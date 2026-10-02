"""Perception backend interface and implementations.

Encapsulates point cloud range-image segmentation and motion estimation
behind a stable architectural boundary returning canonical PerceptionResult contracts,
with zero-copy device-resident tensors (DevicePerceptionResult) for Torch pipelines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Any, Sequence

import numpy as np
import torch

from ..core.contracts import LiDARFrame, PerceptionResult
from ..core.config import PerceptionConfig, SensorConfig
from ..core.exceptions import ConfigurationError, ContractError, PerceptionError
from ..model import RangeUNet, predict
from ..frames import DatasetInfo, make_features, prev_in_ego
from ..sim import CLASSES as SIM_CLASSES, VEHICLE, PERSON
from .. import features_torch


def _prev_in_ego_dev(frame: dict[str, Any], history: list[Any], device: torch.device) -> list[Any]:
    """Transform previous sweep points to current ego frame on device."""
    inv = torch.as_tensor(np.linalg.inv(frame["pose"]), device=device)
    src = frame.get("prev")
    if src is None:
        src = history
    out = []
    for item in list(src or [])[:2]:
        if item is None:
            out.append(None)
            continue
        pw, ring = item
        pw = torch.as_tensor(pw).to(device, torch.float64)
        out.append(((pw @ inv[:3, :3].T + inv[:3, 3]).float(), ring))
    out += [None] * (2 - len(out))
    return out


@dataclass
class DevicePerceptionResult:
    """Device-resident perception inference results (PyTorch Tensors on active device).

    Keeps probability and motion tensors on-device for direct zero-copy consumption
    by Torch grid engines without GPU -> CPU -> GPU round-trips.
    """
    class_probabilities: torch.Tensor    # (N, C) on device
    moving_probabilities: torch.Tensor   # (N,) on device
    semantic_predictions: torch.Tensor   # (N,) on device
    is_moving: torch.Tensor              # (N,) bool on device
    device: torch.device
    pts_world: torch.Tensor | None = None
    point_indices: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_host(self) -> PerceptionResult:
        """Convert device tensors to canonical host PerceptionResult contract."""
        return PerceptionResult(
            class_probabilities=self.class_probabilities.detach().cpu().numpy().astype(np.float32),
            moving_probabilities=self.moving_probabilities.detach().cpu().numpy().astype(np.float32),
            semantic_predictions=self.semantic_predictions.detach().cpu().numpy().astype(np.int64),
            is_moving=self.is_moving.detach().cpu().numpy().astype(bool),
            point_indices=self.point_indices.detach().cpu().numpy().astype(np.int64) if self.point_indices is not None else None,
            metadata=dict(self.metadata, device=str(self.device)),
        )


class PerceptionBackend(ABC):
    """Abstract interface for perception inference backends."""

    @abstractmethod
    def predict(self, frame: LiDARFrame) -> PerceptionResult:
        """Run perception inference on a single LiDAR frame and return canonical host PerceptionResult."""
        ...

    def predict_device(self, frame: LiDARFrame, dev_math: bool = False) -> DevicePerceptionResult:
        """Run perception inference and retain tensors on the active computing device."""
        res = self.predict(frame)
        dev = self.device
        return DevicePerceptionResult(
            class_probabilities=torch.as_tensor(res.class_probabilities, device=dev),
            moving_probabilities=torch.as_tensor(res.moving_probabilities, device=dev),
            semantic_predictions=torch.as_tensor(res.semantic_predictions, device=dev),
            is_moving=torch.as_tensor(res.is_moving, device=dev, dtype=torch.bool),
            device=dev,
            metadata=res.metadata,
        )

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of semantic classes supported."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Active computing device."""
        ...

    def reset(self) -> None:
        """Reset internal temporal states / sweep history."""
        pass


class RangeUNetBackend(PerceptionBackend):
    """Range-image U-Net perception backend (segmentation + motion residual).

    Wraps the RangeUNet architecture, managing feature extraction, model inference,
    temporal sweep history, and per-point classification.
    """

    def __init__(
        self,
        config: PerceptionConfig,
        sensor_config: SensorConfig,
        device: torch.device,
        features_engine: str = "numpy",
    ) -> None:
        self.config = config
        self.sensor_config = sensor_config
        self._device = device
        self.features_engine = features_engine

        # Build sensor/dataset metadata bridge
        self.dataset_info = DatasetInfo(
            name=sensor_config.name,
            n_rows=sensor_config.n_rows,
            n_cols=sensor_config.n_cols,
            class_names=list(SIM_CLASSES),
            active=np.asarray(config.active_classes, bool),
            source=sensor_config.source_description or f"{sensor_config.name} sensor",
            hz=sensor_config.hz,
        )

        # Instantiate model architecture
        self.model = RangeUNet()

        # Load weights if checkpoint is provided
        if config.checkpoint_path is not None:
            if not os.path.exists(config.checkpoint_path):
                raise ConfigurationError(
                    f"Perception checkpoint file not found at: {config.checkpoint_path}"
                )
            try:
                state_dict = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
            except Exception as exc:
                raise PerceptionError(f"Failed to load checkpoint '{config.checkpoint_path}': {exc}") from exc

        self.model.to(self._device).eval()
        self.history: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self) -> None:
        """Clear sweep history."""
        self.history.clear()

    def predict_device(self, frame: LiDARFrame, dev_math: bool = False) -> DevicePerceptionResult:
        """Run range-image inference and retain result tensors on-device."""
        if not isinstance(frame, LiDARFrame):
            raise ContractError(f"Expected LiDARFrame, got {type(frame).__name__}")

        legacy_frame = frame.to_legacy_dict()
        dev = self._device

        # Resolve temporal history for motion residuals
        hist = [self.history[-k] if len(self.history) >= k else None for k in (1, 2)]
        if dev_math:
            pts_dev = torch.as_tensor(frame.pts, device=dev, dtype=torch.float32)
            prev = _prev_in_ego_dev(legacy_frame, hist, dev)
            frame_input = dict(legacy_frame, pts=pts_dev)
            feats, idx, row, col = features_torch.make_features(
                frame_input, self.dataset_info, prev, dev
            )
            row_dev = torch.as_tensor(row, device=dev)
            col_dev = torch.as_tensor(col, device=dev)
            pose_dev = torch.as_tensor(frame.pose, device=dev)
            pw_dev = pts_dev.double() @ pose_dev[:3, :3].T + pose_dev[:3, 3]
        elif self.features_engine == "torch" and dev.type != "mps":
            prev = prev_in_ego(legacy_frame, hist)
            feats, idx, row, col = features_torch.make_features(
                legacy_frame, self.dataset_info, prev, dev
            )
            row_dev = torch.as_tensor(row, device=dev)
            col_dev = torch.as_tensor(col, device=dev)
            pw_dev = None
        else:
            prev = prev_in_ego(legacy_frame, hist)
            feats, idx, row, col = make_features(legacy_frame, self.dataset_info, prev)
            row_dev = torch.as_tensor(row, device=dev)
            col_dev = torch.as_tensor(col, device=dev)
            pw_dev = None

        # Model inference on device without host transfer
        try:
            probs_img, pmove_img = predict(
                self.model,
                feats,
                active=self.dataset_info.active,
                fp16=self.config.fp16,
                to_host=False,
            )
        except Exception as exc:
            raise PerceptionError(f"RangeUNet device inference failed: {exc}") from exc

        # Per-point gather on device
        r_clamped = row_dev.clamp(0, self.sensor_config.n_rows - 1)
        c_clamped = col_dev.clamp(0, self.sensor_config.n_cols - 1)

        P = probs_img[r_clamped, c_clamped]
        pmove = pmove_img[r_clamped, c_clamped]
        cls = P.argmax(dim=1)
        thresh = float(self.config.confidence_threshold)
        is_moving = (pmove > thresh) & ((cls == VEHICLE) | (cls == PERSON))

        # Maintain sweep history (store world coordinates in host memory)
        if pw_dev is not None:
            pw_host = pw_dev.detach().cpu().numpy()
        else:
            pw_host = frame.pts.astype(np.float64) @ frame.pose[:3, :3].T + frame.pose[:3, 3]
        self.history.append((pw_host, frame.ring))
        self.history = self.history[-2:]

        return DevicePerceptionResult(
            class_probabilities=P,
            moving_probabilities=pmove,
            semantic_predictions=cls,
            is_moving=is_moving,
            device=dev,
            pts_world=pw_dev,
            point_indices=torch.arange(len(frame.pts), device=dev, dtype=torch.long),
            metadata={
                "backend": "range_unet",
                "device": str(dev),
                "features_engine": self.features_engine,
                "device_resident": True,
            },
        )

    def predict(self, frame: LiDARFrame) -> PerceptionResult:
        """Run range-image inference and return canonical host PerceptionResult."""
        dev_res = self.predict_device(frame, dev_math=False)
        return dev_res.to_host()
