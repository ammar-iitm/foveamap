"""Perception backend interface and implementations.

Encapsulates point cloud range-image segmentation and motion estimation
behind a stable architectural boundary returning canonical PerceptionResult contracts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Sequence

import numpy as np
import torch

from ..core.contracts import LiDARFrame, PerceptionResult
from ..core.config import PerceptionConfig, SensorConfig
from ..core.exceptions import ConfigurationError, ContractError, PerceptionError
from ..model import RangeUNet, predict
from ..frames import DatasetInfo, make_features, prev_in_ego
from ..sim import CLASSES as SIM_CLASSES, VEHICLE, PERSON
from .. import features_torch


class PerceptionBackend(ABC):
    """Abstract interface for perception inference backends."""

    @abstractmethod
    def predict(self, frame: LiDARFrame) -> PerceptionResult:
        """Run perception inference on a single LiDAR frame and return canonical PerceptionResult."""
        ...

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

    def predict(self, frame: LiDARFrame) -> PerceptionResult:
        """Run range-image inference on a canonical LiDARFrame."""
        if not isinstance(frame, LiDARFrame):
            raise ContractError(f"Expected LiDARFrame, got {type(frame).__name__}")

        legacy_frame = frame.to_legacy_dict()

        # Resolve temporal history for motion residuals
        if frame.prev_sweeps is not None:
            prev = prev_in_ego(legacy_frame, frame.prev_sweeps)
        else:
            hist = [self.history[-k] if len(self.history) >= k else None for k in (1, 2)]
            prev = prev_in_ego(legacy_frame, hist)

        # Extract 8-channel range image features
        try:
            if self.features_engine == "torch" and self._device.type != "mps":
                feats, idx, row, col = features_torch.make_features(
                    legacy_frame, self.dataset_info, prev, self._device
                )
                row_host = row.cpu().numpy()
                col_host = col.cpu().numpy()
            else:
                feats, idx, row, col = make_features(legacy_frame, self.dataset_info, prev)
                row_host = np.asarray(row)
                col_host = np.asarray(col)
        except Exception as exc:
            raise PerceptionError(f"Feature extraction failed: {exc}") from exc

        # Model inference
        try:
            probs_img, pmove_img = predict(
                self.model,
                feats,
                active=self.dataset_info.active,
                fp16=self.config.fp16,
                to_host=True,
            )
        except Exception as exc:
            raise PerceptionError(f"RangeUNet inference failed: {exc}") from exc

        # Per-point gather
        # Clamp row and col to image boundaries in case of outlier ring values
        r_clamped = np.clip(row_host, 0, self.sensor_config.n_rows - 1)
        c_clamped = np.clip(col_host, 0, self.sensor_config.n_cols - 1)

        P = probs_img[r_clamped, c_clamped]               # shape (N, C)
        pmove = pmove_img[r_clamped, c_clamped]           # shape (N,)
        cls = P.argmax(axis=1)                           # shape (N,)
        is_moving = (pmove > self.config.confidence_threshold) & ((cls == VEHICLE) | (cls == PERSON))

        # Maintain sweep history
        pw = frame.pts.astype(np.float64) @ frame.pose[:3, :3].T + frame.pose[:3, 3]
        self.history.append((pw, frame.ring))
        self.history = self.history[-2:]

        return PerceptionResult(
            class_probabilities=P.astype(np.float32),
            moving_probabilities=pmove.astype(np.float32),
            semantic_predictions=cls.astype(np.int64),
            is_moving=is_moving.astype(bool),
            point_indices=np.arange(len(frame.pts), dtype=np.int64),
            metadata={
                "backend": "range_unet",
                "device": str(self._device),
                "features_engine": self.features_engine,
            },
        )
