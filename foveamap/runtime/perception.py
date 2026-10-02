"""Perception subsystem architecture and model integration for FoveaMap.

Defines a stable, model-agnostic perception boundary returning canonical
PerceptionResult contracts, with zero-copy device-resident tensors (DevicePerceptionResult)
for high-performance Torch/CUDA pipelines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import time
from typing import Any, Sequence, Type

import numpy as np
import torch

from ..core.contracts import LiDARFrame, PerceptionResult
from ..core.config import PerceptionConfig, SensorConfig
from ..core.exceptions import ConfigurationError, ContractError, PerceptionError, NumericalConsistencyError
from .device import DeviceContext
from ..model import RangeUNet, predict
from ..frames import DatasetInfo, make_features, prev_in_ego
from ..sim import CLASSES as SIM_CLASSES, ROAD, SIDEWALK, BUILDING, VEHICLE, PERSON
from .. import features_torch


class CheckpointNotFoundError(PerceptionError, ConfigurationError):
    """Raised when a configured perception model checkpoint path does not exist."""
    pass


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

    Keeps probability, motion, and correspondence tensors on-device for direct
    zero-copy consumption by GPU grid engines without GPU -> CPU -> GPU round-trips.
    """
    class_probabilities: torch.Tensor    # (N, C) on device
    moving_probabilities: torch.Tensor   # (N,) on device
    semantic_predictions: torch.Tensor   # (N,) on device
    is_moving: torch.Tensor              # (N,) bool on device
    device: torch.device
    confidence: torch.Tensor | None = None
    pts_world: torch.Tensor | None = None
    point_indices: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_host(self) -> PerceptionResult:
        """Convert device tensors to canonical host PerceptionResult contract."""
        cp = self.class_probabilities.detach().cpu().numpy().astype(np.float32)
        mp = self.moving_probabilities.detach().cpu().numpy().astype(np.float32)
        sp = self.semantic_predictions.detach().cpu().numpy().astype(np.int64)
        mv = self.is_moving.detach().cpu().numpy().astype(bool)

        conf = None
        if self.confidence is not None:
            conf = self.confidence.detach().cpu().numpy().astype(np.float32)

        indices = None
        if self.point_indices is not None:
            if isinstance(self.point_indices, (tuple, list)):
                indices = (
                    self.point_indices[0].detach().cpu().numpy().astype(np.int64),
                    self.point_indices[1].detach().cpu().numpy().astype(np.int64),
                )
            elif isinstance(self.point_indices, torch.Tensor):
                indices = self.point_indices.detach().cpu().numpy().astype(np.int64)

        return PerceptionResult(
            class_probabilities=cp,
            moving_probabilities=mp,
            semantic_predictions=sp,
            is_moving=mv,
            confidence=conf,
            point_indices=indices,
            metadata=dict(self.metadata, device=str(self.device)),
        )


class PerceptionBackend(ABC):
    """Abstract interface for perception inference backends."""

    @abstractmethod
    def predict(
        self,
        frame: LiDARFrame,
        *,
        device: DeviceContext | None = None,
    ) -> PerceptionResult:
        """Run perception inference on a single LiDAR frame and return canonical host PerceptionResult."""
        ...

    @abstractmethod
    def predict_device(
        self,
        frame: LiDARFrame,
        dev_math: bool = False,
    ) -> DevicePerceptionResult:
        """Run perception inference and retain tensors on the active computing device."""
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

    @property
    def name(self) -> str:
        """Backend identifier name."""
        return self.__class__.__name__

    @property
    def model_name(self) -> str:
        """Underlying model architecture name."""
        return self.__class__.__name__

    @property
    def checkpoint_path(self) -> str | None:
        """Path to loaded weights checkpoint, if any."""
        return None

    @property
    def is_ready(self) -> bool:
        """True if the backend is initialized and ready for inference."""
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        """Diagnostic metadata describing backend configuration."""
        return {
            "name": self.name,
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "device": str(self.device),
            "checkpoint_path": self.checkpoint_path,
        }

    def initialize(self) -> None:
        """Initialize or warm up the backend."""
        pass

    def reset(self) -> None:
        """Reset internal temporal states / sweep history."""
        pass


class RangeUNetBackend(PerceptionBackend):
    """Range-image U-Net perception backend (segmentation + motion residual).

    Wraps the RangeUNet architecture, managing feature extraction, model inference,
    temporal sweep history, per-point classification, and latency profiling.
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
            if not os.path.isfile(config.checkpoint_path):
                raise CheckpointNotFoundError(
                    f"Perception checkpoint file not found at: {config.checkpoint_path}"
                )
            try:
                state_dict = torch.load(config.checkpoint_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict)
            except Exception as exc:
                raise PerceptionError(f"Failed to load checkpoint '{config.checkpoint_path}': {exc}") from exc

        self.model.to(self._device).eval()
        self.history: list[tuple[np.ndarray, np.ndarray]] = []
        self._is_ready: bool = True

    @property
    def name(self) -> str:
        return "range_unet"

    @property
    def model_name(self) -> str:
        return "RangeUNet"

    @property
    def checkpoint_path(self) -> str | None:
        return self.config.checkpoint_path

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def reset(self) -> None:
        """Clear sweep history."""
        self.history.clear()

    @torch.inference_mode()
    def predict_device(self, frame: LiDARFrame, dev_math: bool = False) -> DevicePerceptionResult:
        """Run range-image inference and retain result tensors on-device."""
        if not isinstance(frame, LiDARFrame):
            raise ContractError(f"Expected LiDARFrame, got {type(frame).__name__}")

        dev = self._device
        n_pts = len(frame.pts)
        t_start = time.perf_counter()

        # Handle empty point cloud (N=0) gracefully
        if n_pts == 0:
            empty_probs = torch.empty((0, self.num_classes), device=dev, dtype=torch.float32)
            empty_move = torch.empty((0,), device=dev, dtype=torch.float32)
            empty_pred = torch.empty((0,), device=dev, dtype=torch.long)
            empty_is_mv = torch.empty((0,), device=dev, dtype=torch.bool)
            empty_conf = torch.empty((0,), device=dev, dtype=torch.float32)
            empty_pw = torch.empty((0, 3), device=dev, dtype=torch.float64) if dev_math else None

            return DevicePerceptionResult(
                class_probabilities=empty_probs,
                moving_probabilities=empty_move,
                semantic_predictions=empty_pred,
                is_moving=empty_is_mv,
                device=dev,
                confidence=empty_conf,
                pts_world=empty_pw,
                point_indices=(
                    torch.empty((0,), device=dev, dtype=torch.long),
                    torch.empty((0,), device=dev, dtype=torch.long),
                ),
                metadata={
                    "backend": self.name,
                    "model_name": self.model_name,
                    "device": str(dev),
                    "features_engine": self.features_engine,
                    "device_resident": True,
                    "timing": {
                        "feature_ms": 0.0,
                        "inference_ms": 0.0,
                        "postprocess_ms": 0.0,
                        "total_ms": 0.0,
                    },
                },
            )

        legacy_frame = frame.to_legacy_dict()

        # -------------------------------------------------------------
        # Stage 1: Feature Extraction
        # -------------------------------------------------------------
        t0 = time.perf_counter()
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

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        feature_ms = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Stage 2: Model Inference
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        try:
            probs_img, pmove_img = predict(
                self.model,
                feats,
                active=self.dataset_info.active,
                fp16=self.config.fp16,
                to_host=False,
            )
        except Exception as exc:
            raise PerceptionError(f"RangeUNet forward pass failed: {exc}") from exc

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Finiteness validation on network outputs
        if not torch.all(torch.isfinite(probs_img)):
            raise PerceptionError("RangeUNet model produced non-finite class probabilities")
        if not torch.all(torch.isfinite(pmove_img)):
            raise PerceptionError("RangeUNet model produced non-finite motion probabilities")

        # -------------------------------------------------------------
        # Stage 3: Per-point Correspondence Gather & Motion Decision
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        r_clamped = row_dev.clamp(0, self.sensor_config.n_rows - 1)
        c_clamped = col_dev.clamp(0, self.sensor_config.n_cols - 1)

        P = probs_img[r_clamped, c_clamped]
        pmove = pmove_img[r_clamped, c_clamped]
        cls = P.argmax(dim=1)
        conf = P.max(dim=1).values
        thresh = float(self.config.confidence_threshold)
        is_moving = (pmove > thresh) & ((cls == VEHICLE) | (cls == PERSON))

        # Maintain sweep history (store world coordinates in host memory)
        if pw_dev is not None:
            pw_host = pw_dev.detach().cpu().numpy()
        else:
            pw_host = frame.pts.astype(np.float64) @ frame.pose[:3, :3].T + frame.pose[:3, 3]
        self.history.append((pw_host, frame.ring))
        self.history = self.history[-2:]

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        postprocess_ms = (time.perf_counter() - t0) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        return DevicePerceptionResult(
            class_probabilities=P,
            moving_probabilities=pmove,
            semantic_predictions=cls,
            is_moving=is_moving,
            device=dev,
            confidence=conf,
            pts_world=pw_dev,
            point_indices=(r_clamped, c_clamped),
            metadata={
                "backend": self.name,
                "model_name": self.model_name,
                "device": str(dev),
                "features_engine": self.features_engine,
                "device_resident": True,
                "timing": {
                    "feature_ms": round(feature_ms, 3),
                    "inference_ms": round(inference_ms, 3),
                    "postprocess_ms": round(postprocess_ms, 3),
                    "total_ms": round(total_ms, 3),
                },
            },
        )

    def predict(
        self,
        frame: LiDARFrame,
        *,
        device: DeviceContext | None = None,
    ) -> PerceptionResult:
        """Run range-image inference and return canonical host PerceptionResult."""
        dev_res = self.predict_device(frame, dev_math=False)
        return dev_res.to_host()


class ClassicalFallbackBackend(PerceptionBackend):
    """Zero-model geometric rule-based perception backend for testing and low-power fallback.

    Applies height thresholds relative to sensor mounting to classify points into
    ROAD, SIDEWALK, and BUILDING classes without requiring PyTorch neural network checkpoints.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        sensor_config: SensorConfig | None = None,
        device: torch.device | None = None,
        ground_z_threshold: float = -0.8,
        obstacle_z_threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        self.config = config or PerceptionConfig()
        self.sensor_config = sensor_config or SensorConfig()
        self._device = device or torch.device("cpu")
        self.ground_z_threshold = ground_z_threshold
        self.obstacle_z_threshold = obstacle_z_threshold

    @property
    def name(self) -> str:
        return "classical_fallback"

    @property
    def model_name(self) -> str:
        return "HeuristicGeometricSegmenter"

    @property
    def num_classes(self) -> int:
        return self.config.num_classes

    @property
    def device(self) -> torch.device:
        return self._device

    def predict_device(self, frame: LiDARFrame, dev_math: bool = False) -> DevicePerceptionResult:
        if not isinstance(frame, LiDARFrame):
            raise ContractError(f"Expected LiDARFrame, got {type(frame).__name__}")

        dev = self._device
        n_pts = len(frame.pts)
        t_start = time.perf_counter()

        if n_pts == 0:
            return DevicePerceptionResult(
                class_probabilities=torch.empty((0, self.num_classes), device=dev, dtype=torch.float32),
                moving_probabilities=torch.empty((0,), device=dev, dtype=torch.float32),
                semantic_predictions=torch.empty((0,), device=dev, dtype=torch.long),
                is_moving=torch.empty((0,), device=dev, dtype=torch.bool),
                device=dev,
                confidence=torch.empty((0,), device=dev, dtype=torch.float32),
                point_indices=torch.empty((0,), device=dev, dtype=torch.long),
                metadata={"backend": self.name, "timing": {"total_ms": 0.0}},
            )

        pts = frame.pts
        # Height relative to sensor mount origin
        rel_z = pts[:, 2] - frame.sensor_origin[2]

        cls = np.full(n_pts, fill_value=SIDEWALK, dtype=np.int64)
        cls[rel_z < self.ground_z_threshold] = ROAD
        cls[rel_z > self.obstacle_z_threshold] = BUILDING

        # Construct one-hot probability tensor
        P_np = np.zeros((n_pts, self.num_classes), dtype=np.float32)
        P_np[np.arange(n_pts), cls] = 1.0

        p_move_np = np.zeros(n_pts, dtype=np.float32)
        is_moving_np = np.zeros(n_pts, dtype=bool)

        P_dev = torch.as_tensor(P_np, device=dev)
        pmove_dev = torch.as_tensor(p_move_np, device=dev)
        cls_dev = torch.as_tensor(cls, device=dev)
        is_moving_dev = torch.as_tensor(is_moving_np, device=dev)
        conf_dev = torch.ones(n_pts, device=dev, dtype=torch.float32)

        pw_dev = None
        if dev_math:
            pts_dev = torch.as_tensor(frame.pts, device=dev, dtype=torch.float32)
            pose_dev = torch.as_tensor(frame.pose, device=dev)
            pw_dev = pts_dev.double() @ pose_dev[:3, :3].T + pose_dev[:3, 3]

        total_ms = (time.perf_counter() - t_start) * 1000.0

        return DevicePerceptionResult(
            class_probabilities=P_dev,
            moving_probabilities=pmove_dev,
            semantic_predictions=cls_dev,
            is_moving=is_moving_dev,
            device=dev,
            confidence=conf_dev,
            pts_world=pw_dev,
            point_indices=torch.arange(n_pts, device=dev, dtype=torch.long),
            metadata={
                "backend": self.name,
                "model_name": self.model_name,
                "device": str(dev),
                "device_resident": True,
                "timing": {
                    "feature_ms": 0.0,
                    "inference_ms": 0.0,
                    "postprocess_ms": round(total_ms, 3),
                    "total_ms": round(total_ms, 3),
                },
            },
        )

    def predict(
        self,
        frame: LiDARFrame,
        *,
        device: DeviceContext | None = None,
    ) -> PerceptionResult:
        dev_res = self.predict_device(frame, dev_math=False)
        return dev_res.to_host()


# ---------------------------------------------------------------------------
# Backend Registry & Factory
# ---------------------------------------------------------------------------
_PERCEPTION_BACKENDS: dict[str, Type[PerceptionBackend]] = {
    "range_unet": RangeUNetBackend,
    "rangeunet": RangeUNetBackend,
    "classical": ClassicalFallbackBackend,
    "classical_fallback": ClassicalFallbackBackend,
    "heuristic": ClassicalFallbackBackend,
}


def register_perception_backend(name: str, backend_cls: Type[PerceptionBackend]) -> None:
    """Register a custom PerceptionBackend implementation."""
    if not issubclass(backend_cls, PerceptionBackend):
        raise TypeError(f"Registered backend class {backend_cls} must subclass PerceptionBackend")
    key = name.strip().lower()
    _PERCEPTION_BACKENDS[key] = backend_cls


def create_perception_backend(
    config: PerceptionConfig,
    sensor_config: SensorConfig,
    device: torch.device,
    features_engine: str = "numpy",
    **kwargs: Any,
) -> PerceptionBackend:
    """Factory creating an instantiated PerceptionBackend according to configuration."""
    backend_key = (config.backend_type or "range_unet").strip().lower()
    if backend_key not in _PERCEPTION_BACKENDS:
        raise ConfigurationError(
            f"Unknown perception backend '{config.backend_type}'. "
            f"Available backends: {sorted(_PERCEPTION_BACKENDS.keys())}"
        )
    backend_cls = _PERCEPTION_BACKENDS[backend_key]
    return backend_cls(
        config=config,
        sensor_config=sensor_config,
        device=device,
        features_engine=features_engine,
        **kwargs,
    )


def list_perception_backends() -> list[str]:
    """Return all registered perception backend types."""
    return sorted(list(_PERCEPTION_BACKENDS.keys()))
