"""Typed production data contracts for FoveaMap.

These contracts define the core data boundaries across the system:
    LiDARFrame       -> Data adapters to Preprocessing/Perception
    PerceptionResult -> Perception to Projection/Mapping
    MapSnapshot      -> Mapping to SDK/Transport/Consumers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from .exceptions import ContractError, NumericalConsistencyError


@dataclass(frozen=True)
class LiDARFrame:
    """Standardized LiDAR sweep input contract.

    Attributes:
        pts: (N, 3) float32 coordinates in the ego frame (x forward, y left, z up).
        intensity: (N,) float32 remission/intensity normalized in [0, 1].
        ring: (N,) int16 laser ring / range-image row index (0 = top beam).
        pose: (4, 4) float64 rigid transformation from ego frame to world frame.
        sensor_origin: (3,) float32 LiDAR origin in the ego frame (default [0, 0, 1.73]).
        timestamp: float timestamp in seconds (epoch or relative).
        frame_id: string identifier for tracking and diagnostics.
        label: optional (N,) int8 ground-truth semantic class ID (-1 = unlabelled/ignore).
        moving: optional (N,) bool ground-truth dynamic status.
        prev_sweeps: optional list of (pts_world (M,3) float64, ring (M,) int16)
                     representing prior sweeps for temporal motion cues.
        metadata: optional dictionary with dataset-specific metadata.
    """
    pts: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    pose: np.ndarray
    sensor_origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.73], dtype=np.float32))
    timestamp: float = 0.0
    frame_id: str = ""
    label: np.ndarray | None = None
    moving: np.ndarray | None = None
    prev_sweeps: list[tuple[np.ndarray, np.ndarray] | None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_points(self) -> int:
        return len(self.pts)

    def validate(self) -> None:
        """Validate invariant shapes and numerical integrity."""
        if not isinstance(self.pts, np.ndarray) or self.pts.ndim != 2 or self.pts.shape[1] != 3:
            raise ContractError(f"LiDARFrame.pts must be (N, 3) ndarray; got shape {getattr(self.pts, 'shape', None)}")
        n = len(self.pts)
        if not isinstance(self.intensity, np.ndarray) or self.intensity.shape != (n,):
            raise ContractError(f"LiDARFrame.intensity must be ({n},) ndarray; got {getattr(self.intensity, 'shape', None)}")
        if not isinstance(self.ring, np.ndarray) or self.ring.shape != (n,):
            raise ContractError(f"LiDARFrame.ring must be ({n},) ndarray; got {getattr(self.ring, 'shape', None)}")
        if not isinstance(self.pose, np.ndarray) or self.pose.shape != (4, 4):
            raise ContractError(f"LiDARFrame.pose must be (4, 4) ndarray; got {getattr(self.pose, 'shape', None)}")
        if not isinstance(self.sensor_origin, np.ndarray) or self.sensor_origin.shape != (3,):
            raise ContractError(f"LiDARFrame.sensor_origin must be (3,) ndarray; got {getattr(self.sensor_origin, 'shape', None)}")
        if self.label is not None and (not isinstance(self.label, np.ndarray) or self.label.shape != (n,)):
            raise ContractError(f"LiDARFrame.label must be ({n},) ndarray; got {getattr(self.label, 'shape', None)}")
        if self.moving is not None and (not isinstance(self.moving, np.ndarray) or self.moving.shape != (n,)):
            raise ContractError(f"LiDARFrame.moving must be ({n},) ndarray; got {getattr(self.moving, 'shape', None)}")

        if not np.all(np.isfinite(self.pts)):
            raise NumericalConsistencyError("LiDARFrame.pts contains NaN or Inf coordinates")
        if not np.all(np.isfinite(self.pose)):
            raise NumericalConsistencyError("LiDARFrame.pose contains NaN or Inf values")

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to existing untyped Frame dictionary for backward compatibility."""
        return {
            "pts": self.pts,
            "inten": self.intensity,
            "ring": self.ring,
            "label": self.label if self.label is not None else np.full(len(self.pts), -1, np.int8),
            "moving": self.moving if self.moving is not None else np.zeros(len(self.pts), bool),
            "pose": self.pose,
            "sensor": self.sensor_origin,
            "prev": self.prev_sweeps,
            "meta": dict(self.metadata, timestamp=self.timestamp, frame_id=self.frame_id),
        }

    @classmethod
    def from_legacy_dict(cls, frame_dict: dict[str, Any]) -> LiDARFrame:
        """Construct LiDARFrame from existing untyped Frame dictionary."""
        pts = np.asarray(frame_dict["pts"], dtype=np.float32)
        inten = np.asarray(frame_dict["inten"], dtype=np.float32)
        ring = np.asarray(frame_dict["ring"], dtype=np.int16)
        pose = np.asarray(frame_dict["pose"], dtype=np.float64)
        sensor = np.asarray(frame_dict.get("sensor", [0.0, 0.0, 1.73]), dtype=np.float32)
        meta = dict(frame_dict.get("meta") or {})
        timestamp = float(meta.get("timestamp", meta.get("t", 0.0)))
        frame_id = str(meta.get("token", meta.get("scan", f"frame_{int(timestamp * 1000)}")))
        label = np.asarray(frame_dict["label"], dtype=np.int8) if "label" in frame_dict else None
        moving = np.asarray(frame_dict["moving"], dtype=bool) if "moving" in frame_dict else None
        prev = frame_dict.get("prev")
        return cls(
            pts=pts,
            intensity=inten,
            ring=ring,
            pose=pose,
            sensor_origin=sensor,
            timestamp=timestamp,
            frame_id=frame_id,
            label=label,
            moving=moving,
            prev_sweeps=prev,
            metadata=meta,
        )


@dataclass(frozen=True)
class PerceptionResult:
    """Model-agnostic output of the perception stage.

    Attributes:
        class_probabilities: (N, NUM_CLASSES) float32 per-point class probabilities.
        moving_probabilities: (N,) float32 per-point probability of being dynamic.
        semantic_predictions: (N,) int64 class prediction (e.g. argmax of probabilities).
        is_moving: (N,) bool binary motion decision.
        point_indices: optional (H, W) or (row, col) mapping points to 2D range image.
        metadata: optional diagnostics (e.g. inference_time_ms, model_name).
    """
    class_probabilities: np.ndarray
    moving_probabilities: np.ndarray
    semantic_predictions: np.ndarray
    is_moving: np.ndarray
    point_indices: tuple[np.ndarray, np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_points(self) -> int:
        return len(self.semantic_predictions)

    def validate(self) -> None:
        n = self.num_points
        if self.class_probabilities.ndim != 2 or self.class_probabilities.shape[0] != n:
            raise ContractError(f"class_probabilities shape {self.class_probabilities.shape} mismatch with point count {n}")
        if self.moving_probabilities.shape != (n,):
            raise ContractError(f"moving_probabilities shape {self.moving_probabilities.shape} mismatch with point count {n}")
        if self.semantic_predictions.shape != (n,):
            raise ContractError(f"semantic_predictions shape {self.semantic_predictions.shape} mismatch with point count {n}")
        if self.is_moving.shape != (n,):
            raise ContractError(f"is_moving shape {self.is_moving.shape} mismatch with point count {n}")
        if not np.all(np.isfinite(self.class_probabilities)):
            raise NumericalConsistencyError("class_probabilities contains NaN or Inf")
        if not np.all(np.isfinite(self.moving_probabilities)):
            raise NumericalConsistencyError("moving_probabilities contains NaN or Inf")


@dataclass(frozen=True)
class MapSnapshot:
    """Published, thread-safe snapshot of the multi-tier foveated 2.5D map.

    Attributes:
        timestamp: float timestamp corresponding to this snapshot.
        frame_id: string frame ID of the latest processed sweep.
        ego_pose: (4, 4) float64 vehicle pose at snapshot creation.
        origins: list of (2,) int64 window origin coordinates per tier in tier cell units.
        tier_states: tuple of TierLayers objects (one per tier).
        dynamic_cells: tuple of dynamic observation dicts per tier.
        metadata: optional dict containing cell counts, memory bytes, and latency stats.
    """
    timestamp: float
    frame_id: str
    ego_pose: np.ndarray
    origins: Sequence[tuple[int, int] | np.ndarray]
    tier_states: tuple[Any, ...]
    dynamic_cells: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tiers(self) -> int:
        return len(self.tier_states)

    @property
    def total_memory_bytes(self) -> int:
        return sum(getattr(s, "nbytes", 0) for s in self.tier_states)

    @property
    def total_cells(self) -> int:
        return sum(s.n * s.n for s in self.tier_states if hasattr(s, "n"))

    def get_tier(self, tier_idx: int) -> Any:
        if tier_idx < 0 or tier_idx >= len(self.tier_states):
            raise IndexError(f"Tier index {tier_idx} out of range [0, {len(self.tier_states)})")
        return self.tier_states[tier_idx]
