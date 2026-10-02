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

    Coordinate Convention:
        Right-handed ISO 8855 vehicle ego frame:
        - X: forward (metres)
        - Y: left (metres)
        - Z: up (metres)
        World frame: origin at initial reference pose, Z up (metres).

    Attributes:
        pts: (N, 3) float32 coordinates in the ego frame.
        intensity: (N,) float32 calibrated remission/intensity in [0, 1].
        ring: (N,) int16 laser ring / range-image row index (0 = top beam).
        pose: (4, 4) float64 rigid transformation from ego frame to world frame.
        sensor_origin: (3,) float32 LiDAR origin in the ego frame (default [0, 0, 1.73]).
        timestamp: float timestamp in seconds (epoch or sequence-relative).
        frame_id: string identifier for tracking and diagnostics.
        label: optional (N,) int8 ground-truth semantic class ID (-1 = unlabelled/ignore).
        moving: optional (N,) bool ground-truth dynamic status (True = moving).
        prev_sweeps: optional list of prior sweeps as (pts_world (M,3) float64, ring (M,) int16)
                     or None entries, providing temporal motion cues.
        metadata: dictionary with dataset-specific or provenance metadata.
    """
    pts: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    pose: np.ndarray
    sensor_origin: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.73], dtype=np.float32))
    timestamp: float = 0.0
    frame_id: str = ""
    source_id: str = ""
    label: np.ndarray | None = None
    moving: np.ndarray | None = None
    prev_sweeps: list[tuple[np.ndarray, np.ndarray] | None] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_points(self) -> int:
        return len(self.pts)

    @property
    def has_annotations(self) -> bool:
        """True if ground-truth semantic or motion annotations are attached."""
        return self.label is not None or self.moving is not None

    def without_annotations(self) -> LiDARFrame:
        """Return a copy of this frame stripped of all ground-truth annotations."""
        return LiDARFrame(
            pts=self.pts,
            intensity=self.intensity,
            ring=self.ring,
            pose=self.pose,
            sensor_origin=self.sensor_origin,
            timestamp=self.timestamp,
            frame_id=self.frame_id,
            source_id=self.source_id,
            label=None,
            moving=None,
            prev_sweeps=self.prev_sweeps,
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        """Validate invariant shapes, dtypes, and numerical integrity."""
        # 1. Point coordinates
        if not isinstance(self.pts, np.ndarray) or self.pts.ndim != 2 or self.pts.shape[1] != 3:
            raise ContractError(f"LiDARFrame.pts must be an (N, 3) ndarray; got shape {getattr(self.pts, 'shape', None)}")
        if not np.issubdtype(self.pts.dtype, np.floating):
            raise ContractError(f"LiDARFrame.pts must have floating dtype; got {self.pts.dtype}")
        n = len(self.pts)

        # 2. Intensity
        if not isinstance(self.intensity, np.ndarray) or self.intensity.shape != (n,):
            raise ContractError(f"LiDARFrame.intensity must be ({n},) ndarray; got shape {getattr(self.intensity, 'shape', None)}")
        if not np.issubdtype(self.intensity.dtype, np.floating):
            raise ContractError(f"LiDARFrame.intensity must have floating dtype; got {self.intensity.dtype}")

        # 3. Ring / laser beam indices
        if not isinstance(self.ring, np.ndarray) or self.ring.shape != (n,):
            raise ContractError(f"LiDARFrame.ring must be ({n},) ndarray; got shape {getattr(self.ring, 'shape', None)}")
        if not np.issubdtype(self.ring.dtype, np.integer):
            raise ContractError(f"LiDARFrame.ring must have integer dtype; got {self.ring.dtype}")

        # 4. Ego pose (4x4 SE(3) matrix)
        if not isinstance(self.pose, np.ndarray) or self.pose.shape != (4, 4):
            raise ContractError(f"LiDARFrame.pose must be (4, 4) ndarray; got shape {getattr(self.pose, 'shape', None)}")
        if not np.issubdtype(self.pose.dtype, np.floating):
            raise ContractError(f"LiDARFrame.pose must have floating dtype; got {self.pose.dtype}")
        if not np.allclose(self.pose[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
            raise ContractError(f"LiDARFrame.pose must have bottom row [0, 0, 0, 1]; got {self.pose[3, :]}")

        # 5. Sensor origin and source ID
        if not isinstance(self.sensor_origin, np.ndarray) or self.sensor_origin.shape != (3,):
            raise ContractError(f"LiDARFrame.sensor_origin must be (3,) ndarray; got shape {getattr(self.sensor_origin, 'shape', None)}")
        if not isinstance(self.source_id, str):
            raise ContractError(f"LiDARFrame.source_id must be a string; got {type(self.source_id)}")

        # 6. Optional annotations
        if self.label is not None:
            if not isinstance(self.label, np.ndarray) or self.label.shape != (n,):
                raise ContractError(f"LiDARFrame.label must be ({n},) ndarray; got shape {getattr(self.label, 'shape', None)}")
            if not np.issubdtype(self.label.dtype, np.integer):
                raise ContractError(f"LiDARFrame.label must have integer dtype; got {self.label.dtype}")

        if self.moving is not None:
            if not isinstance(self.moving, np.ndarray) or self.moving.shape != (n,):
                raise ContractError(f"LiDARFrame.moving must be ({n},) ndarray; got shape {getattr(self.moving, 'shape', None)}")
            if self.moving.dtype != bool and not np.issubdtype(self.moving.dtype, np.integer):
                raise ContractError(f"LiDARFrame.moving must have boolean dtype; got {self.moving.dtype}")

        # 7. Optional previous sweeps
        if self.prev_sweeps is not None:
            for idx, item in enumerate(self.prev_sweeps):
                if item is not None:
                    if not isinstance(item, (tuple, list)) or len(item) != 2:
                        raise ContractError(f"prev_sweeps[{idx}] must be a 2-tuple (pw, ring) or None; got {type(item)}")
                    pw, r = item
                    if not isinstance(pw, np.ndarray) or pw.ndim != 2 or pw.shape[1] != 3:
                        raise ContractError(f"prev_sweeps[{idx}][0] must be (M, 3) ndarray; got {getattr(pw, 'shape', None)}")
                    if not isinstance(r, np.ndarray) or r.shape != (len(pw),):
                        raise ContractError(f"prev_sweeps[{idx}][1] must be ({len(pw)},) ndarray; got {getattr(r, 'shape', None)}")

        # 8. Numerical finiteness check
        if not np.all(np.isfinite(self.pts)):
            raise NumericalConsistencyError("LiDARFrame.pts contains NaN or Inf coordinates")
        if not np.all(np.isfinite(self.intensity)):
            raise NumericalConsistencyError("LiDARFrame.intensity contains NaN or Inf values")
        if not np.all(np.isfinite(self.pose)):
            raise NumericalConsistencyError("LiDARFrame.pose contains NaN or Inf values")
        if not np.all(np.isfinite(self.sensor_origin)):
            raise NumericalConsistencyError("LiDARFrame.sensor_origin contains NaN or Inf values")

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to existing untyped Frame dictionary for complete backward compatibility."""
        legacy_meta = dict(self.metadata)
        if "timestamp" not in legacy_meta:
            legacy_meta["timestamp"] = self.timestamp
        if "frame_id" not in legacy_meta:
            legacy_meta["frame_id"] = self.frame_id
        if "source_id" not in legacy_meta and self.source_id:
            legacy_meta["source_id"] = self.source_id

        return {
            "pts": self.pts,
            "inten": self.intensity,
            "ring": self.ring,
            "label": self.label if self.label is not None else np.full(len(self.pts), -1, np.int8),
            "moving": self.moving if self.moving is not None else np.zeros(len(self.pts), bool),
            "pose": self.pose,
            "sensor": self.sensor_origin,
            "prev": self.prev_sweeps,
            "meta": legacy_meta,
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
        frame_id = str(meta.get("token", meta.get("scan", meta.get("frame_id", f"frame_{int(timestamp * 1000)}"))))
        source_id = str(meta.get("source_id", meta.get("source", "")))
        label = np.asarray(frame_dict["label"], dtype=np.int8) if "label" in frame_dict and frame_dict["label"] is not None else None
        moving = np.asarray(frame_dict["moving"], dtype=bool) if "moving" in frame_dict and frame_dict["moving"] is not None else None
        prev = frame_dict.get("prev")
        return cls(
            pts=pts,
            intensity=inten,
            ring=ring,
            pose=pose,
            sensor_origin=sensor,
            timestamp=timestamp,
            frame_id=frame_id,
            source_id=source_id,
            label=label,
            moving=moving,
            prev_sweeps=prev,
            metadata=meta,
        )


@dataclass(frozen=True)
class PerceptionResult:
    """Model-agnostic output of the perception stage.

    Attributes:
        class_probabilities: (N, C) float32 per-point class probabilities in [0, 1].
        moving_probabilities: (N,) float32 per-point probability of being dynamic in [0, 1].
        semantic_predictions: (N,) int64 class prediction (0 <= class_id < C).
        is_moving: (N,) bool binary motion decision.
        confidence: optional (N,) float32 classification confidence scores in [0, 1].
        point_indices: optional (N,) or (row, col) mapping points to 2D range image.
        metadata: optional diagnostics (e.g. inference_time_ms, model_name).
    """
    class_probabilities: np.ndarray
    moving_probabilities: np.ndarray
    semantic_predictions: np.ndarray
    is_moving: np.ndarray
    confidence: np.ndarray | None = None
    point_indices: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_points(self) -> int:
        return len(self.semantic_predictions)

    @property
    def point_confidence(self) -> np.ndarray:
        """Per-point classification confidence (explicit or max class probability)."""
        if self.confidence is not None:
            return self.confidence
        if len(self.class_probabilities) == 0:
            return np.empty((0,), dtype=np.float32)
        return self.class_probabilities.max(axis=-1).astype(np.float32)

    def validate(self) -> None:
        """Validate shapes, probability bounds, class IDs, and numerical consistency."""
        n = self.num_points
        if not isinstance(self.class_probabilities, np.ndarray) or self.class_probabilities.ndim != 2 or self.class_probabilities.shape[0] != n:
            raise ContractError(f"class_probabilities shape {getattr(self.class_probabilities, 'shape', None)} mismatch with point count {n}")
        if not isinstance(self.moving_probabilities, np.ndarray) or self.moving_probabilities.shape != (n,):
            raise ContractError(f"moving_probabilities shape {getattr(self.moving_probabilities, 'shape', None)} mismatch with point count {n}")
        if not isinstance(self.semantic_predictions, np.ndarray) or self.semantic_predictions.shape != (n,):
            raise ContractError(f"semantic_predictions shape {getattr(self.semantic_predictions, 'shape', None)} mismatch with point count {n}")
        if not np.issubdtype(self.semantic_predictions.dtype, np.integer):
            raise ContractError(f"semantic_predictions must have integer dtype; got {self.semantic_predictions.dtype}")
        if not isinstance(self.is_moving, np.ndarray) or self.is_moving.shape != (n,):
            raise ContractError(f"is_moving shape {getattr(self.is_moving, 'shape', None)} mismatch with point count {n}")

        # Check finiteness
        if not np.all(np.isfinite(self.class_probabilities)):
            raise NumericalConsistencyError("class_probabilities contains NaN or Inf")
        if not np.all(np.isfinite(self.moving_probabilities)):
            raise NumericalConsistencyError("moving_probabilities contains NaN or Inf")

        # Check probability bounds
        if np.any(self.class_probabilities < -1e-5) or np.any(self.class_probabilities > 1.0 + 1e-4):
            raise NumericalConsistencyError("class_probabilities has values outside [0, 1]")
        if np.any(self.moving_probabilities < -1e-5) or np.any(self.moving_probabilities > 1.0 + 1e-4):
            raise NumericalConsistencyError("moving_probabilities has values outside [0, 1]")

        # Check class index bounds
        num_classes = self.class_probabilities.shape[1]
        if n > 0 and num_classes > 0:
            if np.any(self.semantic_predictions < 0) or np.any(self.semantic_predictions >= num_classes):
                raise ContractError(
                    f"semantic_predictions contains class IDs outside [0, {num_classes - 1}]"
                )

        # Check optional confidence
        if self.confidence is not None:
            if not isinstance(self.confidence, np.ndarray) or self.confidence.shape != (n,):
                raise ContractError(f"confidence shape {getattr(self.confidence, 'shape', None)} mismatch with point count {n}")
            if not np.all(np.isfinite(self.confidence)):
                raise NumericalConsistencyError("confidence contains NaN or Inf")
            if np.any(self.confidence < -1e-5) or np.any(self.confidence > 1.0 + 1e-4):
                raise NumericalConsistencyError("confidence has values outside [0, 1]")

        # Check optional point_indices
        if self.point_indices is not None:
            if isinstance(self.point_indices, (tuple, list)):
                if len(self.point_indices) != 2:
                    raise ContractError("point_indices tuple must contain exactly 2 elements (row, col)")
                r, c = self.point_indices
                if not isinstance(r, np.ndarray) or r.shape != (n,) or not isinstance(c, np.ndarray) or c.shape != (n,):
                    raise ContractError(f"point_indices (row, col) elements must both have shape ({n},)")
            elif isinstance(self.point_indices, np.ndarray):
                if self.point_indices.shape[0] != n:
                    raise ContractError(f"point_indices shape {self.point_indices.shape} mismatch with point count {n}")
            else:
                raise ContractError(f"point_indices must be tuple, list, or ndarray; got {type(self.point_indices)}")


@dataclass(frozen=True)
class MapSnapshot:
    """Published snapshot of the multi-tier foveated 2.5D map.

    Ownership & Immutability Semantics:
        - `MapSnapshot` itself is a frozen dataclass container preventing attribute reassignment.
        - `tier_states` contains host `TierLayers` instances produced by the grid engine (`snapshot()`).
        - The internal NumPy arrays inside `TierLayers` are newly allocated host copies, completely
          detached from live grid memory updates (the grid can proceed updating without altering this snapshot).
        - However, because standard NumPy arrays in Python are mutable by default, consumers must
          treat `tier_states` arrays as READ-ONLY to avoid corrupting shared downstream consumer state.
        - Deeper zero-copy / buffer-locking mechanics are deferred to the transport layer.

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

    def __post_init__(self) -> None:
        if not isinstance(self.ego_pose, np.ndarray) or self.ego_pose.shape != (4, 4):
            raise ContractError(f"ego_pose must be (4, 4) ndarray; got {getattr(self.ego_pose, 'shape', None)}")
        if len(self.origins) != len(self.tier_states):
            raise ContractError(f"Number of origins ({len(self.origins)}) must match tier_states ({len(self.tier_states)})")

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
