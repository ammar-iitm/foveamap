"""TF and coordinate-frame policy (ROS 2 independent).

Frame semantics (must match deployment TF tree):
- ``sensor_frame``: the frame the incoming PointCloud2 points are expressed
  in (the cloud's own ``frame_id``).
- ``base_frame`` (default ``"base_link"``): the vehicle frame the FoveaMap
  runtime consumes (LiDARFrame ego convention: X forward, Y left, Z up).
- ``world_frame`` (default ``"map"``): the stable mapping frame; the runtime
  pose maps ego -> world.

Policy:
- When ``sensor_frame == base_frame`` no transform is applied (explicitly
  recorded in provenance, not silently assumed elsewhere).
- Otherwise a transform for the cloud timestamp is REQUIRED from the injected
  :class:`TransformProvider`. Missing transforms raise
  :class:`MissingTransformError` (typed failure -> frame drop with metrics);
  identity is NEVER silently substituted.
- Stale transforms (older than ``max_tf_age_s`` relative to the cloud stamp)
  are rejected unless ``allow_stale_tf`` is set, in which case use is recorded
  in provenance. Timestamps are never silently discarded or mixed: the output
  LiDARFrame keeps the cloud stamp.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence
import math

import numpy as np

from foveamap.core.exceptions import DataAdapterError


class MissingTransformError(DataAdapterError):
    """Required TF transform is unavailable for the requested timestamp."""


class StaleTransformError(DataAdapterError):
    """Available TF transform is older than the configured max age."""


@dataclass(frozen=True)
class RigidTransform:
    """Timestamped rigid transform parent <- child (translation + quaternion)."""

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    stamp: float = 0.0
    parent_frame: str = ""
    child_frame: str = ""

    def matrix(self) -> np.ndarray:
        """Homogeneous (4, 4) float64 matrix."""
        x, y, z, w = (float(v) for v in self.rotation_xyzw)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1e-12:
            raise DataAdapterError(f"Degenerate TF quaternion {self.rotation_xyzw!r}")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = np.asarray(self.translation, dtype=np.float64).reshape(3)
        return M


class TransformProvider(ABC):
    """Source of timestamped transforms (tf2 buffer in production, fake in tests)."""

    @abstractmethod
    def lookup(self, parent_frame: str, child_frame: str, stamp: float) -> RigidTransform:
        """Return parent <- child at ``stamp`` or raise MissingTransformError."""
        ...


@dataclass
class DictTransformProvider(TransformProvider):
    """In-memory provider for tests/tools: exact (parent, child) entries."""

    transforms: dict[tuple[str, str], RigidTransform] = field(default_factory=dict)

    def lookup(self, parent_frame: str, child_frame: str, stamp: float) -> RigidTransform:
        try:
            return self.transforms[(parent_frame, child_frame)]
        except KeyError:
            raise MissingTransformError(
                f"No transform {child_frame!r} -> {parent_frame!r} at t={stamp}"
            ) from None


@dataclass(frozen=True)
class FramePolicy:
    """Validated TF/frame configuration (populated from ROS params in config.py)."""

    base_frame: str = "base_link"
    world_frame: str = "map"
    require_tf: bool = True
    max_tf_age_s: float = 0.2
    allow_stale_tf: bool = False

    def __post_init__(self) -> None:
        if not self.base_frame or not self.world_frame:
            raise DataAdapterError("base_frame and world_frame must be non-empty")
        if self.max_tf_age_s < 0:
            raise DataAdapterError(f"max_tf_age_s must be non-negative, got {self.max_tf_age_s}")


def resolve_ego_points(
    pts_sensor: np.ndarray,
    sensor_frame: str,
    stamp: float,
    policy: FramePolicy,
    provider: TransformProvider | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Express sensor-frame points in the base (ego) frame.

    Returns ``(pts_ego, provenance)``. Raises MissingTransformError /
    StaleTransformError instead of guessing when a required transform is
    absent or stale. Provenance records exactly what happened.
    """
    pts = np.asarray(pts_sensor, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3), {"tf": "empty_cloud_no_transform", "sensor_frame": sensor_frame}
    if not sensor_frame:
        raise DataAdapterError("PointCloud2 has empty frame_id; cannot resolve coordinates")
    if sensor_frame == policy.base_frame:
        return pts.reshape(-1, 3), {
            "tf": "identity_same_frame",
            "sensor_frame": sensor_frame,
            "base_frame": policy.base_frame,
        }
    if provider is None:
        if policy.require_tf:
            raise MissingTransformError(
                f"Transform {sensor_frame!r} -> {policy.base_frame!r} required but no provider configured"
            )
        return pts.reshape(-1, 3), {"tf": "unchecked_passthrough_tf_not_required"}
    tf = provider.lookup(policy.base_frame, sensor_frame, stamp)
    age = float(stamp) - float(tf.stamp)
    if age < -1e-6:
        raise StaleTransformError(f"Transform from the future (age {age:.3f}s) for t={stamp}")
    stale = age > policy.max_tf_age_s
    if stale and not policy.allow_stale_tf:
        raise StaleTransformError(
            f"Transform age {age:.3f}s exceeds max_tf_age_s={policy.max_tf_age_s} for t={stamp}"
        )
    M = tf.matrix()
    R, t = M[:3, :3], M[:3, 3]
    Pts = pts.reshape(-1, 3)
    ego = Pts @ R.T + t
    return ego, {
        "tf": "transformed_stale_accepted" if stale else "transformed",
        "sensor_frame": sensor_frame,
        "base_frame": policy.base_frame,
        "tf_stamp": float(tf.stamp),
        "tf_age_s": age,
    }


def resolve_world_pose(
    base_frame: str,
    world_frame: str,
    stamp: float,
    policy: FramePolicy,
    provider: TransformProvider | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve the ego/base -> world/map pose for ``stamp``.

    Returns ``(pose_4x4, provenance)``. The SAME cloud timestamp is used for
    the sensor->base and base->world lookups, so frames can never mix
    timestamps. Rules:
    - ``world_frame == base_frame``: explicit same-frame operation, identity
      pose with ``local_map_origin`` provenance (valid and documented).
    - Otherwise a provider lookup ``(world_frame, base_frame)`` at ``stamp``
      is REQUIRED: missing -> MissingTransformError, stale -> StaleTransformError
      (unless ``allow_stale_tf``), future-dated -> StaleTransformError. The
      ``provider is None`` case therefore fails loudly instead of silently
      substituting identity.
    """
    if not base_frame or not world_frame:
        raise DataAdapterError("base_frame and world_frame must be non-empty for pose resolution")
    if world_frame == base_frame:
        return np.eye(4, dtype=np.float64), {
            "pose": "identity_same_frame_operation",
            "base_frame": base_frame,
            "world_frame": world_frame,
        }
    if provider is None:
        raise MissingTransformError(
            f"World transform {base_frame!r} -> {world_frame!r} required at t={stamp} "
            "but no TF provider is configured (configure world_frame == base_frame "
            "for explicit local-map operation)"
        )
    tf = provider.lookup(world_frame, base_frame, stamp)
    age = float(stamp) - float(tf.stamp)
    if age < -1e-6:
        raise StaleTransformError(f"World transform from the future (age {age:.3f}s) for t={stamp}")
    stale = age > policy.max_tf_age_s
    if stale and not policy.allow_stale_tf:
        raise StaleTransformError(
            f"World transform age {age:.3f}s exceeds max_tf_age_s={policy.max_tf_age_s} for t={stamp}"
        )
    return tf.matrix(), {
        "pose": "transformed_stale_accepted" if stale else "transformed",
        "base_frame": base_frame,
        "world_frame": world_frame,
        "tf_stamp": float(tf.stamp),
        "tf_age_s": age,
    }
