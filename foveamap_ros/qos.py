"""QoS descriptors (rclpy independent) + rationale.

Rationale:
- LiDAR input: SENSOR_DATA-like (best-effort, volatile, depth 5). A stale
  sweep is worse than a dropped one for a 10 Hz stream; reliability would add
  latency without value.
- /foveamap/grid + /points_labeled: SENSOR_DATA-like (best-effort, volatile,
  depth 1). Snapshots are high-rate and idempotent; late subscribers get the
  next frame. Depth 1 also bounds DDS queueing on top of the node queue.
- /foveamap/metrics: reliable + transient-local, depth 1, so late-joining
  diagnostic tools immediately see the latest counters.

All choices are overridable via ``to_rclpy()`` kwargs; the defaults above are
what the node uses. ``to_rclpy()`` requires rclpy and raises a clear error
without it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QosProfile:
    reliability: str = "best_effort"  # "best_effort" | "reliable"
    durability: str = "volatile"  # "volatile" | "transient_local"
    history: str = "keep_last"  # "keep_last" | "keep_all"
    depth: int = 5
    deadline_s: float | None = None
    lifespan_s: float | None = None

    def __post_init__(self) -> None:
        if self.reliability not in ("best_effort", "reliable"):
            raise ValueError(f"reliability must be best_effort/reliable, got {self.reliability!r}")
        if self.durability not in ("volatile", "transient_local"):
            raise ValueError(f"durability must be volatile/transient_local, got {self.durability!r}")
        if self.history not in ("keep_last", "keep_all"):
            raise ValueError(f"history must be keep_last/keep_all, got {self.history!r}")
        if self.history == "keep_all":
            raise ValueError("keep_all is forbidden: queues must stay bounded")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")


LIDAR_INPUT_QOS = QosProfile(reliability="best_effort", durability="volatile",
                             history="keep_last", depth=5)
GRID_OUTPUT_QOS = QosProfile(reliability="best_effort", durability="volatile",
                             history="keep_last", depth=1)
POINTS_OUTPUT_QOS = QosProfile(reliability="best_effort", durability="volatile",
                               history="keep_last", depth=1)
METRICS_OUTPUT_QOS = QosProfile(reliability="reliable", durability="transient_local",
                                history="keep_last", depth=1)


def to_rclpy(profile: QosProfile) -> Any:
    """Convert to rclpy.qos.QoSProfile (requires a ROS 2 environment)."""
    try:
        from rclpy.qos import QoSProfile as RclpyQos, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    except Exception as exc:
        raise RuntimeError(f"rclpy QoS unavailable (ROS 2 not installed): {exc}") from exc
    return RclpyQos(
        reliability=(ReliabilityPolicy.RELIABLE if profile.reliability == "reliable"
                     else ReliabilityPolicy.BEST_EFFORT),
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if profile.durability == "transient_local"
                    else DurabilityPolicy.VOLATILE),
        history=HistoryPolicy.KEEP_LAST,
        depth=int(profile.depth),
    )
