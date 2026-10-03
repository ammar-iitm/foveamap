"""FoveaMap ROS 2 integration adapter (Phase 8).

Thin translation layer between ROS 2 messages and the canonical FoveaMap
contracts. This package MUST NOT contain mapping, perception, or traversability
algorithms: all computation stays in :mod:`foveamap`, which remains fully usable
without ROS 2 installed (no module in this package is imported by core code,
and importing this package never requires ``rclpy``).

Layout:
    pointcloud.py  PointCloud2 <-> arrays/LiDARFrame (no rclpy dependency)
    frames.py      TF/frame policy (transform math + provider interface)
    config.py      ROS parameter dict -> FoveaMapConfig (+ ROS IO settings)
    qos.py         QoS descriptors + documented rationale (rclpy optional)
    messages.py    snapshot -> output payloads (/grid, /points_labeled, /metrics)
    diagnostics.py stage-separated metrics aggregation
    node.py        orchestration (ROS-independent core + guarded rclpy wrapper)
    msg/           future .msg definitions for a colcon message package
"""

__all__ = ["ROS2_AVAILABLE", "ROS2_MISSING_REASON"]

try:
    import rclpy  # noqa: F401
    ROS2_AVAILABLE = True
    ROS2_MISSING_REASON = ""
except Exception as exc:  # pragma: no cover - environment dependent
    ROS2_AVAILABLE = False
    ROS2_MISSING_REASON = f"rclpy unavailable: {exc}"
