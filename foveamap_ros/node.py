"""ROS 2 node orchestration (Phase 8).

Two layers:
- :class:`FoveaMapNodeCore`: ROS-independent pipeline orchestration —
  bounded input queue (DROP_OLDEST/DROP_NEWEST), PointCloud2 conversion with
  TF policy, ``FoveaMapRuntime`` processing, snapshot publication payloads,
  and metrics. Fully unit-testable without ROS 2.
- :class:`FoveaMapRosNode`: thin rclpy wrapper (subscriptions, publishers,
  parameters, lifecycle). Importing or instantiating it without rclpy raises
  an explicit error; it is only exercised by ROS-2-guarded tests.

Lifecycle states: CREATED -> CONFIGURED -> ACTIVE -> (INACTIVE on
deactivate) -> SHUTDOWN. Only ACTIVE accepts frames. A failed frame never
partially updates the map: conversion/TF failures happen before
``runtime.process``; runtime exceptions are counted and the previous map is
kept.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from foveamap.core.contracts import LiDARFrame, MapSnapshot
from foveamap.core.exceptions import DataAdapterError, FoveaMapError
from foveamap.runtime.runtime import FoveaMapRuntime
from .config import RosNodeConfig, from_ros_params
from .diagnostics import MetricsAggregator
from .frames import MissingTransformError, TransformProvider, resolve_ego_points, resolve_world_pose
from .messages import build_grid_payload, build_points_payload
from .pointcloud import cloud_to_arrays

CREATED = "CREATED"
CONFIGURED = "CONFIGURED"
ACTIVE = "ACTIVE"
INACTIVE = "INACTIVE"
SHUTDOWN = "SHUTDOWN"


@dataclass
class QueuedCloud:
    msg: Any
    enqueued_monotonic: float = 0.0


class FoveaMapNodeCore:
    """ROS-independent orchestration around an owned FoveaMapRuntime."""

    def __init__(
        self,
        config: RosNodeConfig | None = None,
        *,
        runtime: FoveaMapRuntime | None = None,
        tf_provider: TransformProvider | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config if config is not None else RosNodeConfig()
        self.tf_provider = tf_provider
        self._clock = clock if clock is not None else time.monotonic
        self.runtime = runtime if runtime is not None else FoveaMapRuntime(config=self.config.core)
        self.state = CREATED
        self.metrics = MetricsAggregator()
        self._queue: deque[QueuedCloud] = deque()
        self.last_snapshot: MapSnapshot | None = None
        self.last_outputs: dict[str, Any] = {}
        self.device_string = str(self.runtime.device_ctx.device)

    # ------------------------------------------------------------- lifecycle
    def configure(self) -> None:
        # SHUTDOWN is terminal: a shut-down node must be recreated, never revived.
        if self.state not in (CREATED, INACTIVE):
            raise FoveaMapError(f"configure() illegal from {self.state}")
        # Startup validation: config already validated by dataclasses; verify
        # the perception backend actually constructed (missing checkpoint
        # raises here, never as silent untrained inference).
        backend = getattr(self.runtime, "perception", None)
        if backend is None or not getattr(backend, "is_ready", False):
            raise FoveaMapError("Perception backend not ready; refusing to configure")
        self.state = CONFIGURED

    def activate(self) -> None:
        if self.state != CONFIGURED:
            raise FoveaMapError(f"activate() illegal from {self.state}")
        self.state = ACTIVE

    def deactivate(self) -> None:
        if self.state == ACTIVE:
            self.state = INACTIVE

    def shutdown(self) -> None:
        self._queue.clear()
        self.state = SHUTDOWN

    # ---------------------------------------------------------------- ingest
    def submit(self, msg: Any) -> bool:
        """Enqueue one cloud; False when dropped by bounded backpressure."""
        self.metrics.note_received()
        if self.state != ACTIVE:
            self.metrics.note_dropped("inactive")
            return False
        if len(self._queue) >= self.config.io.max_queue:
            if self.config.io.drop_policy == "oldest":
                self._queue.popleft()
                self.metrics.note_dropped("backpressure_oldest")
            else:
                self.metrics.note_dropped("backpressure_newest")
                return False
        self._queue.append(QueuedCloud(msg=msg, enqueued_monotonic=self._clock()))
        return True

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    def spin_once(self) -> MapSnapshot | None:
        """Process at most one queued cloud synchronously (deterministic)."""
        if self.state != ACTIVE or not self._queue:
            return None
        item = self._queue.popleft()
        t0 = self._clock()
        try:
            frame = self._convert(item.msg)
        except (DataAdapterError, MissingTransformError) as exc:
            self.metrics.note_error("INPUT", type(exc).__name__)
            self.metrics.observe_latency("INPUT", self._clock() - t0)
            return None
        except Exception as exc:  # never let one frame kill the node
            self.metrics.note_error("INPUT", f"unexpected_{type(exc).__name__}")
            return None
        self.metrics.observe_latency("INPUT", self._clock() - t0)
        t1 = self._clock()
        try:
            snapshot = self.runtime.process(frame)
        except FoveaMapError as exc:
            self.metrics.note_error("RUNTIME", type(exc).__name__)
            return None
        except Exception as exc:
            self.metrics.note_error("RUNTIME", f"unexpected_{type(exc).__name__}")
            return None
        self.metrics.observe_latency("RUNTIME", self._clock() - t1)
        t2 = self._clock()
        self.last_snapshot = snapshot
        self.metrics.note_frame_done(self._clock())
        self.last_outputs = self._publish(snapshot, frame)
        self.metrics.observe_latency("PUBLICATION", self._clock() - t2)
        return snapshot

    # -------------------------------------------------------------- convert
    def _convert(self, msg: Any) -> LiDARFrame:
        arrays = cloud_to_arrays(msg, intensity_mode=self.config.io.intensity_mode)
        pts_sensor = arrays["pts"]
        stamp = float(arrays["timestamp"])
        policy = self.config.io.frame_policy()
        ego, tf_prov = resolve_ego_points(
            pts_sensor,
            arrays["frame_id"],
            stamp,
            policy,
            self.tf_provider,
        )
        # Base -> world uses the SAME cloud timestamp: frames, sensor TF and
        # world TF can never mix timestamps. Identity pose is only legal for
        # explicitly configured same-frame (world == base) operation.
        pose, pose_prov = resolve_world_pose(
            policy.base_frame,
            policy.world_frame,
            stamp,
            policy,
            self.tf_provider,
        )
        origin = np.asarray(self.config.io.sensor_origin, dtype=np.float32)
        return LiDARFrame(
            pts=ego.astype(np.float32),
            intensity=arrays["intensity"],
            ring=arrays["ring"],
            pose=pose,
            sensor_origin=origin,
            timestamp=stamp,
            frame_id=str(arrays["frame_id"]),
            source_id="ros/node",
            time_offsets=arrays.get("time_offsets"),
            metadata={
                "intensity_provenance": arrays["intensity_provenance"],
                "ring_available": bool(arrays["ring_available"]),
                "timestamp_provenance": "ros_header_stamp",
                "time_provenance": arrays.get("time_provenance", "absent_no_per_point_timing"),
                "tf_provenance": tf_prov,
                "pose_provenance": pose_prov,
            },
        )

    # -------------------------------------------------------------- publish
    def _publish(self, snapshot: MapSnapshot, frame: LiDARFrame) -> dict[str, Any]:
        cost_max = int(self.config.core.terrain.traversable_cost_max)
        grid_payload = build_grid_payload(snapshot, cost_max=cost_max)
        perception = getattr(self.runtime, "last_perception", None)
        if perception is not None and getattr(perception, "num_points", 0):
            pts_world = getattr(frame, "pts", np.zeros((0, 3), dtype=np.float32))
            classes = np.asarray(perception.semantic_predictions).reshape(-1)
            conf = np.asarray(perception.point_confidence).reshape(-1)
            moving = np.asarray(perception.is_moving).reshape(-1)
        else:
            pts_world = np.zeros((0, 3), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.int32)
            conf = np.zeros((0,), dtype=np.float32)
            moving = np.zeros((0,), dtype=bool)
        points_msg = build_points_payload(
            pts_world, classes, conf, moving,
            frame_id=str(snapshot.frame_id), timestamp=float(snapshot.timestamp),
            max_points=int(self.config.io.publish_points_max),
        )
        metrics_extra = {
            "device": self.device_string,
            "cuda_note": ("cuda_unavailable_cpu_execution"
                          if "cuda" not in self.device_string else "cuda_device_selected"),
            "queue_depth": len(self._queue),
            "grid_bytes": int(grid_payload["serialized_size_bytes"]),
            "temporal": dict(snapshot.temporal_metadata) if snapshot.temporal_metadata else {},
        }
        metrics = self.metrics.report(metrics_extra)
        return {
            self.config.io.grid_topic: grid_payload,
            self.config.io.points_topic: points_msg,
            self.config.io.metrics_topic: metrics,
        }


class FoveaMapRosNode:
    """Thin rclpy wrapper (requires a ROS 2 environment)."""

    def __init__(self, ros_params: dict[str, Any] | None = None, **kwargs: Any) -> None:
        try:
            import rclpy  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "FoveaMapRosNode requires rclpy (ROS 2 not installed on this machine)"
            ) from exc
        config = from_ros_params(ros_params or {})
        tf_provider = kwargs.get("tf_provider")
        self.core = FoveaMapNodeCore(config=config, tf_provider=tf_provider)
        self.state = CREATED

    def configure(self) -> None:
        self.core.configure()
        self.state = CONFIGURED

    def activate(self) -> None:
        self.core.activate()
        self.state = ACTIVE

    def deactivate(self) -> None:
        self.core.deactivate()
        self.state = INACTIVE

    def shutdown(self) -> None:
        self.core.shutdown()
        self.state = SHUTDOWN
