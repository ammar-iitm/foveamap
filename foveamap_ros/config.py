"""ROS parameter mapping into the authoritative FoveaMapConfig (Phase 8).

Single-source rule: ROS 2 parameters are a transport encoding only. Every
value lands in :class:`FoveaMapConfig` (or the ROS IO envelope below) and is
validated by the existing frozen dataclasses. No duplicate grid/terrain/
dynamic semantics live here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from foveamap.core.config import (
    DynamicConfig,
    FoveaMapConfig,
    GridConfig,
    PerceptionConfig,
    PreprocessConfig,
    RuntimeConfig,
    SensorConfig,
    TerrainConfig,
    TierConfig,
)
from foveamap.core.exceptions import ConfigurationError
from .frames import FramePolicy


@dataclass(frozen=True)
class RosIOConfig:
    """ROS transport settings (topics/frames/queues), NOT map semantics."""

    input_topic: str = "/lidar/points"
    grid_topic: str = "/foveamap/grid"
    points_topic: str = "/foveamap/points_labeled"
    metrics_topic: str = "/foveamap/metrics"
    base_frame: str = "base_link"
    world_frame: str = "map"
    require_tf: bool = True
    max_tf_age_s: float = 0.2
    allow_stale_tf: bool = False
    max_queue: int = 2
    drop_policy: str = "oldest"  # "oldest" (DROP_OLDEST) or "newest"
    sensor_origin: tuple[float, float, float] = (0.0, 0.0, 1.73)
    intensity_mode: str = "auto"
    publish_points_max: int = 12000

    def __post_init__(self) -> None:
        for name in ("input_topic", "grid_topic", "points_topic", "metrics_topic"):
            if not getattr(self, name).startswith("/"):
                raise ConfigurationError(f"ROS topic {name} must be absolute, got {getattr(self, name)!r}")
        if not self.base_frame or not self.world_frame:
            raise ConfigurationError("base_frame and world_frame must be non-empty")
        if self.max_tf_age_s < 0:
            raise ConfigurationError(f"max_tf_age_s must be non-negative, got {self.max_tf_age_s}")
        if self.max_queue < 1:
            raise ConfigurationError(f"max_queue must be >= 1, got {self.max_queue}")
        if self.drop_policy not in ("oldest", "newest"):
            raise ConfigurationError(f"drop_policy must be 'oldest' or 'newest', got {self.drop_policy!r}")
        if len(self.sensor_origin) != 3:
            raise ConfigurationError("sensor_origin must have exactly 3 components")
        if self.publish_points_max < 0:
            raise ConfigurationError(f"publish_points_max must be >= 0, got {self.publish_points_max}")

    def frame_policy(self) -> FramePolicy:
        return FramePolicy(
            base_frame=self.base_frame,
            world_frame=self.world_frame,
            require_tf=self.require_tf,
            max_tf_age_s=self.max_tf_age_s,
            allow_stale_tf=self.allow_stale_tf,
        )


def _tier_list(value: Any) -> tuple[TierConfig, ...]:
    tiers = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            tiers.append(TierConfig(float(item[0]), float(item[1]), float(item[2])))
        elif isinstance(item, dict):
            tiers.append(TierConfig(float(item["cell_size_m"]), float(item["half_extent_m"]),
                                    float(item.get("alpha", 0.5))))
        else:
            raise ConfigurationError(f"Invalid tier spec {item!r}; use [cell_m, half_m, alpha] or dict")
    return tuple(tiers)


@dataclass
class RosNodeConfig:
    """Combined validated configuration: core truth + ROS envelope."""

    core: FoveaMapConfig = field(default_factory=FoveaMapConfig)
    io: RosIOConfig = field(default_factory=RosIOConfig)


def from_ros_params(params: dict[str, Any]) -> RosNodeConfig:
    """Build validated config from a flat ROS parameter dict.

    Unknown keys raise (typos must not silently pass). Nested sections use
    ``section.key`` names; ``grid.tiers`` accepts a list of
    ``[cell_m, half_m, alpha]`` triples.
    """
    params = dict(params)
    known = {
        "sensor.name", "sensor.n_rows", "sensor.n_cols", "sensor.fov_up_deg",
        "sensor.fov_down_deg", "sensor.sensor_height_m", "sensor.hz",
        "sensor.min_range_m", "sensor.max_range_m",
        "grid.profile", "grid.tiers", "grid.fuse",
        "perception.backend_type", "perception.checkpoint_path",
        "perception.num_classes", "perception.fp16",
        "perception.confidence_threshold",
        "terrain.vehicle_clearance_m", "terrain.step_threshold_m",
        "terrain.depression_threshold_m", "terrain.depression_window_m",
        "terrain.roughness_threshold_m", "terrain.slope_threshold_rad",
        "terrain.slope_critical_rad", "terrain.stale_age_threshold",
        "terrain.max_stale_age", "terrain.enable_ray_clearing",
        "terrain.free_clear_frames", "terrain.traversable_cost_max",
        "runtime.device", "runtime.grid_engine", "runtime.features_engine",
        "runtime.enable_profiling",
        "preprocess.enabled", "preprocess.min_range_m", "preprocess.max_range_m",
        "dynamic.activation_frames", "dynamic.missing_tolerance_frames",
        "dynamic.stale_frames", "dynamic.confidence_threshold",
        "dynamic.correspondence_distance_m", "dynamic.max_tracks",
        "io.input_topic", "io.grid_topic", "io.points_topic", "io.metrics_topic",
        "io.base_frame", "io.world_frame", "io.require_tf", "io.max_tf_age_s",
        "io.allow_stale_tf", "io.max_queue", "io.drop_policy",
        "io.sensor_origin", "io.intensity_mode", "io.publish_points_max",
    }
    unknown = sorted(set(params) - known)
    if unknown:
        raise ConfigurationError(f"Unknown ROS parameters (typo?): {unknown}")

    base = FoveaMapConfig()
    get = lambda k, d: params.get(k, d)  # noqa: E731

    sensor = SensorConfig(
        name=str(get("sensor.name", base.sensor.name)),
        n_rows=int(get("sensor.n_rows", base.sensor.n_rows)),
        n_cols=int(get("sensor.n_cols", base.sensor.n_cols)),
        fov_up_deg=float(get("sensor.fov_up_deg", base.sensor.fov_up_deg)),
        fov_down_deg=float(get("sensor.fov_down_deg", base.sensor.fov_down_deg)),
        sensor_height_m=float(get("sensor.sensor_height_m", base.sensor.sensor_height_m)),
        hz=float(get("sensor.hz", base.sensor.hz)),
        min_range_m=float(get("sensor.min_range_m", base.sensor.min_range_m)),
        max_range_m=float(get("sensor.max_range_m", base.sensor.max_range_m)),
    )
    if "grid.tiers" in params:
        grid = GridConfig(profile_name=str(get("grid.profile", "custom")),
                          tiers=_tier_list(params["grid.tiers"]),
                          fuse=bool(get("grid.fuse", True)))
    else:
        grid = GridConfig(profile_name=str(get("grid.profile", base.grid.profile_name)),
                          tiers=base.grid.tiers, fuse=bool(get("grid.fuse", base.grid.fuse)))
    perception = PerceptionConfig(
        backend_type=str(get("perception.backend_type", base.perception.backend_type)),
        checkpoint_path=params.get("perception.checkpoint_path", base.perception.checkpoint_path),
        num_classes=int(get("perception.num_classes", base.perception.num_classes)),
        fp16=bool(get("perception.fp16", base.perception.fp16)),
        confidence_threshold=float(get("perception.confidence_threshold", base.perception.confidence_threshold)),
    )
    terrain = TerrainConfig(
        vehicle_clearance_m=float(get("terrain.vehicle_clearance_m", base.terrain.vehicle_clearance_m)),
        step_threshold_m=float(get("terrain.step_threshold_m", base.terrain.step_threshold_m)),
        depression_threshold_m=float(get("terrain.depression_threshold_m", base.terrain.depression_threshold_m)),
        depression_window_m=float(get("terrain.depression_window_m", base.terrain.depression_window_m)),
        roughness_threshold_m=float(get("terrain.roughness_threshold_m", base.terrain.roughness_threshold_m)),
        slope_threshold_rad=float(get("terrain.slope_threshold_rad", base.terrain.slope_threshold_rad)),
        slope_critical_rad=float(get("terrain.slope_critical_rad", base.terrain.slope_critical_rad)),
        stale_age_threshold=int(get("terrain.stale_age_threshold", base.terrain.stale_age_threshold)),
        max_stale_age=int(get("terrain.max_stale_age", base.terrain.max_stale_age)),
        enable_ray_clearing=bool(get("terrain.enable_ray_clearing", base.terrain.enable_ray_clearing)),
        free_clear_frames=int(get("terrain.free_clear_frames", base.terrain.free_clear_frames)),
        traversable_cost_max=int(get("terrain.traversable_cost_max", base.terrain.traversable_cost_max)),
    )
    runtime = RuntimeConfig(
        device=str(get("runtime.device", base.runtime.device)),
        grid_engine=str(get("runtime.grid_engine", base.runtime.grid_engine)),
        features_engine=str(get("runtime.features_engine", base.runtime.features_engine)),
        enable_profiling=bool(get("runtime.enable_profiling", base.runtime.enable_profiling)),
    )
    preprocess = PreprocessConfig(
        enabled=bool(get("preprocess.enabled", base.preprocess.enabled)),
        min_range_m=float(get("preprocess.min_range_m", base.preprocess.min_range_m)),
        max_range_m=float(get("preprocess.max_range_m", base.preprocess.max_range_m)),
    )
    dynamic = DynamicConfig(
        activation_frames=int(get("dynamic.activation_frames", base.dynamic.activation_frames)),
        missing_tolerance_frames=int(get("dynamic.missing_tolerance_frames", base.dynamic.missing_tolerance_frames)),
        stale_frames=int(get("dynamic.stale_frames", base.dynamic.stale_frames)),
        confidence_threshold=float(get("dynamic.confidence_threshold", base.dynamic.confidence_threshold)),
        correspondence_distance_m=float(get("dynamic.correspondence_distance_m", base.dynamic.correspondence_distance_m)),
        max_tracks=int(get("dynamic.max_tracks", base.dynamic.max_tracks)),
    )
    io = RosIOConfig(
        input_topic=str(get("io.input_topic", "/lidar/points")),
        grid_topic=str(get("io.grid_topic", "/foveamap/grid")),
        points_topic=str(get("io.points_topic", "/foveamap/points_labeled")),
        metrics_topic=str(get("io.metrics_topic", "/foveamap/metrics")),
        base_frame=str(get("io.base_frame", "base_link")),
        world_frame=str(get("io.world_frame", "map")),
        require_tf=bool(get("io.require_tf", True)),
        max_tf_age_s=float(get("io.max_tf_age_s", 0.2)),
        allow_stale_tf=bool(get("io.allow_stale_tf", False)),
        max_queue=int(get("io.max_queue", 2)),
        drop_policy=str(get("io.drop_policy", "oldest")),
        sensor_origin=tuple(float(v) for v in get("io.sensor_origin", (0.0, 0.0, 1.73))),
        intensity_mode=str(get("io.intensity_mode", "auto")),
        publish_points_max=int(get("io.publish_points_max", 12000)),
    )
    core = FoveaMapConfig(sensor=sensor, grid=grid, perception=perception,
                          terrain=terrain, runtime=runtime, preprocess=preprocess,
                          dynamic=dynamic)
    return RosNodeConfig(core=core, io=io)
