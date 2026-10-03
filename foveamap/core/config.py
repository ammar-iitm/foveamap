"""Structured, validated configuration dataclasses for FoveaMap.

Extracts all hardcoded parameters and magic numbers into explicit,
composable, and type-checked configuration objects.

RUNTIME INTEGRATION:
    These configuration dataclasses establish formal schemas, defaults, and invariant
    validation rules for the entire FoveaMap pipeline. The canonical runtime
    (`FoveaMapRuntime`) directly consumes `FoveaMapConfig` to configure sensors,
    grid resolution tiers, neural network perception backends, and traversability heuristics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

from .exceptions import ConfigurationError

# Default traversability priors (0 = free ... 254 = lethal, UNKNOWN = 255)
DEFAULT_COST_PRIOR = [200] * 256
# Road=0, Parking=2, Sidewalk=1, Terrain=3, Veg=4, Building=5, Pole=6, Vehicle=7, Person=8
DEFAULT_CLASS_COSTS = {
    0: 0,     # ROAD
    1: 110,   # SIDEWALK
    2: 10,    # PARKING
    3: 150,   # TERRAIN
    4: 254,   # VEGETATION
    5: 254,   # BUILDING
    6: 254,   # POLE
    7: 254,   # VEHICLE
    8: 254,   # PERSON
}
for cls_id, cost in DEFAULT_CLASS_COSTS.items():
    DEFAULT_COST_PRIOR[cls_id] = cost


@dataclass(frozen=True)
class TierConfig:
    """Configuration for a single concentric resolution tier.

    Attributes:
        cell_size_m: cell width and height in metres.
        half_extent_m: half-extent of the tier window in metres.
        alpha: temporal EMA update weight for static observations (0 < alpha <= 1.0).
    """
    cell_size_m: float
    half_extent_m: float
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.cell_size_m <= 0:
            raise ConfigurationError(f"cell_size_m must be positive, got {self.cell_size_m}")
        if self.half_extent_m <= 0:
            raise ConfigurationError(f"half_extent_m must be positive, got {self.half_extent_m}")
        if not (0.0 < self.alpha <= 1.0):
            raise ConfigurationError(f"alpha must be in (0, 1], got {self.alpha}")


@dataclass(frozen=True)
class GridConfig:
    """Configuration for the multi-tier foveated grid engine.

    Attributes:
        profile_name: descriptor name ("spec", "graded", "custom", etc.).
        tiers: tuple of TierConfig ordered from finest to coarsest.
        fuse: whether temporal fusion is enabled across frames.
    """
    profile_name: str = "spec"
    tiers: tuple[TierConfig, ...] = field(default_factory=lambda: (
        TierConfig(0.05, 10.0, alpha=0.3),
        TierConfig(0.50, 100.0, alpha=0.5),
    ))
    fuse: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def base_resolution(self) -> float:
        return self.tiers[0].cell_size_m

    @property
    def coarse_resolution(self) -> float:
        return self.tiers[-1].cell_size_m

    def validate(self) -> None:
        if not self.tiers:
            raise ConfigurationError("GridConfig must contain at least one tier")

        base = self.tiers[0].cell_size_m
        coarse = self.tiers[-1].cell_size_m

        for k, t in enumerate(self.tiers):
            ratio = round(t.cell_size_m / base)
            if abs(ratio * base - t.cell_size_m) > 1e-9:
                raise ConfigurationError(f"Tier {k} cell size {t.cell_size_m} is not an integer multiple of base {base}")
            if abs(round(coarse / t.cell_size_m) * t.cell_size_m - coarse) > 1e-9:
                raise ConfigurationError(f"Coarsest cell {coarse} is not a multiple of tier {k} cell {t.cell_size_m}")
            if abs(round(2 * t.half_extent_m / coarse) * coarse - 2 * t.half_extent_m) > 1e-9:
                raise ConfigurationError(f"Tier {k} extent {t.half_extent_m} must be whole coarse cells")
            if k > 0:
                prev = self.tiers[k - 1]
                if t.cell_size_m <= prev.cell_size_m:
                    raise ConfigurationError(f"Tier {k} cell size must exceed tier {k-1}")
                if t.half_extent_m <= prev.half_extent_m:
                    raise ConfigurationError(f"Tier {k} half extent must exceed tier {k-1}")

    @classmethod
    def from_preset(cls, profile: str = "spec", fuse: bool = True) -> GridConfig:
        presets = {
            "spec": (
                TierConfig(0.05, 10.0, alpha=0.3),
                TierConfig(0.50, 100.0, alpha=0.5),
            ),
            "graded": (
                TierConfig(0.05, 10.0, alpha=0.3),
                TierConfig(0.10, 25.0, alpha=0.4),
                TierConfig(0.50, 100.0, alpha=0.5),
            ),
            "uniform5": (
                TierConfig(0.05, 100.0, alpha=0.5),
            ),
            "uniform50": (
                TierConfig(0.50, 100.0, alpha=0.5),
            ),
        }
        if profile not in presets:
            raise ConfigurationError(f"Unknown grid profile {profile!r}; available presets: {list(presets.keys())}")
        return cls(profile_name=profile, tiers=presets[profile], fuse=fuse)


@dataclass(frozen=True)
class SensorConfig:
    """Sensor physical and scan geometry parameters."""
    name: str = "sim"
    n_rows: int = 64
    n_cols: int = 1024
    fov_up_deg: float = 2.0
    fov_down_deg: float = -24.9
    sensor_height_m: float = 1.73
    hz: float = 10.0
    min_range_m: float = 0.5
    max_range_m: float = 100.0
    source_description: str = ""

    def __post_init__(self) -> None:
        if self.n_rows <= 0 or self.n_cols <= 0:
            raise ConfigurationError(f"Sensor dimensions must be positive, got {self.n_rows}x{self.n_cols}")
        if self.fov_up_deg <= self.fov_down_deg:
            raise ConfigurationError(f"fov_up_deg ({self.fov_up_deg}) must exceed fov_down_deg ({self.fov_down_deg})")
        if self.min_range_m < 0 or self.max_range_m <= self.min_range_m:
            raise ConfigurationError(f"Invalid range bounds: [{self.min_range_m}, {self.max_range_m}]")
        if self.hz <= 0:
            raise ConfigurationError(f"hz must be positive, got {self.hz}")


def find_default_checkpoint() -> str | None:
    """Locate default RangeUNet checkpoint file in standard repository location if present."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(pkg_dir, "..", "..", "checkpoints", "range_unet.pt")),
        os.path.abspath(os.path.join(pkg_dir, "..", "checkpoints", "range_unet.pt")),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@dataclass(frozen=True)
class PerceptionConfig:
    """Perception model and inference settings."""
    backend_type: str = "range_unet"
    checkpoint_path: str | None = field(default_factory=find_default_checkpoint)
    num_classes: int = 9
    fp16: bool = True
    confidence_threshold: float = 0.5
    active_classes: tuple[bool, ...] = field(default_factory=lambda: (True,) * 9)
    strict_validation: bool = False

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ConfigurationError(f"num_classes must be positive, got {self.num_classes}")
        if len(self.active_classes) != self.num_classes:
            raise ConfigurationError(f"active_classes length ({len(self.active_classes)}) != num_classes ({self.num_classes})")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")


@dataclass(frozen=True)
class TerrainConfig:
    """Geometric and traversability analysis thresholds.

    traversable_cost_max (unitless cost, valid [0, 254], default 180) is the
    authoritative traversal boundary: a known, non-dynamic cell with
    cost < traversable_cost_max is traversable. It aligns with the derivation
    rule that clamps step-impaired drivable cells to >= 180 (non-traversable).
    """
    vehicle_clearance_m: float = 2.5
    step_threshold_m: float = 0.08
    depression_threshold_m: float = 0.05
    depression_window_m: float = 2.5
    roughness_threshold_m: float = 0.04
    slope_threshold_rad: float = 0.25
    slope_critical_rad: float = 0.40
    stale_age_threshold: int = 20
    max_stale_age: int = 100
    enable_ray_clearing: bool = False
    free_clear_frames: int = 3
    traversable_cost_max: int = 180
    cost_priors: tuple[int, ...] = field(default_factory=lambda: tuple(DEFAULT_COST_PRIOR))
    def __post_init__(self) -> None:
        if self.vehicle_clearance_m <= 0:
            raise ConfigurationError(f"vehicle_clearance_m must be positive, got {self.vehicle_clearance_m}")
        if self.step_threshold_m <= 0:
            raise ConfigurationError(f"step_threshold_m must be positive, got {self.step_threshold_m}")
        if self.depression_threshold_m <= 0:
            raise ConfigurationError(f"depression_threshold_m must be positive, got {self.depression_threshold_m}")
        if self.depression_window_m <= 0:
            raise ConfigurationError(f"depression_window_m must be positive, got {self.depression_window_m}")
        if self.roughness_threshold_m <= 0:
            raise ConfigurationError(f"roughness_threshold_m must be positive, got {self.roughness_threshold_m}")
        if self.slope_threshold_rad <= 0:
            raise ConfigurationError(f"slope_threshold_rad must be positive, got {self.slope_threshold_rad}")
        if self.slope_critical_rad <= self.slope_threshold_rad:
            raise ConfigurationError(f"slope_critical_rad ({self.slope_critical_rad}) must exceed slope_threshold_rad ({self.slope_threshold_rad})")
        if self.stale_age_threshold < 0:
            raise ConfigurationError(f"stale_age_threshold must be non-negative, got {self.stale_age_threshold}")
        if self.max_stale_age <= self.stale_age_threshold:
            raise ConfigurationError(f"max_stale_age ({self.max_stale_age}) must exceed stale_age_threshold ({self.stale_age_threshold})")
        if self.free_clear_frames <= 0:
            raise ConfigurationError(f"free_clear_frames must be positive, got {self.free_clear_frames}")
        if not (0 <= self.traversable_cost_max <= 254):
            raise ConfigurationError(f"traversable_cost_max must be in [0, 254], got {self.traversable_cost_max}")
        if len(self.cost_priors) != 256:
            raise ConfigurationError("cost_priors must have length 256")

@dataclass(frozen=True)
class DynamicConfig:
    """Temporal dynamic world-model thresholds (Phase 6).

    All frame counts are in processed-frame units. Confidence lives in [0, 1].
    Distances are metres in the world frame; correspondence operates on
    post-scroll grid coordinates and world positions, so ego motion is already
    compensated by the foveated window scroll.

    Attributes:
        activation_frames: consecutive observed frames required to promote
            OBSERVED -> ACTIVE_DYNAMIC.
        missing_tolerance_frames: frames a confirmed track may go unobserved
            while remaining TEMPORARILY_MISSING (still dynamic-occupied).
        stale_frames: missing frames after which a track becomes STALE
            (released, static fallback) and beyond which it is REMOVED.
        confidence_threshold: dynamic confidence required alongside
            activation_frames for ACTIVE_DYNAMIC promotion.
        initial_confidence: confidence assigned to a newly created track.
        hit_increment: confidence added per observed frame (capped at 1.0).
        decay_factor: multiplicative confidence decay per missing frame.
        correspondence_distance_m: max world-frame distance for associating a
            dynamic observation with an existing track. Association is
            cross-tier by world proximity: the foveated engines report the
            same physical object in several tiers (integer-lattice mip-up),
            and all of those observations feed one track anchored at the
            finest reporting tier.
        dynamic_classes: semantic class IDs treated as dynamic evidence
            (default vehicle/person, matching the perception motion gate).
        max_tracks: hard bound on live dynamic tracks (deterministic eviction
            of stalest tracks when exceeded).
    """
    activation_frames: int = 2
    missing_tolerance_frames: int = 2
    stale_frames: int = 4
    confidence_threshold: float = 0.5
    initial_confidence: float = 0.5
    hit_increment: float = 0.3
    decay_factor: float = 0.75
    correspondence_distance_m: float = 1.5
    dynamic_classes: tuple[int, ...] = (7, 8)
    max_tracks: int = 20000

    def __post_init__(self) -> None:
        if self.activation_frames < 1:
            raise ConfigurationError(f"activation_frames must be >= 1, got {self.activation_frames}")
        if self.missing_tolerance_frames < 0:
            raise ConfigurationError(f"missing_tolerance_frames must be >= 0, got {self.missing_tolerance_frames}")
        if self.stale_frames <= self.missing_tolerance_frames:
            raise ConfigurationError(
                f"stale_frames ({self.stale_frames}) must exceed "
                f"missing_tolerance_frames ({self.missing_tolerance_frames})"
            )
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ConfigurationError(f"confidence_threshold must be in (0, 1], got {self.confidence_threshold}")
        if not (0.0 <= self.initial_confidence <= 1.0):
            raise ConfigurationError(f"initial_confidence must be in [0, 1], got {self.initial_confidence}")
        if not (0.0 < self.hit_increment <= 1.0):
            raise ConfigurationError(f"hit_increment must be in (0, 1], got {self.hit_increment}")
        if not (0.0 < self.decay_factor < 1.0):
            raise ConfigurationError(f"decay_factor must be in (0, 1), got {self.decay_factor}")
        if self.correspondence_distance_m <= 0:
            raise ConfigurationError(f"correspondence_distance_m must be positive, got {self.correspondence_distance_m}")
        if not self.dynamic_classes:
            raise ConfigurationError("dynamic_classes must be non-empty")
        if self.max_tracks < 1:
            raise ConfigurationError(f"max_tracks must be >= 1, got {self.max_tracks}")


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution backend and runtime parameters."""
    device: str = "auto"
    grid_engine: str = "numpy"
    features_engine: str = "numpy"
    enable_profiling: bool = False

    def __post_init__(self) -> None:
        if self.grid_engine not in ("numpy", "torch"):
            raise ConfigurationError(f"grid_engine must be 'numpy' or 'torch', got {self.grid_engine!r}")
        if self.features_engine not in ("numpy", "torch"):
            raise ConfigurationError(f"features_engine must be 'numpy' or 'torch', got {self.features_engine!r}")

    @classmethod
    def cpu_profile(cls) -> RuntimeConfig:
        """Explicit CPU/development profile: all-NumPy on CPU, no CUDA needed."""
        return cls(device="cpu", grid_engine="numpy", features_engine="numpy")

    @classmethod
    def gpu_profile(cls) -> RuntimeConfig:
        """Explicit GPU/production profile: Torch engines with auto device.

        ``device="auto"`` selects CUDA where physically present and falls back
        to CPU otherwise (GPU performance itself is validated separately).
        """
        return cls(device="auto", grid_engine="torch", features_engine="torch")


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for raw LiDAR point cloud preprocessing.

    Attributes:
        enabled: whether preprocessing filtering is active.
        min_range_m: minimum radial range from sensor in metres.
        max_range_m: maximum radial range from sensor in metres.
        z_min_m: optional minimum Z height in ego frame in metres.
        z_max_m: optional maximum Z height in ego frame in metres.
        remove_invalid: whether to discard non-finite points (NaN, Inf).
        remove_self_hits: whether to filter out points near the vehicle ego center.
        self_hit_radius_m: radius in metres in ego XY plane to consider vehicle self-hits.
    """
    enabled: bool = True
    min_range_m: float = 0.5
    max_range_m: float = 100.0
    z_min_m: float | None = None
    z_max_m: float | None = None
    remove_invalid: bool = True
    remove_self_hits: bool = True
    self_hit_radius_m: float = 1.0

    def __post_init__(self) -> None:
        if self.min_range_m < 0:
            raise ConfigurationError(f"min_range_m must be non-negative, got {self.min_range_m}")
        if self.max_range_m <= self.min_range_m:
            raise ConfigurationError(f"max_range_m ({self.max_range_m}) must exceed min_range_m ({self.min_range_m})")
        if self.z_min_m is not None and self.z_max_m is not None and self.z_max_m <= self.z_min_m:
            raise ConfigurationError(f"z_max_m ({self.z_max_m}) must exceed z_min_m ({self.z_min_m})")
        if self.self_hit_radius_m < 0:
            raise ConfigurationError(f"self_hit_radius_m must be non-negative, got {self.self_hit_radius_m}")


@dataclass(frozen=True)
class FoveaMapConfig:
    """Top-level unified system configuration."""
    sensor: SensorConfig = field(default_factory=SensorConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    dynamic: DynamicConfig = field(default_factory=DynamicConfig)

    @classmethod
    def cpu_dev(cls) -> FoveaMapConfig:
        """Explicit CPU development profile: all-NumPy, CPU device, standard preprocessing."""
        return cls(
            runtime=RuntimeConfig.cpu_profile(),
            preprocess=PreprocessConfig(enabled=True),
        )

    @classmethod
    def gpu_dev(cls, checkpoint_path: str | None = None) -> FoveaMapConfig:
        """Explicit GPU production profile: Torch engines with auto device and FP16 inference."""
        return cls(
            runtime=RuntimeConfig.gpu_profile(),
            perception=PerceptionConfig(
                backend_type="range_unet",
                checkpoint_path=checkpoint_path,
                fp16=True,
            ),
            preprocess=PreprocessConfig(enabled=True),
        )

    @classmethod
    def benchmark(cls, grid_engine: str = "torch", enable_profiling: bool = True) -> FoveaMapConfig:
        """Benchmark profile: runtime profiling enabled, specified grid engine."""
        return cls(
            runtime=RuntimeConfig(
                device="auto",
                grid_engine=grid_engine,
                features_engine=grid_engine,
                enable_profiling=enable_profiling,
            ),
            preprocess=PreprocessConfig(enabled=True),
        )

    @classmethod
    def demo(cls) -> FoveaMapConfig:
        """Canonical demo profile: profiling enabled, safe bounds, auto device."""
        return cls(
            runtime=RuntimeConfig(
                device="auto",
                grid_engine="numpy",
                features_engine="numpy",
                enable_profiling=True,
            ),
            preprocess=PreprocessConfig(enabled=True),
        )

    @classmethod
    def ros2(cls) -> FoveaMapConfig:
        """ROS 2 deployment profile: standardized automotive sensor bounds and real-time settings."""
        return cls(
            sensor=SensorConfig(
                n_rows=64,
                min_range_m=0.5,
                max_range_m=100.0,
            ),
            runtime=RuntimeConfig(
                device="auto",
                grid_engine="numpy",
                features_engine="numpy",
                enable_profiling=False,
            ),
            preprocess=PreprocessConfig(enabled=True, remove_invalid=True, remove_self_hits=True),
        )

    @classmethod
    def from_env(cls) -> FoveaMapConfig:
        """Construct FoveaMapConfig resolved deterministically from environment variables:

        FOVEAMAP_PROFILE: cpu | gpu | demo | bench | ros2 (default: cpu)
        FOVEAMAP_DEVICE: auto | cpu | cuda | cuda:0
        FOVEAMAP_GRID_ENGINE: numpy | torch
        FOVEAMAP_FEATURES_ENGINE: numpy | torch
        FOVEAMAP_CHECKPOINT: path to model weights
        FOVEAMAP_PROFILING: 1 | 0 | true | false
        """
        profile = os.environ.get("FOVEAMAP_PROFILE", "cpu").strip().lower()
        if profile in ("gpu", "gpu_dev", "cuda"):
            ckpt = os.environ.get("FOVEAMAP_CHECKPOINT")
            base = cls.gpu_dev(checkpoint_path=ckpt)
        elif profile in ("bench", "benchmark"):
            base = cls.benchmark()
        elif profile in ("demo", "canonical_demo"):
            base = cls.demo()
        elif profile in ("ros", "ros2"):
            base = cls.ros2()
        else:
            base = cls.cpu_dev()

        # Apply granular environment variable overrides
        device = os.environ.get("FOVEAMAP_DEVICE")
        grid_engine = os.environ.get("FOVEAMAP_GRID_ENGINE")
        features_engine = os.environ.get("FOVEAMAP_FEATURES_ENGINE")
        profiling_str = os.environ.get("FOVEAMAP_PROFILING")

        runtime_kwargs = {
            "device": device if device is not None else base.runtime.device,
            "grid_engine": grid_engine if grid_engine is not None else base.runtime.grid_engine,
            "features_engine": features_engine if features_engine is not None else base.runtime.features_engine,
            "enable_profiling": (profiling_str.strip().lower() in ("1", "true", "yes"))
            if profiling_str is not None else base.runtime.enable_profiling,
        }

        ckpt_env = os.environ.get("FOVEAMAP_CHECKPOINT")
        perception_kwargs = {}
        if ckpt_env:
            perception_kwargs["checkpoint_path"] = ckpt_env

        runtime = RuntimeConfig(**runtime_kwargs)
        perception = (
            PerceptionConfig(**{**base.perception.__dict__, **perception_kwargs})
            if perception_kwargs else base.perception
        )

        return cls(
            sensor=base.sensor,
            grid=base.grid,
            perception=perception,
            terrain=base.terrain,
            runtime=runtime,
            preprocess=base.preprocess,
            dynamic=base.dynamic,
        )

