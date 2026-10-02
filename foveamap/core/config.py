"""Structured, validated configuration dataclasses for FoveaMap.

Extracts all hardcoded parameters and magic numbers into explicit,
composable, and type-checked configuration objects.

NOTE ON RUNTIME WIRING:
    At this stage (Phase 1), these configuration dataclasses establish the
    formal schemas and invariant validation rules for the production system.
    Existing legacy runtime modules (pipeline.py, grid.py, model.py, etc.)
    continue to operate using their existing internal defaults.
    Active migration of the runtime execution loop to consume these configuration
    objects directly is scheduled for Phase 2.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class PerceptionConfig:
    """Perception model and inference settings."""
    backend_type: str = "range_unet"
    checkpoint_path: str | None = None
    num_classes: int = 9
    fp16: bool = True
    confidence_threshold: float = 0.5
    active_classes: tuple[bool, ...] = field(default_factory=lambda: (True,) * 9)

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ConfigurationError(f"num_classes must be positive, got {self.num_classes}")
        if len(self.active_classes) != self.num_classes:
            raise ConfigurationError(f"active_classes length ({len(self.active_classes)}) != num_classes ({self.num_classes})")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(f"confidence_threshold must be in [0, 1], got {self.confidence_threshold}")


@dataclass(frozen=True)
class TerrainConfig:
    """Geometric and traversability analysis thresholds."""
    vehicle_clearance_m: float = 2.5
    step_threshold_m: float = 0.08
    depression_threshold_m: float = 0.05
    depression_window_m: float = 2.5
    roughness_threshold_m: float = 0.04
    stale_age_threshold: int = 20
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
        if self.stale_age_threshold < 0:
            raise ConfigurationError(f"stale_age_threshold must be non-negative, got {self.stale_age_threshold}")
        if len(self.cost_priors) != 256:
            raise ConfigurationError("cost_priors must have length 256")


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution backend and runtime parameters."""
    device: str = "auto"
    grid_engine: str = "numpy"
    features_engine: str = "numpy"
    enable_profiling: bool = True

    def __post_init__(self) -> None:
        if self.grid_engine not in ("numpy", "torch"):
            raise ConfigurationError(f"grid_engine must be 'numpy' or 'torch', got {self.grid_engine!r}")
        if self.features_engine not in ("numpy", "torch"):
            raise ConfigurationError(f"features_engine must be 'numpy' or 'torch', got {self.features_engine!r}")


@dataclass(frozen=True)
class FoveaMapConfig:
    """Top-level unified system configuration."""
    sensor: SensorConfig = field(default_factory=SensorConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
