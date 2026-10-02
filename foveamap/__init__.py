"""FoveaMap: Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception."""
from __future__ import annotations

__version__ = "0.1.0"

from .core.contracts import LiDARFrame, PerceptionResult, MapSnapshot
from .core.config import FoveaMapConfig, GridConfig, TierConfig
from .core.exceptions import (
    FoveaMapError,
    ConfigurationError,
    ContractError,
    DataAdapterError,
    PerceptionError,
    MappingError,
    NumericalConsistencyError,
)

__all__ = [
    "__version__",
    "LiDARFrame",
    "PerceptionResult",
    "MapSnapshot",
    "FoveaMapConfig",
    "GridConfig",
    "TierConfig",
    "FoveaMapError",
    "ConfigurationError",
    "ContractError",
    "DataAdapterError",
    "PerceptionError",
    "MappingError",
    "NumericalConsistencyError",
]
