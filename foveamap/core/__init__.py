"""Core domain layer of FoveaMap: contracts, configuration, and exceptions."""
from __future__ import annotations

from .contracts import LiDARFrame, PerceptionResult, MapSnapshot
from .config import (
    TierConfig,
    GridConfig,
    SensorConfig,
    PerceptionConfig,
    TerrainConfig,
    RuntimeConfig,
    FoveaMapConfig,
)
from .exceptions import (
    FoveaMapError,
    ConfigurationError,
    ContractError,
    DataAdapterError,
    PerceptionError,
    MappingError,
    NumericalConsistencyError,
)

__all__ = [
    # Contracts
    "LiDARFrame",
    "PerceptionResult",
    "MapSnapshot",
    # Configuration
    "TierConfig",
    "GridConfig",
    "SensorConfig",
    "PerceptionConfig",
    "TerrainConfig",
    "RuntimeConfig",
    "FoveaMapConfig",
    # Exceptions
    "FoveaMapError",
    "ConfigurationError",
    "ContractError",
    "DataAdapterError",
    "PerceptionError",
    "MappingError",
    "NumericalConsistencyError",
]
