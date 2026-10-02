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
    PreprocessConfig,
    FoveaMapConfig,
)
from .ontology import (
    CANONICAL_CLASSES,
    NUM_CLASSES,
    SemanticOntology,
    DEFAULT_ONTOLOGY,
    ROAD,
    SIDEWALK,
    PARKING,
    TERRAIN,
    VEGETATION,
    BUILDING,
    POLE,
    VEHICLE,
    PERSON,
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
    "PreprocessConfig",
    "FoveaMapConfig",
    # Exceptions
    "FoveaMapError",
    "ConfigurationError",
    "ContractError",
    "DataAdapterError",
    "PerceptionError",
    "MappingError",
    "NumericalConsistencyError",
    # Ontology
    "CANONICAL_CLASSES",
    "NUM_CLASSES",
    "SemanticOntology",
    "DEFAULT_ONTOLOGY",
]
