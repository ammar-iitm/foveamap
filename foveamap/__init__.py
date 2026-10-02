"""FoveaMap: Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception."""
from __future__ import annotations

__version__ = "0.1.0"

from .core.contracts import LiDARFrame, PerceptionResult, MapSnapshot
from .core.config import FoveaMapConfig, GridConfig, TierConfig, PreprocessConfig
from .core.ontology import (
    SemanticOntology,
    CANONICAL_CLASSES,
    NUM_CLASSES,
    DEFAULT_ONTOLOGY,
)
from .core.exceptions import (
    FoveaMapError,
    ConfigurationError,
    ContractError,
    DataAdapterError,
    PerceptionError,
    MappingError,
    NumericalConsistencyError,
)

from .runtime import (
    FoveaMapRuntime,
    resolve_device,
    PerceptionBackend,
    RangeUNetBackend,
    ClassicalFallbackBackend,
    create_perception_backend,
)
from .data import LiDARSource, LiDARPreprocessor, create_source

__all__ = [
    "__version__",
    "LiDARFrame",
    "PerceptionResult",
    "MapSnapshot",
    "FoveaMapConfig",
    "GridConfig",
    "TierConfig",
    "PreprocessConfig",
    "LiDARSource",
    "LiDARPreprocessor",
    "create_source",
    "FoveaMapRuntime",
    "resolve_device",
    "PerceptionBackend",
    "RangeUNetBackend",
    "ClassicalFallbackBackend",
    "create_perception_backend",
    "FoveaMapError",
    "ConfigurationError",
    "ContractError",
    "DataAdapterError",
    "PerceptionError",
    "MappingError",
    "NumericalConsistencyError",
    "SemanticOntology",
    "CANONICAL_CLASSES",
    "NUM_CLASSES",
    "DEFAULT_ONTOLOGY",
]

