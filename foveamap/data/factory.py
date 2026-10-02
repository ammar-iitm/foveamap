"""Extensible source factory and registry for LiDAR ingestion."""
from __future__ import annotations

from typing import Type
from .source import LiDARSource
from .sim import SimulatorSource
from .kitti import SemanticKITTISource
from .nuscenes import NuScenesSource
from .file import (
    FileLiDARSource,
    BinFileSource,
    PcdFileSource,
    NpyFileSource,
    FileSequenceSource,
)
from ..core.exceptions import ConfigurationError

_SOURCE_REGISTRY: dict[str, Type[LiDARSource]] = {
    "sim": SimulatorSource,
    "simulator": SimulatorSource,
    "kitti": SemanticKITTISource,
    "semantickitti": SemanticKITTISource,
    "nusc": NuScenesSource,
    "nuscenes": NuScenesSource,
    "file": FileLiDARSource,
    "bin": BinFileSource,
    "pcd": PcdFileSource,
    "npy": NpyFileSource,
    "sequence": FileSequenceSource,
    "dir": FileSequenceSource,
}


def register_source(source_type: str, source_cls: Type[LiDARSource]) -> None:
    """Register a custom LiDARSource class in the global factory."""
    if not issubclass(source_cls, LiDARSource):
        raise TypeError(f"Registered source class {source_cls} must subclass LiDARSource")
    key = source_type.strip().lower()
    _SOURCE_REGISTRY[key] = source_cls


def create_source(source_type: str, **kwargs) -> LiDARSource:
    """Instantiate a configured LiDARSource by name."""
    key = source_type.strip().lower()
    if key not in _SOURCE_REGISTRY:
        raise ConfigurationError(
            f"Unknown source type '{source_type}'. Available source types: {sorted(_SOURCE_REGISTRY.keys())}"
        )
    return _SOURCE_REGISTRY[key](**kwargs)


def list_sources() -> list[str]:
    """Return list of all registered source types."""
    return sorted(list(_SOURCE_REGISTRY.keys()))


# Explicit aliases for consistency with create_perception_backend
create_lidar_source = create_source
register_lidar_source = register_source
list_lidar_sources = list_sources
