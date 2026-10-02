"""Data ingestion, source abstractions, and preprocessing layer for FoveaMap."""
from __future__ import annotations

from .source import LiDARSource
from .preprocess import LiDARPreprocessor
from .sim import SimulatorSource
from .kitti import SemanticKITTISource
from .nuscenes import NuScenesSource
from .file import (
    FileLiDARSource,
    BinFileSource,
    PcdFileSource,
    NpyFileSource,
    FileSequenceSource,
    parse_pcd,
    parse_bin,
    parse_npy,
)
from .factory import create_source, register_source, list_sources

__all__ = [
    # Base abstractions
    "LiDARSource",
    "LiDARPreprocessor",
    # Adapters
    "SimulatorSource",
    "SemanticKITTISource",
    "NuScenesSource",
    # File sources
    "FileLiDARSource",
    "BinFileSource",
    "PcdFileSource",
    "NpyFileSource",
    "FileSequenceSource",
    "parse_pcd",
    "parse_bin",
    "parse_npy",
    # Factory
    "create_source",
    "register_source",
    "list_sources",
]
