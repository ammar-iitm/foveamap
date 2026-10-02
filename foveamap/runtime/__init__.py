"""FoveaMap runtime package.

Exposes the unified core runtime orchestrator, hardware device resolution,
and perception backend abstractions.
"""
from __future__ import annotations

from .device import DeviceContext, resolve_device, sync_device
from .perception import (
    PerceptionBackend,
    RangeUNetBackend,
    ClassicalFallbackBackend,
    DevicePerceptionResult,
    create_perception_backend,
    register_perception_backend,
)
from .runtime import FoveaMapRuntime

__all__ = [
    "DeviceContext",
    "resolve_device",
    "sync_device",
    "PerceptionBackend",
    "RangeUNetBackend",
    "ClassicalFallbackBackend",
    "DevicePerceptionResult",
    "create_perception_backend",
    "register_perception_backend",
    "FoveaMapRuntime",
]
