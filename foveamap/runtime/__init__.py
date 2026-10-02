"""FoveaMap runtime package.

Exposes the unified core runtime orchestrator, hardware device resolution,
and perception backend abstractions.
"""
from __future__ import annotations

from .device import DeviceContext, resolve_device, sync_device
from .perception import PerceptionBackend, RangeUNetBackend, DevicePerceptionResult
from .runtime import FoveaMapRuntime

__all__ = [
    "DeviceContext",
    "resolve_device",
    "sync_device",
    "PerceptionBackend",
    "RangeUNetBackend",
    "DevicePerceptionResult",
    "FoveaMapRuntime",
]
