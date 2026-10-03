"""Public SDK for FoveaMap (Phase 9).

Stable, typed, versioned boundary over the authoritative core:

    consumer -> foveamap.sdk -> FoveaMapRuntime -> core

The SDK never reimplements mapping, traversability, or lifecycle policy; it
delegates to :class:`FoveaMapRuntime` and :class:`MapSnapshot` and only
translates results into frozen public types. Importing this package never
requires ROS 2 or CUDA.
"""

API_VERSION = "1"

from .errors import SDKError, SDKLifecycleError, SDKQueryError
from .types import (
    HealthReport,
    QueryResult,
    RuntimeMetrics,
    RuntimeStatus,
    SnapshotView,
)
from .client import FoveaMap

__all__ = [
    "API_VERSION",
    "FoveaMap",
    "HealthReport",
    "QueryResult",
    "RuntimeMetrics",
    "RuntimeStatus",
    "SDKError",
    "SDKLifecycleError",
    "SDKQueryError",
    "SnapshotView",
]
