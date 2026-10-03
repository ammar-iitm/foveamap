"""Public SDK exception hierarchy (Phase 9).

Reuses the core hierarchy wherever the core already has the right meaning;
SDK-specific errors exist only for SDK-owned concerns (lifecycle, query
translation, client misuse). The original exception is always chained.
"""
from __future__ import annotations

from foveamap.core.exceptions import ConfigurationError, FoveaMapError


class SDKError(FoveaMapError):
    """Base class for SDK-owned failures."""


class SDKLifecycleError(SDKError):
    """Illegal lifecycle transition or use outside the ACTIVE state."""


class SDKQueryError(SDKError):
    """A query could not be answered (no snapshot, bad coordinates)."""


class SDKConfigError(SDKError, ConfigurationError):
    """SDK-level configuration misuse (unknown keys, wrong shapes).

    Subclasses both hierarchies so SDK callers catch ``SDKConfigError``
    while core-compat code still sees a ``ConfigurationError``.
    """
