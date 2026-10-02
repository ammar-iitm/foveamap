"""Domain exception hierarchy for FoveaMap."""
from __future__ import annotations


class FoveaMapError(Exception):
    """Base exception for all FoveaMap domain, mapping, and pipeline errors."""


class ConfigurationError(FoveaMapError):
    """Raised when configuration values are invalid, inconsistent, or missing."""


class ContractError(FoveaMapError):
    """Raised when data violates a required contract, invariant, or dimension."""


class DataAdapterError(FoveaMapError):
    """Raised when sensor, frame, or dataset input data is missing, corrupted, or unsupported."""


class PerceptionError(FoveaMapError):
    """Raised when perception model execution or prediction fails."""


class MappingError(FoveaMapError):
    """Raised when spatial indexing, grid updating, or temporal fusion encounters an invalid state."""


class NumericalConsistencyError(FoveaMapError):
    """Raised when numerical values (e.g., coordinates, bounds, probabilities) contain NaN or Inf."""
