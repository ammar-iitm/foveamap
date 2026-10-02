"""Base abstractions and protocols for LiDAR data ingestion.

All dataset readers, sensor drivers, simulator wrappers, and file readers
subclass `LiDARSource` to emit canonical `LiDARFrame` contracts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Any

from ..core.contracts import LiDARFrame
from ..core.config import SensorConfig


class LiDARSource(ABC):
    """Abstract base class for all deterministic LiDAR data sources.

    Subclasses wrap concrete datasets, live sensors, simulators, or recorded files
    and emit canonical `LiDARFrame` instances.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[LiDARFrame]:
        """Iterate sequentially over canonical LiDAR frames."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Total number of available frames in this source, if known."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset iteration state to the beginning for deterministic replay."""
        ...

    def __getitem__(self, index: int) -> LiDARFrame:
        """Random access to a specific frame by index (if supported)."""
        raise NotImplementedError(
            f"Source '{self.source_id}' ({self.__class__.__name__}) does not support indexed random access."
        )

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique identifier or name describing this data source."""
        ...

    @property
    def sensor_config(self) -> SensorConfig | None:
        """Associated sensor configuration, if known or inferable."""
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Dataset or source-level provenance and calibration metadata."""
        return {}

    def close(self) -> None:
        """Release underlying system or file resources."""
        pass

    def __enter__(self) -> LiDARSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
