"""Tests for data source factory and runtime stream integration."""
from __future__ import annotations

from typing import Iterator
import numpy as np
import pytest

from foveamap.core.config import FoveaMapConfig, RuntimeConfig
from foveamap.core.contracts import LiDARFrame, MapSnapshot
from foveamap.core.exceptions import ConfigurationError
from foveamap.data.source import LiDARSource
from foveamap.data.factory import create_source, register_source, list_sources
from foveamap.runtime import FoveaMapRuntime
from foveamap.pipeline import FoveaMapPipeline


class DummyCustomSource(LiDARSource):
    def __init__(self, count: int = 3) -> None:
        self.count = count
        self._idx = 0

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[LiDARFrame]:
        for i in range(self.count):
            yield LiDARFrame(
                pts=np.ones((10, 3), dtype=np.float32) * float(i + 1),
                intensity=np.ones(10, dtype=np.float32),
                ring=np.zeros(10, dtype=np.int16),
                pose=np.eye(4, dtype=np.float64),
                frame_id=f"custom_{i}",
                source_id=self.source_id,
            )

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        return "custom_dummy"


def test_factory_create_simulator_source():
    source = create_source("simulator", n_steps=3, seed=99)
    assert isinstance(source, LiDARSource)
    assert len(source) == 3
    assert source.source_id == "simulator"


def test_factory_register_and_create_custom_source():
    register_source("dummy_test", DummyCustomSource)
    assert "dummy_test" in list_sources()

    source = create_source("dummy_test", count=4)
    assert isinstance(source, DummyCustomSource)
    assert len(source) == 4
    frames = list(source)
    assert len(frames) == 4
    assert frames[0].frame_id == "custom_0"


def test_factory_rejects_unknown_source():
    with pytest.raises(ConfigurationError):
        create_source("unknown_nonexistent_source")


def test_runtime_process_source_stream():
    source = create_source("simulator", n_steps=3, seed=42)
    cfg = FoveaMapConfig(runtime=RuntimeConfig(device="cpu", grid_engine="numpy"))
    runtime = FoveaMapRuntime(cfg)

    snapshots = list(runtime.process_source(source, max_frames=2))
    assert len(snapshots) == 2
    for s in snapshots:
        assert isinstance(s, MapSnapshot)
        assert s.num_tiers == 2


def test_legacy_pipeline_step_source_stream():
    source = create_source("simulator", n_steps=3, seed=42)
    pipeline = FoveaMapPipeline(grid="numpy", device="cpu")

    outputs = list(pipeline.step_source(source, max_frames=2))
    assert len(outputs) == 2
    for out in outputs:
        assert "stats" in out
        assert "dyn" in out
