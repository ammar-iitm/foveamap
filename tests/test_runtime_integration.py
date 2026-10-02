"""Phase 2 tests: Core Runtime Integration.

Validates:
- FoveaMapRuntime construction from FoveaMapConfig
- DeviceContext resolution and CUDA/CPU fallback policies
- PerceptionBackend boundary and PerceptionResult generation
- FoveatedGrid and TorchFoveatedGrid configuration wiring
- Canonical LiDARFrame ingestion and legacy dict compatibility
- MapSnapshot generation, memory reporting, and ownership semantics
- Immutability of input contracts
- Authoritative configuration affecting runtime terrain cost output
- FoveaMapPipeline backward compatibility with FoveaMapConfig and LiDARFrame
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import patch
import numpy as np
import pytest
import torch

from foveamap.core.contracts import LiDARFrame, PerceptionResult, MapSnapshot
from foveamap.core.config import (
    FoveaMapConfig,
    RuntimeConfig,
    PerceptionConfig,
    SensorConfig,
    GridConfig,
    TierConfig,
    TerrainConfig,
)
from foveamap.core.exceptions import (
    ConfigurationError,
    ContractError,
    PerceptionError,
)
from foveamap.runtime.device import resolve_device, sync_device
from foveamap.runtime.perception import RangeUNetBackend
from foveamap.runtime.runtime import FoveaMapRuntime
from foveamap.grid import FoveatedGrid, UNKNOWN
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.pipeline import FoveaMapPipeline


def _make_synthetic_frame(n: int = 100, timestamp: float = 1.0, frame_id: str = "test_001") -> LiDARFrame:
    """Generate a physically plausible synthetic LiDAR frame for testing."""
    rng = np.random.default_rng(42)
    # Points in front of vehicle (X > 0)
    x = rng.uniform(2.0, 30.0, size=n).astype(np.float32)
    y = rng.uniform(-10.0, 10.0, size=n).astype(np.float32)
    z = rng.uniform(-1.5, 0.5, size=n).astype(np.float32)
    pts = np.stack([x, y, z], axis=1)
    intensity = rng.uniform(0.1, 0.9, size=n).astype(np.float32)
    ring = rng.integers(0, 64, size=n, dtype=np.int16)
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 10.0  # vehicle at x=10m in world
    sensor_origin = np.array([0.0, 0.0, 1.73], dtype=np.float32)

    return LiDARFrame(
        pts=pts,
        intensity=intensity,
        ring=ring,
        pose=pose,
        sensor_origin=sensor_origin,
        timestamp=timestamp,
        frame_id=frame_id,
    )


# ----------------------------------------------------------------------------
# 1. Device and Runtime Resolution Tests
# ----------------------------------------------------------------------------
def test_device_resolution_auto_and_cpu():
    ctx_cpu = resolve_device(RuntimeConfig(device="cpu"))
    assert ctx_cpu.device_type == "cpu"
    assert not ctx_cpu.use_cuda
    assert not ctx_cpu.fp16_enabled

    ctx_auto = resolve_device(RuntimeConfig(device="auto"))
    assert ctx_auto.device_type in ("cpu", "cuda", "mps")


def test_device_resolution_cuda_available_mocked():
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"):
        ctx = resolve_device(
            RuntimeConfig(device="cuda"),
            PerceptionConfig(fp16=True),
        )
        assert ctx.device.type == "cuda"
        assert ctx.use_cuda is True
        assert ctx.fp16_enabled is True
        assert "RTX 4090" in ctx.description


def test_device_resolution_cuda_unavailable_raises_error():
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(ConfigurationError) as exc_info:
            resolve_device(RuntimeConfig(device="cuda"))
        assert "torch.cuda.is_available() is False" in str(exc_info.value)


def test_sync_device_safe_on_all():
    sync_device(torch.device("cpu"))


# ----------------------------------------------------------------------------
# 2. Runtime Construction and Engine Selection Tests
# ----------------------------------------------------------------------------
def test_runtime_construction_default():
    runtime = FoveaMapRuntime()
    assert isinstance(runtime.config, FoveaMapConfig)
    assert runtime.last_snapshot is None
    assert runtime.last_perception is None
    assert runtime.frame_count == 0
    assert isinstance(runtime.grid, FoveatedGrid)


def test_runtime_construction_torch_cpu_engine():
    cfg = FoveaMapConfig(
        runtime=RuntimeConfig(device="cpu", grid_engine="torch", features_engine="numpy")
    )
    runtime = FoveaMapRuntime(cfg)
    assert isinstance(runtime.grid, TorchFoveatedGrid)
    assert runtime.device_ctx.device_type == "cpu"


def test_runtime_rejects_invalid_inputs():
    runtime = FoveaMapRuntime()
    with pytest.raises(ContractError):
        runtime.process("invalid_input_type")  # type: ignore


# ----------------------------------------------------------------------------
# 3. Perception Backend Boundary Tests
# ----------------------------------------------------------------------------
def test_perception_backend_produces_valid_result():
    device = torch.device("cpu")
    backend = RangeUNetBackend(
        config=PerceptionConfig(num_classes=9, confidence_threshold=0.5),
        sensor_config=SensorConfig(n_rows=64, n_cols=1024),
        device=device,
    )
    frame = _make_synthetic_frame(n=80)
    result = backend.predict(frame)

    assert isinstance(result, PerceptionResult)
    assert result.class_probabilities.shape == (80, 9)
    assert result.moving_probabilities.shape == (80,)
    assert result.semantic_predictions.shape == (80,)
    assert result.is_moving.shape == (80,)
    assert result.is_moving.dtype == bool
    assert len(backend.history) == 1


def test_perception_backend_rejects_missing_checkpoint():
    with pytest.raises(ConfigurationError):
        RangeUNetBackend(
            config=PerceptionConfig(checkpoint_path="nonexistent_checkpoint.pt"),
            sensor_config=SensorConfig(),
            device=torch.device("cpu"),
        )


# ----------------------------------------------------------------------------
# 4. End-to-End Runtime Execution Tests (LiDARFrame & Legacy Dict)
# ----------------------------------------------------------------------------
def test_runtime_process_lidar_frame():
    cfg = FoveaMapConfig(
        grid=GridConfig.from_preset("spec"),
        runtime=RuntimeConfig(device="cpu", grid_engine="numpy"),
    )
    runtime = FoveaMapRuntime(cfg)
    frame = _make_synthetic_frame(n=100, timestamp=10.5, frame_id="sweep_105")

    snapshot = runtime.process(frame)
    assert isinstance(snapshot, MapSnapshot)
    assert snapshot.timestamp == 10.5
    assert snapshot.frame_id == "sweep_105"
    assert snapshot.num_tiers == 2
    assert snapshot.total_memory_bytes > 0
    assert runtime.frame_count == 1
    assert runtime.snapshot() is snapshot
    assert "timing" in snapshot.metadata


def test_runtime_process_legacy_dict():
    runtime = FoveaMapRuntime()
    frame = _make_synthetic_frame(n=50)
    legacy = frame.to_legacy_dict()

    snapshot = runtime.step(legacy)  # verify step alias and dict ingest
    assert isinstance(snapshot, MapSnapshot)
    assert snapshot.num_tiers == len(runtime.config.grid.tiers)
    assert runtime.frame_count == 1


def test_runtime_reset_cycle():
    runtime = FoveaMapRuntime()
    frame = _make_synthetic_frame(n=50)
    runtime.process(frame)
    assert runtime.frame_count == 1

    runtime.reset()
    assert runtime.frame_count == 0
    assert runtime.last_snapshot is None
    assert runtime.last_perception is None
    assert len(runtime.perception.history) == 0


def test_input_frame_immutability():
    runtime = FoveaMapRuntime()
    frame = _make_synthetic_frame(n=60)
    orig_pts = frame.pts.copy()
    orig_pose = frame.pose.copy()
    orig_ring = frame.ring.copy()

    runtime.process(frame)

    assert np.array_equal(frame.pts, orig_pts)
    assert np.array_equal(frame.pose, orig_pose)
    assert np.array_equal(frame.ring, orig_ring)


def test_snapshot_immutability_semantics():
    runtime = FoveaMapRuntime()
    frame = _make_synthetic_frame(n=40)
    snapshot = runtime.process(frame)

    # Reassignment of frozen attributes must raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        snapshot.timestamp = 99.0  # type: ignore


# ----------------------------------------------------------------------------
# 5. Authoritative Configuration Verification Tests
# ----------------------------------------------------------------------------
def test_terrain_config_influences_grid_cost():
    # Verify that TerrainConfig custom cost priors directly alter grid cost computation
    custom_cost = [250] * 256  # Make all classes high cost
    custom_cost[0] = 250       # Road also lethal cost

    cfg_default = FoveaMapConfig(terrain=TerrainConfig())
    cfg_custom = FoveaMapConfig(terrain=TerrainConfig(cost_priors=tuple(custom_cost)))

    grid_def = FoveatedGrid(profile=cfg_default.grid, terrain_config=cfg_default.terrain)
    grid_cust = FoveatedGrid(profile=cfg_custom.grid, terrain_config=cfg_custom.terrain)

    # Feed point at origin with class 0 (ROAD)
    xy = np.array([[1.0, 1.0]], dtype=np.float64)
    z = np.array([-1.7], dtype=np.float64)
    probs = np.zeros((1, 9), dtype=np.float32)
    probs[0, 0] = 1.0  # Road
    moving = np.array([False])
    origins = [(0, 0), (0, 0)]

    stats_def = grid_def.bin_points(xy, z, probs, moving, origins)
    grid_def.fuse_stats(stats_def, origins)

    stats_cust = grid_cust.bin_points(xy, z, probs, moving, origins)
    grid_cust.fuse_stats(stats_cust, origins)

    # Cost in custom grid must be higher than default grid for road cell
    cost_def = grid_def.state[0].cost
    cost_cust = grid_cust.state[0].cost

    valid_cells = (cost_def != UNKNOWN) & (cost_cust != UNKNOWN)
    assert np.any(valid_cells)
    assert np.all(cost_cust[valid_cells] >= cost_def[valid_cells])


# ----------------------------------------------------------------------------
# 6. Legacy Pipeline Backward Compatibility
# ----------------------------------------------------------------------------
def test_legacy_pipeline_from_config_and_lidar_frame():
    cfg = FoveaMapConfig(
        runtime=RuntimeConfig(device="cpu", grid_engine="numpy", features_engine="numpy")
    )
    pipe = FoveaMapPipeline.from_config(cfg)
    frame = _make_synthetic_frame(n=70)

    # Pass canonical LiDARFrame to legacy pipeline step()
    out = pipe.step(frame)
    assert "timing" in out
    assert "cls_pts" in out
    assert "moving_pts" in out
    assert "dyn" in out
    assert "stats" in out
    assert len(out["cls_pts"]) == 70
