"""Phase 11 Productization and Real-World Integration Tests.

Validates:
- Standard deployment profiles & environment configuration (Parts 6 & 16)
- Product-level runtime lifecycle contract and state transitions (Part 5)
- Observability and authoritative runtime metrics (Part 8)
- Real input data contract provenance and annotations (Part 7)
- Baseline comparison engine vs. uniform 5 cm grid (Part 10)
- Unified CLI commands (Parts 11, 15, 20)
- HTTP boundary enhancements and security path traversal prevention (Parts 9 & 14)
- Failure recovery & empty frame handling (Part 13)
"""
from __future__ import annotations

import os
import urllib.request
import urllib.error
import json
import pytest
import numpy as np
import torch

from foveamap.core.config import FoveaMapConfig, GridConfig
from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import ContractError
from foveamap.runtime.runtime import FoveaMapRuntime
from foveamap.benchmarks.baseline import compare_uniform_baseline, compute_grid_geometry
from foveamap.cli import main
from foveamap.sdk.client import FoveaMap
from foveamap.sdk.http import FoveaMapHttpServer


# -----------------------------------------------------------------------------
# 1. Deployment Profiles & Environment Configuration
# -----------------------------------------------------------------------------
def test_deployment_profiles_defaults():
    cpu_cfg = FoveaMapConfig.cpu_dev()
    assert cpu_cfg.runtime.device == "cpu"
    assert cpu_cfg.runtime.grid_engine == "numpy"
    assert cpu_cfg.runtime.features_engine == "numpy"

    gpu_cfg = FoveaMapConfig.gpu_dev()
    assert gpu_cfg.runtime.grid_engine == "torch"
    assert gpu_cfg.runtime.features_engine == "torch"
    assert gpu_cfg.perception.backend_type == "range_unet"
    assert gpu_cfg.perception.fp16 is True

    bench_cfg = FoveaMapConfig.benchmark()
    assert bench_cfg.runtime.enable_profiling is True

    demo_cfg = FoveaMapConfig.demo()
    assert demo_cfg.runtime.enable_profiling is True

    ros_cfg = FoveaMapConfig.ros2()
    assert ros_cfg.sensor.n_rows == 64
    assert ros_cfg.preprocess.remove_invalid is True
    assert ros_cfg.preprocess.remove_self_hits is True


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("FOVEAMAP_PROFILE", "gpu")
    monkeypatch.setenv("FOVEAMAP_DEVICE", "cuda:0")
    monkeypatch.setenv("FOVEAMAP_PROFILING", "1")

    cfg = FoveaMapConfig.from_env()
    assert cfg.runtime.device == "cuda:0"
    assert cfg.runtime.grid_engine == "torch"
    assert cfg.runtime.enable_profiling is True

    monkeypatch.setenv("FOVEAMAP_PROFILE", "cpu")
    monkeypatch.setenv("FOVEAMAP_DEVICE", "cpu")
    monkeypatch.setenv("FOVEAMAP_GRID_ENGINE", "numpy")
    monkeypatch.setenv("FOVEAMAP_PROFILING", "0")

    cfg2 = FoveaMapConfig.from_env()
    assert cfg2.runtime.device == "cpu"
    assert cfg2.runtime.grid_engine == "numpy"
    assert cfg2.runtime.enable_profiling is False


# -----------------------------------------------------------------------------
# 2. Runtime Contract & Lifecycle
# -----------------------------------------------------------------------------
def test_runtime_lifecycle_transitions():
    runtime = FoveaMapRuntime(config=FoveaMapConfig.cpu_dev())
    assert runtime.lifecycle == "INITIALIZED"

    state = runtime.configure()
    assert state == "CONFIGURED"
    assert runtime.lifecycle == "CONFIGURED"

    state = runtime.start()
    assert state == "ACTIVE"
    assert runtime.lifecycle == "ACTIVE"

    state = runtime.stop()
    assert state == "STOPPED"
    assert runtime.lifecycle == "STOPPED"

    # Processing in STOPPED state must be rejected explicitly (Part 13)
    frame = LiDARFrame(
        pts=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        intensity=np.array([0.5], dtype=np.float32),
        ring=np.array([0], dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="f0",
    )
    with pytest.raises(ContractError, match="STOPPED state"):
        runtime.process(frame)

    # Re-starting or resetting restores active status
    runtime.start()
    assert runtime.lifecycle == "ACTIVE"
    snap = runtime.process(frame)
    assert snap.frame_id == "f0"

    runtime.reset()
    assert runtime.lifecycle == "CONFIGURED"
    assert runtime.frame_count == 0


def test_runtime_empty_frame_handling():
    runtime = FoveaMapRuntime(config=FoveaMapConfig.cpu_dev())
    runtime.start()

    # Empty frame (0 points)
    empty_frame = LiDARFrame(
        pts=np.zeros((0, 3), dtype=np.float32),
        intensity=np.zeros(0, dtype=np.float32),
        ring=np.zeros(0, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="empty_sweep",
    )

    # Must produce valid MapSnapshot without raising index error
    snap = runtime.process(empty_frame)
    assert snap.frame_id == "empty_sweep"
    assert snap.metadata.get("empty_frame") is True
    assert snap.metadata.get("points") == 0

    metrics = runtime.get_metrics()
    assert metrics["frames_dropped"] == 1
    assert metrics["frames_processed"] == 1


# -----------------------------------------------------------------------------
# 3. Observability & Runtime Metrics
# -----------------------------------------------------------------------------
def test_runtime_get_metrics_authoritative():
    runtime = FoveaMapRuntime(config=FoveaMapConfig.cpu_dev())
    runtime.start()

    pts = np.array([[5.0, 0.0, 0.0], [6.0, 1.0, 0.0]], dtype=np.float32)
    frame = LiDARFrame(
        pts=pts,
        intensity=np.array([0.5, 0.8], dtype=np.float32),
        ring=np.array([0, 1], dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="f1",
    )
    runtime.process(frame)

    metrics = runtime.get_metrics()
    assert metrics["lifecycle"] == "ACTIVE"
    assert metrics["frames_processed"] == 1
    assert metrics["frames_dropped"] == 0
    assert metrics["backend"] is not None
    assert metrics["device"] == "cpu"
    assert metrics["grid_engine"] == "numpy"
    assert "memory" in metrics
    assert metrics["memory"]["allocated_bytes"] > 0
    assert metrics["memory"]["reduction_ratio"] >= 30.0
    assert metrics["memory"]["under_8mb_target"] is True


# -----------------------------------------------------------------------------
# 4. Data Contract Provenance
# -----------------------------------------------------------------------------
def test_lidar_frame_provenance():
    # Real dataset frame
    kitti_frame = LiDARFrame(
        pts=np.zeros((10, 3), dtype=np.float32),
        intensity=np.zeros(10, dtype=np.float32),
        ring=np.zeros(10, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        source_id="semantickitti/00",
        label=np.zeros(10, dtype=np.int8),
    )
    assert kitti_frame.has_semantics is True
    assert kitti_frame.has_motion is False
    assert kitti_frame.data_origin_category == "REAL"

    # Synthetic simulation frame
    sim_frame = LiDARFrame(
        pts=np.zeros((10, 3), dtype=np.float32),
        intensity=np.zeros(10, dtype=np.float32),
        ring=np.zeros(10, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        source_id="sim",
        moving=np.zeros(10, dtype=bool),
    )
    assert sim_frame.has_semantics is False
    assert sim_frame.has_motion is True
    assert sim_frame.data_origin_category == "SYNTHETIC"

    # Test frame
    test_frame = LiDARFrame(
        pts=np.zeros((10, 3), dtype=np.float32),
        intensity=np.zeros(10, dtype=np.float32),
        ring=np.zeros(10, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="golden_sample",
    )
    assert test_frame.data_origin_category == "TEST"


# -----------------------------------------------------------------------------
# 5. Baseline Comparison Engine
# -----------------------------------------------------------------------------
def test_baseline_comparison_engine():
    geom_fovea = compute_grid_geometry("spec")
    assert geom_fovea["total_cells"] == 320000
    assert geom_fovea["total_bytes"] == 5120000

    geom_uniform = compute_grid_geometry("uniform5")
    assert geom_uniform["total_cells"] == 16000000
    assert geom_uniform["total_bytes"] == 256000000

    rep = compare_uniform_baseline(n_points=500, profile="spec", run_empirical=True, seed=1)
    assert rep["status"] == "VERIFIED"
    assert rep["foveamap_cells"] == 320000
    assert rep["uniform_cells"] == 16000000
    assert rep["memory_saving_ratio"] == 50.0
    assert rep["cell_reduction_ratio"] == 50.0
    assert rep["under_8mb_target"] is True
    assert rep["exceeds_30x_target"] is True
    assert "empirical" in rep
    assert rep["empirical"]["foveamap_update_ms"] > 0
    assert rep["empirical"]["uniform_update_ms"] > 0


# -----------------------------------------------------------------------------
# 6. CLI Commands
# -----------------------------------------------------------------------------
def test_cli_info():
    code = main(["info"])
    assert code == 0


def test_cli_compare():
    code = main(["compare", "--no-empirical"])
    assert code == 0


def test_cli_demo_short():
    code = main(["demo", "--frames", "2", "--seed", "100"])
    assert code == 0


# -----------------------------------------------------------------------------
# 7. HTTP Server & Security
# -----------------------------------------------------------------------------
def test_http_server_endpoints_and_security(tmp_path):
    # Create mock dashboard directory
    dash_dir = tmp_path / "dashboard"
    dash_dir.mkdir()
    (dash_dir / "index.html").write_text("<h1>FoveaMap Dashboard</h1>", encoding="utf-8")
    (dash_dir / "data.json").write_text('{"status": "ok"}', encoding="utf-8")

    sdk = FoveaMap()
    sdk.configure()
    sdk.start()

    server = FoveaMapHttpServer(sdk, host="127.0.0.1", port=0, dashboard_dir=dash_dir)
    url = server.start_background()

    try:
        # GET /health
        with urllib.request.urlopen(f"{url}/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] in ("healthy", "degraded")

        # GET /status
        with urllib.request.urlopen(f"{url}/status") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["lifecycle"] == "ACTIVE"

        # GET /metrics
        with urllib.request.urlopen(f"{url}/metrics") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert "frames_received" in data

        # GET / (serves index.html)
        with urllib.request.urlopen(f"{url}/") as resp:
            assert resp.status == 200
            assert "FoveaMap Dashboard" in resp.read().decode()

        # GET /data.json
        with urllib.request.urlopen(f"{url}/data.json") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode())
            assert data["status"] == "ok"

        # Security test: Path traversal prevention (Part 14)
        # Attempt to escape dashboard directory with ../
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{url}/../outside.txt")
        assert exc_info.value.code in (403, 404)

    finally:
        server.stop()
        sdk.close()
