"""Comprehensive test suite for Phase 4 Perception Architecture and Model Integration."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from foveamap.core.contracts import LiDARFrame, PerceptionResult
from foveamap.core.config import PerceptionConfig, SensorConfig, FoveaMapConfig, RuntimeConfig
from foveamap.core.exceptions import ConfigurationError, ContractError, PerceptionError
from foveamap.runtime.perception import (
    PerceptionBackend,
    RangeUNetBackend,
    ClassicalFallbackBackend,
    DevicePerceptionResult,
    create_perception_backend,
    register_perception_backend,
    list_perception_backends,
)
from foveamap.runtime import FoveaMapRuntime
from foveamap.sim import simulate_sequence
from foveamap.frames import sim_frames
from foveamap.data.sim import SimulatorSource


CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "checkpoints", "range_unet.pt"
)


@pytest.fixture
def test_frame() -> LiDARFrame:
    """Deterministic simulated test sweep."""
    source = SimulatorSource(n_steps=1, seed=42)
    return source[0]


# ---------------------------------------------------------------------------
# 1. Backend Lifecycle & Contract Tests
# ---------------------------------------------------------------------------
def test_range_unet_backend_initialization():
    p_cfg = PerceptionConfig(
        checkpoint_path=CHECKPOINT_PATH if os.path.isfile(CHECKPOINT_PATH) else None,
        num_classes=9,
        fp16=True,
    )
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    dev = torch.device("cpu")

    backend = RangeUNetBackend(p_cfg, s_cfg, dev, features_engine="numpy")
    assert isinstance(backend, PerceptionBackend)
    assert backend.name == "range_unet"
    assert backend.model_name == "RangeUNet"
    assert backend.num_classes == 9
    assert backend.device == dev
    assert backend.is_ready is True
    assert backend.model.training is False  # Must be in eval mode


def test_range_unet_backend_missing_checkpoint():
    p_cfg = PerceptionConfig(
        checkpoint_path="nonexistent_fake_checkpoint_12345.pt",
        num_classes=9,
    )
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    with pytest.raises(PerceptionError, match="checkpoint file not found"):
        RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))


def test_classical_fallback_backend(test_frame):
    backend = ClassicalFallbackBackend(device=torch.device("cpu"))
    assert backend.name == "classical_fallback"
    assert backend.num_classes == 9

    res = backend.predict(test_frame)
    assert isinstance(res, PerceptionResult)
    assert res.num_points == test_frame.num_points
    assert res.class_probabilities.shape == (test_frame.num_points, 9)
    assert np.all(res.is_moving == False)
    assert res.point_confidence.max() <= 1.0


# ---------------------------------------------------------------------------
# 2. Input Frame Variations & Edge Cases
# ---------------------------------------------------------------------------
def test_perception_predict_normal_frame(test_frame):
    p_cfg = PerceptionConfig(
        checkpoint_path=CHECKPOINT_PATH if os.path.isfile(CHECKPOINT_PATH) else None,
        num_classes=9,
        fp16=False,
    )
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"), features_engine="numpy")

    res = backend.predict(test_frame)
    assert isinstance(res, PerceptionResult)
    assert res.num_points == test_frame.num_points

    # Output contract validation
    assert res.class_probabilities.shape == (test_frame.num_points, 9)
    assert res.moving_probabilities.shape == (test_frame.num_points,)
    assert res.semantic_predictions.shape == (test_frame.num_points,)
    assert res.is_moving.shape == (test_frame.num_points,)

    # Probability bounds
    assert np.all(res.class_probabilities >= 0.0)
    assert np.all(res.class_probabilities <= 1.0 + 1e-4)
    assert np.all(res.moving_probabilities >= 0.0)
    assert np.all(res.moving_probabilities <= 1.0 + 1e-4)

    # Class ID bounds
    assert np.all(res.semantic_predictions >= 0)
    assert np.all(res.semantic_predictions < 9)

    # Confidence and point indices
    assert res.point_confidence.shape == (test_frame.num_points,)
    assert res.point_indices is not None


def test_perception_predict_empty_frame():
    p_cfg = PerceptionConfig(num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))

    empty_frame = LiDARFrame(
        pts=np.empty((0, 3), dtype=np.float32),
        intensity=np.empty((0,), dtype=np.float32),
        ring=np.empty((0,), dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="empty",
    )

    res = backend.predict(empty_frame)
    assert isinstance(res, PerceptionResult)
    assert res.num_points == 0
    assert res.class_probabilities.shape == (0, 9)
    assert res.moving_probabilities.shape == (0,)
    assert res.semantic_predictions.shape == (0,)
    assert res.is_moving.shape == (0,)
    assert res.point_confidence.shape == (0,)


def test_perception_predict_single_point_frame():
    p_cfg = PerceptionConfig(num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))

    single_frame = LiDARFrame(
        pts=np.array([[12.0, 3.0, 0.5]], dtype=np.float32),
        intensity=np.array([0.6], dtype=np.float32),
        ring=np.array([32], dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="single",
    )

    res = backend.predict(single_frame)
    assert res.num_points == 1
    assert res.class_probabilities.shape == (1, 9)
    assert 0 <= res.semantic_predictions[0] < 9


def test_perception_rejects_invalid_frame():
    p_cfg = PerceptionConfig(num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))

    with pytest.raises(ContractError):
        backend.predict({"not_a": "lidar_frame"})  # type: ignore


# ---------------------------------------------------------------------------
# 3. Device Execution & Zero-Copy Flow
# ---------------------------------------------------------------------------
def test_device_perception_result_to_host(test_frame):
    p_cfg = PerceptionConfig(num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    dev = torch.device("cpu")
    backend = RangeUNetBackend(p_cfg, s_cfg, dev)

    dev_res = backend.predict_device(test_frame, dev_math=False)
    assert isinstance(dev_res, DevicePerceptionResult)
    assert torch.is_tensor(dev_res.class_probabilities)
    assert dev_res.class_probabilities.device == dev

    host_res = dev_res.to_host()
    assert isinstance(host_res, PerceptionResult)
    assert isinstance(host_res.class_probabilities, np.ndarray)
    np.testing.assert_allclose(
        host_res.class_probabilities, dev_res.class_probabilities.numpy()
    )


# ---------------------------------------------------------------------------
# 4. Latency Timing Instrumentation
# ---------------------------------------------------------------------------
def test_perception_timing_instrumentation(test_frame):
    p_cfg = PerceptionConfig(num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))

    res = backend.predict(test_frame)
    timing = res.metadata.get("timing")
    assert timing is not None
    assert "feature_ms" in timing
    assert "inference_ms" in timing
    assert "postprocess_ms" in timing
    assert "total_ms" in timing

    assert timing["total_ms"] > 0
    assert timing["feature_ms"] >= 0
    assert timing["inference_ms"] >= 0
    assert timing["postprocess_ms"] >= 0


# ---------------------------------------------------------------------------
# 5. Golden Frame Perception Test (Determinism)
# ---------------------------------------------------------------------------
def test_golden_frame_perception_determinism(test_frame):
    p_cfg = PerceptionConfig(
        checkpoint_path=CHECKPOINT_PATH if os.path.isfile(CHECKPOINT_PATH) else None,
        num_classes=9,
        fp16=False,
    )
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))

    # Run 1
    backend.reset()
    res1 = backend.predict(test_frame)

    # Run 2
    backend.reset()
    res2 = backend.predict(test_frame)

    # Verify identical output predictions
    np.testing.assert_array_equal(res1.semantic_predictions, res2.semantic_predictions)
    np.testing.assert_array_equal(res1.is_moving, res2.is_moving)
    np.testing.assert_allclose(res1.class_probabilities, res2.class_probabilities, atol=1e-5)
    np.testing.assert_allclose(res1.moving_probabilities, res2.moving_probabilities, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Perception Factory & Registry
# ---------------------------------------------------------------------------
def test_perception_factory_creation():
    p_cfg = PerceptionConfig(backend_type="range_unet", num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    dev = torch.device("cpu")

    backend = create_perception_backend(p_cfg, s_cfg, dev)
    assert isinstance(backend, RangeUNetBackend)

    p_cfg_classical = PerceptionConfig(backend_type="classical", num_classes=9)
    backend_class = create_perception_backend(p_cfg_classical, s_cfg, dev)
    assert isinstance(backend_class, ClassicalFallbackBackend)

    assert "range_unet" in list_perception_backends()
    assert "classical" in list_perception_backends()

    with pytest.raises(ConfigurationError):
        create_perception_backend(PerceptionConfig(backend_type="unknown_model"), s_cfg, dev)


# ---------------------------------------------------------------------------
# 7. Runtime End-to-End Ingestion -> Perception -> Mapping
# ---------------------------------------------------------------------------
def test_runtime_with_perception_backend(test_frame):
    cfg = FoveaMapConfig(
        runtime=RuntimeConfig(device="cpu", grid_engine="numpy"),
        perception=PerceptionConfig(num_classes=9, fp16=False),
    )
    runtime = FoveaMapRuntime(cfg)
    assert isinstance(runtime.perception, PerceptionBackend)

    snapshot = runtime.process(test_frame)
    assert snapshot is not None
    assert runtime.last_perception is not None
    assert runtime.last_perception.num_points == test_frame.num_points


# ---------------------------------------------------------------------------
# 8. Conditional CUDA Test (Physically Verified when Available)
# ---------------------------------------------------------------------------
def test_cuda_perception_execution(test_frame):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable on current host (CPU environment)")

    dev = torch.device("cuda:0")
    p_cfg = PerceptionConfig(
        checkpoint_path=CHECKPOINT_PATH if os.path.isfile(CHECKPOINT_PATH) else None,
        num_classes=9,
        fp16=True,
    )
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, dev, features_engine="torch")

    dev_res = backend.predict_device(test_frame, dev_math=True)
    assert dev_res.device.type == "cuda"
    assert dev_res.class_probabilities.is_cuda is True
    assert dev_res.is_moving.is_cuda is True

    host_res = dev_res.to_host()
    assert host_res.num_points == test_frame.num_points
    assert np.all(np.isfinite(host_res.class_probabilities))
