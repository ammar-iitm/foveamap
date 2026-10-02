"""Comprehensive test suite for Phase 4 Perception Architecture and Model Integration."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest
import torch

from foveamap.core.contracts import LiDARFrame, PerceptionResult
from foveamap.core.config import PerceptionConfig, SensorConfig, FoveaMapConfig, RuntimeConfig
from foveamap.core.exceptions import ConfigurationError, ContractError, PerceptionError, NumericalConsistencyError
from foveamap.core.ontology import CANONICAL_CLASSES, NUM_CLASSES, ROAD, SIDEWALK, BUILDING, VEHICLE, PERSON
from foveamap.runtime.perception import (
    CheckpointNotFoundError,
    PerceptionBackend,
    RangeUNetBackend,
    ClassicalFallbackBackend,
    DevicePerceptionResult,
    DeviceTemporalState,
    DeviceTemporalSweep,
    create_perception_backend,
    register_perception_backend,
    list_perception_backends,
)
from foveamap.runtime import FoveaMapRuntime
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
# 1. Backend Lifecycle & Checkpoint Safety Tests
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


def test_range_unet_backend_checkpoint_none_rejected():
    """Production configuration must reject checkpoint_path=None to avoid random weights."""
    p_cfg = PerceptionConfig(checkpoint_path=None, num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    dev = torch.device("cpu")

    with pytest.raises(ConfigurationError, match="requires a valid checkpoint_path in production"):
        RangeUNetBackend(p_cfg, s_cfg, dev, allow_untrained=False)


def test_range_unet_backend_missing_checkpoint():
    """Non-existent checkpoint file must raise CheckpointNotFoundError (both PerceptionError and ConfigurationError)."""
    p_cfg = PerceptionConfig(
        checkpoint_path="nonexistent_fake_checkpoint_12345.pt",
        num_classes=9,
    )
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    with pytest.raises(CheckpointNotFoundError, match="checkpoint file not found"):
        RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))


def test_range_unet_backend_num_classes_mismatch():
    """Model architecture and canonical ontology require num_classes=9; reject incompatible configs."""
    p_cfg = PerceptionConfig(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=5,
        active_classes=(True,) * 5,
    )
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    with pytest.raises(ConfigurationError, match="require num_classes=9"):
        RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"))


def test_range_unet_backend_untrained_opt_in():
    """Explicit allow_untrained=True permits structural testing without checkpoint."""
    p_cfg = PerceptionConfig(checkpoint_path=None, num_classes=9)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    backend = RangeUNetBackend(p_cfg, s_cfg, torch.device("cpu"), allow_untrained=True)
    assert backend.is_ready is True
    assert backend.checkpoint_path is None


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
# 4. Multi-Frame Temporal State & Device Residency Tests
# ---------------------------------------------------------------------------
def test_temporal_state_device_residency_and_boundedness():
    """Verify that multi-frame temporal history stays on device and is bounded to 2 sweeps."""
    dev = torch.device("cpu")
    backend = RangeUNetBackend(
        config=PerceptionConfig(num_classes=9),
        sensor_config=SensorConfig(n_rows=64, n_cols=1024),
        device=dev,
        features_engine="torch",
    )
    source = SimulatorSource(n_steps=4, seed=123)

    # Frame 1
    res1 = backend.predict_device(source[0], dev_math=True)
    assert len(backend._device_temporal_state) == 1
    assert backend._device_temporal_state.sweeps[0].pts_world.device == dev
    assert len(backend.history) == 1

    # Frame 2
    res2 = backend.predict_device(source[1], dev_math=True)
    assert len(backend._device_temporal_state) == 2
    assert len(backend.history) == 2

    # Frame 3 (history must be bounded at max 2 sweeps)
    res3 = backend.predict_device(source[2], dev_math=True)
    assert len(backend._device_temporal_state) == 2
    assert len(backend.history) == 2

    # Frame 4
    res4 = backend.predict_device(source[3], dev_math=True)
    assert len(backend._device_temporal_state) == 2

    # Check reset behavior
    backend.reset()
    assert len(backend._device_temporal_state) == 0
    assert len(backend.history) == 0


def test_temporal_state_100_frames_longevity():
    """Verify that long streams do not leak memory or accumulate autograd graphs."""
    backend = RangeUNetBackend(
        config=PerceptionConfig(num_classes=9),
        sensor_config=SensorConfig(n_rows=64, n_cols=1024),
        device=torch.device("cpu"),
        features_engine="torch",
    )
    frame = SimulatorSource(n_steps=1, seed=42)[0]

    for _ in range(100):
        res = backend.predict_device(frame, dev_math=True)
        assert len(backend._device_temporal_state) <= 2

    assert len(backend._device_temporal_state) == 2
    for sw in backend._device_temporal_state.sweeps:
        assert sw.pts_world.grad_fn is None
        assert sw.pts_world.requires_grad is False


def test_explicit_prev_sweeps_precedence():
    """Verify that dataset-supplied prev_sweeps take precedence over internal temporal state."""
    backend = RangeUNetBackend(
        config=PerceptionConfig(num_classes=9),
        sensor_config=SensorConfig(n_rows=64, n_cols=1024),
        device=torch.device("cpu"),
        features_engine="torch",
    )
    source = SimulatorSource(n_steps=2, seed=123)
    f0 = source[0]
    f1 = source[1]

    pw0 = (f0.pts.astype(np.float64) @ f0.pose[:3, :3].T + f0.pose[:3, 3]).astype(np.float64)
    r0 = f0.ring.copy()
    explicit_prev = [(pw0, r0), None]

    frame_with_prev = LiDARFrame(
        pts=f1.pts,
        intensity=f1.intensity,
        ring=f1.ring,
        pose=f1.pose,
        sensor_origin=f1.sensor_origin,
        timestamp=f1.timestamp,
        frame_id="frame_with_explicit_prev",
        prev_sweeps=explicit_prev,
    )

    res_dev = backend.predict_device(frame_with_prev, dev_math=True)
    assert isinstance(res_dev, DevicePerceptionResult)
    assert res_dev.num_points == len(f1.pts)

    res_host = backend.predict(frame_with_prev)
    assert isinstance(res_host, PerceptionResult)
    assert res_host.num_points == len(f1.pts)


def test_predict_device_argument_validation(test_frame):
    """Verify that backend.predict() validates requested device matching backend device."""
    backend = RangeUNetBackend(
        config=PerceptionConfig(num_classes=9),
        sensor_config=SensorConfig(),
        device=torch.device("cpu"),
    )
    # Valid matching device
    res = backend.predict(test_frame, device=torch.device("cpu"))
    assert isinstance(res, PerceptionResult)

    # Incompatible device request
    incompatible = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    with pytest.raises(ConfigurationError, match="initialized on device"):
        backend.predict(test_frame, device=incompatible)


# ---------------------------------------------------------------------------
# 5. Contract Negative & Strict Validation Tests
# ---------------------------------------------------------------------------
def test_contract_negative_cases_host_perception_result():
    """Negative tests for PerceptionResult validating rejection of all invalid shapes and values."""
    n = 10
    cp = np.full((n, 9), 1.0 / 9.0, dtype=np.float32)
    mp = np.zeros(n, dtype=np.float32)
    sp = np.zeros(n, dtype=np.int64)
    mv = np.zeros(n, dtype=bool)

    # 1. Shape mismatch (wrong point count in class_probabilities)
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp[:5], moving_probabilities=mp, semantic_predictions=sp, is_moving=mv)

    # 2. Shape mismatch in moving_probabilities
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp[:5], semantic_predictions=sp, is_moving=mv)

    # 3. Shape mismatch in semantic_predictions
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp[:5], is_moving=mv)

    # 4. Shape mismatch in is_moving
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv[:5])

    # 5. Non-finite values (NaN / Inf in class_probabilities)
    cp_nan = cp.copy()
    cp_nan[0, 0] = np.nan
    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=cp_nan, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv)

    cp_inf = cp.copy()
    cp_inf[0, 0] = np.inf
    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=cp_inf, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv)

    # 6. Probabilities outside [0, 1]
    cp_neg = cp.copy()
    cp_neg[0, 0] = -0.5
    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=cp_neg, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv)

    cp_high = cp.copy()
    cp_high[0, 0] = 1.5
    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=cp_high, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv)

    # 7. Invalid class IDs (< 0 or >= num_classes)
    sp_neg = sp.copy()
    sp_neg[0] = -1
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp_neg, is_moving=mv)

    sp_oob = sp.copy()
    sp_oob[0] = 9
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp_oob, is_moving=mv)

    # 8. Invalid confidence shape or bounds
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv, confidence=np.ones(5, dtype=np.float32))

    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv, confidence=np.full(n, 2.0, dtype=np.float32))

    # 9. Invalid point_indices shape or coordinates < -1
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv, point_indices=(np.zeros(5, dtype=np.int64), np.zeros(n, dtype=np.int64)))

    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv, point_indices=(np.full(n, -2, dtype=np.int64), np.zeros(n, dtype=np.int64)))

    # 10. Wrong dtypes
    with pytest.raises(ContractError):
        PerceptionResult(class_probabilities=cp.astype(np.int32), moving_probabilities=mp, semantic_predictions=sp, is_moving=mv)


def test_device_perception_result_validation():
    """Verify DevicePerceptionResult.validate() method."""
    dev = torch.device("cpu")
    n = 10
    cp = torch.full((n, 9), 1.0 / 9.0, device=dev, dtype=torch.float32)
    mp = torch.zeros(n, device=dev, dtype=torch.float32)
    sp = torch.zeros(n, device=dev, dtype=torch.long)
    mv = torch.zeros(n, device=dev, dtype=torch.bool)

    # Valid
    valid_res = DevicePerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv, device=dev)
    assert valid_res.num_points == n

    # Non-tensor
    with pytest.raises(ContractError):
        DevicePerceptionResult(class_probabilities=cp.numpy(), moving_probabilities=mp, semantic_predictions=sp, is_moving=mv, device=dev)  # type: ignore

    # Non-finite
    cp_bad = cp.clone()
    cp_bad[0, 0] = float("nan")
    with pytest.raises(NumericalConsistencyError):
        DevicePerceptionResult(class_probabilities=cp_bad, moving_probabilities=mp, semantic_predictions=sp, is_moving=mv, device=dev)

    # Class ID out of bounds
    sp_bad = sp.clone()
    sp_bad[0] = 15
    with pytest.raises(ContractError):
        DevicePerceptionResult(class_probabilities=cp, moving_probabilities=mp, semantic_predictions=sp_bad, is_moving=mv, device=dev)


def test_device_perception_result_structural_vs_strict():
    """Verify that structural validation runs unconditionally, while numerical checks are gated."""
    dev = torch.device("cpu")
    n = 10
    cp = torch.full((n, 9), 1.0 / 9.0, device=dev, dtype=torch.float32)
    mp = torch.zeros(n, device=dev, dtype=torch.float32)
    sp = torch.zeros(n, device=dev, dtype=torch.long)
    mv = torch.zeros(n, device=dev, dtype=torch.bool)

    # 1. Structural failure: shape mismatch always raises even with strict=False
    cp_mismatch = torch.full((n + 2, 9), 1.0 / 9.0, device=dev, dtype=torch.float32)
    res_mismatch = DevicePerceptionResult.__new__(DevicePerceptionResult)
    res_mismatch.class_probabilities = cp_mismatch
    res_mismatch.moving_probabilities = mp
    res_mismatch.semantic_predictions = sp
    res_mismatch.is_moving = mv
    res_mismatch.device = dev
    res_mismatch.confidence = None
    res_mismatch.pts_world = None
    res_mismatch.point_indices = None
    res_mismatch.metadata = {}
    res_mismatch.strict_validation = False
    with pytest.raises(ContractError):
        res_mismatch.validate(strict=False)

    # 2. Structural pass with non-strict mode: numerical out-of-bounds does NOT raise when strict=False
    sp_bad = sp.clone()
    sp_bad[0] = 15
    res_num = DevicePerceptionResult.__new__(DevicePerceptionResult)
    res_num.class_probabilities = cp
    res_num.moving_probabilities = mp
    res_num.semantic_predictions = sp_bad
    res_num.is_moving = mv
    res_num.device = dev
    res_num.confidence = None
    res_num.pts_world = None
    res_num.point_indices = None
    res_num.metadata = {}
    res_num.strict_validation = False
    # Calling validate(strict=False) should pass structural checks without numerical failure
    res_num.validate(strict=False)

    # But calling validate(strict=True) must raise ContractError on numerical violation
    with pytest.raises(ContractError):
        res_num.validate(strict=True)


# ---------------------------------------------------------------------------
# 6. Latency Timing Instrumentation
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
# 7. Golden Frame Perception Test (Determinism)
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
# 8. Perception Factory & Registry
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
# 9. Runtime End-to-End Ingestion -> Perception -> Mapping
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


def test_point_correspondence_gather_and_consistency():
    """Verify that per-point semantic predictions, probabilities, and confidence correspond directly."""
    p_cfg = PerceptionConfig(num_classes=9, fp16=False)
    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    dev = torch.device("cpu")
    backend = RangeUNetBackend(p_cfg, s_cfg, dev, allow_untrained=True)

    # Construct frame with distinct points, including duplicates and boundary points
    pts = np.array([
        [10.0, 0.0, 0.0],    # directly forward
        [10.0, 0.0, 0.0],    # duplicate forward point (should map to identical row/col)
        [0.0, 10.0, 0.0],    # left 90 deg
        [0.0, -10.0, 0.0],   # right -90 deg
        [-10.0, 0.0, 0.0],   # behind 180 deg
    ], dtype=np.float32)
    n = len(pts)
    frame = LiDARFrame(
        pts=pts,
        intensity=np.full(n, 0.5, dtype=np.float32),
        ring=np.array([32, 32, 20, 45, 10], dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        sensor_origin=np.array([0.0, 0.0, 1.73], dtype=np.float32),
        timestamp=1.0,
        frame_id="corr_test",
    )

    dev_res = backend.predict_device(frame)
    assert dev_res.point_indices is not None
    assert isinstance(dev_res.point_indices, tuple) and len(dev_res.point_indices) == 2
    rows, cols = dev_res.point_indices
    assert rows.shape == (n,)
    assert cols.shape == (n,)

    # Verify all coordinates are within sensor image dimensions
    assert (rows >= 0).all() and (rows < s_cfg.n_rows).all()
    assert (cols >= 0).all() and (cols < s_cfg.n_cols).all()

    # Point 0 and Point 1 are identical in (x,y,z) and ring -> must have identical row/col
    assert rows[0].item() == rows[1].item()
    assert cols[0].item() == cols[1].item()
    torch.testing.assert_close(dev_res.class_probabilities[0], dev_res.class_probabilities[1])

    # Verify correspondence between probabilities, predictions, and confidence for every point
    for i in range(n):
        expected_cls = dev_res.class_probabilities[i].argmax()
        assert dev_res.semantic_predictions[i] == expected_cls
        expected_conf = dev_res.class_probabilities[i].max()
        torch.testing.assert_close(dev_res.point_confidence[i], expected_conf)

    # Verify to_host preserves correspondence arrays as numpy integer ndarrays
    host_res = dev_res.to_host()
    assert host_res.point_indices is not None
    assert isinstance(host_res.point_indices, tuple) and len(host_res.point_indices) == 2
    h_rows, h_cols = host_res.point_indices
    assert isinstance(h_rows, np.ndarray) and isinstance(h_cols, np.ndarray)
    assert h_rows.shape == (n,) and h_cols.shape == (n,)
    np.testing.assert_array_equal(h_rows, rows.numpy())
    np.testing.assert_array_equal(h_cols, cols.numpy())


# ---------------------------------------------------------------------------
# 10. Conditional CUDA Tests (Physically Verified when Available)
# ---------------------------------------------------------------------------
def test_cuda_perception_execution(test_frame):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable on current host (CPU environment); physical CUDA execution required")

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


def test_cuda_cpu_parity_when_available(test_frame):
    """Test CPU vs CUDA numerical parity when CUDA hardware is physically available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable on current host (CPU environment); physical CUDA execution required")

    s_cfg = SensorConfig(name="sim", n_rows=64, n_cols=1024)
    p_cfg_cpu = PerceptionConfig(checkpoint_path=CHECKPOINT_PATH, num_classes=9, fp16=False)
    p_cfg_cuda = PerceptionConfig(checkpoint_path=CHECKPOINT_PATH, num_classes=9, fp16=True)

    backend_cpu = RangeUNetBackend(p_cfg_cpu, s_cfg, torch.device("cpu"), features_engine="torch")
    backend_cuda = RangeUNetBackend(p_cfg_cuda, s_cfg, torch.device("cuda:0"), features_engine="torch")

    res_cpu = backend_cpu.predict(test_frame)
    res_cuda = backend_cuda.predict(test_frame)

    # Class predictions should match with high fidelity
    cls_match = np.mean(res_cpu.semantic_predictions == res_cuda.semantic_predictions)
    assert cls_match >= 0.99, f"Semantic prediction parity {cls_match:.3f} below 0.99"

    # Probabilities within reasonable FP32 vs FP16 mixed precision tolerance
    np.testing.assert_allclose(res_cuda.class_probabilities, res_cpu.class_probabilities, atol=2e-2)
    np.testing.assert_allclose(res_cuda.moving_probabilities, res_cpu.moving_probabilities, atol=2e-2)
