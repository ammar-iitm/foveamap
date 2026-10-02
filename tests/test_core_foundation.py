"""Tests for FoveaMap Phase 1 foundation: contracts, config, exceptions, packaging."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import pytest
import numpy as np

import foveamap
from foveamap.core.contracts import LiDARFrame, PerceptionResult, MapSnapshot
from foveamap.core.config import (
    TierConfig,
    GridConfig,
    SensorConfig,
    PerceptionConfig,
    TerrainConfig,
    RuntimeConfig,
    FoveaMapConfig,
)
from foveamap.core.exceptions import (
    FoveaMapError,
    ConfigurationError,
    ContractError,
    DataAdapterError,
    PerceptionError,
    MappingError,
    NumericalConsistencyError,
)
from foveamap.grid import TierLayers


# ----------------------------------------------------------------------------
# Packaging & Imports
# ----------------------------------------------------------------------------
def test_package_metadata():
    assert hasattr(foveamap, "__version__")
    assert foveamap.__version__ == "0.1.0"
    assert foveamap.LiDARFrame is LiDARFrame
    assert foveamap.FoveaMapConfig is FoveaMapConfig
    assert hasattr(foveamap, "PreprocessConfig")
    assert hasattr(foveamap, "LiDARSource")
    assert hasattr(foveamap, "LiDARPreprocessor")
    assert hasattr(foveamap, "create_source")


# ----------------------------------------------------------------------------
# Exceptions Hierarchy
# ----------------------------------------------------------------------------
def test_exception_hierarchy():
    for exc_cls in (
        ConfigurationError,
        ContractError,
        DataAdapterError,
        PerceptionError,
        MappingError,
        NumericalConsistencyError,
    ):
        err = exc_cls("test failure")
        assert isinstance(err, FoveaMapError)
        assert isinstance(err, Exception)


# ----------------------------------------------------------------------------
# LiDARFrame Contract
# ----------------------------------------------------------------------------
def _make_valid_frame_data(n=100):
    pts = np.zeros((n, 3), dtype=np.float32)
    inten = np.zeros(n, dtype=np.float32)
    ring = np.zeros(n, dtype=np.int16)
    pose = np.eye(4, dtype=np.float64)
    sensor = np.array([0.0, 0.0, 1.73], dtype=np.float32)
    return pts, inten, ring, pose, sensor


def test_lidar_frame_valid_construction():
    pts, inten, ring, pose, sensor = _make_valid_frame_data(50)
    frame = LiDARFrame(
        pts=pts,
        intensity=inten,
        ring=ring,
        pose=pose,
        sensor_origin=sensor,
        timestamp=1.23,
        frame_id="sweep_001",
    )
    assert frame.num_points == 50
    assert frame.timestamp == 1.23
    assert frame.frame_id == "sweep_001"


def test_lidar_frame_rejects_invalid_shapes():
    pts, inten, ring, pose, sensor = _make_valid_frame_data(50)
    # Invalid pts shape
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts[:, :2], intensity=inten, ring=ring, pose=pose, sensor_origin=sensor)
    # Mismatched intensity length
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts, intensity=inten[:10], ring=ring, pose=pose, sensor_origin=sensor)
    # Mismatched ring length
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts, intensity=inten, ring=ring[:10], pose=pose, sensor_origin=sensor)
    # Invalid pose shape
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts, intensity=inten, ring=ring, pose=pose[:3, :3], sensor_origin=sensor)


def test_lidar_frame_rejects_invalid_types_and_pose():
    pts, inten, ring, pose, sensor = _make_valid_frame_data(50)
    # Non-floating pts
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts.astype(np.int32), intensity=inten, ring=ring, pose=pose, sensor_origin=sensor)
    # Non-integer ring
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts, intensity=inten, ring=ring.astype(np.float32), pose=pose, sensor_origin=sensor)
    # Invalid SE(3) pose bottom row
    bad_pose = pose.copy()
    bad_pose[3, :] = [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts, intensity=inten, ring=ring, pose=bad_pose, sensor_origin=sensor)

    # Non-orthogonal 3x3 rotation block
    bad_rot = pose.copy()
    bad_rot[:3, :3] = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts, intensity=inten, ring=ring, pose=bad_rot, sensor_origin=sensor)

    # Reflection matrix (det = -1)
    refl_pose = pose.copy()
    refl_pose[0, 0] = -1.0
    with pytest.raises(ContractError):
        LiDARFrame(pts=pts, intensity=inten, ring=ring, pose=refl_pose, sensor_origin=sensor)

    # Negative intensity
    bad_inten = inten.copy()
    bad_inten[0] = -0.5
    with pytest.raises(NumericalConsistencyError):
        LiDARFrame(pts=pts, intensity=bad_inten, ring=ring, pose=pose, sensor_origin=sensor)


def test_lidar_frame_rejects_nan_and_inf():
    pts, inten, ring, pose, sensor = _make_valid_frame_data(50)
    bad_pts = pts.copy()
    bad_pts[0, 0] = np.nan
    with pytest.raises(NumericalConsistencyError):
        LiDARFrame(pts=bad_pts, intensity=inten, ring=ring, pose=pose, sensor_origin=sensor)

    bad_pose = pose.copy()
    bad_pose[0, 0] = np.inf
    with pytest.raises(NumericalConsistencyError):
        LiDARFrame(pts=pts, intensity=inten, ring=ring, pose=bad_pose, sensor_origin=sensor)


def test_lidar_frame_prev_sweeps_validation():
    pts, inten, ring, pose, sensor = _make_valid_frame_data(50)
    pw = np.zeros((30, 3), dtype=np.float64)
    r = np.zeros(30, dtype=np.int16)
    # Valid prev_sweeps with None and valid tuple
    frame = LiDARFrame(
        pts=pts, intensity=inten, ring=ring, pose=pose, sensor_origin=sensor,
        prev_sweeps=[(pw, r), None]
    )
    assert len(frame.prev_sweeps) == 2

    # Malformed prev_sweeps item
    with pytest.raises(ContractError):
        LiDARFrame(
            pts=pts, intensity=inten, ring=ring, pose=pose, sensor_origin=sensor,
            prev_sweeps=["invalid_entry"]
        )


def test_lidar_frame_legacy_dict_roundtrip():
    pts, inten, ring, pose, sensor = _make_valid_frame_data(50)
    label = np.ones(50, dtype=np.int8)
    moving = np.zeros(50, dtype=bool)
    frame = LiDARFrame(
        pts=pts,
        intensity=inten,
        ring=ring,
        pose=pose,
        sensor_origin=sensor,
        timestamp=4.56,
        frame_id="sweep_042",
        label=label,
        moving=moving,
        metadata={"t": 42, "sequence": "08"},
    )
    legacy = frame.to_legacy_dict()
    assert "pts" in legacy and "inten" in legacy and "ring" in legacy
    assert "label" in legacy and "moving" in legacy and "pose" in legacy
    assert legacy["meta"]["sequence"] == "08"
    assert legacy["meta"]["t"] == 42

    reconstructed = LiDARFrame.from_legacy_dict(legacy)
    assert reconstructed.num_points == frame.num_points
    assert np.array_equal(reconstructed.pts, frame.pts)
    assert np.array_equal(reconstructed.intensity, frame.intensity)
    assert np.array_equal(reconstructed.ring, frame.ring)
    assert np.array_equal(reconstructed.label, frame.label)
    assert reconstructed.metadata["sequence"] == "08"


# ----------------------------------------------------------------------------
# PerceptionResult Contract
# ----------------------------------------------------------------------------
def test_perception_result_valid_construction():
    n = 100
    c_probs = np.full((n, 9), 1.0 / 9.0, dtype=np.float32)
    m_probs = np.zeros(n, dtype=np.float32)
    preds = np.zeros(n, dtype=np.int64)
    is_mv = np.zeros(n, dtype=bool)

    res = PerceptionResult(
        class_probabilities=c_probs,
        moving_probabilities=m_probs,
        semantic_predictions=preds,
        is_moving=is_mv,
    )
    assert res.num_points == n


def test_perception_result_rejects_dimension_mismatch():
    n = 100
    c_probs = np.full((n, 9), 1.0 / 9.0, dtype=np.float32)
    m_probs = np.zeros(n, dtype=np.float32)
    preds = np.zeros(n, dtype=np.int64)
    is_mv = np.zeros(n, dtype=bool)

    with pytest.raises(ContractError):
        PerceptionResult(
            class_probabilities=c_probs[:50],
            moving_probabilities=m_probs,
            semantic_predictions=preds,
            is_moving=is_mv,
        )


def test_perception_result_rejects_nan_and_out_of_bounds():
    n = 50
    c_probs = np.full((n, 9), 1.0 / 9.0, dtype=np.float32)
    m_probs = np.zeros(n, dtype=np.float32)
    preds = np.zeros(n, dtype=np.int64)
    is_mv = np.zeros(n, dtype=bool)

    # NaN in class probabilities
    bad_c = c_probs.copy()
    bad_c[0, 0] = np.nan
    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=bad_c, moving_probabilities=m_probs,
                         semantic_predictions=preds, is_moving=is_mv)

    # Probability > 1.0
    bad_p = c_probs.copy()
    bad_p[0, 0] = 1.5
    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=bad_p, moving_probabilities=m_probs,
                         semantic_predictions=preds, is_moving=is_mv)

    # Probability < 0.0
    bad_m = m_probs.copy()
    bad_m[0] = -0.1
    with pytest.raises(NumericalConsistencyError):
        PerceptionResult(class_probabilities=c_probs, moving_probabilities=bad_m,
                         semantic_predictions=preds, is_moving=is_mv)


# ----------------------------------------------------------------------------
# MapSnapshot Contract & Ownership Semantics
# ----------------------------------------------------------------------------
def test_map_snapshot_construction_and_semantics():
    s0 = TierLayers(400)
    s1 = TierLayers(400)
    snap = MapSnapshot(
        timestamp=10.0,
        frame_id="frame_100",
        ego_pose=np.eye(4),
        origins=[(-200, -200), (-200, -200)],
        tier_states=(s0, s1),
    )
    assert snap.num_tiers == 2
    assert snap.total_cells == 320000
    assert snap.total_memory_bytes == s0.nbytes + s1.nbytes
    assert snap.get_tier(0) is s0

    # FrozenInstanceError when attempting attribute reassignment
    with pytest.raises(FrozenInstanceError):
        snap.timestamp = 20.0  # type: ignore

    # Out of range tier index
    with pytest.raises(IndexError):
        snap.get_tier(99)

    # Origins count mismatch
    with pytest.raises(ContractError):
        MapSnapshot(
            timestamp=10.0,
            frame_id="frame_100",
            ego_pose=np.eye(4),
            origins=[(-200, -200)],  # only 1 origin for 2 tiers
            tier_states=(s0, s1),
        )


def test_map_snapshot_detachment_and_immutability():
    s0 = TierLayers(400)
    external_pose = np.eye(4)
    external_origins = [(-200, -200)]
    external_meta = {"run_id": "test_123"}
    snap = MapSnapshot(
        timestamp=10.0,
        frame_id="frame_100",
        ego_pose=external_pose,
        origins=external_origins,
        tier_states=(s0,),
        metadata=external_meta,
    )
    # Mutating external inputs must NOT mutate published snapshot
    external_pose[0, 3] = 999.0
    external_origins[0] = (0, 0)
    external_meta["run_id"] = "corrupted"

    assert snap.ego_pose[0, 3] == 0.0
    assert snap.origins[0] == (-200, -200)
    assert snap.metadata["run_id"] == "test_123"

    # Mutating published snapshot's ego_pose directly must fail (write-protected array)
    with pytest.raises(ValueError):
        snap.ego_pose[0, 3] = 999.0


# ----------------------------------------------------------------------------
# Configuration & Validation
# ----------------------------------------------------------------------------
def test_tier_config_validation():
    t = TierConfig(0.05, 10.0, alpha=0.3)
    assert t.cell_size_m == 0.05
    assert t.half_extent_m == 10.0

    with pytest.raises(ConfigurationError):
        TierConfig(-0.05, 10.0)
    with pytest.raises(ConfigurationError):
        TierConfig(0.05, -10.0)
    with pytest.raises(ConfigurationError):
        TierConfig(0.05, 10.0, alpha=0.0)
    with pytest.raises(ConfigurationError):
        TierConfig(0.05, 10.0, alpha=1.5)


def test_grid_config_all_presets():
    # Verify all 4 supported presets construct and validate cleanly
    spec = GridConfig.from_preset("spec")
    assert len(spec.tiers) == 2
    assert spec.base_resolution == 0.05
    assert spec.coarse_resolution == 0.50

    graded = GridConfig.from_preset("graded")
    assert len(graded.tiers) == 3

    u5 = GridConfig.from_preset("uniform5")
    assert len(u5.tiers) == 1
    assert u5.base_resolution == 0.05

    u50 = GridConfig.from_preset("uniform50")
    assert len(u50.tiers) == 1
    assert u50.base_resolution == 0.50

    with pytest.raises(ConfigurationError):
        GridConfig.from_preset("nonexistent")


def test_grid_config_invariants():
    # Inverted tier resolutions must fail
    with pytest.raises(ConfigurationError):
        GridConfig(tiers=(TierConfig(0.50, 10.0), TierConfig(0.05, 100.0)))

    # Non-integer multiple of base must fail
    with pytest.raises(ConfigurationError):
        GridConfig(tiers=(TierConfig(0.05, 10.0), TierConfig(0.07, 100.0)))

    # Empty tiers must fail
    with pytest.raises(ConfigurationError):
        GridConfig(tiers=())


def test_sensor_config_validation():
    cfg = SensorConfig(n_rows=64, n_cols=1024)
    assert cfg.n_rows == 64

    with pytest.raises(ConfigurationError):
        SensorConfig(n_rows=0)
    with pytest.raises(ConfigurationError):
        SensorConfig(fov_up_deg=-10.0, fov_down_deg=5.0)
    with pytest.raises(ConfigurationError):
        SensorConfig(min_range_m=50.0, max_range_m=10.0)
    with pytest.raises(ConfigurationError):
        SensorConfig(hz=0.0)


def test_perception_config_validation():
    cfg = PerceptionConfig(num_classes=9, confidence_threshold=0.5)
    assert cfg.num_classes == 9

    with pytest.raises(ConfigurationError):
        PerceptionConfig(num_classes=0)
    with pytest.raises(ConfigurationError):
        PerceptionConfig(confidence_threshold=1.5)
    with pytest.raises(ConfigurationError):
        PerceptionConfig(num_classes=9, active_classes=(True,) * 8)


def test_terrain_config_validation():
    cfg = TerrainConfig()
    assert cfg.vehicle_clearance_m == 2.5
    assert len(cfg.cost_priors) == 256

    with pytest.raises(ConfigurationError):
        TerrainConfig(vehicle_clearance_m=0.0)
    with pytest.raises(ConfigurationError):
        TerrainConfig(cost_priors=(0,) * 10)


def test_runtime_config_validation():
    rc = RuntimeConfig(grid_engine="numpy", features_engine="torch")
    assert rc.grid_engine == "numpy"

    with pytest.raises(ConfigurationError):
        RuntimeConfig(grid_engine="invalid_engine")


def test_foveamap_unified_config():
    config = FoveaMapConfig()
    assert config.grid.profile_name == "spec"
    assert config.sensor.n_rows == 64
    assert config.perception.num_classes == 9
    assert config.terrain.vehicle_clearance_m == 2.5
    assert config.runtime.grid_engine == "numpy"
