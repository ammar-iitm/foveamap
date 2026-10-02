"""Tests for LiDARPreprocessor and PreprocessConfig."""
from __future__ import annotations

import numpy as np
import pytest

from foveamap.core.contracts import LiDARFrame
from foveamap.core.config import PreprocessConfig
from foveamap.core.exceptions import ConfigurationError, NumericalConsistencyError
from foveamap.data.preprocess import LiDARPreprocessor


def _make_test_frame() -> LiDARFrame:
    # 6 points:
    # 0: origin [0, 0, 1.73] -> dist 0 -> self-hit and range < min_range
    # 1: close [0.5, 0.5, 1.73] -> dist ~0.71 -> within self-hit radius 1.0m
    # 2: valid mid-range [10.0, 0.0, 0.0] -> dist ~10.15m, z=0
    # 3: valid far [50.0, 10.0, 1.0] -> dist ~51.0m, z=1.0
    # 4: out of max range [150.0, 0.0, 0.0] -> dist ~150m > 100m
    # 5: extreme height [20.0, 0.0, 10.0] -> z=10.0
    pts = np.array([
        [0.0, 0.0, 1.73],
        [0.5, 0.5, 1.73],
        [10.0, 0.0, 0.0],
        [50.0, 10.0, 1.0],
        [150.0, 0.0, 0.0],
        [20.0, 0.0, 10.0],
    ], dtype=np.float32)
    intensity = np.array([0.1, 0.2, 0.5, 0.8, 0.9, 0.3], dtype=np.float32)
    ring = np.arange(6, dtype=np.int16)
    pose = np.eye(4, dtype=np.float64)
    label = np.array([0, 1, 2, 7, 5, 4], dtype=np.int8)
    moving = np.array([False, False, False, True, False, False], dtype=bool)

    return LiDARFrame(
        pts=pts,
        intensity=intensity,
        ring=ring,
        pose=pose,
        label=label,
        moving=moving,
        frame_id="test_filter_001",
        source_id="test_source",
    )


def test_preprocess_config_validation():
    with pytest.raises(ConfigurationError):
        PreprocessConfig(min_range_m=-1.0)
    with pytest.raises(ConfigurationError):
        PreprocessConfig(min_range_m=50.0, max_range_m=10.0)
    with pytest.raises(ConfigurationError):
        PreprocessConfig(z_min_m=5.0, z_max_m=2.0)
    with pytest.raises(ConfigurationError):
        PreprocessConfig(self_hit_radius_m=-0.5)


def test_preprocessor_range_and_self_hit_filtering():
    frame = _make_test_frame()
    cfg = PreprocessConfig(
        min_range_m=1.0,
        max_range_m=100.0,
        remove_self_hits=True,
        self_hit_radius_m=1.0,
    )
    pre = LiDARPreprocessor(cfg)
    filtered = pre.process(frame)

    assert isinstance(filtered, LiDARFrame)
    # Points 0 (range 0, self-hit), 1 (self-hit < 1.0m), and 4 (range 150m > 100m) should be filtered out
    # Points 2, 3, 5 should be retained
    assert filtered.num_points == 3
    np.testing.assert_allclose(filtered.pts[0], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(filtered.pts[1], [50.0, 10.0, 1.0])
    np.testing.assert_allclose(filtered.pts[2], [20.0, 0.0, 10.0])

    # Metadata should record preprocessing stats
    stats = filtered.metadata["preprocessing"]
    assert stats["original_points"] == 6
    assert stats["retained_points"] == 3
    assert stats["filtered_points"] == 3


def test_preprocessor_z_bounds_filtering():
    frame = _make_test_frame()
    cfg = PreprocessConfig(
        min_range_m=1.0,
        max_range_m=100.0,
        z_min_m=-0.5,
        z_max_m=2.0,
        self_hit_radius_m=1.0,
    )
    pre = LiDARPreprocessor(cfg)
    filtered = pre.process(frame)

    # Point 5 has z=10.0 > 2.0, so only points 2 and 3 remain
    assert filtered.num_points == 2
    np.testing.assert_allclose(filtered.pts[0], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(filtered.pts[1], [50.0, 10.0, 1.0])


def test_preprocessor_input_immutability():
    frame = _make_test_frame()
    pts_copy = frame.pts.copy()
    cfg = PreprocessConfig(min_range_m=2.0)
    pre = LiDARPreprocessor(cfg)
    _ = pre.process(frame)

    # Input frame must remain unmodified
    np.testing.assert_array_equal(frame.pts, pts_copy)
    assert frame.num_points == 6


def test_preprocessor_deterministic_execution():
    frame = _make_test_frame()
    cfg = PreprocessConfig(min_range_m=1.0, max_range_m=80.0, z_min_m=-1.0, z_max_m=5.0)
    pre = LiDARPreprocessor(cfg)

    out1 = pre.process(frame)
    out2 = pre.process(frame)

    np.testing.assert_array_equal(out1.pts, out2.pts)
    np.testing.assert_array_equal(out1.intensity, out2.intensity)
    np.testing.assert_array_equal(out1.ring, out2.ring)
    assert out1.metadata == out2.metadata


def test_preprocessor_pass_through_when_disabled():
    frame = _make_test_frame()
    cfg = PreprocessConfig(enabled=False)
    pre = LiDARPreprocessor(cfg)
    out = pre.process(frame)

    assert out is frame
    assert out.num_points == 6


def test_preprocessor_empty_result_handling():
    frame = _make_test_frame()
    cfg = PreprocessConfig(min_range_m=500.0, max_range_m=1000.0)
    pre = LiDARPreprocessor(cfg)
    out = pre.process(frame)

    assert out.num_points == 0
    assert out.pts.shape == (0, 3)
    assert out.intensity.shape == (0,)
    assert out.ring.shape == (0,)
    assert out.metadata["preprocessing"]["retained_points"] == 0


def test_preprocessor_edge_case_n0():
    empty_frame = LiDARFrame(
        pts=np.empty((0, 3), dtype=np.float32),
        intensity=np.empty((0,), dtype=np.float32),
        ring=np.empty((0,), dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="empty_frame",
    )
    pre = LiDARPreprocessor(PreprocessConfig(min_range_m=1.0, max_range_m=100.0))
    out = pre.process(empty_frame)

    assert out.num_points == 0
    assert out.pts.shape == (0, 3)
    assert out.metadata["preprocessing"]["original_points"] == 0
    assert out.metadata["preprocessing"]["retained_points"] == 0


def test_preprocessor_edge_case_n1():
    single_pt_frame = LiDARFrame(
        pts=np.array([[15.0, 0.0, 1.0]], dtype=np.float32),
        intensity=np.array([0.5], dtype=np.float32),
        ring=np.array([10], dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="single_pt",
    )
    pre = LiDARPreprocessor(PreprocessConfig(min_range_m=1.0, max_range_m=100.0))
    out = pre.process(single_pt_frame)

    assert out.num_points == 1
    assert out.pts.shape == (1, 3)
    np.testing.assert_allclose(out.pts[0], [15.0, 0.0, 1.0])


def test_preprocessor_edge_case_large_n():
    n_points = 100_000
    rng = np.random.default_rng(42)
    # Generate points uniformly between 0 and 150m
    r = rng.uniform(0.1, 150.0, size=n_points).astype(np.float32)
    theta = rng.uniform(-np.pi, np.pi, size=n_points).astype(np.float32)
    pts = np.stack([
        r * np.cos(theta),
        r * np.sin(theta),
        rng.uniform(-2.0, 5.0, size=n_points).astype(np.float32),
    ], axis=1)

    large_frame = LiDARFrame(
        pts=pts,
        intensity=rng.uniform(0.0, 1.0, size=n_points).astype(np.float32),
        ring=(rng.integers(0, 64, size=n_points)).astype(np.int16),
        pose=np.eye(4, dtype=np.float64),
        frame_id="large_cloud",
    )

    pre = LiDARPreprocessor(PreprocessConfig(
        min_range_m=2.0,
        max_range_m=80.0,
        remove_self_hits=True,
        self_hit_radius_m=2.0,
    ))
    out = pre.process(large_frame)

    assert isinstance(out, LiDARFrame)
    assert 0 < out.num_points < n_points
    # Verify all retained points are within [2.0, 80.0] range
    d = np.linalg.norm(out.pts - out.sensor_origin, axis=1)
    assert np.all(d >= 2.0)
    assert np.all(d <= 80.0)
    # Verify input was untouched
    assert large_frame.num_points == n_points


def test_preprocessor_all_points_in_self_hit():
    # Points all within 0.5m of [0, 0, 1.73]
    pts = np.array([
        [0.1, 0.0, 1.73],
        [0.0, 0.2, 1.73],
        [-0.1, -0.1, 1.73],
    ], dtype=np.float32)
    frame = LiDARFrame(
        pts=pts,
        intensity=np.ones(3, dtype=np.float32),
        ring=np.zeros(3, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
    )
    pre = LiDARPreprocessor(PreprocessConfig(min_range_m=0.01, self_hit_radius_m=1.0, remove_self_hits=True))
    out = pre.process(frame)
    assert out.num_points == 0
    assert out.metadata["preprocessing"]["filtered_points"] == 3

