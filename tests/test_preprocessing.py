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
