"""Tests for Phase 3 data contracts and extended LiDARFrame metadata."""
from __future__ import annotations

import numpy as np
import pytest

from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import ContractError, NumericalConsistencyError


def test_lidar_frame_source_id_and_annotations():
    pts = np.array([[10.0, 0.0, 0.0], [20.0, 1.0, -0.5]], dtype=np.float32)
    intensity = np.array([0.5, 0.8], dtype=np.float32)
    ring = np.array([0, 1], dtype=np.int16)
    pose = np.eye(4, dtype=np.float64)
    label = np.array([0, 7], dtype=np.int8)
    moving = np.array([False, True], dtype=bool)

    frame = LiDARFrame(
        pts=pts,
        intensity=intensity,
        ring=ring,
        pose=pose,
        source_id="simulator_run_01",
        label=label,
        moving=moving,
    )

    assert frame.source_id == "simulator_run_01"
    assert frame.has_annotations is True
    assert frame.label is not None
    assert frame.moving is not None

    # Strip annotations
    clean_frame = frame.without_annotations()
    assert clean_frame.has_annotations is False
    assert clean_frame.label is None
    assert clean_frame.moving is None
    assert clean_frame.source_id == "simulator_run_01"
    np.testing.assert_array_equal(clean_frame.pts, pts)


def test_lidar_frame_legacy_dict_preserves_source_id():
    pts = np.ones((5, 3), dtype=np.float32)
    intensity = np.zeros(5, dtype=np.float32)
    ring = np.zeros(5, dtype=np.int16)
    pose = np.eye(4, dtype=np.float64)

    frame = LiDARFrame(
        pts=pts,
        intensity=intensity,
        ring=ring,
        pose=pose,
        source_id="semantickitti/08",
        frame_id="000100",
        timestamp=10.0,
    )

    legacy = frame.to_legacy_dict()
    assert legacy["meta"]["source_id"] == "semantickitti/08"
    assert legacy["meta"]["frame_id"] == "000100"

    recovered = LiDARFrame.from_legacy_dict(legacy)
    assert recovered.source_id == "semantickitti/08"
    assert recovered.frame_id == "000100"
    assert recovered.timestamp == 10.0
    np.testing.assert_array_equal(recovered.pts, pts)


def test_lidar_frame_rejects_invalid_source_id():
    pts = np.ones((5, 3), dtype=np.float32)
    intensity = np.zeros(5, dtype=np.float32)
    ring = np.zeros(5, dtype=np.int16)
    pose = np.eye(4, dtype=np.float64)

    with pytest.raises(ContractError):
        LiDARFrame(
            pts=pts,
            intensity=intensity,
            ring=ring,
            pose=pose,
            source_id=12345,  # type: ignore
        )
