"""Tests for LiDAR data source adapters (Simulator, SemanticKITTI, nuScenes, File-based)."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import DataAdapterError
from foveamap.data.source import LiDARSource
from foveamap.data.sim import SimulatorSource
from foveamap.data.kitti import SemanticKITTISource
from foveamap.data.nuscenes import NuScenesSource
from foveamap.data.file import (
    BinFileSource,
    PcdFileSource,
    NpyFileSource,
    FileSequenceSource,
    parse_pcd,
    parse_bin,
    parse_npy,
)
from foveamap.sim import simulate_sequence, save_sequence
from foveamap.frames import sim_frames
from mock_semantickitti import write_mock as write_kitti_mock
from mock_nuscenes import write_mock as write_nusc_mock


# ---------------------------------------------------------------------------
# SimulatorSource Tests
# ---------------------------------------------------------------------------
def test_simulator_source_lifecycle_and_determinism():
    source = SimulatorSource(n_steps=5, seed=42)
    assert isinstance(source, LiDARSource)
    assert len(source) == 5
    assert source.source_id == "simulator"
    assert source.sensor_config.n_rows == 64

    # Iteration
    frames1 = list(source)
    assert len(frames1) == 5
    for i, f in enumerate(frames1):
        assert isinstance(f, LiDARFrame)
        assert f.frame_id == f"sim_{i:04d}"
        assert f.source_id == "simulator"

    # Reset and replay
    source.reset()
    frames2 = list(source)
    assert len(frames2) == 5
    for f1, f2 in zip(frames1, frames2):
        np.testing.assert_array_equal(f1.pts, f2.pts)
        np.testing.assert_array_equal(f1.pose, f2.pose)
        assert f1.frame_id == f2.frame_id

    # Random access indexing
    frame_idx2 = source[2]
    np.testing.assert_array_equal(frame_idx2.pts, frames1[2].pts)

    with pytest.raises(IndexError):
        _ = source[10]


# ---------------------------------------------------------------------------
# SemanticKITTISource Tests
# ---------------------------------------------------------------------------
def test_semantickitti_source_adapter(tmp_path):
    p = str(tmp_path / "sim.npz")
    save_sequence(p, simulate_sequence(seed=123, n_frames=4))
    raw_frames, _ = sim_frames(p)
    root = str(tmp_path / "kitti")
    write_kitti_mock(root, raw_frames)

    source = SemanticKITTISource(root=root, sequence="08")
    assert isinstance(source, LiDARSource)
    assert len(source) == 4
    assert source.source_id == "semantickitti/08"
    assert source.sensor_config.n_rows == 64

    # Iterate
    frames = list(source)
    assert len(frames) == 4
    for f in frames:
        assert isinstance(f, LiDARFrame)
        assert f.source_id == "semantickitti/08"
        assert f.pts.shape[1] == 3
        assert len(f.ring) == len(f.pts)

    # Deterministic replay after reset
    source.reset()
    f_replay = next(iter(source))
    np.testing.assert_array_equal(f_replay.pts, frames[0].pts)


def test_semantickitti_source_raises_on_missing_dir(tmp_path):
    with pytest.raises(DataAdapterError):
        SemanticKITTISource(root=str(tmp_path / "nonexistent"), sequence="08")


# ---------------------------------------------------------------------------
# NuScenesSource Tests
# ---------------------------------------------------------------------------
def test_nuscenes_source_adapter(tmp_path):
    root = str(tmp_path / "nusc")
    write_nusc_mock(root, scenes=(("scene-0103", 5),), n_sweeps=15)

    source = NuScenesSource(root=root, version="v1.0-mini", scene_name="scene-0103")
    assert isinstance(source, LiDARSource)
    assert len(source) == 2  # 2 keyframes in mock scene
    assert source.source_id == "nuscenes/scene-0103"
    assert source.sensor_config.n_rows == 64

    frames = list(source)
    assert len(frames) == 2
    for f in frames:
        assert isinstance(f, LiDARFrame)
        assert f.source_id == "nuscenes/scene-0103"
        assert f.pts.ndim == 2

    # Reset
    source.reset()
    assert next(iter(source)).frame_id == frames[0].frame_id


def test_nuscenes_source_raises_on_invalid_scene(tmp_path):
    root = str(tmp_path / "nusc_empty")
    write_nusc_mock(root, scenes=(("scene-0103", 5),), n_sweeps=15)
    with pytest.raises(DataAdapterError):
        NuScenesSource(root=root, scene_name="scene-nonexistent")


# ---------------------------------------------------------------------------
# File-Based Sources Tests (.bin, .pcd, .npy, FileSequenceSource)
# ---------------------------------------------------------------------------
def test_bin_file_source(tmp_path):
    bin_file = tmp_path / "scan.bin"
    # Create 50 points with 4 floats (x, y, z, intensity)
    pts = np.random.default_rng(1).uniform(1.0, 30.0, size=(50, 4)).astype(np.float32)
    pts.tofile(str(bin_file))

    source = BinFileSource(bin_file)
    assert len(source) == 1
    assert "scan" in source.source_id
    frame = source[0]
    assert isinstance(frame, LiDARFrame)
    assert frame.num_points == 50
    np.testing.assert_allclose(frame.pts, pts[:, :3])


def test_pcd_file_source_ascii(tmp_path):
    pcd_file = tmp_path / "cloud_ascii.pcd"
    content = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity ring
SIZE 4 4 4 4 2
TYPE F F F F I
COUNT 1 1 1 1 1
WIDTH 3
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 3
DATA ascii
1.0 2.0 3.0 0.5 0
4.0 5.0 6.0 0.8 1
7.0 8.0 9.0 0.2 2
"""
    pcd_file.write_text(content, encoding="ascii")

    source = PcdFileSource(pcd_file)
    assert len(source) == 1
    frame = source[0]
    assert frame.num_points == 3
    np.testing.assert_allclose(frame.pts[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(frame.intensity[1], 0.8)
    assert frame.ring[2] == 2


def test_pcd_file_source_binary(tmp_path):
    pcd_file = tmp_path / "cloud_binary.pcd"
    header = (
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH 4\n"
        "HEIGHT 1\n"
        "POINTS 4\n"
        "DATA binary\n"
    ).encode("ascii")

    pts = np.arange(16, dtype=np.float32).reshape(4, 4)
    with open(pcd_file, "wb") as fh:
        fh.write(header)
        fh.write(pts.tobytes())

    source = PcdFileSource(pcd_file)
    assert len(source) == 1
    frame = source[0]
    assert frame.num_points == 4
    np.testing.assert_allclose(frame.pts[0], [0.0, 1.0, 2.0])


def test_npy_file_source(tmp_path):
    npy_file = tmp_path / "points.npy"
    data = np.ones((25, 3), dtype=np.float32) * 5.0
    np.save(str(npy_file), data)

    source = NpyFileSource(npy_file)
    assert len(source) == 1
    frame = source[0]
    assert frame.num_points == 25
    np.testing.assert_allclose(frame.pts[0], [5.0, 5.0, 5.0])


def test_file_sequence_source(tmp_path):
    seq_dir = tmp_path / "scans"
    seq_dir.mkdir()
    # Create 3 files: 00.bin, 01.bin, 02.bin
    for i in range(3):
        data = np.full((10, 4), fill_value=float(i + 1), dtype=np.float32)
        data.tofile(str(seq_dir / f"{i:02d}.bin"))

    source = FileSequenceSource(seq_dir, pattern="*.bin", hz=10.0)
    assert len(source) == 3
    frames = list(source)
    assert len(frames) == 3

    assert frames[0].frame_id == "00"
    assert frames[1].frame_id == "01"
    assert frames[2].frame_id == "02"
    assert frames[0].timestamp == 0.0
    assert abs(frames[1].timestamp - 0.1) < 1e-5
    assert abs(frames[2].timestamp - 0.2) < 1e-5
    np.testing.assert_allclose(frames[1].pts[0], [2.0, 2.0, 2.0])
