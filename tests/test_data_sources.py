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
    FileLiDARSource,
    BinFileSource,
    PcdFileSource,
    NpyFileSource,
    FileSequenceSource,
    parse_pcd,
    parse_bin,
    parse_npy,
    normalize_intensity,
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


# ---------------------------------------------------------------------------
# Intensity Normalization Tests
# ---------------------------------------------------------------------------
def test_intensity_normalization_policies():
    # 1. Auto mode with 16-bit sensor data (> 255.0)
    raw_16bit = np.array([0.0, 1000.0, 65535.0], dtype=np.float32)
    norm, prov = normalize_intensity(raw_16bit, mode="auto")
    assert prov == "auto_scaled_16bit"
    assert norm.max() <= 1.0
    np.testing.assert_allclose(norm, [0.0, 1000.0 / 65535.0, 1.0], atol=1e-5)

    # 2. Auto mode with 8-bit sensor data (> 1.5 and <= 255.0)
    raw_8bit = np.array([0.0, 128.0, 255.0], dtype=np.float32)
    norm, prov = normalize_intensity(raw_8bit, mode="auto")
    assert prov == "auto_scaled_8bit"
    assert norm.max() <= 1.0
    np.testing.assert_allclose(norm, [0.0, 128.0 / 255.0, 1.0], atol=1e-5)

    # 3. Auto mode with already normalized unit data (<= 1.0)
    raw_unit = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    norm, prov = normalize_intensity(raw_unit, mode="auto")
    assert prov == "auto_unit_range"
    np.testing.assert_array_equal(norm, raw_unit)

    # 4. Explicit modes
    norm_255, prov = normalize_intensity(raw_8bit, mode="scale_255")
    assert prov == "scale_255"
    np.testing.assert_allclose(norm_255, [0.0, 128.0 / 255.0, 1.0], atol=1e-5)

    norm_raw, prov = normalize_intensity(raw_8bit, mode="raw")
    assert prov == "raw"
    np.testing.assert_array_equal(norm_raw, raw_8bit)

    # 5. Invalid mode
    with pytest.raises(DataAdapterError):
        normalize_intensity(raw_8bit, mode="invalid_unknown_mode")


# ---------------------------------------------------------------------------
# PCD Parser Edge Cases & Heterogeneous Binary Fields
# ---------------------------------------------------------------------------
def test_pcd_parser_heterogeneous_binary_fields(tmp_path):
    pcd_file = tmp_path / "heterogeneous.pcd"
    # Layout: x (F, 4), y (F, 4), z (F, 4), intensity (F, 4), ring (U, 2)
    header = (
        "VERSION 0.7\n"
        "FIELDS x y z intensity ring\n"
        "SIZE 4 4 4 4 2\n"
        "TYPE F F F F U\n"
        "COUNT 1 1 1 1 1\n"
        "WIDTH 3\n"
        "HEIGHT 1\n"
        "POINTS 3\n"
        "DATA binary\n"
    ).encode("ascii")

    # Define custom struct: 4 float32s + 1 uint16 (total 18 bytes per point)
    dtype = np.dtype([
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("ring", np.uint16),
    ])
    records = np.zeros(3, dtype=dtype)
    records[0] = (1.0, 2.0, 3.0, 0.4, 10)
    records[1] = (4.0, 5.0, 6.0, 0.8, 20)
    records[2] = (7.0, 8.0, 9.0, 1.0, 30)

    with open(pcd_file, "wb") as fh:
        fh.write(header)
        fh.write(records.tobytes())

    parsed, prov = parse_pcd(pcd_file)
    assert parsed["pts"].shape == (3, 3)
    np.testing.assert_allclose(parsed["pts"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(parsed["pts"][2], [7.0, 8.0, 9.0])
    np.testing.assert_allclose(parsed["intensity"], [0.4, 0.8, 1.0])
    assert parsed["ring"].dtype == np.int16
    np.testing.assert_array_equal(parsed["ring"], [10, 20, 30])


def test_pcd_parser_rejects_unsupported_or_malformed(tmp_path):
    # 1. Missing DATA section
    no_data_file = tmp_path / "no_data.pcd"
    no_data_file.write_text("VERSION 0.7\nFIELDS x y z\nWIDTH 1\n", encoding="ascii")
    with pytest.raises(DataAdapterError, match="no DATA section found"):
        parse_pcd(no_data_file)

    # 2. Binary compressed (LZF) PCD
    lzf_file = tmp_path / "compressed.pcd"
    lzf_file.write_text("VERSION 0.7\nFIELDS x y z\nDATA binary_compressed\n", encoding="ascii")
    with pytest.raises(DataAdapterError, match="Binary compressed"):
        parse_pcd(lzf_file)

    # 3. Missing coordinate fields (no z)
    no_z_file = tmp_path / "no_z.pcd"
    no_z_file.write_text("VERSION 0.7\nFIELDS x y\nSIZE 4 4\nTYPE F F\nCOUNT 1 1\nWIDTH 1\nDATA ascii\n1 2\n", encoding="ascii")
    with pytest.raises(DataAdapterError, match="must contain x, y, and z"):
        parse_pcd(no_z_file)

    # 4. Truncated binary file
    trunc_file = tmp_path / "trunc.pcd"
    header = "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 10\nPOINTS 10\nDATA binary\n".encode("ascii")
    with open(trunc_file, "wb") as fh:
        fh.write(header)
        fh.write(b"\x00" * 12)  # only 1 point (12 bytes) instead of 10 points (120 bytes)
    with pytest.raises(DataAdapterError, match="Truncated binary PCD"):
        parse_pcd(trunc_file)


# ---------------------------------------------------------------------------
# BIN and NPY Parser Error Handling
# ---------------------------------------------------------------------------
def test_bin_parser_error_handling(tmp_path):
    # Truncated binary: 13 bytes is not a multiple of 4 bytes (float32)
    bad_bin = tmp_path / "corrupt.bin"
    bad_bin.write_bytes(b"\x00" * 13)
    with pytest.raises(DataAdapterError):
        parse_bin(bad_bin)

    # Length not divisible by columns
    col_bin = tmp_path / "col.bin"
    # 7 floats is not divisible by 4
    np.ones(7, dtype=np.float32).tofile(str(col_bin))
    with pytest.raises(DataAdapterError):
        parse_bin(col_bin, columns=4)


def test_npy_npz_archive_parsing(tmp_path):
    # .npz archive with 'pts', 'intensity', 'ring'
    npz_file = tmp_path / "archive.npz"
    pts = np.ones((10, 3), dtype=np.float32)
    intensity = np.full(10, 0.5, dtype=np.float32)
    ring = np.arange(10, dtype=np.int16)
    np.savez(str(npz_file), pts=pts, intensity=intensity, ring=ring)

    parsed, prov = parse_npy(npz_file)
    assert parsed["pts"].shape == (10, 3)
    np.testing.assert_array_equal(parsed["ring"], ring)

    # 1D array should be rejected
    bad_npy = tmp_path / "bad.npy"
    np.save(str(bad_npy), np.ones(10, dtype=np.float32))
    with pytest.raises(DataAdapterError):
        parse_npy(bad_npy)


# ---------------------------------------------------------------------------
# Sensor Origin Policy & Provenance Tests
# ---------------------------------------------------------------------------
def test_sensor_origin_policy(tmp_path):
    npy_file = tmp_path / "origin_test.npy"
    np.save(str(npy_file), np.ones((5, 3), dtype=np.float32))

    # 1. Default policy: unknown origin defaults to [0, 0, 0] (optical center)
    s_default = FileLiDARSource(npy_file)
    np.testing.assert_allclose(s_default.sensor_origin, [0.0, 0.0, 0.0])
    frame_default = s_default[0]
    assert frame_default.metadata["sensor_origin_provenance"] == "default_lidar_center"

    # 2. User-supplied origin: explicitly configured
    s_custom = FileLiDARSource(npy_file, sensor_origin=[0.0, 0.0, 1.73])
    np.testing.assert_allclose(s_custom.sensor_origin, [0.0, 0.0, 1.73])
    frame_custom = s_custom[0]
    assert frame_custom.metadata["sensor_origin_provenance"] == "user_supplied"


# ---------------------------------------------------------------------------
# Timestamp Provenance Tracking Tests
# ---------------------------------------------------------------------------
def test_timestamp_provenance_across_adapters(tmp_path):
    # 1. Simulator: synthetic
    sim_source = SimulatorSource(n_steps=2, seed=1)
    sim_frame = sim_source[0]
    assert sim_frame.metadata["timestamp_provenance"] == "synthetic"
    assert sim_frame.metadata["sensor_origin_provenance"] == "simulated_model_mount"

    # 2. SemanticKITTI: derived or recorded
    p = str(tmp_path / "sim.npz")
    save_sequence(p, simulate_sequence(seed=42, n_frames=2))
    raw_frames, _ = sim_frames(p)
    kitti_root = str(tmp_path / "kitti_prov")
    write_kitti_mock(kitti_root, raw_frames)
    kitti_source = SemanticKITTISource(root=kitti_root, sequence="08")
    kitti_frame = kitti_source[0]
    assert "timestamp_provenance" in kitti_frame.metadata
    assert kitti_frame.metadata["sensor_origin_provenance"] == "dataset_calibrated_mount"

    # 3. FileSequenceSource: derived from sequence index
    seq_dir = tmp_path / "prov_seq"
    seq_dir.mkdir()
    np.ones((5, 4), dtype=np.float32).tofile(str(seq_dir / "00.bin"))
    seq_source = FileSequenceSource(seq_dir, pattern="*.bin", hz=10.0)
    seq_frame = seq_source[0]
    assert seq_frame.metadata["timestamp_provenance"] == "derived_from_sequence_index"
    assert seq_frame.metadata["sequence_index"] == 0


# ---------------------------------------------------------------------------
# Simulator Evaluation Truth Isolation Tests
# ---------------------------------------------------------------------------
def test_simulator_evaluation_truth_isolation():
    sim_source = SimulatorSource(n_steps=3, seed=99)
    assert hasattr(sim_source, "evaluation_truth")
    assert "cross_x" in sim_source.evaluation_truth

    frame = sim_source[0]
    # Truth is stored in dedicated evaluation_truth key, not polluting root metadata
    assert "evaluation_truth" in frame.metadata
    assert frame.has_annotations is True

    # Perception caller can safely strip annotations
    unannotated = frame.without_annotations()
    assert unannotated.has_annotations is False
    assert unannotated.label is None
    assert unannotated.moving is None
    np.testing.assert_array_equal(unannotated.pts, frame.pts)


# ---------------------------------------------------------------------------
# FileSequenceSource Edge Cases
# ---------------------------------------------------------------------------
def test_file_sequence_source_rejects_empty_or_missing_directory(tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(DataAdapterError, match="No files matching"):
        FileSequenceSource(empty_dir, pattern="*.bin")

    with pytest.raises(DataAdapterError, match="Directory not found"):
        FileSequenceSource(tmp_path / "non_existent_dir_12345", pattern="*.bin")

