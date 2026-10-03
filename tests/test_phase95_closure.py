"""Phase 9.5 development-closure regression suite.

Covers the cross-phase fixes: runtime profiles, remote/archive hardening,
per-point timestamp contract + plumbing, license/packaging presence, and the
class-representation memory invariant (Option A: 16-byte cells kept).
"""
import os

import numpy as np
import pytest

from foveamap.core.config import RuntimeConfig
from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import ConfigurationError, NumericalConsistencyError, ContractError
from foveamap.data.preprocess import LiDARPreprocessor
from foveamap.grid import FoveatedGrid


def _frame(**over):
    n = 8
    kw = dict(
        pts=np.column_stack([np.linspace(1.0, 3.0, n), np.full(n, 0.0),
                             np.full(n, -1.7)]).astype(np.float32),
        intensity=np.ones(n, dtype=np.float32),
        ring=np.zeros(n, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        timestamp=0.0,
        frame_id="closure",
        source_id="test",
    )
    kw.update(over)
    return LiDARFrame(**kw)


# ------------------------------------------------------- runtime profiles
def test_cpu_and_gpu_profiles_explicit():
    cpu = RuntimeConfig.cpu_profile()
    assert (cpu.device, cpu.grid_engine, cpu.features_engine) == ("cpu", "numpy", "numpy")
    gpu = RuntimeConfig.gpu_profile()
    assert (gpu.device, gpu.grid_engine, gpu.features_engine) == ("auto", "torch", "torch")
    # Defaults stay development-safe (no silent CUDA requirement).
    default = RuntimeConfig()
    assert default.grid_engine == "numpy" and default.features_engine == "numpy"


# ------------------------------------------------------- remote/archive bounds
def test_remote_range_file_rejects_oversize():
    from foveamap.remote_zip import HTTPRangeFile
    import urllib.request

    class _Resp:
        status = 206
        headers = {"Content-Range": "bytes 0-0/999999999999"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    real = urllib.request.urlopen
    urllib.request.urlopen = lambda req, timeout=None: _Resp()
    try:
        with pytest.raises(OSError):
            HTTPRangeFile("https://example.com/huge.zip", max_size=10)
    finally:
        urllib.request.urlopen = real


def test_prepare_script_rejects_traversal_members(tmp_path):
    import sys
    import zipfile
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import prepare_semantickitti as prep

    evil = os.path.join(str(tmp_path), "evil.zip")
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../../escape_calib.txt", b"evil")
        z.writestr("dataset/sequences/00/calib.txt", b"ok")
    out = os.path.join(str(tmp_path), "root")
    with pytest.raises(ValueError):
        prep.extract(evil, out, lambda n: True)
    assert not os.path.exists(os.path.join(str(tmp_path), "escape_calib.txt"))
    # Legitimate nested members still extract.
    good = os.path.join(str(tmp_path), "good.zip")
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("dataset/sequences/00/calib.txt", b"ok")
    assert prep.extract(good, out, lambda n: True) == 1
    with open(os.path.join(out, "dataset", "sequences", "00", "calib.txt"), "rb") as fh:
        assert fh.read() == b"ok"


def test_fetch_scans_rejects_unsafe_sequence_ids():
    from foveamap import semantickitti as SK
    with pytest.raises(ValueError):
        SK.fetch_scans("/nonexistent-root-xyz", {"../evil": [0]}, url="https://example.com/x.zip")
    with pytest.raises(ValueError):
        SK.fetch_scans("/nonexistent-root-xyz", {"00": [-1]}, url="https://example.com/x.zip")


# ------------------------------------------------------- per-point timestamps
def test_time_offsets_absent_by_default():
    assert _frame().time_offsets is None


def test_time_offsets_valid_and_aligned_by_preprocess():
    n = 8
    offs = np.linspace(-0.1, 0.0, n, dtype=np.float32)
    f = _frame(time_offsets=offs)
    proc = LiDARPreprocessor()
    out = proc.process(f)
    assert out.time_offsets is not None
    assert len(out.time_offsets) == len(out.pts)
    # Filtering keeps alignment: dropped far points drop their offsets.
    assert np.all(np.abs(out.time_offsets) <= 0.1 + 1e-6)


def test_time_offsets_static_sensor_ok():
    f = _frame(time_offsets=np.zeros(8, dtype=np.float32))
    assert f.time_offsets is not None


def test_time_offsets_malformed_rejected():
    with pytest.raises((ContractError, NumericalConsistencyError)):
        _frame(time_offsets=np.full(8, np.nan, dtype=np.float32))
    with pytest.raises((ContractError, NumericalConsistencyError)):
        _frame(time_offsets=np.zeros(7, dtype=np.float32))
    with pytest.raises((ContractError, NumericalConsistencyError)):
        _frame(time_offsets=np.zeros(8, dtype=np.int32))


def test_time_offsets_out_of_range_rejected():
    with pytest.raises(NumericalConsistencyError):
        _frame(time_offsets=np.full(8, 3600.0, dtype=np.float32))


def test_time_offsets_legacy_roundtrip():
    offs = np.linspace(-0.05, 0.05, 8, dtype=np.float32)
    f = _frame(time_offsets=offs)
    back = LiDARFrame.from_legacy_dict(f.to_legacy_dict())
    np.testing.assert_allclose(back.time_offsets, offs, rtol=1e-6)


def test_ros_time_channel_flows_to_frame():
    from foveamap_ros.pointcloud import (
        FLOAT32, RosHeader, RosPointCloud2, RosPointField, RosStamp, cloud_to_arrays,
    )
    n = 4
    rec = np.zeros(n, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32),
                             ("time", np.float32)])
    rec["x"], rec["y"], rec["z"] = 1.0, 0.0, -1.7
    rec["time"] = np.array([-0.08, -0.02, 0.0, 0.05], dtype=np.float32)
    msg = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
                RosPointField("z", 8, FLOAT32), RosPointField("time", 12, FLOAT32)],
        height=1, width=n, point_step=16, row_step=16 * n, data=rec.tobytes(),
        header=RosHeader(RosStamp(77, 0), "base_link"))
    out = cloud_to_arrays(msg)
    np.testing.assert_allclose(out["time_offsets"], rec["time"], rtol=1e-6)
    assert out["time_provenance"] == "ros_time_channel"


# ------------------------------------------------------- packaging / memory
def test_license_present_and_package_coherent():
    root = os.path.join(os.path.dirname(__file__), "..")
    with open(os.path.join(root, "LICENSE")) as fh:
        text = fh.read()
    assert "MIT License" in text and "FoveaMap Team" in text
    import foveamap
    assert foveamap.__version__ == "0.1.0"


def test_cell_representation_stays_16_bytes():
    # Option A (ARCHITECTURE_DECISIONS D1): compressed dominant+secondary
    # representation; persistent 16 B/cell must hold on the required profile.
    g = FoveatedGrid("spec")
    per_cell = sum(getattr(g.state[0], f).nbytes for f in g.state[0].FIELDS) // (g.state[0].n ** 2)
    assert per_cell == 16
    rep = g.memory_report()
    assert rep["allocated_bytes"] == 5120000
    assert rep["under_8mb_target"] is True
