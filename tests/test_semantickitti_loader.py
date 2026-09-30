"""SemanticKITTI loader on a simulated drive written in the SemanticKITTI layout."""
import io
import json
import os
import zipfile

import numpy as np
import pytest

from foveamap import semantickitti as SK
from foveamap.frames import make_features, prev_in_ego, world_points
from foveamap.sim import FOV_DOWN, FOV_UP
from mock_semantickitti import write_mock


@pytest.fixture(scope="module")
def kitti(drive, tmp_path_factory):
    frames, _ = drive
    root = str(tmp_path_factory.mktemp("kitti"))
    T0 = write_mock(root, frames)
    return root, frames, T0


def test_frames_match_simulator(kitti):
    root, sim, T0 = kitti
    ds = SK.SemanticKITTI(root)
    poses = ds.poses_ego("08")
    info = SK.kitti_info()
    for i, s in enumerate(sim):
        f = ds.frame("08", i, poses, info, remove_close=0.0)
        order = np.random.default_rng(i).permutation(len(s["pts"]))
        np.testing.assert_allclose(f["pts"], s["pts"][order], atol=1e-5)
        np.testing.assert_array_equal(f["label"], s["label"][order])
        np.testing.assert_array_equal(f["moving"], s["moving"][order])
        # world points: KITTI's world is scan 0's ego frame
        sim_world = world_points(dict(s, pts=s["pts"][order]))
        expect = (np.linalg.inv(T0) @ np.c_[sim_world, np.ones(len(sim_world))].T).T[:, :3]
        np.testing.assert_allclose(world_points(f), expect, atol=2e-4)
        assert f["ring"].min() >= 0 and f["ring"].max() < info.n_rows
        assert [p is None for p in f["prev"]] == [i < 1, i < 2]
        if i >= 1:                                          # previous scan lands in the same world frame
            s_prev = sim[i - 1]
            o = np.random.default_rng(i - 1).permutation(len(s_prev["pts"]))
            pw = world_points(dict(s_prev, pts=s_prev["pts"][o]))
            e = (np.linalg.inv(T0) @ np.c_[pw, np.ones(len(pw))].T).T[:, :3]
            np.testing.assert_allclose(f["prev"][0][0], e, atol=2e-4)
        feats, idx, _, _ = make_features(f, info, prev_in_ego(f))
        assert feats.shape == (8, 64, 1024) and (idx >= 0).mean() > 0.3


def test_rows_from_elevation():
    elev = np.array([FOV_UP - 1e-6, FOV_DOWN + 1e-6, 10.0, -40.0, (FOV_UP + FOV_DOWN) / 2])
    pts = np.c_[np.ones(5), np.zeros(5), np.tan(np.radians(elev))]
    np.testing.assert_array_equal(SK.rows_from_elevation(pts), [0, 63, 0, 63, 32])


def test_label_map_covers_kitti_ids():
    kitti_ids = [0, 1, 10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 40, 44, 48, 49, 50, 51, 52, 60, 70, 71,
                 72, 80, 81, 99, 252, 253, 254, 255, 256, 257, 258, 259]
    mapped = SK.LUT[kitti_ids]
    assert list(mapped[:2]) == [-1, -1] and (mapped[2:] >= 0).all()
    assert SK.IS_MOVING[[252, 254]].all() and not SK.IS_MOVING[[10, 30, 40]].any()


def test_cache_and_needed_scans(kitti, tmp_path):
    root, sim, _ = kitti
    # the mock only has sequence 08 (val)
    idx = SK.build_cache(root, str(tmp_path), splits=("val",), stride=2)
    assert idx["val"] == ["08"] and idx["counts"]["08"] == len(range(0, len(sim), 2))
    got = list(SK.iter_cached(str(tmp_path), "val"))
    assert len(got) == SK.count_cached(str(tmp_path), "val") and [f["meta"]["scan"] for f in got] == [0, 2]
    assert SK.cached_info(str(tmp_path)).n_rows == 64
    assert SK.needed_scans(SK.selected_scans(25, 10)) == [0, 8, 9, 10, 18, 19, 20]


def test_fetch_scans_from_zip(kitti, tmp_path, monkeypatch):
    """fetch_scans pulls only the wanted members; a local zip stands in for the remote one."""
    root, sim, _ = kitti
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i in range(len(sim)):
            name = f"dataset/sequences/08/velodyne/{i:06d}.bin"
            z.write(os.path.join(root, name), name)
    import foveamap.remote_zip as rz
    monkeypatch.setattr(rz, "open_remote_zip", lambda url: zipfile.ZipFile(io.BytesIO(buf.getvalue())))
    out = str(tmp_path / "fetched")
    assert SK.fetch_scans(out, {"08": [0, 2]}, url="unused", workers=2, log=lambda *a: None) == 2
    got = sorted(os.listdir(os.path.join(out, "dataset", "sequences", "08", "velodyne")))
    assert got == ["000000.bin", "000002.bin"]
    assert SK.fetch_scans(out, {"08": [0, 2]}, url="unused", workers=2, log=lambda *a: None) == 0   # already there

    # not enough room (e.g. a nearly full Google Drive): stop before downloading anything
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10 ** 12, 10 ** 12, 10 ** 6))
    small = str(tmp_path / "full")
    with pytest.raises(SK.NotEnoughSpace, match="larger stride"):
        SK.fetch_scans(small, {"08": [1, 3]}, url="unused", workers=2, log=lambda *a: None, reserve=0)
    assert not os.path.exists(os.path.join(small, "dataset"))



def test_parallel_cache_matches_serial(kitti, tmp_path):
    root, _, _ = kitti
    a, b = str(tmp_path / "serial"), str(tmp_path / "parallel")
    SK.build_cache(root, a, splits=("val",), stride=1, workers=1)
    SK.build_cache(root, b, splits=("val",), stride=1, workers=2)
    assert json.load(open(os.path.join(a, "index.json"))) == json.load(open(os.path.join(b, "index.json")))
    fa, fb = list(SK.iter_cached(a, "val")), list(SK.iter_cached(b, "val"))
    assert len(fa) == len(fb) > 0
    for x, y in zip(fa, fb):
        for k in ("pts", "label", "moving", "ring", "pose"):
            np.testing.assert_array_equal(x[k], y[k])
    assert not [f for f in os.listdir(b) if f.endswith(".part")]


def test_fetch_logs_a_summary(kitti, tmp_path, monkeypatch):
    root, sim, _ = kitti
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i in range(len(sim)):
            name = f"dataset/sequences/08/velodyne/{i:06d}.bin"
            z.write(os.path.join(root, name), name)
    import foveamap.remote_zip as rz
    monkeypatch.setattr(rz, "open_remote_zip", lambda url: zipfile.ZipFile(io.BytesIO(buf.getvalue())))
    lines = []
    SK.fetch_scans(str(tmp_path / "f"), {"08": list(range(len(sim)))}, url="unused", workers=2,
                   log=lines.append, progress_every=0.001)
    assert lines[0].startswith(f"fetching {len(sim)} scans") and "GB to download" in lines[1]
    assert lines[-1].startswith(f"fetched {len(sim)} scans (") and "GB) in" in lines[-1]
