"""prepare_semantickitti.py, the training data path and the benchmark on a mock SemanticKITTI."""
import os
import zipfile

import pytest

from foveamap import semantickitti as SK
from mock_semantickitti import write_mock


@pytest.fixture(scope="module")
def kitti(drive, tmp_path_factory):
    frames, _ = drive
    root = str(tmp_path_factory.mktemp("kitti"))
    T0 = write_mock(root, frames)
    return root, frames, T0


def _zip(path, root, names):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            z.write(os.path.join(root, n), n)


def test_prepare_script_train_arrays_and_benchmark(kitti, tmp_path, monkeypatch):
    """prepare_semantickitti.py end to end against local zips in the KITTI layout, then the
    training data path (arrays only, no training) and a 2-frame benchmark on its cache."""
    import runpy
    import sys
    import urllib.request
    import foveamap.remote_zip as rz
    root, sim, _ = kitti
    seq = "dataset/sequences/08"
    n = len(sim)
    zips = {SK.LABELS_URL: tmp_path / "labels.zip", SK.CALIB_URL: tmp_path / "calib.zip",
            SK.VELODYNE_URL: tmp_path / "velodyne.zip"}
    _zip(zips[SK.LABELS_URL], root, [f"{seq}/poses.txt"] + [f"{seq}/labels/{i:06d}.label" for i in range(n)])
    _zip(zips[SK.CALIB_URL], root, [f"{seq}/calib.txt"])
    _zip(zips[SK.VELODYNE_URL], root, [f"{seq}/velodyne/{i:06d}.bin" for i in range(n)])
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda url, dst: __import__("shutil").copy(zips[url], dst))
    monkeypatch.setattr(rz, "open_remote_zip", lambda url: zipfile.ZipFile(zips[url]))
    data, cache = tmp_path / "kitti", tmp_path / "cache"
    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "prepare_semantickitti.py")
    monkeypatch.setattr(sys, "argv", [script, "--root", str(data), "--out", str(cache), "--splits", "val",
                                      "--stride", "3", "--workers", "2"])
    runpy.run_path(script, run_name="__main__")
    # stride 3 over 4 scans -> frames 0, 3; scan 3 needs 1 and 2 for its motion cue; scan 0 needs nothing
    vel = sorted(os.listdir(data / "dataset" / "sequences" / "08" / "velodyne"))
    assert vel == ["000000.bin", "000001.bin", "000002.bin", "000003.bin"]
    labels = sorted(os.listdir(data / "dataset" / "sequences" / "08" / "labels"))
    assert labels == ["000000.label", "000003.label"]          # labels only for frames
    frames = list(SK.iter_cached(str(cache), "val"))
    assert [f["meta"]["scan"] for f in frames] == [0, 3] and all(p is not None for p in frames[1]["prev"])

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from train import load_frames
    from foveamap.frames import frames_to_training_arrays
    args = __import__("argparse").Namespace(dataset="semantickitti", cache=str(cache), train_split="train",
                                            val_split="val")
    it, info, count = load_frames(args, "val")
    X, Y, M, R = frames_to_training_arrays(it, info, count)
    assert X.shape == (2, 8, 64, 1024) and (Y >= 0).any() and info.name == "semantickitti"

    bench = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_benchmark.py")
    monkeypatch.setattr(sys, "argv", [bench, "--dataset", "semantickitti", "--cache", str(cache), "--max-frames", "2",
                                      "--ckpt", os.path.join(os.path.dirname(__file__), "..", "checkpoints", "range_unet.pt"),
                                      "--device", "cpu", "--grid", "torch", "--export", "none",
                                      "--out", str(tmp_path / "bench")])
    runpy.run_path(bench, run_name="__main__")
