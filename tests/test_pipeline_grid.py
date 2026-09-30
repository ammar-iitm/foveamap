"""The pipeline gives the same map with either grid engine."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foveamap.frames import SIM_INFO  # noqa: E402
from foveamap.pipeline import FoveaMapPipeline, run_benchmark, to_host  # noqa: E402
from test_grid_torch import _state_mismatch  # noqa: E402

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "range_unet.pt")


def test_pipeline_engines_agree(drive):
    frames, _ = drive
    ref = FoveaMapPipeline(CKPT, SIM_INFO, device="cpu", grid="numpy")
    tor = FoveaMapPipeline(CKPT, SIM_INFO, device="cpu", grid="torch", features="numpy")
    for fr in frames:
        a, b = ref.step(fr), tor.step(fr)
        np.testing.assert_array_equal(a["cls_pts"], to_host(b["cls_pts"]))
        np.testing.assert_array_equal(a["moving_pts"], to_host(b["moving_pts"]))
        for sa, sb in zip(a["stats"], to_host(b["stats"])):
            np.testing.assert_array_equal(sa["key"], sb["key"])
            np.testing.assert_array_equal(sa["n_pts"], sb["n_pts"])
        for ra, rb in zip(ref.grid.snapshot(), tor.grid.snapshot()):
            frac, _ = _state_mismatch(ra, rb, 2e-3)
            assert frac <= 1e-4


def test_benchmark_runs_with_torch_grid(drive, tmp_path):
    frames, truth = drive
    s, per_frame = run_benchmark(frames, SIM_INFO, CKPT, str(tmp_path), truth=truth, n_uniform=1,
                                 device="cpu", grid="torch")
    assert s["grid_engine"] == "torch" and s["points_lost"] == 0 and s["nesting_ok"]
    assert s["export_mode"] == "async"
    assert all(f["png_bytes"] > 0 and f["export_ms"] > 0 for f in per_frame)
    assert len(os.listdir(tmp_path / "frames")) == len(frames)
    assert "export" not in per_frame[0]["timing_ms"]          # background export is not latency


def test_benchmark_export_modes(drive, tmp_path):
    frames, truth = drive
    out_after = tmp_path / "after"
    s, per_frame = run_benchmark(frames, SIM_INFO, CKPT, str(out_after), truth=truth, n_uniform=1,
                                 device="cpu", grid="torch", export="after")
    assert s["export_mode"] == "after"
    assert all(f["png_bytes"] > 0 and f["export_ms"] > 0 for f in per_frame)
    assert len(os.listdir(out_after / "frames")) == len(frames)
    assert "export" not in per_frame[0]["timing_ms"]


def test_torch_features_in_pipeline(drive):
    frames, _ = drive
    ref = FoveaMapPipeline(CKPT, SIM_INFO, device="cpu", grid="numpy")
    tor = FoveaMapPipeline(CKPT, SIM_INFO, device="cpu", grid="torch")
    assert tor.features == "torch"
    for fr in frames:
        a, b = ref.step(fr), tor.step(fr)
        # features differ only at rare tie pixels, so a handful of points may change class
        assert (a["cls_pts"] != to_host(b["cls_pts"])).mean() < 1e-3
    for ra, rb in zip(ref.grid.snapshot(), tor.grid.snapshot()):
        assert (ra.cls != rb.cls).mean() < 1e-3
