"""The live server's worker: frames through the pipeline, state and tiles for the dashboard."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from live_server import Live  # noqa: E402
from foveamap.frames import SIM_INFO  # noqa: E402
from foveamap.pipeline import FoveaMapPipeline  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoints", "range_unet.pt")


def test_live_state_and_tiles(drive):
    frames, _ = drive
    pipe = FoveaMapPipeline(CKPT, SIM_INFO, device="cpu", grid="torch")
    live = Live(pipe, frames, SIM_INFO, rate=10.0, loop=False)
    assert live.state()["frame"] is None and live.tiles() == (None, b"")
    live.warm_up()
    assert pipe.grid.origins is None                       # warm-up leaves an empty map
    for fr in frames:
        live.step(fr)
    st = live.state()
    s, f = st["summary"], st["frame"]
    assert s["live"] and s["frames_processed"] == len(frames) and f["t"] == len(frames) - 1
    assert f["binned"] == f["in_window"] > 0                # no point lost, as in the benchmark
    assert {"preprocess", "inference", "projection", "fusion", "publish", "objects"} <= set(f["timing_ms"])
    assert s["fps"] > 0 and s["latency_ms"]["p95"] >= s["latency_ms"]["p50"]
    assert "sparse_voxel_5cm_one_sweep" in s["memory_bytes"]
    t, body = live.tiles()
    assert t == f["t"] and len(body) == sum(tr["n"] ** 2 * 3 for tr in s["tiers"])
