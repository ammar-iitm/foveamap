"""Profile the GPU pipeline with torch.profiler: where the time goes inside each stage.

    python scripts/profile_pipeline.py --dataset semantickitti --cache /content/cache/semantickitti --scene 08
    python scripts/profile_pipeline.py --dataset sim --device cpu          # simulated drive, for a quick look

Runs `--warmup` frames, then profiles `--frames` frames of features, network, binning, fusion, map
snapshot and object extraction. Prints the CPU and GPU time of each stage per frame, the operations
with the most GPU and CPU time, and how often per frame the host waited for the GPU (synchronisations,
device-to-host copies, and the ops that need one: nonzero, unique, item). The profiler adds overhead,
so read the stage times relative to each other, not as latency.
"""
import argparse
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap import features_torch, pipeline as P  # noqa: E402
from foveamap.objects import extract_objects, report  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STAGES = ("features", "network", "binning", "fusion", "snapshot", "objects")
WAITS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpy", "cudaMemcpyAsync",
         "aten::nonzero", "aten::unique", "aten::_unique2", "aten::item", "aten::_local_scalar_dense")


def labelled(name, fn):
    def run(*a, **k):
        with record_function(f"stage:{name}"):
            return fn(*a, **k)
    return run


def total(evt, kind):
    for attr in (f"{kind}_time_total", "device_time_total" if kind == "cuda" else None):
        if attr and hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="semantickitti", choices=["sim", "nuscenes", "semantickitti"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--seq", default=os.path.join(ROOT, "data", "demo.npz"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--frames", type=int, default=20)
    a = ap.parse_args()

    if a.dataset == "sim":
        from foveamap.frames import SIM_INFO, sim_frames
        frames, info = sim_frames(a.seq)[0], SIM_INFO
        ckpt = a.ckpt or os.path.join(ROOT, "checkpoints", "range_unet.pt")
    else:
        mod = __import__(f"foveamap.{a.dataset}", fromlist=["load_scene", "cached_info"])
        cache = a.cache or os.path.join(ROOT, "cache", a.dataset)
        scene = a.scene or ("08" if a.dataset == "semantickitti" else "scene-0103")
        frames, info = mod.load_scene(cache, scene), mod.cached_info(cache)
        ckpt = a.ckpt or os.path.join(ROOT, "checkpoints", f"range_unet_{a.dataset}.pt")
    frames = frames[:a.warmup + a.frames]

    pipe = P.FoveaMapPipeline(ckpt, info, grid="torch", device=a.device)
    features_torch.make_features = labelled("features", features_torch.make_features)
    P.predict = labelled("network", P.predict)
    for name, attr in (("binning", "bin_points"), ("fusion", "fuse_stats"), ("snapshot", "snapshot")):
        setattr(pipe.grid, attr, labelled(name, getattr(pipe.grid, attr)))
    objects = labelled("objects", lambda snap, dyn: report(extract_objects(pipe.grid, snap, dyn)))

    def frame(f):
        r = pipe.step(f)
        snap = pipe.grid.snapshot()
        objects(snap, P.to_host(r["dyn"]))

    for f in frames[:a.warmup]:
        frame(f)
    cuda = pipe.device.type == "cuda"
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=acts) as prof:
        for f in frames[a.warmup:]:
            frame(f)
        P._sync(pipe.device)
    n = len(frames) - a.warmup
    ka = prof.key_averages()
    print(f"\n{n} frames on {pipe.device} ({torch.cuda.get_device_name(pipe.device) if cuda else 'no CUDA'})")
    print(f"{'stage':10s} {'CPU ms/frame':>13s} {'GPU ms/frame':>13s}")
    by = {e.key: e for e in ka}
    for s in STAGES:
        e = by.get(f"stage:{s}")
        if e is not None:
            print(f"{s:10s} {total(e, 'cpu') / 1e3 / n:13.2f} {total(e, 'cuda') / 1e3 / n:13.2f}")
    print("\nhost waits per frame:")
    for w in WAITS:
        if w in by:
            print(f"  {w:28s} {by[w].count / n:6.1f}  ({total(by[w], 'cpu') / 1e3 / n:.2f} ms CPU)")
    for key in (["self_cuda_time_total", "self_device_time_total"] if cuda else []) + ["self_cpu_time_total"]:
        try:
            print(f"\nby {key}:\n" + ka.table(sort_by=key, row_limit=25, max_name_column_width=48))
        except (KeyError, AttributeError, RuntimeError, ValueError):
            continue
        if key != "self_cpu_time_total":
            break   # one GPU table is enough
