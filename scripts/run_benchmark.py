"""Run a drive through the full pipeline, print the benchmark, export dashboard data.

    python scripts/run_benchmark.py                                   # simulated demo drive
    python scripts/run_benchmark.py --grid torch                      # grid engine in PyTorch (GPU if present)
    python scripts/run_benchmark.py --dataset nuscenes --scene scene-0103 \
        --ckpt checkpoints/range_unet_nuscenes.pt --out dashboard/data
    python scripts/run_benchmark.py --dataset semantickitti --scene 08 --max-frames 100 --grid torch
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap.pipeline import run_benchmark  # noqa: E402
from foveamap.frames import SIM_INFO, sim_frames  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim", choices=["sim", "nuscenes", "semantickitti"])
    ap.add_argument("--seq", default=os.path.join(ROOT, "data", "demo.npz"), help="simulated sequence (.npz)")
    ap.add_argument("--cache", default=None, help="frame cache (default: cache/<dataset>)")
    ap.add_argument("--scene", default=None,
                    help="nuScenes scene (default scene-0103) or SemanticKITTI sequence (default 08)")
    ap.add_argument("--max-frames", type=int, default=None, help="use only the first n frames")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "dashboard", "data"))
    ap.add_argument("--profile", default="spec")
    ap.add_argument("--device", default=None)
    ap.add_argument("--grid", default="numpy", choices=["numpy", "torch"],
                    help="grid engine: NumPy on the CPU, or PyTorch on the model's device")
    ap.add_argument("--features", default=None, choices=["numpy", "torch"],
                    help="range-image features engine (default: same as --grid)")
    ap.add_argument("--export", default="after", choices=["after", "async", "none"],
                    help="dashboard PNG export: 'after' (encode after the timed loop, default), "
                         "'async' (background thread during the run), or 'none'")
    args = ap.parse_args()

    if args.dataset == "sim":
        frames, truth = sim_frames(args.seq)
        info = SIM_INFO
        ckpt = args.ckpt or os.path.join(ROOT, "checkpoints", "range_unet.pt")
    elif args.dataset == "nuscenes":
        from foveamap.nuscenes import load_scene, cached_info
        cache = args.cache or os.path.join(ROOT, "cache", "nuscenes")
        args.scene = args.scene or "scene-0103"
        frames, truth = load_scene(cache, args.scene), None
        info = cached_info(cache)
        info.source = f"nuScenes {args.scene} · {info.n_rows}-beam Lidar · labelled keyframes at 2 Hz"
        ckpt = args.ckpt or os.path.join(ROOT, "checkpoints", "range_unet_nuscenes.pt")
    else:
        from foveamap.semantickitti import load_scene, cached_info
        cache = args.cache or os.path.join(ROOT, "cache", "semantickitti")
        args.scene = args.scene or "08"
        frames, truth = load_scene(cache, args.scene), None
        info = cached_info(cache)
        with open(os.path.join(cache, "index.json")) as fh:
            stride = json.load(fh)["stride"]
        info.source = f"SemanticKITTI sequence {args.scene} · 64-beam Lidar · every {stride}th scan"
        info.hz = 10.0 / stride
        ckpt = args.ckpt or os.path.join(ROOT, "checkpoints", "range_unet_semantickitti.pt")
    if args.max_frames:
        frames = frames[:args.max_frames]

    summary, _ = run_benchmark(frames, info, ckpt, args.out, truth=truth, profile=args.profile, device=args.device,
                               grid=args.grid, features=args.features, export=args.export)
    s = {k: v for k, v in summary.items() if not k.endswith("_index") and "by_class" not in k}
    print(json.dumps(s, indent=2))
