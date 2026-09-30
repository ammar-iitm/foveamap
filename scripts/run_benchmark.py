"""Run a drive through the full pipeline, print the benchmark, export dashboard data.

    python scripts/run_benchmark.py                                   # simulated demo drive
    python scripts/run_benchmark.py --grid torch                      # grid engine in PyTorch (GPU if present)
    python scripts/run_benchmark.py --dataset nuscenes --scene scene-0103 \
        --ckpt checkpoints/range_unet_nuscenes.pt --out dashboard/data
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
    ap.add_argument("--dataset", default="sim", choices=["sim", "nuscenes"])
    ap.add_argument("--seq", default=os.path.join(ROOT, "data", "demo.npz"), help="simulated sequence (.npz)")
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache", "nuscenes"))
    ap.add_argument("--scene", default="scene-0103", help="nuScenes scene name")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "dashboard", "data"))
    ap.add_argument("--profile", default="spec")
    ap.add_argument("--device", default=None)
    ap.add_argument("--grid", default="numpy", choices=["numpy", "torch"],
                    help="grid engine: NumPy on the CPU, or PyTorch on the model's device")
    ap.add_argument("--features", default=None, choices=["numpy", "torch"],
                    help="range-image features engine (default: same as --grid)")
    args = ap.parse_args()

    if args.dataset == "sim":
        frames, truth = sim_frames(args.seq)
        info = SIM_INFO
        ckpt = args.ckpt or os.path.join(ROOT, "checkpoints", "range_unet.pt")
    else:
        from foveamap.nuscenes import load_scene, cached_info
        frames, truth = load_scene(args.cache, args.scene), None
        info = cached_info(args.cache)
        info.source = f"nuScenes {args.scene} · {info.n_rows}-beam Lidar · labelled keyframes at 2 Hz"
        ckpt = args.ckpt or os.path.join(ROOT, "checkpoints", "range_unet_nuscenes.pt")

    summary, _ = run_benchmark(frames, info, ckpt, args.out, truth=truth, profile=args.profile, device=args.device,
                               grid=args.grid, features=args.features)
    s = {k: v for k, v in summary.items() if not k.endswith("_index") and "by_class" not in k}
    print(json.dumps(s, indent=2))
