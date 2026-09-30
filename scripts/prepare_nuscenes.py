"""Convert nuScenes(-mini) + lidarseg into cached FoveaMap frames.

    python scripts/prepare_nuscenes.py --dataroot /content/nuscenes --out cache/nuscenes
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap.nuscenes import build_cache  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "cache", "nuscenes"))
    ap.add_argument("--splits", default="mini_train,mini_val")
    a = ap.parse_args()
    build_cache(a.dataroot, a.out, a.version, a.splits.split(","))
