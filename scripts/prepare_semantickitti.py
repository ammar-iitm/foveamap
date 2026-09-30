"""Fetch the SemanticKITTI data FoveaMap uses and cache it as frames.

    python scripts/prepare_semantickitti.py --root /content/kitti --out cache/semantickitti --stride 10

Downloads the SemanticKITTI labels (179 MB, includes poses) and the KITTI
calibration (0.6 MB), then pulls only the needed scans out of the ~85 GB KITTI
velodyne zip with HTTP range requests: every `stride`-th scan of each sequence
plus the two scans before it (the motion cue). Stride 10 is ~5,700 training
and ~1,200 validation scans, about 14 GB.

KITTI is CC BY-NC-SA 3.0 and SemanticKITTI CC BY-NC-SA 4.0 (non-commercial).
"""
import argparse
import os
import sys
import time
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap import semantickitti as SK  # noqa: E402


def download(url, dst):
    if os.path.exists(dst):
        return dst
    print(f"downloading {url}", flush=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    return dst


def extract(zip_path, root, keep):
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if keep(n) and not n.endswith("/")
                   and not os.path.exists(os.path.join(root, n))]
        for n in members:
            z.extract(n, root)
    return len(members)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="where the KITTI files go")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "cache", "semantickitti"))
    ap.add_argument("--stride", type=int, default=10, help="use every n-th scan as a frame")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    splits = a.splits.split(",")
    seqs = [s for sp in splits for s in SK.SPLITS[sp]]
    os.makedirs(a.root, exist_ok=True)
    t0 = time.time()

    lz = download(SK.LABELS_URL, os.path.join(a.root, "data_odometry_labels.zip"))
    cz = download(SK.CALIB_URL, os.path.join(a.root, "data_odometry_calib.zip"))
    extract(cz, a.root, lambda n: n.endswith("calib.txt") and any(f"/{s}/" in n for s in seqs))
    extract(lz, a.root, lambda n: n.endswith("poses.txt") and any(f"/{s}/" in n for s in seqs))

    ds = SK.SemanticKITTI(a.root)
    frames, wanted = {}, {}
    for s in seqs:
        n = len(SK.read_poses(os.path.join(ds.seq_dir(s), "poses.txt")))
        frames[s] = SK.selected_scans(n, a.stride)
        wanted[s] = SK.needed_scans(frames[s])
    labels = {f"dataset/sequences/{s}/labels/{i:06d}.label" for s in seqs for i in frames[s]}
    print(f"extracted {extract(lz, a.root, lambda n: n in labels)} label files", flush=True)
    SK.fetch_scans(a.root, wanted, workers=a.workers)
    print(f"data ready in {time.time() - t0:.0f}s; building the frame cache", flush=True)
    SK.build_cache(a.root, a.out, splits, stride=a.stride)
    print(f"done in {time.time() - t0:.0f}s")
