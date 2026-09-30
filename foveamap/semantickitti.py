"""SemanticKITTI loader -> FoveaMap Frames, and a fetcher for the scans it needs.

Expected layout (the KITTI odometry + SemanticKITTI zips, extracted):
    <root>/dataset/sequences/<SS>/velodyne/<NNNNNN>.bin   float32 x, y, z, remission
    <root>/dataset/sequences/<SS>/labels/<NNNNNN>.label   uint32, low 16 bits = semantic id
    <root>/dataset/sequences/<SS>/poses.txt              3x4 cam0 poses (in the labels zip)
    <root>/dataset/sequences/<SS>/calib.txt              Tr: velodyne -> cam0 (in the calib zip)

The Lidar is an HDL-64E 1.73 m above the ground, the sensor the simulator
models, and SemanticKITTI's classes map onto all 9 FoveaMap classes. The .bin
files carry no laser id, so a point's range-image row comes from its
elevation angle over the simulator's field of view (the RangeNet++ projection).
Scans are 10 Hz, so the motion cue uses the previous two scans (0.1 s, 0.2 s).
"""
from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .frames import DatasetInfo, transform
from .sim import (CLASSES, ROAD, SIDEWALK, PARKING, TERRAIN, VEGETATION, BUILDING, POLE,
                  VEHICLE, PERSON, N_BEAMS, FOV_UP, FOV_DOWN, SENSOR_H)

VELODYNE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
LABELS_URL = "http://www.semantic-kitti.org/assets/data_odometry_labels.zip"
CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip"

SPLITS = {"train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"], "val": ["08"]}

# SemanticKITTI raw semantic id -> FoveaMap class (semantic-kitti.yaml ids; 0 unlabeled, 1 outlier -> ignore)
RAW_TO_CLASS = {
    40: ROAD, 60: ROAD,                                  # road, lane-marking
    44: PARKING,
    48: SIDEWALK, 49: SIDEWALK,                          # sidewalk, other-ground
    72: TERRAIN,
    70: VEGETATION, 71: VEGETATION,                      # vegetation, trunk
    50: BUILDING, 51: BUILDING, 52: BUILDING, 99: BUILDING,   # building, fence, other-structure, other-object
    80: POLE, 81: POLE,                                  # pole, traffic-sign
    10: VEHICLE, 11: VEHICLE, 13: VEHICLE, 15: VEHICLE, 16: VEHICLE, 18: VEHICLE, 20: VEHICLE,
    252: VEHICLE, 256: VEHICLE, 257: VEHICLE, 258: VEHICLE, 259: VEHICLE,     # moving vehicles
    30: PERSON, 31: PERSON, 32: PERSON,                  # person, bicyclist, motorcyclist
    253: PERSON, 254: PERSON, 255: PERSON,               # moving bicyclist, person, motorcyclist
}
MOVING_RAW = (252, 253, 254, 255, 256, 257, 258, 259)

LUT = np.full(1 << 16, -1, np.int8)
for _raw, _cls in RAW_TO_CLASS.items():
    LUT[_raw] = _cls
IS_MOVING = np.zeros(1 << 16, bool)
IS_MOVING[list(MOVING_RAW)] = True

S_UP = np.eye(4)
S_UP[2, 3] = SENSOR_H           # ego frame = velodyne frame moved down to the ground


def kitti_info(n_cols=1024):
    return DatasetInfo("semantickitti", N_BEAMS, n_cols, class_names=list(CLASSES),
                       active=np.ones(len(CLASSES), bool),
                       source="SemanticKITTI · 64-beam HDL-64E · 10 Hz", hz=10.0)


def rows_from_elevation(pts_sensor, n_rows=N_BEAMS, fov_up=FOV_UP, fov_down=FOV_DOWN):
    """Range-image row (0 = top) from elevation angle, clipped to the image."""
    elev = np.degrees(np.arctan2(pts_sensor[:, 2], np.hypot(pts_sensor[:, 0], pts_sensor[:, 1])))
    r = np.floor((fov_up - elev) / (fov_up - fov_down) * n_rows)
    return np.clip(r, 0, n_rows - 1).astype(np.int16)


def read_scan(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_labels(path):
    return np.fromfile(path, dtype=np.uint32) & 0xFFFF


def read_calib_tr(path):
    """Tr (velodyne -> cam0) as a 4x4 matrix."""
    with open(path) as fh:
        for line in fh:
            key, _, vals = line.partition(":")
            if key.strip() == "Tr":
                T = np.eye(4)
                T[:3, :4] = np.array(vals.split(), np.float64).reshape(3, 4)
                return T
    raise ValueError(f"no Tr line in {path}")


def read_poses(path):
    """(N, 4, 4) cam0 poses, one per scan."""
    raw = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    P = np.tile(np.eye(4), (len(raw), 1, 1))
    P[:, :3, :4] = raw
    return P


class SemanticKITTI:
    def __init__(self, root):
        self.root = root

    def seq_dir(self, seq):
        return os.path.join(self.root, "dataset", "sequences", seq)

    def scan_path(self, seq, i):
        return os.path.join(self.seq_dir(seq), "velodyne", f"{i:06d}.bin")

    def label_path(self, seq, i):
        return os.path.join(self.seq_dir(seq), "labels", f"{i:06d}.label")

    def poses_ego(self, seq):
        """(N, 4, 4) ego -> world. World = scan 0's ego frame, z up, ground near z = 0."""
        Tr = read_calib_tr(os.path.join(self.seq_dir(seq), "calib.txt"))
        P = read_poses(os.path.join(self.seq_dir(seq), "poses.txt"))
        T_wv = np.linalg.inv(Tr) @ P @ Tr                 # velodyne -> world (velodyne axes of scan 0)
        return S_UP @ T_wv @ np.linalg.inv(S_UP)

    def frame(self, seq, i, poses, info, prev_steps=(1, 2), remove_close=1.0):
        raw = read_scan(self.scan_path(seq, i))
        sem = read_labels(self.label_path(seq, i))
        assert len(sem) == len(raw), f"label/point count mismatch for {seq}/{i:06d}"
        keep = np.hypot(raw[:, 0], raw[:, 1]) > remove_close
        raw, sem = raw[keep], sem[keep]
        label = LUT[sem]
        prev = []
        for k in prev_steps:
            j = i - k
            if j < 0 or not os.path.exists(self.scan_path(seq, j)):
                prev.append(None)
                continue
            r = read_scan(self.scan_path(seq, j))
            r = r[np.hypot(r[:, 0], r[:, 1]) > remove_close]
            pw = transform(poses[j] @ S_UP, r[:, :3].astype(np.float64))      # sensor -> ego -> world
            prev.append((pw.astype(np.float32), rows_from_elevation(r[:, :3], info.n_rows)))
        return dict(
            pts=(raw[:, :3] + np.array([0, 0, SENSOR_H], np.float32)).astype(np.float32),
            inten=np.clip(raw[:, 3], 0, 1).astype(np.float32),
            ring=rows_from_elevation(raw[:, :3], info.n_rows),
            label=label, moving=IS_MOVING[sem] & np.isin(label, (VEHICLE, PERSON)),
            pose=poses[i], sensor=np.array([0, 0, SENSOR_H], np.float32), prev=prev,
            meta=dict(sequence=seq, scan=i),
        )


def selected_scans(n_scans, stride, start=0):
    """Scans used as frames: every `stride`-th, starting at `start`."""
    return list(range(start, n_scans, stride))


def needed_scans(frames, prev_steps=(1, 2)):
    """Scans to fetch: each frame plus the earlier scans its motion cue uses."""
    need = set()
    for i in frames:
        need.add(i)
        need.update(i - k for k in prev_steps if i - k >= 0)
    return sorted(need)


def _cache_sequence(root, out_dir, seq, stride, info):
    """Build one sequence's pickle (unless present). Returns its frame count."""
    ds = SemanticKITTI(root)
    path = os.path.join(out_dir, f"{seq}.pkl")
    ids = selected_scans(len(read_poses(os.path.join(ds.seq_dir(seq), "poses.txt"))), stride)
    if not os.path.exists(path):
        poses = ds.poses_ego(seq)
        frames = [ds.frame(seq, i, poses, info) for i in ids]
        with open(path + ".part", "wb") as fh:
            pickle.dump(frames, fh, protocol=4)
        os.replace(path + ".part", path)
    return len(ids)


def build_cache(root, out_dir, splits=("train", "val"), stride=10, info=None, workers=1):
    """Convert the selected scans to cached Frame lists (one pickle per sequence),
    `workers` sequences at a time (processes; each holds one sequence in memory)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    info = info or kitti_info()
    os.makedirs(out_dir, exist_ok=True)
    index = {"n_rows": info.n_rows, "n_cols": info.n_cols, "stride": stride, "counts": {}}
    seqs = [(split, seq) for split in splits for seq in SPLITS[split]]
    for split in splits:
        index[split] = list(SPLITS[split])
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            jobs = {ex.submit(_cache_sequence, root, out_dir, seq, stride, info): (split, seq) for split, seq in seqs}
            for job in as_completed(jobs):
                split, seq = jobs[job]
                index["counts"][seq] = job.result()
                print(f"  {split}: sequence {seq} ({index['counts'][seq]} frames)", flush=True)
    else:
        for split, seq in seqs:
            index["counts"][seq] = _cache_sequence(root, out_dir, seq, stride, info)
            print(f"  {split}: sequence {seq} ({index['counts'][seq]} frames)", flush=True)
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump(index, fh)
    return index


def cached_info(out_dir):
    with open(os.path.join(out_dir, "index.json")) as fh:
        return kitti_info(json.load(fh).get("n_cols", 1024))


def iter_cached(out_dir, split):
    """Frames of a split, one sequence in memory at a time."""
    with open(os.path.join(out_dir, "index.json")) as fh:
        index = json.load(fh)
    for seq in index[split]:
        with open(os.path.join(out_dir, f"{seq}.pkl"), "rb") as fh:
            yield from pickle.load(fh)


def count_cached(out_dir, split):
    with open(os.path.join(out_dir, "index.json")) as fh:
        index = json.load(fh)
    return sum(index["counts"][s] for s in index[split])


def load_scene(out_dir, seq):
    with open(os.path.join(out_dir, f"{seq}.pkl"), "rb") as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------------
# Fetching only the scans that are used (the velodyne zip is ~85 GB)
# ----------------------------------------------------------------------------
class NotEnoughSpace(OSError):
    pass


def fetch_scans(root, wanted, url=VELODYNE_URL, workers=16, log=print, reserve=1 << 30, progress_every=30):
    """Extract {sequence: [scan ids]} from the remote velodyne zip into <root>, skipping existing files.

    Checks first that the scans (sizes from the zip's index) plus `reserve` bytes fit in the free
    space at <root> (on a mounted Google Drive, the Drive's remaining quota) and raises
    NotEnoughSpace before downloading anything if they don't. Logs progress every
    `progress_every` seconds."""
    import shutil
    import threading
    import time
    from .remote_zip import open_remote_zip

    todo = [(s, i) for s, ids in wanted.items() for i in ids
            if not os.path.exists(os.path.join(root, "dataset", "sequences", s, "velodyne", f"{i:06d}.bin"))]
    log(f"fetching {len(todo)} scans ({sum(len(v) for v in wanted.values()) - len(todo)} already present)")
    if not todo:
        return 0
    z = open_remote_zip(url)
    need = sum(z.getinfo(f"dataset/sequences/{s}/velodyne/{i:06d}.bin").file_size for s, i in todo)
    os.makedirs(root, exist_ok=True)
    free = shutil.disk_usage(root).free
    log(f"{need / 1e9:.1f} GB to download, {free / 1e9:.1f} GB free at {root}")
    if need + reserve > free:
        raise NotEnoughSpace(f"the scans need {need / 1e9:.1f} GB (+{reserve / 1e9:.0f} GB headroom) but only "
                             f"{free / 1e9:.1f} GB is free at {root}; use a larger stride or free up space")
    chunks = [todo[k::workers] for k in range(workers)]
    lock, got = threading.Lock(), [0, 0]          # scans, bytes
    stop = threading.Event()

    def work(chunk):
        z = open_remote_zip(url)              # one connection state per thread
        for s, i in chunk:
            name = f"dataset/sequences/{s}/velodyne/{i:06d}.bin"
            dst = os.path.join(root, name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            data = z.read(name)
            with open(dst + ".part", "wb") as fh:
                fh.write(data)
            os.replace(dst + ".part", dst)
            with lock:
                got[0] += 1
                got[1] += len(data)
        return len(chunk)

    t0 = time.time()

    def report():
        while not stop.wait(progress_every):
            with lock:
                n, b = got
            rate = b / max(time.time() - t0, 1e-6)
            eta = (need - b) / rate if rate > 0 else float("inf")
            log(f"  {n}/{len(todo)} scans, {b / 1e9:.1f}/{need / 1e9:.1f} GB, "
                f"{rate / 1e6:.0f} MB/s, about {eta / 60:.0f} min left")

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    done = 0
    try:
        with ThreadPoolExecutor(workers) as ex:
            for n in ex.map(work, [c for c in chunks if c]):
                done += n
    finally:
        stop.set()
    log(f"fetched {done} scans ({got[1] / 1e9:.1f} GB) in {(time.time() - t0) / 60:.1f} min")
    return done
