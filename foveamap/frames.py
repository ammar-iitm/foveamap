"""Sensor-agnostic Lidar frames and the network's input features.

Every data source (the simulator, nuScenes, later SemanticKITTI or CARLA) is
turned into the same ``Frame`` dict, so the network, the grid engine and the
dashboard never need to know where points came from.

Frame keys
    pts      (N, 3) float32  points in the ego frame (x forward, y left, z up)
    inten    (N,)   float32  intensity in [0, 1]
    ring     (N,)   int16    range-image row of the laser that fired (0 = top beam)
    label    (N,)   int8     FoveaMap class id, -1 = ignore / unlabelled
    moving   (N,)   bool     ground-truth moving flag (False when unknown)
    pose     (4, 4) float64  ego -> world transform at this sweep
    sensor   (3,)   float32  Lidar origin in the ego frame
    prev     list of (pts_world (M,3) float64, ring (M,)) for the sweeps
             ~0.1 s and ~0.2 s earlier, or None to let the pipeline use its
             own history (fine when frames arrive at 10 Hz)
    meta     dict (token, timestamp, ...)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import os

import numpy as np

from .sim import CLASSES as SIM_CLASSES, NUM_CLASSES, SENSOR_H, load_sequence

IN_CH = 8


@dataclass
class DatasetInfo:
    name: str
    n_rows: int                  # range-image height (number of lasers)
    n_cols: int = 1024           # range-image width (azimuth bins)
    class_names: list = field(default_factory=lambda: list(SIM_CLASSES))
    active: np.ndarray = field(default_factory=lambda: np.ones(NUM_CLASSES, bool))
    source: str = ""
    hz: float = 10.0


SIM_INFO = DatasetInfo("sim", 64, 1024, source="Simulated 64-beam drive", hz=10.0)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def transform(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def pixel_coords(pts_ego, ring, sensor, n_cols):
    """Column from azimuth around the sensor (col 0 = behind, sweeping clockwise,
    same convention as the simulator), row = ring. Returns (row, col, range)."""
    d = pts_ego - sensor
    az = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    col = np.rint((180.0 - az) / 360.0 * n_cols).astype(np.int64) % n_cols
    rng = np.linalg.norm(d, axis=1)
    return ring.astype(np.int64), col, rng


def range_image(pts_ego, ring, sensor, n_rows, n_cols):
    """Closest point per pixel. Returns range image (0 = empty), index image
    (-1 = empty) and each point's (row, col)."""
    row, col, rng = pixel_coords(pts_ego, ring, sensor, n_cols)
    ok = (row >= 0) & (row < n_rows) & (rng > 0.5)
    flat = row * n_cols + col
    order = np.argsort(-rng)                     # far first, so the nearest write wins
    order = order[ok[order]]
    idx = np.full(n_rows * n_cols, -1, np.int64)
    idx[flat[order]] = order
    img = np.zeros(n_rows * n_cols, np.float32)
    has = idx >= 0
    img[has] = rng[idx[has]]
    return img.reshape(n_rows, n_cols), idx.reshape(n_rows, n_cols), row, col


def row_elevations(pts, row, sensor, n_rows):
    """Mean elevation angle (deg) of each range-image row in this sweep."""
    d = pts - sensor
    elev = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    cnt = np.bincount(row, minlength=n_rows)[:n_rows]
    mean = np.bincount(row, weights=elev, minlength=n_rows)[:n_rows] / np.maximum(cnt, 1)
    have = cnt > 0
    if have.sum() >= 2:                         # fill empty rows by interpolation
        idx = np.arange(n_rows)
        mean = np.interp(idx, idx[have], mean[have])
    return mean


def rows_from_elevation(pts, sensor, elev_of_row):
    """Nearest row for points seen from `sensor`; -1 outside the vertical field of view."""
    d = pts - sensor
    elev = np.degrees(np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])))
    asc = elev_of_row[::-1]                     # rows are top-first; np.interp needs ascending x
    n = len(elev_of_row)
    r = np.rint(np.interp(elev, asc, np.arange(n)[::-1].astype(np.float64))).astype(np.int64)
    spacing = abs(asc[-1] - asc[0]) / max(n - 1, 1)
    r[(elev > asc[-1] + spacing / 2) | (elev < asc[0] - spacing / 2)] = -1
    return r


def make_features(frame, info: DatasetInfo, prev_ego=None):
    """8-channel range-image features for one frame.

    prev_ego: list of 2 (pts_in_current_ego, ring) or None entries (motion cue).
    Returns feats (8, H, W) float32, idx (H, W) and per-point (row, col).
    """
    H, W = info.n_rows, info.n_cols
    pts, sensor = frame["pts"], frame["sensor"]
    rimg, idx, row, col = range_image(pts, frame["ring"], sensor, H, W)
    valid = idx >= 0
    sel = np.where(valid, idx, 0)
    xyz = np.where(valid[..., None], pts[sel], 0.0)
    inten = np.where(valid, frame["inten"][sel], 0.0)
    feats = [rimg / 50.0, xyz[..., 0] / 50.0, xyz[..., 1] / 50.0, xyz[..., 2] / 3.0, inten, valid.astype(np.float32)]
    elev_of_row = row_elevations(pts, row, sensor, H)
    for p in (prev_ego or [None, None])[:2]:
        if p is None or len(p[0]) == 0:
            feats.append(np.zeros((H, W), np.float32))
            continue
        # older sweeps were taken from another position: a laser id no longer
        # sits on the same row, so re-assign rows by elevation seen from here
        prow = rows_from_elevation(p[0], sensor, elev_of_row)
        pr, _, _, _ = range_image(p[0], prow, sensor, H, W)
        both = valid & (pr > 0)
        res = np.where(both, np.abs(rimg - pr) / np.maximum(rimg, 1e-3), 0.0)
        feats.append(np.clip(res * 5, 0, 1).astype(np.float32))
    return np.stack(feats).astype(np.float32), idx, row, col


def prev_in_ego(frame, history=None):
    """Previous sweeps expressed in the current ego frame."""
    inv = np.linalg.inv(frame["pose"])
    src = frame.get("prev")
    if src is None:
        src = history or []
    out = []
    for item in list(src)[:2]:
        if item is None:
            out.append(None)
            continue
        pw, ring = item
        out.append((transform(inv, pw).astype(np.float32), ring))
    while len(out) < 2:
        out.append(None)
    return out


def label_images(frame, idx):
    valid = idx >= 0
    sel = np.where(valid, idx, 0)
    lab = np.where(valid, frame["label"][sel], -1).astype(np.int8)
    mov = np.where(valid, frame["moving"][sel], False)
    return lab, mov


def world_points(frame):
    return transform(frame["pose"], frame["pts"].astype(np.float64))


# ----------------------------------------------------------------------------
# Simulator adapter
# ----------------------------------------------------------------------------
def sim_frames(path_or_seq):
    """Turn a simulated sequence (.npz) into Frames. prev = the two previous
    frames (the simulator runs at 10 Hz, so they are 0.1 s and 0.2 s old)."""
    seq = load_sequence(path_or_seq) if isinstance(path_or_seq, str) else path_or_seq
    T = seq["range"].shape[0]
    frames = []
    for t in range(T):
        valid = seq["valid"][t]
        rows, _ = np.nonzero(valid)
        pose = np.eye(4)
        pose[:2, 3] = seq["ego"][t][:2]
        frames.append(dict(
            pts=seq["xyz"][t][valid].astype(np.float32),
            inten=seq["intensity"][t][valid].astype(np.float32),
            ring=rows.astype(np.int16),
            label=seq["label"][t][valid].astype(np.int8),
            moving=seq["moving"][t][valid].astype(bool),
            pose=pose,
            sensor=np.array([0, 0, SENSOR_H], np.float32),
            prev=None,
            meta=dict(t=t),
        ))
    for t, f in enumerate(frames):
        f["prev"] = [(world_points(frames[t - k]), frames[t - k]["ring"]) if t - k >= 0 else None for k in (1, 2)]
    truth = dict(potholes=seq["potholes"], cross_x=float(seq["cross_x"]))
    return frames, truth


def frames_to_training_arrays(frames, info: DatasetInfo, n=None, log=None, every=250, out_dir=None):
    """Features + per-pixel targets for a list of frames (used by training).
    frames may be any iterable when n (the number of frames) is given.
    log: called with a progress line every `every` frames.
    out_dir: write the arrays to memory-mapped .npy files there instead of holding them in RAM
    (about 1.4 MB a frame for 64 x 1024 images; stride-5 SemanticKITTI is ~5 GB)."""
    F = len(frames) if n is None else n
    shapes = dict(X=((F, IN_CH, info.n_rows, info.n_cols), np.float16), Y=((F, info.n_rows, info.n_cols), np.int8),
                  M=((F, info.n_rows, info.n_cols), bool), R=((F, info.n_rows, info.n_cols), np.float16))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        X, Y, M, R = (np.lib.format.open_memmap(os.path.join(out_dir, f"{k}.npy"), mode="w+", dtype=dt, shape=sh)
                      for k, (sh, dt) in shapes.items())
    else:
        X, Y, M, R = (np.zeros(sh, dt) for sh, dt in shapes.values())
    Y[:] = -1
    for i, f in enumerate(frames):
        feats, idx, _, _ = make_features(f, info, prev_in_ego(f))
        X[i] = feats
        Y[i], M[i] = label_images(f, idx)
        R[i] = feats[0] * 50.0
        if log and (i + 1) % every == 0:
            log(f"prepared {i + 1}/{F} frames")
    return X, Y, M, R
