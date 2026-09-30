"""PyTorch port of `frames.make_features` (range image + 8-channel features on any device).

Same channels and conventions as the NumPy version, which stays the
reference (`tests/test_features_torch.py` checks parity):

* "Nearest point wins" per pixel is a per-pixel `amin` of range, then the
  point index among the points at that range (`amax` on exact ties, where the
  NumPy argsort order is arbitrary anyway).
* `np.interp` (row elevations, re-assigning older sweeps to rows) becomes a
  `searchsorted`-based linear interpolation with the same end clamping.
* No boolean-mask indexing: masked-out points scatter a neutral value
  instead, so building the features never waits on the GPU (no nonzero sync).
"""
from __future__ import annotations

import numpy as np
import torch

from .frames import DatasetInfo


def _t(a, device, dtype=None):
    return torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a).to(device, dtype)


def interp(x, xp, fp):
    """np.interp for 1-D tensors: xp ascending, values outside clamp to the end points."""
    n = xp.numel()
    i = torch.searchsorted(xp, x, right=True).clamp(1, n - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    dx = x1 - x0
    y = f0 + (x - x0) / torch.where(dx == 0, 1, dx) * (f1 - f0)
    y = torch.where(x <= xp[0], fp[0], y)
    return torch.where(x >= xp[-1], fp[-1], y)


def _elev(d):
    return torch.rad2deg(torch.atan2(d[:, 2], torch.hypot(d[:, 0], d[:, 1])))


def range_image(pts, ring, sensor, n_rows, n_cols):
    """Device version of `frames.range_image`: (range img, index img, row, col)."""
    d = pts - sensor
    az = torch.rad2deg(torch.atan2(d[:, 1], d[:, 0]))
    col = torch.round((180.0 - az) / 360.0 * n_cols).long() % n_cols      # round half to even, like np.rint
    rng = torch.linalg.vector_norm(d, dim=1)
    row = ring.long()
    ok = (row >= 0) & (row < n_rows) & (rng > 0.5)
    flat = torch.where(ok, row * n_cols + col, 0)      # masked points scatter neutral values into pixel 0
    HW = n_rows * n_cols
    best = torch.full((HW,), float("inf"), dtype=rng.dtype, device=pts.device)
    best.scatter_reduce_(0, flat, torch.where(ok, rng, float("inf")), "amin")
    win = ok & (rng == best[flat])
    pid = torch.arange(len(pts), device=pts.device)
    idx = torch.full((HW,), -1, dtype=torch.long, device=pts.device)
    idx.scatter_reduce_(0, flat, torch.where(win, pid, -1), "amax")
    img = torch.where(idx >= 0, rng[idx.clamp(min=0)], 0.0)
    return img.view(n_rows, n_cols), idx.view(n_rows, n_cols), row, col


def row_elevations(pts, row, sensor, n_rows):
    elev = _elev(pts - sensor)
    dev = pts.device
    r = row.clamp(0, n_rows)                    # rows past the image go to a spill bin
    cnt = torch.zeros(n_rows + 1, dtype=elev.dtype, device=dev).index_add_(0, r, torch.ones_like(elev))[:n_rows]
    mean = torch.zeros(n_rows + 1, dtype=elev.dtype, device=dev).index_add_(0, r, elev)[:n_rows] / cnt.clamp(min=1)
    have = cnt > 0
    # fill empty rows by linear interpolation between the nearest rows with data (clamped at the ends)
    ar = torch.arange(n_rows, device=dev)
    lo = torch.cummax(torch.where(have, ar, -1), 0).values
    hi = torch.cummin(torch.where(have, ar, n_rows).flip(0), 0).values.flip(0)
    lo_c, hi_c = lo.clamp(0, n_rows - 1), hi.clamp(0, n_rows - 1)
    t = (ar - lo_c) / (hi_c - lo_c).clamp(min=1)
    m_lo, m_hi = mean[lo_c], mean[hi_c]
    filled = torch.where((lo >= 0) & (hi < n_rows), m_lo + t * (m_hi - m_lo), torch.where(lo >= 0, m_lo, m_hi))
    return torch.where(have.sum() >= 2, filled, mean)


def rows_from_elevation(pts, sensor, elev_of_row):
    elev = _elev(pts - sensor)
    asc = elev_of_row.flip(0)
    n = elev_of_row.numel()
    rows_desc = torch.arange(n - 1, -1, -1, device=pts.device, dtype=elev.dtype)
    r = torch.round(interp(elev, asc, rows_desc)).long()
    spacing = (asc[-1] - asc[0]).abs() / max(n - 1, 1)
    return torch.where((elev > asc[-1] + spacing / 2) | (elev < asc[0] - spacing / 2), -1, r)


def make_features(frame, info: DatasetInfo, prev_ego=None, device="cpu"):
    """Device version of `frames.make_features`. Returns feats (8, H, W) float32 and
    idx (H, W), row, col, all as tensors on `device`."""
    dev = torch.device(device)
    H, W = info.n_rows, info.n_cols
    pts = _t(frame["pts"], dev, torch.float32)
    sensor = _t(frame["sensor"], dev, torch.float32)
    rimg, idx, row, col = range_image(pts, _t(frame["ring"], dev), sensor, H, W)
    valid = idx >= 0
    sel = idx.clamp(min=0)
    xyz = torch.where(valid[..., None], pts[sel], 0.0)
    inten = torch.where(valid, _t(frame["inten"], dev, torch.float32)[sel], 0.0)
    feats = [rimg / 50.0, xyz[..., 0] / 50.0, xyz[..., 1] / 50.0, xyz[..., 2] / 3.0, inten, valid.float()]
    elev_of_row = row_elevations(pts, row, sensor, H)
    for p in (prev_ego or [None, None])[:2]:
        if p is None or len(p[0]) == 0:
            feats.append(torch.zeros((H, W), device=dev))
            continue
        # older sweeps were taken from another position: re-assign rows by elevation seen from here
        ppts = _t(p[0], dev, torch.float32)
        prow = rows_from_elevation(ppts, sensor, elev_of_row)
        pr, _, _, _ = range_image(ppts, prow, sensor, H, W)
        both = valid & (pr > 0)
        res = torch.where(both, (rimg - pr).abs() / rimg.clamp(min=1e-3), 0.0)
        feats.append((res * 5).clamp(0, 1))
    return torch.stack(feats), idx, row, col
