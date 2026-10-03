"""Object-level output: obstacle cells of the map grouped into objects.

Each object has a class (vehicle, person or pole), a moving flag, an oriented
box and a top height. Objects come from the published map snapshot, so they
are computed on the host after the map step:

* Cells: those whose effective class is an object class and that were observed
  within `max_age` frames, plus this frame's moving cells (the dynamic layer,
  which is never fused into the static map). Each tier contributes only the
  cells not covered by a finer tier, so nothing is counted twice.
* Grouping: per class, 8-connected components on the coarsest tier's raster
  (0.5 m in the spec profile). That bridges the gaps between scan rings without
  merging a person into the car next to them.
* Box: principal axes of the member cells (area-weighted), extents from the
  cells' projections plus half a cell; yaw is the long axis in the world frame.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .sim import POLE, VEHICLE, PERSON

OBJECT_CLASSES = (VEHICLE, PERSON, POLE)
MIN_CELLS = {VEHICLE: 3, PERSON: 2, POLE: 2}     # fewer member cells than this is treated as noise
MATCH_DIST = {VEHICLE: 1.5, PERSON: 0.75, POLE: 0.75}   # m, centre distance for a detection to match
FIELDS = ("cls", "moving", "x", "y", "length", "width", "yaw", "z_top", "n_cells")


def _members(grid, snap, dyn, max_age):
    """Object-class cells of every tier: (cls, moving, x, y, cell size, z_max, coarse i, coarse j)."""
    tiers, origins = grid.tiers, grid.origins
    tc, oc = tiers[-1], np.asarray(origins[-1])
    out = []
    for k, (t, s, o) in enumerate(zip(tiers, snap, origins)):
        eff = getattr(s, "eff_cls", s.cls)
        cls = np.where(np.isin(eff, OBJECT_CLASSES) & (s.age <= max_age), eff, 255).astype(np.uint8)
        moving = np.zeros(cls.shape, bool)
        d = dyn[k] if dyn is not None else None
        if d is not None and len(d["i"]):
            di, dj = np.asarray(d["i"]), np.asarray(d["j"])
            cls[di, dj] = np.asarray(d["cls"], np.uint8)
            moving[di, dj] = True
        sel = cls != 255
        if k > 0:
            sel &= ~grid.inner_mask(k)             # covered by the finer tier
        i, j = np.nonzero(sel)
        if not len(i):
            continue
        o = np.asarray(o)
        r = tc.ratio // t.ratio
        out.append(dict(
            cls=cls[i, j], moving=moving[i, j],
            x=(o[0] + i + 0.5) * t.cell, y=(o[1] + j + 0.5) * t.cell,
            cell=np.full(len(i), t.cell), z=s.z_max[i, j].astype(np.float32),
            ci=(o[0] + i) // r - oc[0], cj=(o[1] + j) // r - oc[1]))
    if not out:
        return None
    return {key: np.concatenate([m[key] for m in out]) for key in out[0]}


def extract_objects(grid, snap, dyn=None, max_age=0):
    """Objects in the map, as a list of dicts with FIELDS.

    grid: the FoveatedGrid (or TorchFoveatedGrid) the snapshot came from, for tiers / origins.
    snap: host TierLayers per tier (`grid.snapshot()`); dyn: host dynamic cells per tier.
    max_age: include static cells observed up to this many frames ago (0 = this frame)."""
    m = _members(grid, snap, dyn, max_age)
    if m is None:
        return []
    n = grid.tiers[-1].n
    objs = []
    for c in OBJECT_CLASSES:
        sel = m["cls"] == c
        if not sel.any():
            continue
        raster = np.zeros((n, n), bool)
        raster[m["ci"][sel], m["cj"][sel]] = True
        lab, nlab = ndimage.label(raster, structure=np.ones((3, 3), bool))
        comp = lab[m["ci"][sel], m["cj"][sel]]
        order = np.argsort(comp, kind="stable")
        comp = comp[order]
        idx = np.nonzero(sel)[0][order]
        starts = np.r_[0, np.nonzero(np.diff(comp))[0] + 1]
        for a, b in zip(starts, np.r_[starts[1:], len(comp)]):
            if b - a < MIN_CELLS[c]:
                continue
            objs.append(_box(c, m, idx[a:b]))
    return objs


def _box(c, m, ids):
    x, y, cell = m["x"][ids], m["y"][ids], m["cell"][ids]
    w = cell * cell
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    dx, dy = x - mx, y - my
    cxx, cyy, cxy = np.average(dx * dx, weights=w), np.average(dy * dy, weights=w), np.average(dx * dy, weights=w)
    yaw = 0.5 * np.arctan2(2 * cxy, cxx - cyy)           # long axis
    u, v = np.cos(yaw), np.sin(yaw)
    pa, pb = dx * u + dy * v, -dx * v + dy * u
    half = cell / 2 * (abs(u) + abs(v))                  # a cell's half-extent along either axis
    a0, a1 = (pa - half).min(), (pa + half).max()
    b0, b1 = (pb - half).min(), (pb + half).max()
    ca, cb = (a0 + a1) / 2, (b0 + b1) / 2
    length, width = a1 - a0, b1 - b0
    if width > length:                                    # keep yaw on the long side
        length, width, yaw = width, length, yaw + np.pi / 2
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    z = m["z"][ids]
    z = z[np.isfinite(z)]
    moving = float(np.sum(w * m["moving"][ids])) > 0.5 * float(np.sum(w))
    return dict(cls=int(c), moving=bool(moving),
                x=float(mx + ca * u - cb * v), y=float(my + ca * v + cb * u),
                length=float(length), width=float(width), yaw=float(yaw),
                z_top=float(z.max()) if len(z) else None, n_cells=int(len(ids)))


def match_objects(pred, gt, ego_xy, max_range=25.0):
    """Greedy per-class matching by centre distance (MATCH_DIST) for ground-truth objects within
    max_range of the ego. Returns {cls: [tp, fp, fn]} and the number of matched pairs whose
    moving flags agree, out of the matched pairs."""
    near = lambda o: np.hypot(o["x"] - ego_xy[0], o["y"] - ego_xy[1]) < max_range
    counts = {c: [0, 0, 0] for c in OBJECT_CLASSES}
    agree = pairs = 0
    for c in OBJECT_CLASSES:
        P = [o for o in pred if o["cls"] == c and near(o)]
        G = [o for o in gt if o["cls"] == c and near(o)]
        if P and G:
            d = np.hypot(np.array([p["x"] for p in P])[:, None] - np.array([g["x"] for g in G])[None],
                         np.array([p["y"] for p in P])[:, None] - np.array([g["y"] for g in G])[None])
        used_p, used_g = set(), set()
        if P and G:
            for flat in np.argsort(d, axis=None):
                i, j = divmod(int(flat), len(G))
                if d[i, j] > MATCH_DIST[c]:
                    break
                if i in used_p or j in used_g:
                    continue
                used_p.add(i); used_g.add(j)
                pairs += 1
                agree += P[i]["moving"] == G[j]["moving"]
        tp = len(used_p)
        counts[c][0] += tp
        counts[c][1] += len(P) - tp
        counts[c][2] += len(G) - tp
    return counts, agree, pairs


def pack(objs):
    """Compact rows for the dashboard: [cls, moving, x, y, length, width, yaw, z_top]."""
    r2 = lambda v: None if v is None else round(v, 2)
    return [[o["cls"], int(o["moving"]), r2(o["x"]), r2(o["y"]), r2(o["length"]), r2(o["width"]),
             round(o["yaw"], 3), r2(o["z_top"])] for o in objs]
