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
IS_OBJECT = np.zeros(256, bool)
IS_OBJECT[list(OBJECT_CLASSES)] = True
MIN_CELLS = {VEHICLE: 3, PERSON: 2, POLE: 2}     # fewer member cells than this is treated as noise
MATCH_DIST = {VEHICLE: 1.5, PERSON: 0.75, POLE: 0.75}   # m, centre distance for a detection to match
FIELDS = ("cls", "moving", "x", "y", "length", "width", "yaw", "z_top", "n_cells", "area")
# Minimum footprint (m^2) of a reported object. Most false objects on real scans are a few
# misclassified cells far from any real object. Chosen from the benchmark's sweep on SemanticKITTI
# sequence 08: per class, the smallest area whose F1 is within 1 point of the best (vehicles were
# still improving at 1 m^2, the largest tried then, so the sweep now goes further).
MIN_AREA = {VEHICLE: 1.0, PERSON: 0.05, POLE: 0.02}
AREA_SWEEP = {VEHICLE: (0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0), PERSON: (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3),
              POLE: (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)}


def _members(grid, snap, dyn, max_age):
    """Object-class cells of every tier: (cls, moving, x, y, cell size, z_max, coarse i, coarse j).
    Moving cells (the dynamic layer) take precedence over static ones in the same place."""
    tiers, origins = grid.tiers, grid.origins
    tc, oc = tiers[-1], np.asarray(origins[-1])
    out = []
    for k, (t, s, o) in enumerate(zip(tiers, snap, origins)):
        n = t.n
        eff = getattr(s, "eff_cls", s.cls)
        static = IS_OBJECT[eff] & (s.age <= max_age)
        d = dyn[k] if dyn is not None else None
        didx = (np.asarray(d["i"], np.int64) * n + np.asarray(d["j"], np.int64)) if d is not None else np.zeros(0, np.int64)
        dcls = np.asarray(d["cls"], np.uint8) if d is not None else np.zeros(0, np.uint8)
        if k > 0:                                       # cells under the finer tier are covered there
            lo, hi = grid.inner_box(k)
            static[lo[0]:hi[0], lo[1]:hi[1]] = False
            di, dj = didx // n, didx % n
            out_box = (di < lo[0]) | (di >= hi[0]) | (dj < lo[1]) | (dj >= hi[1])
            didx, dcls = didx[out_box], dcls[out_box]
        sidx = np.flatnonzero(static)
        if len(didx):
            sidx = sidx[~np.isin(sidx, didx)]
        idx = np.concatenate([sidx, didx])
        if not len(idx):
            continue
        i, j = idx // n, idx % n
        o = np.asarray(o)
        r = tc.ratio // t.ratio
        out.append(dict(
            cls=np.concatenate([eff.ravel()[sidx], dcls]).astype(np.uint8),
            moving=np.r_[np.zeros(len(sidx), bool), np.ones(len(didx), bool)],
            x=(o[0] + i + 0.5) * t.cell, y=(o[1] + j + 0.5) * t.cell,
            cell=np.full(len(i), t.cell), z=s.z_max.ravel()[idx].astype(np.float32),
            ci=(o[0] + i) // r - oc[0], cj=(o[1] + j) // r - oc[1]))
    if not out:
        return None
    return {key: np.concatenate([m[key] for m in out]) for key in out[0]}


def report(objs):
    """The objects worth reporting: those at least MIN_AREA in footprint."""
    return [o for o in objs if o["area"] >= MIN_AREA[o["cls"]]]


def extract_objects(grid, snap, dyn=None, max_age=0):
    """Objects in the map, as a list of dicts with FIELDS.

    grid: the FoveatedGrid (or TorchFoveatedGrid) the snapshot came from, for tiers / origins.
    snap: host TierLayers per tier (`grid.snapshot()`); dyn: host dynamic cells per tier.
    max_age: include static cells observed up to this many frames ago (0 = this frame).
    Every box is computed at once from per-object sums (no per-object Python work)."""
    m = _members(grid, snap, dyn, max_age)
    if m is None:
        return []
    comp = np.zeros(len(m["cls"]), np.int64)            # object id per member cell, 0 = none
    next_id = 0
    for c in OBJECT_CLASSES:
        sel = m["cls"] == c
        if not sel.any():
            continue
        ci, cj = m["ci"][sel], m["cj"][sel]
        i0, j0 = ci.min(), cj.min()                     # label only the box around this class's cells
        raster = np.zeros((ci.max() - i0 + 1, cj.max() - j0 + 1), bool)
        raster[ci - i0, cj - j0] = True
        lab, nlab = ndimage.label(raster, structure=np.ones((3, 3), bool))
        comp[sel] = lab[ci - i0, cj - j0] + next_id
        next_id += nlab
    ids, g = np.unique(comp, return_inverse=True)       # every member belongs to some component
    count = np.bincount(g)
    first = np.zeros(len(ids), np.int64)
    first[g[::-1]] = np.arange(len(g))[::-1]            # one member of each component
    cls = m["cls"][first].astype(np.int64)
    keep = count >= np.array([MIN_CELLS.get(int(c), 1) for c in cls])
    if not keep.any():
        return []
    sel = keep[g]                                        # drop small components, renumber the rest
    g = np.cumsum(keep)[g[sel]] - 1
    cls, count = cls[keep], count[keep]
    x, y, cell = m["x"][sel], m["y"][sel], m["cell"][sel]
    w = cell * cell
    sw = lambda v: np.bincount(g, w * v, minlength=len(cls))
    W = sw(np.ones_like(w))
    mx, my = sw(x) / W, sw(y) / W
    dx, dy = x - mx[g], y - my[g]
    cxx, cyy, cxy = sw(dx * dx) / W, sw(dy * dy) / W, sw(dx * dy) / W
    yaw = 0.5 * np.arctan2(2 * cxy, cxx - cyy)                # long axis
    u, v = np.cos(yaw), np.sin(yaw)
    pa, pb = dx * u[g] + dy * v[g], -dx * v[g] + dy * u[g]
    half = cell / 2 * (np.abs(u) + np.abs(v))[g]           # a cell's half-extent along either axis
    order = np.argsort(g, kind="stable")
    starts = np.r_[0, np.nonzero(np.diff(g[order]))[0] + 1]
    red = lambda f, a: f.reduceat(a[order], starts)
    a0, a1 = red(np.minimum, pa - half), red(np.maximum, pa + half)
    b0, b1 = red(np.minimum, pb - half), red(np.maximum, pb + half)
    ca, cb = (a0 + a1) / 2, (b0 + b1) / 2
    length, width = a1 - a0, b1 - b0
    swap = width > length                                  # keep yaw on the long side
    length, width = np.where(swap, width, length), np.where(swap, length, width)
    yaw = (np.where(swap, yaw + np.pi / 2, yaw) + np.pi / 2) % np.pi - np.pi / 2
    cx, cy = mx + ca * u - cb * v, my + ca * v + cb * u
    z_top = red(np.fmax, m["z"][sel].astype(np.float64))   # NaN only where no member has a height
    moving = sw(m["moving"][sel].astype(np.float64)) > 0.5 * W
    return [dict(cls=int(cls[k]), moving=bool(moving[k]), x=float(cx[k]), y=float(cy[k]),
                 length=float(length[k]), width=float(width[k]), yaw=float(yaw[k]),
                 z_top=None if np.isnan(z_top[k]) else float(z_top[k]), n_cells=int(count[k]),
                 area=float(W[k]))
            for k in range(len(cls))]


FRAGMENT_DIST = 3.0     # m: a false detection this close to a real object is counted as a fragment of it


def match_objects(pred, gt, ego_xy, max_range=25.0):
    """Greedy per-class matching by centre distance (MATCH_DIST) for objects within max_range of
    the ego. Returns ({cls: [tp, fp, fn]}, moving flags agreeing, matched pairs, diagnostics).

    diagnostics[cls] lists the sizes (cells) of true positives, false positives and misses, and
    for each false positive whether a ground-truth object of the same class (a fragment) or of
    another class (a misclassification) lies within FRAGMENT_DIST."""
    near = lambda o: np.hypot(o["x"] - ego_xy[0], o["y"] - ego_xy[1]) < max_range
    counts = {c: [0, 0, 0] for c in OBJECT_CLASSES}
    diag = {c: dict(tp_cells=[], fp_cells=[], fn_cells=[], fp_fragment=[], fp_other_class=[]) for c in OBJECT_CLASSES}
    gxy = np.array([[o["x"], o["y"]] for o in gt]).reshape(-1, 2)
    gcls = np.array([o["cls"] for o in gt])
    agree = pairs = 0
    for c in OBJECT_CLASSES:
        P = [o for o in pred if o["cls"] == c and near(o)]
        G = [o for o in gt if o["cls"] == c and near(o)]
        used_p, used_g = set(), set()
        if P and G:
            d = np.hypot(np.array([p["x"] for p in P])[:, None] - np.array([g["x"] for g in G])[None],
                         np.array([p["y"] for p in P])[:, None] - np.array([g["y"] for g in G])[None])
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
        dg = diag[c]
        for i, o in enumerate(P):
            if i in used_p:
                dg["tp_cells"].append(o["n_cells"])
                continue
            dg["fp_cells"].append(o["n_cells"])
            dist = np.hypot(gxy[:, 0] - o["x"], gxy[:, 1] - o["y"]) if len(gxy) else np.zeros(0)
            dg["fp_fragment"].append(bool(np.any((dist < FRAGMENT_DIST) & (gcls == c))))
            dg["fp_other_class"].append(bool(np.any((dist < FRAGMENT_DIST) & (gcls != c))))
        dg["fn_cells"] += [o["n_cells"] for j, o in enumerate(G) if j not in used_g]
    return counts, agree, pairs, diag


def pack(objs):
    """Compact rows for the dashboard: [cls, moving, x, y, length, width, yaw, z_top]."""
    r2 = lambda v: None if v is None else round(v, 2)
    return [[o["cls"], int(o["moving"]), r2(o["x"]), r2(o["y"]), r2(o["length"]), r2(o["width"]),
             round(o["yaw"], 3), r2(o["z_top"])] for o in objs]
