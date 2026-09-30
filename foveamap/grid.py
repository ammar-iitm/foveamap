"""Variable-resolution (foveated) 2.5D grid engine.

Design (see Architecture Vision, sections 5-7):

* A stack of concentric, square, world-aligned tiers. Default "spec" profile:
      tier 0: 5 cm cells, +/-10 m      (400 x 400)
      tier 1: 50 cm cells, +/-100 m    (400 x 400)
* Every tier's cell size is an integer multiple of the finest cell, and all
  tiers share the world origin. A point's fine index f = floor(x / 0.05) is
  computed ONCE in integer space; its index in tier k is f // ratio_k.
  Floor-division of integers is exact, so a coarse cell is always an exact
  union of fine cells -> no misalignment and no double counting.
* Tier windows are snapped to the coarsest lattice, so the fine box moves in
  whole 50 cm steps and never splits a coarse cell.
* Every point inside the outer window is binned into every tier that
  contains it. This is the "mip-up": tier 1 is complete everywhere and can
  be read alone by a planner that wants one resolution.

Persistent record per cell = 16 bytes (structure-of-arrays):
    count u16 | z_min f16 | z_max f16 | ground f16 | rough f16 |
    cls u8 | conf u8 | flags u8 (bits0-3 flags, bits4-7 ground class) |
    clear u8 (2 cm units) | cost u8 | age u8
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import uniform_filter

from .sim import (NUM_CLASSES, GROUND_CLASSES, ROAD, PARKING, SIDEWALK, TERRAIN,
                  VEGETATION, BUILDING, POLE, VEHICLE, PERSON)

UNKNOWN = 255
F_DYNAMIC, F_OVERHANG, F_STEP, F_DEPRESSION = 1, 2, 4, 8

# traversability prior per class (0 free ... 254 lethal)
COST_PRIOR = np.full(256, 200, np.int32)
COST_PRIOR[[ROAD, PARKING, SIDEWALK, TERRAIN, VEGETATION, BUILDING, POLE, VEHICLE, PERSON]] = \
    [0, 10, 110, 150, 254, 254, 254, 254, 254]
GROUND_MASK = np.zeros(NUM_CLASSES, bool)
GROUND_MASK[list(GROUND_CLASSES)] = True
DRIVABLE = np.zeros(256, bool)
DRIVABLE[[ROAD, PARKING]] = True
VEHICLE_CLEARANCE = 2.5      # m: an obstacle higher than this above ground can be driven under
STEP_THRESH = 0.08           # m: curb / pothole edge
DEPRESSION_THRESH = 0.05     # m below the local drivable surface
DEPRESSION_WIN = 2.5         # m, window for the local drivable-surface reference


@dataclass
class Tier:
    cell: float       # metres
    half: float       # half-extent, metres
    ratio: int        # cell / finest cell (integer)
    n: int            # cells per side
    alpha: float      # temporal EMA weight for new observations


PROFILES = {
    "spec": ((0.05, 10.0), (0.5, 100.0)),
    "graded": ((0.05, 10.0), (0.10, 25.0), (0.5, 100.0)),
    "uniform5": ((0.05, 100.0),),
    "uniform50": ((0.5, 100.0),),
}


class TierLayers:
    """The 16-byte-per-cell persistent state of one tier."""
    FIELDS = dict(count=np.uint16, z_min=np.float16, z_max=np.float16, ground=np.float16,
                  rough=np.float16, cls=np.uint8, conf=np.uint8, flags=np.uint8,
                  clear=np.uint8, cost=np.uint8, age=np.uint8)

    def __init__(self, n):
        self.n = n
        self.count = np.zeros((n, n), np.uint16)
        self.z_min = np.full((n, n), np.nan, np.float16)
        self.z_max = np.full((n, n), np.nan, np.float16)
        self.ground = np.full((n, n), np.nan, np.float16)
        self.rough = np.full((n, n), np.nan, np.float16)
        self.cls = np.full((n, n), UNKNOWN, np.uint8)
        self.conf = np.zeros((n, n), np.uint8)
        self.flags = np.full((n, n), 0xF0, np.uint8)      # ground class nibble 0xF = unknown
        self.clear = np.full((n, n), UNKNOWN, np.uint8)
        self.cost = np.full((n, n), UNKNOWN, np.uint8)
        self.age = np.full((n, n), UNKNOWN, np.uint8)

    @property
    def nbytes(self):
        return sum(getattr(self, f).nbytes for f in self.FIELDS)

    def shifted(self, d):
        """Copy of this state moved by d = (di, dj) cells (window scroll)."""
        out = TierLayers(self.n)
        di, dj = int(d[0]), int(d[1])
        n = self.n
        if abs(di) >= n or abs(dj) >= n:
            return out
        src = (slice(max(di, 0), n + min(di, 0)), slice(max(dj, 0), n + min(dj, 0)))
        dst = (slice(max(-di, 0), n + min(-di, 0)), slice(max(-dj, 0), n + min(-dj, 0)))
        for f in self.FIELDS:
            getattr(out, f)[dst] = getattr(self, f)[src]
        return out


class FoveatedGrid:
    def __init__(self, profile="spec", fuse=True):
        spec = PROFILES[profile] if isinstance(profile, str) else profile
        base, coarse = spec[0][0], spec[-1][0]
        self.base, self.coarse = base, coarse
        self.tiers: list[Tier] = []
        for k, (cell, half) in enumerate(spec):
            ratio = round(cell / base)
            assert abs(ratio * base - cell) < 1e-9, "tier cell must be an integer multiple of the finest cell"
            assert abs(round(coarse / cell) * cell - coarse) < 1e-9, "coarsest cell must be a multiple of every tier cell"
            assert abs(round(2 * half / coarse) * coarse - 2 * half) < 1e-9, "tier extent must be whole coarse cells"
            if k:
                assert cell > spec[k - 1][0] and half > spec[k - 1][1]
            self.tiers.append(Tier(cell, half, ratio, round(2 * half / cell), 0.3 if k == 0 else 0.5))
        self.coarse_ratio = round(coarse / base)
        self.fuse = fuse
        self.state = [TierLayers(t.n) for t in self.tiers]
        self.origins = None

    # ------------------------------------------------------------------ utils
    @property
    def nbytes(self):
        return sum(s.nbytes for s in self.state)

    @property
    def n_cells(self):
        return sum(t.n * t.n for t in self.tiers)

    def window_origins(self, ego_xy):
        """Integer origin (in each tier's own cell units) of every tier window."""
        ec = np.floor(np.asarray(ego_xy, np.float64) / self.coarse).astype(np.int64)
        out = []
        for t in self.tiers:
            half_coarse = round(t.half / self.coarse)
            out.append((ec - half_coarse) * round(self.coarse / t.cell))
        return out

    def fine_index(self, xy_world):
        return np.floor(np.asarray(xy_world, np.float64) / self.base).astype(np.int64)

    # -------------------------------------------------------------- binning
    def bin_points(self, xy_world, z, probs, moving, origins):
        """Scatter-reduce points into every tier. Returns per-tier compact stats.

        probs:  (N, C) class probabilities (one-hot for ground truth)
        moving: (N,) bool, predicted moving flag
        """
        f = self.fine_index(xy_world)
        cls = probs.argmax(1)
        is_ground = GROUND_MASK[cls]
        static = ~moving
        out = []
        for t, org in zip(self.tiers, origins):
            ij = f // t.ratio - org
            inw = (ij >= 0).all(1) & (ij < t.n).all(1)
            idx = np.nonzero(inw)[0]
            key = ij[idx, 0] * t.n + ij[idx, 1]
            uk, inv = np.unique(key, return_inverse=True)
            M = len(uk)
            zz, st, gg = z[idx], static[idx], is_ground[idx] & static[idx]
            order = np.argsort(inv, kind="stable")
            starts = np.searchsorted(inv[order], np.arange(M))

            def rmin(v):
                return np.minimum.reduceat(v[order], starts) if M else np.zeros(0)

            def rmax(v):
                return np.maximum.reduceat(v[order], starts) if M else np.zeros(0)

            bc = lambda w: np.bincount(inv, weights=w, minlength=M)  # noqa: E731
            n_static = bc(st.astype(np.float64))
            n_ground = bc(gg.astype(np.float64))
            gsum, gsq = bc(np.where(gg, zz, 0.0)), bc(np.where(gg, zz * zz, 0.0))
            P = probs[idx]
            p_static = np.stack([bc(P[:, c] * st) for c in range(P.shape[1])], 1)
            p_all = np.stack([bc(P[:, c]) for c in range(P.shape[1])], 1)
            mv = moving[idx]
            c_mv = cls[idx]
            out.append(dict(
                key=uk, n_pts=np.bincount(inv, minlength=M), n_static=n_static,
                z_min=rmin(np.where(st, zz, np.inf)), z_max=rmax(np.where(st, zz, -np.inf)),
                ground=np.where(n_ground > 0, gsum / np.maximum(n_ground, 1), np.nan),
                rough=np.where(n_ground > 1, np.sqrt(np.maximum(gsq / np.maximum(n_ground, 1) - (gsum / np.maximum(n_ground, 1)) ** 2, 0)), np.nan),
                zmin_ng=rmin(np.where(st & ~is_ground[idx], zz, np.inf)),
                p_static=p_static, p_all=p_all,
                n_dyn=bc(mv.astype(np.float64)),
                n_dyn_person=bc((mv & (c_mv == PERSON)).astype(np.float64)),
                n_in=len(idx),
            ))
        return out

    # --------------------------------------------------------------- update
    def update(self, xy_world, z, probs, moving, ego_xy):
        """Process one frame. Returns (map_layers_per_tier, frame_stats_per_tier)."""
        origins = self.window_origins(ego_xy)
        stats = self.bin_points(xy_world, z, probs, moving, origins)
        if self.origins is not None and self.fuse:
            self.state = [s.shifted(o - oo) for s, o, oo in zip(self.state, origins, self.origins)]
        elif not self.fuse:
            self.state = [TierLayers(t.n) for t in self.tiers]
        self.origins = origins
        dyn = []
        for t, s, st in zip(self.tiers, self.state, stats):
            dyn.append(self._fuse_tier(t, s, st))
            self._derive(t, s)
        return dyn, stats

    def _fuse_tier(self, t, s: TierLayers, st):
        n = t.n
        key = st["key"]
        obs = st["n_static"] > 0
        k = key[obs]
        i, j = k // n, k % n
        # ---- class: blend new probabilities with the stored (cls, conf)
        p = st["p_static"][obs]
        p = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
        old_c = s.cls[i, j].astype(np.int64)
        old_conf = s.conf[i, j] / 255.0
        a = t.alpha if self.fuse else 1.0
        q = a * p
        has_old = old_c != UNKNOWN
        q[np.nonzero(has_old)[0], old_c[has_old]] += (1 - a) * old_conf[has_old]
        new_c = q.argmax(1)
        s.cls[i, j] = new_c
        s.conf[i, j] = np.clip(q.max(1) / np.maximum(q.sum(1), 1e-9) * 255, 0, 255).astype(np.uint8)
        # ground class (argmax over ground classes only), kept in the flags high nibble
        pg = np.where(GROUND_MASK[None, :], p, 0.0)
        gcls = np.where(pg.sum(1) > 0.05, pg.argmax(1), 0xF)
        # ---- heights
        g_new = st["ground"][obs]
        g_old = s.ground[i, j].astype(np.float64)
        both = np.isfinite(g_new) & np.isfinite(g_old)
        g = np.where(both, (1 - a) * g_old + a * g_new, np.where(np.isfinite(g_new), g_new, g_old))
        s.ground[i, j] = g
        r_new, r_old = st["rough"][obs], s.rough[i, j].astype(np.float64)
        s.rough[i, j] = np.where(np.isfinite(r_new), np.where(np.isfinite(r_old), (1 - a) * r_old + a * r_new, r_new), r_old)
        s.z_min[i, j] = st["z_min"][obs]
        s.z_max[i, j] = st["z_max"][obs]
        s.count[i, j] = np.minimum(st["n_static"][obs], 65535)
        zng = st["zmin_ng"][obs]
        clear = np.where(np.isfinite(zng) & np.isfinite(g), zng - g, np.nan)
        s.clear[i, j] = np.where(np.isfinite(clear), np.clip(clear / 0.02, 0, 254), UNKNOWN).astype(np.uint8)
        old_flags = s.flags[i, j]
        keep_g = np.where(gcls == 0xF, old_flags >> 4, gcls)
        s.flags[i, j] = (keep_g << 4).astype(np.uint8)
        # ---- age
        age = s.age.astype(np.int32) + 1
        age[s.age == UNKNOWN] = UNKNOWN
        age[i, j] = 0
        s.age[:] = np.minimum(age, UNKNOWN).astype(np.uint8)
        # ---- dynamic layer: this frame only, never fused
        dyn_cells = st["n_dyn"] > 0
        dk = key[dyn_cells]
        dcls = np.where(st["n_dyn_person"][dyn_cells] * 2 > st["n_dyn"][dyn_cells], PERSON, VEHICLE)
        return dict(i=dk // n, j=dk % n, cls=dcls)

    def _derive(self, t, s: TierLayers):
        """Flags + traversability cost from the fused layers (vectorised over the tier)."""
        cls = s.cls.astype(np.int32)
        gcls = (s.flags >> 4).astype(np.int32)
        ground = s.ground.astype(np.float32)
        valid_g = np.isfinite(ground)
        flags = np.zeros_like(s.flags)
        # overhang: obstacle points start well above the ground
        clear_m = np.where(s.clear != UNKNOWN, s.clear * 0.02, np.nan)
        overhang = np.isfinite(clear_m) & (clear_m > 0.5) & valid_g
        flags |= np.where(overhang, F_OVERHANG, 0).astype(np.uint8)
        passable_under = overhang & (clear_m >= VEHICLE_CLEARANCE) & (gcls != 0xF)
        eff_cls = np.where(passable_under, gcls, cls)
        # step edges: height jump to a neighbour 1 or 2 cells away
        gz = np.where(valid_g, ground, np.nan)
        step = np.zeros_like(gz)
        for d in (1, 2):
            for ax in (0, 1):
                a = np.roll(gz, d, ax)
                diff = np.abs(gz - a)
                if ax == 0:
                    diff[:d, :] = np.nan
                else:
                    diff[:, :d] = np.nan
                step = np.fmax(step, np.nan_to_num(diff, nan=0.0))
                b = np.roll(diff, -d, ax)
                step = np.fmax(step, np.nan_to_num(b, nan=0.0))
        stepf = step > STEP_THRESH
        flags |= np.where(stepf & valid_g, F_STEP, 0).astype(np.uint8)
        # depressions (potholes): below the local mean of drivable ground
        drv = DRIVABLE[np.clip(eff_cls, 0, 255)] & valid_g
        win = max(3, int(round(DEPRESSION_WIN / t.cell)) | 1)
        num = uniform_filter(np.where(drv, gz, 0.0), win, mode="constant")
        den = uniform_filter(drv.astype(np.float32), win, mode="constant")
        ref = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)
        dep = drv & (gz < ref - DEPRESSION_THRESH)
        flags |= np.where(dep, F_DEPRESSION, 0).astype(np.uint8)
        # cost
        cost = COST_PRIOR[np.clip(eff_cls, 0, 255)].copy()
        cost = np.where(passable_under, cost + 20, cost)
        cost = np.where(stepf & DRIVABLE[np.clip(eff_cls, 0, 255)], np.maximum(cost, 180), cost)
        cost = np.where(stepf & (cost < 180), np.maximum(cost, 140), cost)
        cost = np.where(dep, np.maximum(cost, 170), cost)
        rough = s.rough.astype(np.float32)
        cost = np.where(np.isfinite(rough) & (rough > 0.04), cost + 30, cost)
        cost = np.where(s.conf < 150, cost + 25, cost)
        stale = (s.age != UNKNOWN) & (s.age > 20)
        cost = np.where(stale, cost + 20, cost)
        cost = np.clip(cost, 0, 254)
        cost = np.where(cls == UNKNOWN, UNKNOWN, cost)
        s.cost[:] = cost.astype(np.uint8)
        s.flags[:] = (s.flags & 0xF0) | flags
        s.eff_cls = np.where(cls == UNKNOWN, UNKNOWN, eff_cls).astype(np.uint8)

    # ------------------------------------------------------------ geometry
    def cell_centres(self, k):
        """World (x, y) of every cell centre in tier k, shape (n, n, 2)."""
        t, org = self.tiers[k], self.origins[k]
        ii = (org[0] + np.arange(t.n) + 0.5) * t.cell
        jj = (org[1] + np.arange(t.n) + 0.5) * t.cell
        return np.stack(np.meshgrid(ii, jj, indexing="ij"), -1)

    def inner_mask(self, k):
        """Cells of tier k that lie inside tier k-1's window (covered at finer resolution)."""
        if k == 0:
            return np.zeros((self.tiers[0].n,) * 2, bool)
        tp, op = self.tiers[k - 1], self.origins[k - 1]
        t, o = self.tiers[k], self.origins[k]
        r = t.ratio // tp.ratio
        lo = op // r - o
        hi = lo + tp.n // r
        m = np.zeros((t.n, t.n), bool)
        m[lo[0]:hi[0], lo[1]:hi[1]] = True
        return m
