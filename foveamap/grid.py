"""Variable-resolution (foveated) 2.5D grid engine (NumPy reference).

Design (see Architecture Vision, sections 5-7 & PRD):

* A stack of concentric, square, world-aligned tiers. Default "spec" profile:
      tier 0: 5 cm cells, +/-10 m      (400 x 400)
      tier 1: 50 cm cells, +/-100 m    (400 x 400)
* Every tier's cell size is an integer multiple of the finest cell, and all
  tiers share the world origin. A point's fine index f = floor(x / 0.05) is
  computed ONCE in integer space; its index in tier k is f // ratio_k.
  Floor-division of integers is exact, so a coarse cell is always an exact
  union of fine cells -> no misalignment and no double counting.
* Tier windows are snapped to the coarsest lattice, so the fine box moves in
  whole coarse steps and never splits a coarse cell.
* Authoritative Tier Assignment:
  Every valid point is assigned to EXACTLY ONE native tier based on distance /
  tier window boundaries.
* True Mip-Up:
  Fine tier cells aggregate upward into parent coarse cells on the exact
  integer lattice, ensuring tier 1 is complete everywhere with zero double-counting
  and no tier boundary seams.
* Static / Dynamic Separation:
  Static layers persist and integrate with confidence EMA.
  The dynamic layer is rebuilt fresh every frame and never leaves ghost trails.

Persistent record per cell = 16 bytes (structure-of-arrays):
    count u16 | z_min f16 | z_max f16 | ground f16 | rough f16 |
    cls u8 | conf u8 (upper nibble primary conf, lower nibble secondary conf) |
    flags u8 (bits0-3 terrain flags, bits4-7 secondary-evidence class, 0xF=none) |
    clear u8 (2 cm units) | cost u8 | age u8

Secondary-evidence semantics (flags upper nibble + conf lower nibble):
    The pair stores the most relevant non-dominant class with its own matching
    confidence. A distinct ground class (road/sidewalk/parking/terrain) is
    preferred when confidently present (overhang/underpass case); otherwise the
    fused runner-up (top-2) is stored. The confidence nibble always belongs to
    the stored class. Effective class (passable-under) uses the stored class
    only when it is a ground class and clearance >= vehicle_clearance_m.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from scipy.ndimage import uniform_filter

from .core.ontology import (
    NUM_CLASSES,
    GROUND_CLASSES,
    DRIVABLE_CLASSES,
    DYNAMIC_CLASSES,
    ROAD,
    PARKING,
    SIDEWALK,
    TERRAIN,
    VEGETATION,
    BUILDING,
    POLE,
    VEHICLE,
    PERSON,
)
from .core.config import GridConfig, TerrainConfig, DynamicConfig, DEFAULT_COST_PRIOR
from .temporal import (
    DynamicWorldModel,
    DynamicObservation,
    ACTIVE_DYNAMIC as DYN_ACTIVE,
    TEMPORARILY_MISSING as DYN_MISSING,
    STALE as DYN_STALE,
)
from .terrain import slope_at_cell, report_layers, is_traversable_cell

UNKNOWN = 255
F_SLOPE, F_OVERHANG, F_STEP, F_DEPRESSION = 1, 2, 4, 8
F_DYNAMIC = 1  # Alias for backward compatibility if referenced

# Traversability prior per class (0 free ... 254 lethal)
COST_PRIOR = np.array(DEFAULT_COST_PRIOR, dtype=np.int32)
GROUND_MASK = np.zeros(NUM_CLASSES, bool)
GROUND_MASK[list(GROUND_CLASSES)] = True
DRIVABLE = np.zeros(256, bool)
DRIVABLE[list(DRIVABLE_CLASSES)] = True

VEHICLE_CLEARANCE = 2.5      # m: an obstacle higher than this above ground can be driven under
STEP_THRESH = 0.08           # m: curb / pothole edge
DEPRESSION_THRESH = 0.05     # m below the local drivable surface
DEPRESSION_WIN = 2.5         # m, window for the local drivable-surface reference
SLOPE_THRESH = 0.25          # rad (~14.3 deg)
SLOPE_CRIT = 0.40            # rad (~22.9 deg)


def _query_slope(s: TierLayers, i: int, j: int, cell_m: float) -> float | None:
    """On-demand slope (radians) for query output; None when unknown.

    Derived from stored ground elevation with the same neighbor policy as
    ``_derive``. Adds no persistent state (see memory model).
    """
    return slope_at_cell(
        np.asarray(s.ground, dtype=np.float64),
        np.isfinite(np.asarray(s.ground, dtype=np.float64)),
        int(i),
        int(j),
        float(cell_m),
    )


@dataclass
class Tier:
    cell: float       # metres
    half: float       # half-extent, metres
    ratio: int        # cell / finest cell (integer)
    n: int            # cells per side
    alpha: float      # temporal EMA weight for new observations


def _build_profiles() -> dict[str, tuple[tuple[float, float], ...]]:
    presets = ("spec", "graded", "uniform5", "uniform50")
    out = {}
    for name in presets:
        cfg = GridConfig.from_preset(name)
        out[name] = tuple((t.cell_size_m, t.half_extent_m) for t in cfg.tiers)
    return out


PROFILES = _build_profiles()


class TierLayers:
    """The 16-byte-per-cell persistent state of one tier."""
    FIELDS = dict(count=np.uint16, z_min=np.float16, z_max=np.float16, ground=np.float16,
                  rough=np.float16, cls=np.uint8, conf=np.uint8, flags=np.uint8,
                  clear=np.uint8, cost=np.uint8, age=np.uint8)

    def __init__(self, n: int, cell_size_m: float = 0.05, half_extent_m: float = 10.0):
        self.n = n
        self.cell = float(cell_size_m)
        self.half = float(half_extent_m)
        self.r = self.cell
        self.count = np.zeros((n, n), np.uint16)
        self.z_min = np.full((n, n), np.nan, np.float16)
        self.z_max = np.full((n, n), np.nan, np.float16)
        self.ground = np.full((n, n), np.nan, np.float16)
        self.rough = np.full((n, n), np.nan, np.float16)
        self.cls = np.full((n, n), UNKNOWN, np.uint8)
        self.conf = np.zeros((n, n), np.uint8)
        self.flags = np.full((n, n), 0xF0, np.uint8)      # upper nibble 0xF = unknown sec/ground class
        self.clear = np.full((n, n), UNKNOWN, np.uint8)
        self.cost = np.full((n, n), UNKNOWN, np.uint8)
        self.age = np.full((n, n), UNKNOWN, np.uint8)
        self.dynamic_mask = np.zeros((n, n), bool)
        self.free_passes = np.zeros((n, n), np.uint8)
        self._eff_cls: np.ndarray | None = None

    @property
    def secondary_class(self) -> np.ndarray:
        """Secondary evidence class ID (0..8, or UNKNOWN=255 for none).

        Upper nibble of ``flags`` stores 0xF when no secondary evidence exists;
        this accessor maps that sentinel to UNKNOWN for parity with the Torch
        engine and the query API.
        """
        nid = (self.flags >> 4).astype(np.uint8)
        return np.where(nid == 0xF, UNKNOWN, nid).astype(np.uint8)

    @property
    def secondary_confidence(self) -> np.ndarray:
        """Secondary evidence confidence (0.0 to 1.0) stored in conf lower nibble.

        This confidence always belongs to :meth:`secondary_class`: when the
        stored nibble prefers a distinct ground class, the confidence is the
        ground-class confidence, otherwise it is the runner-up confidence.
        """
        return ((self.conf & 0x0F).astype(np.float32) / 15.0)

    @property
    def primary_confidence(self) -> np.ndarray:
        """Primary confidence (0.0 to 1.0) stored in conf upper nibble."""
        return ((self.conf >> 4).astype(np.float32) / 15.0)

    @property
    def dynamic(self) -> np.ndarray:
        """Active dynamic observation mask for current frame."""
        return self.dynamic_mask

    @property
    def nbytes(self) -> int:
        """Total bytes for all persistent fields in this tier (exactly 16 bytes per cell)."""
        return sum(getattr(self, k).nbytes for k in self.FIELDS)

    @property
    def aux_nbytes(self) -> int:
        """Total bytes for transient/auxiliary buffers (dynamic mask, free-space streak, cached views)."""
        b = self.dynamic_mask.nbytes + self.free_passes.nbytes
        if self._eff_cls is not None:
            b += self._eff_cls.nbytes
        return b

    @property
    def eff_cls(self) -> np.ndarray:
        """Effective class considering overhang clearance (passable under)."""
        if self._eff_cls is not None:
            return self._eff_cls
        cls = self.cls.astype(np.int32)
        gcls = (self.flags >> 4).astype(np.int32)
        valid_g = np.isfinite(self.ground)
        clear_m = np.where(self.clear != UNKNOWN, self.clear * 0.02, np.nan)
        overhang = np.isfinite(clear_m) & (clear_m > 0.5) & valid_g
        is_ground = np.isin(gcls, list(GROUND_CLASSES))
        passable_under = overhang & (clear_m >= VEHICLE_CLEARANCE) & is_ground
        eff = np.where(passable_under, gcls, cls)
        return np.where(cls == UNKNOWN, UNKNOWN, eff).astype(np.uint8)

    @eff_cls.setter
    def eff_cls(self, val: np.ndarray) -> None:
        self._eff_cls = val

    @property
    def min_z(self) -> np.ndarray:
        return self.z_min

    @property
    def max_z(self) -> np.ndarray:
        return self.z_max

    @property
    def roughness(self) -> np.ndarray:
        return self.rough

    @property
    def clearance(self) -> np.ndarray:
        return self.clear

    def copy(self) -> TierLayers:
        out = TierLayers(self.n, cell_size_m=self.cell, half_extent_m=self.half)
        for f in self.FIELDS:
            getattr(out, f)[:] = getattr(self, f)
        out.dynamic_mask[:] = self.dynamic_mask
        out.free_passes[:] = self.free_passes
        if self._eff_cls is not None:
            out._eff_cls = self._eff_cls.copy()
        return out

    def shifted(self, d: Sequence[int]) -> TierLayers:
        """Copy of this state moved by d = (di, dj) cells (window scroll)."""
        out = TierLayers(self.n, cell_size_m=self.cell, half_extent_m=self.half)
        di, dj = int(d[0]), int(d[1])
        n = self.n
        if abs(di) >= n or abs(dj) >= n:
            return out
        src = (slice(max(di, 0), n + min(di, 0)), slice(max(dj, 0), n + min(dj, 0)))
        dst = (slice(max(-di, 0), n + min(-di, 0)), slice(max(-dj, 0), n + min(-dj, 0)))
        for f in self.FIELDS:
            getattr(out, f)[dst] = getattr(self, f)[src]
        out.dynamic_mask[dst] = self.dynamic_mask[src]
        out.free_passes[dst] = self.free_passes[src]
        if self._eff_cls is not None:
            out._eff_cls = np.full((n, n), UNKNOWN, np.uint8)
            out._eff_cls[dst] = self._eff_cls[src]
        return out


class FoveatedGrid:
    def __init__(self, profile: str | GridConfig | Sequence[tuple[float, float]] = "spec",
                 fuse: bool = True, terrain_config: TerrainConfig | None = None,
                 dynamic_config: DynamicConfig | None = None):
        if isinstance(profile, GridConfig):
            spec = tuple((t.cell_size_m, t.half_extent_m) for t in profile.tiers)
            fuse = profile.fuse
            alphas = tuple(t.alpha for t in profile.tiers)
        elif isinstance(profile, str):
            if profile in PROFILES:
                spec = PROFILES[profile]
            else:
                cfg = GridConfig.from_preset(profile)
                spec = tuple((t.cell_size_m, t.half_extent_m) for t in cfg.tiers)
            alphas = None
        else:
            spec = profile
            alphas = None

        base, coarse = spec[0][0], spec[-1][0]
        self.base, self.coarse = float(base), float(coarse)
        self.tiers: list[Tier] = []
        for k, (cell, half) in enumerate(spec):
            ratio = round(cell / base)
            assert abs(ratio * base - cell) < 1e-9, "tier cell must be an integer multiple of the finest cell"
            assert abs(round(coarse / cell) * cell - coarse) < 1e-9, "coarsest cell must be a multiple of every tier cell"
            assert abs(round(2 * half / coarse) * coarse - 2 * half) < 1e-9, "tier extent must be whole coarse cells"
            if k:
                assert cell > spec[k - 1][0] and half > spec[k - 1][1]
            alpha = alphas[k] if alphas is not None else (0.3 if k == 0 else 0.5)
            self.tiers.append(Tier(float(cell), float(half), int(ratio), round(2 * half / cell), float(alpha)))
        self.coarse_ratio = round(coarse / base)
        self.fuse = fuse
        self.terrain = terrain_config if terrain_config is not None else TerrainConfig()
        self.state = [TierLayers(t.n, cell_size_m=t.cell, half_extent_m=t.half) for t in self.tiers]
        self.origins: list[np.ndarray] | None = None
        self.last_diagnostics: dict[str, Any] = {}
        # Phase 6 temporal dynamic world model (bounded lifecycle over the
        # per-frame dynamic observation masks; never writes static arrays).
        self.dynamic_config = dynamic_config if dynamic_config is not None else DynamicConfig()
        self.temporal = DynamicWorldModel(self.dynamic_config)
        self.frame_index = -1
        self._last_timestamp: float | None = None

    # ------------------------------------------------------------------ utils
    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.state)

    @property
    def n_cells(self) -> int:
        return sum(t.n * t.n for t in self.tiers)

    def window_origins(self, ego_xy: Sequence[float] | np.ndarray) -> list[np.ndarray]:
        """Integer origin (in each tier's own cell units) of every tier window."""
        ec = np.floor(np.asarray(ego_xy, np.float64) / self.coarse).astype(np.int64)
        out = []
        for t in self.tiers:
            half_coarse = round(t.half / self.coarse)
            out.append((ec - half_coarse) * round(self.coarse / t.cell))
        return out

    def fine_index(self, xy_world: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.floor(np.asarray(xy_world, np.float64) / self.base).astype(np.int64)

    # ------------------------------------------------ authoritative tier assign
    def assign_native_tiers(self, xy_world: np.ndarray, origins: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
        """Assign each point to exactly one native tier based on concentric bounds.

        Invariants:
        1. Points falling in tier 0 window are assigned to tier 0.
        2. Points falling in tier k window but outside finer tiers are assigned to tier k.
        3. Points outside all windows are marked -1 (filtered).
        4. points_in == native_points_assigned + filtered_points.
        """
        n_pts = len(xy_world)
        if n_pts == 0:
            diag = {"points_in": 0, "native_points_assigned": 0, "filtered_points": 0,
                    "native_counts_per_tier": [0] * len(self.tiers)}
            return np.zeros((0,), dtype=np.int32), diag

        f = self.fine_index(xy_world)
        native_tiers = np.full(n_pts, -1, dtype=np.int32)
        unassigned = np.ones(n_pts, dtype=bool)

        for k, (t, org) in enumerate(zip(self.tiers, origins)):
            ij = f // t.ratio - org
            in_tier = (ij[:, 0] >= 0) & (ij[:, 0] < t.n) & (ij[:, 1] >= 0) & (ij[:, 1] < t.n)
            assign_mask = unassigned & in_tier
            native_tiers[assign_mask] = k
            unassigned[assign_mask] = False

        n_assigned = int((native_tiers >= 0).sum())
        n_filtered = int((native_tiers == -1).sum())
        diag = {
            "points_in": n_pts,
            "native_points_assigned": n_assigned,
            "filtered_points": n_filtered,
            "native_counts_per_tier": [int((native_tiers == k).sum()) for k in range(len(self.tiers))],
        }
        assert n_pts == n_assigned + n_filtered, f"Invariant violated: {n_pts} != {n_assigned} + {n_filtered}"
        self.last_diagnostics = diag
        return native_tiers, diag

    # -------------------------------------------------------------- binning
    def bin_points(self, xy_world: np.ndarray, z: np.ndarray, probs: np.ndarray,
                   moving: np.ndarray, origins: list[np.ndarray]) -> list[dict[str, Any]]:
        """Authoritative single-native binning + exact integer-lattice mip-up.

        Each classified point is assigned to exactly one native tier.
        Coarse tiers aggregate child cells from finer tiers via exact integer-lattice mip-up.
        """
        n_pts = len(xy_world)
        if n_pts == 0:
            return [
                dict(
                    key=np.zeros((0,), dtype=np.int64),
                    n_pts=np.zeros((0,), dtype=np.int64),
                    n_static=np.zeros((0,), dtype=np.float64),
                    n_ground=np.zeros((0,), dtype=np.float64),
                    z_min=np.zeros((0,), dtype=np.float32),
                    z_max=np.zeros((0,), dtype=np.float32),
                    ground=np.zeros((0,), dtype=np.float32),
                    rough=np.zeros((0,), dtype=np.float32),
                    zmin_ng=np.zeros((0,), dtype=np.float32),
                    p_static=np.zeros((0, probs.shape[1] if probs.ndim == 2 else NUM_CLASSES), dtype=np.float64),
                    p_all=np.zeros((0, probs.shape[1] if probs.ndim == 2 else NUM_CLASSES), dtype=np.float64),
                    n_dyn=np.zeros((0,), dtype=np.float64),
                    n_dyn_person=np.zeros((0,), dtype=np.float64),
                    zmin_dyn=np.zeros((0,), dtype=np.float32),
                    zmax_dyn=np.zeros((0,), dtype=np.float32),
                    n_in=0,
                )
                for _ in self.tiers
            ]

        f = self.fine_index(xy_world)
        probs_arr = np.asarray(probs, dtype=np.float64)
        cls = probs_arr.argmax(1)
        is_ground = GROUND_MASK[cls]
        moving_arr = np.asarray(moving, dtype=bool)
        static = ~moving_arr

        native_tiers, _ = self.assign_native_tiers(xy_world, origins)

        # 1. Native binning for each tier (non-overlapping point assignments)
        native_stats = []
        for k, (t, org) in enumerate(zip(self.tiers, origins)):
            idx = np.nonzero(native_tiers == k)[0]
            if len(idx) == 0:
                native_stats.append(dict(
                    key=np.zeros(0, np.int64), n_pts=np.zeros(0, np.int64),
                    n_static=np.zeros(0, np.float64), n_ground=np.zeros(0, np.float64),
                    z_min=np.zeros(0, np.float32), z_max=np.zeros(0, np.float32),
                    ground=np.zeros(0, np.float32), rough=np.zeros(0, np.float32),
                    zmin_ng=np.zeros(0, np.float32),
                    p_static=np.zeros((0, probs_arr.shape[1]), np.float64),
                    p_all=np.zeros((0, probs_arr.shape[1]), np.float64),
                    n_dyn=np.zeros(0, np.float64), n_dyn_person=np.zeros(0, np.float64),
                    zmin_dyn=np.zeros(0, np.float32), zmax_dyn=np.zeros(0, np.float32),
                    n_in=0,
                ))
                continue

            ij = f[idx] // t.ratio - org
            key = ij[:, 0] * t.n + ij[:, 1]
            uk, inv = np.unique(key, return_inverse=True)
            M = len(uk)
            zz, st, gg = z[idx], static[idx], is_ground[idx] & static[idx]
            order = np.argsort(inv, kind="stable")
            starts = np.searchsorted(inv[order], np.arange(M))

            def rmin(v: np.ndarray) -> np.ndarray:
                return np.minimum.reduceat(v[order], starts) if M else np.zeros(0, np.float32)

            def rmax(v: np.ndarray) -> np.ndarray:
                return np.maximum.reduceat(v[order], starts) if M else np.zeros(0, np.float32)

            bc = lambda w: np.bincount(inv, weights=w, minlength=M)  # noqa: E731
            n_static = bc(st.astype(np.float64))
            n_ground = bc(gg.astype(np.float64))
            gsum, gsq = bc(np.where(gg, zz, 0.0)), bc(np.where(gg, zz * zz, 0.0))
            P = probs_arr[idx]
            p_static = np.stack([bc(P[:, c] * st) for c in range(P.shape[1])], 1)
            p_all = np.stack([bc(P[:, c]) for c in range(P.shape[1])], 1)
            mv = moving_arr[idx]
            c_mv = cls[idx]

            native_stats.append(dict(
                key=uk,
                n_pts=np.bincount(inv, minlength=M).astype(np.int64),
                n_static=n_static,
                n_ground=n_ground,
                z_min=rmin(np.where(st, zz, np.inf)),
                z_max=rmax(np.where(st, zz, -np.inf)),
                ground=np.where(n_ground > 0, gsum / np.maximum(n_ground, 1), np.nan),
                rough=np.where(n_ground > 1, np.sqrt(np.maximum(gsq / np.maximum(n_ground, 1) - (gsum / np.maximum(n_ground, 1)) ** 2, 0)), np.nan),
                zmin_ng=rmin(np.where(st & ~is_ground[idx], zz, np.inf)),
                p_static=p_static,
                p_all=p_all,
                n_dyn=bc(mv.astype(np.float64)),
                n_dyn_person=bc((mv & (c_mv == PERSON)).astype(np.float64)),
                zmin_dyn=rmin(np.where(mv, zz, np.inf)),
                zmax_dyn=rmax(np.where(mv, zz, -np.inf)),
                n_in=len(idx),
            ))

        # 2. Integer-lattice Mip-Up: aggregate child cells into parent tiers
        fused_stats = list(native_stats)
        for k in range(len(self.tiers) - 1):
            child_st = fused_stats[k]
            child_t = self.tiers[k]
            parent_t = self.tiers[k + 1]
            child_org = origins[k]
            parent_org = origins[k + 1]
            fused_stats[k + 1] = self._mip_up_tier(child_st, child_t, parent_t, child_org, parent_org, fused_stats[k + 1])

        return fused_stats

    def _mip_up_tier(self, child_st: dict[str, Any], child_t: Tier, parent_t: Tier,
                     child_org: np.ndarray, parent_org: np.ndarray,
                     parent_native: dict[str, Any]) -> dict[str, Any]:
        """Aggregate child tier cells into parent tier on exact integer lattice."""
        k_c = child_st["key"]
        if len(k_c) == 0:
            return parent_native

        r = parent_t.ratio // child_t.ratio
        fi = (k_c // child_t.n) + child_org[0]
        fj = (k_c % child_t.n) + child_org[1]
        pi = fi // r - parent_org[0]
        pj = fj // r - parent_org[1]
        k_p = pi * parent_t.n + pj

        uk_p, inv_p = np.unique(k_p, return_inverse=True)
        M_p = len(uk_p)
        order_p = np.argsort(inv_p, kind="stable")
        starts_p = np.searchsorted(inv_p[order_p], np.arange(M_p))

        def rmin_p(v: np.ndarray) -> np.ndarray:
            return np.minimum.reduceat(v[order_p], starts_p) if M_p else np.zeros(0, np.float32)

        def rmax_p(v: np.ndarray) -> np.ndarray:
            return np.maximum.reduceat(v[order_p], starts_p) if M_p else np.zeros(0, np.float32)

        bc_p = lambda w: np.bincount(inv_p, weights=w, minlength=M_p)  # noqa: E731

        n_pts_p = bc_p(child_st["n_pts"]).astype(np.int64)
        n_static_p = bc_p(child_st["n_static"])
        cg = child_st["ground"]
        c_valid = np.isfinite(cg)
        ng_c = child_st["n_ground"]

        ng_p = bc_p(np.where(c_valid, ng_c, 0.0))
        g_sum_p = bc_p(np.where(c_valid, cg * ng_c, 0.0))
        ground_p = np.where(ng_p > 0, g_sum_p / np.maximum(ng_p, 1), np.nan)

        cr = np.nan_to_num(child_st["rough"], nan=0.0)
        g_diff = cg - ground_p[inv_p]
        var_sum_p = bc_p(np.where(c_valid, ng_c * (cr ** 2 + g_diff ** 2), 0.0))
        var_p = np.where(ng_p > 1, np.maximum(0.0, var_sum_p / np.maximum(ng_p, 1)), np.nan)
        rough_p = np.where(np.isfinite(var_p), np.sqrt(var_p), np.nan)

        z_min_p = rmin_p(child_st["z_min"])
        z_max_p = rmax_p(child_st["z_max"])
        zmin_ng_p = rmin_p(child_st["zmin_ng"])

        p_static_p = np.stack([bc_p(child_st["p_static"][:, c]) for c in range(child_st["p_static"].shape[1])], 1)
        p_all_p = np.stack([bc_p(child_st["p_all"][:, c]) for c in range(child_st["p_all"].shape[1])], 1)
        n_dyn_p = bc_p(child_st["n_dyn"])
        n_dyn_person_p = bc_p(child_st["n_dyn_person"])
        zmin_dyn_p = rmin_p(child_st.get("zmin_dyn", np.full(len(k_c), np.inf, np.float32)))
        zmax_dyn_p = rmax_p(child_st.get("zmax_dyn", np.full(len(k_c), -np.inf, np.float32)))

        # Combine with parent native (disjoint sets due to coarse-cell snapped boundary)
        if len(parent_native["key"]) == 0:
            return dict(
                key=uk_p, n_pts=n_pts_p, n_static=n_static_p, n_ground=ng_p,
                z_min=z_min_p, z_max=z_max_p, ground=ground_p, rough=rough_p,
                zmin_ng=zmin_ng_p, p_static=p_static_p, p_all=p_all_p,
                n_dyn=n_dyn_p, n_dyn_person=n_dyn_person_p,
                zmin_dyn=zmin_dyn_p, zmax_dyn=zmax_dyn_p,
                n_in=int(n_pts_p.sum()),
            )

        all_keys = np.concatenate([parent_native["key"], uk_p])
        order = np.argsort(all_keys)

        return dict(
            key=all_keys[order],
            n_pts=np.concatenate([parent_native["n_pts"], n_pts_p])[order],
            n_static=np.concatenate([parent_native["n_static"], n_static_p])[order],
            n_ground=np.concatenate([parent_native["n_ground"], ng_p])[order],
            z_min=np.concatenate([parent_native["z_min"], z_min_p])[order],
            z_max=np.concatenate([parent_native["z_max"], z_max_p])[order],
            ground=np.concatenate([parent_native["ground"], ground_p])[order],
            rough=np.concatenate([parent_native["rough"], rough_p])[order],
            zmin_ng=np.concatenate([parent_native["zmin_ng"], zmin_ng_p])[order],
            p_static=np.concatenate([parent_native["p_static"], p_static_p], axis=0)[order],
            p_all=np.concatenate([parent_native["p_all"], p_all_p], axis=0)[order],
            n_dyn=np.concatenate([parent_native["n_dyn"], n_dyn_p])[order],
            n_dyn_person=np.concatenate([parent_native["n_dyn_person"], n_dyn_person_p])[order],
            zmin_dyn=np.concatenate([parent_native.get("zmin_dyn", np.full(len(parent_native["key"]), np.inf)), zmin_dyn_p])[order],
            zmax_dyn=np.concatenate([parent_native.get("zmax_dyn", np.full(len(parent_native["key"]), -np.inf)), zmax_dyn_p])[order],
            n_in=parent_native["n_in"] + int(n_pts_p.sum()),
        )

    # --------------------------------------------------------------- update
    def update(self, xy_world: np.ndarray, z: np.ndarray, probs: np.ndarray,
               moving: np.ndarray, ego_xy: Sequence[float] | np.ndarray,
               sensor_origin: Sequence[float] | np.ndarray | None = None,
               timestamp: float | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Process one frame. Returns (dynamic_cells_per_tier, frame_stats_per_tier)."""
        origins = self.window_origins(ego_xy)
        stats = self.bin_points(xy_world, z, probs, moving, origins)
        dyn = self.fuse_stats(stats, origins, sensor_origin=sensor_origin, timestamp=timestamp)
        return dyn, stats

    def _new_layers(self, t: Tier) -> TierLayers:
        return TierLayers(t.n, cell_size_m=t.cell, half_extent_m=t.half)

    def snapshot(self) -> list[TierLayers]:
        """Host copy of every tier's state, safe to read while the grid keeps updating."""
        return [s.copy() for s in self.state]

    def fuse_stats(self, stats: list[dict[str, Any]], origins: list[np.ndarray],
                   sensor_origin: Sequence[float] | np.ndarray | None = None,
                   timestamp: float | None = None) -> list[dict[str, Any]]:
        """Scroll state to origins and fuse stats. Returns structured dynamic observations.

        Temporal order per frame: scroll (ego compensation) -> static fusion
        -> dynamic lifecycle update -> terrain derivation -> ray clearing.
        ``timestamp`` feeds world-frame velocity evidence; when omitted, a
        deterministic 10 Hz clock derived from the frame counter is used.
        """
        if self.origins is not None and self.fuse:
            deltas = [tuple(int(a) - int(b) for a, b in zip(o, oo)) for o, oo in zip(origins, self.origins)]
            self.state = [s.shifted(o - oo) for s, o, oo in zip(self.state, origins, self.origins)]
            self.temporal.on_scroll(deltas, [t.n for t in self.tiers])
        elif not self.fuse:
            self.state = [self._new_layers(t) for t in self.tiers]
            # Unfused operation is frame-independent: no cross-frame lifecycle.
            self.temporal.reset()
        self.origins = origins
        dyn = []
        for t, s, st in zip(self.tiers, self.state, stats):
            dyn.append(self._fuse_tier(t, s, st))
            self._derive(t, s)

        # Phase 6 temporal dynamic update from frame-isolated observations.
        # Static arrays are untouched here; only the bounded lifecycle store
        # advances. Torch dyn dicts carry device tensors: the sparse
        # host conversion below is an intentional per-tier boundary over
        # dynamic cells only (never per-point data).
        self.frame_index += 1
        if timestamp is None:
            ts = (self._last_timestamp + 0.1) if self._last_timestamp is not None else float(self.frame_index) * 0.1
        else:
            ts = float(timestamp)
        self._last_timestamp = ts
        if self.fuse:
            self.temporal.update(
                self._dynamic_observations(dyn, origins),
                frame_idx=self.frame_index,
                timestamp=ts,
            )

        # Conservative ray clearing if enabled and sensor origin provided
        if getattr(self.terrain, "enable_ray_clearing", False) and sensor_origin is not None:
            self._clear_rays(sensor_origin, stats)

        return dyn

    def _dynamic_observations(
        self, dyn: list[dict[str, Any]], origins: list[np.ndarray]
    ) -> list[DynamicObservation]:
        """Convert per-tier dynamic observation dicts to world-frame evidence.

        Accepts both NumPy arrays (NumPy engine) and torch tensors (Torch
        engine; converted sparsely to host here as a documented boundary).
        """
        import torch as _torch

        obs: list[DynamicObservation] = []
        for k, (t, d, org) in enumerate(zip(self.tiers, dyn, origins)):
            ii = d.get("i")
            jj = d.get("j")
            if ii is None or jj is None or len(ii) == 0:
                continue
            if _torch.is_tensor(ii):
                # Intentional sparse host boundary: dynamic cells only.
                ii_h = ii.detach().cpu().numpy().ravel()
                jj_h = jj.detach().cpu().numpy().ravel()
                cls_h = d["cls"].detach().cpu().numpy().ravel()
                conf_h = (d["conf"].detach().cpu().numpy().ravel().astype(np.float64) / 255.0)
                cnt_h = d["count"].detach().cpu().numpy().ravel()
            else:
                ii_h = np.asarray(ii).ravel()
                jj_h = np.asarray(jj).ravel()
                cls_h = np.asarray(d["cls"]).ravel()
                conf_h = np.asarray(d["conf"]).ravel().astype(np.float64) / 255.0
                cnt_h = np.asarray(d["count"]).ravel()
            org_h = np.asarray(org)
            for ci, cj, cc, cf, cn in zip(ii_h, jj_h, cls_h, conf_h, cnt_h):
                ci_i, cj_i = int(ci), int(cj)
                obs.append(DynamicObservation(
                    tier=k,
                    i=ci_i,
                    j=cj_i,
                    x=float((int(org_h[0]) + ci_i + 0.5) * t.cell),
                    y=float((int(org_h[1]) + cj_i + 0.5) * t.cell),
                    cls=int(cc),
                    conf=float(min(1.0, max(0.0, cf))),
                    count=int(cn),
                ))
        return obs

    def _fuse_tier(self, t: Tier, s: TierLayers, st: dict[str, Any]) -> dict[str, Any]:
        n = t.n
        key = st["key"]
        obs = st["n_static"] > 0
        k = key[obs]
        i, j = k // n, k % n

        # Reset free passes for observed cells
        s.free_passes[i, j] = 0

        # ---- class: blend new probabilities with stored (cls, conf)
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
        q_sum = np.maximum(q.sum(1), 1e-9)
        # Persistent secondary evidence (16-byte cell: flags upper nibble + conf lower nibble).
        # Semantics (documented, unambiguous): the stored pair is the most relevant
        # non-dominant class with its own matching confidence. A distinct ground
        # class is preferred when confidently present (overhang/underpass case);
        # otherwise the fused runner-up (top-2) is stored. Confidence always
        # belongs to the stored class — never mixed.
        q_sec = q.copy()
        q_sec[np.arange(len(new_c)), new_c] = -1.0
        sec_c = q_sec.argmax(1)
        sec_conf_runner = np.maximum(q_sec.max(1), 0.0) / q_sum
        has_runner = sec_conf_runner > 0.02
        # Ground-class evidence from the fused distribution (stable across frames).
        pg = np.where(GROUND_MASK[None, :], q, 0.0)
        pg_sum = pg.sum(1)
        gcls = np.where(pg_sum > 0.05 * q_sum, pg.argmax(1), 0xF)
        gconf = np.where(pg_sum > 0.0, pg.max(1) / q_sum, 0.0)
        use_ground = (gcls != 0xF) & (gcls != new_c) & (gconf > 0.02)
        final_sec = np.where(use_ground, gcls, np.where(has_runner, sec_c, 0xF)).astype(np.uint8)
        final_sec_conf = np.where(use_ground, gconf, np.where(has_runner, sec_conf_runner, 0.0))
        has_sec = use_ground | has_runner
        sec_c_id = np.where(has_sec, final_sec, 0xF).astype(np.uint8)
        sec_conf_4bit = np.clip(np.round(final_sec_conf * 15.0), 0, 15).astype(np.uint8)
        prim_conf_4bit = np.clip(np.round((q.max(1) / q_sum) * 15.0), 0, 15).astype(np.uint8)
        s.conf[i, j] = (prim_conf_4bit << 4) | (sec_conf_4bit & 0x0F)

        # Ground/secondary nibble stores the secondary evidence class above.
        keep_sec = sec_c_id

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
        s.count[i, j] = np.minimum(st["n_static"][obs], 65535).astype(np.uint16)
        zng = st["zmin_ng"][obs]
        clear = np.where(np.isfinite(zng) & np.isfinite(g), zng - g, np.nan)
        s.clear[i, j] = np.where(np.isfinite(clear), np.clip(clear / 0.02, 0, 254), UNKNOWN).astype(np.uint8)
        s.flags[i, j] = (keep_sec << 4).astype(np.uint8)

        # ---- age & stale transitions
        age = s.age.astype(np.int32) + 1
        age[s.age == UNKNOWN] = UNKNOWN
        age[i, j] = 0

        # Cells unobserved beyond max_stale_age revert to UNKNOWN
        max_stale = self.terrain.max_stale_age if self.terrain is not None else 100
        stale_timeout = age >= max_stale
        s.cls[stale_timeout] = UNKNOWN
        s.conf[stale_timeout] = 0
        s.cost[stale_timeout] = UNKNOWN
        s.count[stale_timeout] = 0
        s.ground[stale_timeout] = np.nan
        s.z_min[stale_timeout] = np.nan
        s.z_max[stale_timeout] = np.nan
        s.rough[stale_timeout] = np.nan
        s.clear[stale_timeout] = UNKNOWN
        s.flags[stale_timeout] = 0xF0
        age[stale_timeout] = UNKNOWN
        s.age[:] = np.minimum(age, UNKNOWN).astype(np.uint8)

        # ---- dynamic layer: rebuilt this frame only, never fused into static state
        s.dynamic_mask.fill(False)
        dyn_cells = st["n_dyn"] > 0
        dk = key[dyn_cells]
        if len(dk) > 0:
            s.dynamic_mask[dk // n, dk % n] = True

        d_count = st["n_dyn"][dyn_cells].astype(np.uint32)
        d_person = st["n_dyn_person"][dyn_cells]
        d_cls = np.where(d_person * 2 > d_count, PERSON, VEHICLE).astype(np.uint8)
        d_conf = np.where(d_cls == PERSON, d_person / np.maximum(d_count, 1), (d_count - d_person) / np.maximum(d_count, 1))
        d_conf_u8 = np.clip(d_conf * 255.0, 0, 255).astype(np.uint8)

        zmin_d = st.get("zmin_dyn", np.full(len(key), np.nan, np.float32))[dyn_cells]
        zmax_d = st.get("zmax_dyn", np.full(len(key), np.nan, np.float32))[dyn_cells]

        return dict(
            i=dk // n,
            j=dk % n,
            cls=d_cls,
            conf=d_conf_u8,
            count=d_count,
            z_min=zmin_d,
            z_max=zmax_d,
        )

    def _derive(self, t: Tier, s: TierLayers) -> None:
        """Derive flags, slope, and traversability cost vectorised across tier."""
        veh_clearance = self.terrain.vehicle_clearance_m if self.terrain is not None else VEHICLE_CLEARANCE
        step_thresh = self.terrain.step_threshold_m if self.terrain is not None else STEP_THRESH
        dep_thresh = self.terrain.depression_threshold_m if self.terrain is not None else DEPRESSION_THRESH
        dep_win = self.terrain.depression_window_m if self.terrain is not None else DEPRESSION_WIN
        cost_prior_arr = np.asarray(self.terrain.cost_priors, dtype=np.int32) if self.terrain is not None else COST_PRIOR
        rough_thresh = self.terrain.roughness_threshold_m if self.terrain is not None else 0.04
        slope_thresh = self.terrain.slope_threshold_rad if self.terrain is not None else SLOPE_THRESH
        slope_crit = self.terrain.slope_critical_rad if self.terrain is not None else SLOPE_CRIT
        stale_thresh = self.terrain.stale_age_threshold if self.terrain is not None else 20

        cls = s.cls.astype(np.int32)
        gcls = (s.flags >> 4).astype(np.int32)
        ground = s.ground.astype(np.float32)
        valid_g = np.isfinite(ground)
        flags = np.zeros_like(s.flags)

        # 1. Overhang: obstacle points start well above ground
        clear_m = np.where(s.clear != UNKNOWN, s.clear * 0.02, np.nan)
        overhang = np.isfinite(clear_m) & (clear_m > 0.5) & valid_g
        flags |= np.where(overhang, F_OVERHANG, 0).astype(np.uint8)
        # Passable only when the stored secondary evidence is a ground class
        # (road/sidewalk/parking/terrain). A non-ground runner-up (pole,
        # vegetation, building) must never become the effective class.
        passable_under = overhang & (clear_m >= veh_clearance) & np.isin(gcls, list(GROUND_CLASSES))
        eff_cls = np.where(passable_under, gcls, cls)

        # 2. Step edges: height jump to neighbour 1 or 2 cells away
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
        stepf = step > step_thresh
        flags |= np.where(stepf & valid_g, F_STEP, 0).astype(np.uint8)

        # 3. Depressions (potholes): below local mean of drivable ground
        drv = DRIVABLE[np.clip(eff_cls, 0, 255)] & valid_g
        win = max(3, int(round(dep_win / t.cell)) | 1)
        num = uniform_filter(np.where(drv, gz, 0.0), win, mode="constant")
        den = uniform_filter(drv.astype(np.float32), win, mode="constant")
        ref = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)
        dep = drv & (gz < ref - dep_thresh)
        flags |= np.where(dep, F_DEPRESSION, 0).astype(np.uint8)

        # 4. 2.5D Slope estimation from ground gradient
        g_down = np.roll(gz, -1, axis=0)
        v_down = np.roll(valid_g, -1, axis=0)
        v_down[-1, :] = False

        g_up = np.roll(gz, 1, axis=0)
        v_up = np.roll(valid_g, 1, axis=0)
        v_up[0, :] = False

        dx_central = (g_down - g_up) / (2.0 * t.cell)
        dx_fwd = (g_down - gz) / t.cell
        dx_bwd = (gz - g_up) / t.cell
        dx = np.where(v_down & v_up, dx_central, np.where(v_down, dx_fwd, np.where(v_up, dx_bwd, 0.0)))

        g_right = np.roll(gz, -1, axis=1)
        v_right = np.roll(valid_g, -1, axis=1)
        v_right[:, -1] = False

        g_left = np.roll(gz, 1, axis=1)
        v_left = np.roll(valid_g, 1, axis=1)
        v_left[:, 0] = False

        dy_central = (g_right - g_left) / (2.0 * t.cell)
        dy_fwd = (g_right - gz) / t.cell
        dy_bwd = (gz - g_left) / t.cell
        dy = np.where(v_right & v_left, dy_central, np.where(v_right, dy_fwd, np.where(v_left, dy_bwd, 0.0)))

        has_slope = valid_g & ((v_down | v_up) | (v_right | v_left))
        slope_grad = np.sqrt(dx ** 2 + dy ** 2)
        slope_rad = np.where(has_slope, np.arctan(slope_grad), 0.0)
        slope_flag = has_slope & (slope_rad > slope_thresh)
        flags |= np.where(slope_flag, F_SLOPE, 0).astype(np.uint8)

        # 5. Traversability cost aggregation
        cost = cost_prior_arr[np.clip(eff_cls, 0, 255)].copy()
        cost = np.where(passable_under, cost + 20, cost)
        cost = np.where(stepf & DRIVABLE[np.clip(eff_cls, 0, 255)], np.maximum(cost, 180), cost)
        cost = np.where(stepf & (cost < 180), np.maximum(cost, 140), cost)
        cost = np.where(dep, np.maximum(cost, 170), cost)

        rough = s.rough.astype(np.float32)
        cost = np.where(np.isfinite(rough) & (rough > rough_thresh), cost + 30, cost)

        # Slope penalty
        slope_excess = np.clip((slope_rad - slope_thresh) / max(slope_crit - slope_thresh, 1e-4), 0.0, 1.0)
        cost = np.where(slope_flag, cost + np.round(slope_excess * 40).astype(np.int32), cost)
        cost = np.where(has_slope & (slope_rad >= slope_crit), np.maximum(cost, 220), cost)

        cost = np.where(s.conf < 150, cost + 25, cost)
        stale = (s.age != UNKNOWN) & (s.age >= stale_thresh)
        cost = np.where(stale, cost + 20, cost)
        cost = np.clip(cost, 0, 254)
        cost = np.where(cls == UNKNOWN, UNKNOWN, cost)

        s.cost[:] = cost.astype(np.uint8)
        s.flags[:] = (s.flags & 0xF0) | flags
        s._eff_cls = np.where(cls == UNKNOWN, UNKNOWN, eff_cls).astype(np.uint8)

    # ---------------------------------------------------- free space ray clearing
    def _clear_rays(self, sensor_origin: Sequence[float] | np.ndarray, stats: list[dict[str, Any]]) -> None:
        """Conservative 2.5D ray clearing along beams to obstacle returns."""
        so = np.asarray(sensor_origin, dtype=np.float64)
        free_thresh = self.terrain.free_clear_frames if self.terrain is not None else 3

        for tier_idx, (t, s, st) in enumerate(zip(self.tiers, self.state, stats)):
            k_obs = st["key"][st["n_static"] > 0]
            if len(k_obs) == 0:
                s.free_passes.fill(0)
                continue
            obs_set = set(k_obs.tolist())
            # Sample subset of obstacle rays for speed
            step_sz = max(1, len(k_obs) // 500)
            sample_keys = k_obs[::step_sz]
            ci = sample_keys // t.n
            cj = sample_keys % t.n
            org = self.origins[tier_idx]
            target_xy = (np.stack([ci, cj], 1) + org + 0.5) * t.cell

            vec = target_xy - so[:2]
            dist = np.hypot(vec[:, 0], vec[:, 1])
            valid = (dist > 1.0) & (dist < 80.0)
            if not valid.any():
                s.free_passes.fill(0)
                continue

            vec = vec[valid]
            dist = dist[valid]
            u = vec / dist[:, None]

            # Track cells traversed by free space rays in the current frame
            traversed_this_frame: set[tuple[int, int]] = set()

            # Ray sampling from 1m up to (dist - 1.5 * cell)
            for d_ray, u_ray in zip(dist, u):
                n_samples = max(1, int((d_ray - 1.5 * t.cell - 1.0) / t.cell))
                if n_samples <= 0:
                    continue
                samples = np.linspace(1.0, d_ray - 1.5 * t.cell, n_samples)
                pts_ray = so[:2] + samples[:, None] * u_ray
                ij_ray = np.floor(pts_ray / t.cell).astype(np.int64) - org
                in_w = (ij_ray[:, 0] >= 0) & (ij_ray[:, 0] < t.n) & (ij_ray[:, 1] >= 0) & (ij_ray[:, 1] < t.n)
                if not in_w.any():
                    continue
                cells = ij_ray[in_w]
                for c_i, c_j in cells:
                    c_key = c_i * t.n + c_j
                    if c_key in obs_set:
                        continue
                    traversed_this_frame.add((int(c_i), int(c_j)))

            # Consecutive frame requirement: cells with active free passes that were NOT traversed in this frame lose streak
            fp_nonzero_i, fp_nonzero_j = np.nonzero(s.free_passes)
            for f_i, f_j in zip(fp_nonzero_i, fp_nonzero_j):
                if (int(f_i), int(f_j)) not in traversed_this_frame:
                    s.free_passes[f_i, f_j] = 0

            # Increment consecutive free passes and clear obstacles reaching threshold
            for c_i, c_j in traversed_this_frame:
                s.free_passes[c_i, c_j] += 1
                if s.free_passes[c_i, c_j] >= free_thresh:
                    # Never clear dynamic obstacles, ground classes, or UNKNOWN.
                    if bool(s.dynamic_mask[c_i, c_j]):
                        s.free_passes[c_i, c_j] = 0
                        continue
                    if s.cls[c_i, c_j] != UNKNOWN and s.cls[c_i, c_j] not in GROUND_CLASSES:
                        s.cls[c_i, c_j] = UNKNOWN
                        s.conf[c_i, c_j] = 0
                        s.cost[c_i, c_j] = UNKNOWN
                        s.count[c_i, c_j] = 0
                        s.z_min[c_i, c_j] = np.nan
                        s.z_max[c_i, c_j] = np.nan
                        s.ground[c_i, c_j] = np.nan
                        s.rough[c_i, c_j] = np.nan
                        s.clear[c_i, c_j] = UNKNOWN
                        s.flags[c_i, c_j] = 0xF0
                        s.age[c_i, c_j] = UNKNOWN
                        s.free_passes[c_i, c_j] = 0

    # ------------------------------------------------------------ geometry
    def cell_centres(self, k: int) -> np.ndarray:
        """World (x, y) of every cell centre in tier k, shape (n, n, 2)."""
        t, org = self.tiers[k], self.origins[k]
        ii = (org[0] + np.arange(t.n) + 0.5) * t.cell
        jj = (org[1] + np.arange(t.n) + 0.5) * t.cell
        return np.stack(np.meshgrid(ii, jj, indexing="ij"), -1)

    def inner_mask(self, k: int) -> np.ndarray:
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

    # ------------------------------------------------------------ Query API
    def query_point(self, x: float, y: float) -> dict[str, Any]:
        """Query state at world coordinate (x, y) with finest available tier."""
        if self.origins is None:
            return {"tier": -1, "is_unknown": True, "is_traversable": False}

        f = self.fine_index(np.array([[x, y]]))
        for k, (t, org) in enumerate(zip(self.tiers, self.origins)):
            ij = f // t.ratio - org
            if 0 <= ij[0, 0] < t.n and 0 <= ij[0, 1] < t.n:
                i, j = int(ij[0, 0]), int(ij[0, 1])
                s = self.state[k]
                cls = int(s.cls[i, j])
                cost = int(s.cost[i, j])
                cnt = int(s.count[i, j])
                age = int(s.age[i, j])
                mask_dyn = bool(s.dynamic[i, j])
                # Phase 6: overlay the bounded temporal lifecycle. The mask is
                # frame-isolated observation; OBSERVED/ACTIVE/MISSING tracks
                # keep a region dynamically occupied after the observation
                # frame. Static arrays are only read here, never written.
                enrich = self.temporal.query_enrichment(k, i, j)
                temporal_occupied = bool(enrich["dynamic_occupied"]) if enrich else False
                dyn = mask_dyn or temporal_occupied
                dyn_state = enrich["dynamic_state"] if enrich else None
                if dyn:
                    if dyn_state == DYN_MISSING:
                        state = "TEMPORARILY_MISSING"
                    elif dyn_state == DYN_ACTIVE:
                        state = "ACTIVE_DYNAMIC"
                    else:
                        state = "OBSERVED_DYNAMIC"
                elif cnt == 0 or cls == UNKNOWN:
                    if enrich is not None and dyn_state == DYN_STALE:
                        # Recently disappeared dynamic object, no static geometry.
                        state = "STALE"
                    else:
                        state = "UNKNOWN"
                elif age >= self.terrain.stale_age_threshold:
                    state = "STALE"
                else:
                    # Fresh static geometry wins over an expired dynamic track.
                    state = "OBSERVED_STATIC"
                return {
                    "tier": k,
                    "cell_size_m": t.cell,
                    "cell_coord": (i, j),
                    "dominant_class": cls,
                    "primary_class": cls,
                    "cls": cls,
                    "conf": int(s.conf[i, j]),
                    "secondary_class": int(s.secondary_class[i, j]),
                    "secondary_confidence": float(s.secondary_confidence[i, j]),
                    "cost": cost,
                    "count": cnt,
                    "dynamic": dyn,
                    "state": state,
                    "ground": float(s.ground[i, j]) if np.isfinite(s.ground[i, j]) else None,
                    "z_min": float(s.z_min[i, j]) if np.isfinite(s.z_min[i, j]) else None,
                    "z_max": float(s.z_max[i, j]) if np.isfinite(s.z_max[i, j]) else None,
                    "rough": float(s.rough[i, j]) if np.isfinite(s.rough[i, j]) else None,
                    "slope_rad": _query_slope(s, i, j, t.cell),
                    "clear": float(s.clear[i, j] * 0.02) if s.clear[i, j] != UNKNOWN else None,
                    "age": age,
                    "flags": int(s.flags[i, j]),
                    "is_unknown": state == "UNKNOWN",
                    "is_traversable": is_traversable_cell(
                        cost, dyn, state != "UNKNOWN", self._trav_max()),
                    "dynamic_state": dyn_state,
                    "dynamic_confidence": float(enrich["dynamic_confidence"]) if enrich else 0.0,
                    "dynamic_age_frames": int(enrich["dynamic_age_frames"]) if enrich else 0,
                    "velocity": enrich["velocity"] if enrich else None,
                }
        return {"tier": -1, "state": "OUT_OF_BOUNDS", "is_unknown": True, "is_traversable": False}

    def _trav_max(self) -> int:
        """Authoritative traversal threshold from configuration."""
        trav = getattr(self.terrain, "traversable_cost_max", 180)
        try:
            return int(trav)
        except (TypeError, ValueError):
            return 180

    def is_traversable(self, x: float, y: float, max_cost: int | None = None) -> bool:
        """Check if world coordinate (x, y) is safely traversable.

        With ``max_cost=None`` (default) the authoritative configured
        threshold applies, agreeing with ``query_point()["is_traversable"]``.
        An explicit ``max_cost`` is a planner policy override.
        """
        q = self.query_point(x, y)
        if q.get("dynamic", False) or q["is_unknown"] or q.get("cost", UNKNOWN) == UNKNOWN:
            return False
        limit = self._trav_max() if max_cost is None else int(max_cost)
        return q.get("cost", 255) < limit

    def get_height(self, x: float, y: float) -> float | None:
        """Get best elevation estimate at world coordinate (x, y)."""
        q = self.query_point(x, y)
        if q.get("ground") is not None:
            return q["ground"]
        if q.get("z_min") is not None:
            return q["z_min"]
        return None

    def query_box(self, min_x: float, min_y: float, max_x: float, max_y: float,
                  tier_idx: int = 0) -> dict[str, Any]:
        """Query subgrid of cells covering bounding box [min_x, min_y, max_x, max_y]."""
        if self.origins is None or tier_idx < 0 or tier_idx >= len(self.tiers):
            return {"cells": np.zeros((0, 2), np.int64), "cost": np.zeros((0,), np.uint8)}
        t = self.tiers[tier_idx]
        org = self.origins[tier_idx]
        s = self.state[tier_idx]
        i0 = max(0, int(np.floor(min_x / t.cell)) - int(org[0]))
        i1 = min(t.n, int(np.floor(max_x / t.cell)) - int(org[0]) + 1)
        j0 = max(0, int(np.floor(min_y / t.cell)) - int(org[1]))
        j1 = min(t.n, int(np.floor(max_y / t.cell)) - int(org[1]) + 1)
        if i0 >= i1 or j0 >= j1:
            return {"cells": np.zeros((0, 2), np.int64), "cost": np.zeros((0,), np.uint8)}
        sub_cost = s.cost[i0:i1, j0:j1]
        sub_cls = s.cls[i0:i1, j0:j1]
        return {
            "tier": tier_idx,
            "slice_i": (i0, i1),
            "slice_j": (j0, j1),
            "cost": sub_cost,
            "cls": sub_cls,
        }

    def reset(self) -> None:
        """Reset internal grid state completely (including temporal dynamics)."""
        for s in self.state:
            s.count.fill(0)
            s.z_min.fill(np.nan)
            s.z_max.fill(np.nan)
            s.ground.fill(np.nan)
            s.rough.fill(np.nan)
            s.cls.fill(UNKNOWN)
            s.conf.fill(0)
            s.flags.fill(0xF0)
            s.clear.fill(UNKNOWN)
            s.cost.fill(UNKNOWN)
            s.age.fill(UNKNOWN)
            s.dynamic_mask.fill(False)
            s.free_passes.fill(0)
            s._eff_cls = None
        self.origins = None
        self.temporal.reset()
        self.frame_index = -1
        self._last_timestamp = None

    def temporal_snapshot(self) -> tuple[dict[str, Any], ...]:
        """Detached copy of live dynamic tracks (world-state publication)."""
        return self.temporal.snapshot_tracks()

    def temporal_stats(self) -> dict[str, Any]:
        """Bounded temporal-model statistics (plain values)."""
        return self.temporal.stats()

    def terrain_report(self) -> dict[str, Any]:
        """Aggregate terrain/traversability diagnostics (plain values, diagnostic-only).

        Vectorized over live tiers; adds no persistent state and allocates
        only small per-tier temporaries (see memory model).
        """
        stale_thresh = self.terrain.stale_age_threshold if self.terrain is not None else 20
        return report_layers(
            self.state,
            [t.cell for t in self.tiers],
            stale_age_threshold=stale_thresh,
            traversable_cost_max=self._trav_max(),
        )

    def memory_report(self) -> dict[str, Any]:
        """Produce honest memory accounting matching PRD acceptance criteria."""
        cells_per_tier = [t.n * t.n for t in self.tiers]
        allocated_cells = sum(cells_per_tier)
        eff_cells = self.tiers[0].n * self.tiers[0].n
        for k in range(1, len(self.tiers)):
            r = self.tiers[k].ratio // self.tiers[k - 1].ratio
            inner_covered = (self.tiers[k - 1].n // r) ** 2
            eff_cells += (self.tiers[k].n * self.tiers[k].n - inner_covered)

        persistent_bytes = sum(s.nbytes for s in self.state)
        aux_bytes = sum(s.aux_nbytes for s in self.state)
        total_allocated = persistent_bytes + aux_bytes

        # Uniform baseline: 5 cm cells over outer extent
        uniform_cells = int((2 * self.tiers[-1].half / self.tiers[0].cell) ** 2)
        uniform_bytes = uniform_cells * 16
        reduction = uniform_bytes / max(persistent_bytes, 1)

        return {
            "cells_per_tier": cells_per_tier,
            "allocated_cells": allocated_cells,
            "effective_non_overlapping_cells": eff_cells,
            "bytes_per_cell": 16,
            "persistent_state_bytes": persistent_bytes,
            "persistent_bytes": persistent_bytes,
            "allocated_bytes": persistent_bytes,
            "auxiliary_bytes": aux_bytes,
            "cache_bytes": 0,
            "total_allocated_bytes": total_allocated,
            "effective_bytes": eff_cells * 16,
            "uniform_baseline_cells": uniform_cells,
            "uniform_baseline_bytes": uniform_bytes,
            "reduction_ratio": reduction,
            "memory_reduction_ratio": reduction,
            "under_8mb_target": total_allocated <= 8 * 1024 * 1024,
            # Phase 6 bounded temporal state (runtime overhead, reported
            # separately from PRD grid-layer map memory above).
            "temporal_tracks": len(self.temporal),
            "temporal_tracks_capacity": int(self.dynamic_config.max_tracks),
            "temporal_bytes_estimate": len(self.temporal) * 192,
            "temporal_stats": self.temporal.stats(),
        }
