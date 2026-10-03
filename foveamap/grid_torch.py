"""PyTorch port of the foveated grid engine (runs on CUDA, MPS or CPU).

Same tiers, window snapping, authoritative tier assignment, true mip-up,
and per-cell semantics as `grid.FoveatedGrid`; the NumPy engine stays the
reference and `tests/test_grid_torch.py` checks parity against it.

* The fine index floor(x / 0.05) is always computed in float64 (on the host
  when the device has no float64, e.g. MPS), so points on cell boundaries
  land in exactly the same cell as in the NumPy engine. Everything after
  that is integer arithmetic.
* Single native tier assignment per point, followed by exact integer-lattice
  mip-up aggregation on the device.
* Per-point statistics are reduced on the device with `torch.unique` +
  `index_add_` / `scatter_reduce_`.
* Persistent 16 bytes per cell layout with honest memory accounting.
* Separate dynamic layer rebuilt fresh each frame.
* 2.5D slope derivation and terrain traversability parity with NumPy reference.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import torch
import torch.nn.functional as F

from .grid import (
    FoveatedGrid,
    Tier,
    TierLayers,
    GROUND_MASK,
    COST_PRIOR,
    DRIVABLE,
    UNKNOWN,
    F_SLOPE,
    F_OVERHANG,
    F_STEP,
    F_DEPRESSION,
    VEHICLE_CLEARANCE,
    STEP_THRESH,
    DEPRESSION_THRESH,
    DEPRESSION_WIN,
    SLOPE_THRESH,
    SLOPE_CRIT,
)
from .core.ontology import PERSON, VEHICLE, NUM_CLASSES, GROUND_CLASSES
from .core.config import GridConfig, TerrainConfig, DynamicConfig
from .temporal import (
    ACTIVE_DYNAMIC as DYN_ACTIVE,
    TEMPORARILY_MISSING as DYN_MISSING,
    STALE as DYN_STALE,
)

# clearance thresholds evaluated in float64 on the stored 2 cm code, exactly as the NumPy engine does
_CLEAR_CODES = np.arange(256) * 0.02
OVERHANG_LUT = (_CLEAR_CODES > 0.5) & (np.arange(256) != UNKNOWN)
PASS_UNDER_LUT = OVERHANG_LUT & (_CLEAR_CODES >= VEHICLE_CLEARANCE)

# Ground-class lookup on device (road/sidewalk/parking/terrain only).
_GROUND_IDS = torch.tensor(list(GROUND_CLASSES), dtype=torch.long)


class TorchTierLayers:
    """Device copy of `grid.TierLayers` (same fields, dtypes and 16 bytes per cell).

    torch has no index_put for uint16 on CPU, so `count` is written through an
    int16 view of the same bits.
    """
    FIELDS = dict(count=torch.uint16, z_min=torch.float16, z_max=torch.float16, ground=torch.float16,
                  rough=torch.float16, cls=torch.uint8, conf=torch.uint8, flags=torch.uint8,
                  clear=torch.uint8, cost=torch.uint8, age=torch.uint8)
    FILL = dict(count=0, z_min=float("nan"), z_max=float("nan"), ground=float("nan"), rough=float("nan"),
                cls=UNKNOWN, conf=0, flags=0xF0, clear=UNKNOWN, cost=UNKNOWN, age=UNKNOWN)

    def __init__(self, n: int, device: torch.device, cell_size_m: float = 0.05, half_extent_m: float = 10.0):
        self.n, self.device = n, device
        self.cell = float(cell_size_m)
        self.half = float(half_extent_m)
        self.r = self.cell
        for f, dt in self.FIELDS.items():
            if f == "count":
                self.count = torch.zeros((n, n), dtype=torch.int16, device=device).view(torch.uint16)
            else:
                setattr(self, f, torch.full((n, n), self.FILL[f], dtype=dt, device=device))
        self.dynamic_mask = torch.zeros((n, n), dtype=torch.bool, device=device)
        self.free_passes = torch.zeros((n, n), dtype=torch.uint8, device=device)
        self._eff_cls: torch.Tensor | None = None

    @property
    def eff_cls(self) -> torch.Tensor:
        """Effective class view (passable under overhangs).

        Uses the stored secondary-evidence class only when it is a ground
        class; a non-ground runner-up never becomes effective.
        """
        if self._eff_cls is not None:
            return self._eff_cls
        cls = self.cls.long()
        gcls = (self.flags >> 4).long()
        ground = self.ground.float()
        valid_g = ground.isfinite()
        clear = self.clear.long()
        pass_under = torch.as_tensor(PASS_UNDER_LUT, device=self.device)
        is_ground = torch.isin(gcls, _GROUND_IDS.to(self.device))
        passable = pass_under[clear.clamp(0, 255)] & valid_g & is_ground
        eff = torch.where(passable, gcls, cls)
        return torch.where(cls == UNKNOWN, UNKNOWN, eff).to(torch.uint8)

    @eff_cls.setter
    def eff_cls(self, val: torch.Tensor) -> None:
        self._eff_cls = val

    @property
    def min_z(self) -> torch.Tensor:
        return self.z_min

    @property
    def max_z(self) -> torch.Tensor:
        return self.z_max

    @property
    def roughness(self) -> torch.Tensor:
        return self.rough

    @property
    def clearance(self) -> torch.Tensor:
        return self.clear

    @property
    def dynamic(self) -> torch.Tensor:
        return self.dynamic_mask

    @property
    def secondary_class(self) -> torch.Tensor:
        """Secondary (runner-up) class ID (0-8 or UNKNOWN if none) stored in flags upper nibble."""
        nid = (self.flags >> 4).long()
        return torch.where(nid == 0x0F, UNKNOWN, nid).to(torch.uint8)

    @property
    def secondary_confidence(self) -> torch.Tensor:
        """Secondary class confidence in [0.0, 1.0] stored in conf lower nibble."""
        return (self.conf & 0x0F).float() / 15.0

    @property
    def primary_confidence(self) -> torch.Tensor:
        """Primary class confidence in [0.0, 1.0] stored in conf upper nibble."""
        return (self.conf >> 4).float() / 15.0

    @property
    def count16(self) -> torch.Tensor:
        return self.count.view(torch.int16)

    @property
    def nbytes(self) -> int:
        return sum(getattr(self, f).nelement() * getattr(self, f).element_size() for f in self.FIELDS)

    @property
    def aux_nbytes(self) -> int:
        b = self.free_passes.nelement() * self.free_passes.element_size() + self.dynamic_mask.nelement() * self.dynamic_mask.element_size()
        if self._eff_cls is not None:
            b += self._eff_cls.nelement() * self._eff_cls.element_size()
        return b

    def shifted(self, d: Sequence[int]) -> TorchTierLayers:
        """Copy of this state moved by d = (di, dj) cells (window scroll)."""
        out = TorchTierLayers(self.n, self.device, cell_size_m=self.cell, half_extent_m=self.half)
        di, dj = int(d[0]), int(d[1])
        n = self.n
        if abs(di) >= n or abs(dj) >= n:
            return out
        src = (slice(max(di, 0), n + min(di, 0)), slice(max(dj, 0), n + min(dj, 0)))
        dst = (slice(max(-di, 0), n + min(-di, 0)), slice(max(-dj, 0), n + min(-dj, 0)))
        out.count16[dst] = self.count16[src]
        for f in self.FIELDS:
            if f != "count":
                getattr(out, f)[dst] = getattr(self, f)[src]
        out.dynamic_mask[dst] = self.dynamic_mask[src]
        out.free_passes[dst] = self.free_passes[src]
        if self._eff_cls is not None:
            out._eff_cls = torch.full((n, n), UNKNOWN, dtype=torch.uint8, device=self.device)
            out._eff_cls[dst] = self._eff_cls[src]
        return out

    def to_numpy(self) -> TierLayers:
        """Host `grid.TierLayers` with the same contents."""
        out = TierLayers(self.n, cell_size_m=self.cell, half_extent_m=self.half)
        for f in self.FIELDS:
            v = self.count16.cpu().numpy().view(np.uint16) if f == "count" else getattr(self, f).cpu().numpy()
            getattr(out, f)[:] = v
        out.dynamic_mask[:] = self.dynamic_mask.cpu().numpy()
        out.free_passes[:] = self.free_passes.cpu().numpy()
        if self._eff_cls is not None:
            out.eff_cls = self._eff_cls.cpu().numpy()
        else:
            out.eff_cls = self.eff_cls.cpu().numpy()
        return out


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchFoveatedGrid(FoveatedGrid):
    def __init__(self, profile: str | GridConfig | Sequence[tuple[float, float]] = "spec",
                 fuse: bool = True, device: Any = None,
                 dtype: torch.dtype = torch.float32,
                 terrain_config: TerrainConfig | None = None,
                 dynamic_config: DynamicConfig | None = None):
        super().__init__(profile, fuse=fuse, terrain_config=terrain_config,
                         dynamic_config=dynamic_config)
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        dev = self.device
        self._ground_mask = torch.as_tensor(GROUND_MASK, device=dev)
        if self.terrain is not None:
            self._cost_prior = torch.as_tensor(self.terrain.cost_priors, dtype=torch.int32, device=dev)
            pass_under = OVERHANG_LUT & (_CLEAR_CODES >= self.terrain.vehicle_clearance_m)
            self._pass_under = torch.as_tensor(pass_under, device=dev)
        else:
            self._cost_prior = torch.as_tensor(COST_PRIOR, device=dev)
            self._pass_under = torch.as_tensor(PASS_UNDER_LUT, device=dev)
        self._drivable = torch.as_tensor(DRIVABLE, device=dev)
        self._overhang = torch.as_tensor(OVERHANG_LUT, device=dev)
        self.state = [self._new_layers(t) for t in self.tiers]

    # ------------------------------------------------------------------ utils
    def _t(self, a: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(a).to(self.device, dtype)

    def fine_index_t(self, xy_world: Any) -> torch.Tensor:
        """floor(xy / base) as int64 on device, computed in float64."""
        if not torch.is_tensor(xy_world):
            return torch.from_numpy(self.fine_index(xy_world)).to(self.device)
        xy = xy_world if xy_world.device.type != "mps" else xy_world.cpu()
        return torch.floor(xy.double() / self.base).long().to(self.device)

    # ------------------------------------------------ authoritative tier assign
    def assign_native_tiers_t(self, xy_world: Any, origins: list[np.ndarray | torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        """Assign each point to exactly one native tier on the device."""
        dev = self.device
        n_pts = len(xy_world)
        if n_pts == 0:
            diag = {"points_in": 0, "native_points_assigned": 0, "filtered_points": 0,
                    "native_counts_per_tier": [0] * len(self.tiers)}
            return torch.zeros((0,), dtype=torch.int32, device=dev), diag

        f = self.fine_index_t(xy_world)
        native_tiers = torch.full((n_pts,), -1, dtype=torch.int32, device=dev)
        unassigned = torch.ones(n_pts, dtype=torch.bool, device=dev)

        for k, (t, org) in enumerate(zip(self.tiers, origins)):
            org_t = torch.as_tensor(org, device=dev)
            ij = torch.div(f, t.ratio, rounding_mode="floor") - org_t
            in_tier = (ij[:, 0] >= 0) & (ij[:, 0] < t.n) & (ij[:, 1] >= 0) & (ij[:, 1] < t.n)
            assign_mask = unassigned & in_tier
            native_tiers[assign_mask] = k
            unassigned[assign_mask] = False

        n_assigned = int((native_tiers >= 0).sum().item())
        n_filtered = int((native_tiers == -1).sum().item())
        diag = {
            "points_in": n_pts,
            "native_points_assigned": n_assigned,
            "filtered_points": n_filtered,
            "native_counts_per_tier": [int((native_tiers == k).sum().item()) for k in range(len(self.tiers))],
        }
        self.last_diagnostics = diag
        return native_tiers, diag

    # -------------------------------------------------------------- binning
    def bin_points(self, xy_world: Any, z: Any, probs: Any, moving: Any,
                   origins: list[Any]) -> list[dict[str, Any]]:
        """Device single-native binning + exact integer-lattice mip-up."""
        dev, fdt = self.device, self.dtype
        n_pts = len(xy_world)
        if n_pts == 0:
            num_cls = probs.shape[1] if (torch.is_tensor(probs) or isinstance(probs, np.ndarray)) and probs.ndim == 2 else NUM_CLASSES
            return [
                dict(
                    key=torch.zeros((0,), dtype=torch.int64, device=dev),
                    n_pts=torch.zeros((0,), dtype=torch.int64, device=dev),
                    n_static=torch.zeros((0,), dtype=fdt, device=dev),
                    n_ground=torch.zeros((0,), dtype=fdt, device=dev),
                    z_min=torch.zeros((0,), dtype=fdt, device=dev),
                    z_max=torch.zeros((0,), dtype=fdt, device=dev),
                    ground=torch.zeros((0,), dtype=fdt, device=dev),
                    rough=torch.zeros((0,), dtype=fdt, device=dev),
                    zmin_ng=torch.zeros((0,), dtype=fdt, device=dev),
                    p_static=torch.zeros((0, num_cls), dtype=fdt, device=dev),
                    p_all=torch.zeros((0, num_cls), dtype=fdt, device=dev),
                    n_dyn=torch.zeros((0,), dtype=fdt, device=dev),
                    n_dyn_person=torch.zeros((0,), dtype=fdt, device=dev),
                    zmin_dyn=torch.zeros((0,), dtype=fdt, device=dev),
                    zmax_dyn=torch.zeros((0,), dtype=fdt, device=dev),
                    n_in=0,
                )
                for _ in self.tiers
            ]

        f = self.fine_index_t(xy_world)
        z = self._t(z, fdt)
        P = self._t(probs, fdt)
        moving = self._t(moving, torch.bool)
        cls = P.argmax(1)
        is_ground = self._ground_mask[cls]
        static = ~moving

        native_tiers, _ = self.assign_native_tiers_t(xy_world, origins)

        # 1. Native binning
        native_stats = []
        for k, (t, org) in enumerate(zip(self.tiers, origins)):
            idx = (native_tiers == k).nonzero().squeeze(1)
            if idx.numel() == 0:
                native_stats.append(dict(
                    key=torch.zeros((0,), dtype=torch.int64, device=dev),
                    n_pts=torch.zeros((0,), dtype=torch.int64, device=dev),
                    n_static=torch.zeros((0,), dtype=fdt, device=dev),
                    n_ground=torch.zeros((0,), dtype=fdt, device=dev),
                    z_min=torch.zeros((0,), dtype=fdt, device=dev),
                    z_max=torch.zeros((0,), dtype=fdt, device=dev),
                    ground=torch.zeros((0,), dtype=fdt, device=dev),
                    rough=torch.zeros((0,), dtype=fdt, device=dev),
                    zmin_ng=torch.zeros((0,), dtype=fdt, device=dev),
                    p_static=torch.zeros((0, P.shape[1]), dtype=fdt, device=dev),
                    p_all=torch.zeros((0, P.shape[1]), dtype=fdt, device=dev),
                    n_dyn=torch.zeros((0,), dtype=fdt, device=dev),
                    n_dyn_person=torch.zeros((0,), dtype=fdt, device=dev),
                    zmin_dyn=torch.zeros((0,), dtype=fdt, device=dev),
                    zmax_dyn=torch.zeros((0,), dtype=fdt, device=dev),
                    n_in=0,
                ))
                continue

            org_t = torch.as_tensor(org, device=dev)
            ij = torch.div(f[idx], t.ratio, rounding_mode="floor") - org_t
            key = ij[:, 0] * t.n + ij[:, 1]
            uk, inv = torch.unique(key, sorted=True, return_inverse=True)
            M = uk.numel()
            zz, st = z[idx], static[idx]
            ng_pt = is_ground[idx]
            gg = ng_pt & st
            inf = torch.tensor(float("inf"), dtype=fdt, device=dev)

            def ssum(v: torch.Tensor) -> torch.Tensor:
                return torch.zeros((M,) + v.shape[1:], dtype=v.dtype, device=dev).index_add_(0, inv, v)

            def smin(v: torch.Tensor) -> torch.Tensor:
                return torch.full((M,), float("inf"), dtype=fdt, device=dev).scatter_reduce_(0, inv, v, "amin")

            def smax(v: torch.Tensor) -> torch.Tensor:
                return torch.full((M,), -float("inf"), dtype=fdt, device=dev).scatter_reduce_(0, inv, v, "amax")

            n_static = ssum(st.to(fdt))
            n_ground = ssum(gg.to(fdt))
            gmean = ssum(torch.where(gg, zz, 0)) / n_ground.clamp(min=1)
            gvar = ssum(torch.where(gg, zz - gmean[inv], 0) ** 2) / n_ground.clamp(min=1)
            nan = torch.tensor(float("nan"), dtype=fdt, device=dev)
            Pi = P[idx]
            mv = moving[idx]

            native_stats.append(dict(
                key=uk,
                n_pts=ssum(torch.ones_like(inv, dtype=torch.int64)),
                n_static=n_static,
                n_ground=n_ground,
                z_min=smin(torch.where(st, zz, inf)),
                z_max=smax(torch.where(st, zz, -inf)),
                ground=torch.where(n_ground > 0, gmean, nan),
                rough=torch.where(n_ground > 1, gvar.sqrt(), nan),
                zmin_ng=smin(torch.where(st & ~ng_pt, zz, inf)),
                p_static=ssum(Pi * st[:, None]),
                p_all=ssum(Pi),
                n_dyn=ssum(mv.to(fdt)),
                n_dyn_person=ssum((mv & (cls[idx] == PERSON)).to(fdt)),
                zmin_dyn=smin(torch.where(mv, zz, inf)),
                zmax_dyn=smax(torch.where(mv, zz, -inf)),
                n_in=int(idx.numel()),
            ))

        # 2. Integer-lattice Mip-Up
        fused_stats = list(native_stats)
        for k in range(len(self.tiers) - 1):
            child_st = fused_stats[k]
            child_t = self.tiers[k]
            parent_t = self.tiers[k + 1]
            child_org = origins[k]
            parent_org = origins[k + 1]
            fused_stats[k + 1] = self._mip_up_tier_t(child_st, child_t, parent_t, child_org, parent_org, fused_stats[k + 1])

        return fused_stats

    def _mip_up_tier_t(self, child_st: dict[str, Any], child_t: Tier, parent_t: Tier,
                       child_org: Any, parent_org: Any, parent_native: dict[str, Any]) -> dict[str, Any]:
        """Aggregate child tier cells into parent tier on exact integer lattice in PyTorch."""
        dev, fdt = self.device, self.dtype
        k_c = child_st["key"]
        if k_c.numel() == 0:
            return parent_native

        r = parent_t.ratio // child_t.ratio
        child_org_t = torch.as_tensor(child_org, device=dev)
        parent_org_t = torch.as_tensor(parent_org, device=dev)

        fi = torch.div(k_c, child_t.n, rounding_mode="floor") + child_org_t[0]
        fj = (k_c % child_t.n) + child_org_t[1]
        pi = torch.div(fi, r, rounding_mode="floor") - parent_org_t[0]
        pj = torch.div(fj, r, rounding_mode="floor") - parent_org_t[1]
        k_p = pi * parent_t.n + pj

        uk_p, inv_p = torch.unique(k_p, sorted=True, return_inverse=True)
        M_p = uk_p.numel()

        def ssum_p(v: torch.Tensor) -> torch.Tensor:
            return torch.zeros((M_p,) + v.shape[1:], dtype=v.dtype, device=dev).index_add_(0, inv_p, v)

        def smin_p(v: torch.Tensor) -> torch.Tensor:
            return torch.full((M_p,), float("inf"), dtype=fdt, device=dev).scatter_reduce_(0, inv_p, v, "amin")

        def smax_p(v: torch.Tensor) -> torch.Tensor:
            return torch.full((M_p,), -float("inf"), dtype=fdt, device=dev).scatter_reduce_(0, inv_p, v, "amax")

        n_pts_p = ssum_p(child_st["n_pts"].to(torch.int64))
        n_static_p = ssum_p(child_st["n_static"])
        cg = child_st["ground"]
        c_valid = cg.isfinite()
        ng_c = child_st["n_ground"]

        ng_p = ssum_p(torch.where(c_valid, ng_c, 0.0))
        g_sum_p = ssum_p(torch.where(c_valid, cg * ng_c, 0.0))
        nan = torch.tensor(float("nan"), dtype=fdt, device=dev)
        ground_p = torch.where(ng_p > 0, g_sum_p / ng_p.clamp(min=1), nan)

        cr = child_st["rough"].nan_to_num(0.0)
        g_diff = cg - ground_p[inv_p]
        var_sum_p = ssum_p(torch.where(c_valid, ng_c * (cr ** 2 + g_diff ** 2), 0.0))
        var_p = torch.where(ng_p > 1, (var_sum_p / ng_p.clamp(min=1)).clamp(min=0.0), nan)
        rough_p = torch.where(var_p.isfinite(), var_p.sqrt(), nan)

        z_min_p = smin_p(child_st["z_min"])
        z_max_p = smax_p(child_st["z_max"])
        zmin_ng_p = smin_p(child_st["zmin_ng"])

        p_static_p = ssum_p(child_st["p_static"])
        p_all_p = ssum_p(child_st["p_all"])
        n_dyn_p = ssum_p(child_st["n_dyn"])
        n_dyn_person_p = ssum_p(child_st["n_dyn_person"])
        zmin_dyn_p = smin_p(child_st.get("zmin_dyn", torch.full_like(z_min_p, float("inf"))))
        zmax_dyn_p = smax_p(child_st.get("zmax_dyn", torch.full_like(z_max_p, -float("inf"))))

        # Combine with parent native
        if parent_native["key"].numel() == 0:
            return dict(
                key=uk_p, n_pts=n_pts_p, n_static=n_static_p, n_ground=ng_p,
                z_min=z_min_p, z_max=z_max_p, ground=ground_p, rough=rough_p,
                zmin_ng=zmin_ng_p, p_static=p_static_p, p_all=p_all_p,
                n_dyn=n_dyn_p, n_dyn_person=n_dyn_person_p,
                zmin_dyn=zmin_dyn_p, zmax_dyn=zmax_dyn_p,
                n_in=int(n_pts_p.sum().item()),
            )

        all_keys = torch.cat([parent_native["key"], uk_p])
        order = torch.argsort(all_keys)

        return dict(
            key=all_keys[order],
            n_pts=torch.cat([parent_native["n_pts"], n_pts_p])[order],
            n_static=torch.cat([parent_native["n_static"], n_static_p])[order],
            n_ground=torch.cat([parent_native["n_ground"], ng_p])[order],
            z_min=torch.cat([parent_native["z_min"], z_min_p])[order],
            z_max=torch.cat([parent_native["z_max"], z_max_p])[order],
            ground=torch.cat([parent_native["ground"], ground_p])[order],
            rough=torch.cat([parent_native["rough"], rough_p])[order],
            zmin_ng=torch.cat([parent_native["zmin_ng"], zmin_ng_p])[order],
            p_static=torch.cat([parent_native["p_static"], p_static_p], dim=0)[order],
            p_all=torch.cat([parent_native["p_all"], p_all_p], dim=0)[order],
            n_dyn=torch.cat([parent_native["n_dyn"], n_dyn_p])[order],
            n_dyn_person=torch.cat([parent_native["n_dyn_person"], n_dyn_person_p])[order],
            zmin_dyn=torch.cat([parent_native.get("zmin_dyn", torch.full_like(parent_native["z_min"], float("inf"))), zmin_dyn_p])[order],
            zmax_dyn=torch.cat([parent_native.get("zmax_dyn", torch.full_like(parent_native["z_max"], -float("inf"))), zmax_dyn_p])[order],
            n_in=parent_native["n_in"] + int(n_pts_p.sum().item()),
        )

    # --------------------------------------------------------------- update
    def _new_layers(self, t: Tier) -> TorchTierLayers:
        return TorchTierLayers(t.n, self.device, cell_size_m=t.cell, half_extent_m=t.half)

    TIER_FIELD_SPECS = (
        ("count", "count16", np.uint16, 2),
        ("z_min", "z_min", np.float16, 2),
        ("z_max", "z_max", np.float16, 2),
        ("ground", "ground", np.float16, 2),
        ("rough", "rough", np.float16, 2),
        ("cls", "cls", np.uint8, 1),
        ("conf", "conf", np.uint8, 1),
        ("flags", "flags", np.uint8, 1),
        ("clear", "clear", np.uint8, 1),
        ("cost", "cost", np.uint8, 1),
        ("age", "age", np.uint8, 1),
        ("eff_cls", "eff_cls", np.uint8, 1),
    )

    def snapshot(self) -> list[TierLayers]:
        """Host copy of the grid state (single packed device-to-host copy)."""
        parts = [getattr(s, attr).view(torch.uint8).reshape(-1)
                 for s in self.state for _, attr, _, _ in self.TIER_FIELD_SPECS]
        packed = torch.cat(parts).cpu().numpy()
        out = []
        offset = 0
        for s in self.state:
            tl = TierLayers.__new__(TierLayers)
            tl.n = s.n
            tl.cell = s.cell
            tl.half = s.half
            tl.r = s.r
            tl.dynamic_mask = s.dynamic_mask.cpu().numpy()
            tl.free_passes = s.free_passes.cpu().numpy()
            n_cells = s.n * s.n
            for name, _, dt, b in self.TIER_FIELD_SPECS:
                sz = n_cells * b
                setattr(tl, name, packed[offset:offset + sz].view(dt).reshape(s.n, s.n))
                offset += sz
            out.append(tl)
        return out

    def _fuse_tier(self, t: Tier, s: TorchTierLayers, st: dict[str, Any]) -> dict[str, Any]:
        n, fdt = t.n, self.dtype
        key = st["key"]
        obs = st["n_static"] > 0
        k = key[obs]
        i, j = k // n, k % n

        # Reset free passes
        s.free_passes[i, j] = 0

        # ---- class
        p = st["p_static"][obs]
        p = p / p.sum(1, keepdim=True).clamp(min=1e-9)
        old_c = s.cls[i, j].long()
        old_conf = s.conf[i, j].to(fdt) / 255.0
        a = t.alpha if self.fuse else 1.0
        q = a * p
        valid_old = old_c != UNKNOWN
        safe_c = torch.where(valid_old, old_c, 0)
        delta = torch.where(valid_old, (1 - a) * old_conf, 0.0)
        q.scatter_add_(1, safe_c[:, None], delta[:, None])
        new_c = q.argmax(1).to(torch.uint8)
        s.cls[i, j] = new_c
        q_sum = q.sum(1).clamp(min=1e-9)
        # Persistent secondary evidence (flags upper nibble + conf lower nibble).
        # Same unambiguous semantics as the NumPy engine: prefer a distinct
        # confident ground class, else the fused runner-up; confidence always
        # belongs to the stored class.
        q_sec = q.clone()
        q_sec[torch.arange(len(new_c), device=q.device), new_c.long()] = -1.0
        sec_c = q_sec.argmax(dim=1)
        sec_conf_runner = q_sec.max(dim=1).values.clamp(min=0.0) / q_sum
        has_runner = sec_conf_runner > 0.02
        pg = torch.where(self._ground_mask[None, :], q, 0)
        pg_sum = pg.sum(1)
        gcls = torch.where(pg_sum > 0.05 * q_sum, pg.argmax(1), 0xF)
        gconf = torch.where(pg_sum > 0, pg.max(dim=1).values / q_sum, 0.0)
        use_ground = (gcls != 0xF) & (gcls != new_c.long()) & (gconf > 0.02)
        final_sec = torch.where(
            use_ground, gcls,
            torch.where(has_runner, sec_c, torch.tensor(0xF, dtype=torch.long, device=q.device)),
        )
        final_sec_conf = torch.where(
            use_ground, gconf,
            torch.where(has_runner, sec_conf_runner, torch.tensor(0.0, dtype=q.dtype, device=q.device)),
        )
        has_sec = use_ground | has_runner
        sec_c_id = torch.where(has_sec, final_sec.to(torch.uint8), torch.tensor(0xF, dtype=torch.uint8, device=q.device))
        sec_conf_4bit = (final_sec_conf * 15.0).round().clamp(0, 15).to(torch.uint8)
        prim_conf_4bit = ((q.max(1).values / q_sum) * 15.0).round().clamp(0, 15).to(torch.uint8)
        s.conf[i, j] = (prim_conf_4bit << 4) | (sec_conf_4bit & 0x0F)

        # Secondary-evidence nibble stores the class selected above.
        keep_sec = sec_c_id.long()

        # ---- heights
        g_new = st["ground"][obs]
        g_old = s.ground[i, j].to(fdt)
        both = g_new.isfinite() & g_old.isfinite()
        g = torch.where(both, (1 - a) * g_old + a * g_new, torch.where(g_new.isfinite(), g_new, g_old))
        s.ground[i, j] = g.half()
        r_new, r_old = st["rough"][obs], s.rough[i, j].to(fdt)
        s.rough[i, j] = torch.where(r_new.isfinite(), torch.where(r_old.isfinite(), (1 - a) * r_old + a * r_new, r_new),
                                    r_old).half()
        s.z_min[i, j] = st["z_min"][obs].half()
        s.z_max[i, j] = st["z_max"][obs].half()
        s.count16[i, j] = st["n_static"][obs].clamp(max=65535).to(torch.int32).to(torch.int16)
        clear = st["zmin_ng"][obs] - g
        s.clear[i, j] = torch.where(clear.isfinite(), (clear / 0.02).clamp(0, 254), UNKNOWN).to(torch.uint8)
        s.flags[i, j] = (keep_sec << 4).to(torch.uint8)

        # ---- age & stale transitions
        seen = s.age != UNKNOWN
        s.age[seen] = (s.age[seen].int() + 1).clamp(max=UNKNOWN).to(torch.uint8)
        s.age[i, j] = 0

        max_stale = self.terrain.max_stale_age if self.terrain is not None else 100
        stale_timeout = (s.age.int() >= max_stale) & seen
        s.cls[stale_timeout] = UNKNOWN
        s.conf[stale_timeout] = 0
        s.cost[stale_timeout] = UNKNOWN
        s.count16[stale_timeout] = 0
        s.ground[stale_timeout] = float("nan")
        s.z_min[stale_timeout] = float("nan")
        s.z_max[stale_timeout] = float("nan")
        s.rough[stale_timeout] = float("nan")
        s.clear[stale_timeout] = UNKNOWN
        s.flags[stale_timeout] = 0xF0
        s.age[stale_timeout] = UNKNOWN

        # ---- dynamic layer
        s.dynamic_mask.zero_()
        dyn_cells = st["n_dyn"] > 0
        dk = key[dyn_cells]
        if len(dk) > 0:
            s.dynamic_mask[dk // n, dk % n] = True

        d_count = st["n_dyn"][dyn_cells].to(torch.int64)
        d_person = st["n_dyn_person"][dyn_cells]
        dcls = torch.where(d_person * 2 > d_count, PERSON, VEHICLE).to(torch.uint8)
        dconf = torch.where(dcls == PERSON, d_person / d_count.clamp(min=1), (d_count - d_person) / d_count.clamp(min=1))
        dconf_u8 = (dconf * 255.0).clamp(0, 255).to(torch.uint8)

        zmin_d = st.get("zmin_dyn", torch.full_like(st["z_min"], float("nan")))[dyn_cells]
        zmax_d = st.get("zmax_dyn", torch.full_like(st["z_max"], float("nan")))[dyn_cells]

        return dict(
            i=dk // n,
            j=dk % n,
            cls=dcls,
            conf=dconf_u8,
            count=d_count,
            z_min=zmin_d,
            z_max=zmax_d,
        )

    def _derive(self, t: Tier, s: TorchTierLayers) -> None:
        """Derive flags, slope, and cost on device."""
        step_thresh = self.terrain.step_threshold_m if self.terrain is not None else STEP_THRESH
        dep_thresh = self.terrain.depression_threshold_m if self.terrain is not None else DEPRESSION_THRESH
        dep_win = self.terrain.depression_window_m if self.terrain is not None else DEPRESSION_WIN
        rough_thresh = self.terrain.roughness_threshold_m if self.terrain is not None else 0.04
        slope_thresh = self.terrain.slope_threshold_rad if self.terrain is not None else SLOPE_THRESH
        slope_crit = self.terrain.slope_critical_rad if self.terrain is not None else SLOPE_CRIT
        stale_thresh = self.terrain.stale_age_threshold if self.terrain is not None else 20

        cls = s.cls.long()
        gcls = (s.flags >> 4).long()
        ground = s.ground.float()
        valid_g = ground.isfinite()
        clear = s.clear.long()

        # 1. Overhang (passable only for ground-class secondary evidence)
        overhang = self._overhang[clear] & valid_g
        is_ground = torch.isin(gcls, _GROUND_IDS.to(s.cls.device))
        passable_under = self._pass_under[clear] & valid_g & is_ground
        eff_cls = torch.where(passable_under, gcls, cls)

        # 2. Steps
        gz = ground
        step = torch.zeros_like(gz)
        for d in (1, 2):
            for ax in (0, 1):
                diff = (gz - gz.roll(d, ax)).abs()
                if ax == 0:
                    diff[:d, :] = float("nan")
                else:
                    diff[:, :d] = float("nan")
                step = torch.fmax(step, diff.nan_to_num(nan=0.0))
                step = torch.fmax(step, diff.roll(-d, ax).nan_to_num(nan=0.0))
        stepf = step > step_thresh

        # 3. Depressions
        drv_cls = self._drivable[eff_cls]
        drv = drv_cls & valid_g
        win = max(3, int(round(dep_win / t.cell)) | 1)
        num = _box(torch.where(drv, gz, 0.0), win)
        den = _box(drv.float(), win)
        ref = torch.where(den > 0.05, num / den.clamp(min=1e-6), float("nan"))
        dep = drv & (gz < ref - dep_thresh)

        # 4. 2.5D Slope
        g_down = gz.roll(-1, 0)
        v_down = valid_g.roll(-1, 0)
        v_down[-1, :] = False

        g_up = gz.roll(1, 0)
        v_up = valid_g.roll(1, 0)
        v_up[0, :] = False

        dx_central = (g_down - g_up) / (2.0 * t.cell)
        dx_fwd = (g_down - gz) / t.cell
        dx_bwd = (gz - g_up) / t.cell
        dx = torch.where(v_down & v_up, dx_central, torch.where(v_down, dx_fwd, torch.where(v_up, dx_bwd, 0.0)))

        g_right = gz.roll(-1, 1)
        v_right = valid_g.roll(-1, 1)
        v_right[:, -1] = False

        g_left = gz.roll(1, 1)
        v_left = valid_g.roll(1, 1)
        v_left[:, 0] = False

        dy_central = (g_right - g_left) / (2.0 * t.cell)
        dy_fwd = (g_right - gz) / t.cell
        dy_bwd = (gz - g_left) / t.cell
        dy = torch.where(v_right & v_left, dy_central, torch.where(v_right, dy_fwd, torch.where(v_left, dy_bwd, 0.0)))

        has_slope = valid_g & ((v_down | v_up) | (v_right | v_left))
        slope_grad = (dx ** 2 + dy ** 2).sqrt()
        slope_rad = torch.where(has_slope, slope_grad.atan(), 0.0)
        slope_flag = has_slope & (slope_rad > slope_thresh)

        flags = (torch.where(overhang, F_OVERHANG, 0) | torch.where(stepf & valid_g, F_STEP, 0)
                 | torch.where(dep, F_DEPRESSION, 0) | torch.where(slope_flag, F_SLOPE, 0))

        # 5. Cost
        cost = self._cost_prior[eff_cls]
        cost = torch.where(passable_under, cost + 20, cost)
        cost = torch.where(stepf & drv_cls, cost.clamp(min=180), cost)
        cost = torch.where(stepf & (cost < 180), cost.clamp(min=140), cost)
        cost = torch.where(dep, cost.clamp(min=170), cost)

        rough = s.rough.float()
        cost = torch.where(rough.isfinite() & (rough > rough_thresh), cost + 30, cost)

        slope_excess = ((slope_rad - slope_thresh) / max(slope_crit - slope_thresh, 1e-4)).clamp(0.0, 1.0)
        cost = torch.where(slope_flag, cost + torch.round(slope_excess * 40).int(), cost)
        cost = torch.where(has_slope & (slope_rad >= slope_crit), cost.clamp(min=220), cost)

        cost = torch.where(s.conf < 150, cost + 25, cost)
        stale = (s.age != UNKNOWN) & (s.age > stale_thresh)
        cost = torch.where(stale, cost + 20, cost).clamp(0, 254)
        unknown = cls == UNKNOWN
        s.cost[:] = torch.where(unknown, UNKNOWN, cost).to(torch.uint8)
        s.flags[:] = (s.flags & 0xF0) | flags.to(torch.uint8)
        s._eff_cls = torch.where(unknown, UNKNOWN, eff_cls).to(torch.uint8)

    def _clear_rays(self, sensor_origin: Any, stats: list[dict[str, Any]]) -> None:
        """Conservative 2.5D ray clearing on device (Torch override).

        Mirrors :meth:`FoveatedGrid._clear_rays` semantics with identical
        consecutive-frame, ground-class, and dynamic-obstacle guards. Only the
        ray-geometry sampling loop is host-side (explicit opt-in boundary: ray
        clearing is disabled by default and never runs in the default Torch hot
        path). Streak updates and obstacle clearing are device-resident masked
        tensor ops with no per-cell host synchronization.
        """
        so = np.asarray(
            sensor_origin.cpu().numpy() if torch.is_tensor(sensor_origin) else sensor_origin,
            dtype=np.float64,
        )
        free_thresh = self.terrain.free_clear_frames if self.terrain is not None else 3

        for tier_idx, (t, s, st) in enumerate(zip(self.tiers, self.state, stats)):
            key = st["key"]
            n_static = st["n_static"]
            if torch.is_tensor(key):
                obs_mask = (n_static > 0).cpu().numpy()
                k_obs = key.detach().cpu().numpy()[obs_mask]
            else:
                k_obs = np.asarray(key)[np.asarray(n_static) > 0]
            if len(k_obs) == 0:
                s.free_passes.zero_()
                continue
            obs_set = set(int(v) for v in k_obs.tolist())
            step_sz = max(1, len(k_obs) // 500)
            sample_keys = k_obs[::step_sz]
            ci = sample_keys // t.n
            cj = sample_keys % t.n
            org = np.asarray(
                self.origins[tier_idx].cpu().numpy()
                if torch.is_tensor(self.origins[tier_idx])
                else self.origins[tier_idx]
            )
            target_xy = (np.stack([ci, cj], 1) + org + 0.5) * t.cell
            vec = target_xy - so[:2]
            dist = np.hypot(vec[:, 0], vec[:, 1])
            valid = (dist > 1.0) & (dist < 80.0)
            if not valid.any():
                s.free_passes.zero_()
                continue
            vec = vec[valid]
            dist = dist[valid]
            u = vec / dist[:, None]

            traversed: set[tuple[int, int]] = set()
            for d_ray, u_ray in zip(dist, u):
                n_samples = max(1, int((d_ray - 1.5 * t.cell - 1.0) / t.cell))
                if n_samples <= 0:
                    continue
                samples = np.linspace(1.0, d_ray - 1.5 * t.cell, n_samples)
                pts_ray = so[:2] + samples[:, None] * u_ray
                ij_ray = np.floor(pts_ray / t.cell).astype(np.int64) - org
                in_w = (
                    (ij_ray[:, 0] >= 0) & (ij_ray[:, 0] < t.n)
                    & (ij_ray[:, 1] >= 0) & (ij_ray[:, 1] < t.n)
                )
                if not in_w.any():
                    continue
                for c_i, c_j in ij_ray[in_w]:
                    if int(c_i) * t.n + int(c_j) in obs_set:
                        continue
                    traversed.add((int(c_i), int(c_j)))

            # Reset streaks not traversed this frame (device-resident).
            if traversed:
                trav_mask = torch.zeros((t.n, t.n), dtype=torch.bool, device=self.device)
                ti = torch.tensor([c[0] for c in traversed], dtype=torch.long, device=self.device)
                tj = torch.tensor([c[1] for c in traversed], dtype=torch.long, device=self.device)
                trav_mask[ti, tj] = True
                s.free_passes[~trav_mask & (s.free_passes > 0)] = 0
                # Increment streaks on-device.
                s.free_passes[ti, tj] = (s.free_passes[ti, tj].int() + 1).clamp(max=255).to(torch.uint8)
            else:
                s.free_passes[s.free_passes > 0] = 0
                continue

            # Clear obstacles reaching threshold: fully device-resident masked
            # ops. Only the ray-geometry sampling above is host-side (explicit
            # opt-in boundary); this application phase performs no host
            # synchronization and no per-cell Python loop.
            fp_hits = s.free_passes[ti, tj]
            due = fp_hits >= int(free_thresh)
            cls_t = s.cls[ti, tj]
            ground_t = torch.isin(cls_t.long(), _GROUND_IDS.to(self.device))
            dyn_t = s.dynamic_mask[ti, tj]
            # Dynamic occupants reaching threshold only reset their streak.
            s.free_passes[ti, tj] = torch.where(
                due & dyn_t,
                torch.zeros((), dtype=torch.uint8, device=self.device),
                fp_hits,
            )
            clear_now = due & (cls_t != UNKNOWN) & ~ground_t & ~dyn_t
            ci, cj = ti[clear_now], tj[clear_now]  # empty-safe: no-op when none
            s.cls[ci, cj] = UNKNOWN
            s.conf[ci, cj] = 0
            s.cost[ci, cj] = UNKNOWN
            s.count16[ci, cj] = 0
            s.ground[ci, cj] = float("nan")
            s.z_min[ci, cj] = float("nan")
            s.z_max[ci, cj] = float("nan")
            s.rough[ci, cj] = float("nan")
            s.clear[ci, cj] = UNKNOWN
            s.flags[ci, cj] = 0xF0
            s.age[ci, cj] = UNKNOWN
            s.free_passes[ci, cj] = 0

    def reset(self) -> None:
        """Reset internal grid state completely (including temporal dynamics)."""
        for s in self.state:
            s.count.zero_()
            s.z_min.fill_(float("nan"))
            s.z_max.fill_(float("nan"))
            s.ground.fill_(float("nan"))
            s.rough.fill_(float("nan"))
            s.cls.fill_(UNKNOWN)
            s.conf.zero_()
            s.flags.fill_(0xF0)
            s.clear.fill_(UNKNOWN)
            s.cost.fill_(UNKNOWN)
            s.age.fill_(UNKNOWN)
            s.dynamic_mask.zero_()
            s.free_passes.zero_()
            s._eff_cls = None
        self.origins = None
        self.temporal.reset()
        self.frame_index = -1
        self._last_timestamp = None

    def query_point(self, x: float, y: float) -> dict[str, Any]:
        """Query spatial cell state at continuous world coordinate (x, y) on device."""
        if self.origins is None:
            return {"tier": -1, "state": "OUT_OF_BOUNDS", "is_unknown": True, "is_traversable": False}

        f = self.fine_index_t(torch.tensor([[x, y]], device=self.device))
        for k, (t, org) in enumerate(zip(self.tiers, self.origins)):
            org_t = torch.as_tensor(org, device=self.device)
            ij = torch.div(f, t.ratio, rounding_mode="floor") - org_t
            if 0 <= ij[0, 0].item() < t.n and 0 <= ij[0, 1].item() < t.n:
                i, j = int(ij[0, 0].item()), int(ij[0, 1].item())
                s = self.state[k]
                cls = int(s.cls[i, j].item())
                cost = int(s.cost[i, j].item())
                cnt = int(s.count16[i, j].item())
                age = int(s.age[i, j].item())
                mask_dyn = bool(s.dynamic[i, j].item())
                # Phase 6 temporal overlay (host-side bounded store; the cell
                # indices above are already host ints, so no device sync here).
                enrich = self.temporal.query_enrichment(k, i, j)
                temporal_occupied = bool(enrich["dynamic_occupied"]) if enrich else False
                dyn = mask_dyn or temporal_occupied
                dyn_state = enrich["dynamic_state"] if enrich else None
                stale_thresh = self.terrain.stale_age_threshold if self.terrain is not None else 20
                if dyn:
                    if dyn_state == DYN_MISSING:
                        state = "TEMPORARILY_MISSING"
                    elif dyn_state == DYN_ACTIVE:
                        state = "ACTIVE_DYNAMIC"
                    else:
                        state = "OBSERVED_DYNAMIC"
                elif cnt == 0 or cls == UNKNOWN:
                    if enrich is not None and dyn_state == DYN_STALE:
                        state = "STALE"
                    else:
                        state = "UNKNOWN"
                elif age >= stale_thresh:
                    state = "STALE"
                else:
                    # Fresh static geometry wins over an expired dynamic track.
                    state = "OBSERVED_STATIC"
                ground_v = s.ground[i, j].item()
                z_min_v = s.z_min[i, j].item()
                z_max_v = s.z_max[i, j].item()
                rough_v = s.rough[i, j].item()
                clear_v = s.clear[i, j].item()
                return {
                    "tier": k,
                    "cell_size_m": t.cell,
                    "cell_coord": (i, j),
                    "dominant_class": cls,
                    "primary_class": cls,
                    "cls": cls,
                    "conf": int(s.conf[i, j].item()),
                    "secondary_class": int(s.secondary_class[i, j].item()),
                    "secondary_confidence": float(s.secondary_confidence[i, j].item()),
                    "cost": cost,
                    "count": cnt,
                    "dynamic": dyn,
                    "state": state,
                    "ground": float(ground_v) if not np.isnan(ground_v) else None,
                    "z_min": float(z_min_v) if not np.isnan(z_min_v) else None,
                    "z_max": float(z_max_v) if not np.isnan(z_max_v) else None,
                    "rough": float(rough_v) if not np.isnan(rough_v) else None,
                    "clear": float(clear_v * 0.02) if clear_v != UNKNOWN else None,
                    "age": age,
                    "flags": int(s.flags[i, j].item()),
                    "is_unknown": state == "UNKNOWN",
                    "is_traversable": (cost < 180) and (state != "UNKNOWN") and not dyn,
                    "dynamic_state": dyn_state,
                    "dynamic_confidence": float(enrich["dynamic_confidence"]) if enrich else 0.0,
                    "dynamic_age_frames": int(enrich["dynamic_age_frames"]) if enrich else 0,
                    "velocity": enrich["velocity"] if enrich else None,
                }
        return {"tier": -1, "state": "OUT_OF_BOUNDS", "is_unknown": True, "is_traversable": False}

    def is_traversable(self, x: float, y: float, max_cost: int = 150) -> bool:
        """Check if world coordinate (x, y) is safely traversable on device."""
        q = self.query_point(x, y)
        if q.get("dynamic", False) or q["is_unknown"] or q.get("cost", UNKNOWN) == UNKNOWN:
            return False
        return q.get("cost", 255) <= max_cost

    def get_height(self, x: float, y: float) -> float | None:
        """Get best elevation estimate at world coordinate (x, y) on device."""
        q = self.query_point(x, y)
        if q.get("ground") is not None:
            return q["ground"]
        if q.get("z_min") is not None:
            return q["z_min"]
        return None

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
            # Phase 6 bounded temporal state (reported separately; see grid.py).
            "temporal_tracks": len(self.temporal),
            "temporal_tracks_capacity": int(self.dynamic_config.max_tracks),
            "temporal_bytes_estimate": len(self.temporal) * 192,
            "temporal_stats": self.temporal.stats(),
        }


def _box(x: torch.Tensor, win: int) -> torch.Tensor:
    """Separable zero-padded box mean = scipy uniform_filter(x, win, mode="constant")."""
    x = x[None, None]
    x = F.avg_pool2d(x, (win, 1), stride=1, padding=(win // 2, 0), count_include_pad=True)
    x = F.avg_pool2d(x, (1, win), stride=1, padding=(0, win // 2), count_include_pad=True)
    return x[0, 0]


def stats_to_numpy(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-tier stats dicts with tensors moved to host NumPy arrays."""
    return [{k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in s.items()} for s in stats]

