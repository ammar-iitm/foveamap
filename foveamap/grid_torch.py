"""PyTorch port of the foveated grid engine (runs on CUDA, MPS or CPU).

Same tiers, window snapping and per-cell semantics as `grid.FoveatedGrid`;
the NumPy engine stays the reference and `tests/test_grid_torch.py` checks
parity against it.

* The fine index floor(x / 0.05) is always computed in float64 (on the host
  when the device has no float64, e.g. MPS), so points on cell boundaries
  land in exactly the same cell as in the NumPy engine. Everything after
  that is integer arithmetic.
* Per-point statistics are reduced on the device with `torch.unique` +
  `index_add_` / `scatter_reduce_` instead of sort + `reduceat` / `bincount`.
  The float math runs in `dtype` (float32 by default); roughness uses a
  two-pass variance so float32 does not lose it to cancellation.
* The persistent state keeps the NumPy engine's 16-byte-per-cell layout, as
  device tensors. Fusion and derived flags / cost are ported op for op; the
  box filter for the pothole reference is a separable `avg_pool2d` with zero
  padding (= scipy `uniform_filter(mode="constant")`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .grid import (FoveatedGrid, TierLayers, GROUND_MASK, COST_PRIOR, DRIVABLE, UNKNOWN,
                   F_OVERHANG, F_STEP, F_DEPRESSION, VEHICLE_CLEARANCE, STEP_THRESH,
                   DEPRESSION_THRESH, DEPRESSION_WIN)
from .sim import PERSON, VEHICLE

# clearance thresholds evaluated in float64 on the stored 2 cm code, exactly as the NumPy engine does
_CLEAR_CODES = np.arange(256) * 0.02
OVERHANG_LUT = (_CLEAR_CODES > 0.5) & (np.arange(256) != UNKNOWN)
PASS_UNDER_LUT = OVERHANG_LUT & (_CLEAR_CODES >= VEHICLE_CLEARANCE)


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

    def __init__(self, n, device):
        self.n, self.device = n, device
        for f, dt in self.FIELDS.items():
            if f == "count":
                self.count = torch.zeros((n, n), dtype=torch.int16, device=device).view(torch.uint16)
            else:
                setattr(self, f, torch.full((n, n), self.FILL[f], dtype=dt, device=device))
        self.eff_cls = torch.full((n, n), UNKNOWN, dtype=torch.uint8, device=device)

    @property
    def count16(self):
        return self.count.view(torch.int16)

    @property
    def nbytes(self):
        return sum(getattr(self, f).nelement() * getattr(self, f).element_size() for f in self.FIELDS)

    def shifted(self, d):
        """Copy of this state moved by d = (di, dj) cells (window scroll)."""
        out = TorchTierLayers(self.n, self.device)
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
        return out

    def to_numpy(self):
        """Host `grid.TierLayers` with the same contents (plus eff_cls)."""
        out = TierLayers(self.n)
        for f in self.FIELDS:
            v = self.count16.cpu().numpy().view(np.uint16) if f == "count" else getattr(self, f).cpu().numpy()
            getattr(out, f)[:] = v
        out.eff_cls = self.eff_cls.cpu().numpy()
        return out


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchFoveatedGrid(FoveatedGrid):
    def __init__(self, profile="spec", fuse=True, device=None, dtype=torch.float32):
        super().__init__(profile, fuse=fuse)
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        dev = self.device
        self._ground_mask = torch.as_tensor(GROUND_MASK, device=dev)
        self._cost_prior = torch.as_tensor(COST_PRIOR, device=dev)
        self._drivable = torch.as_tensor(DRIVABLE, device=dev)
        self._overhang = torch.as_tensor(OVERHANG_LUT, device=dev)
        self._pass_under = torch.as_tensor(PASS_UNDER_LUT, device=dev)
        self.state = [self._new_layers(t.n) for t in self.tiers]

    # ------------------------------------------------------------------ utils
    def _t(self, a, dtype):
        return torch.as_tensor(a).to(self.device, dtype)

    def fine_index_t(self, xy_world):
        """floor(xy / base) as int64 on the device, computed in float64."""
        if not torch.is_tensor(xy_world):
            return torch.from_numpy(self.fine_index(xy_world)).to(self.device)
        xy = xy_world if xy_world.device.type != "mps" else xy_world.cpu()
        return torch.floor(xy.double() / self.base).long().to(self.device)

    # -------------------------------------------------------------- binning
    def bin_points(self, xy_world, z, probs, moving, origins):
        """Device version of `FoveatedGrid.bin_points`: same keys, tensors as values."""
        dev, fdt = self.device, self.dtype
        f = self.fine_index_t(xy_world)
        z = self._t(z, fdt)
        P = self._t(probs, fdt)
        moving = self._t(moving, torch.bool)
        cls = P.argmax(1)
        is_ground = self._ground_mask[cls]
        static = ~moving
        out = []
        for t, org in zip(self.tiers, origins):
            ij = torch.div(f, t.ratio, rounding_mode="floor") - torch.as_tensor(org, device=dev)
            inw = ((ij >= 0) & (ij < t.n)).all(1)
            idx = inw.nonzero().squeeze(1)
            key = ij[idx, 0] * t.n + ij[idx, 1]
            uk, inv = torch.unique(key, sorted=True, return_inverse=True)
            M = uk.numel()
            zz, st = z[idx], static[idx]
            ng_pt = is_ground[idx]
            gg = ng_pt & st
            inf = torch.tensor(float("inf"), dtype=fdt, device=dev)

            def ssum(v):
                return torch.zeros((M,) + v.shape[1:], dtype=v.dtype, device=dev).index_add_(0, inv, v)

            def smin(v):
                return torch.full((M,), float("inf"), dtype=fdt, device=dev).scatter_reduce_(0, inv, v, "amin")

            def smax(v):
                return torch.full((M,), -float("inf"), dtype=fdt, device=dev).scatter_reduce_(0, inv, v, "amax")

            n_static = ssum(st.to(fdt))
            n_ground = ssum(gg.to(fdt))
            gmean = ssum(torch.where(gg, zz, 0)) / n_ground.clamp(min=1)
            gvar = ssum(torch.where(gg, zz - gmean[inv], 0) ** 2) / n_ground.clamp(min=1)
            nan = torch.tensor(float("nan"), dtype=fdt, device=dev)
            Pi = P[idx]
            mv = moving[idx]
            out.append(dict(
                key=uk, n_pts=ssum(torch.ones_like(inv)), n_static=n_static,
                z_min=smin(torch.where(st, zz, inf)), z_max=smax(torch.where(st, zz, -inf)),
                ground=torch.where(n_ground > 0, gmean, nan),
                rough=torch.where(n_ground > 1, gvar.sqrt(), nan),
                zmin_ng=smin(torch.where(st & ~ng_pt, zz, inf)),
                p_static=ssum(Pi * st[:, None]), p_all=ssum(Pi),
                n_dyn=ssum(mv.to(fdt)),
                n_dyn_person=ssum((mv & (cls[idx] == PERSON)).to(fdt)),
                n_in=int(idx.numel()),
            ))
        return out


    # --------------------------------------------------------------- update
    def _new_layers(self, n):
        return TorchTierLayers(n, self.device)

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

    def snapshot(self):
        """Host copy of the grid state (single packed device-to-host copy)."""
        parts = [getattr(s, attr).view(torch.uint8).reshape(-1)
                 for s in self.state for _, attr, _, _ in self.TIER_FIELD_SPECS]
        packed = torch.cat(parts).cpu().numpy()
        out = []
        offset = 0
        for s in self.state:
            tl = TierLayers.__new__(TierLayers)        # fields come from the packed buffer, skip allocating them
            tl.n = s.n
            n_cells = s.n * s.n
            for name, _, dt, b in self.TIER_FIELD_SPECS:
                sz = n_cells * b
                setattr(tl, name, packed[offset:offset + sz].view(dt).reshape(s.n, s.n))
                offset += sz
            out.append(tl)
        return out

    def _fuse_tier(self, t, s: TorchTierLayers, st):
        n, fdt = t.n, self.dtype
        key = st["key"]
        obs = st["n_static"] > 0
        k = key[obs]
        i, j = k // n, k % n
        # ---- class: blend new probabilities with the stored (cls, conf)
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
        s.cls[i, j] = q.argmax(1).to(torch.uint8)
        s.conf[i, j] = (q.max(1).values / q.sum(1).clamp(min=1e-9) * 255).clamp(0, 255).to(torch.uint8)
        # ground class (argmax over ground classes only), kept in the flags high nibble
        pg = torch.where(self._ground_mask[None, :], p, 0)
        gcls = torch.where(pg.sum(1) > 0.05, pg.argmax(1), 0xF)
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
        clear = st["zmin_ng"][obs] - g                  # nan / inf where either side is missing
        s.clear[i, j] = torch.where(clear.isfinite(), (clear / 0.02).clamp(0, 254), UNKNOWN).to(torch.uint8)
        keep_g = torch.where(gcls == 0xF, (s.flags[i, j] >> 4).long(), gcls)
        s.flags[i, j] = (keep_g << 4).to(torch.uint8)
        # ---- age
        seen = s.age != UNKNOWN
        s.age[seen] = (s.age[seen].int() + 1).clamp(max=UNKNOWN).to(torch.uint8)
        s.age[i, j] = 0
        # ---- dynamic layer: this frame only, never fused
        dyn_cells = st["n_dyn"] > 0
        dk = key[dyn_cells]
        dcls = torch.where(st["n_dyn_person"][dyn_cells] * 2 > st["n_dyn"][dyn_cells], PERSON, VEHICLE)
        return dict(i=dk // n, j=dk % n, cls=dcls)

    def _derive(self, t, s: TorchTierLayers):
        """Flags + traversability cost from the fused layers (vectorised over the tier)."""
        cls = s.cls.long()
        gcls = (s.flags >> 4).long()
        ground = s.ground.float()
        valid_g = ground.isfinite()
        clear = s.clear.long()
        # overhang: obstacle points start well above the ground
        overhang = self._overhang[clear] & valid_g
        passable_under = self._pass_under[clear] & valid_g & (gcls != 0xF)
        eff_cls = torch.where(passable_under, gcls, cls)
        # step edges: height jump to a neighbour 1 or 2 cells away
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
        stepf = step > STEP_THRESH
        # depressions (potholes): below the local mean of drivable ground
        drv_cls = self._drivable[eff_cls]
        drv = drv_cls & valid_g
        win = max(3, int(round(DEPRESSION_WIN / t.cell)) | 1)
        num = _box(torch.where(drv, gz, 0.0), win)
        den = _box(drv.float(), win)
        ref = torch.where(den > 0.05, num / den.clamp(min=1e-6), float("nan"))
        dep = drv & (gz < ref - DEPRESSION_THRESH)
        flags = (torch.where(overhang, F_OVERHANG, 0) | torch.where(stepf & valid_g, F_STEP, 0)
                 | torch.where(dep, F_DEPRESSION, 0))
        # cost
        cost = self._cost_prior[eff_cls]
        cost = torch.where(passable_under, cost + 20, cost)
        cost = torch.where(stepf & drv_cls, cost.clamp(min=180), cost)
        cost = torch.where(stepf & (cost < 180), cost.clamp(min=140), cost)
        cost = torch.where(dep, cost.clamp(min=170), cost)
        rough = s.rough.float()
        cost = torch.where(rough.isfinite() & (rough > 0.04), cost + 30, cost)
        cost = torch.where(s.conf < 150, cost + 25, cost)
        stale = (s.age != UNKNOWN) & (s.age > 20)
        cost = torch.where(stale, cost + 20, cost).clamp(0, 254)
        unknown = cls == UNKNOWN
        s.cost[:] = torch.where(unknown, UNKNOWN, cost).to(torch.uint8)
        s.flags[:] = (s.flags & 0xF0) | flags.to(torch.uint8)
        s.eff_cls = torch.where(unknown, UNKNOWN, eff_cls).to(torch.uint8)


def _box(x, win):
    """Separable zero-padded box mean, = scipy uniform_filter(x, win, mode="constant") for odd win."""
    x = x[None, None]
    x = F.avg_pool2d(x, (win, 1), stride=1, padding=(win // 2, 0), count_include_pad=True)
    x = F.avg_pool2d(x, (1, win), stride=1, padding=(0, win // 2), count_include_pad=True)
    return x[0, 0]


def stats_to_numpy(stats):
    """Per-tier stats dicts with tensors moved to host NumPy arrays."""
    return [{k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in s.items()} for s in stats]
