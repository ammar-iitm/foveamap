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
"""
from __future__ import annotations

import numpy as np
import torch

from .grid import FoveatedGrid, GROUND_MASK
from .sim import PERSON


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchFoveatedGrid(FoveatedGrid):
    def __init__(self, profile="spec", fuse=True, device=None, dtype=torch.float32):
        super().__init__(profile, fuse=fuse)
        self.device = torch.device(device) if device is not None else default_device()
        self.dtype = dtype
        self._ground_mask = torch.as_tensor(GROUND_MASK, device=self.device)

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


def stats_to_numpy(stats):
    """Per-tier stats dicts with tensors moved to host NumPy arrays."""
    return [{k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in s.items()} for s in stats]
