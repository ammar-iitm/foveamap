"""Terrain & traversability shared primitives (Phase 7).

This module owns the *interpretation* of the foveated grid's terrain fields;
the mapping engines (`grid.FoveatedGrid`, `grid_torch.TorchFoveatedGrid`)
own the fields themselves and the per-frame derivation (`_derive`).

Field semantics (all tiers, both engines):
- ``ground`` (float16, metres): EMA-fused local surface elevation estimate.
  NaN means no ground evidence (unknown terrain, never safe terrain).
- ``z_min`` / ``z_max`` (float16, metres): observed vertical extent of
  static points in the cell (objects above ground included).
- ``rough`` (float16, metres): ground-height standard deviation (sigma)
  within the cell, fused with EMA and combined across mip-up levels by
  variance combination. NaN when fewer than 2 ground points support it.
- ``clear`` (uint8, 2 cm units): ``zmin_non_ground - ground`` clearance;
  UNKNOWN (255) when either endpoint is missing.
- ``cost`` (uint8, 0..254, UNKNOWN=255): deterministic traversability cost
  (see cost model below). UNKNOWN means insufficient evidence and is never
  traversable.
- ``cls`` / ``conf`` / ``flags``: semantic evidence and terrain flags
  (F_SLOPE, F_STEP, F_DEPRESSION, F_OVERHANG in the low nibble).

Slope convention: radians everywhere (``slope_threshold_rad``,
``slope_critical_rad``). ``slope = atan(|grad|)`` with gradients in
physical units (``tier resolution`` per cell step) — never pixel indices.

Cost model (identical in both engines, see ``_derive``):
``cost = semantic_prior[eff_cls]``
``+ 20`` if passable-under overhang, ``max(cost,180)`` for steps on drivable
(``max(cost,140)`` for steps elsewhere), ``max(cost,170)`` for depressions,
``+30`` if roughness exceeds threshold, slope excess ``0..40`` above
``slope_threshold_rad`` (``max(cost,220)`` at/above critical),
``+25`` if classification confidence < 150, ``+20`` if stale,
clipped to 254; UNKNOWN cells keep cost 255.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

UNKNOWN_COST = 255
# Fallback traversal boundary, used only when no TerrainConfig value is
# available (mirrors TerrainConfig.traversable_cost_max default). All grid,
# snapshot, and report APIs take the configured value; this constant is the
# documented default, not a second source of truth.
TRAVERSABLE_COST_MAX = 180


def is_traversable_cell(cost: int, dynamic: bool, known: bool, cost_max: int = TRAVERSABLE_COST_MAX) -> bool:
    """One authoritative traversability policy for every public API.

    UNKNOWN / OUT_OF_BOUNDS (known=False) -> False; dynamic occupancy ->
    False; lethal/high cost (cost >= cost_max, including UNKNOWN=255) ->
    False; otherwise True. STALE cells follow the same cost rule (staleness
    is already priced into cost via the stale penalty).
    """
    if not known:
        return False
    if dynamic:
        return False
    return int(cost) < int(cost_max)


def slope_at_cell(
    ground: np.ndarray,
    valid: np.ndarray,
    i: int,
    j: int,
    cell_m: float,
) -> float | None:
    """Slope magnitude at cell (i, j) in radians, or None if unknown.

    Replicates the neighbor-selection policy of the grid ``_derive`` methods
    exactly: one-sided differences at window edges, central differences when
    both neighbors are valid, and no estimate unless the centre is valid and
    at least one orthogonal neighbor is valid. Gradients use physical units
    (``cell_m`` per step), so every tier yields the same slope for the same
    physical surface.
    """
    n0, n1 = ground.shape
    if not (0 <= i < n0 and 0 <= j < n1):
        return None
    if not bool(valid[i, j]):
        return None
    gz = float(ground[i, j])
    if not np.isfinite(gz):
        return None

    def get(di: int, dj: int) -> float | None:
        ii, jj = i + di, j + dj
        if 0 <= ii < n0 and 0 <= jj < n1 and bool(valid[ii, jj]):
            v = float(ground[ii, jj])
            return v if np.isfinite(v) else None
        return None

    g_down, g_up = get(-1, 0), get(1, 0)
    g_right, g_left = get(0, -1), get(0, 1)
    # NOTE: axis convention mirrors _derive: axis 0 is treated as x.
    if g_down is not None and g_up is not None:
        dx = (g_down - g_up) / (2.0 * cell_m)
    elif g_down is not None:
        dx = (g_down - gz) / cell_m
    elif g_up is not None:
        dx = (gz - g_up) / cell_m
    else:
        dx = None
    if g_right is not None and g_left is not None:
        dy = (g_right - g_left) / (2.0 * cell_m)
    elif g_right is not None:
        dy = (g_right - gz) / cell_m
    elif g_left is not None:
        dy = (gz - g_left) / cell_m
    else:
        dy = None
    if dx is None and dy is None:
        return None
    dx = dx if dx is not None else 0.0
    dy = dy if dy is not None else 0.0
    return float(np.arctan(np.sqrt(dx * dx + dy * dy)))


def report_tier(
    ground: np.ndarray,
    rough: np.ndarray,
    cost: np.ndarray,
    dynamic_mask: np.ndarray,
    age: np.ndarray,
    cell_m: float,
    stale_age_threshold: int = 20,
    traversable_cost_max: int = TRAVERSABLE_COST_MAX,
) -> dict[str, Any]:
    """Aggregate terrain/traversability diagnostics for one tier (plain values).

    All inputs are host NumPy arrays from a single tier. Slope statistics are
    computed vectorially with the same central/edge policy basis as
    ``slope_at_cell`` (unknown slopes excluded, never zero-filled).
    Traversable counts obey the unified ``is_traversable_cell`` policy, so
    dynamic occupancy blocks traversal here exactly as in query APIs.
    """
    ground = np.asarray(ground, dtype=np.float64)
    valid = np.isfinite(ground)
    n_cells = int(valid.size)
    known = int(valid.sum())
    unknown = n_cells - known

    cost = np.asarray(cost)
    known_cost = cost != UNKNOWN_COST
    dyn = np.asarray(dynamic_mask, dtype=bool)
    trav_mask = known_cost & ~dyn & (cost < int(traversable_cost_max))
    traversable = int(trav_mask.sum())
    nontraversable = int((known_cost & ~trav_mask).sum())
    unknown_cost = int((~known_cost).sum())

    rough = np.asarray(rough, dtype=np.float64)
    rough_known = rough[np.isfinite(rough)]
    mean_rough = float(rough_known.mean()) if rough_known.size else None
    max_rough = float(rough_known.max()) if rough_known.size else None

    # Vectorized slope magnitude with edge-aware neighbor selection.
    slope = _slope_field(ground, valid, cell_m)
    slope_known = slope[np.isfinite(slope)]
    mean_slope = float(slope_known.mean()) if slope_known.size else None
    max_slope = float(slope_known.max()) if slope_known.size else None

    age = np.asarray(age)
    stale = int(((age != 255) & (age >= stale_age_threshold)).sum())
    dynamic = int(np.asarray(dynamic_mask, dtype=bool).sum())

    return {
        "cells": n_cells,
        "cell_m": float(cell_m),
        "known_ground": known,
        "unknown_ground": unknown,
        "known_ground_pct": round(100.0 * known / max(n_cells, 1), 3),
        "slope_cells": int(slope_known.size),
        "mean_slope_rad": None if mean_slope is None else round(mean_slope, 5),
        "max_slope_rad": None if max_slope is None else round(max_slope, 5),
        "mean_roughness_m": None if mean_rough is None else round(mean_rough, 5),
        "max_roughness_m": None if max_rough is None else round(max_rough, 5),
        "traversable_cells": traversable,
        "nontraversable_cells": nontraversable,
        "unknown_cost_cells": unknown_cost,
        "dynamic_cells": dynamic,
        "stale_cells": stale,
    }


def _slope_field(ground: np.ndarray, valid: np.ndarray, cell_m: float) -> np.ndarray:
    """Per-cell slope magnitude (radians); NaN where evidence is insufficient."""
    out = np.full(ground.shape, np.nan, dtype=np.float64)
    gz = np.where(valid, ground, np.nan)
    v = valid
    # Neighbor values with window-edge invalidation (mirrors _derive).
    g_d = np.roll(gz, -1, axis=0)
    v_d = np.roll(v, -1, axis=0)
    v_d[-1, :] = False
    g_u = np.roll(gz, 1, axis=0)
    v_u = np.roll(v, 1, axis=0)
    v_u[0, :] = False
    g_r = np.roll(gz, -1, axis=1)
    v_r = np.roll(v, -1, axis=1)
    v_r[:, -1] = False
    g_l = np.roll(gz, 1, axis=1)
    v_l = np.roll(v, 1, axis=1)
    v_l[:, 0] = False
    with np.errstate(invalid="ignore", divide="ignore"):
        dx_c = (g_d - g_u) / (2.0 * cell_m)
        dx_f = (g_d - gz) / cell_m
        dx_b = (gz - g_u) / cell_m
        dx = np.where(v_d & v_u, dx_c, np.where(v_d, dx_f, np.where(v_u, dx_b, 0.0)))
        dy_c = (g_r - g_l) / (2.0 * cell_m)
        dy_f = (g_r - gz) / cell_m
        dy_b = (gz - g_l) / cell_m
        dy = np.where(v_r & v_l, dy_c, np.where(v_r, dy_f, np.where(v_l, dy_b, 0.0)))
        mag = np.sqrt(dx * dx + dy * dy)
        has = v & ((v_d | v_u) | (v_r | v_l))
        out = np.where(has, np.arctan(mag), np.nan)
    return out


def report_layers(
    layers: Sequence[Any],
    cell_sizes_m: Sequence[float],
    stale_age_threshold: int = 20,
    traversable_cost_max: int = TRAVERSABLE_COST_MAX,
) -> dict[str, Any]:
    """Full-map terrain report from tier layer objects (plain values).

    Accepts NumPy ``TierLayers`` (or equivalent duck-typed) objects; Torch
    grids convert via their existing host snapshot first (diagnostic-only
    boundary, never hot path). Adds zero persistent state.
    """
    per_tier = [
        report_tier(
            np.asarray(s.ground),
            np.asarray(s.rough),
            np.asarray(s.cost),
            np.asarray(s.dynamic),
            np.asarray(s.age),
            float(cell_m),
            stale_age_threshold=int(stale_age_threshold),
            traversable_cost_max=int(traversable_cost_max),
        )
        for s, cell_m in zip(layers, cell_sizes_m)
    ]
    return {"per_tier": per_tier, "total": summarize_reports(per_tier)}


def summarize_reports(tier_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-tier reports into map totals (plain values)."""
    total_cells = sum(r["cells"] for r in tier_reports)
    known = sum(r["known_ground"] for r in tier_reports)
    slope_n = sum(r["slope_cells"] for r in tier_reports)
    max_slope = max((r["max_slope_rad"] for r in tier_reports if r["max_slope_rad"] is not None), default=None)
    # Weighted mean over slope-bearing cells (per-tier means are 5-decimal
    # rounded diagnostics; sufficient for aggregate reporting).
    mean_num = sum((r["mean_slope_rad"] or 0.0) * r["slope_cells"] for r in tier_reports)
    mean_slope = round(mean_num / slope_n, 5) if slope_n else None
    return {
        "tiers": len(tier_reports),
        "total_cells": total_cells,
        "known_ground": known,
        "unknown_ground": sum(r["unknown_ground"] for r in tier_reports),
        "known_ground_pct": round(100.0 * known / max(total_cells, 1), 3),
        "slope_cells": slope_n,
        "mean_slope_rad": mean_slope,
        "max_slope_rad": max_slope,
        "traversable_cells": sum(r["traversable_cells"] for r in tier_reports),
        "nontraversable_cells": sum(r["nontraversable_cells"] for r in tier_reports),
        "unknown_cost_cells": sum(r["unknown_cost_cells"] for r in tier_reports),
        "dynamic_cells": sum(r["dynamic_cells"] for r in tier_reports),
        "stale_cells": sum(r["stale_cells"] for r in tier_reports),
    }
