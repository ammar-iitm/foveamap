"""Snapshot -> ROS output payloads (plain-data builders, no rclpy).

Topics:
- ``/foveamap/grid``: decimated per-tier cell summaries built from the public
  snapshot/query contract (tier, centre xy, state, class, cost, ground,
  dynamic, traversability). Complete snapshots are published for correctness;
  ``serialized_size_bytes`` is measured so bandwidth is observable, and the
  builder caps cells per tier (``max_cells_per_tier``) deterministically
  (uniform stride) for bandwidth control. Deltas are future work.
- ``/foveamap/points_labeled``: per-point (x, y, z, class, confidence,
  dynamic) plus header; serializable via :func:`build_pointcloud2`.
- ``/foveamap/metrics``: the diagnostics payload (see diagnostics.py).

Internal ``TierLayers`` arrays are only read, never retained: payloads own
fresh lists/scalars. Timestamps/frame IDs come from the snapshot.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from foveamap.core.contracts import MapSnapshot
from .pointcloud import build_pointcloud2, ros_stamp_from_seconds, RosPointCloud2

UNKNOWN = 255


def _tier_cell_rows(tier: Any, origin: Any, cell_m: float, max_cells: int | None,
                    cost_max: int) -> list[dict[str, Any]]:
    n = int(getattr(tier, "n", 0))
    if n <= 0:
        return []
    ox, oy = int(origin[0]), int(origin[1])
    stride = 1
    if max_cells is not None and max_cells > 0:
        stride = max(1, int(np.ceil((n * n) / float(max_cells))))
    rows: list[dict[str, Any]] = []
    count = np.asarray(tier.count)
    cls = np.asarray(tier.cls)
    cost = np.asarray(tier.cost)
    ground = np.asarray(tier.ground, dtype=np.float64)
    dyn = np.asarray(tier.dynamic, dtype=bool)
    for i in range(0, n, stride):
        for j in range(0, n, stride):
            c = int(cost[i, j])
            if int(count[i, j]) == 0 and int(cls[i, j]) == UNKNOWN and not bool(dyn[i, j]):
                continue  # skip empty unknown cells (bandwidth)
            g = float(ground[i, j]) if np.isfinite(ground[i, j]) else None
            rows.append({
                "i": int(i),
                "j": int(j),
                "x": float((ox + i + 0.5) * cell_m),
                "y": float((oy + j + 0.5) * cell_m),
                "cls": int(cls[i, j]),
                "cost": c,
                "ground": g,
                "dynamic": bool(dyn[i, j]),
                "traversable": bool(c != UNKNOWN and c < cost_max and not bool(dyn[i, j])),
            })
    return rows


def build_grid_payload(
    snapshot: MapSnapshot,
    *,
    max_cells_per_tier: int | None = 20000,
    cost_max: int = 180,
) -> dict[str, Any]:
    """Build the ``/foveamap/grid`` payload from a detached snapshot."""
    tiers: list[dict[str, Any]] = []
    for idx, tier in enumerate(snapshot.tier_states):
        cell_m = float(getattr(tier, "cell", getattr(tier, "r", 0.0)) or 0.0)
        if not cell_m:
            cfgs = snapshot.metadata.get("tier_configs", [])
            if idx < len(cfgs):
                cell_m = float(cfgs[idx].get("cell_size_m", 0.0))
        rows = _tier_cell_rows(tier, snapshot.origins[idx], cell_m, max_cells_per_tier, int(cost_max))
        tiers.append({"tier": idx, "cell_m": cell_m, "cells": rows})
    payload = {
        "stamp_sec": int(np.floor(float(snapshot.timestamp))),
        "stamp_nanosec": int(round((float(snapshot.timestamp) % 1.0) * 1e9)),
        "frame_id": str(snapshot.frame_id),
        "origins": [[int(a), int(b)] for a, b in snapshot.origins],
        "tiers": tiers,
    }
    payload["serialized_size_bytes"] = len(json.dumps(payload).encode("utf-8"))
    return payload


def build_points_payload(
    xyz_world: np.ndarray,
    classes: np.ndarray,
    confidence: np.ndarray,
    is_moving: np.ndarray,
    *,
    frame_id: str = "",
    timestamp: float = 0.0,
    max_points: int = 12000,
) -> RosPointCloud2:
    """Build the ``/foveamap/points_labeled`` cloud (deterministic stride)."""
    xyz = np.asarray(xyz_world, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    stride = max(1, int(np.ceil(n / float(max_points)))) if max_points and max_points > 0 else 1
    sel = np.arange(0, n, stride)
    stamp = ros_stamp_from_seconds(float(timestamp))
    return build_pointcloud2(
        xyz[sel],
        extra={
            "label": (np.asarray(classes).reshape(-1)[sel].astype(np.int32), 5),
            "confidence": (np.asarray(confidence).reshape(-1)[sel].astype(np.float32), 7),
            "moving": (np.asarray(is_moving).reshape(-1)[sel].astype(np.uint8), 2),
        },
        frame_id=str(frame_id),
        stamp_sec=stamp.sec,
        stamp_nanosec=stamp.nanosec,
    )


def payload_size_bytes(payload: dict[str, Any]) -> int:
    """Measure JSON-serialized size (bandwidth observability)."""
    return len(json.dumps(payload).encode("utf-8"))
