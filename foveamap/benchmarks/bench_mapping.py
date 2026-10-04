"""Production 1,000-Frame Mapping Engine Benchmark Runner.

Benchmarks FoveaMap's multi-tier foveated grid mapping engine (NumPy and PyTorch),
measuring latency percentiles (p50, p95, p99), sustained FPS, peak memory,
VRAM, points in/assigned/dropped, dynamic cell count, and stale cell count.

Usage:
    python benchmarks/bench_mapping.py --engine torch --frames 1000
    python benchmarks/bench_mapping.py --engine numpy --frames 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
import numpy as np

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from foveamap.grid import FoveatedGrid
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.core.config import FoveaMapConfig, GridConfig, TerrainConfig
from foveamap.core.ontology import NUM_CLASSES, VEHICLE, PERSON, ROAD, BUILDING, VEGETATION


def generate_synthetic_sweep(
    frame_idx: int,
    ego_xy: np.ndarray,
    n_pts: int = 50000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic classified LiDAR sweep with realistic static & dynamic objects."""
    if rng is None:
        rng = np.random.default_rng(frame_idx)

    # 1. Spatial distribution: dense near range, sparser far range
    radii = rng.power(0.5, n_pts) * 75.0  # up to 75m
    angles = rng.uniform(-np.pi, np.pi, n_pts)
    px = ego_xy[0] + radii * np.cos(angles)
    py = ego_xy[1] + radii * np.sin(angles)
    pz = rng.normal(-1.5, 0.4, n_pts)

    # 2. Semantic classes
    probs = np.zeros((n_pts, NUM_CLASSES), dtype=np.float32)
    moving = np.zeros(n_pts, dtype=bool)

    # Drivable road points near ground
    road_mask = (pz < -1.2) & (radii < 40.0)
    probs[road_mask, ROAD] = 0.92

    # Static buildings / obstacles at mid-range
    bldg_mask = (pz >= -1.2) & (radii > 15.0) & (radii < 60.0)
    probs[bldg_mask, BUILDING] = 0.88

    # Dynamic moving vehicle
    veh_center = ego_xy + np.array([12.0 + frame_idx * 0.2, 3.5])
    veh_dist = np.hypot(px - veh_center[0], py - veh_center[1])
    veh_mask = veh_dist < 2.5
    probs[veh_mask, VEHICLE] = 0.96
    moving[veh_mask] = True
    pz[veh_mask] = rng.uniform(-0.8, 0.8, veh_mask.sum())

    # Dynamic pedestrian
    ped_center = ego_xy + np.array([6.0, 5.0 - frame_idx * 0.05])
    ped_dist = np.hypot(px - ped_center[0], py - ped_center[1])
    ped_mask = ped_dist < 1.0
    probs[ped_mask, PERSON] = 0.90
    moving[ped_mask] = True
    pz[ped_mask] = rng.uniform(-1.0, 0.5, ped_mask.sum())

    # Fill remainder with vegetation
    rem_mask = ~(road_mask | bldg_mask | veh_mask | ped_mask)
    probs[rem_mask, VEGETATION] = 0.85
    # Normalize probabilities
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-6)

    xy_world = np.column_stack([px, py])
    return xy_world, pz.astype(np.float32), probs, moving


def run_mapping_benchmark(
    engine: str = "torch",
    profile: str = "spec",
    n_frames: int = 1000,
    n_pts: int = 50000,
    device: str = "cpu",
) -> dict:
    """Execute sustained multi-frame mapping benchmark."""
    print(f"============================================================")
    print(f"FOVEAMAP PHASE 5 MAPPING ENGINE BENCHMARK")
    print(f"Engine:    {engine.upper()}")
    print(f"Profile:   {profile}")
    print(f"Device:    {device}")
    print(f"Frames:    {n_frames}")
    print(f"Points/fr: {n_pts}")
    print(f"============================================================")

    # Initialize grid engine
    cfg = FoveaMapConfig()
    if engine == "torch":
        dev = torch.device(device)
        grid = TorchFoveatedGrid(profile=profile, device=dev, fuse=True, terrain_config=cfg.terrain)
    else:
        grid = FoveatedGrid(profile=profile, fuse=True, terrain_config=cfg.terrain)

    tracemalloc.start()
    latencies_ms: list[float] = []
    points_in_list: list[int] = []
    points_assigned_list: list[int] = []
    dropped_points_list: list[int] = []
    dynamic_cells_list: list[int] = []
    stale_cells_list: list[int] = []

    ego_xy = np.array([0.0, 0.0], dtype=np.float64)
    ego_vel = np.array([0.15, 0.02], dtype=np.float64)  # ~5.4 km/h

    rng = np.random.default_rng(42)

    # Warmup
    print("Warming up engine (10 frames)...")
    for w in range(10):
        xy, z, p, m = generate_synthetic_sweep(w, ego_xy, n_pts=n_pts, rng=rng)
        if engine == "torch":
            grid.update(xy, z, p, m, ego_xy)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            grid.update(xy, z, p, m, ego_xy)
    grid.reset()

    print(f"Benchmarking {n_frames} consecutive frames...")
    t_start = time.perf_counter()

    for f_idx in range(n_frames):
        ego_xy += ego_vel
        xy, z, p, m = generate_synthetic_sweep(f_idx, ego_xy, n_pts=n_pts, rng=rng)

        # Timed mapping step (binning + mip-up + fusion + terrain)
        t0 = time.perf_counter()
        if engine == "torch":
            dyn, stats = grid.update(xy, z, p, m, ego_xy)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
        else:
            dyn, stats = grid.update(xy, z, p, m, ego_xy)
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(t_elapsed)

        # Collect diagnostics
        diag = grid.last_diagnostics
        p_in = diag.get("points_in", len(xy))
        p_assigned = diag.get("native_points_assigned", 0)
        p_dropped = diag.get("filtered_points", 0)

        points_in_list.append(p_in)
        points_assigned_list.append(p_assigned)
        dropped_points_list.append(p_dropped)

        # Dynamic cells count
        n_dyn = sum(len(d["i"]) for d in dyn) if dyn else 0
        dynamic_cells_list.append(n_dyn)

        # Stale cells count
        if f_idx % 50 == 0 or f_idx == n_frames - 1:
            stale_cnt = 0
            for s in grid.state:
                if engine == "torch":
                    age = s.age
                    stale_cnt += int(((age != 255) & (age >= 20)).sum().item())
                else:
                    age = s.age
                    stale_cnt += int(((age != 255) & (age >= 20)).sum())
            stale_cells_list.append(stale_cnt)

        if (f_idx + 1) % 200 == 0 or f_idx == n_frames - 1:
            curr_fps = (f_idx + 1) / (time.perf_counter() - t_start)
            print(f"  Frame {f_idx + 1:4d}/{n_frames} | Latency: {t_elapsed:6.2f} ms | FPS: {curr_fps:5.1f}")

    total_time_s = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_vram_mb = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Compute percentiles
    lats = np.array(latencies_ms)
    p50 = float(np.percentile(lats, 50))
    p90 = float(np.percentile(lats, 90))
    p95 = float(np.percentile(lats, 95))
    p99 = float(np.percentile(lats, 99))
    p_mean = float(np.mean(lats))
    p_min = float(np.min(lats))
    p_max = float(np.max(lats))
    fps = n_frames / total_time_s

    mem_rep = grid.memory_report()

    summary = {
        "engine": engine,
        "device": device,
        "profile": profile,
        "total_frames": n_frames,
        "total_time_s": round(total_time_s, 3),
        "fps": round(fps, 1),
        "latency_ms": {
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "mean": round(p_mean, 2),
            "min": round(p_min, 2),
            "max": round(p_max, 2),
        },
        "points": {
            "total_points_in": int(np.sum(points_in_list)),
            "mean_points_per_frame": int(np.mean(points_in_list)),
            "mean_points_assigned": int(np.mean(points_assigned_list)),
            "mean_dropped_points": int(np.mean(dropped_points_list)),
        },
        "memory": {
            "bytes_per_cell": mem_rep["bytes_per_cell"],
            "allocated_cells": mem_rep["allocated_cells"],
            "allocated_bytes": mem_rep["allocated_bytes"],
            "allocated_mb": round(mem_rep["allocated_bytes"] / (1024 * 1024), 3),
            "effective_bytes": mem_rep["effective_bytes"],
            "peak_rss_mb": round(peak_mem / (1024 * 1024), 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "uniform_baseline_bytes": mem_rep["uniform_baseline_bytes"],
            "reduction_ratio": round(mem_rep["reduction_ratio"], 1),
            "under_8mb_target": mem_rep["under_8mb_target"],
        },
        "state_dynamics": {
            "mean_dynamic_cells": round(float(np.mean(dynamic_cells_list)), 1),
            "final_stale_cells": stale_cells_list[-1] if stale_cells_list else 0,
        },
    }

    print("\n============================================================")
    print("PHASE 5 BENCHMARK RESULTS")
    print("============================================================")
    print(f"Frames Processed:   {n_frames}")
    print(f"Sustained FPS:      {summary['fps']} Hz")
    print(f"Latency p50:        {summary['latency_ms']['p50']} ms")
    print(f"Latency p95:        {summary['latency_ms']['p95']} ms")
    print(f"Latency p99:        {summary['latency_ms']['p99']} ms")
    print(f"Latency mean:       {summary['latency_ms']['mean']} ms")
    print(f"Points In / Frame:  {summary['points']['mean_points_per_frame']}")
    print(f"Points Assigned:    {summary['points']['mean_points_assigned']}")
    print(f"Points Filtered:    {summary['points']['mean_dropped_points']}")
    print(f"Dynamic Cells:      {summary['state_dynamics']['mean_dynamic_cells']}")
    print(f"Allocated Map Mem:  {summary['memory']['allocated_mb']} MB (target <= 8 MB: {summary['memory']['under_8mb_target']})")
    print(f"Peak Working RSS:   {summary['memory']['peak_rss_mb']} MB")
    print(f"Peak VRAM:          {summary['memory']['peak_vram_mb']} MB")
    print(f"Reduction vs 5cm:   {summary['memory']['reduction_ratio']}x (target >= 30x)")
    print("============================================================\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoveaMap Phase 5 Mapping Engine Benchmark")
    parser.add_argument("--engine", choices=["torch", "numpy"], default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", choices=["spec", "graded"], default="spec")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--points", type=int, default=50000)
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    results = run_mapping_benchmark(
        engine=args.engine,
        profile=args.profile,
        n_frames=args.frames,
        n_pts=args.points,
        device=args.device,
    )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"Benchmark results written to {args.out}")
