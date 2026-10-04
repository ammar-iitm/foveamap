"""Unified command-line interface for FoveaMap (Phase 11).

Provides developer, operator, and deployment commands:
- foveamap info: Inspect platform, CUDA, backends, sources, and profiles.
- foveamap demo: Run canonical live perception & mapping demonstration.
- foveamap compare: Reproducible FoveaMap vs. uniform 5 cm grid benchmark.
- foveamap run: Ingest and process LiDAR files or directories.
- foveamap serve: Launch local HTTP API and interactive dashboard.
- foveamap bench: Run sustained multi-frame mapping benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core.config import FoveaMapConfig, GridConfig
from .core.contracts import LiDARFrame
from .core.ontology import CANONICAL_CLASSES
from .data.factory import list_sources, create_source
from .runtime.runtime import FoveaMapRuntime
from .runtime.perception import list_perception_backends
from .benchmarks.baseline import compare_uniform_baseline
from .sdk.client import FoveaMap
from .sdk.http import FoveaMapHttpServer
from .sdk.types import SnapshotView


def cmd_info(args: argparse.Namespace) -> int:
    """Display system diagnostics, detected hardware, engines, and profiles."""
    print("=" * 60)
    print("FoveaMap System Diagnostics & Runtime Capabilities")
    print("=" * 60)
    print(f"Python Version:       {sys.version.split()[0]} ({sys.platform})")
    print(f"PyTorch Version:      {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available:       {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device Count:    {torch.cuda.device_count()}")
        print(f"Current Device:       {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"Apple Silicon MPS:    Available")
    else:
        print(f"Hardware Mode:        CPU-Only Execution")

    print("\n--- Available Deployment Profiles ---")
    profiles = ["cpu_dev", "gpu_dev", "benchmark", "demo", "ros2"]
    for p in profiles:
        print(f"  * {p:<12} (FoveaMapConfig.{p}())")

    print("\n--- Registered Data Sources ---")
    for s in list_sources():
        print(f"  * {s}")

    print("\n--- Registered Perception Backends ---")
    for b in list_perception_backends():
        print(f"  * {b}")

    print("\n--- Standard Grid Profiles ---")
    for g in ("spec", "graded", "uniform5", "uniform50"):
        cfg = GridConfig.from_preset(g)
        n_tiers = len(cfg.tiers)
        res_str = ", ".join(f"{t.cell_size_m*100:.0f}cm (+/-{t.half_extent_m:.0f}m)" for t in cfg.tiers)
        print(f"  * {g:<10} ({n_tiers} tier{'s' if n_tiers>1 else ''}: {res_str})")

    print("=" * 60)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Run reproducible comparison against uniform 5 cm baseline."""
    print("Executing reproducible FoveaMap vs. Uniform Baseline Comparison...")
    rep = compare_uniform_baseline(
        n_points=args.points,
        profile=args.profile,
        run_empirical=not args.no_empirical,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(rep, indent=2))
        return 0

    print("\n" + "=" * 65)
    print(f"FOVEAMAP vs. UNIFORM 5 CM BASELINE (Profile: {rep['foveamap_profile']})")
    print("=" * 65)
    print(f"{'Metric':<30} | {'FoveaMap':<15} | {'Uniform 5cm':<15}")
    print("-" * 65)
    print(f"{'Cell Count':<30} | {rep['foveamap_cells']:<15,d} | {rep['uniform_cells']:<15,d}")
    print(f"{'Memory (MB)':<30} | {rep['foveamap_memory_mb']:<15.2f} | {rep['uniform_memory_mb']:<15.2f}")
    print(f"{'Memory Reduction Ratio':<30} | {rep['memory_saving_ratio']:<15.1f}x | 1.0x (baseline)")
    print(f"{'Under 8 MB Target':<30} | {str(rep['under_8mb_target']):<15} | False")
    print(f"{'Exceeds 30x Saving Target':<30} | {str(rep['exceeds_30x_target']):<15} | False")

    if "empirical" in rep:
        emp = rep["empirical"]
        print("-" * 65)
        print(f"{'Points In Sweep':<30} | {emp['n_points']:<15,d} | {emp['n_points']:<15,d}")
        print(f"{'Update Latency (ms)':<30} | {emp['foveamap_update_ms']:<15.2f} | {emp['uniform_update_ms']:<15.2f}")
        print(f"{'Update Speedup Ratio':<30} | {emp['update_speedup_ratio']:<15.1f}x | 1.0x (baseline)")
        print(f"{'Throughput (FPS)':<30} | {emp['foveamap_fps']:<15.1f} | {emp['uniform_fps']:<15.1f}")
    print("=" * 65 + "\n")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run canonical live demonstration using simulated 64-beam LiDAR frames."""
    print("=" * 65)
    print("Starting FoveaMap Canonical Demonstration")
    print("=" * 65)

    cfg = FoveaMapConfig.demo()
    runtime = FoveaMapRuntime(config=cfg)
    runtime.start()

    print(f"Device:       {runtime.device_ctx.device}")
    print(f"Grid Engine:  {runtime.config.runtime.grid_engine}")
    print(f"Perception:   {runtime.perception.name}")
    print(f"Frames:       {args.frames}")

    source = create_source("sim", n_steps=args.frames, seed=args.seed)
    print(f"Source:       SimulatorSource (64-beam LiDAR, procedural street)")
    print("-" * 65)

    last_snap = None
    t_start = time.perf_counter()

    for idx, frame in enumerate(source):
        t0 = time.perf_counter()
        snap = runtime.process(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        last_snap = snap

        timing_str = " ".join(f"{k}:{v*1000:.1f}ms" for k, v in runtime.last_timing.items() if k != "total")
        n_dyn = len(snap.dynamic_cells)
        n_tracks = len(snap.dynamic_tracks)
        print(f"Frame {idx+1:2d}/{args.frames} | pts: {frame.num_points:5d} | total: {dt:5.1f}ms | dyn: {n_dyn:3d} | tracks: {n_tracks:2d} | {timing_str}")

    total_time = time.perf_counter() - t_start
    avg_fps = args.frames / max(total_time, 1e-4)

    print("-" * 65)
    print(f"Processed {args.frames} frames in {total_time:.2f}s ({avg_fps:.1f} FPS)")

    # Execute canonical sample queries
    print("\n--- Authoritative 2.5D Map Queries ---")
    view = SnapshotView(last_snap)
    q_ego = view.query_point(0.0, 0.0)
    cls_id = q_ego.dominant_class
    cls_name = CANONICAL_CLASSES[cls_id] if cls_id is not None and 0 <= cls_id < len(CANONICAL_CLASSES) else "UNKNOWN"
    res_cm = f"{q_ego.cell_size_m * 100:.0f}cm" if q_ego.cell_size_m else "N/A"
    elev = f"{q_ego.ground_m:.2f} m" if q_ego.ground_m is not None else "N/A"
    q_fwd = view.query_point(5.0, 0.0)
    cls_id_f = q_fwd.dominant_class
    cls_name_f = CANONICAL_CLASSES[cls_id_f] if cls_id_f is not None and 0 <= cls_id_f < len(CANONICAL_CLASSES) else "UNKNOWN"
    res_cm_f = f"{q_fwd.cell_size_m * 100:.0f}cm" if q_fwd.cell_size_m else "N/A"
    elev_f = f"{q_fwd.ground_m:.2f} m" if q_fwd.ground_m is not None else "N/A"
    print(f"Query 5m Forward (5.0, 0.0):")
    print(f"  * Tier:            {q_fwd.tier} (resolution {res_cm_f})")
    print(f"  * Semantic Class:  {cls_name_f} (id={cls_id_f})")
    print(f"  * Elevation:       {elev_f}")
    print(f"  * Traversable:     {q_fwd.traversable} (cost={q_fwd.cost})")
    print(f"  * Cell State:      {q_fwd.state}")


    # Metrics summary
    metrics = runtime.get_metrics()
    mem = metrics.get("memory", {})
    print("\n--- Memory & Resolution Invariants ---")
    print(f"  * Foveated Memory:  {mem.get('allocated_bytes', 0) / (1024*1024):.2f} MB (<= 8 MB target: {mem.get('under_8mb_target')})")
    print(f"  * Baseline Memory:  {mem.get('uniform_baseline_bytes', 0) / (1024*1024):.2f} MB")
    print(f"  * Reduction Ratio:  {mem.get('reduction_ratio', 0):.1f}x (>= 30x target: {mem.get('reduction_ratio', 0) >= 30})")

    if args.serve:
        print("\nStarting HTTP dashboard server...")
        sdk_session = FoveaMap(config=cfg, runtime=runtime)
        sdk_session._snapshot = last_snap
        dashboard_dir = Path(__file__).parents[1] / "dashboard"
        server = FoveaMapHttpServer(sdk_session, host=args.host, port=args.port, dashboard_dir=dashboard_dir)
        url = server.start_background()
        print(f"Dashboard available at: {url}")
        print("Press Ctrl+C to stop server.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping server...")
            server.stop()

    print("=" * 65)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch FoveaMap HTTP API server with optional interactive dashboard."""
    cfg = FoveaMapConfig.from_env()
    sdk = FoveaMap(config=cfg)
    sdk.configure()
    sdk.start()

    dashboard_dir = None
    if args.dashboard:
        dashboard_dir = Path(__file__).parents[1] / "dashboard"
        if not dashboard_dir.is_dir():
            print(f"Warning: Dashboard directory {dashboard_dir} not found; serving API only.")
            dashboard_dir = None

    server = FoveaMapHttpServer(sdk, host=args.host, port=args.port, dashboard_dir=dashboard_dir)
    url = server.start_background()
    print("=" * 65)
    print(f"FoveaMap HTTP Service Active")
    print(f"Base URL:     {url}")
    print(f"Endpoints:    {url}/health, {url}/status, {url}/metrics, {url}/map/query, {url}/map/snapshot")
    if dashboard_dir:
        print(f"Dashboard:    {url}/")
    print("Press Ctrl+C to terminate.")
    print("=" * 65)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping FoveaMap HTTP Server...")
        server.stop()
        sdk.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run FoveaMap pipeline on a file (.pcd, .bin, .npy) or directory."""
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist.", file=sys.stderr)
        return 1

    cfg = FoveaMapConfig.from_env()
    runtime = FoveaMapRuntime(config=cfg)
    runtime.start()

    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".pcd":
            source = create_source("pcd", path=input_path)
        elif suffix == ".bin":
            source = create_source("bin", path=input_path)
        elif suffix == ".npy":
            source = create_source("npy", path=input_path)
        else:
            print(f"Error: Unsupported file format '{suffix}'. Supported: .pcd, .bin, .npy", file=sys.stderr)
            return 1
    else:
        source = create_source("sequence", directory=input_path)

    print(f"Processing {len(source)} frame(s) from {input_path.name}...")
    for i, frame in enumerate(source):
        t0 = time.perf_counter()
        snap = runtime.process(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"Frame {i+1:3d}/{len(source)} | id: {frame.frame_id} | pts: {frame.num_points:5d} | time: {dt:5.1f}ms")

    metrics = runtime.get_metrics()
    print(f"\nCompleted {metrics['frames_processed']} frames. FPS: {metrics['fps']}")
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"Saved metrics to {out_path}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Run sustained mapping engine benchmark."""
    from .benchmarks import run_mapping_benchmark

    print(f"Starting {args.frames}-frame benchmark on engine={args.engine}...")
    summary = run_mapping_benchmark(
        engine=args.engine,
        profile=args.profile,
        n_frames=args.frames,
        n_pts=args.points,
        device=args.device,
    )
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved benchmark summary to {out_path}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create unified CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="foveamap",
        description="FoveaMap: Adaptive Variable-Resolution 2.5D LiDAR Mapping",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # info
    p_info = subparsers.add_parser("info", help="Display platform, GPU, backend, and profile capabilities")
    p_info.set_defaults(func=cmd_info)

    # demo
    p_demo = subparsers.add_parser("demo", help="Run canonical live perception and mapping demonstration")
    p_demo.add_argument("--frames", type=int, default=15, help="Number of simulated LiDAR frames to process (default: 15)")
    p_demo.add_argument("--seed", type=int, default=42, help="Random seed for deterministic replay")
    p_demo.add_argument("--serve", action="store_true", help="Launch interactive dashboard server after demo")
    p_demo.add_argument("--host", default="127.0.0.1", help="Dashboard host (default: 127.0.0.1)")
    p_demo.add_argument("--port", type=int, default=8080, help="Dashboard port (default: 8080)")
    p_demo.set_defaults(func=cmd_demo)

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare FoveaMap against uniform 5 cm baseline")
    p_comp.add_argument("--points", type=int, default=50000, help="Points per test sweep (default: 50,000)")
    p_comp.add_argument("--profile", choices=["spec", "graded"], default="spec", help="FoveaMap profile (default: spec)")
    p_comp.add_argument("--no-empirical", action="store_true", help="Skip live execution latency measurement")
    p_comp.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_comp.add_argument("--json", action="store_true", help="Output raw JSON instead of table")
    p_comp.set_defaults(func=cmd_compare)

    # run
    p_run = subparsers.add_parser("run", help="Process point cloud files or sequences")
    p_run.add_argument("input", help="Path to .pcd, .bin, .npy file or sequence directory")
    p_run.add_argument("--out", default=None, help="Optional output JSON path for run metrics")
    p_run.set_defaults(func=cmd_run)

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch local HTTP API and dashboard server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Loopback host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_serve.add_argument("--dashboard", action="store_true", default=True, help="Serve interactive web dashboard")
    p_serve.add_argument("--api-only", dest="dashboard", action="store_false", help="Serve API endpoints only")
    p_serve.set_defaults(func=cmd_serve)

    # bench
    p_bench = subparsers.add_parser("bench", help="Run sustained multi-frame mapping benchmark")
    p_bench.add_argument("--engine", choices=["numpy", "torch"], default="numpy", help="Grid engine (default: numpy)")
    p_bench.add_argument("--device", default="cpu", help="Device (default: cpu)")
    p_bench.add_argument("--profile", choices=["spec", "graded"], default="spec", help="Grid profile (default: spec)")
    p_bench.add_argument("--frames", type=int, default=100, help="Frames to benchmark (default: 100)")
    p_bench.add_argument("--points", type=int, default=50000, help="Points per frame (default: 50000)")
    p_bench.add_argument("--out", default=None, help="Output JSON results path")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
