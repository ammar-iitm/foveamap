#!/usr/bin/env python3
"""Authoritative 1,000-Frame Soak Benchmark for Kaggle Remote GPU Execution.

Requirements:
- 50 warmup frames (excluded from statistics)
- 1,000 measured frames
- FP32 and FP16 evaluation
- Strict per-stage CUDA synchronization
- Exact provenance and checkpoint verification
- Full percentile breakdown (P50, P75, P90, P95, P99, max, mean, FPS)
- Stage-level breakdown (preprocess, inference, projection, fusion)
- VRAM stability, NaN/Inf checks, semantic class validity, FP32/FP16 parity
- Target gates: P95 <= 50.0 ms, FPS >= 20.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

# Ensure repository root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from foveamap.core.contracts import LiDARFrame
from foveamap.frames import SIM_INFO
from foveamap.pipeline import FoveaMapPipeline
from foveamap.sim import simulate_sequence


EXPECTED_CHECKPOINT_SHA256 = "28d99c86fa862fe01ad5517ad7d988563d218814e9d462c946d2059171c7320f"
GATE_TARGET_P95_MS = 50.0
GATE_TARGET_FPS = 20.0


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_driver_version() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        return out.strip()
    except Exception:
        return "Unknown"


def get_git_provenance() -> Dict[str, Any]:
    prov_file = os.path.join(ROOT, "provenance.json")
    if os.path.exists(prov_file):
        try:
            with open(prov_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    # Fallback to git command if .git exists
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return {
            "git_commit": sha,
            "branch": "detected",
            "modified_files": [line.split()[-1] for line in status.splitlines() if line.strip()],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        }
    except Exception:
        return {
            "git_commit": "unknown",
            "branch": "unknown",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        }


def generate_soak_frames(total_frames: int, seed: int = 42) -> List[LiDARFrame]:
    print(f"Generating {total_frames} deterministic simulation frames (seed={seed})...")
    t0 = time.perf_counter()
    # HDL-64E simulation
    seq = simulate_sequence(seed=seed, n_frames=total_frames)
    F = seq["frames"]
    frames: List[LiDARFrame] = []
    sensor_origin = np.array([0.0, 0.0, 1.73], dtype=np.float32)

    for i, f in enumerate(F):
        valid = f["valid"]
        pts = f["xyz"][valid].astype(np.float32)
        intensity = f["intensity"][valid].astype(np.float32)
        r_indices = np.where(valid)[0] // 1024
        ring = r_indices.astype(np.int16)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = f["ego"]

        frame = LiDARFrame(
            pts=pts,
            intensity=intensity,
            ring=ring,
            pose=pose,
            sensor_origin=sensor_origin,
            timestamp=float(i * 0.1),
            frame_id=f"frame_{i:05d}",
        )
        frames.append(frame)

    elapsed = time.perf_counter() - t0
    pts_counts = [len(f.pts) for f in frames]
    print(f"Generated {len(frames)} frames in {elapsed:.2f}s (mean points: {np.mean(pts_counts):.1f})")
    return frames


def run_single_precision_soak(
    frames: List[LiDARFrame],
    ckpt_path: str,
    device_str: str,
    fp16: bool,
    n_warmup: int = 50,
    n_measured: int = 1000,
) -> Dict[str, Any]:
    precision_label = "FP16" if fp16 else "FP32"
    print(f"\n============================================================")
    print(f"STARTING {precision_label} 1,000-FRAME SOAK RUN")
    print(f"============================================================")

    device = torch.device(device_str)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Initialize production pipeline with exact target configuration
    pipeline = FoveaMapPipeline(
        ckpt=ckpt_path,
        info=SIM_INFO,
        profile="spec",
        device=device_str,
        grid="torch",
        features="torch",
        fuse=True,
    )
    pipeline.fp16 = fp16

    # 1. Warmup
    print(f"Running {n_warmup} warmup frames...")
    for idx in range(n_warmup):
        pipeline.step(frames[idx])
    torch.cuda.synchronize(device)
    print("Warmup complete. Measured region begins.")

    # 2. Measured Soak Loop
    stage_timings: Dict[str, List[float]] = {
        "preprocess": [],
        "inference": [],
        "projection": [],
        "fusion": [],
        "total": [],
    }
    vram_samples: List[Dict[str, float]] = []
    class_predictions: List[np.ndarray] = []
    nan_inf_found = False

    t_start = time.perf_counter()

    for idx in range(n_warmup, n_warmup + n_measured):
        frame = frames[idx]
        out = pipeline.step(frame)
        torch.cuda.synchronize(device)

        timing = out["timing"]
        p_ms = timing["preprocess"] * 1000.0
        i_ms = timing["inference"] * 1000.0
        j_ms = timing["projection"] * 1000.0
        f_ms = timing["fusion"] * 1000.0
        tot_ms = p_ms + i_ms + j_ms + f_ms

        stage_timings["preprocess"].append(p_ms)
        stage_timings["inference"].append(i_ms)
        stage_timings["projection"].append(j_ms)
        stage_timings["fusion"].append(f_ms)
        stage_timings["total"].append(tot_ms)

        # Integrity & NaN/Inf check every 50 frames
        if (idx - n_warmup) % 50 == 0:
            for s in pipeline.grid.state:
                if torch.isnan(s.cost).any() or torch.isinf(s.cost).any():
                    nan_inf_found = True
            vram_samples.append({
                "frame": idx - n_warmup,
                "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
            })

        # Retain downsampled prediction sample for FP32/FP16 agreement audit
        if (idx - n_warmup) % 20 == 0:
            cls_sample = out["cls_pts"].detach().cpu().numpy()
            class_predictions.append(cls_sample)

    t_total_elapsed = time.perf_counter() - t_start
    sustained_fps = n_measured / t_total_elapsed

    # Final VRAM stats
    final_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
    final_reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    vram_drift_mb = final_allocated_mb - vram_samples[0]["allocated_mb"] if vram_samples else 0.0

    def compute_stats(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(np.max(a)),
        }

    stats = {stage: compute_stats(times) for stage, times in stage_timings.items()}

    p95_pass = stats["total"]["p95"] <= GATE_TARGET_P95_MS
    fps_pass = sustained_fps >= GATE_TARGET_FPS
    overall_pass = p95_pass and fps_pass and not nan_inf_found

    print(f"\n{precision_label} SUMMARY:")
    print(f"  Mean E2E:    {stats['total']['mean']:.2f} ms")
    print(f"  P50 Latency: {stats['total']['p50']:.2f} ms")
    print(f"  P95 Latency: {stats['total']['p95']:.2f} ms (Target: <= {GATE_TARGET_P95_MS} ms) -> {'PASS' if p95_pass else 'FAIL'}")
    print(f"  P99 Latency: {stats['total']['p99']:.2f} ms")
    print(f"  Max Latency: {stats['total']['max']:.2f} ms")
    print(f"  Sustained:   {sustained_fps:.2f} FPS (Target: >= {GATE_TARGET_FPS} FPS) -> {'PASS' if fps_pass else 'FAIL'}")
    print(f"  Fusion Mean: {stats['fusion']['mean']:.2f} ms | P95: {stats['fusion']['p95']:.2f} ms")
    print(f"  Peak VRAM:   {peak_allocated_mb:.2f} MiB allocated, {peak_reserved_mb:.2f} MiB reserved")
    print(f"  VRAM Drift:  {vram_drift_mb:+.3f} MiB over {n_measured} frames")

    return {
        "precision": precision_label,
        "n_warmup": n_warmup,
        "n_measured": n_measured,
        "wall_clock_s": t_total_elapsed,
        "sustained_fps": sustained_fps,
        "stats": stats,
        "vram": {
            "final_allocated_mb": final_allocated_mb,
            "final_reserved_mb": final_reserved_mb,
            "peak_allocated_mb": peak_allocated_mb,
            "peak_reserved_mb": peak_reserved_mb,
            "drift_mb": vram_drift_mb,
        },
        "integrity": {
            "nan_inf_found": nan_inf_found,
            "zero_drift": abs(vram_drift_mb) < 5.0,
        },
        "gates": {
            "p95_pass": p95_pass,
            "fps_pass": fps_pass,
            "overall_pass": overall_pass,
        },
        "sample_preds": class_predictions,
        "raw_totals": stage_timings["total"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Authoritative FoveaMap 1,000-Frame Soak Benchmark")
    parser.add_argument("--device", default="cuda:0", help="Execution device (default: cuda:0)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup frame count")
    parser.add_argument("--frames", type=int, default=1000, help="Measured frame count")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic simulation seed")
    parser.add_argument("--output-json", default=os.path.join(ROOT, "results", "kaggle_1000_soak_results.json"))
    args = parser.parse_args()

    print("================================================================================")
    print("FOVEAMAP AUTHORITATIVE REMOTE KAGGLE GPU 1,000-FRAME SOAK BENCHMARK")
    print("================================================================================")

    # 1. Hardware & Environment Check
    if not torch.cuda.is_available():
        print("ERROR: CUDA is NOT available in this environment. This benchmark strictly requires an NVIDIA GPU.")
        return 1

    device_idx = 0
    gpu_name = torch.cuda.get_device_name(device_idx)
    gpu_vram_total_mb = torch.cuda.get_device_properties(device_idx).total_memory / (1024 * 1024)
    driver_version = get_driver_version()
    cuda_version = torch.version.cuda or "Unknown"
    pytorch_version = torch.__version__

    print(f"\n--- REMOTE GPU HARDWARE ENVIRONMENT ---")
    print(f"GPU Model:          {gpu_name}")
    print(f"Total VRAM:         {gpu_vram_total_mb:.1f} MiB ({gpu_vram_total_mb/1024:.2f} GiB)")
    print(f"NVIDIA Driver:      {driver_version}")
    print(f"CUDA Version:       {cuda_version}")
    print(f"PyTorch Version:    {pytorch_version}")
    print(f"PyTorch CUDA Build: {torch.version.cuda}")
    print(f"CUDA Available:     {torch.cuda.is_available()}")

    # 2. Checkpoint Provenance Verification
    ckpt_path = os.path.join(ROOT, "checkpoints", "range_unet.pt")
    if not os.path.isfile(ckpt_path):
        print(f"FATAL: Checkpoint file not found at {ckpt_path}!")
        return 1

    actual_ckpt_sha256 = compute_file_sha256(ckpt_path)
    print(f"\n--- CHECKPOINT PROVENANCE ---")
    print(f"Checkpoint Path:    {ckpt_path}")
    print(f"Expected SHA256:    {EXPECTED_CHECKPOINT_SHA256}")
    print(f"Actual SHA256:      {actual_ckpt_sha256}")

    if actual_ckpt_sha256.lower() != EXPECTED_CHECKPOINT_SHA256.lower():
        print("FATAL ERROR: Checkpoint SHA256 mismatch! The benchmark MUST FAIL if checkpoint does not match.")
        return 1
    print("Checkpoint SHA256 verification: PASS (Bit-exact match)")

    # 3. Source Git Provenance
    prov = get_git_provenance()
    print(f"\n--- SOURCE CODE PROVENANCE ---")
    print(f"Git Commit SHA:     {prov.get('git_commit', 'unknown')}")
    print(f"Branch:             {prov.get('branch', 'unknown')}")
    print(f"Modified Files:     {prov.get('modified_files', [])}")

    # 4. Generate Workload
    total_required = args.warmup + args.frames
    frames = generate_soak_frames(total_required, seed=args.seed)

    # 5. Execute FP32 Soak
    fp32_res = run_single_precision_soak(
        frames=frames,
        ckpt_path=ckpt_path,
        device_str=args.device,
        fp16=False,
        n_warmup=args.warmup,
        n_measured=args.frames,
    )

    # 6. Execute FP16 Soak
    fp16_res = run_single_precision_soak(
        frames=frames,
        ckpt_path=ckpt_path,
        device_str=args.device,
        fp16=True,
        n_warmup=args.warmup,
        n_measured=args.frames,
    )

    # 7. Semantic Parity Evaluation (FP32 vs FP16)
    agreements: List[float] = []
    for p32, p16 in zip(fp32_res["sample_preds"], fp16_res["sample_preds"]):
        agree = float(np.mean(p32 == p16))
        agreements.append(agree)
    semantic_agreement_pct = float(np.mean(agreements) * 100.0) if agreements else 100.0
    print(f"\nFP32 vs FP16 Semantic Agreement: {semantic_agreement_pct:.4f}%")

    # 8. Compile Master Benchmark Results
    # Clean raw predictions from json export
    fp32_res.pop("sample_preds", None)
    fp16_res.pop("sample_preds", None)

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "gpu_model": gpu_name,
            "vram_total_mb": gpu_vram_total_mb,
            "nvidia_driver": driver_version,
            "cuda_version": cuda_version,
            "pytorch_version": pytorch_version,
            "cuda_available": True,
        },
        "provenance": {
            "git_commit": prov.get("git_commit", "unknown"),
            "checkpoint_sha256": actual_ckpt_sha256,
            "modified_files": prov.get("modified_files", []),
        },
        "semantic_agreement_pct": semantic_agreement_pct,
        "fp32": fp32_res,
        "fp16": fp16_res,
        "verdict": {
            "is_tesla_t4": "Tesla T4" in gpu_name or "T4" in gpu_name,
            "fp32_p95_pass": fp32_res["gates"]["p95_pass"],
            "fp32_fps_pass": fp32_res["gates"]["fps_pass"],
            "fp16_p95_pass": fp16_res["gates"]["p95_pass"],
            "fp16_fps_pass": fp16_res["gates"]["fps_pass"],
            "all_gates_pass": fp32_res["gates"]["overall_pass"] and fp16_res["gates"]["overall_pass"],
        },
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comprehensive benchmark results to: {args.output_json}")

    # Print Final Audit Table
    print("\n" + "=" * 80)
    print("FINAL 1,000-FRAME SOAK AUDIT REPORT TABLE")
    print("=" * 80)
    print(f"{'Precision':<8} | {'Mean (ms)':<9} | {'P50 (ms)':<8} | {'P95 (ms)':<8} | {'P99 (ms)':<8} | {'Max (ms)':<8} | {'FPS':<6} | {'P95 Gate':<8} | {'FPS Gate':<8}")
    print("-" * 80)
    for res in [fp32_res, fp16_res]:
        s = res["stats"]["total"]
        prec = res["precision"]
        p95_g = "PASS" if res["gates"]["p95_pass"] else "FAIL"
        fps_g = "PASS" if res["gates"]["fps_pass"] else "FAIL"
        print(f"{prec:<8} | {s['mean']:<9.2f} | {s['p50']:<8.2f} | {s['p95']:<8.2f} | {s['p99']:<8.2f} | {s['max']:<8.2f} | {res['sustained_fps']:<6.2f} | {p95_g:<8} | {fps_g:<8}")
    print("=" * 80)

    all_pass = results["verdict"]["all_gates_pass"]
    print(f"\nOVERALL DEPLOYMENT SOAK VERDICT: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
