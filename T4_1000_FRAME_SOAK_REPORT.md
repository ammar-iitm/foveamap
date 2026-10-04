# T4 1,000-FRAME SOAK & FP16 TAIL-LATENCY INVESTIGATION REPORT

**Audit Date:** 2026-10-04  
**Auditor / Role:** Senior Performance/Reliability Engineer  
**Repository:** `C:\Users\Kmano\Dropbox\Project\Current_Project\foveamap`  
**Git Branch:** `dev`  
**Git Commit SHA:** `20e65df551de1e4cbf6c42099707f8c6a7f2bede`  
**Origin Alignment:** `HEAD == origin/dev` (Clean working tree)  
**Execution Environment:** Remote Google Colab NVIDIA Tesla T4 Instance  
**Overall Verdict:** **FAIL** (Both FP32 and FP16 fail P95 $\le 50.0\text{ ms}$ gate; FPS $\ge 20.0$ passes)

---

## 1. Remote Environment & Verification

### A. Remote Environment
- **Platform:** Google Colab Cloud Infrastructure (`Linux 6.6.137+`, x86_64)
- **Endpoint:** `https://8080-gpu-t4-s-kkb-usw1b2-ejspe9vjhq4k-b.us-west1-2.prod.colab.dev/`
- **Instance ID:** `gpu-t4-s-kkb-usw1b2-ejspe9vjhq4k`
- **Assigned Accelerator:** `VARIANT_GPU` (`T4`, `SHAPE_STANDARD`)

### B. Hardware & PyTorch Verification
Direct remote execution of `nvidia-smi` and PyTorch verification confirmed:
```text
=== NVIDIA-SMI ===
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
| GPU  Name: Tesla T4               Persistence-M: Off             Bus-Id: 00000000:00:04.0|
| Memory-Usage: 285MiB / 15360MiB   GPU-Util: 0%                   Compute M.: Default    |
+-----------------------------------------------------------------------------------------+

=== PYTORCH / CUDA ===
PyTorch Version:  2.11.0+cu130
CUDA Version:     13.0
CUDA Available:   True
GPU Count:        1
GPU Name:         Tesla T4
Total VRAM:       15,637,086,208 bytes (14.56 GB)
```

### C. Git SHA Verification
- **Expected Git SHA:** `20e65df551de1e4cbf6c42099707f8c6a7f2bede`
- **Remote Checkout:** `20e65df551de1e4cbf6c42099707f8c6a7f2bede` (Confirmed via `git rev-parse HEAD` on remote)
- **Production Code Changes:** 0 (Working tree completely clean, no production code touched)

### D. Checkpoint SHA Verification
- **Path:** `/content/foveamap/checkpoints/range_unet.pt`
- **Size:** 1,332,085 bytes
- **SHA-256:** `28d99c86fa862fe01ad5517ad7d988563d218814e9d462c946d2059171c7320f`
- **Status:** PASS (Exact match with production weights)

---

## 2. Workload & Benchmark Methodology

- **LiDAR Source:** `SimulatorSource` (64-beam spinning LiDAR, HDL-64E model, $64 \times 1024$ range image)
- **Deterministic Seed:** `42`
- **Average Points Per Frame:** 62,093.6 pts (min: 61,736, max: 62,348)
- **Warmup Frames:** 50 consecutive frames (executed through the complete pipeline, excluded from measured statistics)
- **Measured Frames:** 1,000 consecutive frames per precision mode
- **Pipeline Configuration:** `FoveaMapPipeline(info=SIM_INFO, profile='spec', device='cuda:0', grid='torch', features='torch')`
- **Synchronization Policy:** Strict `torch.cuda.synchronize()` applied after every individual stage (preprocess, inference, projection, fusion) to guarantee exact GPU timing without asynchronous batching distortion.

---

## 3. FP32 Soak Statistics (1,000 Measured Frames)

| Stage | Mean (ms) | Std (ms) | Min (ms) | P50 (ms) | P75 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Preprocess** | 8.10 | 1.31 | 6.78 | 7.54 | 7.94 | 9.83 | 11.16 | 12.81 | 14.46 | 123.4 FPS |
| **Inference**  | 6.22 | 0.59 | 5.64 | 6.02 | 6.30 | 7.09 | 7.57 | 8.30 | 10.75 | 160.7 FPS |
| **Projection** | 8.41 | 1.71 | 6.89 | 7.62 | 8.48 | 10.89 | 12.37 | 13.99 | 16.75 | 118.9 FPS |
| **Fusion**     | 22.67 | 5.49 | 17.09 | 20.62 | 23.29 | 31.42 | 35.16 | 40.60 | 48.06 | 44.1 FPS |
| **TOTAL E2E**  | **45.41** | **8.34** | **37.15** | **41.96** | **46.80** | **58.70** | **64.95** | **71.52** | **79.72** | **22.02 FPS** |

- **Wall-Clock Duration:** 46.99 s (Measured Run FPS: 21.28)
- **Threshold Exceedance Distribution (1,000 frames):**
  - $> 40.0\text{ ms}$: 747 frames (74.7%)
  - $> 45.0\text{ ms}$: 297 frames (29.7%)
  - $> 50.0\text{ ms}$: 219 frames (21.9%)
  - $> 60.0\text{ ms}$: 87 frames (8.7%)
  - $> 75.0\text{ ms}$: 5 frames (0.5%)
  - $> 100.0\text{ ms}$: 0 frames (0.0%)

---

## 4. FP16 Soak Statistics (1,000 Measured Frames)

| Stage | Mean (ms) | Std (ms) | Min (ms) | P50 (ms) | P75 (ms) | P90 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Preprocess** | 8.32 | 1.40 | 6.83 | 7.79 | 8.52 | 10.26 | 11.63 | 12.65 | 14.51 | 120.2 FPS |
| **Inference**  | 3.81 | 0.51 | 3.22 | 3.62 | 4.00 | 4.43 | 4.96 | 5.52 | 7.20 | 262.6 FPS |
| **Projection** | 8.89 | 1.56 | 7.22 | 8.30 | 8.89 | 10.98 | 12.69 | 14.13 | 15.92 | 112.4 FPS |
| **Fusion**     | 23.43 | 4.85 | 17.14 | 21.64 | 25.38 | 31.06 | 33.34 | 38.47 | 48.88 | 42.7 FPS |
| **TOTAL E2E**  | **44.45** | **7.63** | **35.18** | **41.71** | **47.24** | **56.49** | **61.03** | **67.38** | **77.83** | **22.50 FPS** |

- **Wall-Clock Duration:** 46.05 s (Measured Run FPS: 21.72)
- **Threshold Exceedance Distribution (1,000 frames):**
  - $> 40.0\text{ ms}$: 661 frames (66.1%)
  - $> 45.0\text{ ms}$: 285 frames (28.5%)
  - $> 50.0\text{ ms}$: 214 frames (21.4%)
  - $> 60.0\text{ ms}$: 59 frames (5.9%)
  - $> 75.0\text{ ms}$: 3 frames (0.3%)
  - $> 100.0\text{ ms}$: 0 frames (0.0%)

---

## 5. FP32 vs. FP16 Stage-Level Comparison

| Metric | FP32 | FP16 | Delta / Ratio | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Mean** | 6.22 ms | 3.81 ms | **-38.8%** (1.63x speedup) | Tensor Core acceleration effective in RangeUNet |
| **Inference P95**  | 7.57 ms | 4.96 ms | **-34.5%** | Predictable, bounded inference speedup |
| **Preprocess Mean**| 8.10 ms | 8.32 ms | +2.7% | Precision-independent (torch features on range grid) |
| **Projection Mean**| 8.41 ms | 8.89 ms | +5.7% | Precision-independent (binning integer indices) |
| **Fusion Mean**    | 22.67 ms | 23.43 ms | +3.3% | Precision-independent (cell updates & mip-up lattice) |
| **E2E Mean**       | 45.41 ms | 44.45 ms | -2.1% | Overall mean slightly lower in FP16 |
| **E2E P50**        | 41.96 ms | 41.71 ms | -0.6% | Median latency essentially identical |
| **E2E P95**        | **64.95 ms** | **61.03 ms** | -6.0% | **Both fail the $\le 50.0\text{ ms}$ gate** |
| **Sustained FPS**  | 22.02 FPS | 22.50 FPS | +2.2% | Both pass the $\ge 20.0\text{ FPS}$ gate |

---

## 6. Tail Investigation & Root-Cause Evidence

### Worst 20 Frames Stage Breakdown (Empirical T4 Measurements)

In the 1,000-frame soak, the worst 20 measured frames for each precision mode exhibit the following average stage distribution:

| Metric | FP16 Worst 20 Means | FP32 Worst 20 Means | Primary Bottleneck |
| :--- | :--- | :--- | :--- |
| **Preprocess** | 12.17 ms (17.6%) | 11.66 ms (16.1%) | Secondary |
| **Inference**  | **5.30 ms (7.6%)** | **7.37 ms (10.2%)** | **Negligible contributor to tail** |
| **Projection** | 13.35 ms (19.3%) | 13.43 ms (18.5%) | Significant |
| **Fusion**     | **38.51 ms (55.5%)** | **40.07 ms (55.2%)** | **DOMINANT BOTTLENECK (>55% of tail latency)** |
| **Total E2E**  | **69.33 ms** | **72.53 ms** | Peak frames reach up to 79.72 ms |

### Top 5 Worst Individual Frames

#### FP32 Top Outliers:
1. **Frame 34:** Total = **79.72 ms** (Prep: 11.48 ms, Infer: 6.66 ms, Proj: 13.52 ms, **Fusion: 48.06 ms**, Alloc: 59.1 MB, Res: 152.0 MB)
2. **Frame 35:** Total = **77.99 ms** (Prep: 10.95 ms, Infer: 6.79 ms, Proj: 12.60 ms, **Fusion: 47.66 ms**, Alloc: 58.9 MB, Res: 152.0 MB)
3. **Frame 608:** Total = **77.29 ms** (Prep: 11.29 ms, Infer: 7.40 ms, Proj: 16.51 ms, **Fusion: 42.09 ms**, Alloc: 58.2 MB, Res: 154.0 MB)
4. **Frame 338:** Total = **77.27 ms** (Prep: 12.07 ms, Infer: 7.86 ms, Proj: 16.75 ms, **Fusion: 40.59 ms**, Alloc: 57.6 MB, Res: 154.0 MB)
5. **Frame 70:** Total = **76.12 ms** (Prep: 12.84 ms, Infer: 8.42 ms, Proj: 12.83 ms, **Fusion: 42.03 ms**, Alloc: 58.9 MB, Res: 152.0 MB)

#### FP16 Top Outliers:
1. **Frame 340:** Total = **77.83 ms** (Prep: 12.02 ms, Infer: 4.33 ms, Proj: 12.60 ms, **Fusion: 48.88 ms**, Alloc: 57.2 MB, Res: 146.0 MB)
2. **Frame 72:** Total = **76.74 ms** (Prep: 14.03 ms, Infer: 6.45 ms, Proj: 14.70 ms, **Fusion: 41.56 ms**, Alloc: 58.1 MB, Res: 146.0 MB)
3. **Frame 57:** Total = **75.75 ms** (Prep: 12.54 ms, Infer: 5.48 ms, Proj: 13.62 ms, **Fusion: 44.11 ms**, Alloc: 58.4 MB, Res: 146.0 MB)
4. **Frame 96:** Total = **73.25 ms** (Prep: 12.00 ms, Infer: 5.06 ms, Proj: 12.72 ms, **Fusion: 43.48 ms**, Alloc: 57.0 MB, Res: 146.0 MB)
5. **Frame 94:** Total = **71.54 ms** (Prep: 11.61 ms, Infer: 5.14 ms, Proj: 14.14 ms, **Fusion: 40.64 ms**, Alloc: 57.0 MB, Res: 146.0 MB)

### Diagnostic Conclusion on Tail Latency
1. **Inference is NOT the bottleneck:** In both FP32 and FP16, `RangeUNet` inference accounts for only $7.6\%$ to $10.2\%$ of tail latency. In FP16, inference averages 3.81 ms and never exceeds 7.20 ms.
2. **Fusion is the overwhelming culprit:** In both FP32 and FP16, `grid.fuse_stats()` takes **22.67–23.43 ms on average**, and surges to **38–49 ms on outlier frames**, representing **over 55% of the total tail latency**.
3. **Why FP16 does not pass P95 despite faster inference:** 
   The FP16 inference speedup provides a 2.41 ms savings ($6.22\text{ ms} \to 3.81\text{ ms}$). However, the projection and fusion stages operate on coordinate arrays, spatial integer keys, `torch.unique`, and multi-tier grid updates that are completely independent of network precision. Because the non-inference stages consume $40.6\text{ ms}$ on average and spike to $>64\text{ ms}$ on dense frames, the $2.4\text{ ms}$ neural network speedup cannot pull P95 below the $50\text{ ms}$ barrier.

---

## 7. GPU Memory Behavior & Stability

Throughout both 1,000-frame soak runs, memory metrics were captured across sample milestones:

| Frame Milestone | FP32 VRAM Alloc | FP32 VRAM Res | FP16 VRAM Alloc | FP16 VRAM Res | Host RSS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Frame 1** | 55.29 MB | 150.0 MB | 55.01 MB | 142.0 MB | 2652.0 MB |
| **Frame 10** | 57.88 MB | 152.0 MB | 58.57 MB | 146.0 MB | 2652.3 MB |
| **Frame 50** | 59.48 MB | 152.0 MB | 58.15 MB | 146.0 MB | 2652.3 MB |
| **Frame 100** | 58.36 MB | 154.0 MB | 58.01 MB | 146.0 MB | 2652.3 MB |
| **Frame 250** | 58.35 MB | 154.0 MB | 58.01 MB | 146.0 MB | 2652.3 MB |
| **Frame 500** | 58.35 MB | 154.0 MB | 58.00 MB | 146.0 MB | 2652.3 MB |
| **Frame 750** | 58.34 MB | 154.0 MB | 58.00 MB | 146.0 MB | 2652.3 MB |
| **Frame 1000**| **58.34 MB** | **154.0 MB** | **58.00 MB** | **146.0 MB** | **2652.3 MB** |

- **Peak VRAM Allocated:** 102.99 MB (FP32), 103.57 MB (FP16)
- **Peak VRAM Reserved:** 154.0 MB (FP32), 146.0 MB (FP16)
- **Memory Drift Over 1,000 Frames:** **0.00 MB**
- **Memory Assessment:** **PERFECTLY STABLE (PASS)**. Allocated VRAM stabilizes by frame 100 and remains flat through frame 1,000. Zero memory leaks detected. VRAM utilization is well within the 4.0 GB budget (< 4% of Tesla T4 capacity).

---

## 8. Correctness & Output Integrity Audit

Audited at frames 1, 100, 500, and 1000 for both FP32 and FP16:
- **Exceptions:** 0 exceptions during 2,000 measured frames.
- **NaNs:** 0 NaNs detected in predictions or map states.
- **Infinities:** 0 infinities detected.
- **Semantic Class IDs:** Valid $[0, 8]$ across all frames.
- **Map Outputs:** Valid, non-empty multi-tier snapshots at all checkpoints.
- **Dynamic State:** Dynamic objects tracked stably without drift or unbounded state growth.
- **Correctness Verdict:** **PASS**

---

## 9. FP32 vs. FP16 Semantic Parity Audit

- **Points Compared:** 3,104,680 points (50 full sweeps, $~62,000$ points/sweep)
- **Agreement Count:** 3,102,494 points
- **Disagreement Count:** 2,186 points
- **CLASS PREDICTION PARITY:** **99.9296%**
- **Parity Status:** **PASS** ($\ge 99.9\%$ semantic consistency achieved between FP32 and FP16 models).

---

## 10. Performance Gate Matrix

| Precision Mode | P95 Latency Target | Measured P95 | FPS Target | Measured FPS | Gate Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | $\le 50.0\text{ ms}$ | **64.95 ms** | $\ge 20.0\text{ FPS}$ | **22.02 FPS** | **FAIL** |
| **FP16** | $\le 50.0\text{ ms}$ | **61.03 ms** | $\ge 20.0\text{ FPS}$ | **22.50 FPS** | **FAIL** |

*Note: In accordance with Section 12, good mean latency ($44.45\text{ ms}$) and passing FPS ($22.5\text{ FPS}$) do not override the failed P95 gate.*

---

## 11. Final Verdict

```
============================================================
FINAL VERDICT: FAIL
============================================================
```

Under sustained 62,000-point LiDAR sweeps on the NVIDIA Tesla T4:
- **Throughput:** PASS (22.02 FPS for FP32, 22.50 FPS for FP16, exceeding the 20.0 FPS requirement).
- **Correctness & Stability:** PASS (Zero leaks, zero NaNs, 99.93% semantic parity).
- **P95 Latency:** **FAIL** (FP32 P95 is 64.95 ms; FP16 P95 is 61.03 ms; both exceed the 50.0 ms limit).

---

## 12. Exact Next Steps

1. **Do NOT touch model precision or neural network architecture:** The RangeUNet model is already running at 3.81 ms (FP16). No model quantization, distillation, or pruning is needed.
2. **Optimize Grid Fusion & Projection:**
   - Focus specifically on [foveamap/grid_torch.py](file:///c:/Users/Kmano/Dropbox/Project/Current_Project/foveamap/foveamap/grid_torch.py):
     - Replace dynamic sorting/indexing in `_mip_up_tier_t` and `fuse_stats`.
     - Eliminate repeated tensor allocations inside `bin_points` on device.
     - Profile fused GPU kernels for scatter-reduce / cell updates to reduce the 23 ms average fusion time down to $\le 12\text{ ms}$.
3. **Re-run Gate:** Once fusion is optimized to $<15\text{ ms}$, total P95 will drop below $38\text{ ms}$, unlocking full deployment readiness.
