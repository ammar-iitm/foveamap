# FoveaMap — Architecture Decisions & PRD Reconciliation (Phase 9.5)

This document records intentional deviations between the PRD / Architecture
Vision PDFs (frozen, Sep 2026) and the implementation. The PDFs cannot be
edited; THIS file is the authoritative reconciliation. Status labels:
IMPLEMENTED (in tree), VERIFIED (tested/measured here), PLANNED, DEFERRED.

## D1. Cell class representation — OPTION A (compressed, 16 bytes kept)

PRD FR-9 asks for a per-cell class histogram (plus mean height). The
implementation keeps the 16-byte persistent cell (count, z_min, z_max,
ground, rough, cls, conf nibbles, flags nibbles, clear, cost, age) with
**dominant class + secondary class evidence** as the v1 compressed
representation (STATUS: IMPLEMENTED, VERIFIED by parity/secondary tests).
Rationale: a full 9-class FP16 histogram would add 18 B/cell and destroy
the <= 8 MB / 50x PRD memory objective (NFR-3/NFR-4, both measured green).
No downstream consumer in Phases 1–9 requires the full histogram. A bounded
auxiliary histogram may be added later only with measured memory accounting.
Mean height is subsumed by `ground` (EMA surface estimate); last-seen time
by `age`; the dynamic flag by the per-frame dynamic layer + Phase 6
lifecycle. Memory accounting distinguishes persistent/auxiliary/temporal
bytes and never reports theory as allocation.

Measured memory table (from `memory_report()`, spec profile required):

```text
Option A (current, dominant + secondary evidence):
  spec:   320,000 cells x 16 B = 5,120,000 persistent (5.76 MB with aux) <= 8 MB PASS
  graded: 570,000 cells x 16 B = 9,120,000 persistent (graded is not the
          <= 8 MB profile; spec is the acceptance profile)
Option B (full 9-class FP16 histogram, +18 B/cell, estimated):
  spec:   320,000 x 34 B = 10,880,000 persistent -> EXCEEDS 8 MB target
Option C (bounded auxiliary histogram): possible later only with measured
  accounting; no current consumer requires it.
```

Downstream loss under Option A: per-class vote margins beyond top-2 are not
retained per cell. Sufficient for v1 because costing, queries, and
snapshots only ever consume dominant/secondary evidence (verified by
contract tests); raw per-point labels remain available upstream of fusion.

## D8. grid_map export — ROS-compatible payload now, grid_map_msgs later

PRD FR-13 allows "ROS grid_map or NumPy" export. v1 provides NumPy tier
export (`export_numpy`, detached copies) plus a ROS-compatible tier/cell
payload (`/foveamap/grid`, documented geometry + semantics) and matching
`.msg` definitions. A `grid_map_msgs/GridMap` converter is intentionally
deferred: it binds deployment message generation (rosidl) without adding
mapping value, and an unvalidated converter would be worse than the
documented boundary. STATUS: NumPy + ROS-compatible payload IMPLEMENTED +
TESTED; `grid_map_msgs` DEFERRED with rationale.

## D2. Perception backbone — RangeUNet actual, sparse-conv future

PRD FR-6 and Vision §4 name sparse 3D conv the default. Actual default:
RangeUNet (8-channel range image, 9 classes + motion) with explicit
ClassicalFallback (`backend_type="classical"`), selected by config
(STATUS: IMPLEMENTED). The backend abstraction (`PerceptionBackend`
factory) makes backbones swappable without touching mapping; swapping in a
sparse-conv backbone later needs no grid changes. Sparse 3D U-Net:
PLANNED/DEFERRED. No claim of equivalence is made.

## D3. Dashboard scope — replay viewer with analysis UI, not live 3D (v1)

Verified in `dashboard/index.html`: 2D top-down canvas with play/pause/step/
scrub, semantic/elevation/traversability/ground-truth layer toggles, overlays
(fovea tier, curb/pothole edges, moving agents, confidence, grid, raw points),
foveated-vs-uniform split compare, per-frame inspector, and latency/memory/
accuracy charts driven by recorded `dashboard/data` (FR-15 metrics, FR-16
toggles/compare, FR-17 replay controls: IMPLEMENTED as replay UI).
Deferred: live streaming mode and 3D view (FR-14 partially met).
Rendering stays outside measured core latency by methodology (`export after`
default). No dashboard redesign is in scope for closure.

## D4. Deployment export — ONNX/TensorRT deferred (PRD FR-20, P2)

Not implemented. No export claims are made anywhere in code or docs.

## D5. Execution profiles — explicit CPU/GPU contracts

`RuntimeConfig.cpu_profile()` (all-NumPy/CPU) and `.gpu_profile()`
(Torch engines, `device="auto"` → CUDA where present, CPU fallback).
Defaults remain `device="auto"`, engines NumPy (development-safe);
production GPU selects the gpu profile explicitly. Remote GPU *performance*
has been authoritatively verified on NVIDIA Tesla T4 (Kaggle remote execution,
Kernel `zesalamander/foveamap-1000-frame-soak-t4`), achieving P95 = 34.51 ms
(FP32) / 34.74 ms (FP16) at 30.70 FPS / 30.07 FPS, comfortably surpassing the
P95 $\le 50.0\text{ ms}$ and $\ge 20.0\text{ FPS}$ deployment gates.

## D6. ROS 2 posture — structural adapter, physical execution deferred

`foveamap_ros` is a tested adapter (conversion/TF/lifecycle/messages/
backpressure/diagnostics). rclpy/colcon execution is UNVERIFIED here (no
ROS 2 on this host); the package is structurally colcon-valid
(`package.xml` + `CMakeLists.txt` + wired `.msg` generation) with
ROS-dependent tests guarded and skipped with reason.

## D7. Accuracy posture — historical numbers are provenance, not proof

`results/kaggle_1000_soak_results.json` contains authoritative 1,000-frame
soak measurements on NVIDIA Tesla T4 (Kernel `zesalamander/foveamap-1000-frame-soak-t4`).
Earlier baseline figures in `results/` are preserved for provenance.
Held-out dataset semantic validation on nuScenes/SemanticKITTI full splits is documented
in Phase 10/11 artifacts; synthetic & 5-frame regression sets are verified locally with
100% test pass.
