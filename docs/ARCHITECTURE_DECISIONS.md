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
production GPU selects the gpu profile explicitly. GPU *performance*
remains UNVERIFIED until the 12 GB Colab/Kaggle session.

## D6. ROS 2 posture — structural adapter, physical execution deferred

`foveamap_ros` is a tested adapter (conversion/TF/lifecycle/messages/
backpressure/diagnostics). rclpy/colcon execution is UNVERIFIED here (no
ROS 2 on this host); the package is structurally colcon-valid
(`package.xml` + `CMakeLists.txt` + wired `.msg` generation) with
ROS-dependent tests guarded and skipped with reason.

## D7. Accuracy posture — historical numbers are provenance, not proof

`results/` and README figures are Colab/T4 measurements from earlier trees
(labeled with hardware/session). They are NOT re-claimed as final-tree
results. NFR-6/7/8 acceptance requires the held-out rerun (Phase 11),
explicitly UNVERIFIED here.
