# FOVEAMAP — TECHNICAL DEPLOYMENT & DEMONSTRATOR IMPLEMENTATION PLAN

**Date:** 2026-10-05  
**Author:** Lead Deployment & Systems Architect  
**Evaluated Git Baseline:** `8736e52641dc89496ac839c63d30901a970d9cd6` (`dev`)  
**Deployment Target:** Local Technical Demonstrator (Vercel-compatible static export via `scripts/build_site.py`)  
**Status:** APPROVED FOR LOCAL IMPLEMENTATION (Public Deployment Frozen)

---

## 1. Current Deployment Architecture

- **Public Hosting Target:** Vercel (URL: `https://foveamap-teal.vercel.app/`).
- **Build System:** Governed by `vercel.json`:
  ```json
  {
    "framework": null,
    "installCommand": "",
    "buildCommand": "python3 scripts/build_site.py",
    "outputDirectory": "site"
  }
  ```
- **Static Artifact Bundle:** `scripts/build_site.py` wraps `dashboard/index.html` into a standalone HTML file and writes to `site/index.html`.
- **Precomputed Replay Data:** Stored in `site/data/`:
  - `metrics.json` (17.9 KB): nuScenes scene-0103 keyframes (40 frames, 2 Hz, 32-beam).
  - `points.b64.txt` (3.2 MB): Base64-encoded decimated 3D points for raw point overlay.
  - `frames/*.png` (40 files, ~100-250 KB each): Encoded multi-tier 2.5D grid states (semantic class, elevation, traversability cost, and dynamic/hazard flags).
- **Alternative Local Runtime Server:** `foveamap serve --dashboard` (`foveamap.sdk.http.FoveaMapHttpServer`) capable of serving static dashboard files and dynamic REST endpoints (`/health`, `/status`, `/metrics`, `/map/query`, `/map/snapshot`).

---

## 2. Existing Frontend Architecture

- **Current Implementation:** `dashboard/index.html` is a monolithic client-side HTML5/CSS3/JavaScript instrument panel (~665 lines, 43.7 KB).
- **Presentation Model:** A single-screen dashboard with:
  - Top header: branding, dataset title, split compare toggle, integrity pill.
  - Main panel (left 2/3): Canvas-based top-down 2.5D map (`#map`), zoom/pan controls, colormap legend, and interactive cell inspector.
  - Rail panel (right 1/3): 4 KPI cards (Pipeline speed, p95 latency, Map memory, 50x saving), and SVG charts (memory comparison, latency breakdown, accuracy by distance).
  - Bottom bar: Base layer chips (Semantic, Elevation, Traversability, Ground truth), overlay chips (5 cm tier, Edges, Moving, Confidence, Cells, Points), and a frame scrubber.
- **Identified Deficiency for Technical Judges:**
  - The current UI plunges the user directly into an unexplained instrument dashboard without answering **"What is FoveaMap and why does it matter?"**.
  - A technical judge looking at the screen cannot discern in 30–60 seconds what the fundamental engineering problem is, why uniform grids fail, how the 8-stage pipeline works, or what evidence validates the system without verbal explanation.

---

## 3. Existing Backend Architecture

- **Runtime Engine (`foveamap/runtime/runtime.py`):** `FoveaMapRuntime` coordinating ingest, preprocessing, RangeUNet inference, foveated projection, temporal fusion, and `MapSnapshot` generation.
- **Pipeline Harness (`foveamap/pipeline.py`):** `FoveaMapPipeline` providing synchronous execution and RGB frame encoding (`encode_tier`, `encode_gt`).
- **HTTP Service Boundary (`foveamap/sdk/http.py`):** Python standard library `ThreadingHTTPServer` exposing REST API with hardened bearer authentication, restricted CORS, and directory traversal protection.
- **CLI Commands (`foveamap/cli.py`):** `info`, `demo`, `compare`, `run`, `serve`, `bench`.

---

## 4. Existing Reusable Components

1. **Client-Side Canvas Rendering Core (`dashboard/index.html`):**
   - High-performance pixel unpacking (`rawFrame(t)`) from compressed PNG tiles.
   - Validated Okabe-Ito colorblind palette (`CLASS_HEX`, `CLASS_RGB`) and Cividis elevation colormap (`CIVIDIS`).
   - Interactive zoom, pan, and coordinate transformation logic.
   - Real-time cell inspector displaying distance, elevation, semantic class, and traversability cost.
2. **Replay Datasets:**
   - Real nuScenes keyframe sequence (`site/data/`, 40 frames).
   - High-density 64-beam simulated drive (`dashboard/data/`, 60 frames).
3. **Build Pipeline:**
   - `scripts/build_site.py` for automated wrapping and static site generation.

---

## 5. Existing Data That Can Be Visualized

- **Concentric Grid Tiers:** 5 cm near-field ($\pm 10\text{ m}$) nested inside 50 cm far-field ($\pm 100\text{ m}$).
- **Semantic Classification:** 9 classes (Road, Sidewalk, Parking, Terrain, Vegetation, Building/Wall, Pole/Sign, Vehicle, Person).
- **Elevation Profiles:** Ground elevation, obstacle height, and multi-level overhangs.
- **Traversability & Hazards:** Cost (0–254), curb steps (flag 64), potholes/depressions (flag 128).
- **Dynamic Objects:** Spatial tracking bounding boxes and velocity vectors.
- **Decimated 3D Point Clouds:** Raw LiDAR returns colored by predicted semantic category.
- **Authoritative T4 Benchmarks:** Tail latencies, stage latencies, memory footprint, and semantic parity.

---

## 6. Existing Replay / Simulation Capabilities

- Deterministic frame replay operating entirely in client-side JavaScript from pre-rendered arrays, requiring zero cloud compute or local GPU during evaluation.
- Local Python simulation engine (`foveamap.sim.simulate_sequence`) capable of generating dynamic obstacle scenarios on CPU.

---

## 7. What Can Be Demonstrated Without Physical LiDAR

- Adaptive variable-resolution mapping with visual near-field vs. far-field density contrast.
- 50x cell count and memory reduction ($16\text{M} \to 320\text{k}$ cells, $256\text{ MB} \to 5.12\text{ MB}$).
- Real-time semantic segmentation, elevation mapping, and traversability assessment.
- Dynamic entity tracking and velocity estimation.
- Split comparison against uniform 50 cm baseline.
- Complete 8-stage pipeline dataflow.
- Rigorous measured T4 GPU benchmark metrics.

---

## 8. What Can Be Demonstrated Without a GPU

- The entire client-side web application runs at 60 FPS on any laptop or desktop browser.
- CPU Python pipeline and regression test suites execute cleanly on any x86_64 / ARM machine.
- Transparent reporting clearly articulates that browser replay visualizes validated offline/cloud inference runs while remote Tesla T4 hardware achieves 31.5 FPS / 33.7 ms P95.

---

## 9. What Must Remain Explicitly Marked as Unavailable Hardware Validation

1. **Physical LiDAR UDP Sensor Stream:** `BLOCKED_EXTERNAL` (Requires physical Ouster/Hesai/Velodyne sensor).
2. **Live In-Vehicle ROS 2 Chassis Bus:** `BLOCKED_EXTERNAL` (Requires vehicle compute platform connected to physical CAN/DDS).

---

## 10. Proposed Deployment Architecture

A cohesive, high-density, **Judge-First Technical Demonstrator** single-page application:

```
+--------------------------------------------------------------------------------+
| FOVEAMAP TECHNICAL DEMONSTRATOR                                                |
+--------------------------------------------------------------------------------+
| 1. EXECUTIVE SUMMARY: "What is FoveaMap and why does it matter?" (30s brief)    |
|    - The Problem: Memory & Compute Explosion of Uniform Grids                  |
|    - The Solution: Concentric Variable-Resolution 2.5D Mapping                 |
|    - High-level Pipeline: LiDAR -> Perception -> Grid -> Fusion -> 2.5D Map    |
+--------------------------------------------------------------------------------+
| 2. INTERACTIVE FOVEATED MAP HERO INSTRUMENT (Main Interactive Experience)      |
|    - Top-Down 2.5D Map Canvas with Zoom & Pan                                  |
|    - Near-Field (5 cm / 10 m) vs Far-Field (50 cm / 100 m) Visualized          |
|    - Real-Time Controls: Semantic, Elevation, Traversability, Ground Truth     |
|    - Overlays: 5 cm Tier, Hazard Edges, Moving Agents, Confidence, Grid, Points|
|    - Interactive Hover Inspector (Class, Height, Cost, Coordinates)            |
|    - 40-Frame Replay Scrubber with Play / Pause / Step Controls                |
+--------------------------------------------------------------------------------+
| 3. ARCHITECTURAL COMPARISON: UNIFORM 5 CM vs FOVEAMAP                          |
|    - Quantitative Matrix: Cell count, Memory, Coverage, Compute                |
|    - Visual Split Compare: Resolution fidelity where it matters                |
+--------------------------------------------------------------------------------+
| 4. 8-STAGE TECHNICAL PIPELINE BREAKDOWN (Interactive & Codebase-Verified)      |
|    - 01. Ingestion -> 02. Preprocessing -> 03. RangeUNet Perception ->         |
|    - 04. 3D-2.5D Projection -> 05. Foveated Binning -> 06. Dynamic Fusion ->   |
|    - 07. Traversability Analysis -> 08. Output / Serialization                 |
+--------------------------------------------------------------------------------+
| 5. AUTHORITATIVE BENCHMARK EVIDENCE (Tesla T4 1,000-Frame Soak Breakdown)      |
|    - FP32 & FP16 P50/P90/P95/P99, Sustained FPS, VRAM Stability, 0 NaN/Inf     |
|    - GPU Soak Benchmark vs Current Browser Demo Distinction                    |
+--------------------------------------------------------------------------------+
| 6. HARDWARE STATUS & DEPLOYMENT HONESTY MATRIX                                 |
|    - Software Demo: AVAILABLE | Tesla T4: VALIDATED                            |
|    - Physical LiDAR: BLOCKED_EXTERNAL | Vehicle ROS 2: BLOCKED_EXTERNAL         |
+--------------------------------------------------------------------------------+
| 7. SYSTEM ARCHITECTURE & ENGINEERING PROVENANCE                                |
|    - Dual Container Specs (Dockerfile, Dockerfile.gpu)                          |
|    - Evaluated Git SHA (8736e52), Checkpoint SHA-256 (28d99c86...), Test Pass  |
+--------------------------------------------------------------------------------+
```

---

## 11. Files That Will Be Added

- `DEPLOYMENT_IMPLEMENTATION_PLAN.md` (this file).
- `DEPLOYMENT_LOCAL_TEST_REPORT.md` (comprehensive local verification report).

---

## 12. Files That Will Be Modified

- `dashboard/index.html`: Upgraded to integrate the Judge-First Technical Demonstrator layout, sections, navigation, and interactive technical components while preserving 100% of the canvas map and replay functionality.
- `scripts/build_site.py`: Ensured to wrap and output the complete demonstrator into `site/index.html`.
- `site/index.html`: Static production build artifact.
- `Final_Audit.md`: Continual logging of deployment engineering steps.

---

## 13. Files That MUST NOT Be Modified (Absolute Architecture Freeze)

- `foveamap/core/` (all modules)
- `foveamap/grid.py`, `foveamap/grid_torch.py`
- `foveamap/runtime/` (all modules: `runtime.py`, `perception.py`, `device.py`)
- `foveamap/temporal.py`, `foveamap/terrain.py`, `foveamap/model.py`, `foveamap/pipeline.py`
- `foveamap/data/` (all modules)
- `foveamap_ros/` (all modules)
- `checkpoints/range_unet.pt`
- `provenance.json`
- `tests/` (all 34 test modules)

---

## 14. Local Test Strategy

1. **Build Test:** Execute `python scripts/build_site.py` and verify `site/index.html` generation.
2. **Local Server Execution:** Launch Python HTTP server on port 8000 serving `site/`.
3. **Functional Inspection:**
   - Verify page loads with HTTP 200.
   - Verify all 7 technical sections are present and readable.
   - Verify canvas map initializes and renders frame 0.
   - Verify replay playback advances frames smoothly.
   - Verify all layer toggles (Semantic, Elevation, Cost, GT) alter canvas rendering correctly.
   - Verify overlay toggles (5 cm tier, Edges, Moving, Confidence, Grid, Points) function.
   - Verify hover inspector displays exact mathematical values.
   - Verify pipeline stages expand with verified technical specifications.
   - Verify benchmark tables reflect exact T4 soak results.
   - Verify zero console errors and zero 404 network requests.
4. **Regression Protection:** Run targeted and fast regression tests (`pytest -q -m "not (cuda or ros2 or slow)"`) to ensure zero impact on core algorithms.

---

## 15. Risks & Mitigations

- **Risk:** Bloated assets slowing initial page load.  
  *Mitigation:* Keep the entire application vanilla (HTML/CSS/JS), zero external UI libraries, keeping total bundle under 80 KB uncompressed (excluding replay frames).
- **Risk:** Canvas resizing or responsiveness breaking on different viewport dimensions.  
  *Mitigation:* Use CSS Grid with aspect-ratio preservation and dynamic canvas resize handlers (`ResizeObserver`).
- **Risk:** Misinterpretation of browser replay as live GPU execution.  
  *Mitigation:* Explicit "DEMO MODE (Deterministic Replay)" status pill in header and clear separation between browser replay and remote Tesla T4 hardware benchmarks.
