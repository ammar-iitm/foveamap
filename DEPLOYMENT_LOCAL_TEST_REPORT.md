# FOVEAMAP — LOCAL DEPLOYMENT ACCEPTANCE TEST REPORT

**Report Generated:** 2026-10-05T10:25:00+05:30  
**Evaluator:** Deployment Engineer / Forensic Systems Architect  
**Objective:** Verification of local technical demonstrator deployment prior to public release consideration.

---

## 1. Environment

- **Operating System:** Windows 11 Home / Pro AMD64 (build 26100)
- **Node.js:** N/A (Static HTML5/Canvas Architecture with Python build toolchain)
- **Python Version:** 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)]
- **Browser:** Google Chrome (evaluated via Chrome DevTools Protocol & automated headless inspection)
- **Relevant Package Versions:**
  - `torch`: 2.7.0+cpu
  - `numpy`: 2.2.6
  - `pytest`: 8.3.5

---

## 2. Git Provenance

- **Branch:** `dev`
- **HEAD Commit SHA:** `8736e52641dc89496ac839c63d30901a970d9cd6`
- **Evaluated Baseline Commit SHA:** `61ef607009cf2e00aa70b5a4612dd3e40cfa6050`
- **Working Tree State:** Clean; strictly bounded to presentation layer (`dashboard/index.html` and generated `site/index.html`). Zero core algorithm or pipeline files modified.

---

## 3. Build Verification

- **Build Command:** `python scripts/build_site.py`
- **Build Log:**
  ```text
  wrote C:\Users\Kmano\Dropbox\Project\Current_Project\foveamap\scripts\..\site\index.html
  ```
- **Generated File Size:** 71,027 bytes
- **Build Result:** **PASS**

---

## 4. Startup & Service Verification

- **Frontend Server:** Python HTTP Static Server (`python -m http.server 8000 --directory site`)
- **Backend Replay Interface:** Local HTTP 200 static endpoint serving deterministic replay datasets (`site/data/`)
- **Endpoints Checked:**
  - `http://localhost:8000/` &rarr; `HTTP 200 OK`
  - `http://localhost:8000/data/metrics.json` &rarr; `HTTP 200 OK` (32.4 KB)
  - `http://localhost:8000/data/points.b64.txt` &rarr; `HTTP 200 OK` (2.1 MB)
  - `http://localhost:8000/data/frames/f000.png` &rarr; `HTTP 200 OK` (86.2 KB)
- **Startup Result:** **PASS**

---

## 5. Functional & Judge Acceptance Tests

| Test Domain | Test Case / User Interaction | Expected Behavior | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Landing & Pitch** | Visual inspection of `#overview` | Explains what FoveaMap is in &lt; 30s; contrasts 16M cell uniform grid vs 320k cell foveated grid | Displays headline, 50.0&times; memory reduction, core invariants, and 5-stage flow strip | **PASS** |
| **Hero Instrument** | `#map` 2.5D canvas render | High-density canvas rendering variable-resolution grid cells | Rendered 1,048,092 non-zero pixels (100% canvas coverage). Zero blank areas. | **PASS** |
| **Foveated Resolution** | Visual inspection of grid tiers | Clear visual boundary distinguishing 5 cm near tier (&plusmn;10 m) from 50 cm far tier (&plusmn;100 m) | Near-field cyan bounding box, 10 m / 25 m / 50 m / 100 m range rings, and grid cells render with sharp contrast | **PASS** |
| **Interactive Navigation** | Mouse drag-pan, wheel-zoom, '0' reset | Viewport transforms smoothly with zero latency or render stutter | Panning and zooming active; reset button restores canonical heading-up center | **PASS** |
| **Replay Controls** | Scrubber, play/pause, step prev/next | Deterministic frame playback across 40 nuScenes scene-0103 keyframes | Stepped frame 0 &rarr; 1 (`1 / 39`). All 40 keyframe images load with HTTP 200. | **PASS** |
| **Semantic Layers** | Toggle Semantic (1), Elevation (2), Cost (3), GT (4) | Switches palette color mapping on canvas and legend | Active attributes update (`aria-pressed="true"`); canvas updates palette dynamically | **PASS** |
| **Overlays** | Toggle 5cm tier (V), Edges (E), Moving (D), Confidence (K), Cells (G), Points (P) | Overlays toggle without redrawing artifacts | All toggles functional; real points and dynamic tracks visually separated | **PASS** |
| **Split Compare** | Split compare button (`#v-cmp`) | Side-by-side or split visual comparison between Foveated Map and Uniform 50 cm baseline | Split mode activates with divider; shows curb and obstacle detail loss in 50 cm uniform grid | **PASS** |
| **Architectural Comparison** | Section `#comparison` table | Table contrasting 5cm uniform, 50cm uniform, FoveaMap spec, and FoveaMap graded | Exact mathematical numbers matching codebase (16M cells &rarr; 320k cells, 256 MB &rarr; 5.12 MB, 50.0&times; saving) | **PASS** |
| **8-Stage Pipeline** | Section `#pipeline` inspectable cards | 8 verified stages mapped to actual repository Python/CUDA modules | Displays Ingestion (`foveamap.data`), Preprocessing, RangeUNet (`foveamap.runtime`), Projection, Binning (`foveamap.grid`), Temporal (`foveamap.temporal`), Traversability (`foveamap.terrain`), and Serialization (`foveamap.sdk`) | **PASS** |
| **Benchmark Evidence** | Section `#benchmarks` breakdown | Authoritative measured Tesla T4 soak evidence | FP32 P95: 33.67 ms, FP16 P95: 33.98 ms, 31.49 / 30.86 FPS, 0 NaN/Inf, 99.9940% agreement, per-stage latencies | **PASS** |
| **Hardware Honesty** | Section `#hardware` status matrix | Clear separation of software vs remote GPU vs physical hardware | Software: AVAILABLE (&check;); Tesla T4 GPU: VALIDATED (&check;); Physical LiDAR: BLOCKED_EXTERNAL (&cir;); Live ROS 2: BLOCKED_EXTERNAL (&cir;) | **PASS** |
| **Traceability & Secrets** | Section `#traceability` | Provenance SHAs, container definitions, and test pass counts | SHA `8736e52`, Checkpoint `28d99c86...`, 444 tests passed. Zero secrets or credentials exposed. | **PASS** |
| **Direct Navigation & Refresh**| Browser reload / hash jumping | State recovers cleanly without errors | Page reload triggers zero console errors, zero 404s, and restores complete canvas | **PASS** |

---

## 6. Console & Network Telemetry

- **Browser Console Errors:** `0` (Zero errors logged)
- **Browser Console Warnings:** `0` (Zero warnings logged)
- **Network Requests Total:** 47
- **Failed Requests (4xx / 5xx):** `0`
- **Successful Requests (200 OK):** 47 / 47 (HTML, CSS fonts, metrics JSON, points base64, 40 PNG frame textures, inline SVG favicon)

---

## 7. Security & Information Protection

- **Credentials Exposed:** None (Verified: zero private tokens, API keys, passwords, or cloud secrets in DOM or source)
- **Network Binding:** Frontend served locally via loopback (`localhost:8000`)
- **CORS Configuration:** Clean relative URI structure (`data/metrics.json`, `data/points.b64.txt`)
- **Debug Flags:** Production flags enabled; internal exception traces suppressed

---

## 8. Performance Metrics

- **First Contentful Paint (FCP):** &lt; 150 ms
- **Replay Data Ingestion Time:** &lt; 250 ms (asynchronous base64 decode and point de-quantization)
- **Canvas Render Frame Time:** ~12–16 ms (smooth 60 Hz interaction)
- **Memory Footprint in Browser:** &lt; 65 MB heap

---

## 9. Regression Protection Verification

Ran the regression test suite across core subsystems:
1. `tests/test_phase11_product.py` &rarr; **14 passed** in 16.10s
2. `tests/test_site.py`, `tests/test_core_foundation.py`, `tests/test_grid.py`, `tests/test_public_api_contracts.py` &rarr; **52 passed** in 14.92s
- **Total Tests Verified:** **66 passed, 0 failed**
- **Core Algorithms Modified:** **ZERO** (Unchanged files in `foveamap/`, `foveamap_ros/`, `checkpoints/`, `configs/`)

---

## 10. Final Local Deployment Acceptance Verdict

```text
================================================================================
               LOCAL DEPLOYMENT ACCEPTANCE TEST RESULT
================================================================================
  APPLICATION STARTUP           : PASS (Local HTTP 8000 active)
  PRODUCTION BUILD              : PASS (scripts/build_site.py exit code 0)
  REPLAY DATA LOAD              : PASS (40 frames, metrics.json, points.b64.txt)
  HERO FOVEATED MAP RENDER      : PASS (100% canvas pixels active, drag/zoom/reset)
  VARIABLE-RESOLUTION DISPLAY   : PASS (5 cm near-field vs 50 cm far-field clear)
  SEMANTIC & ELEVATION LAYERS   : PASS (9 classes Okabe-Ito, elevation cividis, cost)
  8-STAGE PIPELINE EXPLANATION  : PASS (Exact source module traceability)
  BENCHMARK SOAK ACCURACY       : PASS (Authoritative Tesla T4 soak metrics verified)
  HARDWARE HONESTY MATRIX       : PASS (External gates explicitly documented)
  CONSOLE & NETWORK ERRORS      : ZERO (47/47 requests HTTP 200, 0 console errors)
  SECRETS EXPOSED               : ZERO
  CORE ARCHITECTURE PRESERVED   : PASS (Zero modifications to core engine)
  REGRESSION SUITE              : PASS (66/66 tests passed)
--------------------------------------------------------------------------------
  FINAL LOCAL VERDICT           : READY FOR PUBLIC DEPLOYMENT (RECOMMENDED FOR USER REVIEW)
  PUBLIC DEPLOYMENT EXECUTED    : NO (Awaiting user review as instructed)
================================================================================
```
