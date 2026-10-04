# FoveaMap Development Guide

For operators, see `README.md` and `docs/QUICKSTART.md`. This document is for
engineers who modify, test, and extend the codebase.

## 1. Prerequisites

- Python 3.11 recommended (declared support: `>=3.9`; CI runs 3.11; local dev
  verified on 3.13). Windows, Linux, and macOS all work for CPU development.
- Git, `pip`, and (optionally) Docker Desktop.
- No GPU, ROS 2 installation, or dataset download is needed for the default
  CPU loop. CUDA and ROS 2 are separate, explicitly-gated workflows below.

## 2. Install (deterministic CPU path)

```bash
git clone https://github.com/ammar-iitm/foveamap.git
cd foveamap
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

What this installs (see `pyproject.toml` / `requirements.txt`):

| Dependency | Role | Notes |
|---|---|---|
| `numpy>=1.24` | NumPy grid engine, geometry | direct |
| `scipy>=1.10` | numerics | direct |
| `pillow>=9.0` | dashboard/benchmark PNG export | direct |
| `torch>=2.0` | perception + Torch grid engine | CPU wheel by default; see §4 for CUDA |
| `pytest>=7.0` (+ `ruff>=0.4` via `[dev]`) | tests + lint | dev-only |

There is no lockfile: direct dependencies carry lower bounds and the Docker
image (`python:3.11-slim` + CPU torch) is the reproducible reference. Do not
add transitive pins without a reproducibility reason.

Verify the install:

```bash
python -m foveamap.cli info
python -m foveamap.cli demo --frames 3
```

## 3. Command surface

Two entry points, each with one job:

- `python scripts/dev.py {test,test-fast,test-unit,demo,info,bench,lint,clean}`
  — developer loop (see `scripts/dev.py`).
- `foveamap {info,demo,compare,run,serve,bench}` (or `python -m foveamap.cli …`)
  — product CLI (see `foveamap/cli.py`).

| Goal | Command |
|---|---|
| Full suite | `python scripts/dev.py test` (= `pytest -q`) |
| Fast CPU loop | `python scripts/dev.py test-fast` (= `pytest -q -m "not cuda and not ros2 and not slow"`) |
| Hardening/contracts subset | `python scripts/dev.py test-unit` |
| Focused file | `pytest -q tests/test_grid.py` |
| Focused test | `pytest -q tests/test_grid.py -k nesting` |
| Demo | `python scripts/dev.py demo` |
| Benchmark (short) | `python scripts/dev.py bench` |
| Lint | `python scripts/dev.py lint` |
| Clean caches | `python scripts/dev.py clean` |

Do not introduce `make`/tox wrappers: this Windows-first repo standardizes on
`scripts/dev.py` (cross-platform) plus the product CLI.

## 4. CUDA development path

CPU development is the default. CUDA is opt-in:

1. Install an NVIDIA driver + a CUDA-enabled torch build (see
   https://pytorch.org for the matching `cuXX` wheel).
2. Confirm: `python -c "import torch; print(torch.cuda.is_available())"`.
3. Run hardware tests explicitly: `pytest -q -m cuda`.
4. Benchmark on GPU: `python -m foveamap.cli bench --engine torch --device cuda --frames 100`.

Rules: CUDA-gated tests (`-m cuda`: `test_cuda_perception_execution`,
`test_cuda_cpu_parity_when_available`, `test_real_cuda_canonicalization`) skip
with a reason on CPU hosts. Never fake CUDA results; CPU CI never requires
CUDA. The canonical T4/Colab procedure (fine-tune + benchmark notebooks) lives
in `notebooks/` and `README.md` ("Real data" sections).

## 5. ROS 2 development path

ROS 2 does **not** run on Windows. The ROS 2 workflow is Linux-only:

1. Use ROS 2 Humble (or newer) on Linux with Python 3.10+.
2. Build the adapter: `cd foveamap_ros && colcon build` (see
   `docs/ROS2_INTEGRATION.md` and `foveamap_ros/package.xml`).
3. Run the adapter suite where rclpy exists: `pytest -q -m ros2`
   (the two `test_phase8_ros.py` integration tests skip with a reason when
   `rclpy` is absent; all other adapter tests run anywhere).

## 6. Test workflow and markers

Markers are registered in `pyproject.toml` (`[tool.pytest.ini_options]`):

| Marker | Meaning |
|---|---|
| `unit` | fast hermetic tests (convention; not yet applied per-file) |
| `integration` | multi-component tests on deterministic fixtures |
| `adversarial` | applied: `tests/test_hardening_p4_adversarial.py` |
| `cuda` | applied: physical-CUDA tests, skip on CPU |
| `ros2` | applied: the two rclpy integration tests |
| `slow` / `benchmark` / `hardware` | reserved for future expensive/hardware gates |

Current state (verified): 34 test modules, 435 passed / 5 skipped on CPU
(no unconditional skips; all skips are hardware-conditional with reasons).
Correctness tests and performance tests are separate: `pytest` never enforces
latency; benchmarks report numbers via `foveamap bench` and `benchmarks/`.

Test independence: grid tests assert integer lattice invariants
(no-lost-points, exact nesting, world alignment after scroll) against both
engines; `grid_torch` parity is checked cell-by-cell against the NumPy engine
plus scrolling-drive parity. Contracts reject NaN/Inf and shape mismatches
(`ContractError`). See `ARCHITECTURE.md` for the oracle structure.

## 7. Fixtures and sample data

- `tests/conftest.py::drive` — deterministic 4-frame simulated drive
  (`seed=7`), written to a tmp dir; the canonical small fixture.
- `tests/mock_nuscenes.py`, `tests/mock_semantickitti.py` — in-repo mock
  datasets in the real file layouts (rotated world/mount, shuffled laser ids).
- `foveamap/data/sim.py` + `foveamap/sim.py` — procedural streets and the
  vectorized 64×1024 ray caster; `create_source("sim", …)` needs no download.
- Real datasets (SemanticKITTI, nuScenes) are **never** in Git: they are
  fetched/cached by `scripts/prepare_*.py` and the Colab notebooks. Keep new
  fixtures small, seeded, license-safe, and clearly TEST vs REAL.

## 8. Contracts (authoritative)

Single sources of truth: `foveamap/core/contracts.py`,
`foveamap/core/ontology.py`, `foveamap/core/config.py`,
`foveamap/core/exceptions.py`, `foveamap/sdk/types.py`.

| Contract | Key fields / units |
|---|---|
| `LiDARFrame` | `pts` (N,3) float32 **ego frame, X-fwd / Y-left / Z-up, meters**; `intensity` ≥ 0; `ring` int16; `pose` (4,4) float64 SE(3); `sensor_origin` [0,0,1.73] m; `timestamp` s; `label` int8 / `moving` bool / `time_offsets` s (≤600) optional |
| `PerceptionResult` | `class_probabilities` (N,9) ∈ [0,1]; `moving_probabilities` (N,) ∈ [0,1]; `semantic_predictions` ∈ [0,9); `is_moving`; `confidence` optional |
| `MapSnapshot` | `ego_pose` (copied, read-only); `origins`, `tier_states`; `dynamic_cells`, `dynamic_tracks` (detached); `query_point(x, y)` finest→coarsest; `is_traversable(x, y, clearance_req)` |
| Ontology (`core/ontology.py`) | 9 classes `road,sidewalk,parking,terrain,vegetation,building,pole,vehicle,person` (ids 0–8, **never renumber**); `GROUND=(0,1,2,3)`, `DRIVABLE=(0,2)`, `DYNAMIC=(7,8)` |
| Errors | `FoveaMapError` → `ConfigurationError, ContractError, DataAdapterError, PerceptionError, MappingError, NumericalConsistencyError` (+ SDK `SDKQueryError`) |

Ownership: `MapSnapshot` views are read-only snapshots — treat returned arrays
as borrowed, copy before mutating. `LiDARFrame.validate()` rejects bad inputs
at the boundary; do not bypass it.

## 9. Configuration

`FoveaMapConfig` (frozen dataclasses) with profiles `cpu_dev()`, `gpu_dev()`,
`benchmark()`, `demo()`, `ros2()`, plus `from_env()` overrides:

| Env var | Values | Default |
|---|---|---|
| `FOVEAMAP_PROFILE` | cpu/gpu/demo/bench/ros2 | cpu |
| `FOVEAMAP_DEVICE` | e.g. cpu, cuda:0 | profile |
| `FOVEAMAP_GRID_ENGINE` | numpy/torch | profile |
| `FOVEAMAP_FEATURES_ENGINE` | numpy/torch | profile |
| `FOVEAMAP_CHECKPOINT` | path | `checkpoints/range_unet.pt` if present |
| `FOVEAMAP_PROFILING` | 1/0 | profile |
| `FOVEAMAP_ALLOW_INSECURE_REMOTE` | 1/true/yes | false (loopback-only) |

Grid presets: `spec` (5 cm ±10 m + 50 cm ±100 m), `graded`
(5/10/50 cm), `uniform5`, `uniform50`. Units are meters/seconds; thresholds
(`conf_thresh`, costs, staleness) live in `PerceptionConfig`/`TerrainConfig`/
`DynamicConfig` — do not scatter magic constants in new code.

## 10. HTTP / SDK workflow

```bash
foveamap serve --port 8000            # API + dashboard (loopback by default)
curl localhost:8000/health
curl localhost:8000/status
```

Python SDK (`foveamap/sdk/client.py`):

```python
from foveamap.sdk.client import FoveaMap
sdk = FoveaMap(); sdk.configure(); sdk.start()
```

Endpoints: `/health /status /metrics /map/query /map/snapshot` (+ `/reset`).
HTTP validates strictly and never leaks stack traces; use SDK errors
(`SDKQueryError`) and server logs when debugging. Binding non-loopback
requires `--allow-insecure-remote` (and `FOVEAMAP_ALLOW_INSECURE_REMOTE`).

## 11. Docker workflow

Use Python directly for the inner loop; use Docker for parity/deployment
checks. The image is CPU-based (`python:3.11-slim`, non-root `appuser`):

```bash
docker build -t foveamap-dev-check .
docker run --rm foveamap-dev-check info
docker run --rm -p 8000:8000 foveamap-dev-check serve --port 8000
```

## 12. Benchmark / profiling workflow

```bash
foveamap bench --engine numpy --frames 100
foveamap bench --engine torch --frames 100
foveamap compare --points 50000
python scripts/run_benchmark.py --grid torch   # full harness incl. dashboard export
```

Report p50/p95, FPS, engine/device, point counts, and host. Keep perf tests
out of the unit suite (`benchmarks/bench_*.py` are harnesses, not gates).

## 13. Debugging and logging

There is no `logging` framework yet: the product CLI prints human-readable
progress to stdout (errors to stderr), and benchmarks print tables. When
debugging: reproduce with a seed (`demo --seed`), shrink to a focused pytest
(`-k …`), inspect `runtime.last_timing` / `get_metrics()`, and check
`validate()` errors (`ContractError` names the field and reason). Do not add
per-point logging or print secrets; keep new error messages actionable
(what failed, where, what to do).

## 14. Extension points (add X without touching unrelated modules)

| Add… | Interface → register → use |
|---|---|
| Perception model | implement backend in `foveamap/runtime/perception.py` (predict → `PerceptionResult`) → `list_perception_backends()` → `PerceptionConfig(backend=…)` |
| Data source | implement `foveamap/data/source.py` protocol → `foveamap/data/factory.py` registry → `create_source(name, …)` |
| Semantic class | **do not renumber**: extend `core/ontology.py` + loaders + cost priors together, with tests |
| Mapping strategy | `foveamap/grid.py` (NumPy) + `grid_torch.py` (parity!) + `GridConfig.from_preset` |
| Terrain metric | `foveamap/terrain.py` + `TerrainConfig` field + Phase-7-style tests |
| Temporal behavior | `foveamap/temporal.py` + `DynamicConfig` + dynamic lifecycle tests |
| SDK endpoint | `foveamap/sdk/client.py` + `sdk/http.py` route + `sdk/types.py` schema + Phase-9-style tests |
| ROS adapter | `foveamap_ros/` module + `package.xml`/`CMakeLists.txt` + Phase-8-style tests |
| Benchmark | `benchmarks/bench_*.py` or `foveamap/benchmarks/` harness; never gate correctness on timing |

Untrained/random-weight inference (`allow_untrained=True`) is development-only
and warns loudly — never deploy it.

## 15. Contribution workflow

1. Branch from `dev`; keep changes scoped with a development-readiness reason.
2. Add/extend tests for behavior changes; keep public contracts stable.
3. Run `python scripts/dev.py lint`, `test-fast`, then full `test`, plus
   `demo --frames 3`.
4. Performance changes must include before/after benchmark numbers.
5. API changes must update `docs/API.md` / `docs/SDK.md` and `ARCHITECTURE.md`
   if the data flow changes.
6. PRs: what changed, why, validation commands, and (if applicable) benchmark
   deltas. No secrets, no absolute local paths, no disabled tests.

## 16. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `torch` import fails | install CPU wheel first: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| CUDA tests skip | expected on CPU hosts; run `-m cuda` only where CUDA exists |
| rclpy tests skip | expected without ROS 2; see §5 |
| `demo` slow on CPU | expected (~2 FPS on CPU); GPU path is documented in §4 |
| Port in use on `serve` | change `--port` or stop the previous server |
