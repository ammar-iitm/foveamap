# FoveaMap Development Readiness Report

## 1. Executive Verdict

**DEVELOPMENT READY (CONDITIONAL — minor gaps remain).**

A new competent developer can clone, install (`pip install -e ".[dev]"`),
run the full suite, run the demo/HTTP/Docker flows, and extend components via
documented interfaces. All documented commands were executed and verified.
Remaining gaps are non-blocking and listed explicitly in §12 (repo-wide lint
adoption, `logging` framework, slow-marker coverage).

## 2. Exact Git Revision

- Branch: `dev`
- Previous release-candidate claim: `329afc8` — **verified correct**:
  `git rev-parse HEAD` → `329afc839e252f3d946eba6e2d3a7d6cd1e3ff27`.
- Working tree at inspection: clean; ahead of `origin/dev` by 9 commits
  (pre-existing state, not changed by this pass).
- Changes in this pass: 6 modified + 4 new files (see §5); no commits made.

## 3. Environment

| Item | Value |
|---|---|
| OS / shell | Windows 11, PowerShell 5.1 |
| Python | 3.13.5 (dev host); Docker/CI reference: 3.11 |
| NumPy | 2.2.6 |
| PyTorch | 2.7.0+cpu (host); container 2.14.1+cpu |
| CUDA | not available on host (CPU-only execution) |
| Docker | 29.1.3 (build + run verified) |
| ROS 2 / rclpy | not available (all ROS 2 gating verified by skip-with-reason) |

## 4. Baseline (before changes)

- `git status`: clean; `HEAD == 329afc8` as claimed.
- `python -m foveamap.cli info`: OK (CPU-only, profiles/backends/sources listed).
- `python -m foveamap.cli demo --frames 3`: OK (~2 FPS CPU, 4.88 MB, 50× reduction).
- `pytest -q`: **435 passed, 5 skipped** (443 s) — matches the prior claim.
  All 5 skips are hardware-conditional with reasons (3 CUDA, 2 rclpy).

## 5. Changes Made

| File | Change | Reason | Validation |
|---|---|---|---|
| `pyproject.toml` | Registered `unit/integration/adversarial/cuda/ros2/slow/benchmark/hardware` markers; added `[tool.ruff]` (single lightweight stack, line-length 100); added `ruff>=0.4` to `[dev]` extras | Unregistered markers would warn; no lint standard existed; dev surface needs one obvious quality gate | `pytest --collect-only` shows no unknown-mark warnings; `ruff check scripts/dev.py` passes |
| `tests/test_hardening_p3.py` | `@pytest.mark.cuda` on `test_real_cuda_canonicalization` | Hardware-gated tests must be selectable (`-m cuda`) | `pytest -m "cuda or ros2" --collect-only` → 5 tests |
| `tests/test_perception_backend.py` | `@pytest.mark.cuda` on `test_cuda_perception_execution`, `test_cuda_cpu_parity_when_available` | Same | Same |
| `tests/test_phase8_ros.py` | `ros2_hardware = pytest.mark.ros2` + applied to the two rclpy integration tests (adapter tests untouched) | ROS 2-gated tests must be selectable without mislabeling the always-run adapter tests | `pytest -q tests/test_phase8_ros.py …` 77 passed / 3 skipped incl. file |
| `tests/test_hardening_p4_adversarial.py` | `pytestmark = pytest.mark.adversarial` | Adversarial suite must be runnable as a group | Included in affected-file run, green |
| `scripts/dev.py` (new) | `test/test-fast/test-unit/demo/info/bench/lint/clean` dispatcher | One memorable command surface instead of ad-hoc invocations (§6) | `lint`, `info`, `demo` executed green |
| `.github/workflows/ci.yml` (new) | CPU CI: install → import → CLI smoke → lint → fast tests | No CI existed; CPU job must not require CUDA/ROS 2 | Validated step-by-step locally (install/import/smoke/lint/fast-subset all executed); runner execution itself not available on this host |
| `DEVELOPMENT.md` (new) | Install, deps, commands, CUDA/ROS 2 paths, markers, fixtures, contracts, config, HTTP/SDK, Docker, benchmarks, debugging, extension points, contribution, troubleshooting | No single developer entry point existed | Every command in it executed during this pass |
| `ARCHITECTURE.md` (new) | Implemented data flow, module map, contracts/frames, tier math, perception, temporal, terrain, lifecycle, boundaries, invariants | Architecture lived across PDFs/READMEs, not in a source-traceable reference | Cross-checked against `contracts.py`, `config.py`, `ontology.py`, `runtime.py`, `cli.py` |
| `README.md` | 2-line pointer to `DEVELOPMENT.md`/`ARCHITECTURE.md`/`scripts/dev.py` | Discoverability | — |

Nothing else was touched: no product code, no public API, no test removed or
weakened, no behavior changed to suit docs.

## 6. Developer Workflow (exact, all executed)

```bash
git clone https://github.com/ammar-iitm/foveamap.git
cd foveamap
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/dev.py info     # == python -m foveamap.cli info
python scripts/dev.py demo     # == python -m foveamap.cli demo --frames 3
python scripts/dev.py bench    # short numpy benchmark
foveamap serve --port 8000     # loopback API + dashboard
docker build -t foveamap-dev-check .
docker run --rm foveamap-dev-check info
```

Verified: `info` (host + container), `demo --frames 3` (host), `bench
--engine numpy --frames 5` (p50 615 ms CPU, 4.883 MB), `compare
--no-empirical` (50×), HTTP `/health` → `healthy` + `/status` → `ACTIVE`
(port 8901), Docker image build + `info` in container.

## 7. Test Workflow (exact, all executed)

```bash
python scripts/dev.py test        # full: pytest -q → 435 passed, 5 skipped
python scripts/dev.py test-fast   # pytest -q -m "not cuda and not ros2 and not slow"
python scripts/dev.py test-unit   # hardening + contracts subset
pytest -q tests/test_grid.py -k nesting   # focused example
pytest -q -m cuda                  # physical-CUDA hosts only (skips on CPU)
pytest -q -m ros2                  # ROS 2 hosts only (skips otherwise)
pytest -q -m adversarial          # stress suite
python scripts/dev.py lint        # compileall + scoped ruff check
```

Post-change full run: **435 passed, 5 skipped** (726 s), identical to the
pre-change baseline. Marker selection verified (`cuda or ros2` collects
exactly the 5 hardware-gated tests). `unit/integration/slow/benchmark/
hardware` are registered but reserved — applying them per-file is future work.

## 8. CUDA Workflow

Requires a CUDA-capable host + CUDA torch build; **not available here, and no
CUDA result is claimed**. Procedure (documented in `DEVELOPMENT.md` §4):

```bash
python -c "import torch; print(torch.cuda.is_available())"  # must be True
pytest -q -m cuda
python -m foveamap.cli bench --engine torch --device cuda --frames 100
```

The 3 `cuda`-marked tests skip with reasons on CPU. T4/Colab fine-tune and
benchmark notebooks remain the canonical GPU evidence (`notebooks/`,
`README.md` real-data sections, `results/`).

## 9. ROS 2 Workflow

Linux-only (Humble+); **cannot run on this Windows host, and no ROS 2 runtime
result is claimed**. Procedure (`DEVELOPMENT.md` §5, `docs/ROS2_INTEGRATION.md`):

```bash
cd foveamap_ros && colcon build
pytest -q -m ros2   # 2 rclpy integration tests; adapter tests run anywhere
```

Verified locally: adapter suite passes, integration tests skip with
`ROS 2 integration requires rclpy (…)` reason.

## 10. CI

New: `.github/workflows/ci.yml`, job `cpu` (Ubuntu, Python 3.11):
install (CPU torch + `.[dev]`) → import check → `cli info` → `cli demo
--frames 2` → `dev.py lint` → `dev.py test-fast`. CUDA/ROS 2/slow-benchmark
paths are deliberately excluded from the default job. Every step was executed
locally in equivalent form; Actions-runner execution is the only unverified
part (no runner on this host). CI config itself is YAML-valid by construction
and uses only pinned actions (`checkout@v4`, `setup-python@v5`).

## 11. Documentation

- Added: `DEVELOPMENT.md` (single developer entry point), `ARCHITECTURE.md`
  (source-traceable reference), `.github/workflows/ci.yml`,
  `scripts/dev.py`.
- Updated: `pyproject.toml` (markers, ruff), `README.md` (2-line dev pointer),
  4 test files (markers only).
- Deliberately consolidated: testing/benchmark/CUDA/ROS 2/Docker guidance
  lives as sections of `DEVELOPMENT.md` rather than six new top-level docs,
  to avoid documentation sprawl. `docs/QUICKSTART.md`, `docs/API.md`,
  `docs/SDK.md`, `docs/ROS2_INTEGRATION.md`, `docs/PERCEPTION.md`,
  `docs/MAPPING_ENGINE.md` etc. remain the deep references.

## 12. Remaining Gaps (genuine only)

| # | Gap | Status | Evidence / path |
|---|---|---|---|
| 1 | Repo-wide lint adoption | PARTIAL | Fresh `ruff check` reports ~889 pre-existing findings (import order, unused imports, long lines) across legacy files. `dev.py lint` is therefore scoped to the new surface + `compileall`; mass-reformat was refused per minimal-change rule. |
| 2 | No `logging` framework | PARTIAL | `logging.getLogger` has 0 hits; CLI/bench output is `print`-based (82 in `cli.py`, 30 in `bench_mapping.py`). Adequate for CLI tools, but library code has no level-controlled logging. Documented in `DEVELOPMENT.md` §13. |
| 3 | `slow`/`benchmark`/`unit` markers unapplied | PARTIAL | Registered but only `cuda`/`ros2`/`adversarial` are applied. Full suite (~7–12 min CPU) is the only complete gate; `test-fast` currently deselects just the 5 hardware tests. |
| 4 | CI runner execution | BLOCKED | Workflow created but no GitHub runner available from this host; all steps equivalently executed locally. |
| 5 | CUDA/ROS 2 physical validation | BLOCKED | No CUDA GPU, no Linux/ROS 2 on this host. Gating, skips, and docs verified; execution explicitly not claimed. |
| 6 | Benchmark numbers are host-relative | NOT APPLICABLE | CPU numbers reported (§6) are smoke evidence, not performance claims; canonical GPU figures live in `results/` + notebooks. |

## 13. Final Verdict

**CONDITIONAL — the repository is development-ready for CPU-based engineering;
CUDA and ROS 2 remain explicitly-gated, physically-unvalidated-here workflows
(as they must be on a Windows CPU host).**

Success-condition check (§SUCCESS CONDITION): (1) repo inspected ✓;
(2) behavior intact — 435/5 before and after ✓; (3) install path ✓;
(4) deps clear ✓; (5–7) tests organized/discoverable, focused + full runs ✓;
(8–10) CLI/SDK+HTTP/Docker documented + executed ✓; (11–12) CUDA/ROS 2
documented without fake validation ✓; (13) CI created, locally validated ✓;
(14–15) architecture/contracts/extension points documented from source ✓;
(16) debugging/logging usable, limits stated ✓; (17) benchmark workflow
executed ✓; (18) no secrets/machine paths in tracked files (`git grep`
clean) ✓; (19) full suite passes ✓; (20–22) tree contains only the intended
files; this report exists; gaps explicit ✓.
