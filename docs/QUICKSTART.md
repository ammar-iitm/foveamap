# FoveaMap Quickstart Guide

This guide provides reproducible, copy-paste instructions for installing, inspecting, benchmarking, serving, and deploying FoveaMap.

---

## 1. Installation

Install FoveaMap in editable mode with development dependencies:

```bash
git clone https://github.com/ammar-iitm/foveamap.git
cd foveamap
pip install -e ".[dev]"
```

Verify installation and environment capabilities:

```bash
foveamap info
```

Or via Python module:

```bash
python -m foveamap info
```

---

## 2. Canonical Demonstration

Run the self-contained live demonstration streaming 64-beam spinning LiDAR frames through perception, foveated grid mapping, dynamic tracking, terrain analysis, and point queries:

```bash
foveamap demo --frames 15
```

To launch the interactive web dashboard immediately following the demonstration:

```bash
foveamap demo --frames 15 --serve --port 8080
```

Open `http://127.0.0.1:8080/` in your browser.

---

## 3. Reproducible Baseline Comparison

Compare FoveaMap's multi-tier foveated architecture against a uniform 5 cm 2.5D grid baseline across the full 100 m extent:

```bash
# Theoretical geometry + live empirical latency benchmark
foveamap compare --points 50000

# Fast theoretical calculation only
foveamap compare --no-empirical

# Output structured JSON
foveamap compare --no-empirical --json
```

**Measured Invariants:**
- FoveaMap (spec profile): 320,000 cells (5.12 MB)
- Uniform 5 cm baseline: 16,000,000 cells (256.00 MB)
- Memory reduction: **50.0×** (exceeds PRD ≥ 30× target; under 8 MB target = `True`)
- Update throughput speedup: **> 50×** faster than uniform grid update.

---

## 4. Ingesting Point Clouds & Sequences

Process individual point cloud files (`.pcd`, `.bin`, `.npy`) or recorded frame sequence directories:

```bash
# Process a single PCD file
foveamap run data/sample.pcd

# Process a directory sequence and record metrics
foveamap run /path/to/kitti/sequence/00/velodyne/ --out results/kitti_00_run.json
```

---

## 5. Local HTTP API & Interactive Dashboard

Start the local HTTP service serving both the REST API and the interactive HTML5 canvas dashboard:

```bash
foveamap serve --port 8000 --dashboard
```

### Endpoints
- `GET /`: Interactive web dashboard
- `GET /health`: Health status (`healthy` / `degraded`)
- `GET /status`: Lifecycle, frame counters, hardware device context
- `GET /metrics`: Live latency breakdown, FPS, memory accounting, active tracks
- `GET /map/snapshot`: Latest published canonical MapSnapshot summary
- `GET /map/query?x=5.0&y=0.0`: Spatial cell query at continuous world coordinate (x, y)
- `POST /frames`: Ingest JSON sweep (`{"pts": [[x,y,z], ...], "frame_id": "f0"}`)
- `POST /reset`: Reset grid and temporal states
- `POST /lifecycle`: Control session state (`{"action": "start"|"stop"|"configure"}`)

---

## 6. Multi-Frame Benchmark Harness

Execute sustained multi-frame benchmarks (NumPy CPU or PyTorch CUDA):

```bash
# CPU benchmark (NumPy engine)
foveamap bench --engine numpy --frames 100 --points 50000

# GPU benchmark (PyTorch engine, requires CUDA device)
foveamap bench --engine torch --device cuda --frames 1000 --points 50000
```

---

## 7. Python SDK Usage

```python
from foveamap import FoveaMap, FoveaMapConfig, LiDARFrame
import numpy as np

# 1. Initialize session using a standard deployment profile
config = FoveaMapConfig.cpu_dev()  # or gpu_dev(), demo(), benchmark(), ros2()
client = FoveaMap(config)
client.configure()
client.start()

# 2. Ingest LiDAR frames
frame = LiDARFrame(
    pts=np.random.uniform(-20, 20, (10000, 3)).astype(np.float32),
    intensity=np.ones(10000, dtype=np.float32),
    ring=np.zeros(10000, dtype=np.int16),
    pose=np.eye(4, dtype=np.float64),
    frame_id="frame_001",
)
snapshot = client.process(frame)

# 3. Query the authoritative 2.5D map
q = client.query_point(5.0, 0.0)
print(f"Cell class: {q.dominant_class}, elevation: {q.ground_m}m, traversable: {q.traversable}")

# 4. Check live session metrics
metrics = client.metrics()
print(f"Processed: {metrics.frames_processed}, FPS: {metrics.fps}")

client.close()
```

---

## 8. Deployment Profiles & Environment Variables

FoveaMap supports deterministic configuration via factory classmethods and environment variables:

| Profile | Method | Device | Grid Engine | Features Engine | Perception |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU Dev** | `FoveaMapConfig.cpu_dev()` | `cpu` | `numpy` | `numpy` | Classical / CPU |
| **GPU Dev** | `FoveaMapConfig.gpu_dev()` | `auto` | `torch` | `torch` | RangeUNet (FP16) |
| **Benchmark** | `FoveaMapConfig.benchmark()` | `auto` | `torch` | `torch` | Profiling enabled |
| **Demo** | `FoveaMapConfig.demo()` | `auto` | `numpy` | `numpy` | Profiling enabled |
| **ROS 2** | `FoveaMapConfig.ros2()` | `auto` | `numpy` | `numpy` | Real-time settings |

### Environment Overrides
You can override settings without modifying code:
```bash
export FOVEAMAP_PROFILE=gpu          # cpu | gpu | demo | bench | ros2
export FOVEAMAP_DEVICE=cuda:0        # auto | cpu | cuda | cuda:0
export FOVEAMAP_GRID_ENGINE=torch    # numpy | torch
export FOVEAMAP_PROFILING=1          # 1 | 0
export FOVEAMAP_CHECKPOINT=checkpoints/best_model.pt
```

Construct from environment:
```python
config = FoveaMapConfig.from_env()
```

---

## 9. Running Tests

```bash
# Run full test suite
pytest

# Run Phase 11 productization tests only
pytest tests/test_phase11_product.py -v
```

---

## 10. Docker Container Deployment

Build and run FoveaMap inside the production container:

```bash
# Build the production container
docker build -t foveamap:latest .

# Inspect runtime capabilities
docker run --rm foveamap:latest info

# Run canonical demonstration
docker run --rm foveamap:latest demo --frames 15

# Run baseline comparison
docker run --rm foveamap:latest compare --no-empirical

# Launch HTTP API & dashboard server
docker run --rm -p 8000:8000 foveamap:latest serve --host 0.0.0.0 --port 8000
```
