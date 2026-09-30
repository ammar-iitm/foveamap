# FoveaMap prototype

A working prototype of the FoveaMap design (see the PRD, Architecture Vision and
Visual Design Document). It turns Lidar sweeps into a **variable-resolution 2.5D
semantic grid**: 5 cm cells within ±10 m and 50 cm cells out to ±100 m. The grid
uses 50× less memory than a uniform 5 cm grid, and no point is lost where the
tiers meet.

It runs on two data sources through the same code:
- **Simulated Lidar**: built in, no download needed, exact labels.
- **Real Lidar from nuScenes-mini**: through the Colab notebook, on a free GPU.

```
sweep ──► features ──► range-image U-Net ──► foveated grid engine ──► fusion + cost ──► dashboard frames
          (8 ch)       (9 classes + moving)   (tier select, scatter-     (EMA, overhang,     + metrics.json
                                               reduce, mip-up)           steps, potholes)
```

## Design documents

| Document | What it covers |
| --- | --- |
| [`docs/FoveaMap_PRD.pdf`](docs/FoveaMap_PRD.pdf) | Product requirements: goals, user stories, functional and non-functional requirements, acceptance criteria, roadmap |
| [`docs/FoveaMap_Architecture_Vision.pdf`](docs/FoveaMap_Architecture_Vision.pdf) | Pipeline, perception model, tiered grid design, projection and fusion rules, latency and memory budget, key decisions |
| [`docs/FoveaMap_Visual_Design.pdf`](docs/FoveaMap_Visual_Design.pdf) | Dashboard layout, colour system, map rendering, components, interaction, accessibility |

## Real data: nuScenes-mini on Colab (no Lidar hardware needed)

1. Open [`notebooks/foveamap_nuscenes_colab.ipynb` in Google Colab](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_nuscenes_colab.ipynb).
2. Set *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. The notebook clones the latest `main` from GitHub.

The notebook downloads nuScenes-mini and its lidarseg labels (about 4 GB) and checks the loader on one frame. It then scores the simulator-trained model on real data, fine-tunes on the 8 `mini_train` scenes, benchmarks a `mini_val` scene, shows the dashboard inline, and zips the results. It takes about 15–25 minutes in total.

If you have the data locally:

```bash
python scripts/prepare_nuscenes.py --dataroot /path/to/nuscenes --out cache/nuscenes
python scripts/train.py --dataset nuscenes --cache cache/nuscenes --init checkpoints/range_unet.pt --epochs 120
python scripts/run_benchmark.py --dataset nuscenes --cache cache/nuscenes --scene scene-0103 \
    --ckpt checkpoints/range_unet_nuscenes.pt --out dashboard/data
```

How the loader maps nuScenes onto FoveaMap:

- **No devkit needed.** It reads the JSON tables directly.
- **Laser rows.** Laser ids are sorted top to bottom by elevation to make range-image rows.
- **Motion cue.** It uses the sweeps 0.1 s and 0.2 s before each keyframe (nuScenes records sweeps at 20 Hz).
- **Moving flags.** They come from annotation boxes whose attribute is moving, or whose speed is above 0.5 m/s.
- **Classes.** The 32 lidarseg classes map to FoveaMap's. nuScenes has no parking or pole/sign class, so the "pole" slot holds barriers and cones and parking is masked out.

## What's in the box

| Path | What it does |
| --- | --- |
| `foveamap/sim.py` | Procedural streets and a vectorised 64 × 1024 Lidar ray caster, with exact per-point labels and moving flags. Scenes have 15 cm curbs, potholes, parking lots, buildings, walls, poles and signs, trees with overhanging canopies, an overhead gantry, parked and moving cars, and walking or crossing pedestrians. |
| `foveamap/frames.py` | Turns any Lidar source into one frame format: ego-frame points, laser row, ego→world pose (with heading), and the earlier sweeps. It also builds the 8-channel range-image features, which use the earlier sweeps as a motion cue. |
| `foveamap/nuscenes.py` | The nuScenes and nuScenes-lidarseg loader, splits, class map, moving flags and frame cache. |
| `foveamap/model.py` | The range-image U-Net, with a 9-class head and a moving/static head. It runs on GPU with FP16 or on CPU, and can mask out classes a dataset doesn't have. |
| `foveamap/grid.py` | **The foveated grid engine.** It uses integer fine indices with floor-division per tier, so tiers nest exactly. Windows snap to the coarse lattice and scroll with the vehicle. Each cell is a 16-byte structure-of-arrays record. The engine also does fused class, ground height, roughness, overhang clearance, curb-step and pothole flags, and traversability cost. |
| `foveamap/grid_torch.py` | The same grid engine in PyTorch, for any `torch.device` (CUDA, MPS or CPU). It keeps the 16-byte cell layout and exact integer nesting, and is checked for parity against `grid.py`. Select it with `--grid torch`. |
| `foveamap/pipeline.py` | The pipeline and benchmark harness. It records per-stage latency, measured memory against uniform baselines, and accuracy by distance band (points and grid cells). It also measures moving IoU, curb and pothole recall (simulator only), and a per-frame integrity check, and exports the dashboard data. |
| `scripts/` | `gen_data.py` (simulated drives), `prepare_nuscenes.py` (frame cache), `train.py` (sim, or nuScenes fine-tune), `run_benchmark.py`, `make_local_view.py`. |
| `notebooks/foveamap_nuscenes_colab.ipynb` | The Colab notebook. `build_notebook.py` generates it. |
| `dashboard/index.html` | The replay dashboard. It rotates the world-aligned grid so the vehicle's heading is always up. |
| `tests/` | Grid invariants (no point lost, exact nesting, world alignment after scrolling, 16 B per cell). There is also a nuScenes loader test on a mock dataset written in nuScenes' exact file layout. The mock uses a rotated world, a rotated sensor mount, shuffled laser ids, lidarseg ids and annotation boxes. Parity tests compare the PyTorch grid engine with the NumPy one, cell by cell and across a scrolling drive, on CPU and on MPS/CUDA when present. |

## Run the simulator version

```bash
pip install -r requirements.txt
python scripts/gen_data.py                            # ~8 min on 2 CPU cores
python scripts/train.py --dataset sim --budget 1000   # ~17 min on CPU
python scripts/run_benchmark.py                       # writes dashboard/data/
python scripts/run_benchmark.py --grid torch          # same, with the PyTorch grid engine (on the GPU if there is one)
python -m pytest -q tests
python scripts/make_local_view.py && cd dashboard && python -m http.server 8000   # open http://localhost:8000/view.html
```

Dashboard keys:

| Key | Action |
| --- | --- |
| `Space` | Play / pause |
| `←` / `→` | Step one frame (`Shift` steps 10) |
| `1`–`4` | Layer: semantic, elevation, traversability, ground truth |
| `C` | Split compare: foveated vs uniform 50 cm |
| `V` | 5 cm tier on/off |
| `E` | Curb and pothole edges |
| `D` | Moving outlines |
| `K` | Confidence fade |
| `G` | Cell grid |
| `P` | Raw points |
| `0` | Reset view |

## Results, simulated drive

Demo drive: 60 frames of a world the model never saw in training. Validation mIoU on a separate drive was 92.1%.

| Check | Result | Target (PRD) | Status |
| --- | --- | --- | --- |
| Map memory, 2-tier foveated (measured bytes) | 5.12 MB | ≤ 8 MB | pass |
| Saving vs uniform 5 cm 2.5D grid (256 MB) | 50× | ≥ 30× | pass |
| Points lost at tier boundaries (60 frames) | 0 | 0 | pass |
| Exact tier nesting, every frame | yes | yes | pass |
| Point mIoU, 0–10 m / 10–25 / 25–50 / 50–100 m | 96.9% / 94.2% / 86.8% / 74.2% | ≥ 70% near | pass |
| Grid-cell mIoU, same bands | 95.4% / 94.1% / 86.3% / 74.2% | ≥ 70% near | pass |
| Drivable IoU on grid, 0–10 m | 99.7% | ≥ 90% | pass |
| Moving-object IoU (points) | 89.7% | reported | — |
| Curb (15 cm) recall within 10 m | 98.9% | ≥ 90% | pass |
| Pothole recall within 10 m | 100.0% | — | — |
| End-to-end latency p50 / p95 (2 vCPU, no GPU) | 318 / 371 ms | ≤ 50 ms on GPU | CPU only |
| Throughput | 3.1 FPS | ≥ 20 FPS on GPU | CPU only |
| Grid engine only vs same engine on uniform 5 cm grid | 119 ms vs 4,041 ms | — | 34× faster |

Stage means (ms): preprocess 38, inference 127, projection 51, fusion 68, publish 39.

Simulated data is easier than real Lidar, so treat these numbers as a check that the pipeline works, not as benchmark claims. The nuScenes notebook produces the real-data numbers.

## Results, real Lidar (nuScenes)

These were measured with the Colab notebook on a T4 GPU. The model was fine-tuned from the simulator checkpoint on the 8 nuScenes-mini training scenes and scored 46.3% mIoU on the 2 validation scenes. The full outputs are in [`results/nuscenes/`](results/nuscenes/).

| Check (scene-0103, 40 keyframes) | Result | Target | Status |
| --- | --- | --- | --- |
| Map memory / saving vs uniform 5 cm | 5.12 MB / 50× | ≤ 8 MB / ≥ 30× | pass |
| Points lost at tier edges | 0 | 0 | pass |
| Drivable IoU on grid, 0–10 m | 94.0% | ≥ 90% | pass |
| Point mIoU, 0–10 m / 10–25 / 25–50 / 50–100 m | 44.8% / 43.5% / 30.8% / 15.5% | ≥ 70% near | fail |
| Moving-object IoU | 37.5% | reported | — |
| p95 latency / throughput, PyTorch grid engine on the GPU | 61 ms / 18.7 FPS | ≤ 50 ms / ≥ 20 FPS | fail (close) |

The latency row comes from a later T4 run that benchmarked both grid engines on the same scene. The other rows, and the files in [`results/nuscenes/`](results/nuscenes/), are from the earlier NumPy-engine run. Drivable IoU (94.0%) and points lost (0) were identical with both engines.

| Stage means, T4 (ms) | NumPy grid (CPU) | PyTorch grid (GPU) |
| --- | --- | --- |
| Preprocess | 19.9 | 19.6 |
| Network | 5.6 | 4.9 |
| Projection | 26.9 | 7.9 |
| Fusion + cost | 83.0 | 16.3 |
| Publish (map snapshot to host) | 1.7 | 4.7 |
| **p50 / p95 end to end** | **136 / 149** | **53 / 61** |
| Background PNG export (not in latency) | 186 | 165 |

The GPU grid engine is 4.5× faster than the NumPy one (24 vs 110 ms for projection + fusion). The biggest remaining stage is preprocessing, which builds the range image on the CPU.

- **Structure holds up.** Memory, point conservation and drivable surface hold up on real data.
- **Accuracy is limited by data.** 8 training scenes are far too few, and the weakest classes are terrain, barriers and cones, pedestrians and sidewalk.
- **Speed: 61 ms p95, close to the 50 ms target.** With the grid engine on the GPU, CPU preprocessing is the biggest stage (about 20 ms). Background PNG export takes about 170 ms per frame, which is longer than a frame, so on Colab's 2 vCPUs it competes with the main thread.
- **Potholes.** The pothole heuristic was tuned on flat simulated roads and flags false potholes on real, cambered ones.

## How the prototype differs from the full design

- **Backbone.** The prototype uses the range-image network (the "low-power fallback" in Architecture section 4) instead of a sparse-conv U-Net. The training loop and the grid engine don't depend on which backbone you use.
- **Grid engine.** The grid engine runs in NumPy or PyTorch (`--grid torch`). The PyTorch engine is parity-tested against NumPy and runs on the GPU. Dashboard PNG export runs on a background thread, reported as `export_ms` and not counted in latency. Feature building (the range image) is still NumPy on the CPU.
- **Not built yet:** 3D view, velocity arrows (no tracker), free-space ray clearing, ROS 2 node, TensorRT export, and a SemanticKITTI loader (it would fill the same frame format as `nuscenes.py`).
