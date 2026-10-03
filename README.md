# FoveaMap prototype

A working prototype of the FoveaMap design (see the PRD, Architecture Vision and
Visual Design Document). It turns Lidar sweeps into a **variable-resolution 2.5D
semantic grid**: 5 cm cells within ±10 m and 50 cm cells out to ±100 m. The grid
uses 50× less memory than a uniform 5 cm grid, and no point is lost where the
tiers meet.

**Live dashboard: [foveamap-teal.vercel.app](https://foveamap-teal.vercel.app)**, a replay of real
Lidar (nuScenes scene-0103) run through the GPU pipeline on a T4.

It runs on three data sources through the same code:
- **Simulated Lidar**: built in, no download needed, exact labels.
- **Real Lidar from SemanticKITTI**: 64-beam, the main real-data training set, through its Colab notebook.
- **Real Lidar from nuScenes-mini**: 32-beam, 8 training scenes, through its Colab notebook.

| Colab notebook (T4 GPU) | What it does |
| --- | --- |
| [**SemanticKITTI**](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_semantickitti_colab.ipynb) | Fine-tunes on SemanticKITTI and scores sequence 08 by distance and class. |
| [**nuScenes-mini**](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_nuscenes_colab.ipynb) | Fine-tunes on nuScenes-mini, compares training recipes, and benchmarks latency with both grid engines. |

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

## Real data: SemanticKITTI on Colab

nuScenes-mini's 8 training scenes are too few to learn real-world classes. SemanticKITTI has 19,130 labelled training scans from a 64-beam HDL-64E, the sensor the simulator models.

1. Register at [cvlibs.net](http://www.cvlibs.net/datasets/kitti/user_register.php) and accept the KITTI terms (CC BY-NC-SA 3.0; SemanticKITTI is CC BY-NC-SA 4.0, non-commercial).
2. Open [`notebooks/foveamap_semantickitti_colab.ipynb` in Google Colab](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_semantickitti_colab.ipynb), set a T4 runtime and *Run all* (about 40–60 minutes the first time). It downloads to the runtime's local disk, builds the frame cache, and saves the cache (about 14 GB at stride 10) to your Google Drive under `MyDrive/foveamap_data/`, so later sessions copy it back in about 5 minutes instead of downloading and rebuilding.

The notebook scores the simulator model on sequence 08, fine-tunes on sequences 00–07, 09 and 10, scores sequence 08 again and benchmarks it. Outside Colab:

```bash
python scripts/prepare_semantickitti.py --root /path/to/kitti --out cache/semantickitti --stride 10
python scripts/train.py --dataset semantickitti --init checkpoints/range_unet.pt --epochs 20
python scripts/run_benchmark.py --dataset semantickitti --scene 08 --max-frames 100 --grid torch
```

How the loader maps SemanticKITTI onto FoveaMap:

- **Only the scans it needs.** The KITTI velodyne zip is 85 GB. `prepare_semantickitti.py` fetches every n-th scan (default 10) and the two scans before it straight from the remote zip with HTTP range requests, about 14 GB at stride 10.
- **Laser rows.** The files carry no laser id, so rows come from elevation over the simulator's field of view (+2° to −24.9°, 64 rows).
- **Frames and poses.** The ego frame is the Lidar frame moved down 1.73 m to the ground. Poses are cam0 poses converted to the Lidar with the calibration, so the world is the first scan's ego frame.
- **Classes and moving flags.** SemanticKITTI's classes map onto all 9 FoveaMap classes, and its moving-car and moving-person labels give the moving flags.

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
| `foveamap/semantickitti.py` | The SemanticKITTI loader, splits, class map, moving flags and frame cache, plus fetching selected scans from the remote KITTI zip (`foveamap/remote_zip.py`). |
| `foveamap/model.py` | The range-image U-Net, with a 9-class head and a moving/static head. It runs on GPU with FP16 or on CPU, and can mask out classes a dataset doesn't have. |
| `foveamap/grid.py` | **The foveated grid engine.** It uses integer fine indices with floor-division per tier, so tiers nest exactly. Windows snap to the coarse lattice and scroll with the vehicle. Each cell is a 16-byte structure-of-arrays record. The engine also does fused class, ground height, roughness, overhang clearance, curb-step and pothole flags (cells clearly below a local plane fit of the drivable ground), and traversability cost. |
| `foveamap/grid_torch.py` | The same grid engine in PyTorch, for any `torch.device` (CUDA, MPS or CPU). It keeps the 16-byte cell layout and exact integer nesting, and is checked for parity against `grid.py`. Select it with `--grid torch`. |
| `foveamap/features_torch.py` | The range-image features in PyTorch, so `--grid torch` keeps the whole path from features to map on the GPU. |
| `foveamap/pipeline.py` | The pipeline and benchmark harness. It records per-stage latency, measured memory against uniform baselines, and accuracy by distance band (points and grid cells). It also measures moving IoU, curb and pothole recall (simulator only), and a per-frame integrity check, and exports the dashboard data. |
| `scripts/` | `gen_data.py` (simulated drives), `prepare_nuscenes.py` and `prepare_semantickitti.py` (frame caches), `train.py` (sim, or nuScenes / SemanticKITTI fine-tune), `run_benchmark.py`, `make_local_view.py`. |
| `notebooks/` | The Colab notebooks: `foveamap_semantickitti_colab.ipynb` (generated by `build_notebook_semantickitti.py`) and `foveamap_nuscenes_colab.ipynb` (generated by `build_notebook.py`). |
| `dashboard/index.html` | The replay dashboard. It rotates the world-aligned grid so the vehicle's heading is always up. |
| `site/data/` | The replay the live dashboard shows. Vercel builds every push to `main` (`vercel.json` runs `scripts/build_site.py`, which wraps the dashboard into `site/index.html`). To publish another run, copy its dashboard data here and push. |
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

Simulated data is easier than real Lidar, so treat these numbers as a check that the pipeline works, not as benchmark claims. The real-data numbers come from the SemanticKITTI and nuScenes notebooks below.

## Results, real Lidar (SemanticKITTI)

Measured with the SemanticKITTI notebook on a T4 GPU (stride 10, 20 epochs, seed 0). The model was fine-tuned from the simulator checkpoint on 1,922 frames from sequences 00–07, 09 and 10, and scored on every 10th scan of sequence 08, which it never saw. The full outputs are in [`results/semantickitti/`](results/semantickitti/).

| Sequence 08, all points | Simulator model | Fine-tuned |
| --- | --- | --- |
| **mIoU** | 10.7% | **58.1%** |
| mIoU 0–10 m / 10–25 / 25–50 / 50–100 m | 9.9% / 10.2% / 8.8% / 0.9% | 62.4% / 53.8% / 40.1% / 18.3% |
| Road / sidewalk / parking | 16.6% / 21.7% / 2.8% | 87.9% / 69.3% / 25.7% |
| Terrain / vegetation / building | 21.5% / 12.1% / 17.3% | 69.5% / 76.4% / 65.5% |
| Pole / vehicle / person | 1.4% / 3.1% / 0.3% | 22.9% / 76.1% / 29.4% |
| Moving-object IoU | 0.8% | 31.7% |

| Check (first 100 frames of sequence 08, 1 s apart) | Result | Target | Status |
| --- | --- | --- | --- |
| Map memory / saving vs uniform 5 cm | 5.12 MB / 50× | ≤ 8 MB / ≥ 30× | pass |
| Points lost at tier edges | 0 | 0 | pass |
| Drivable IoU on grid, 0–10 m | 90.9% | ≥ 90% | pass |
| Point mIoU, 0–10 m / 10–25 / 25–50 / 50–100 m | 59.9% / 51.3% / 40.4% / 11.9% | ≥ 70% near | fail |
| p50 / p95 latency, features + grid engine on the GPU | 36 / 46 ms | ≤ 50 ms p95 | pass |
| Throughput, same run | 26.2 FPS | ≥ 20 FPS | pass |

KITTI scans average about 123,000 points, nearly 5 times as many as nuScenes keyframes (about 26,000), so every stage has more to do. The first run kept the point transforms on the CPU in float64 and missed the target. Moving them to the GPU fixed it:

| Stage means, T4 (ms) | Transforms on the CPU | Transforms on the GPU |
| --- | --- | --- |
| Preprocess | 22.2 | 12.5 |
| Network | 4.6 | 4.2 |
| Projection | 16.2 | 7.2 |
| Fusion + cost | 14.1 | 12.5 |
| Publish (map snapshot to host) | 2.1 | 1.7 |
| **p50 / p95 end to end** | **47 / 149** | **36 / 46** |
| Throughput (FPS) | 16.9 | 26.2 |

The p95 margin is thin and comes from one run. Colab's shared vCPUs have busy spells that slow every stage at once for a few seconds: in two runs with the CPU transforms, the slow stretches fell in different places (frames 26–36 and 90–98, then 50–58) and those frames had normal point counts, so they come from the machine, not the data. The run with GPU transforms is `seq08_metrics.json` / `benchmark_seq08.log`; the earlier one is `*_cpu_transforms*`.

- **The simulator alone doesn't carry over to real Lidar.** It scores 10.7% on SemanticKITTI, about the same as on nuScenes (9.8%), even though KITTI's 64-beam HDL-64E is the sensor it simulates. The gap comes from simulated versus real scenes, not the beam count.
- **Fine-tuning brings it to 58.1% mIoU**, with road at 87.9% and vehicles at 76.1%, against 46.3% after fine-tuning on nuScenes-mini's 8 scenes.
- **Weak classes.** Most poles and people are found (recall 67% and 61%), but too many other points are labelled as them, so their IoU stays low (22.9% and 29.4%). Parking (25.7%) is flat ground that looks like road or sidewalk.
- **Reproducible.** Two training runs with seed 0 gave identical scores, down to the per-class IoUs.
- **Training loss was still falling** after 20 epochs, so longer training may help.

## Results, real Lidar (nuScenes)

These were measured with the Colab notebook on a T4 GPU. The model was fine-tuned from the simulator checkpoint on the 8 nuScenes-mini training scenes and scored 46.3% mIoU on the 2 validation scenes. The full outputs are in [`results/nuscenes/`](results/nuscenes/).

| Check (scene-0103, 40 keyframes) | Result | Target | Status |
| --- | --- | --- | --- |
| Map memory / saving vs uniform 5 cm | 5.12 MB / 50× | ≤ 8 MB / ≥ 30× | pass |
| Points lost at tier edges | 0 | 0 | pass |
| Drivable IoU on grid, 0–10 m | 94.0% | ≥ 90% | pass |
| Point mIoU, 0–10 m / 10–25 / 25–50 / 50–100 m | 44.8% / 43.5% / 30.8% / 15.4% | ≥ 70% near | fail |
| Moving-object IoU | 37.5% | reported | — |
| p50 / p95 latency, features + grid engine on the GPU | 30 / 31 ms | ≤ 50 ms p95 | pass |
| Throughput, same run | 33.4 FPS | ≥ 20 FPS | pass |

All rows come from the run with features and grid engine on the GPU. [`results/nuscenes/`](results/nuscenes/) has its metrics and per-frame log (`scene-0103_metrics.json`, `benchmark_scene-0103.log`), next to the NumPy and concurrent-export runs from the same session (`*_numpy*`, `*_torch_async*`). The NumPy engine gives the same accuracy apart from the 50–100 m band (15.5% point mIoU), where rare tie pixels in the GPU range image change a few points.

Latency is the map pipeline: sweep in, fused map snapshot on the host out. The dashboard PNGs (map and ground-truth tiles) are a benchmark artifact, so the benchmark encodes them after the timed loop by default (`--export after`) and reports that time separately. Encoding them concurrently on the same 2 vCPUs (`--export async`) starves the pipeline thread and raises p95 to 87 ms.

| Stage means, T4 (ms) | NumPy (CPU) | GPU, PNG export concurrent | GPU, PNG export after the loop |
| --- | --- | --- | --- |
| Preprocess | 24.3 | 14.5 | 6.8 |
| Network | 7.4 | 6.2 | 3.0 |
| Projection | 33.1 | 10.3 | 5.3 |
| Fusion + cost | 94.1 | 22.7 | 10.8 |
| Publish (map snapshot to host) | 1.8 | 4.8 | 4.0 |
| **p50 / p95 end to end** | **137 / 250** | **56 / 87** | **30 / 31** |
| Throughput (FPS) | 6.2 | 17.1 | 33.4 |
| PNG export per frame (not in latency) | 208 | 173 | 141 |

All three columns come from one Colab session. The NumPy column exports concurrently, as it did before `--export` existed. With PNG export after the loop, 37 of the 38 timed frames take 28.6–31.1 ms and the slowest takes 41.7 ms. Earlier runs, before the snapshot copy was packed and with export concurrent, measured 53 / 61 ms (GPU grid, CPU features) and 43 / 63 ms (GPU grid and features).

- **Structure holds up.** Memory, point conservation and drivable surface hold up on real data.
- **Accuracy is limited by data.** 8 training scenes are far too few, and the weakest classes are terrain, barriers and cones, pedestrians and sidewalk.
- **Speed: 31 ms p95 at 33 FPS on a T4.** This is with the features and grid engine on the GPU, 4.5× faster at p50 than NumPy. It excludes the dashboard's PNG encoding (about 140 ms per frame). Encoding concurrently on Colab's 2 vCPUs pushes p95 to 87 ms.
- **Potholes.** The first pothole test (5 cm below the mean of drivable ground within 2.5 m) flagged 11% of drivable 5 cm cells and 20% of 50 cm cells in this replay, where there are no potholes. The flags followed whole laser rings: each of the 32 lasers sits a few cm higher or lower than its neighbours, and the test read the low rings as holes. A cell now counts as a pothole when it is below a least-squares plane of the drivable ground within 5 m by at least 5 cm and three times the ground's scatter about that plane. The plane absorbs grade, camber and pitch error; the scatter term absorbs the ring offsets. Re-scored offline on this replay's stored cell heights (1.8 cm steps), the false flags fall to 0.09% (5 cm cells) and 0.32% (50 cm cells), while the simulated drive still finds every pothole. The trade-off is that a pothole has to stand out from the local ring noise: with ±5 cm ring offsets a 10 cm pothole is barely flagged.

## How the prototype differs from the full design

- **Backbone.** The prototype uses the range-image network (the "low-power fallback" in Architecture section 4) instead of a sparse-conv U-Net. The training loop and the grid engine don't depend on which backbone you use.
- **Grid engine.** The grid engine runs in NumPy or PyTorch (`--grid torch`). The PyTorch engine is parity-tested against NumPy and runs on the GPU. With `--grid torch`, the range-image features are also built on the GPU (`foveamap/features_torch.py`, also parity-tested; override with `--features numpy`). Dashboard PNG export is encoded after the timed loop by default and reported as `export_ms`, not counted in latency (`--export async` encodes on a background thread during the run instead, and `none` skips it).
- **Not built yet:** 3D view, velocity arrows (no tracker), free-space ray clearing, ROS 2 node, and TensorRT export.
