# FoveaMap prototype

A working prototype of the FoveaMap design (see the PRD, Architecture Vision and
Visual Design Document). It turns Lidar sweeps into a **variable-resolution 2.5D
semantic grid**: 5 cm cells within ±10 m and 50 cm cells out to ±100 m. The grid
uses 50× less memory than a uniform 5 cm grid, and no point is lost where the
tiers meet.

**Live dashboard: [foveamap-teal.vercel.app](https://foveamap-teal.vercel.app)**, replays of real
Lidar run through the GPU pipeline on a T4, with object boxes. Switch between them in the header:
SemanticKITTI sequence 08 (64-beam, 100 frames 1 s apart, p50 33–35 ms; p95 44–73 ms depending on the Colab machine) and
[nuScenes scene-0103](https://foveamap-teal.vercel.app/?data=nuscenes) (32-beam, 40 keyframes at 2 Hz, 30 / 43 ms at 31 FPS).
The hosted site replays recorded benchmark runs; to watch the pipeline process frames in real time, use the [live view](#live-view).

It runs on three data sources through the same code:
- **Simulated Lidar**: built in, no download needed, exact labels.
- **Real Lidar from SemanticKITTI**: 64-beam, the main real-data training set, through its Colab notebook.
- **Real Lidar from nuScenes-mini**: 32-beam, 8 training scenes, through its Colab notebook.

| Colab notebook (T4 GPU) | What it does |
| --- | --- |
| [**SemanticKITTI**](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_semantickitti_colab.ipynb) | Fine-tunes on SemanticKITTI and scores sequence 08 by distance and class. |
| [**Benchmark only**](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_benchmark_colab.ipynb) | Re-runs the benchmark with the latest code on SemanticKITTI sequence 08 or nuScenes scene-0103, profiles it and zips the dashboard frames. SemanticKITTI uses the cache and model its notebook saved to Drive (about 10 minutes); nuScenes makes and saves its own on the first run. |
| [**nuScenes-mini**](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_nuscenes_colab.ipynb) | Fine-tunes on nuScenes-mini, compares training recipes, and benchmarks latency with both grid engines. |

```
sweep ──► features ──► range-image U-Net ──► foveated grid engine ──► fusion + cost ──► objects ──► dashboard frames
          (8 ch)       (9 classes + moving)   (tier select, scatter-     (EMA, overhang,     (boxes,     + metrics.json
                                               reduce, mip-up)           steps, potholes)    class,
                                                                                             moving)
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
python scripts/train.py --dataset semantickitti --init checkpoints/range_unet.pt --epochs 40
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
| `foveamap/grid.py` | **The foveated grid engine.** It uses integer fine indices with floor-division per tier, so tiers nest exactly. Windows snap to the coarse lattice and scroll with the vehicle. Each cell is a 16-byte structure-of-arrays record. The engine also does fused class, ground height, roughness, overhang clearance, curb-step and pothole flags (5 cm cells clearly below a local plane fit of the drivable ground), and traversability cost. |
| `foveamap/grid_torch.py` | The same grid engine in PyTorch, for any `torch.device` (CUDA, MPS or CPU). It keeps the 16-byte cell layout and exact integer nesting, and is checked for parity against `grid.py`. Select it with `--grid torch`. |
| `foveamap/features_torch.py` | The range-image features in PyTorch, so `--grid torch` keeps the whole path from features to map on the GPU. |
| `foveamap/objects.py` | Object-level output: groups obstacle cells into objects with a class, moving flag, oriented box and top height, and matches them against ground-truth objects. |
| `foveamap/pipeline.py` | The pipeline and benchmark harness. It records per-stage latency, measured memory against uniform baselines, and accuracy by distance band (points and grid cells). It also measures moving IoU, curb and pothole recall (simulator only), the share of drivable cells flagged as potholes, object precision and recall by class, and a per-frame integrity check, and exports the dashboard data. |
| `scripts/` | `gen_data.py` (simulated drives), `prepare_nuscenes.py` and `prepare_semantickitti.py` (frame caches), `train.py` (sim, or nuScenes / SemanticKITTI fine-tune), `run_benchmark.py`, `make_local_view.py`. |
| `notebooks/` | The Colab notebooks: `foveamap_semantickitti_colab.ipynb` (generated by `build_notebook_semantickitti.py`), `foveamap_benchmark_colab.ipynb` (`build_notebook_benchmark.py`) and `foveamap_nuscenes_colab.ipynb` (`build_notebook.py`). |
| `scripts/live_server.py` | The live view: runs the pipeline on a recorded drive at sensor rate and serves the dashboard as it goes (see [Live view](#live-view)). |
| `dashboard/index.html` | The dashboard: replays a recorded run, or shows the live server's frames as they come. It rotates the world-aligned grid so the vehicle's heading is always up. |
| `site/data/` | The replays the live dashboard shows, one folder per dataset listed in `site/data/datasets.json` (the first is the default; `?data=<id>` or the header switch picks another). Now SemanticKITTI sequence 08 (run 1 at `8d45de1`) and nuScenes scene-0103 (run 1 at `8e654bd`), both from the benchmark-only notebook; the older nuScenes replay, from before objects and the pothole plane fit, is in the git history. Vercel builds every push to `main` (`vercel.json` runs `scripts/build_site.py`, which wraps the dashboard into `site/index.html` and checks every listed dataset). To publish another run, copy its dashboard data into `site/data/<id>/`, list it in `datasets.json` and push. |
| `tests/` | Grid invariants (no point lost, exact nesting, world alignment after scrolling, 16 B per cell). There is also a nuScenes loader test on a mock dataset written in nuScenes' exact file layout. The mock uses a rotated world, a rotated sensor mount, shuffled laser ids, lidarseg ids and annotation boxes. Parity tests compare the PyTorch grid engine with the NumPy one, cell by cell and across a scrolling drive, on CPU and on MPS/CUDA when present. |

## Live view

The hosted dashboard replays recorded benchmark runs: a static site can't run a GPU. The live view runs the pipeline in real time instead. `scripts/live_server.py` feeds a recorded drive in one frame per sensor tick, as a Lidar driver would, through features, network, grid, fusion and objects, and serves the dashboard, which shows each frame as it is processed. The pipeline only ever uses the current and earlier sweeps, so this is the same work it would do on a live sensor; what's missing is a sensor driver (a ROS 2 node is not built yet).

```bash
python scripts/live_server.py --dataset sim --device cpu          # simulated drive, no GPU needed
python scripts/live_server.py --dataset semantickitti --cache cache/semantickitti --scene 08 --rate 10
# then open http://localhost:8000/
```

On Colab, section 6 of the [benchmark-only notebook](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_benchmark_colab.ipynb) starts it on the same data and opens it through Colab's port proxy.

- **Rate.** `--rate` is how many frames per second are fed in; by default the recording's own (1 Hz for SemanticKITTI's every 10th scan, 2 Hz for nuScenes keyframes). A frame the pipeline can't start on time is skipped and counted as late, not queued, as with a real sensor.
- **What updates.** The map, object boxes, integrity check and the latency of every stage, with rolling p50 / p95 and FPS over the last 200 frames. The first frame is processed before the clock starts (on CUDA it compiles the derive step).
- **What doesn't.** Accuracy and the ground-truth layer need labels, so the live view leaves them out; the benchmark measures them on the same recordings. Map tiles are encoded only when the browser asks for them, so drawing never sits on the pipeline's path.

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
| `O` | Object boxes |
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
| Objects within 25 m, precision / recall: vehicle, person, pole | 99.7% / 75.8%, 87.2% / 78.9%, 96.3% / 89.7% | reported | — |
| Moving flag agrees with ground truth, matched objects | 87.0% | reported | — |
| End-to-end latency p50 / p95, laptop CPU (Apple M5 Pro, NumPy grid engine) | 68 / 71 ms | ≤ 50 ms on GPU | CPU only |
| Throughput, same runs | 14.6 FPS | ≥ 20 FPS on GPU | CPU only |
| Grid engine only vs same engine on uniform 5 cm grid | 33 ms vs 649 ms | — | 20× faster |

Stage means (ms): preprocess 13, network 22, projection 19, fusion 14, publish under 1, objects 1. Measured at `2d501d6` with `python scripts/run_benchmark.py --grid numpy --device cpu --export none`, two runs that agree within 1 ms. The PyTorch grid engine on the same CPU gives 66 / 68–70 ms; on the laptop's GPU (Apple MPS) the network drops to 7–8 ms but the grid stages slow down, for 66–67 / 76–86 ms, since this drive is too small to keep that GPU busy. An earlier run on a 2-vCPU cloud machine, before most of the speed work, measured 318 / 371 ms. Real-data latency on a T4 is in the SemanticKITTI and nuScenes results below.

**Objects.** `foveamap/objects.py` groups the map's vehicle, person and pole cells (seen this frame) and this frame's moving cells into objects, each with a class, a moving flag, an oriented box and a top height. Cells are grouped per class into 8-connected components on the 0.5 m raster, using the 5 cm cells where the fine tier has them; that bridges the gaps between scan rings without merging a person into the car beside them. The benchmark scores them against objects extracted the same way from a grid built from the ground-truth labels (a detection matches when its centre is within 1.5 m for vehicles, 0.75 m for people and poles), so the score measures how perception errors carry through to objects, not the grouping itself. Reported objects must have a minimum footprint (1 m² for vehicles, 0.05 m² for people, 0.02 m² for poles), chosen on real data (see the SemanticKITTI results); on this clean simulated drive the filter costs recall (vehicles 95.5% → 75.8%, people 81.4% → 78.9%, poles 90.8% → 89.7%). The dashboard draws the boxes (`O`).

Simulated data is easier than real Lidar, so treat these numbers as a check that the pipeline works, not as benchmark claims. The real-data numbers come from the SemanticKITTI and nuScenes notebooks below.

## Results, real Lidar (SemanticKITTI)

Measured with the SemanticKITTI notebook on a T4 GPU (stride 10, 40 epochs, seed 0). The model was fine-tuned from the simulator checkpoint on 1,922 frames from sequences 00–07, 09 and 10, and scored on every 10th scan of sequence 08, which it never saw. The full outputs are in [`results/semantickitti/`](results/semantickitti/); the 20-epoch run's files end in `_epochs20`.

| Sequence 08, all points | Simulator model | Fine-tuned, 20 epochs | Fine-tuned, 40 epochs |
| --- | --- | --- | --- |
| **mIoU** | 10.7% | 58.1% | **59.7%** |
| mIoU 0–10 m / 10–25 / 25–50 / 50–100 m | 9.9% / 10.2% / 8.8% / 0.9% | 62.4% / 53.8% / 40.1% / 18.3% | 64.2% / 55.8% / 41.6% / 17.9% |
| Road / sidewalk / parking | 16.6% / 21.7% / 2.8% | 87.9% / 69.3% / 25.7% | 88.6% / 71.2% / 27.9% |
| Terrain / vegetation / building | 21.5% / 12.1% / 17.3% | 69.5% / 76.4% / 65.5% | 70.0% / 77.4% / 67.5% |
| Pole / vehicle / person | 1.4% / 3.1% / 0.3% | 22.9% / 76.1% / 29.4% | 24.0% / 78.0% / 33.2% |
| Moving-object IoU | 0.8% | 31.7% | 29.7% |

| Check (40-epoch model, first 100 frames of sequence 08, 1 s apart) | Result | Target | Status |
| --- | --- | --- | --- |
| Map memory / saving vs uniform 5 cm | 5.12 MB / 50× | ≤ 8 MB / ≥ 30× | pass |
| Points lost at tier edges | 0 | 0 | pass |
| Drivable IoU on grid, 0–10 m | 91.9% | ≥ 90% | pass |
| Point mIoU, 0–10 m / 10–25 / 25–50 / 50–100 m | 61.4% / 52.9% / 41.8% / 12.9% | ≥ 70% near | fail |
| p50 / p95 latency, full pipeline with object extraction, on the GPU (three sessions, two runs each) | 33.7 / 43.7 and 33.2 / 43.8 ms; 35.0 / 49.5 and 35.2 / 49.0 ms; 33.2 / 73.4 and 35.2 / 54.1 ms | ≤ 50 ms p95 | pass in two sessions, **fail** in the third |
| Throughput, same runs | 28.4 / 28.3; 25.9 / 26.3; 24.8 / 25.7 FPS | ≥ 20 FPS | pass |

KITTI scans average about 123,000 points, nearly 5 times as many as nuScenes keyframes (about 26,000), so every stage has more to do. The first run kept the point transforms on the CPU in float64 and missed the target. Moving them to the GPU fixed it (both columns use the 20-epoch model):

| Stage means, T4 (ms) | Transforms on the CPU | Transforms on the GPU |
| --- | --- | --- |
| Preprocess | 22.2 | 12.5 |
| Network | 4.6 | 4.2 |
| Projection | 16.2 | 7.2 |
| Fusion + cost | 14.1 | 12.5 |
| Publish (map snapshot to host) | 2.1 | 1.7 |
| **p50 / p95 end to end** | **47 / 149** | **36 / 46** |
| Throughput (FPS) | 16.9 | 26.2 |

Before objects were added the p95 margin was thin: 46 ms in both runs with GPU transforms, against 50 ms. Colab's shared vCPUs have busy spells that slow every stage at once for a few seconds: in two runs with the CPU transforms, the slow stretches fell in different places (frames 26–36 and 90–98, then 50–58) and those frames had normal point counts, so they come from the machine, not the data. The 40-epoch run is `seq08_metrics.json` / `benchmark_seq08.log`, the 20-epoch run with GPU transforms `*_epochs20*`, and the earlier one `*_cpu_transforms*`.

Adding object extraction and the pothole plane fit first pushed p95 to 57 ms (two runs at `569268b`, `seq08_objects_run*`). A `torch.profiler` run on the T4 (`scripts/profile_pipeline.py`, `profile_seq08_eager.txt`) showed why: the pipeline is CPU-bound, with about 17 ms of GPU kernels in a 33 ms map frame and the rest spent launching about 1,000 small operations, and the costliest single operation was a float64 matrix multiply in the sweep transforms (4.7 ms). Three changes brought it back under the target:

| Typical frame (50 fastest of 98), T4, ms | Before (`569268b`) | After (`8d45de1`) |
| --- | --- | --- |
| Preprocess | 11.6 | 9.6 |
| Network | 4.2 | 4.2 |
| Projection | 6.8 | 5.5 |
| Fusion + cost | 11.9 | 8.6 |
| Publish | 1.6 | 1.6 |
| Objects (host CPU) | 4.4 | 3.3 |
| **Total** | **40.5** | **32.8** |
| **p50 / p95, all frames** | **41.6 / 57.0** | **33.7 / 43.7** |

- **Sweep transforms written per coordinate** instead of as float64 (N, 3) @ (3, 3) matmuls, which a T4 runs slowly (same result to 6e-14 m).
- **The derive step (flags and cost) compiled with `torch.compile` on CUDA.** It is about 160 element-wise ops per tier with fixed shapes; compiled, it matches the eager version exactly. The first frame of a run includes compiling it (up to half a minute) and is not counted.
- **Object extraction without full-tier temporaries,** and the pothole plane's constants built once per device.

Blocks of 10–15 consecutive slower frames still appear at different places in each run (the shared vCPUs; unmounting Google Drive did not remove them). In the first session they peaked at 41–48 ms; a second session with the same code (`seq08_session2_run*`) had more of them and one 161 ms frame, its typical frame was 1 ms slower (33.9 ms), and p95 came to 49.0–49.5 ms. The margin under 50 ms therefore depends on the Colab machine. The first session's runs are `seq08_compiled_run*` with `profile_seq08_compiled.txt`.

A third session (`seq08_session3_run*`, `profile_seq08_session3.txt`) followed two more fixes: fusion finds its observed cells once instead of indexing with boolean masks, which cuts the host's waits for the GPU (stream synchronisations from 52 to 25 per frame, host-to-device copies from 60 to 35), and the previous sweeps are uploaded as float32. The typical frame got faster, 31.9–32.6 ms, the best so far, but that machine had the most stalls yet (frames of 77–118 ms scattered through the first run), and p95 came to 73.4 and 54.1 ms. The median is steady at 33–35 ms across all three sessions; the 95th percentile is set by Colab's shared machine, and on a busy one it misses 50 ms. The benchmark notebook now logs CPU steal time and the GPU's clock and power every 0.5 s next to each frame, to tell which it is.

| Objects within 25 m (40-epoch model, first 100 frames of sequence 08) | Precision | Recall |
| --- | --- | --- |
| Vehicle (at least 1 m²) | 64.7% | 69.8% |
| Person (at least 0.05 m²) | 12.8% | 28.1% |
| Pole / sign (at least 0.02 m²) | 23.9% | 61.5% |

Moving flags agree with ground truth on 95% of matched objects. Without a size filter, precision was 34%, 9% and 16%: most false objects on real scans are a handful of misclassified cells, not near any real object (only about 12% are fragments of a real one). The minimum footprints come from the benchmark's sweep on this same sequence (per class, the smallest area whose F1 is within 1 point of the best), so they are tuned on the data they are scored on; it is one number per class. People stay poor at any size, matching the network's 33% person IoU: object quality is limited by perception, not by grouping.

- **The simulator alone doesn't carry over to real Lidar.** It scores 10.7% on SemanticKITTI, about the same as on nuScenes (9.8%), even though KITTI's 64-beam HDL-64E is the sensor it simulates. The gap comes from simulated versus real scenes, not the beam count.
- **Fine-tuning brings it to 59.7% mIoU**, with road at 88.6% and vehicles at 78.0%, against 46.3% after fine-tuning on nuScenes-mini's 8 scenes.
- **Longer training helps a little.** Going from 20 to 40 epochs raised mIoU by 1.6 points and every class's IoU, while moving-object IoU fell by 2 points and the 50–100 m band by 0.4. Training loss was still falling slowly at the end, but near-range mIoU (64.2%) is still short of 70%: more data (a smaller stride) is the likelier fix than more epochs.
- **Weak classes.** Most poles and people are found (recall 70% and 63%), but too many other points are labelled as them, so their IoU stays low (24.0% and 33.2%). Parking (27.9%) is flat ground that looks like road or sidewalk.
- **Reproducible.** Two 20-epoch training runs with seed 0 gave identical scores, down to the per-class IoUs. (Scoring the same model in different Colab sessions can differ by a few dozen of about 52 million points, from GPU rounding.)

## Results, real Lidar (nuScenes)

Measured on a T4 GPU with the benchmark-only notebook at `8e654bd` (current code, with object extraction), using a model fine-tuned from the simulator checkpoint on the 8 nuScenes-mini training scenes for 120 epochs. The first fine-tune with this recipe scored 46.3% mIoU on the 2 validation scenes. The full outputs are in [`results/nuscenes/`](results/nuscenes/); the latest runs are `scene-0103_compiled_run*` with `profile_scene-0103_compiled.txt`.

| Check (scene-0103, 40 keyframes) | Result | Target | Status |
| --- | --- | --- | --- |
| Map memory / saving vs uniform 5 cm | 5.12 MB / 50× | ≤ 8 MB / ≥ 30× | pass |
| Points lost at tier edges | 0 | 0 | pass |
| Drivable IoU on grid, 0–10 m | 94.1% | ≥ 90% | pass |
| Point mIoU, 0–10 m / 10–25 / 25–50 / 50–100 m | 44.1% / 43.7% / 30.9% / 15.2% | ≥ 70% near | fail |
| Moving-object IoU | 37.8% | reported | — |
| Pothole flags on drivable cells within 10 m (no potholes in this scene) | 0.039% | — | — |
| Objects within 25 m, precision / recall: vehicle, person | 65.1% / 65.4%, 39.5% / 18.3% | reported | — |
| p50 / p95 latency, full pipeline with objects, on the GPU (run 1) | 29.6 / 43.3 ms | ≤ 50 ms p95 | pass |
| Throughput, same run | 30.7 FPS | ≥ 20 FPS | pass |

Run 2 measured p50 / p95 38.1 / 98.8 ms: frames 7–10 ran 2–4 times slower in every stage at once, the network included, a pause of the machine about 2 s long, and with only 38 timed frames those few frames set p95. Outside it both runs take about 28–31 ms a frame. nuScenes has no pole or sign class, so that slot holds barriers and cones; none of the 16 in range were found as objects (64 false ones), and the moving flag agrees on 75% of matched objects.

The rest of this section is the earlier session, before objects and the pothole plane fit, which compared the engines.

In that session, [`results/nuscenes/`](results/nuscenes/) has the GPU run's metrics and per-frame log (`scene-0103_metrics.json`, `benchmark_scene-0103.log`), next to the NumPy and concurrent-export runs from the same session (`*_numpy*`, `*_torch_async*`). The NumPy engine gives the same accuracy apart from the 50–100 m band (15.5% point mIoU), where rare tie pixels in the GPU range image change a few points.

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
- **Speed: 43 ms p95 at 31 FPS on a T4 with objects** (31 ms p95 at 33 FPS before objects were added). This is with the features and grid engine on the GPU, 4.5× faster at p50 than NumPy. It excludes the dashboard's PNG encoding (about 140 ms per frame). Encoding concurrently on Colab's 2 vCPUs pushes p95 to 87 ms.
- **Potholes.** The first pothole test (5 cm below the mean of drivable ground within 2.5 m) flagged 11% of drivable 5 cm cells and 20% of 50 cm cells in this replay, where there are no potholes. The flags followed whole laser rings: each of the 32 lasers sits a few cm higher or lower than its neighbours, and the test read the low rings as holes. A cell now counts as a pothole when it is below a least-squares plane of the drivable ground within 5 m by at least 5 cm and three times the ground's scatter about that plane. The plane absorbs grade, camber and pitch error; the scatter term absorbs the ring offsets. Re-scored offline on this replay's stored cell heights (1.8 cm steps), the false flags fall to 0.09% (5 cm cells) and 0.32% (50 cm cells), while the simulated drive still finds every pothole. With the current code, 0.039% of drivable cells within 10 m are flagged in this scene and 0.057% on SemanticKITTI sequence 08; neither has potholes. Potholes are now flagged on the 5 cm tier only: a 0.25–0.6 m pothole is about one 50 cm cell, too few to tell from noise, and skipping the coarse tier keeps the plane fit cheap. The trade-off is that a pothole has to stand out from the local ring noise: with ±5 cm ring offsets a 10 cm pothole is barely flagged. The benchmark reports `pothole_flag_rate_drivable_10m`, which on a road without potholes is the false-alarm rate.

## Why a range-image network

The problem statement names PointNet++ and sparse CNNs as examples ("e.g.") of a perception backbone. FoveaMap uses a range-image U-Net instead, the "low-power fallback" of Architecture section 4, for these reasons:

- **It leaves room for the map in the latency budget.** On a T4 the network takes 4.2 ms of a typical 33 ms SemanticKITTI frame (FP16, 64 × 1024 input). Features, binning, fusion and object extraction take the rest, and the whole pipeline meets p95 ≤ 50 ms (43.7 ms on SemanticKITTI, 43.3 ms on nuScenes).
- **It is small and needs nothing beyond stock PyTorch.** 327,434 parameters, a 1.3 MB checkpoint, plain 2-D convolutions: the same code runs on CUDA, Apple MPS and the CPU. Sparse 3-D convolutions need compiled libraries (spconv, TorchSparse or MinkowskiEngine) built for a particular CUDA version, and PointNet++ needs custom neighbour-search ops.
- **It matches the sensor.** A spinning Lidar produces a range image natively: rows are lasers, columns are azimuth. The input is a dense image with no voxelisation step, and the motion cue (range residuals against the two previous sweeps) is just two more image channels.
- **Accuracy is in the same range as the alternatives.** On the SemanticKITTI single-scan test set (19 classes), published results are about 20% mIoU for PointNet++, 52% for RangeNet++ and 60% for SalsaNext (both range-image networks), and roughly 60–68% for sparse-voxel networks such as SPVNAS and Cylinder3D, depending on the paper and version. Range-image networks give up a few points against sparse convolutions for speed; PointNet++ is far behind on this data. Our 59.7% is on 9 merged classes, trained on every 10th scan, so it is not directly comparable to these numbers.
- **It can be replaced.** The grid engine, fusion, object extraction and training loop only need per-point class probabilities and a moving probability, so a sparse-conv backbone could take its place without changing them.

The limits are known, and they explain the weakest results:

- **One point per pixel.** 64 × 1024 is 65,536 pixels for about 123,000 points in a KITTI scan, so at most about half the points are seen directly; the rest take their pixel's prediction, which blurs object boundaries. RangeNet++ adds a k-nearest-neighbour clean-up step for this; FoveaMap does not yet.
- **Azimuth resolution.** A column is 0.35°, about 30 cm at 50 m, so a person is one or two columns wide there. That is one reason the 50–100 m band and the small classes (poles 24%, people 33% IoU) are the weakest.
- The natural next steps are a k-NN or point-based refinement head on the range-image output, or a sparse-voxel backbone where a faster GPU leaves room for it.

## How the prototype differs from the full design

- **Backbone.** The prototype uses the range-image network (the "low-power fallback" in Architecture section 4) instead of a sparse-conv U-Net. The training loop and the grid engine don't depend on which backbone you use. See [Why a range-image network](#why-a-range-image-network).
- **Grid engine.** The grid engine runs in NumPy or PyTorch (`--grid torch`). The PyTorch engine is parity-tested against NumPy and runs on the GPU. With `--grid torch`, the range-image features are also built on the GPU (`foveamap/features_torch.py`, also parity-tested; override with `--features numpy`). Dashboard PNG export is encoded after the timed loop by default and reported as `export_ms`, not counted in latency (`--export async` encodes on a background thread during the run instead, and `none` skips it). On CUDA the derive step (flags and traversability cost) runs through `torch.compile`; set `FOVEAMAP_COMPILE=0` to run it eagerly.
- **Not built yet:** 3D view, object tracking across frames (so no object IDs or velocity arrows), free-space ray clearing, ROS 2 node, and TensorRT export.
