# Results in full

Every measurement behind the summary in the [README](../README.md#at-a-glance): the simulated drive, SemanticKITTI and nuScenes, including the runs that missed a target and what was changed. The raw metrics, per-frame logs and profiles are in [`results/`](../results/).

## Results, real Lidar (SemanticKITTI)

Measured with the SemanticKITTI notebook on a T4 GPU (stride 10, 40 epochs, seed 0). The model was fine-tuned from the simulator checkpoint on 1,922 frames from sequences 00–07, 09 and 10, and scored on every 10th scan of sequence 08, which it never saw. The full outputs are in [`results/semantickitti/`](../results/semantickitti/); the 20-epoch run's files end in `_epochs20`.

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

A fourth session (`seq08_session4_run*`, `monitor_seq08_session4_run*.csv`, `profile_seq08_session4.txt`) benchmarked the published model, twice as wide and trained on every 5th scan, with the benchmark notebook's machine monitor running: p50 / p95 33.8 / 46.0 ms and 33.6 / 48.0 ms, 27.5 and 27.1 FPS, the network at 7.4 ms. Both runs met the target. The monitor (CPU steal and busy time from `/proc/stat`, GPU clock, power and utilisation from `nvidia-smi` every 0.5 s) explains the tail:

- **No CPU was stolen by the host**: steal was 0.0% in every sample of both runs.
- **The GPU was not the cause**: its SM clock stayed at 585 MHz throughout and it was busy about 9% of the time, the same for slow and fast frames.
- **The slow frames are CPU-bound**: frames over 45 ms (7 and 11 of 98) came while the VM's two vCPUs were 85–89% busy, against 56% for the 50 fastest frames, which is about what the pipeline itself uses. Their extra time is spread over every stage that launches GPU work (preprocessing, projection and fusion each 3–15 ms over their medians), not one stage. The profiler agrees: 2.1 s of CPU time against 0.38 s of GPU time.

So the pipeline is limited by launching GPU work from one CPU thread, and on two shared vCPUs any other load on the machine delays it. More cores, or fewer and larger kernels per frame, would shrink the tail; the GPU has room to spare.

| Objects within 25 m (40-epoch model, first 100 frames of sequence 08) | Precision | Recall |
| --- | --- | --- |
| Vehicle (at least 1 m²) | 64.7% | 69.8% |
| Person (at least 0.05 m²) | 12.8% | 28.1% |
| Pole / sign (at least 0.02 m²) | 23.9% | 61.5% |

Moving flags agree with ground truth on 95% of matched objects. Without a size filter, precision was 34%, 9% and 16%: most false objects on real scans are a handful of misclassified cells, not near any real object (only about 12% are fragments of a real one). The minimum footprints come from the benchmark's sweep on this same sequence (per class, the smallest area whose F1 is within 1 point of the best), so they are tuned on the data they are scored on; it is one number per class. People stay poor at any size, matching the network's 33% person IoU: object quality is limited by perception, not by grouping.

- **The simulator alone doesn't carry over to real Lidar.** It scores 10.7% on SemanticKITTI, about the same as on nuScenes (9.8%), even though KITTI's 64-beam HDL-64E is the sensor it simulates. The gap comes from simulated versus real scenes, not the beam count.
- **Fine-tuning brings it to 59.7% mIoU**, with road at 88.6% and vehicles at 78.0%, against 46.3% after fine-tuning on nuScenes-mini's 8 scenes.
- **Longer training helps a little.** Going from 20 to 40 epochs raised mIoU by 1.6 points and every class's IoU, while moving-object IoU fell by 2 points and the 50–100 m band by 0.4. Training loss was still falling slowly at the end, but near-range mIoU (64.2%) is still short of 70%: more data (a smaller stride) is the likelier fix than more epochs.

### Twice the training data (stride 5)

Fine-tuning on every 5th scan instead of every 10th (3,834 frames, 40 epochs, 32 minutes on a T4) and scoring every 5th scan of sequence 08 (so not the identical frames as the stride-10 column, but the same held-out drive):

| Sequence 08 | Stride 10, 40 epochs | Stride 5, 40 epochs |
| --- | --- | --- |
| **mIoU** | 59.7% | **60.3%** |
| mIoU 0–10 m / 10–25 / 25–50 / 50–100 m | 64.2% / 55.8% / 41.6% / 17.9% | 64.8% / 56.4% / 42.1% / 19.4% |
| Road / sidewalk / parking | 88.6% / 71.2% / 27.9% | 88.7% / 72.3% / 29.1% |
| Terrain / vegetation / building | 70.0% / 77.4% / 67.5% | 66.0% / 79.5% / 67.2% |
| Pole / vehicle / person | 24.0% / 78.0% / 33.2% | 25.6% / 80.6% / 34.0% |
| Moving-object IoU | 29.7% | **35.9%** |

Twice the data bought 0.6 points of mIoU, near range included, and 6 points of moving-object IoU; terrain lost 4. So data is not what holds near-range accuracy at 64–65%. The likelier limit is the network: 327k parameters, against about 6.7M for SalsaNext and 50M for RangeNet++, and it uses only 4–5 ms of a 30 ms frame, so it can grow. The benchmark of this model (first 100 frames of every 5th scan of sequence 08, so 0.5 s apart) gave p50 / p95 30.6 / 61.3 ms, 27.8 FPS, drivable IoU 93.3%, pothole flag rate 0.056%, objects within 25 m vehicle 66.3% / 70.7%, person 12.7% / 26.6%, pole 23.7% / 57.8% precision / recall, and moving flags agreeing on 96.2% of matched objects. Files: `*stride5*` (without `width2`).

### A network twice as wide (the published model)

The same stride-5 data and 40 epochs, with every channel width doubled (`train.py --width 2`: 1.3M parameters instead of 327k). A wider network cannot start from the simulator checkpoint, so it trained from scratch (46 minutes on a T4). Scored on the same frames as the stride-5 column above:

| Sequence 08 | Standard width (327k) | **Twice the width (1.3M)** |
| --- | --- | --- |
| **mIoU** | 60.3% | **62.9%** |
| mIoU 0–10 m / 10–25 / 25–50 / 50–100 m | 64.8% / 56.4% / 42.1% / 19.4% | **67.0%** / 60.4% / 45.5% / 17.7% |
| Road / sidewalk / parking | 88.7% / 72.3% / 29.1% | 89.7% / 74.4% / 29.8% |
| Terrain / vegetation / building | 66.0% / 79.5% / 67.2% | 66.6% / 81.0% / 71.0% |
| Pole / vehicle / person | 25.6% / 80.6% / 34.0% | 30.1% / 82.3% / 41.5% |
| Moving-object IoU | 35.9% | **44.4%** |

Every class gained; only the 50–100 m band fell (1.7 points). Where twice the data bought 0.6 points, twice the width bought 2.6 overall and 2.2 at 0–10 m, so the network's size was the larger limit. Its training loss ended at about 0.06 against 0.10 for the standard network, a much larger gap than on the held-out drive: it is starting to fit its training drives, so augmentation (`--aug`) is the next thing to try. Its benchmark (same 100 frames) gave p50 / p95 33.3 / 45.6 ms, 27.7 FPS, with the network at 7.3 ms (4.8 ms at standard width); drivable IoU 94.9%, pothole flag rate 0.059%, objects within 25 m vehicle 69.1% / 71.8%, person 17.6% / 28.1%, pole 33.5% / 63.0% precision / recall, and moving flags agreeing on 95.1% of matched objects. Files: `*width2*`.

### Twice the width with augmentation, at stride 10

A run meant to add augmentation (`--aug`: each training sweep's range and x, y, z scaled by a random ±5%, intensity by ±20%) to the published recipe ran with the notebook's default stride of 10 instead of 5, so it trained on 1,922 frames and was scored on every 10th scan of sequence 08. It is not a clean test of augmentation, since the data halved at the same time, but it is informative:

| Sequence 08 | Standard width, stride 10 | Twice the width, stride 5 (published) | Twice the width + augmentation, stride 10 |
| --- | --- | --- | --- |
| **mIoU** | 59.7% | 62.9% | 62.8% |
| mIoU 0–10 m / 10–25 / 25–50 / 50–100 m | 64.2% / 55.8% / 41.6% / 17.9% | 67.0% / 60.4% / 45.5% / 17.7% | **67.2%** / 59.2% / 45.1% / **20.3%** |
| Road / sidewalk / parking | 88.6% / 71.2% / 27.9% | 89.7% / 74.4% / 29.8% | 90.9% / 74.1% / 30.3% |
| Terrain / vegetation / building | 70.0% / 77.4% / 67.5% | 66.6% / 81.0% / 71.0% | 68.4% / 78.8% / 71.7% |
| Pole / vehicle / person | 24.0% / 78.0% / 33.2% | 30.1% / 82.3% / 41.5% | 29.2% / 81.0% / 40.5% |
| Moving-object IoU | 29.7% | **44.4%** | 39.8% |

The columns are scored on different frames of the same held-out drive (every 10th, every 5th and every 10th scan), so differences of a point or less are within that. With half the data, augmentation brought the wide network to the published model's accuracy near range (67.2% against 67.0%) and beyond it at 50–100 m, but not for moving objects. Its training loss ended at 0.083 against 0.061 without augmentation, so it fits its training drives less closely, as intended. The run that answers whether augmentation adds to twice the data is the same recipe at stride 5. Its benchmark (first 100 frames of every 10th scan) gave p50 / p95 33.4 / 46.8 ms, 27.4 FPS, network 7.4 ms, drivable IoU 93.1%, pothole flag rate 0.061%, objects within 25 m vehicle 68.3% / 71.8%, person 15.5% / 26.6%, pole 30.8% / 64.1% precision / recall. Files: `*width2_aug*`.

### Peak GPU memory (NFR-5)

Measured in the same benchmark, with the twice-as-wide model loaded before the counters were reset: PyTorch's allocator held at most 138 MB (104 MB of it in tensors), and the whole GPU had 451 MB in use, including the CUDA context, of the T4's 14.9 GB. The PRD's target is at most 4 GB, so the pipeline uses about a ninth of it. The map's own layers are 5.12 MB of that.
- **Weak classes.** Most poles and people are found (recall 70% and 63%), but too many other points are labelled as them, so their IoU stays low (24.0% and 33.2%). Parking (27.9%) is flat ground that looks like road or sidewalk.
- **Reproducible.** Two 20-epoch training runs with seed 0 gave identical scores, down to the per-class IoUs. (Scoring the same model in different Colab sessions can differ by a few dozen of about 52 million points, from GPU rounding.)

## Results, real Lidar (nuScenes)

Measured on a T4 GPU with the benchmark-only notebook at `8e654bd` (current code, with object extraction), using a model fine-tuned from the simulator checkpoint on the 8 nuScenes-mini training scenes for 120 epochs. The first fine-tune with this recipe scored 46.3% mIoU on the 2 validation scenes. The full outputs are in [`results/nuscenes/`](../results/nuscenes/); the latest runs are `scene-0103_compiled_run*` with `profile_scene-0103_compiled.txt`.

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

In that session, [`results/nuscenes/`](../results/nuscenes/) has the GPU run's metrics and per-frame log (`scene-0103_metrics.json`, `benchmark_scene-0103.log`), next to the NumPy and concurrent-export runs from the same session (`*_numpy*`, `*_torch_async*`). The NumPy engine gives the same accuracy apart from the 50–100 m band (15.5% point mIoU), where rare tie pixels in the GPU range image change a few points.

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
