# Real data: SemanticKITTI and nuScenes

How to run FoveaMap on real Lidar without Lidar hardware, through the Colab notebooks on a free T4 GPU, or locally if you have the data, and how each loader maps its dataset onto FoveaMap's frames and classes. Back to the [README](../README.md).

## SemanticKITTI on Colab

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
- **Stride 5 fits a standard Colab runtime.** Every 5th scan doubles the training data (about 3,800 frames, 26 GB of scans, 28 GB of cache). The cache is written and read 100 frames at a time, and `train.py --arrays DIR` keeps the training arrays in memory-mapped files instead of RAM (the notebook does this below stride 10); training from them gives exactly the same weights.
- **Laser rows.** The files carry no laser id, so rows come from elevation over the simulator's field of view (+2° to −24.9°, 64 rows).
- **Frames and poses.** The ego frame is the Lidar frame moved down 1.73 m to the ground. Poses are cam0 poses converted to the Lidar with the calibration, so the world is the first scan's ego frame.
- **Classes and moving flags.** SemanticKITTI's classes map onto all 9 FoveaMap classes, and its moving-car and moving-person labels give the moving flags.

## nuScenes-mini on Colab (no Lidar hardware needed)

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
