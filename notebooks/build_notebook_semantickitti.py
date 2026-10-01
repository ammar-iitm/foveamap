"""Builds foveamap_semantickitti_colab.ipynb (kept as code so the notebook stays reviewable)."""
import json
import os

cells = []


def md(src):
    cells.append(dict(cell_type="markdown", metadata={}, source=src.strip("\n").splitlines(True)))


def code(src):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=src.strip("\n").splitlines(True)))


md("""
<a href="https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_semantickitti_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

md("""
# FoveaMap on SemanticKITTI

nuScenes-mini has only 8 training scenes, far too few to learn real-world classes. SemanticKITTI has 19,130 labelled training scans from a 64-beam HDL-64E, the sensor the simulator models, and its classes map onto all 9 FoveaMap classes (including poles, parking and moving cars and people).

This notebook:
1. Fetches every 10th scan of the training sequences (00–07, 09, 10) and of the validation sequence 08, plus the two scans before each (the motion cue). It pulls only those scans out of the 85 GB KITTI zip, about 14 GB in total, and builds the frame cache from them. The cache is saved to your Google Drive, so later sessions copy it back instead of downloading and rebuilding.
2. Measures the **simulator-trained** model on sequence 08 before fine-tuning.
3. **Fine-tunes** it on the training sequences.
4. Scores sequence 08 again (mIoU by distance band and per class), and runs the pipeline benchmark on it.

**Before you start:** use *Runtime → Change runtime type → T4 GPU*. The first run takes roughly 40–60 minutes, about half of it data preparation; later runs copy the prepared data from Drive in about 5 minutes.

**Terms:** KITTI is licensed [CC BY-NC-SA 3.0](http://www.cvlibs.net/datasets/kitti/) and SemanticKITTI [CC BY-NC-SA 4.0](http://www.semantic-kitti.org/) (non-commercial). Register at [cvlibs.net](http://www.cvlibs.net/datasets/kitti/user_register.php) and accept the KITTI terms before downloading.
""")

md("## 1. Setup")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: Runtime > Change runtime type > T4 GPU"
""")
md("""
The finished frame cache is kept in your **Google Drive** (`MyDrive/foveamap_data/`), so it is built once and copied back by later sessions. Colab asks for permission to access your Drive. Set `USE_DRIVE = False` to build everything in this runtime only.

The raw KITTI files go to this runtime's local disk, which is much faster than Drive for thousands of small files, and are not kept.

This cell also clones the latest `main` from [GitHub](https://github.com/ammar-iitm/foveamap), replacing any older copy of the code.
""")
code("""
import os, sys, shutil, importlib
REPO = 'https://github.com/ammar-iitm/foveamap.git'
USE_DRIVE = True
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
KITTI = '/content/kitti'                    # raw scans, labels, poses, calibration (local disk, not kept)
CACHE = '/content/cache/semantickitti'      # frames built from them

%cd /content
shutil.rmtree('/content/foveamap', ignore_errors=True)
!git clone -q --depth 1 $REPO /content/foveamap
if not os.path.exists('/content/foveamap/foveamap/semantickitti.py'):
    raise FileNotFoundError('This copy of the code has no SemanticKITTI loader (foveamap/semantickitti.py).')
for m in [m for m in list(sys.modules) if m == 'foveamap' or m.startswith('foveamap.')]:
    del sys.modules[m]
importlib.invalidate_caches()
%cd /content/foveamap
!pip -q install -r requirements.txt
!git log --oneline -1
""")

md("""
## 2. Get the frame cache

**If the cache for this `STRIDE` is already in Drive** (from an earlier session), it is copied into this runtime, about 5 minutes.

**Otherwise** (first time, about 20–30 minutes): downloads the SemanticKITTI labels (179 MB, includes poses) and the KITTI calibration, fetches only the scans that are used from the KITTI velodyne zip (progress every 30 s), builds the frame cache, then saves it to Drive (about 14 GB) for next time.
""")
code("""
STRIDE = 10     # every 10th scan: ~1,900 training and ~400 validation frames
DRIVE_CACHE = f'/content/drive/MyDrive/foveamap_data/semantickitti_cache_stride{STRIDE}'
if USE_DRIVE:
    drive.mount('/content/drive')
if USE_DRIVE and os.path.exists(f'{DRIVE_CACHE}/index.json'):          # index.json is saved last: its presence means complete
    print('Frame cache found in Drive; copying it into this runtime')
    !mkdir -p $CACHE && cp "$DRIVE_CACHE"/*.pkl "$DRIVE_CACHE"/index.json $CACHE/
else:
    !python scripts/prepare_semantickitti.py --root $KITTI --out $CACHE --stride $STRIDE
    if not os.path.exists(f'{CACHE}/index.json'):                       # written last by the cache build
        raise RuntimeError('Preparing SemanticKITTI failed (see above). Run this cell again: '
                           'scans already fetched are kept, so it resumes where it stopped.')
    if USE_DRIVE:
        print('Saving the frame cache to Drive for later sessions')
        !mkdir -p "$DRIVE_CACHE" && cp $CACHE/*.pkl "$DRIVE_CACHE"/ && cp $CACHE/index.json "$DRIVE_CACHE"/
        drive.flush_and_unmount()                                      # wait until everything has reached Drive
        print(f'Saved to {DRIVE_CACHE}')
!du -sh $CACHE
""")

md("""
## 3. Simulator model on sequence 08, before fine-tuning
""")
code("""
import json
sys.path.insert(0, 'scripts')
from train import evaluate
from foveamap.semantickitti import iter_cached, cached_info
from foveamap.model import load_model

info = cached_info(CACHE)
zero_shot = evaluate(load_model('checkpoints/range_unet.pt'), iter_cached(CACHE, 'val'), info)
json.dump(zero_shot, open('checkpoints/range_unet_sim_on_semantickitti_val.json', 'w'), indent=1)
print(f"mIoU {100 * zero_shot['miou']:.1f}%  moving IoU {zero_shot['moving_iou']}")
for n, v in zero_shot['iou_by_class'].items():
    print(f'  {n:12s} IoU {v}')
""")

md("""
## 4. Fine-tune on the training sequences (about 15 minutes on a T4)

Starts from the simulator checkpoint. Progress prints every 250 steps. `FLAGS` takes the recipe options from `scripts/train.py` (`--reset-head`, `--balance`, `--aug`).
""")
code("""
EPOCHS = 20
FLAGS = ''
!python scripts/train.py --dataset semantickitti --cache $CACHE --init checkpoints/range_unet.pt \\
    --out checkpoints/range_unet_semantickitti.pt --epochs $EPOCHS $FLAGS 2>&1 | tee train_semantickitti.log | awk '!/^step/ || (++n % 10 == 0)'
""")

md("""
## 5. Results on sequence 08, and the pipeline benchmark

The benchmark runs the first 100 frames of sequence 08 (every 10th scan, so 1 s apart) with the features and grid engine on the GPU.
""")
code("""
!python scripts/run_benchmark.py --dataset semantickitti --cache $CACHE --scene 08 --max-frames 100 \\
    --grid torch --out results/semantickitti > benchmark_semantickitti.log
!tail -n 3 benchmark_semantickitti.log
""")
code("""
import json, pandas as pd                 # reads files only, so it also works after a runtime restart
zero_shot = json.load(open('checkpoints/range_unet_sim_on_semantickitti_val.json'))
ft = json.load(open('checkpoints/range_unet_semantickitti_val.json'))
S = json.load(open('results/semantickitti/metrics.json'))['summary']
pct = lambda v: '—' if v is None else f'{100 * v:.1f}%'
rows = [('mIoU, all points', pct(zero_shot['miou']), pct(ft['miou']))]
rows += [(f'mIoU {b}', pct(z), pct(f)) for b, z, f in zip(['0-10 m', '10-25 m', '25-50 m', '50-100 m'],
                                                          zero_shot['miou_by_band'], ft['miou_by_band'])]
rows += [(f'IoU {c}', pct(zero_shot['iou_by_class'][c]), pct(v)) for c, v in ft['iou_by_class'].items()]
rows += [('Moving-object IoU', pct(zero_shot['moving_iou']), pct(ft['moving_iou']))]
display(pd.DataFrame(rows, columns=['Sequence 08', 'Simulator model', 'Fine-tuned']))
print(f"Pipeline on the GPU: p50 / p95 {S['latency_ms']['p50']:.0f} / {S['latency_ms']['p95']:.0f} ms, "
      f"{S['fps']:.1f} FPS, points lost {S['points_lost']}, drivable IoU 0-10 m {pct(S['drivable_iou_grid_0_10'])}")
""")

md("""
## 6. Download the results
""")
code("""
!zip -qr /content/foveamap_semantickitti_results.zip checkpoints/range_unet_semantickitti_val.json checkpoints/range_unet_sim_on_semantickitti_val.json train_semantickitti.log benchmark_semantickitti.log results/semantickitti/metrics.json
from google.colab import files
files.download('/content/foveamap_semantickitti_results.zip')
""")

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=0)
out = os.path.join(os.path.dirname(__file__), "foveamap_semantickitti_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out)
