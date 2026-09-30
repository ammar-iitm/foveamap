"""Builds foveamap_nuscenes_colab.ipynb (kept as code so the notebook stays reviewable)."""
import json
import os

cells = []


def md(src):
    cells.append(dict(cell_type="markdown", metadata={}, source=src.strip("\n").splitlines(True)))


def code(src):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=src.strip("\n").splitlines(True)))


md("""
# FoveaMap on real Lidar: nuScenes-mini

This notebook runs the FoveaMap prototype on **real Lidar data** from nuScenes-mini (10 scenes, 32-beam Velodyne, about 4 GB). It works through these steps:

1. Downloads nuScenes-mini and the nuScenes-lidarseg labels.
2. Checks the loader on one frame and plots it.
3. Measures how the **simulator-trained** model does on real data, before any fine-tuning.
4. **Fine-tunes** the model on the 8 `mini_train` scenes, using the GPU.
5. Runs the full benchmark on a `mini_val` scene with both grid engines (NumPy on the CPU, PyTorch on the GPU): latency, memory, and accuracy by distance.
6. Exports the dashboard data and zips everything for download.

**Before you start:** use *Runtime → Change runtime type → T4 GPU*. The whole notebook takes roughly 15–25 minutes.

nuScenes is free for non-commercial use under its [terms of use](https://www.nuscenes.org/terms-of-use).
""")

md("## 1. Setup")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: Runtime > Change runtime type > T4 GPU"
""")
md("""
Get the code: this cell clones the latest `main` from [GitHub](https://github.com/ammar-iitm/foveamap), replacing any older copy in this runtime. To run a local copy instead, set `USE_ZIP = True` and upload the code zip when asked (or set `ZIP_IN_DRIVE` to its path in Google Drive).
""")
code("""
import os, sys, shutil, zipfile, importlib
REPO = 'https://github.com/ammar-iitm/foveamap.git'
USE_ZIP = False       # True: upload a code zip instead of cloning
ZIP_IN_DRIVE = None   # e.g. '/content/drive/MyDrive/foveamap.zip'

%cd /content
shutil.rmtree('/content/foveamap', ignore_errors=True)            # always start from a fresh copy
if not USE_ZIP:
    !git clone -q --depth 1 $REPO /content/foveamap
elif ZIP_IN_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    zipfile.ZipFile(ZIP_IN_DRIVE).extractall('/content')
else:
    from google.colab import files
    up = files.upload()
    zipfile.ZipFile(next(iter(up))).extractall('/content')
if not os.path.exists('/content/foveamap/foveamap/grid_torch.py'):
    raise FileNotFoundError('This copy of the code has no PyTorch grid engine (foveamap/grid_torch.py).')
for m in [m for m in list(sys.modules) if m == 'foveamap' or m.startswith('foveamap.')]:
    del sys.modules[m]                                            # forget any old import
importlib.invalidate_caches()
%cd /content/foveamap
!pip -q install -r requirements.txt
!git log --oneline -1 2>/dev/null; ls foveamap
""")

md("""
## 2. Download nuScenes-mini + lidarseg

If a download fails (for example, the site asks you to log in), register for free at [nuscenes.org](https://www.nuscenes.org/nuscenes#download). Then download **Mini** (`v1.0-mini.tgz`) and **nuScenes-lidarseg → Mini** (`nuScenes-lidarseg-mini-v1.0.tar.bz2`), put both files in `/content/`, and rerun this cell.
""")
code("""
DATAROOT = '/content/nuscenes'
os.makedirs(DATAROOT, exist_ok=True)

if not os.path.exists('/content/v1.0-mini.tgz'):
    !wget -q --show-progress -O /content/v1.0-mini.tgz https://www.nuscenes.org/data/v1.0-mini.tgz
if not os.path.exists('/content/nuScenes-lidarseg-mini-v1.0.tar.bz2'):
    !wget -q --show-progress -O /content/nuScenes-lidarseg-mini-v1.0.tar.bz2 https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2

!tar -xzf /content/v1.0-mini.tgz -C $DATAROOT
!tar -xjf /content/nuScenes-lidarseg-mini-v1.0.tar.bz2 -C $DATAROOT

need = ['v1.0-mini/sample_data.json', 'v1.0-mini/lidarseg.json', 'samples/LIDAR_TOP', 'sweeps/LIDAR_TOP', 'lidarseg/v1.0-mini']
missing = [p for p in need if not os.path.exists(os.path.join(DATAROOT, p))]
print('missing:', missing if missing else 'nothing, data is ready')
""")

md("## 3. Check the loader on one frame")
code("""
import numpy as np, matplotlib.pyplot as plt
from foveamap.nuscenes import NuScenesLite, nuscenes_info
from foveamap.frames import make_features, prev_in_ego

nusc = NuScenesLite(DATAROOT, 'v1.0-mini')
info = nuscenes_info(nusc.n_rings())
print('lasers:', info.n_rows, '| mini_train:', nusc.scene_names('mini_train'), '| mini_val:', nusc.scene_names('mini_val'))

toks = nusc.keyframe_tokens('scene-0103')
tok = toks[min(10, len(toks) - 1)]
f = nusc.frame(tok, info)
counts = np.bincount(f['label'][f['label'] >= 0], minlength=9)
for n, c, a in zip(info.class_names, counts, info.active):
    if a: print(f'{n:24s} {c:7d}')
print('moving points:', int(f['moving'].sum()), '| prev sweeps:', [p is not None for p in f['prev']])

feats, idx, _, _ = make_features(f, info, prev_in_ego(f))
COL = np.array(['#009E73','#F0E442','#3DBE93','#C9BD3A','#8FA39A','#A7B1BD','#DCE3EA','#E69F00','#CC79A7'])
fig, ax = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw=dict(width_ratios=[1, 1.6]))
m = f['label'] >= 0
ax[0].scatter(-f['pts'][m, 1], f['pts'][m, 0], s=0.3, c=COL[f['label'][m]])
ax[0].scatter(-f['pts'][f['moving'], 1], f['pts'][f['moving'], 0], s=2, c='red', label='moving')
ax[0].set_aspect('equal'); ax[0].set_xlim(-40, 40); ax[0].set_ylim(-30, 50); ax[0].set_title('Ego frame, heading up (labels)'); ax[0].legend()
ax[1].imshow(feats[0], aspect='auto', cmap='viridis'); ax[1].set_title(f'Range image, {info.n_rows} x {info.n_cols}')
plt.show()
plt.figure(figsize=(16, 2.5)); plt.imshow(feats[6], aspect='auto', vmax=1); plt.title('Motion residual vs. 0.1 s earlier (bright = moved)'); plt.show()
""")

md("## 4. Convert all 10 scenes to cached frames (takes 2–4 minutes)")
code("""
!python scripts/prepare_nuscenes.py --dataroot $DATAROOT --out cache/nuscenes
!du -sh cache/nuscenes
""")

md("""
## 5. Simulator model on real data, before fine-tuning

This measures the gap between the simulator and real data: how well the model trained only on simulated streets labels real nuScenes points.
""")
code("""
import sys, json
sys.path.insert(0, 'scripts')
from train import evaluate
from foveamap.nuscenes import load_cached, cached_info
from foveamap.model import load_model

info = cached_info('cache/nuscenes')
val = load_cached('cache/nuscenes', 'mini_val')
zero_shot = evaluate(load_model('checkpoints/range_unet.pt'), val, info)
print(json.dumps(zero_shot, indent=1))
""")

md("""
## 6. Fine-tune on `mini_train` (GPU, about 5 minutes on a T4)

Training starts from the simulator checkpoint. The unused `parking` class is masked out, because nuScenes has no parking label.
""")
code("""
!python scripts/train.py --dataset nuscenes --cache cache/nuscenes --init checkpoints/range_unet.pt \\
    --out checkpoints/range_unet_nuscenes.pt --epochs 120
""")

md("""
## 7. Full benchmark on a `mini_val` scene, and export the dashboard data

On a T4 the network takes only a few milliseconds, so the grid engine decides the latency. The scene runs twice: once with the NumPy grid engine on Colab's CPU, and once with the PyTorch engine on the GPU (`--grid torch`). The PyTorch run goes to `dashboard/data`. PNG export for the dashboard runs on a background thread and is reported separately (`export_ms`), not as latency.
""")
code("""
SCENE = 'scene-0103'   # or 'scene-0916'
for GRID, OUT in (('numpy', 'results/numpy'), ('torch', 'dashboard/data')):
    !python scripts/run_benchmark.py --dataset nuscenes --cache cache/nuscenes --scene $SCENE \\
        --ckpt checkpoints/range_unet_nuscenes.pt --grid $GRID --out $OUT > benchmark_{SCENE}_{GRID}.log
    !tail -n 3 benchmark_{SCENE}_{GRID}.log
""")
code("""
import pandas as pd
R = {g: json.load(open(f'{d}/metrics.json'))['summary'] for g, d in (('numpy', 'results/numpy'), ('torch', 'dashboard/data'))}
cmp = lambda f: {g: f(s) for g, s in R.items()}
pd.DataFrame({
    'p50 / p95 latency (ms)': cmp(lambda s: f"{s['latency_ms']['p50']:.0f} / {s['latency_ms']['p95']:.0f}"),
    'Throughput (FPS)': cmp(lambda s: f"{s['fps']:.1f}"),
    'Grid engine (projection + fusion, ms)': cmp(lambda s: f"{s['grid_only_ms']:.1f}"),
    **{f'{k} (ms)': cmp(lambda s, k=k: f"{s['stages_ms'][k]:.1f}") for k in R['torch']['stages_ms']},
    'Background export (ms, not in latency)': cmp(lambda s: f"{s['export_ms']:.0f}"),
    'Points lost': cmp(lambda s: s['points_lost']),
    'Drivable IoU on grid, 0-10 m': cmp(lambda s: f"{100*s['drivable_iou_grid_0_10']:.1f}%"),
}).T.rename(columns={'numpy': 'NumPy grid (CPU)', 'torch': 'PyTorch grid (GPU)'})
""")
code("""
p95 = R['torch']['latency_ms']['p95']
print(f"GPU grid engine p95 = {p95:.0f} ms -> {'PASS' if p95 <= 50 else 'FAIL'} (target <= 50 ms)")
""")
code("""
S = R['torch']
ft = json.load(open('checkpoints/range_unet_nuscenes_val.json'))
pct = lambda v: '—' if v is None else f'{100*v:.1f}%'
rows = [
    ('Hardware', S['hardware']),
    ('End-to-end latency p50 / p95', f"{S['latency_ms']['p50']:.0f} / {S['latency_ms']['p95']:.0f} ms"),
    ('Throughput', f"{S['fps']:.1f} FPS"),
    ('Stage means (ms)', ', '.join(f'{k} {v:.0f}' for k, v in S['stages_ms'].items())),
    ('Map memory', f"{S['memory_bytes']['foveated_spec']/1e6:.2f} MB"),
    ('Saving vs uniform 5 cm', f"{S['memory_saving_vs_uniform5']:.0f}x"),
    ('Points lost at tier edges', S['points_lost']),
    ('Val mIoU, simulator model (zero-shot)', pct(zero_shot['miou'])),
    ('Val mIoU, fine-tuned', pct(ft['miou'])),
    ('Point mIoU by band (0-10/10-25/25-50/50-100 m)', ' / '.join(pct(v) for v in S['point_miou_by_band'])),
    ('Grid-cell mIoU by band', ' / '.join(pct(v) for v in S['grid_miou_by_band'])),
    ('Drivable IoU on grid, 0-10 m', pct(S['drivable_iou_grid_0_10'])),
    ('Moving-object IoU', pct(S['moving_iou'])),
]
pd.set_option('display.max_colwidth', 120)
pd.DataFrame(rows, columns=['Check', 'Result'])
""")

md("""
## 8. View the dashboard here

This serves the exported dashboard from the Colab machine and opens it in a frame below.
""")
code("""
!python scripts/make_local_view.py
import subprocess, time
subprocess.Popen(['python', '-m', 'http.server', '8008'], cwd='dashboard', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
from google.colab import output
output.serve_kernel_port_as_iframe(8008, path='/view.html', height=900)
""")

md("""
## 9. Download the results

Send the zip back and I can publish the real-data dashboard and update the docs with these numbers.
""")
code("""
!zip -qr /content/foveamap_nuscenes_results.zip dashboard results/numpy/metrics.json checkpoints/range_unet_nuscenes.pt checkpoints/range_unet_nuscenes_val.json benchmark_*.log
from google.colab import files
files.download('/content/foveamap_nuscenes_results.zip')
""")

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=0)
out = os.path.join(os.path.dirname(__file__), "foveamap_nuscenes_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out)
