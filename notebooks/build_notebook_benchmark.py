"""Builds foveamap_semantickitti_benchmark_colab.ipynb (kept as code so the notebook stays reviewable)."""
import json
import os

cells = []


def md(src):
    cells.append(dict(cell_type="markdown", metadata={}, source=src.strip("\n").splitlines(True)))


def code(src):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=src.strip("\n").splitlines(True)))


md("""
<a href="https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_semantickitti_benchmark_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

md("""
# FoveaMap: SemanticKITTI benchmark only

Re-runs the pipeline benchmark on SemanticKITTI sequence 08 with the latest code, without downloading data or training. It copies the frame cache for sequence 08 and the fine-tuned model from your Google Drive, where the [main SemanticKITTI notebook](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_semantickitti_colab.ipynb) saved them, runs the benchmark `RUNS` times, and prints latency by stage, pothole false alarms, object precision and recall, and the object-size sweep.

**Before you start:** use *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*. It takes about 10 minutes. Run the main notebook once first if you haven't: this one only reads what it saved in `MyDrive/foveamap_data/`.
""")

md("## 1. Setup")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: Runtime > Change runtime type > T4 GPU"
""")
code("""
import os, sys, shutil, importlib, json, subprocess
STRIDE, EPOCHS, FLAGS = 10, 40, ''     # which saved model to use: the main notebook's settings when it trained
SCENE, FRAMES, RUNS = '08', 100, 2     # benchmark the first FRAMES frames of SCENE, RUNS times

from google.colab import drive
drive.mount('/content/drive')
DATA = '/content/drive/MyDrive/foveamap_data'
CACHE = '/content/cache/semantickitti'

%cd /content
shutil.rmtree('/content/foveamap', ignore_errors=True)
!git clone -q --depth 1 https://github.com/ammar-iitm/foveamap.git /content/foveamap
for m in [m for m in list(sys.modules) if m == 'foveamap' or m.startswith('foveamap.')]:
    del sys.modules[m]
importlib.invalidate_caches()
%cd /content/foveamap
!pip -q install -r requirements.txt
!git log --oneline -1
""")

md("""
## 2. Copy the frame cache and the model from Drive

Only sequence 08 and the cache index are copied (a few GB). Each copy is checked against the file in Drive, so a failed copy stops here with a clear message.
""")
code("""
def restore(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} is not in your Drive. Run the main SemanticKITTI notebook once first: '
                                'it saves the frame cache and the fine-tuned model there.')
    size = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(f'already here: {dst}')
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f'copying {src} ({size / 1e9:.2f} GB)', flush=True)
    shutil.copyfile(src, dst + '.part')
    os.replace(dst + '.part', dst)
    if os.path.getsize(dst) != size:
        raise OSError(f'the copy of {src} is incomplete; run this cell again')

DRIVE_CACHE = f'{DATA}/semantickitti_cache_stride{STRIDE}'
DRIVE_CKPT = f'{DATA}/checkpoints_stride{STRIDE}_epochs{EPOCHS}{FLAGS.replace(" ", "")}'
restore(f'{DRIVE_CACHE}/index.json', f'{CACHE}/index.json')
restore(f'{DRIVE_CACHE}/{SCENE}.pkl', f'{CACHE}/{SCENE}.pkl')
restore(f'{DRIVE_CKPT}/range_unet_semantickitti.pt', 'checkpoints/range_unet_semantickitti.pt')
drive.flush_and_unmount()       # the Drive client's background work competes for the 2 vCPUs while timing
print('ready (Drive unmounted until the results are saved)')
""")

md("""
## 3. Benchmark

Each run writes its metrics, per-frame log and dashboard frames to `results/semantickitti_run<N>/`. Colab's shared CPUs have busy spells, so compare the runs: a p95 that differs a lot between them is the machine, not the code. Google Drive stays unmounted while timing, and on CUDA the first frames include compiling the derive step (they are not counted).
""")
code("""
import pandas as pd
pct = lambda v: '—' if v is None else f'{100 * v:.1f}%'

def run(n):
    out, log = f'results/semantickitti_run{n}', f'benchmark_run{n}.log'
    cmd = [sys.executable, 'scripts/run_benchmark.py', '--dataset', 'semantickitti', '--cache', CACHE,
           '--scene', SCENE, '--max-frames', str(FRAMES), '--grid', 'torch', '--out', out]
    with open(log, 'w') as fh:
        if subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode:
            print(open(log).read()[-3000:])
            raise RuntimeError(f'benchmark run {n} failed (log above)')
    return json.load(open(f'{out}/metrics.json'))['summary']

for n in range(1, RUNS + 1):
    S = run(n)
    L = S['latency_ms']
    print(f"\\n=== run {n}: p50 / p95 / p99 {L['p50']:.1f} / {L['p95']:.1f} / {L['p99']:.1f} ms, {S['fps']:.1f} FPS")
    print('stage means (ms):', {k: round(v, 1) for k, v in S['stages_ms'].items()})
    print(f"points lost {S['points_lost']}, drivable IoU 0-10 m {pct(S['drivable_iou_grid_0_10'])}, "
          f"pothole flag rate {100 * S['pothole_flag_rate_drivable_10m']:.3f}%")
    print('point mIoU by band:', [pct(v) for v in S['point_miou_by_band']])

O = S['objects_within_25m']
display(pd.DataFrame([(c, pct(v['precision']), pct(v['recall']), v['tp'], v['fp'], v['fn']) for c, v in O.items()],
                     columns=['Objects within 25 m', 'Precision', 'Recall', 'TP', 'FP', 'FN']))
print('moving flag agrees on matched objects:', pct(S['objects_moving_flag_agreement']))
rows = [(c, r['min_area_m2'], pct(r['precision']), pct(r['recall']), pct(r['f1']))
        for c, rs in S['objects_area_sweep'].items() for r in rs]
display(pd.DataFrame(rows, columns=['Class', 'Min area (m²)', 'Precision', 'Recall', 'F1']))
""")

md("""
## 4. Profile: where the time goes

Runs `torch.profiler` over 20 frames: the CPU and GPU time of each stage, the operations with the most GPU and CPU time, and how often the host waits for the GPU. The profiler adds overhead, so read it relative to the benchmark above. The output is also saved to `profile.txt`.
""")
code("""
!python scripts/profile_pipeline.py --dataset semantickitti --cache $CACHE --scene $SCENE 2>&1 | grep -v "^USDT\\|^STAGE:" | tee profile.txt
""")

md("""
## 5. Download the results

The zip holds the profile, every run's metrics and log, and run 1's dashboard frames. A copy is also saved to `MyDrive/foveamap_data/benchmark_results/`.
""")
code("""
import time
name = f'foveamap_semantickitti_benchmark_{time.strftime("%Y%m%d_%H%M")}.zip'
files_ = ['profile.txt'] + [f'benchmark_run{n}.log' for n in range(1, RUNS + 1)] + \\
         [f'results/semantickitti_run{n}/metrics.json' for n in range(1, RUNS + 1)] + ['results/semantickitti_run1']
!zip -qr /content/$name {' '.join(files_)}
drive.mount('/content/drive')
os.makedirs(f'{DATA}/benchmark_results', exist_ok=True)
shutil.copy(f'/content/{name}', f'{DATA}/benchmark_results/{name}')
from google.colab import files
files.download(f'/content/{name}')
""")

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=0)
out = os.path.join(os.path.dirname(__file__), "foveamap_semantickitti_benchmark_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out)
