"""Builds foveamap_benchmark_colab.ipynb (kept as code so the notebook stays reviewable)."""
import json
import os

cells = []


def md(src):
    cells.append(dict(cell_type="markdown", metadata={}, source=src.strip("\n").splitlines(True)))


def code(src):
    cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                      source=src.strip("\n").splitlines(True)))


md("""
<a href="https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_benchmark_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
""")

md("""
# FoveaMap: benchmark only

Re-runs the pipeline benchmark with the latest code on **SemanticKITTI** sequence 08 or **nuScenes-mini** scene-0103 (set `DATASET` below), runs it `RUNS` times, profiles it, and zips the results with the dashboard frames. It prints latency by stage, pothole false alarms, object precision and recall, and the object-size sweep.

What it needs, from `MyDrive/foveamap_data/`:
- **SemanticKITTI:** the frame cache and fine-tuned model that the [main SemanticKITTI notebook](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_semantickitti_colab.ipynb) saved there. Run that notebook once first. About 10 minutes.
- **nuScenes:** nothing. The first run downloads nuScenes-mini (about 4 GB), builds the frame cache, fine-tunes for 120 epochs, and saves the cache and model to Drive, about 30–40 minutes. Later runs take about 10 minutes.

**Before you start:** use *Runtime → Change runtime type → T4 GPU*, set `DATASET`, then *Runtime → Run all*. nuScenes is CC BY-NC-SA 4.0 (non-commercial); if the download asks you to log in, see the [nuScenes notebook](https://colab.research.google.com/github/ammar-iitm/foveamap/blob/main/notebooks/foveamap_nuscenes_colab.ipynb) for the manual steps.
""")

md("## 1. Setup")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: Runtime > Change runtime type > T4 GPU"
""")
code("""
DATASET = 'semantickitti'   # or 'nuscenes'
RUNS = 2                    # benchmark runs, to tell the code from the machine's busy spells

import os, sys, shutil, importlib, json, subprocess
from google.colab import drive
drive.mount('/content/drive')
DATA = '/content/drive/MyDrive/foveamap_data'
if DATASET == 'semantickitti':
    STRIDE, EPOCHS, FLAGS = 10, 40, ''                  # which saved model: the main notebook's settings
    SCENE, FRAMES = '08', 100                           # the first 100 frames of sequence 08
    DRIVE_CACHE = f'{DATA}/semantickitti_cache_stride{STRIDE}'
    DRIVE_CKPT = f'{DATA}/checkpoints_stride{STRIDE}_epochs{EPOCHS}{FLAGS.replace(" ", "")}'
elif DATASET == 'nuscenes':
    EPOCHS = 120
    SCENE, FRAMES = 'scene-0103', None                  # all 40 keyframes
    DRIVE_CACHE = f'{DATA}/nuscenes_mini_cache'
    DRIVE_CKPT = f'{DATA}/checkpoints_nuscenes_epochs{EPOCHS}'
else:
    raise ValueError(f'DATASET must be semantickitti or nuscenes, not {DATASET!r}')
CACHE = f'/content/cache/{DATASET}'
CKPT = f'checkpoints/range_unet_{DATASET}.pt'

%cd /content
shutil.rmtree('/content/foveamap', ignore_errors=True)
!git clone -q --depth 1 https://github.com/ammar-iitm/foveamap.git /content/foveamap
for m in [m for m in list(sys.modules) if m == 'foveamap' or m.startswith('foveamap.')]:
    del sys.modules[m]
importlib.invalidate_caches()
%cd /content/foveamap
!pip -q install -r requirements.txt
!git log --oneline -1
import torch
if not torch.cuda.is_available():
    raise RuntimeError('This runtime has no GPU, so the benchmark would measure the CPU instead. Choose '
                       'Runtime > Change runtime type > T4 GPU (if Colab offers none, the free GPU time '
                       'may be used up for now) and run all again.')
print('GPU:', torch.cuda.get_device_name(0))
""")

md("""
## 2. Data and model

Copied from Drive when they are there; each copy is checked against its source, so a failed copy stops here with a clear message. For nuScenes, whatever is missing is made and saved to Drive first.
""")
code("""
def restore(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f'{src} is not in your Drive. For SemanticKITTI, run the main SemanticKITTI '
                                'notebook once first: it saves the frame cache and the fine-tuned model there.')
    size = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        print(f'already here: {dst}')
        return
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    print(f'copying {src} ({size / 1e9:.2f} GB)', flush=True)
    shutil.copyfile(src, dst + '.part')
    os.replace(dst + '.part', dst)
    if os.path.getsize(dst) != size:
        raise OSError(f'the copy of {src} is incomplete; run this cell again')

def save(files, folder):
    os.makedirs(folder, exist_ok=True)
    for f in files:
        restore(f, f'{folder}/{os.path.basename(f)}')

have_cache = os.path.exists(f'{DRIVE_CACHE}/index.json')
have_model = os.path.exists(f'{DRIVE_CKPT}/range_unet_{DATASET}.pt')
if DATASET == 'semantickitti' or (have_cache and have_model):
    restore(f'{DRIVE_CACHE}/index.json', f'{CACHE}/index.json')
    restore(f'{DRIVE_CACHE}/{SCENE}.pkl', f'{CACHE}/{SCENE}.pkl')
else:                                                    # nuScenes, first time
    if have_cache:
        for f in sorted(os.listdir(DRIVE_CACHE)):
            restore(f'{DRIVE_CACHE}/{f}', f'{CACHE}/{f}')
    else:
        DATAROOT = '/content/nuscenes'
        os.makedirs(DATAROOT, exist_ok=True)
        if not os.path.exists('/content/v1.0-mini.tgz'):
            !wget -q --show-progress -O /content/v1.0-mini.tgz https://www.nuscenes.org/data/v1.0-mini.tgz
        if not os.path.exists('/content/nuScenes-lidarseg-mini-v1.0.tar.bz2'):
            !wget -q --show-progress -O /content/nuScenes-lidarseg-mini-v1.0.tar.bz2 https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2
        !tar -xzf /content/v1.0-mini.tgz -C $DATAROOT
        !tar -xjf /content/nuScenes-lidarseg-mini-v1.0.tar.bz2 -C $DATAROOT
        !python scripts/prepare_nuscenes.py --dataroot $DATAROOT --out $CACHE
        if not os.path.exists(f'{CACHE}/index.json'):
            raise RuntimeError('Building the nuScenes cache failed (see above).')
        save([f'{CACHE}/{f}' for f in sorted(os.listdir(CACHE))], DRIVE_CACHE)
    if not have_model:
        print(f'fine-tuning for {EPOCHS} epochs', flush=True)
        !python scripts/train.py --dataset nuscenes --cache $CACHE --init checkpoints/range_unet.pt \\
            --out $CKPT --epochs $EPOCHS 2>&1 | tee train_nuscenes.log | awk '!/^step/ || (++n % 10 == 0)'
        if not os.path.exists(CKPT):
            raise RuntimeError('Fine-tuning failed (see above).')
        save([CKPT, CKPT.replace('.pt', '_val.json'), 'train_nuscenes.log'], DRIVE_CKPT)
restore(f'{DRIVE_CKPT}/range_unet_{DATASET}.pt', CKPT)
drive.flush_and_unmount()       # the Drive client's background work competes for the 2 vCPUs while timing
print('ready (Drive unmounted until the results are saved)')
""")

md("""
## 3. Benchmark

Each run writes its metrics, per-frame log and dashboard frames to `results/<DATASET>_run<N>/`. Colab's shared CPUs have busy spells, so compare the runs: a p95 that differs a lot between them is the machine, not the code. On CUDA the first frames include compiling the derive step (they are not counted).
""")
code("""
import pandas as pd, threading, time
pct = lambda v: '—' if v is None else f'{100 * v:.1f}%'
def monitor(path, stop):
    # every 0.5 s: CPU steal and busy share (/proc/stat), and the GPU's SM clock, power, temperature,
    # performance state and utilisation (one long-running nvidia-smi), to line slow frames up with
    # what the machine was doing. Each line is stamped on arrival, so no clock or time zone to match.
    gpu = {'sm_mhz': '', 'power_w': '', 'temp_c': '', 'pstate': '', 'gpu_util': ''}
    def read_gpu(proc):
        for line in proc.stdout:
            v = [x.strip() for x in line.split(',')]
            if len(v) == 5:
                gpu.update(sm_mhz=v[0], power_w=v[1], temp_c=v[2], pstate=v[3].lstrip('P'), gpu_util=v[4])
    smi = None
    try:
        smi = subprocess.Popen(['nvidia-smi', '--query-gpu=clocks.sm,power.draw,temperature.gpu,pstate,utilization.gpu',
                                '--format=csv,noheader,nounits', '-lms', '500'], stdout=subprocess.PIPE, text=True)
        threading.Thread(target=read_gpu, args=(smi,), daemon=True).start()
    except OSError as e:
        print('no GPU monitor:', e)
    cpu = lambda: [int(v) for v in open('/proc/stat').readline().split()[1:9]]
    prev = cpu()
    with open(path, 'w') as fh:
        fh.write('time,steal_pct,busy_pct,sm_mhz,power_w,temp_c,pstate,gpu_util\\n')
        while not stop.wait(0.5):
            cur = cpu()
            d = [c - p for c, p in zip(cur, prev)]
            prev, tot = cur, max(sum(d), 1)       # user nice system idle iowait irq softirq steal
            fh.write(f"{time.time():.3f},{100 * d[7] / tot:.1f},{100 * (tot - d[3] - d[4]) / tot:.1f},"
                     f"{gpu['sm_mhz']},{gpu['power_w']},{gpu['temp_c']},{gpu['pstate']},{gpu['gpu_util']}\\n")
            fh.flush()
    if smi:
        smi.terminate()

def safe_monitor(path, stop):
    try:
        monitor(path, stop)
    except Exception as e:                        # a monitor problem must not stop the benchmark
        print(f'machine monitor stopped: {type(e).__name__}: {e}')

def run(n):
    out, log = f'results/{DATASET}_run{n}', f'benchmark_{DATASET}_run{n}.log'
    cmd = [sys.executable, 'scripts/run_benchmark.py', '--dataset', DATASET, '--cache', CACHE,
           '--scene', SCENE, '--ckpt', CKPT, '--grid', 'torch', '--out', out]
    if FRAMES:
        cmd += ['--max-frames', str(FRAMES)]
    stop = threading.Event()
    mon = threading.Thread(target=safe_monitor, args=(f'monitor_{DATASET}_run{n}.csv', stop), daemon=True)
    mon.start()
    try:
        with open(log, 'w') as fh:
            if subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode:
                print(open(log).read()[-3000:])
                raise RuntimeError(f'benchmark run {n} failed (log above)')
    finally:
        stop.set()
        mon.join()
    try:
        m = pd.read_csv(f'monitor_{DATASET}_run{n}.csv')
        print(f'run {n} machine: CPU steal max {m.steal_pct.max():.0f}%, mean {m.steal_pct.mean():.1f}%; '
              f'GPU SM clock {m.sm_mhz.min()}-{m.sm_mhz.max()} MHz, {m.temp_c.max()} C max, '
              f'P-states {sorted(m.pstate.dropna().unique())}')
    except Exception as e:
        print(f'run {n}: no machine monitor data ({e})')
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
!python scripts/profile_pipeline.py --dataset $DATASET --cache $CACHE --scene $SCENE --ckpt $CKPT 2>&1 | grep -v "^USDT\\\\|^STAGE:" | tee profile.txt
""")

md("""
## 5. Download the results

The zip holds the profile, every run's metrics, log and machine monitor (CPU steal, GPU clock and power every 0.5 s), and run 1's dashboard frames. A copy is also saved to `MyDrive/foveamap_data/benchmark_results/`.
""")
code("""
import time
name = f'foveamap_{DATASET}_benchmark_{time.strftime("%Y%m%d_%H%M")}.zip'
files_ = ['profile.txt'] + [f'benchmark_{DATASET}_run{n}.log' for n in range(1, RUNS + 1)] + \\
         [f'monitor_{DATASET}_run{n}.csv' for n in range(1, RUNS + 1)] + \\
         [f'results/{DATASET}_run{n}/metrics.json' for n in range(1, RUNS + 1)] + [f'results/{DATASET}_run1']
!zip -qr /content/$name {' '.join(files_)}
drive.mount('/content/drive')
os.makedirs(f'{DATA}/benchmark_results', exist_ok=True)
shutil.copy(f'/content/{name}', f'{DATA}/benchmark_results/{name}')
from google.colab import files
files.download(f'/content/{name}')
""")

md("""
## 6. Live view (optional)

Runs the pipeline live on the same recording: one frame per sensor tick, through features, network, grid, fusion and objects, and opens the dashboard as it goes. The map, objects and latency update as each frame is processed; frames the pipeline can't keep up with are skipped and counted, not queued. `RATE` is how many frames per second are fed in: the recording's own rate by default (1 Hz for SemanticKITTI's every 10th scan, 2 Hz for nuScenes keyframes); a T4 keeps up with 10 Hz or more. Accuracy is not computed live (it needs ground truth; section 3 measures it).

Run `live.terminate()` to stop it.
""")
code("""
import time, urllib.request
RATE, PORT = None, 8000
cmd = [sys.executable, 'scripts/live_server.py', '--dataset', DATASET, '--cache', CACHE, '--scene', SCENE,
       '--ckpt', CKPT, '--port', str(PORT)] + (['--rate', str(RATE)] if RATE else [])
live = subprocess.Popen(cmd, stdout=open('live.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(180):                      # the first frame compiles the derive step: up to a minute
    if live.poll() is not None:
        print(open('live.log').read())
        raise RuntimeError('the live server stopped (log above)')
    try:
        if json.load(urllib.request.urlopen(f'http://localhost:{PORT}/live/state.json'))['frame']:
            break
    except Exception:
        pass
    time.sleep(1)
from google.colab import output
output.serve_kernel_port_as_window(PORT, anchor_text='Open the live view')
""")

nb = dict(cells=cells, metadata=dict(
    accelerator="GPU", colab=dict(provenance=[], gpuType="T4"),
    kernelspec=dict(display_name="Python 3", name="python3"),
    language_info=dict(name="python")), nbformat=4, nbformat_minor=0)
out = os.path.join(os.path.dirname(__file__), "foveamap_benchmark_colab.ipynb")
with open(out, "w") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out)
