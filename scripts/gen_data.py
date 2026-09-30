"""Generate simulated sequences: train / val / demo (different random worlds)."""
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap.sim import simulate_sequence, save_sequence  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "data")

JOBS = [(f"train_{i:02d}", 100 + i, 36) for i in range(8)] + [("val_00", 900, 30), ("demo", 7, 60)]


def run(job):
    name, seed, n = job
    path = os.path.join(OUT, name + ".npz")
    if os.path.exists(path):
        return name, 0.0
    t0 = time.time()
    save_sequence(path, simulate_sequence(seed, n_frames=n))
    return name, time.time() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with Pool(2) as p:
        for name, dt in p.imap_unordered(run, JOBS):
            print(f"{name}: {dt:.1f}s", flush=True)
