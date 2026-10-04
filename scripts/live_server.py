"""Live view: run the pipeline on a recorded drive at sensor rate and serve the dashboard as it goes.

    python scripts/live_server.py --dataset semantickitti --cache /content/cache/semantickitti --scene 08
    python scripts/live_server.py --dataset sim --device cpu        # simulated drive, no GPU needed

Then open http://localhost:8000/ (on Colab: google.colab.output.serve_kernel_port_as_window(8000)).

A worker thread feeds one frame per sensor period (`--rate`, default the recording's own rate)
through the full pipeline: features, network, grid, fusion and objects. It keeps the latest map
snapshot and rolling latency figures. The web server only answers requests: /live/state.json gives
the latest frame and the rolling numbers, /live/tiles.bin its map tiles as raw RGB, encoded when the
browser asks, so encoding never sits on the pipeline's path. When the pipeline falls behind the
sensor it does not queue frames: it skips to the next due frame and counts the late one.
Accuracy needs ground truth, so the live view does not compute it; run the benchmark for that.
"""
import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap.grid import FoveatedGrid, DRIVABLE  # noqa: E402,F401
from foveamap.objects import extract_objects, pack, report  # noqa: E402
from foveamap.pipeline import FoveaMapPipeline, encode_tier, hardware_label, to_host  # noqa: E402
from build_site import wrap  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
WARM = 2              # frames left out of the rolling figures (start-up, compiling)


def load(args):
    """(frames, info, checkpoint) for --dataset, as run_benchmark.py loads them."""
    if args.dataset == "sim":
        from foveamap.frames import SIM_INFO, sim_frames
        frames, info = sim_frames(args.seq)[0], SIM_INFO
        ckpt = os.path.join(ROOT, "checkpoints", "range_unet.pt")
    else:
        mod = __import__(f"foveamap.{args.dataset}", fromlist=["load_scene", "cached_info"])
        cache = args.cache or os.path.join(ROOT, "cache", args.dataset)
        scene = args.scene or ("08" if args.dataset == "semantickitti" else "scene-0103")
        frames, info = mod.load_scene(cache, scene), mod.cached_info(cache)
        if args.dataset == "semantickitti":
            with open(os.path.join(cache, "index.json")) as fh:
                stride = json.load(fh)["stride"]
            info.source = f"SemanticKITTI sequence {scene} · 64-beam Lidar · every {stride}th scan"
            info.hz = 10.0 / stride
        else:
            info.source = f"nuScenes {scene} · {info.n_rows}-beam Lidar · labelled keyframes at 2 Hz"
        ckpt = os.path.join(ROOT, "checkpoints", f"range_unet_{args.dataset}.pt")
    if args.max_frames:
        frames = frames[:args.max_frames]
    return frames, info, args.ckpt or ckpt


def memory_bytes(grid):
    """The dashboard's memory comparison, from cell counts (the benchmark measures the same layouts)."""
    t5, t50 = round(200 / 0.05), round(200 / 0.5)
    return {"foveated_spec": grid.nbytes, "foveated_graded": FoveatedGrid("graded").nbytes,
            "uniform_50cm_2.5d": t50 * t50 * 16, "uniform_5cm_2.5d": t5 * t5 * 16,
            "occupancy_2d_5cm": t5 * t5, "dense_voxel_5cm": t5 * t5 * 160 * 2}


class Live:
    def __init__(self, pipe, frames, info, rate, loop):
        self.pipe, self.frames, self.info, self.rate, self.loop = pipe, frames, info, rate, loop
        self.lock = threading.Lock()
        self.latest = None                     # (frame dict, snapshot, dynamic cells)
        self.history = deque(maxlen=100)       # recent frame dicts, for the latency sparkline
        self.totals = deque(maxlen=200)        # recent end-to-end ms, for the rolling percentiles
        self.n, self.late, self.done = 0, 0, False
        g = pipe.grid
        mem = memory_bytes(g)
        self.summary = dict(
            live=True, dataset=info.name, source=info.source, sensor_rows=info.n_rows, hz=rate,
            hardware=hardware_label(pipe.device, pipe.grid_engine), grid_engine=pipe.grid_engine,
            features_engine=pipe.features, tiers=[dict(cell=t.cell, half=t.half, n=t.n) for t in g.tiers],
            classes=list(info.class_names), active_classes=[bool(a) for a in np.asarray(info.active, bool)],
            memory_bytes=mem, memory_saving_vs_uniform5=mem["uniform_5cm_2.5d"] / mem["foveated_spec"])

    def reset(self):
        g = self.pipe.grid
        g.state = [g._new_layers(t.n) for t in g.tiers]
        g.origins, self.pipe.history = None, []

    def warm_up(self):
        """One frame through every stage before the sensor clock starts (on CUDA the first one
        compiles the derive step, which can take half a minute), then an empty map again."""
        r = self.pipe.step(self.frames[0])
        snap = self.pipe.grid.snapshot()
        extract_objects(self.pipe.grid, snap, to_host(r["dyn"]))
        self.reset()

    def run(self):
        self.warm_up()
        period, due = 1.0 / self.rate, time.perf_counter()
        while True:
            for fr in self.frames:
                now = time.perf_counter()
                if now < due:
                    time.sleep(due - now)
                elif now - due > period:       # behind the sensor by a whole frame: skip, don't queue
                    self.late += 1
                    due += period
                    continue
                self.step(fr)
                due += period
            if not self.loop:
                break
            self.reset()
        self.done = True

    def step(self, fr):
        pipe = self.pipe
        r = pipe.step(fr)
        t0 = time.perf_counter()
        snap = pipe.grid.snapshot()
        dyn = to_host(r["dyn"])
        r["timing"]["publish"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        objs = report(extract_objects(pipe.grid, snap, dyn))
        r["timing"]["objects"] = time.perf_counter() - t0
        timing = {k: round(v * 1000, 2) for k, v in r["timing"].items()}
        total = round(sum(timing.values()), 2)
        # integrity, as the benchmark checks it (bookkeeping, after the timed stages)
        g, pw = pipe.grid, to_host(r["pw"])
        fo = g.fine_index(pw[:, :2]) // g.tiers[-1].ratio - g.origins[-1]
        mem = self.summary["memory_bytes"]
        if "raw_sweep" not in mem:                 # one sweep's sparse 5 cm voxels and raw size, as the benchmark
            mem["sparse_voxel_5cm_one_sweep"] = int(len(np.unique(np.floor(pw / 0.05).astype(np.int64), axis=0)) * 10)
            mem["raw_sweep"] = int(len(fr["pts"]) * 16)
        pose = fr["pose"]
        f = dict(t=self.n, ego=[round(float(pose[0, 3]), 3), round(float(pose[1, 3]), 3)],
                 yaw=round(float(np.arctan2(pose[1, 0], pose[0, 0])), 5),
                 origins=[[int(o[0]), int(o[1])] for o in g.origins], timing_ms=timing, total_ms=total,
                 binned=int(to_host(r["stats"][-1]["n_pts"]).sum()),
                 in_window=int(np.sum(((fo >= 0) & (fo < g.tiers[-1].n)).all(1))), nest_ok=True,
                 objects=pack(objs))
        with self.lock:
            self.latest = (f, snap, dyn)
            self.history.append(f)
            if self.n >= WARM:
                self.totals.append(total)
            self.n += 1

    def state(self):
        with self.lock:
            f = self.latest[0] if self.latest else None
            hist, totals = list(self.history), np.array(self.totals)
            n, late = self.n, self.late
        s = dict(self.summary, frames_processed=n, late_frames=late, finished=self.done)
        if len(totals):
            s.update(latency_ms=dict(p50=float(np.percentile(totals, 50)), p95=float(np.percentile(totals, 95)),
                                     p99=float(np.percentile(totals, 99)), mean=float(totals.mean())),
                     fps=float(1000.0 / totals.mean()),
                     grid_only_ms=float(np.mean([h["timing_ms"]["projection"] + h["timing_ms"]["fusion"]
                                                 for h in hist[-len(totals):]])))
        return dict(summary=s, frame=f, history=hist)

    def tiles(self):
        """(frame number, raw RGB of every tier's map tile, concatenated) for the latest frame."""
        with self.lock:
            if self.latest is None:
                return None, b""
            f, snap, dyn = self.latest
        tiers = self.pipe.grid.tiers
        return f["t"], b"".join(encode_tier(s, d, t.n).tobytes() for s, d, t in zip(snap, dyn, tiers))


def serve(live, port, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self.send(200, page, "text/html; charset=utf-8")
            elif path == "/live/state.json":
                self.send(200, json.dumps(live.state()).encode(), "application/json")
            elif path == "/live/tiles.bin":
                t, body = live.tiles()
                self.send(200 if t is not None else 204, body, "application/octet-stream",
                          {"X-Frame": str(t if t is not None else -1)})
            elif path == "/favicon.ico":
                self.send(204, b"", "image/x-icon")
            else:
                self.send(404, b"not found", "text/plain")

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"live view on http://localhost:{port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="semantickitti", choices=["sim", "nuscenes", "semantickitti"])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--seq", default=os.path.join(ROOT, "data", "demo.npz"), help="simulated drive (.npz)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--grid", default="torch", choices=["numpy", "torch"])
    ap.add_argument("--rate", type=float, default=None, help="frames per second fed in (default: the recording's)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-loop", action="store_true", help="stop after one pass instead of starting over")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    frames, info, ckpt = load(a)
    pipe = FoveaMapPipeline(ckpt, info, device=a.device, grid=a.grid)
    live = Live(pipe, frames, info, a.rate or info.hz or 10.0, loop=not a.no_loop)
    threading.Thread(target=live.run, daemon=True).start()
    with open(os.path.join(ROOT, "dashboard", "index.html"), encoding="utf-8") as fh:
        page = wrap(fh.read()).encode("utf-8")
    serve(live, a.port, page)
