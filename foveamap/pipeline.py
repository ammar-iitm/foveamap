"""End-to-end FoveaMap pipeline: sweep -> features -> network -> foveated grid.

Also the benchmark harness (PRD FR-18/19): per-stage latency, map memory vs.
uniform baselines, accuracy by distance band, integrity, and the frame
export consumed by the web dashboard.
"""
from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from .sim import NUM_CLASSES, ROAD, PARKING, VEHICLE, PERSON
from .model import load_model, predict, pick_device
from .frames import DatasetInfo, make_features, prev_in_ego, transform
from .grid import (FoveatedGrid, UNKNOWN, DRIVABLE, F_DYNAMIC, F_OVERHANG, F_STEP, F_DEPRESSION)
from .grid_torch import TorchFoveatedGrid
from . import features_torch
from .objects import OBJECT_CLASSES, extract_objects, match_objects, pack

BANDS = [(0, 10), (10, 25), (25, 50), (50, 100)]
BAND_NAMES = ["0–10 m", "10–25 m", "25–50 m", "50–100 m"]


def hardware_label(device, grid="numpy"):
    if device.type == "cuda":
        where = "GPU" if grid == "torch" else "CPU"
        return f"{torch.cuda.get_device_name(device)} GPU (FP16 inference) + {where} grid engine"
    engine = "PyTorch" if grid == "torch" else "NumPy"
    return f"{os.cpu_count()} vCPU, no GPU (CPU PyTorch, {engine} grid engine)"


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def to_host(x):
    """Tensors (also inside dicts / lists) -> NumPy arrays."""
    if torch.is_tensor(x):
        return x.cpu().numpy()
    if isinstance(x, dict):
        return {k: to_host(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_host(v) for v in x)
    return x


def _prev_in_ego_dev(frame, history, device):
    """`frames.prev_in_ego` on the device: (pts float32 tensor, ring) entries."""
    inv = torch.as_tensor(np.linalg.inv(frame["pose"]), device=device)
    src = frame.get("prev")
    if src is None:
        src = history
    out = []
    for item in list(src or [])[:2]:
        if item is None:
            out.append(None)
            continue
        pw, ring = item
        pw = torch.as_tensor(pw).to(device, torch.float64)
        out.append(((pw @ inv[:3, :3].T + inv[:3, 3]).float(), ring))
    out += [None] * (2 - len(out))
    return out


class FoveaMapPipeline:
    """sweep -> features -> network -> foveated grid, with per-stage timing."""

    def __init__(self, ckpt, info: DatasetInfo, profile="spec", fuse=True, device=None, grid="numpy",
                 features=None):
        """grid / features: "numpy" (CPU) or "torch" (on the model's device); features follows grid by default."""
        if not torch.cuda.is_available():
            torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.device = pick_device(device)
        self.model = load_model(ckpt, self.device)
        self.info = info
        self.grid_engine = grid
        self.features = features or grid
        if self.features not in ("numpy", "torch"):
            raise ValueError(f"unknown features engine {self.features!r}")
        if grid == "torch":
            self.grid = TorchFoveatedGrid(profile, fuse=fuse, device=self.device)
        elif grid == "numpy":
            self.grid = FoveatedGrid(profile, fuse=fuse)
        else:
            raise ValueError(f"unknown grid engine {grid!r}")
        self.history = []          # last 2 sweeps as (pts_world, ring), newest last

    def step(self, frame):
        """One sweep. With the torch engine, predictions stay on the device through
        binning and fusion, and the per-point / per-cell outputs are tensors (see to_host)."""
        on_dev = self.grid_engine == "torch"
        # torch features and grid: each sweep goes to the device once and every point transform
        # runs there (float64, except on MPS, which has none)
        dev_math = on_dev and self.features == "torch" and self.device.type != "mps"
        T = {}
        t0 = time.perf_counter()
        hist = [self.history[-k] if len(self.history) >= k else None for k in (1, 2)]
        if dev_math:
            pts = torch.as_tensor(frame["pts"]).to(self.device, torch.float32)
            prev = _prev_in_ego_dev(frame, hist, self.device)
            feats, idx, row, col = features_torch.make_features(dict(frame, pts=pts), self.info, prev, self.device)
        elif self.features == "torch":
            prev = prev_in_ego(frame, hist)
            feats, idx, row, col = features_torch.make_features(frame, self.info, prev, self.device)
        else:
            prev = prev_in_ego(frame, hist)
            feats, idx, row, col = make_features(frame, self.info, prev)
        _sync(self.device)
        T["preprocess"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        probs, pmove = predict(self.model, feats, self.info.active, to_host=not on_dev)
        _sync(self.device)
        T["inference"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if on_dev:
            row, col = (torch.as_tensor(a).to(self.device) for a in (row, col))
        else:
            row, col = to_host(row), to_host(col)
        P = probs[row, col]                                   # every point takes its pixel's prediction
        cls = P.argmax(1)
        moving = (pmove[row, col] > 0.5) & ((cls == VEHICLE) | (cls == PERSON))
        if dev_math:
            pose = torch.as_tensor(frame["pose"], device=self.device)
            pw = pts.double() @ pose[:3, :3].T + pose[:3, 3]
        else:
            pw = transform(frame["pose"], frame["pts"].astype(np.float64))
        ego_xy = frame["pose"][:2, 3]
        origins = self.grid.window_origins(ego_xy)
        stats = self.grid.bin_points(pw[:, :2], pw[:, 2], P, moving, origins)
        _sync(self.device)
        T["projection"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        dyn = self.grid.fuse_stats(stats, origins)
        _sync(self.device)
        T["fusion"] = time.perf_counter() - t0

        self.history.append((pw, frame["ring"]))
        self.history = self.history[-2:]
        return dict(timing=T, cls_pts=cls, moving_pts=moving, stats=stats, dyn=dyn,
                    pts=frame["pts"], pw=pw, ego_xy=ego_xy)


# ----------------------------------------------------------------------------
# Dashboard encoding
# ----------------------------------------------------------------------------
def encode_tier(state, dyn, n):
    """Pack one tier into an RGB tile (see dashboard/README for the bit layout)."""
    cls = getattr(state, "eff_cls", state.cls).astype(np.int32)
    disp = np.where(cls == UNKNOWN, 0, cls + 1)
    flags = state.flags & 0x0F
    dmask = np.zeros((n, n), bool)
    dmask[dyn["i"], dyn["j"]] = True
    disp[dyn["i"], dyn["j"]] = dyn["cls"] + 1
    R = disp | np.where(dmask, 16, 0) | np.where(flags & F_OVERHANG, 32, 0) \
        | np.where(flags & F_STEP, 64, 0) | np.where(flags & F_DEPRESSION, 128, 0)
    z = state.z_max.astype(np.float32)
    G = np.where(np.isfinite(z), np.clip((z + 0.5) / 4.5 * 254, 0, 254) + 1, 0)
    cost = np.where(dmask, 254, state.cost.astype(np.int32))
    age = state.age.astype(np.int32)
    ab = np.where(age == 0, 0, np.where(age <= 10, 1, np.where(age <= 20, 2, 3)))
    ab = np.where(age == UNKNOWN, 3, ab)
    B = ((cost >> 2) << 2) | ab
    img = np.stack([R, G, B], -1).astype(np.uint8)
    return img[::-1, ::-1]          # rows = -x (forward is up), cols = -y (left is left)


def encode_gt(gt_stats, conf, n):
    cls = np.zeros(n * n, np.int32)
    k = gt_stats["key"]
    c = gt_stats["p_all"].argmax(1)
    mv = gt_stats["n_dyn"] > 0
    cls[k] = (c + 1) | np.where(mv, 16, 0)
    R = cls.reshape(n, n)
    img = np.stack([R, conf.astype(np.int32), np.zeros_like(R)], -1).astype(np.uint8)
    return img[::-1, ::-1]


# ----------------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------------
def confusion(pred, gt, k=NUM_CLASSES):
    return np.bincount(gt * k + pred, minlength=k * k).reshape(k, k)


def ious_from(cm):
    tp = np.diag(cm).astype(np.float64)
    den = cm.sum(0) + cm.sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)


def _export_frame(snap, dyn, gstats, tiers, path):
    """Background job: encode the map tiles, the publish PNG and the dashboard frame.
    Runs off the critical path; its time is reported as export_ms, not latency."""
    t0 = time.perf_counter()
    tiles = [encode_tier(s, d, tr.n) for s, d, tr in zip(snap, dyn, tiers)]
    pub_buf = io.BytesIO()
    Image.fromarray(np.concatenate(tiles, 1)).save(pub_buf, format="PNG", compress_level=1)
    gt_tiles = [encode_gt(gs, s.conf, tr.n) for gs, s, tr in zip(gstats, snap, tiers)]
    buf = io.BytesIO()
    Image.fromarray(np.concatenate(tiles + gt_tiles, 1)).save(buf, format="PNG", optimize=False, compress_level=6)
    if path is not None:
        with open(path, "wb") as fh:
            fh.write(buf.getvalue())
    return (time.perf_counter() - t0) * 1000, len(buf.getvalue())


def run_benchmark(frames, info: DatasetInfo, ckpt, out_dir, truth=None, profile="spec",
                  export="after", n_uniform=3, device=None, grid="numpy", features=None):
    """Run frames through the pipeline; write metrics.json, frames/*.png and
    points.b64.txt for the dashboard. truth: simulator-only curb/pothole geometry.
    export: 'after' (default: keep each frame's map snapshot, ~5 MB, and encode the PNGs
    after the timed loop), 'async' (encode on a background thread during the run; on a
    small machine it competes with the pipeline), or 'none' / False.
    grid / features: "numpy" or "torch" (on the model's device); features follows grid by default."""
    if export is True:
        export_mode = "async"
    elif export is False:
        export_mode = "none"
    elif isinstance(export, str) and export.lower() in ("async", "after", "none"):
        export_mode = export.lower()
    else:
        raise ValueError(f"unknown export mode {export!r}; choose 'async', 'after', or 'none'")
    T = len(frames)
    pipe = FoveaMapPipeline(ckpt, info, profile, device=device, grid=grid, features=features)
    gt_grid = FoveatedGrid(profile, fuse=False)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for old in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, old))
    C = NUM_CLASSES
    cm_pts = np.zeros((len(BANDS), C, C), np.int64)
    cm_grid = np.zeros((len(BANDS), C, C), np.int64)
    mov_tp = np.zeros(len(BANDS)); mov_un = np.zeros(len(BANDS))
    curb_hit = curb_tot = pot_hit = pot_tot = 0
    obj_counts = {c: np.zeros(3, np.int64) for c in OBJECT_CLASSES}   # tp, fp, fn within 25 m
    obj_agree = obj_pairs = 0
    dep_hit = dep_tot = 0           # drivable cells within 10 m flagged as potholes (false alarms on real roads)
    per_frame, pts_blob, pts_index = [], [], []
    rng = np.random.default_rng(0)
    from scipy.ndimage import binary_dilation
    if export_mode == "async":
        exporter = ThreadPoolExecutor(max_workers=1)
        exports = []
    elif export_mode == "after":
        export_queue = []

    for t, fr in enumerate(frames):
        r = pipe.step(fr)
        ego = r["ego_xy"]

        # ---- publish: host snapshot of the map; tile/PNG encoding runs in the background
        t0 = time.perf_counter()
        snap = pipe.grid.snapshot()
        dyn = to_host(r["dyn"])
        r["timing"]["publish"] = time.perf_counter() - t0

        # ---- objects: obstacle cells grouped into classified, boxed objects (host, timed)
        t0 = time.perf_counter()
        objs = extract_objects(pipe.grid, snap, dyn)
        r["timing"]["objects"] = time.perf_counter() - t0
        for k in ("cls_pts", "moving_pts", "stats", "pw"):    # benchmark bookkeeping, not timed
            r[k] = to_host(r[k])

        # ground truth grid for the same frame (benchmark only, not timed)
        lab = fr["label"].astype(np.int64)
        has = lab >= 0
        gm = fr["moving"]
        gstats = gt_grid.bin_points(r["pw"][has, :2], r["pw"][has, 2], np.eye(C)[lab[has]], gm[has], pipe.grid.origins)
        # reference objects: the same extraction on a grid fused from this frame's labels
        gdyn = gt_grid.fuse_stats(gstats, pipe.grid.origins)
        cnt, agree, pairs = match_objects(objs, extract_objects(gt_grid, gt_grid.state, gdyn), ego)
        for c in OBJECT_CLASSES:
            obj_counts[c] += np.asarray(cnt[c])
        obj_agree += agree; obj_pairs += pairs
        path = os.path.join(frames_dir, f"f{t:03d}.png") if export_mode != "none" else None
        if export_mode == "async":
            exports.append(exporter.submit(_export_frame, snap, dyn, gstats, pipe.grid.tiers, path))
        elif export_mode == "after":
            export_queue.append((snap, dyn, gstats, pipe.grid.tiers, path))

        # ---- point accuracy by distance band (ego-frame horizontal distance) ----
        d = np.hypot(r["pts"][:, 0], r["pts"][:, 1])
        for b, (lo, hi) in enumerate(BANDS):
            m = (d >= lo) & (d < hi) & has
            cm_pts[b] += confusion(r["cls_pts"][m], lab[m])
            pm, gmm = r["moving_pts"][m], gm[m]
            mov_tp[b] += np.sum(pm & gmm); mov_un[b] += np.sum(pm | gmm)

        # ---- grid accuracy (single-frame prediction vs rasterised ground truth) ---
        for k, (ps, gs) in enumerate(zip(r["stats"], gstats)):
            common, ip, ig = np.intersect1d(ps["key"], gs["key"], return_indices=True)
            pc = ps["p_all"][ip].argmax(1); gc = gs["p_all"][ig].argmax(1)
            cen = pipe.grid.cell_centres(k).reshape(-1, 2)[common] - ego[:2]
            dist = np.hypot(cen[:, 0], cen[:, 1])
            finer = pipe.grid.inner_mask(k).reshape(-1)[common]
            for b, (lo, hi) in enumerate(BANDS):
                m = (dist >= lo) & (dist < hi) & ~finer
                cm_grid[b] += confusion(pc[m], gc[m])

        # ---- pothole flags on drivable ground within 10 m, this frame's cells ----
        s0 = snap[0]
        cen0 = pipe.grid.cell_centres(0)
        drv0 = DRIVABLE[s0.eff_cls] & (s0.age == 0) & \
            (np.hypot(cen0[..., 0] - ego[0], cen0[..., 1] - ego[1]) < 10)
        dep_hit += int(np.sum(drv0 & ((s0.flags & F_DEPRESSION) > 0))); dep_tot += int(np.sum(drv0))

        # ---- curb + pothole detection (simulator only: needs exact geometry) ----
        if truth is not None:
            s0, tr0 = snap[0], pipe.grid.tiers[0]
            cen = pipe.grid.cell_centres(0)
            near = np.hypot(cen[..., 0] - ego[0], cen[..., 1] - ego[1]) < 10
            obs = s0.age == 0
            step_d = binary_dilation((s0.flags & (F_STEP | F_DEPRESSION)) > 0, iterations=2)
            curb = (np.abs(np.abs(cen[..., 1]) - 7.0) < tr0.cell) & (np.abs(cen[..., 0] - truth["cross_x"]) > 8) & near & obs
            curb_hit += np.sum(curb & step_d); curb_tot += np.sum(curb)
            for px, py, pr, _ in truth["potholes"]:
                ph = ((cen[..., 0] - px) ** 2 + (cen[..., 1] - py) ** 2 < (pr * 0.7) ** 2) & near & obs
                pot_hit += np.sum(ph & step_d); pot_tot += np.sum(ph)

        # ---- integrity: every point in the window binned, fine tier nests in coarse
        g = pipe.grid
        n_in = int(r["stats"][-1]["n_pts"].sum())                 # sum of per-cell counts
        n_pts = int(len(r["pts"]))
        fo = g.fine_index(r["pw"][:, :2]) // g.tiers[-1].ratio - g.origins[-1]
        in_window = int(np.sum(((fo >= 0) & (fo < g.tiers[-1].n)).all(1)))   # independent window test
        nest_ok = True
        for k in range(1, len(g.tiers)):
            m = g.inner_mask(k).reshape(-1)[r["stats"][k]["key"]]
            nest_ok &= int(r["stats"][k]["n_pts"][m].sum()) == int(r["stats"][k - 1]["n_in"])

        # ---- points overlay for the dashboard (decimated, ego frame) ------------
        sel = rng.choice(n_pts, size=min(12000, n_pts), replace=False)
        px = np.clip(np.round(r["pts"][sel, 0] * 100), -32000, 32000).astype(np.int16)
        py = np.clip(np.round(r["pts"][sel, 1] * 100), -32000, 32000).astype(np.int16)
        pc = (r["cls_pts"][sel] | (r["moving_pts"][sel] << 4)).astype(np.uint8)
        pts_index.append([int(sum(len(b) for b in pts_blob)), int(len(sel))])
        pts_blob.append(np.concatenate([px.view(np.uint8), py.view(np.uint8), pc]).tobytes())

        pose = fr["pose"]
        timing_ms = {k: round(v * 1000, 2) for k, v in r["timing"].items()}
        per_frame.append(dict(
            t=t, ego=[round(float(ego[0]), 3), round(float(ego[1]), 3)],
            yaw=round(float(np.arctan2(pose[1, 0], pose[0, 0])), 5),
            origins=[[int(o[0]), int(o[1])] for o in g.origins],
            timing_ms=timing_ms, total_ms=round(sum(timing_ms.values()), 2),
            points=n_pts, binned=n_in, in_window=in_window, nest_ok=bool(nest_ok),
            cells_updated=int(sum(len(s["key"]) for s in r["stats"])),
            objects=pack(objs),
        ))
        print(f"frame {t:3d}  total {per_frame[-1]['total_ms']:7.1f} ms  "
              + "  ".join(f"{k} {v:.0f}" for k, v in timing_ms.items()), flush=True)

    if export_mode == "async":
        exporter.shutdown(wait=True)
        for f, job in zip(per_frame, exports):
            f["export_ms"], f["png_bytes"] = job.result()
            f["export_ms"] = round(f["export_ms"], 2)
    elif export_mode == "after":
        for f, args in zip(per_frame, export_queue):
            f["export_ms"], f["png_bytes"] = _export_frame(*args)
            f["export_ms"] = round(f["export_ms"], 2)
    else:
        for f in per_frame:
            f["export_ms"], f["png_bytes"] = 0.0, 0

    # ------------------------------------------------------------------ summary
    warm = per_frame[2:] if T > 3 else per_frame
    totals = np.array([f["total_ms"] for f in warm])
    stages = {k: float(np.mean([f["timing_ms"][k] for f in warm])) for k in per_frame[0]["timing_ms"]}
    pts_iou = [ious_from(cm) for cm in cm_pts]
    grid_iou = [ious_from(cm) for cm in cm_grid]
    all_pts = ious_from(cm_pts.sum(0))
    drv = lambda cm: _binary_iou(cm, (ROAD, PARKING))  # noqa: E731

    # memory: measured bytes of the live structure vs. baselines of the same extent
    mem = {"foveated_spec": pipe.grid.nbytes}
    mem["foveated_graded"] = FoveatedGrid("graded").nbytes
    mem["uniform_50cm_2.5d"] = FoveatedGrid("uniform50").nbytes
    u5 = FoveatedGrid("uniform5", fuse=False)
    mem["uniform_5cm_2.5d"] = u5.nbytes
    mem["occupancy_2d_5cm"] = 4000 * 4000 * 1
    mem["dense_voxel_5cm"] = 4000 * 4000 * 160 * 2
    vk = np.unique(np.floor(r["pw"] / 0.05).astype(np.int64), axis=0)   # sparse 5 cm voxel hash, one sweep
    mem["sparse_voxel_5cm_one_sweep"] = int(len(vk) * 10)
    mem["raw_sweep"] = int(len(r["pts"]) * 16)
    u_times = []
    for fr in frames[:min(n_uniform, T)]:            # uniform 5 cm grid, same code path
        pw = transform(fr["pose"], fr["pts"].astype(np.float64))
        lab = np.clip(fr["label"].astype(np.int64), 0, C - 1)
        t0 = time.perf_counter()
        u5.update(pw[:, :2], pw[:, 2], np.eye(C)[lab], np.zeros(len(pw), bool), fr["pose"][:2, 3])
        u_times.append((time.perf_counter() - t0) * 1000)
    del u5

    active = np.asarray(info.active, bool)
    summary = dict(
        dataset=info.name, source=info.source, sensor_rows=info.n_rows, hz=info.hz,
        profile=profile,
        hardware=hardware_label(pipe.device, grid),
        grid_engine=grid,
        features_engine=pipe.features,
        export_mode=export_mode,
        frames=T,
        latency_ms=dict(p50=float(np.percentile(totals, 50)), p95=float(np.percentile(totals, 95)),
                        p99=float(np.percentile(totals, 99)), mean=float(totals.mean())),
        fps=float(1000.0 / totals.mean()),
        grid_only_ms=float(np.mean([f["timing_ms"]["projection"] + f["timing_ms"]["fusion"] for f in warm])),
        uniform5_grid_ms=float(np.mean(u_times)),
        stages_ms=stages,
        export_ms=float(np.mean([f["export_ms"] for f in warm])),     # background thread, not in latency
        memory_bytes=mem,
        memory_saving_vs_uniform5=mem["uniform_5cm_2.5d"] / mem["foveated_spec"],
        cells=dict(foveated_spec=pipe.grid.n_cells, uniform_5cm=16_000_000),
        tiers=[dict(cell=tr.cell, half=tr.half, n=tr.n) for tr in pipe.grid.tiers],
        classes=list(info.class_names),
        active_classes=[bool(a) for a in active],
        bands=BAND_NAMES,
        point_miou_by_band=[_nanmean(x[active]) for x in pts_iou],
        grid_miou_by_band=[_nanmean(x[active]) for x in grid_iou],
        point_iou_by_class=[_f(x) for x in all_pts],
        grid_iou_by_class_band=[[_f(x) for x in g] for g in grid_iou],
        point_iou_by_class_band=[[_f(x) for x in g] for g in pts_iou],
        point_miou=_nanmean(all_pts[active]),
        drivable_iou_grid_0_10=drv(cm_grid[0]),
        drivable_iou_points_0_10=drv(cm_pts[0]),
        moving_iou_by_band=[float(a / b) if b else None for a, b in zip(mov_tp, mov_un)],
        moving_iou=float(mov_tp.sum() / max(mov_un.sum(), 1)) if mov_un.sum() else None,
        curb_recall_10m=float(curb_hit / max(curb_tot, 1)) if truth is not None else None,
        pothole_recall_10m=float(pot_hit / max(pot_tot, 1)) if truth is not None else None,
        pothole_flag_rate_drivable_10m=float(dep_hit / max(dep_tot, 1)),
        points_lost=int(sum(f["in_window"] - f["binned"] for f in per_frame)),
        nesting_ok=all(f["nest_ok"] for f in per_frame),
        objects_within_25m={info.class_names[c]: dict(
            precision=float(tp / (tp + fp)) if tp + fp else None, recall=float(tp / (tp + fn)) if tp + fn else None,
            tp=int(tp), fp=int(fp), fn=int(fn)) for c, (tp, fp, fn) in obj_counts.items()},
        objects_moving_flag_agreement=float(obj_agree / obj_pairs) if obj_pairs else None,
        objects_per_frame=float(np.mean([len(f["objects"]) for f in per_frame])),
        points_file_index=pts_index,
    )
    if export_mode != "none":
        import base64
        with open(os.path.join(out_dir, "points.b64.txt"), "w") as fh:     # artifacts serve text, not raw binary
            fh.write(base64.b64encode(b"".join(pts_blob)).decode())
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(dict(summary=summary, frames=per_frame), fh)
    return summary, per_frame


def _binary_iou(cm, pos):
    pos = list(pos)
    tp = cm[np.ix_(pos, pos)].sum()
    fp = cm[:, pos].sum() - tp
    fn = cm[pos, :].sum() - tp
    return float(tp / max(tp + fp + fn, 1))


def _nanmean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def _f(x):
    return None if not np.isfinite(x) else round(float(x), 4)
