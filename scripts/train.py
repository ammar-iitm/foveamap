"""Train (or fine-tune) the range-image segmentation + motion network.

    # simulator, from scratch, CPU budget in seconds
    python scripts/train.py --dataset sim --budget 1000

    # nuScenes-mini, fine-tune from the simulator checkpoint (GPU if available)
    python scripts/train.py --dataset nuscenes --cache cache/nuscenes \
        --init checkpoints/range_unet.pt --out checkpoints/range_unet_nuscenes.pt --epochs 60
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap.sim import NUM_CLASSES  # noqa: E402
from foveamap.model import RangeUNet, predict, pick_device  # noqa: E402
from foveamap.frames import SIM_INFO, sim_frames, frames_to_training_arrays, make_features, prev_in_ego  # noqa: E402
from foveamap.pipeline import BANDS, BAND_NAMES, confusion, ious_from  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def load_frames(args, split):
    if args.dataset == "sim":
        pat = "train_*.npz" if split == "train" else "val_*.npz"
        frames = []
        for p in sorted(glob.glob(os.path.join(ROOT, "data", pat))):
            frames += sim_frames(p)[0]
        return frames, SIM_INFO
    from foveamap.nuscenes import load_cached, cached_info
    sp = {"train": args.train_split, "val": args.val_split}[split]
    return load_cached(args.cache, sp), cached_info(args.cache)


def evaluate(model, frames, info):
    """Per-point mIoU by distance band + moving IoU (every point, not just one per pixel)."""
    C = NUM_CLASSES
    cm = np.zeros((len(BANDS), C, C), np.int64)
    tp = un = 0
    for f in frames:
        feats, idx, row, col = make_features(f, info, prev_in_ego(f))
        probs, pm = predict(model, feats, info.active)
        pred = probs[row, col].argmax(1)
        mv = (pm[row, col] > 0.5) & np.isin(pred, (7, 8))
        lab = f["label"].astype(np.int64)
        has = lab >= 0
        d = np.hypot(f["pts"][:, 0], f["pts"][:, 1])
        for b, (lo, hi) in enumerate(BANDS):
            m = has & (d >= lo) & (d < hi)
            cm[b] += confusion(pred[m], lab[m])
        tp += np.sum(mv & f["moving"] & has); un += np.sum((mv | f["moving"]) & has)
    act = np.asarray(info.active, bool)
    all_iou = ious_from(cm.sum(0))
    band = [float(np.nanmean(ious_from(c)[act])) if c.sum() else None for c in cm]
    return dict(miou=float(np.nanmean(all_iou[act])), miou_by_band=band,
                iou_by_class={n: (None if not np.isfinite(v) else round(float(v), 4))
                              for n, v, a in zip(info.class_names, all_iou, act) if a},
                moving_iou=float(tp / un) if un else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim", choices=["sim", "nuscenes"])
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache", "nuscenes"))
    ap.add_argument("--train-split", default="mini_train")
    ap.add_argument("--val-split", default="mini_val")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=float(os.environ.get("TRAIN_BUDGET_S", 900)), help="seconds")
    ap.add_argument("--epochs", type=float, default=None, help="passes over the training frames (overrides --budget)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(ROOT, "checkpoints", "range_unet.pt" if args.dataset == "sim" else "range_unet_nuscenes.pt")

    dev = pick_device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    frames, info = load_frames(args, "train")
    X, Y, M, R = frames_to_training_arrays(frames, info)
    print(f"{args.dataset}: {len(X)} training frames, range image {info.n_rows} x {info.n_cols}, device {dev}", flush=True)

    valid = Y >= 0
    freq = np.bincount(Y[valid].astype(np.int64), minlength=NUM_CLASSES).astype(np.float64)
    cw = (freq.sum() / np.maximum(freq, 1)) ** 0.5
    cw[~np.asarray(info.active, bool) | (freq == 0)] = 0.0
    cw = torch.tensor(cw / cw[cw > 0].mean(), dtype=torch.float32, device=dev)
    pos_w = torch.tensor(min(50.0, (~M & valid).sum() / max(M.sum(), 1)) ** 0.5, device=dev)

    model = RangeUNet()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print("initialised from", args.init)
    model.to(dev)
    base_lr = args.lr or (1e-3 if args.init else 3e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    gpu = dev.type == "cuda"
    CROP, BATCH = (512, 16) if gpu else (256, 6)
    CROP = min(CROP, info.n_cols)
    steps_per_epoch = max(1, int(len(X) * info.n_cols / CROP / BATCH))
    total_steps = int(args.epochs * steps_per_epoch) if args.epochs else None
    scaler = torch.amp.GradScaler("cuda", enabled=gpu) if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=gpu)
    t0, step, hist = time.time(), 0, []

    def progress():
        return step / total_steps if total_steps else (time.time() - t0) / args.budget

    while progress() < 1.0:
        idx = rng.integers(0, len(X), BATCH)
        c0 = rng.integers(0, info.n_cols, BATCH)
        cols = (c0[:, None] + np.arange(CROP)[None]) % info.n_cols
        xb = torch.from_numpy(np.stack([X[i][:, :, c] for i, c in zip(idx, cols)]).astype(np.float32))
        yb = torch.from_numpy(np.stack([Y[i][:, c] for i, c in zip(idx, cols)]).astype(np.int64))
        mb = torch.from_numpy(np.stack([M[i][:, c] for i, c in zip(idx, cols)]).astype(np.float32))
        rb = torch.from_numpy(np.stack([R[i][:, c] for i, c in zip(idx, cols)]).astype(np.float32))
        if rng.random() < 0.5:                                   # mirror left/right
            xb = torch.flip(xb, [-1]); xb[:, 2] = -xb[:, 2]
            yb = torch.flip(yb, [-1]); mb = torch.flip(mb, [-1]); rb = torch.flip(rb, [-1])
        xb, yb, mb, rb = xb.to(dev), yb.to(dev), mb.to(dev), rb.to(dev)
        w = 1.0 + torch.clamp(rb / 25.0, 0, 3)                 # range-aware weighting (Arch. section 4)
        lr = base_lr * min(1.0, (step + 1) / 50) * max(0.05, 1 - progress())
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=gpu):
            sem, mot = model(xb)
        sem, mot = sem.float(), mot.float()
        vm = (yb >= 0).float()
        ce = F.cross_entropy(sem, yb, weight=cw, ignore_index=-1, reduction="none")
        l_sem = (ce * w * vm).sum() / (w * vm).sum().clamp_min(1)
        bce = F.binary_cross_entropy_with_logits(mot, mb, pos_weight=pos_w, reduction="none")
        l_mot = (bce * vm).sum() / vm.sum().clamp_min(1)
        loss = l_sem + 0.5 * l_mot
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        hist.append(float(loss.detach())); step += 1
        if step % 25 == 0:
            print(f"step {step:5d}  {time.time() - t0:6.0f}s  loss {np.mean(hist[-25:]):.3f}  lr {lr:.5f}", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(model.cpu().state_dict(), out)
    model.to(dev).eval()
    print("saved", out, f"({step} steps, {time.time() - t0:.0f}s)")

    vframes, _ = load_frames(args, "val")
    res = evaluate(model, vframes, info)
    print(f"val mIoU {res['miou']:.3f}  moving IoU {res['moving_iou']}")
    for n, v in zip(BAND_NAMES, res["miou_by_band"]):
        print(f"  {n:9s} mIoU {v}")
    for n, v in res["iou_by_class"].items():
        print(f"  {n:22s} IoU {v}")
    with open(out.replace(".pt", "_val.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
