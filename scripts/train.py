"""Train (or fine-tune) the range-image segmentation + motion network.

    # simulator, from scratch, CPU budget in seconds
    python scripts/train.py --dataset sim --budget 1000

    # nuScenes-mini, fine-tune from the simulator checkpoint (GPU if available)
    python scripts/train.py --dataset nuscenes --cache cache/nuscenes \
        --init checkpoints/range_unet.pt --out checkpoints/range_unet_nuscenes.pt --epochs 60
    # recipe options: --reset-head (fresh class layer), --balance (rare-class frames), --aug

    # SemanticKITTI (cache from scripts/prepare_semantickitti.py), fine-tune from the simulator
    python scripts/train.py --dataset semantickitti --init checkpoints/range_unet.pt --epochs 30
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
    """(frames, info, n). SemanticKITTI frames are a generator (one sequence in memory at a time)."""
    if args.dataset == "sim":
        pat = "train_*.npz" if split == "train" else "val_*.npz"
        frames = []
        for p in sorted(glob.glob(os.path.join(ROOT, "data", pat))):
            frames += sim_frames(p)[0]
        return frames, SIM_INFO, len(frames)
    sp = {"train": args.train_split, "val": args.val_split}[split]
    if args.dataset == "semantickitti":
        from foveamap.semantickitti import iter_cached, count_cached, cached_info
        return iter_cached(args.cache, sp), cached_info(args.cache), count_cached(args.cache, sp)
    from foveamap.nuscenes import load_cached, cached_info
    frames = load_cached(args.cache, sp)
    return frames, cached_info(args.cache), len(frames)


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
    total = cm.sum(0)
    all_iou = ious_from(total)
    band = [float(np.nanmean(ious_from(c)[act])) if c.sum() else None for c in cm]
    return dict(miou=float(np.nanmean(all_iou[act])), miou_by_band=band,
                iou_by_class={n: (None if not np.isfinite(v) else round(float(v), 4))
                              for n, v, a in zip(info.class_names, all_iou, act) if a},
                moving_iou=float(tp / un) if un else None,
                points_by_class={n: int(c) for n, c, a in zip(info.class_names, total.sum(1), act) if a},
                confusion=total.tolist())          # rows = label, cols = prediction, all bands


def label_stats(Y, M, n_classes, chunk=128):
    """(labelled pixels per class, moving pixels, static labelled pixels), read a chunk of frames at a
    time so memory-mapped arrays never need a full-size temporary."""
    freq, n_mov, n_static = np.zeros(n_classes), 0, 0
    for i in range(0, len(Y), chunk):
        y, m = np.asarray(Y[i:i + chunk]), np.asarray(M[i:i + chunk])
        v = y >= 0
        freq += np.bincount(y[v].astype(np.int64), minlength=n_classes)
        n_mov += int(m.sum())
        n_static += int((~m & v).sum())
    return freq, n_mov, n_static


def balanced_frame_probs(Y, class_w):
    """Sampling probability per frame, proportional to its class-weighted labelled-pixel mass."""
    mass = np.array([class_w[y[y >= 0].astype(np.int64)].sum() for y in Y], np.float64)
    return mass / mass.sum()


def augment(xb, rb):
    """Random per-sample scale (+/-5%) of range and xyz, and intensity jitter (+/-20%) on valid pixels.
    xb: (B, 8, H, W) features (range, x, y, z, intensity, valid, res1, res2); rb: (B, H, W) range."""
    B = xb.shape[0]
    s = torch.empty(B, 1, 1).uniform_(0.95, 1.05)
    xb = xb.clone()
    xb[:, 0:4] = xb[:, 0:4] * s[:, None]
    xb[:, 4] = (xb[:, 4] * torch.empty(B, 1, 1).uniform_(0.8, 1.2)).clamp(0, 1) * xb[:, 5]
    return xb, rb * s


def top_confusions(cm, names, active, k=2):
    """For each class: recall and the classes its points are most often predicted as."""
    cm = np.asarray(cm, np.float64)
    out = {}
    for i, (n, a) in enumerate(zip(names, active)):
        tot = cm[i].sum()
        if not a or tot == 0:
            continue
        others = [(names[j], cm[i, j] / tot) for j in np.argsort(-cm[i]) if j != i and cm[i, j] > 0][:k]
        out[n] = dict(recall=round(float(cm[i, i] / tot), 4), predicted_as=[(m, round(float(f), 4)) for m, f in others])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sim", choices=["sim", "nuscenes", "semantickitti"])
    ap.add_argument("--cache", default=None, help="frame cache (default: cache/<dataset>)")
    ap.add_argument("--train-split", default=None, help="default: mini_train (nuScenes), train (SemanticKITTI)")
    ap.add_argument("--val-split", default=None, help="default: mini_val (nuScenes), val (SemanticKITTI)")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=float(os.environ.get("TRAIN_BUDGET_S", 900)), help="seconds")
    ap.add_argument("--epochs", type=float, default=None, help="passes over the training frames (overrides --budget)")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--reset-head", action="store_true",
                    help="with --init: re-initialise the class layer (the dataset's classes differ from the checkpoint's)")
    ap.add_argument("--balance", action="store_true",
                    help="sample frames in proportion to their class-weighted pixel mass (more rare-class frames)")
    ap.add_argument("--aug", action="store_true", help="random scale (+/-5%%) and intensity jitter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arrays", default=None,
                    help="keep the training arrays in memory-mapped .npy files in this directory instead of RAM "
                         "(needed for stride-5 SemanticKITTI on a 12.7 GB runtime)")
    args = ap.parse_args()
    args.cache = args.cache or os.path.join(ROOT, "cache", args.dataset)
    kitti = args.dataset == "semantickitti"
    args.train_split = args.train_split or ("train" if kitti else "mini_train")
    args.val_split = args.val_split or ("val" if kitti else "mini_val")
    out = args.out or os.path.join(ROOT, "checkpoints", "range_unet.pt" if args.dataset == "sim"
                                   else f"range_unet_{args.dataset}.pt")

    dev = pick_device(args.device)
    if dev.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    frames, info, n_frames = load_frames(args, "train")
    print(f"preparing {n_frames} training frames", flush=True)
    X, Y, M, R = frames_to_training_arrays(frames, info, n_frames, log=lambda m: print(m, flush=True),
                                           out_dir=args.arrays)
    del frames
    print(f"{args.dataset}: {len(X)} training frames, range image {info.n_rows} x {info.n_cols}, device {dev}", flush=True)

    freq, n_mov, n_static = label_stats(Y, M, NUM_CLASSES)
    print("training pixels by class:", ", ".join(f"{n} {100 * c / freq.sum():.2f}%"
                                                  for n, c, a in zip(info.class_names, freq, info.active) if a))
    cw = (freq.sum() / np.maximum(freq, 1)) ** 0.5
    cw[~np.asarray(info.active, bool) | (freq == 0)] = 0.0
    cw = torch.tensor(cw / cw[cw > 0].mean(), dtype=torch.float32, device=dev)
    pos_w = torch.tensor(min(50.0, n_static / max(n_mov, 1)) ** 0.5, device=dev)

    model = RangeUNet()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print("initialised from", args.init)
        if args.reset_head:
            model.sem.reset_parameters()
            print("class layer re-initialised")

    frame_p = None
    if args.balance:                                  # frames rich in rare classes are drawn more often
        frame_p = balanced_frame_probs(Y, cw.cpu().numpy())
        print(f"balanced sampling: frame weights {frame_p.min() * len(Y):.2f}x to {frame_p.max() * len(Y):.2f}x of uniform")
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
        idx = rng.choice(len(X), BATCH, p=frame_p) if frame_p is not None else rng.integers(0, len(X), BATCH)
        c0 = rng.integers(0, info.n_cols, BATCH)
        cols = (c0[:, None] + np.arange(CROP)[None]) % info.n_cols
        xb = torch.from_numpy(np.stack([X[i][:, :, c] for i, c in zip(idx, cols)]).astype(np.float32))
        yb = torch.from_numpy(np.stack([Y[i][:, c] for i, c in zip(idx, cols)]).astype(np.int64))
        mb = torch.from_numpy(np.stack([M[i][:, c] for i, c in zip(idx, cols)]).astype(np.float32))
        rb = torch.from_numpy(np.stack([R[i][:, c] for i, c in zip(idx, cols)]).astype(np.float32))
        if rng.random() < 0.5:                                   # mirror left/right
            xb = torch.flip(xb, [-1]); xb[:, 2] = -xb[:, 2]
            yb = torch.flip(yb, [-1]); mb = torch.flip(mb, [-1]); rb = torch.flip(rb, [-1])
        if args.aug:
            xb, rb = augment(xb, rb)
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
    del X, Y, M, R                                    # free the training arrays (~2.5 GB at stride 10) before validation

    vframes, _, _ = load_frames(args, "val")
    res = evaluate(model, vframes, info)
    print(f"val mIoU {res['miou']:.3f}  moving IoU {res['moving_iou']}")
    for n, v in zip(BAND_NAMES, res["miou_by_band"]):
        print(f"  {n:9s} mIoU {v}")
    res["top_confusions"] = top_confusions(res["confusion"], info.class_names, info.active)
    for n, v in res["iou_by_class"].items():
        tc = res["top_confusions"].get(n, {})
        conf = ", ".join(f"{m} {100 * f:.0f}%" for m, f in tc.get("predicted_as", []))
        print(f"  {n:22s} IoU {v}  recall {tc.get('recall')}  points {res['points_by_class'][n]:8d}  confused with: {conf}")
    with open(out.replace(".pt", "_val.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
