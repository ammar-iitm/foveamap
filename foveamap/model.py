"""Segmentation + motion network.

Prototype choice: a range-image U-Net (the "low-power fallback" row in
Architecture Vision section 4). It is fully convolutional, so the same
weights run on a 64-beam simulated sensor and on nuScenes' 32-beam sensor.

Input (8 channels, H x W; built in frames.make_features): range, x, y, z,
intensity, valid mask, and two motion-residual images (current range vs. the
range of the sweeps ~0.1 s and ~0.2 s earlier, re-projected into the current
frame; LMNet-style motion cue).
Outputs: 9-class logits and a moving/static logit per pixel.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sim import NUM_CLASSES
from .frames import IN_CH


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class RangeUNet(nn.Module):
    def __init__(self, c=(16, 32, 64, 96)):
        super().__init__()
        self.e1 = cbr(IN_CH, c[0])
        self.e2 = cbr(c[0], c[1])
        self.e3 = nn.Sequential(cbr(c[1], c[2]), cbr(c[2], c[2]))
        self.e4 = nn.Sequential(cbr(c[2], c[3]), cbr(c[3], c[3]))
        self.d3 = cbr(c[3] + c[2], c[2])
        self.d2 = cbr(c[2] + c[1], c[1])
        self.d1 = cbr(c[1] + c[0] + 2, 16)       # + the 2 residual channels again (sharp motion edges)
        self.sem = nn.Conv2d(16, NUM_CLASSES, 1)
        self.mot = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)                                   # H x W
        e2 = self.e2(F.max_pool2d(e1, (1, 2)))            # H x W/2
        e3 = self.e3(F.max_pool2d(e2, 2))                 # H/2 x W/4
        e4 = self.e4(F.max_pool2d(e3, 2))                 # H/4 x W/8
        d3 = self.d3(torch.cat([F.interpolate(e4, size=e3.shape[-2:], mode="nearest"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1, x[:, 6:8]], 1))
        return self.sem(d1), self.mot(d1).squeeze(1)


def pick_device(pref: str | None = None):
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(ckpt, device=None):
    device = pick_device(device) if not isinstance(device, torch.device) else device
    m = RangeUNet()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def predict(model, feats, active=None, fp16=True, to_host=True):
    """feats: (8, H, W) numpy or tensor -> probs (H, W, C) numpy, p_move (H, W) numpy.
    active: boolean class mask; inactive classes are never predicted.
    to_host=False returns device tensors instead (no sync, no copy)."""
    dev = next(model.parameters()).device
    x = feats if torch.is_tensor(feats) else torch.from_numpy(np.asarray(feats, np.float32))
    x = x[None].to(dev, torch.float32)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(fp16 and dev.type == "cuda")):
        sem, mot = model(x)
    sem = sem.float()
    if active is not None and not np.all(active):
        sem[:, torch.from_numpy(~np.asarray(active, bool)).to(dev)] = -1e4
    probs = torch.softmax(sem[0], 0).permute(1, 2, 0)
    pm = torch.sigmoid(mot[0].float())
    if not to_host:
        return probs, pm
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return probs.cpu().numpy(), pm.cpu().numpy()
