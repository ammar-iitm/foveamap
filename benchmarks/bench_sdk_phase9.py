"""Phase 9 SDK overhead probe (diagnostic, CPU only, no system claims)."""
import time

import numpy as np
import torch

from foveamap.sdk import FoveaMap
from foveamap.runtime.perception import ClassicalFallbackBackend
from foveamap.core.contracts import LiDARFrame

backend = ClassicalFallbackBackend(device=torch.device("cpu"))
m = FoveaMap(perception_backend=backend)
m.configure()
m.start()

n = 2000
xs = np.linspace(1, 30, n)
pts = np.column_stack([xs, np.zeros(n), np.full(n, -1.7)]).astype(np.float32)
frame = LiDARFrame(
    pts=pts, intensity=np.ones(n, dtype=np.float32), ring=np.zeros(n, dtype=np.int16),
    pose=np.eye(4, dtype=np.float64), timestamp=0.0, frame_id="p", source_id="p")

# Warm up once (model paths, grid allocation).
m.process(frame)
m.reset()
reps = 20
t0 = time.perf_counter()
for _ in range(reps):
    m._runtime.process(frame)
t1 = time.perf_counter()
m.reset()
t2 = time.perf_counter()
for _ in range(reps):
    m.process(frame)
t3 = time.perf_counter()
view = m.snapshot()
t4 = time.perf_counter()
for _ in range(200):
    view.query_point(5.0, 0.0)
t5 = time.perf_counter()
import json as _json
t6 = time.perf_counter()
for _ in range(20):
    _json.dumps(view.to_dict())
t7 = time.perf_counter()

raw_ms = (t1 - t0) / reps * 1000.0
sdk_ms = (t3 - t2) / reps * 1000.0
print("raw_runtime_ms=%.2f sdk_process_ms=%.2f overhead_ms=%.3f" % (raw_ms, sdk_ms, sdk_ms - raw_ms))
print("query200_ms=%.1f snap_todict20_ms=%.1f" % ((t5 - t4) * 1000.0, (t7 - t6) * 1000.0))
print("device:", m.status().device, "| cuda:", m.metrics().cuda_note)
