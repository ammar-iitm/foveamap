"""Phase 8 CPU performance probe (diagnostic, not a benchmark claim)."""
import time

import numpy as np

from foveamap_ros.config import from_ros_params
from foveamap_ros.node import FoveaMapNodeCore
from foveamap_ros.pointcloud import (
    FLOAT32, RosHeader, RosPointCloud2, RosPointField, RosStamp,
)

cfg = from_ros_params({"perception.backend_type": "classical", "runtime.grid_engine": "numpy"})
core = FoveaMapNodeCore(config=cfg)
core.configure()
core.activate()

rng = np.random.default_rng(0)
n = 20000
xyz = np.column_stack([rng.uniform(1, 60, n), rng.uniform(-20, 20, n),
                       rng.normal(-1.0, 0.5, n)]).astype(np.float32)
msg = RosPointCloud2(
    fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
            RosPointField("z", 8, FLOAT32)],
    height=1, width=n, point_step=12, row_step=12 * n, data=xyz.tobytes(),
    header=RosHeader(RosStamp(100, 0), "base_link"))

t0 = time.perf_counter()
frame = core._convert(msg)
t1 = time.perf_counter()
snap = core.runtime.process(frame)
t2 = time.perf_counter()
core.last_snapshot = snap
out = core._publish(snap, frame)
t3 = time.perf_counter()
print("points:", n)
print("convert_ms=%.1f" % ((t1 - t0) * 1000.0))
print("runtime_ms=%.1f" % ((t2 - t1) * 1000.0))
print("serialize_ms=%.1f" % ((t3 - t2) * 1000.0))
print("grid_bytes:", out["/foveamap/grid"]["serialized_size_bytes"])
print("points_bytes:", len(out["/foveamap/points_labeled"].data))
