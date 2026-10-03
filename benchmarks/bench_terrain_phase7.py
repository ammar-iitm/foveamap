import time

import numpy as np

from foveamap.grid import FoveatedGrid
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.core.ontology import NUM_CLASSES, ROAD

rng = np.random.default_rng(0)
n = 5000
xy = rng.uniform(-30, 30, (n, 2))
z = (rng.normal(-1.7, 0.3, n)).astype(np.float32)
p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
p[:, ROAD] = 1.0
m = np.zeros(n, bool)

for name, g in (("numpy", FoveatedGrid("spec")), ("torch", TorchFoveatedGrid("spec", device="cpu"))):
    g.update(xy, z, p, m, (0.0, 0.0))
    t0 = time.perf_counter()
    rep = g.terrain_report()
    t1 = time.perf_counter()
    t0q = time.perf_counter()
    for k in range(200):
        g.query_point(float(xy[k, 0]), float(xy[k, 1]))
    t1q = time.perf_counter()
    mem = g.memory_report()
    total = rep["total"]
    print(name, "report_ms=%.1f" % ((t1 - t0) * 1000.0), "known=", total["known_ground"],
          "trav=", total["traversable_cells"], "persistent=", mem["allocated_bytes"],
          "aux=", mem["auxiliary_bytes"], "q200_ms=%.1f" % ((t1q - t0q) * 1000.0))
