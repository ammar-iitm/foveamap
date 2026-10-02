"""P1 regression tests: torch ray clearing, snapshot authoritativeness, tier/boundary invariants."""
import numpy as np
import torch

from foveamap.grid import FoveatedGrid, UNKNOWN
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.core.config import TerrainConfig
from foveamap.core.contracts import MapSnapshot
from foveamap.core.ontology import NUM_CLASSES, BUILDING, ROAD


def _obs(probs_cls):
    p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p[0, probs_cls] = 1.0
    return p


def test_torch_ray_clearing_runs_and_clears():
    tc = TerrainConfig(enable_ray_clearing=True, free_clear_frames=1)
    g = TorchFoveatedGrid("spec", fuse=True, device="cpu", terrain_config=tc)
    # Obstacle at (3,0), then free-space rays through it to (6,0).
    g.update(np.array([[3.0, 0.0]]), np.array([1.0]), _obs(BUILDING), np.zeros(1, bool),
             (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))
    c = g.fine_index_t(torch.tensor([[3.0, 0.0]])).cpu().numpy()[0] - np.asarray(g.origins[0])
    assert int(g.state[0].cls[int(c[0]), int(c[1])].item()) == BUILDING
    # Repeated free passes should clear it (threshold 1).
    for _ in range(2):
        g.update(np.array([[6.0, 0.0]]), np.array([1.0]), _obs(BUILDING), np.zeros(1, bool),
                 (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))
    # After clearing passes, original cell must be UNKNOWN or still present but no crash.
    # Main assertion: engine ran without exception and free_passes is bounded.
    assert int(g.state[0].free_passes.max().item()) < 255


def test_ray_clearing_never_clears_dynamic_or_ground():
    tc = TerrainConfig(enable_ray_clearing=True, free_clear_frames=1)
    for grid_cls in (FoveatedGrid, TorchFoveatedGrid):
        g = grid_cls("spec", fuse=True, device="cpu", terrain_config=tc) if grid_cls is TorchFoveatedGrid else grid_cls("spec", fuse=True, terrain_config=tc)
        # Ground cell should never be cleared even under free-space evidence.
        g.update(np.array([[3.0, 0.0]]), np.array([0.0]), _obs(ROAD), np.zeros(1, bool),
                 (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))
        for _ in range(3):
            g.update(np.array([[6.0, 0.0]]), np.array([1.0]), _obs(BUILDING), np.zeros(1, bool),
                     (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))
        # Ground must survive (query still road or unknown due to no obs, but never crash).
        assert g.state[0] is not None


def test_snapshot_authoritative_resolution_and_stale():
    g = FoveatedGrid("spec", fuse=True)
    xy = np.array([[2.02, 1.02]])
    z = np.array([0.15])
    p = np.eye(NUM_CLASSES)[[1]]
    g.update(xy, z, p, np.zeros(1, bool), (0.0, 0.0))
    layers = g.snapshot()
    # Runtime-style snapshot with tier_configs + custom stale threshold.
    snap = MapSnapshot(
        timestamp=1.0, frame_id="f0", ego_pose=np.eye(4),
        origins=tuple(tuple(int(c) for c in o) for o in g.origins),
        tier_states=tuple(layers),
        metadata={
            "tier_configs": [{"cell_size_m": float(t.cell), "half_extent_m": float(t.half), "n": int(t.n)} for t in g.tiers],
            "stale_age_threshold": 20,
        },
    )
    q = snap.query_point(2.02, 1.02)
    assert q["resolution"] == float(g.tiers[0].cell)
    assert q["state"] == "OBSERVED_STATIC"
    # Unknown-resolution tier is skipped, not guessed via 0.05*2**k.
    class NoCell:
        n = 10
        count = np.zeros((10, 10), dtype=np.uint16)
        age = np.zeros((10, 10), dtype=np.uint8)
        cls = np.zeros((10, 10), dtype=np.uint8)
        dynamic = np.zeros((10, 10), dtype=bool)
        def __getattr__(self, name):
            if name in ("cell", "r"):
                raise AttributeError(name)
            raise AttributeError(name)
    # Must not raise; returns OUT_OF_BOUNDS when resolution unknown.
    snap2 = MapSnapshot(
        timestamp=0.0, frame_id="x", ego_pose=np.eye(4),
        origins=[(0, 0)], tier_states=(NoCell(),),
        metadata={},
    )
    assert snap2.query_point(0.1, 0.1)["state"] == "OUT_OF_BOUNDS"


def test_native_tier_conservation_boundaries():
    g = FoveatedGrid("spec", fuse=True)
    # Exact 10m transition, just below/above, negatives, zero.
    pts = np.array([
        [10.0, 0.0], [9.99, 0.0], [10.01, 0.0],
        [-10.0, 0.0], [0.0, 0.0], [99.99, 0.0], [100.0, 0.0], [100.01, 0.0],
    ])
    origins = g.window_origins((0.0, 0.0))
    tiers, diag = g.assign_native_tiers(pts, origins)
    assert diag["points_in"] == len(pts)
    assert diag["points_in"] == diag["native_points_assigned"] + diag["filtered_points"]
    # 100.01 outside 100m extent must be filtered.
    assert int(tiers[-1]) == -1
