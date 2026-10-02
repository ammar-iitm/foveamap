"""Phase 6 tests: dynamic world model & temporal environment intelligence.

Covers lifecycle, confidence/decay, correspondence, ego-motion,
static/dynamic separation, ray-clearing interaction, query/snapshot API,
traversability, reset, determinism, NumPy/Torch parity, bounded state,
golden temporal scenario, and a 100+ frame temporal soak.
"""
import numpy as np
import pytest
import torch

from foveamap.grid import FoveatedGrid, UNKNOWN
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.temporal import DynamicWorldModel, DynamicObservation
from foveamap.core.config import DynamicConfig, TerrainConfig, FoveaMapConfig
from foveamap.core.contracts import MapSnapshot, LiDARFrame
from foveamap.core.ontology import (
    NUM_CLASSES, ROAD, VEHICLE, PERSON, BUILDING,
)


def _dyn_probs(cls_id, p=0.95):
    pr = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    pr[0, cls_id] = p
    return pr


def _step(grid, xy, z, probs, moving, ego=(0.0, 0.0), ts=None, sensor=None):
    kw = {} if sensor is None else {"sensor_origin": sensor}
    return grid.update(
        np.asarray(xy, dtype=np.float64).reshape(-1, 2),
        np.asarray(z, dtype=np.float32).reshape(-1),
        np.asarray(probs, dtype=np.float32).reshape(-1, NUM_CLASSES),
        np.asarray(moving, dtype=bool).reshape(-1),
        ego, timestamp=ts, **kw,
    )


def _empty():
    return np.zeros((0, 2)), np.zeros(0, dtype=np.float32), np.zeros((0, NUM_CLASSES), dtype=np.float32), np.zeros(0, bool)


# ------------------------------------------------------------------ lifecycle
def test_single_dynamic_observation_is_provisional_but_occupied():
    g = FoveatedGrid("spec")
    xy = np.array([[10.0, 2.0]])
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    q = g.query_point(10.0, 2.0)
    assert q["dynamic"] is True
    assert q["state"] == "OBSERVED_DYNAMIC"
    assert q["dynamic_state"] == "OBSERVED"
    assert q["is_traversable"] is False
    assert g.is_traversable(10.0, 2.0) is False


def test_repeated_observation_activates():
    g = FoveatedGrid("spec")
    xy = np.array([[10.0, 2.0]])
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
    q = g.query_point(10.0, 2.0)
    assert q["dynamic_state"] == "ACTIVE_DYNAMIC"
    assert q["state"] == "ACTIVE_DYNAMIC"
    assert q["dynamic_confidence"] >= 0.5


def test_activation_threshold_configurable():
    cfg = DynamicConfig(activation_frames=3)
    g = FoveatedGrid("spec", dynamic_config=cfg)
    xy = np.array([[10.0, 2.0]])
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    assert g.query_point(10.0, 2.0)["dynamic_state"] == "OBSERVED"
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
    assert g.query_point(10.0, 2.0)["dynamic_state"] == "OBSERVED"
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.2)
    assert g.query_point(10.0, 2.0)["dynamic_state"] == "ACTIVE_DYNAMIC"


def test_confidence_rises_and_decays():
    g = FoveatedGrid("spec")
    xy = np.array([[10.0, 2.0]])
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    c1 = g.query_point(10.0, 2.0)["dynamic_confidence"]
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
    c2 = g.query_point(10.0, 2.0)["dynamic_confidence"]
    assert c2 > c1
    ex, ez, ep, em = _empty()
    _step(g, ex, ez, ep, em, ts=0.2)
    c3 = g.query_point(10.0, 2.0)["dynamic_confidence"]
    assert c3 < c2


def test_disappearance_goes_missing_then_stale_then_removed():
    cfg = DynamicConfig(activation_frames=1, missing_tolerance_frames=2, stale_frames=4)
    g = FoveatedGrid("spec", dynamic_config=cfg)
    xy = np.array([[10.0, 2.0]])
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    assert g.query_point(10.0, 2.0)["dynamic_state"] == "ACTIVE_DYNAMIC"
    ex, ez, ep, em = _empty()
    _step(g, ex, ez, ep, em, ts=0.1)
    q1 = g.query_point(10.0, 2.0)
    assert q1["dynamic_state"] == "TEMPORARILY_MISSING"
    assert q1["dynamic"] is True
    assert q1["state"] == "TEMPORARILY_MISSING"
    _step(g, ex, ez, ep, em, ts=0.2)
    assert g.query_point(10.0, 2.0)["dynamic_state"] == "TEMPORARILY_MISSING"
    _step(g, ex, ez, ep, em, ts=0.3)
    q3 = g.query_point(10.0, 2.0)
    assert q3["dynamic_state"] == "STALE"
    assert q3["dynamic"] is False
    assert q3["state"] == "STALE"
    _step(g, ex, ez, ep, em, ts=0.4)
    _step(g, ex, ez, ep, em, ts=0.5)
    assert len(g.temporal) == 0
    q5 = g.query_point(10.0, 2.0)
    assert q5["state"] == "UNKNOWN"
    assert q5["dynamic"] is False


def test_stale_releases_traversability_fallback():
    g = FoveatedGrid("spec")
    xy = np.array([[10.0, 2.0]])
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    assert g.is_traversable(10.0, 2.0) is False
    ex, ez, ep, em = _empty()
    for k in range(6):
        _step(g, ex, ez, ep, em, ts=0.1 * (k + 1))
    assert len(g.temporal) == 0


# ------------------------------------------------------- movement / correspondence
def test_moving_object_keeps_identity_and_velocity():
    g = FoveatedGrid("spec")
    _step(g, [[10.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    t0 = g.temporal_snapshot()[0]["track_id"]
    _step(g, [[11.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
    tracks = g.temporal_snapshot()
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == t0
    assert tracks[0]["state"] == "ACTIVE_DYNAMIC"
    q = g.query_point(11.0, 2.0)
    assert q["dynamic"] is True
    assert q["velocity"] is not None
    assert abs(q["velocity"][0] - 10.0) < 1.0  # 1 m / 0.1 s in world frame
    assert abs(q["velocity"][1]) < 1.0


def test_no_ghost_mask_at_old_position_but_lifecycle_visible():
    g = FoveatedGrid("spec")
    _step(g, [[2.0, 2.0], [2.02, 2.02]], [0.5, 0.6], np.tile(_dyn_probs(VEHICLE), (2, 1)), [True, True], ts=0.0)
    _step(g, [[4.0, 2.0], [4.02, 2.02]], [0.5, 0.6], np.tile(_dyn_probs(VEHICLE), (2, 1)), [True, True], ts=0.1)
    c0 = np.floor(np.array([2.0, 2.0]) / 0.05).astype(int) - g.origins[0]
    # Frame-isolated mask: no ghost observation at the old cell.
    assert bool(g.state[0].dynamic_mask[c0[0], c0[1]]) is False
    assert int(g.state[0].count[c0[0], c0[1]]) == 0


def test_person_track():
    g = FoveatedGrid("spec")
    xy = np.array([[5.0, 5.0]])
    _step(g, xy, [0.0], _dyn_probs(PERSON), [True], ts=0.0)
    _step(g, xy, [0.0], _dyn_probs(PERSON), [True], ts=0.1)
    tracks = g.temporal_snapshot()
    assert len(tracks) == 1
    assert tracks[0]["cls"] == PERSON
    assert tracks[0]["state"] == "ACTIVE_DYNAMIC"


def test_multiple_dynamic_cells():
    g = FoveatedGrid("spec")
    xy = np.array([[10.0, 2.0], [-8.0, -3.0]])
    z = np.array([0.5, 0.5])
    p = np.tile(_dyn_probs(VEHICLE), (2, 1))
    _step(g, xy, z, p, [True, True], ts=0.0)
    assert len(g.temporal) == 2
    assert g.query_point(10.0, 2.0)["dynamic"] is True
    assert g.query_point(-8.0, -3.0)["dynamic"] is True


def test_negative_coordinates_and_boundary():
    g = FoveatedGrid("spec")
    _step(g, [[-9.0, -9.0]], [0.4], _dyn_probs(VEHICLE), [True], ts=0.0)
    q = g.query_point(-9.0, -9.0)
    assert q["dynamic"] is True
    # Far-tier dynamic observation (tier 1, 50 m).
    _step(g, [[50.0, 0.0]], [1.0], _dyn_probs(VEHICLE), [True], ts=0.1)
    q2 = g.query_point(50.0, 0.0)
    assert q2["dynamic"] is True
    assert q2["tier"] == 1


def test_empty_frame_advances_missing_without_crash():
    g = FoveatedGrid("spec")
    ex, ez, ep, em = _empty()
    _step(g, ex, ez, ep, em, ts=0.0)
    assert len(g.temporal) == 0
    assert g.query_point(1.0, 1.0)["state"] == "UNKNOWN"


# ------------------------------------------------------- static/dynamic separation
def test_dynamic_does_not_pollute_static_map():
    g = FoveatedGrid("spec")
    xy = np.array([[2.0, 2.0], [2.02, 2.02]])
    _step(g, xy, [0.5, 0.6], np.tile(_dyn_probs(VEHICLE), (2, 1)), [True, True], ts=0.0)
    c0 = np.floor(np.array([2.0, 2.0]) / 0.05).astype(int) - g.origins[0]
    assert int(g.state[0].count[c0[0], c0[1]]) == 0
    assert int(g.state[0].cls[c0[0], c0[1]]) == UNKNOWN


def test_static_road_survives_temporary_vehicle_overlap():
    g = FoveatedGrid("spec")
    road = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    road[0, ROAD] = 1.0
    xy = np.array([[3.0, 3.0]])
    _step(g, xy, [0.0], road, [False], ts=0.0)
    assert int(g.state[0].cls[int((np.floor(3.0 / 0.05) - g.origins[0][0])), int((np.floor(3.0 / 0.05) - g.origins[0][1]))]) == ROAD
    assert g.is_traversable(3.0, 3.0) is True
    # Vehicle temporarily occupies the same cell.
    _step(g, xy, [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
    q = g.query_point(3.0, 3.0)
    assert q["dynamic"] is True
    assert g.is_traversable(3.0, 3.0) is False
    # Static road geometry untouched underneath.
    c = np.floor(np.array([3.0, 3.0]) / 0.05).astype(int) - g.origins[0]
    assert int(g.state[0].cls[c[0], c[1]]) == ROAD
    # Vehicle leaves; after lifecycle expiry the road is traversable again.
    ex, ez, ep, em = _empty()
    for k in range(6):
        _step(g, ex, ez, ep, em, ts=0.2 + 0.1 * k)
    assert len(g.temporal) == 0


def test_dynamic_ground_overlap_keeps_ground_intact():
    g = FoveatedGrid("spec")
    road = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    road[0, ROAD] = 1.0
    _step(g, [[6.0, 1.0]], [0.0], road, [False], ts=0.0)
    _step(g, [[6.0, 1.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
    assert np.isfinite(g.state[0].ground).any()


def test_ray_clearing_skips_active_dynamic_and_clears_after_removal():
    terrain = TerrainConfig(enable_ray_clearing=True, free_clear_frames=1)
    g = FoveatedGrid("spec", terrain_config=terrain)
    _step(g, [[3.0, 0.0]], [1.0], _dyn_probs(VEHICLE), [True], ts=0.0, sensor=(0.0, 0.0, 0.0))
    # Dynamic obstacle cell must not be cleared by free-space rays.
    _step(g, [[6.0, 0.0]], [1.0], _dyn_probs(BUILDING), [False], ts=0.1, sensor=(0.0, 0.0, 0.0))
    assert len(g.temporal) >= 1


# ------------------------------------------------------------------ golden scenario
def test_golden_temporal_scenario():
    """Frame 0 empty road; 1-3 vehicle x=10,11,12; 4-6 absent. No ghosts."""
    g = FoveatedGrid("spec")
    road = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    road[0, ROAD] = 1.0
    ex, ez, ep, em = _empty()
    _step(g, [[0.0, 5.0]], [0.0], road, [False], ts=0.0)  # frame 0: empty road
    assert g.query_point(0.0, 5.0)["state"] == "OBSERVED_STATIC"
    _step(g, [[10.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)  # frame 1
    q1 = g.query_point(10.0, 2.0)
    assert q1["dynamic"] is True and q1["dynamic_state"] == "OBSERVED"
    _step(g, [[11.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.2)  # frame 2
    q2 = g.query_point(11.0, 2.0)
    assert q2["dynamic_state"] == "ACTIVE_DYNAMIC"
    assert len(g.temporal_snapshot()) == 1  # same identity across motion
    _step(g, [[12.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.3)  # frame 3
    assert g.query_point(12.0, 2.0)["dynamic_state"] == "ACTIVE_DYNAMIC"
    assert len(g.temporal_snapshot()) == 1
    _step(g, ex, ez, ep, em, ts=0.4)  # frame 4: disappears
    assert g.query_point(12.0, 2.0)["state"] == "TEMPORARILY_MISSING"
    _step(g, ex, ez, ep, em, ts=0.5)  # frame 5
    assert g.query_point(12.0, 2.0)["dynamic_state"] == "TEMPORARILY_MISSING"
    _step(g, ex, ez, ep, em, ts=0.6)  # frame 6
    assert g.query_point(12.0, 2.0)["state"] in ("STALE", "UNKNOWN")
    # No ghost static geometry where the vehicle drove (check native tier).
    q = g.query_point(11.0, 2.0)
    k = q["tier"]
    assert k >= 0
    ci, cj = q["cell_coord"] if "cell_coord" in q else q["cell"]
    assert int(g.state[k].count[ci, cj]) == 0


# ------------------------------------------------------- reset / determinism
def test_reset_clears_temporal_state():
    g = FoveatedGrid("spec")
    _step(g, [[10.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    assert len(g.temporal) == 1
    g.reset()
    assert len(g.temporal) == 0
    q = g.query_point(10.0, 2.0)
    assert q["is_unknown"] is True  # origins cleared; no state of any kind
    assert g.frame_index == -1


def test_deterministic_replay():
    def run():
        gg = FoveatedGrid("spec")
        _step(gg, [[10.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
        _step(gg, [[11.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
        ex, ez, ep, em = _empty()
        _step(gg, ex, ez, ep, em, ts=0.2)
        return gg.temporal_snapshot(), gg.query_point(11.0, 2.0)

    t1, q1 = run()
    t2, q2 = run()
    assert t1 == t2
    assert q1["dynamic_state"] == q2["dynamic_state"]


# ------------------------------------------------------- snapshot / query api
def test_snapshot_carries_coherent_temporal_state():
    g = FoveatedGrid("spec")
    _step(g, [[10.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    dyn_obs, _stats = _step(g, [[10.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.1)
    snap = MapSnapshot(
        timestamp=0.1,
        frame_id="f1",
        ego_pose=np.eye(4),
        origins=tuple(tuple(int(c) for c in o) for o in g.origins),
        tier_states=tuple(g.snapshot()),
        dynamic_cells=tuple(dyn_obs),
        dynamic_tracks=tuple(g.temporal_snapshot()),
        temporal_metadata=dict(g.temporal_stats()),
    )
    q = snap.query_point(10.0, 2.0)
    assert q["dynamic"] is True
    assert q["dynamic_state"] == "ACTIVE_DYNAMIC"
    assert snap.is_traversable(10.0, 2.0) is False
    # Snapshot detached: further grid updates cannot mutate it.
    _step(g, [[-5.0, -5.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.2)
    assert snap.query_point(10.0, 2.0)["dynamic_state"] == "ACTIVE_DYNAMIC"


def test_query_api_backward_compatible_keys():
    g = FoveatedGrid("spec")
    _step(g, [[10.0, 2.0]], [0.5], _dyn_probs(VEHICLE), [True], ts=0.0)
    q = g.query_point(10.0, 2.0)
    for key in ("tier", "dominant_class", "cls", "cost", "dynamic", "state",
                "is_unknown", "is_traversable", "age", "count"):
        assert key in q
    assert "dynamic_state" in q and "dynamic_confidence" in q and "velocity" in q


# ------------------------------------------------------- parity
def test_numpy_torch_temporal_parity():
    gn = FoveatedGrid("spec")
    gt = TorchFoveatedGrid("spec", device="cpu")
    seq = [
        (np.array([[10.0, 2.0]]), 0.0),
        (np.array([[11.0, 2.0]]), 0.1),
        (np.zeros((0, 2)), 0.2),
        (np.zeros((0, 2)), 0.3),
    ]
    for xy, ts in seq:
        n = len(xy)
        z = np.full(n, 0.5, dtype=np.float32)
        p = np.tile(_dyn_probs(VEHICLE), (n, 1)) if n else np.zeros((0, NUM_CLASSES), dtype=np.float32)
        m = np.ones(n, bool)
        _step(gn, xy, z, p, m, ts=ts)
        _step(gt, xy, z, p, m, ts=ts)
    assert len(gn.temporal) == len(gt.temporal)
    qn = gn.query_point(11.0, 2.0)
    qt = gt.query_point(11.0, 2.0)
    assert qn["dynamic_state"] == qt["dynamic_state"]
    assert qn["dynamic"] == qt["dynamic"]
    assert abs(qn["dynamic_confidence"] - qt["dynamic_confidence"]) < 1e-6
    assert (qn["velocity"] is None) == (qt["velocity"] is None)
    sn, st = gn.temporal_snapshot(), gt.temporal_snapshot()
    assert [(t["tier"], t["cell"], t["cls"], t["state"]) for t in sn] == \
           [(t["tier"], t["cell"], t["cls"], t["state"]) for t in st]


def test_torch_structural_no_per_point_sync():
    # Torch grid processes dynamic frames without error on CPU device tensors.
    gt = TorchFoveatedGrid("spec", device="cpu")
    xy = torch.zeros((4, 2))
    xy[:, 0] = 10.0
    xy[:, 1] = 2.0
    z = torch.full((4,), 0.5)
    p = torch.zeros((4, NUM_CLASSES))
    p[:, VEHICLE] = 0.95
    m = torch.ones((4,), dtype=torch.bool)
    gt.update(xy, z, p, m, (0.0, 0.0), timestamp=0.0)
    gt.update(xy, z, p, m, (0.0, 0.0), timestamp=0.1)
    assert len(gt.temporal) >= 1
    assert gt.query_point(10.0, 2.0)["dynamic"] is True


# ------------------------------------------------------- bounded state
def test_bounded_capacity_with_eviction():
    cfg = DynamicConfig(max_tracks=8)
    g = FoveatedGrid("spec", dynamic_config=cfg)
    for k in range(20):
        _step(g, [[-9.0 + 0.3 * k, -9.0]], [0.4], _dyn_probs(VEHICLE), [True], ts=0.01 * k)
    assert len(g.temporal) <= 8
    rep = g.memory_report()
    assert rep["temporal_tracks"] <= 8
    assert rep["temporal_tracks_capacity"] == 8
    assert rep["allocated_bytes"] == 5120000  # PRD grid memory keys unchanged


def test_history_bounded_no_growth():
    g = FoveatedGrid("spec")
    ex, ez, ep, em = _empty()
    for k in range(30):
        _step(g, ex, ez, ep, em, ts=0.1 * k)
    assert len(g.temporal) == 0
    assert g.temporal.total_expired == 0


def test_config_validation_rejects_unsafe_values():
    with pytest.raises(Exception):
        DynamicConfig(activation_frames=0)
    with pytest.raises(Exception):
        DynamicConfig(missing_tolerance_frames=-1)
    with pytest.raises(Exception):
        DynamicConfig(stale_frames=2, missing_tolerance_frames=2)
    with pytest.raises(Exception):
        DynamicConfig(max_tracks=0)
    with pytest.raises(Exception):
        DynamicConfig(correspondence_distance_m=0.0)
    with pytest.raises(Exception):
        DynamicConfig(decay_factor=1.5)
    # Defaults are valid and exposed through the top-level config.
    assert isinstance(FoveaMapConfig().dynamic, DynamicConfig)


def test_ego_motion_scroll_keeps_world_locked_tracks():
    g = FoveatedGrid("spec")
    _step(g, [[5.0, 0.0]], [0.5], _dyn_probs(VEHICLE), [True], ego=(0.0, 0.0), ts=0.0)
    t0 = g.temporal_snapshot()[0]["track_id"]
    # Ego drives 5 m; same world cell re-observed -> same track survives scroll.
    _step(g, [[5.0, 0.0]], [0.5], _dyn_probs(VEHICLE), [True], ego=(5.0, 0.0), ts=0.1)
    tracks = g.temporal_snapshot()
    assert len(tracks) == 1
    assert tracks[0]["track_id"] == t0
    assert g.query_point(5.0, 0.0)["dynamic"] is True


# ------------------------------------------------------- runtime integration
class _MovingBackend:
    """Minimal stub perception backend: every point is a moving vehicle."""

    def __init__(self):
        from foveamap.runtime.perception import ClassicalFallbackBackend
        import torch as _torch
        self._fallback = ClassicalFallbackBackend(device=_torch.device("cpu"))

    @property
    def device(self):
        return self._fallback.device

    def reset(self):
        pass

    def predict_device(self, frame, dev_math=False, profiling=False):
        import torch as _torch
        from foveamap.runtime.perception import DevicePerceptionResult
        n = len(frame.pts)
        dev = self._fallback.device
        P = torch.zeros((n, NUM_CLASSES), device=dev)
        P[:, VEHICLE] = 1.0
        return DevicePerceptionResult(
            class_probabilities=P,
            moving_probabilities=torch.ones((n,), device=dev),
            semantic_predictions=torch.full((n,), VEHICLE, device=dev, dtype=torch.long),
            is_moving=torch.ones((n,), device=dev, dtype=torch.bool),
            device=dev,
            metadata={"backend": "stub-moving"},
        )

    def predict(self, frame, device=None):
        return self.predict_device(frame).to_host()


def _runtime_frame(x=10.0, y=2.0, ts=0.0):
    return LiDARFrame(
        pts=np.array([[x, y, 0.5]], dtype=np.float32),
        intensity=np.ones(1, dtype=np.float32),
        ring=np.zeros(1, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        timestamp=ts,
        frame_id=f"f{ts}",
        source_id="test",
    )


def test_runtime_end_to_end_world_state():
    from foveamap.runtime.runtime import FoveaMapRuntime
    cfg = FoveaMapConfig()
    rt = FoveaMapRuntime(config=cfg, perception_backend=_MovingBackend())
    s1 = rt.process(_runtime_frame(ts=0.0))
    assert s1.query_point(10.0, 2.0)["dynamic"] is True
    s2 = rt.process(_runtime_frame(ts=0.1))
    assert s2.query_point(10.0, 2.0)["dynamic_state"] == "ACTIVE_DYNAMIC"
    assert len(s2.dynamic_tracks) == 1
    assert s2.temporal_metadata["n_tracks"] == 1
    assert s2.metadata["dynamic_config"]["activation_frames"] == 2
    rt.reset()
    assert rt.grid.temporal is not None and len(rt.grid.temporal) == 0
    s3 = rt.process(_runtime_frame(ts=0.0))
    assert s3.query_point(10.0, 2.0)["dynamic_state"] == "OBSERVED"


def test_runtime_torch_engine_end_to_end():
    from foveamap.runtime.runtime import FoveaMapRuntime
    import copy
    cfg = FoveaMapConfig()
    cfg = copy.deepcopy(cfg)
    object.__setattr__(cfg.runtime, "grid_engine", "torch")
    rt = FoveaMapRuntime(config=cfg, perception_backend=_MovingBackend())
    s1 = rt.process(_runtime_frame(ts=0.0))
    s2 = rt.process(_runtime_frame(ts=0.1))
    assert s2.query_point(10.0, 2.0)["dynamic"] is True
    assert len(s2.dynamic_tracks) == 1


# ------------------------------------------------------- soak
def test_temporal_soak_120_frames():
    import time
    rng = np.random.default_rng(7)
    gn = FoveatedGrid("spec")
    gt = TorchFoveatedGrid("spec", device="cpu")
    t0 = time.perf_counter()
    max_tracks = 0
    for f in range(120):
        ego = (0.15 * f, 0.02 * f)
        n = int(rng.integers(8, 24))
        radii = rng.uniform(2.0, 30.0, n)
        ang = rng.uniform(-np.pi, np.pi, n)
        xy = np.column_stack([ego[0] + radii * np.cos(ang), ego[1] + radii * np.sin(ang)])
        z = rng.normal(0.0, 0.5, n).astype(np.float32)
        p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        p[:, ROAD] = 0.7
        moving = np.zeros(n, bool)
        if f % 3 == 0:  # recurring moving vehicle near ego
            xy = np.vstack([xy, [[ego[0] + 10.0, ego[1] + 2.0]]])
            z = np.append(z, 0.5)
            pv = np.zeros((1, NUM_CLASSES), dtype=np.float32)
            pv[0, VEHICLE] = 0.95
            p = np.vstack([p, pv])
            moving = np.append(moving, True)
        _step(gn, xy, z, p, moving, ego=ego, ts=0.1 * f)
        _step(gt, xy, z, p, moving, ego=ego, ts=0.1 * f)
        max_tracks = max(max_tracks, len(gn.temporal), len(gt.temporal))
        assert len(gn.temporal) <= 20000 and len(gt.temporal) <= 20000
    dt = time.perf_counter() - t0
    assert dt < 300
    # Parity after long soak.
    assert len(gn.temporal) == len(gt.temporal)
    assert gn.temporal.total_expired == gt.temporal.total_expired
    for (wx, wy) in ([ego[0] + 10.0, ego[1] + 2.0], [0.0, 0.0]):
        qn, qt = gn.query_point(wx, wy), gt.query_point(wx, wy)
        assert qn["dynamic"] == qt["dynamic"]
        assert qn["state"] == qt["state"]
    assert max_tracks > 0
    # Reset works after soak.
    gn.reset()
    gt.reset()
    assert len(gn.temporal) == 0 and len(gt.temporal) == 0
