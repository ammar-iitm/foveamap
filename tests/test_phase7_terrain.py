"""Phase 7 tests: terrain & traversability engine.

Validates analytical terrain geometry (known planes give known slopes),
roughness/step/depression semantics, obstacle vs ground separation, dynamic
exclusion, unknown handling, cost bounds, tier-aware physical units, NumPy /
Torch parity, runtime integration, and edge cases.
"""
import math

import numpy as np
import pytest
import torch

from foveamap.grid import FoveatedGrid, UNKNOWN, F_SLOPE, F_STEP, F_DEPRESSION, F_OVERHANG
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.terrain import slope_at_cell, report_layers
from foveamap.core.config import TerrainConfig, FoveaMapConfig, DEFAULT_CLASS_COSTS
from foveamap.core.contracts import LiDARFrame, MapSnapshot
from foveamap.core.exceptions import NumericalConsistencyError
from foveamap.core.ontology import NUM_CLASSES, ROAD, SIDEWALK, VEHICLE, PERSON, BUILDING


def _probs(cls_id):
    p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p[0, cls_id] = 1.0
    return p


def _patch(x0, x1, y0, y1, step, z_fn, cls_id, moving=False):
    """Classified point patch over a rectangle with height function z(x, y)."""
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    z = np.asarray([z_fn(x, y) for x, y in xy], dtype=np.float32)
    n = len(xy)
    p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    p[:, cls_id] = 1.0
    return xy, z, p, np.full(n, moving, dtype=bool)


def _update(grid, xy, z, probs, moving, ego=(0.0, 0.0)):
    return grid.update(
        np.asarray(xy, dtype=np.float64),
        np.asarray(z, dtype=np.float32),
        np.asarray(probs, dtype=np.float32),
        np.asarray(moving, dtype=bool),
        ego,
    )


# ---------------------------------------------------------------- flat plane
def test_flat_terrain_zero_slope_and_road_cost():
    for grid in (FoveatedGrid("spec"), TorchFoveatedGrid("spec", device="cpu")):
        xy, z, p, m = _patch(1.0, 3.0, 1.0, 3.0, 0.05, lambda x, y: -1.7, ROAD)
        _update(grid, xy, z, p, m)
        q = grid.query_point(2.0, 2.0)
        assert q["state"] == "OBSERVED_STATIC"
        assert q["slope_rad"] == pytest.approx(0.0, abs=1e-6)
        assert q["rough"] is None or abs(q["rough"]) < 0.01
        assert q["cost"] == 0  # road prior, no penalties
        assert q["ground"] == pytest.approx(-1.7, abs=0.01)
        assert q["is_traversable"] is True


def test_planar_slope_matches_analytical_gradient():
    a = 0.1
    expected = math.atan(a)
    for grid in (FoveatedGrid("spec"), TorchFoveatedGrid("spec", device="cpu")):
        xy, z, p, m = _patch(1.0, 4.0, 1.0, 3.0, 0.05, lambda x, y: -1.7 + a * x, ROAD)
        _update(grid, xy, z, p, m)
        q = grid.query_point(2.5, 2.0)
        assert q["slope_rad"] == pytest.approx(expected, abs=0.02)
        # Below the 0.25 rad threshold: no slope flag, no slope penalty.
        assert not (q["flags"] & F_SLOPE)
        assert q["cost"] == 0


def test_steep_slope_sets_flag_and_penalty_consistently():
    a = 0.3
    expected = math.atan(a)
    for grid in (FoveatedGrid("spec"), TorchFoveatedGrid("spec", device="cpu")):
        xy, z, p, m = _patch(1.0, 4.0, 1.0, 3.0, 0.05, lambda x, y: -1.7 + a * x, ROAD)
        _update(grid, xy, z, p, m)
        q = grid.query_point(2.5, 2.0)
        assert q["slope_rad"] == pytest.approx(expected, abs=0.02)
        assert q["flags"] & F_SLOPE
        # Cost equals the documented slope-excess term for the measured slope.
        excess = min(max((q["slope_rad"] - 0.25) / (0.40 - 0.25), 0.0), 1.0)
        assert q["cost"] == round(excess * 40)


def test_critical_slope_clamps_cost_lethal():
    a = 0.5  # atan(0.5) ~= 0.464 rad >= 0.40 critical
    for grid in (FoveatedGrid("spec"), TorchFoveatedGrid("spec", device="cpu")):
        xy, z, p, m = _patch(1.0, 4.0, 1.0, 3.0, 0.05, lambda x, y: -1.7 + a * x, ROAD)
        _update(grid, xy, z, p, m)
        q = grid.query_point(2.5, 2.0)
        assert q["slope_rad"] >= 0.40
        assert q["cost"] >= 220
        assert q["is_traversable"] is False


def test_slope_units_are_radians_not_degrees():
    a = 0.1
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(1.0, 4.0, 1.0, 3.0, 0.05, lambda x, y: -1.7 + a * x, ROAD)
    _update(g, xy, z, p, m)
    q = g.query_point(2.5, 2.0)
    assert q["slope_rad"] < 0.2  # radians; degrees would read ~5.7
    assert q["slope_rad"] == pytest.approx(math.atan(a), abs=0.02)


# ------------------------------------------------------------------ roughness
def test_rough_terrain_raises_cost_by_documented_amount():
    rng = np.random.default_rng(0)
    g = FoveatedGrid("spec")
    n = 12
    # Cell interior: all points land in the single cell containing (2.02, 2.02).
    xy = np.full((n, 2), [2.02, 2.02]) + rng.uniform(-0.01, 0.01, (n, 2))
    z = rng.normal(0.0, 0.1, n).astype(np.float32)
    p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    p[:, ROAD] = 1.0
    _update(g, xy, z, p, np.zeros(n, bool))
    q = g.query_point(2.02, 2.02)
    # Expectation derived from the actual draws (population std, ddof=0, as
    # the engine computes), not from the nominal distribution sigma.
    assert float(z.std()) > 0.04  # guard: draws must exceed the threshold
    assert q["rough"] == pytest.approx(float(z.std()), abs=0.01)
    assert q["cost"] == 30  # road prior 0 + documented roughness penalty 30


# ------------------------------------------------------------------ step/edge
def test_sharp_step_flags_discontinuity_and_cost():
    g = FoveatedGrid("spec")
    xy_l, z_l, p_l, m_l = _patch(1.0, 2.0, 1.0, 3.0, 0.05, lambda x, y: -1.7, ROAD)
    xy_r, z_r, p_r, m_r = _patch(2.0, 3.0, 1.0, 3.0, 0.05, lambda x, y: -1.2, ROAD)
    xy = np.vstack([xy_l, xy_r])
    z = np.concatenate([z_l, z_r])
    p = np.vstack([p_l, p_r])
    m = np.concatenate([m_l, m_r])
    _update(g, xy, z, p, m)
    q = g.query_point(1.97, 2.0)
    assert q["flags"] & F_STEP
    assert q["cost"] >= 180  # step on drivable surface is non-traversable


def test_pothole_depression_penalty():
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(-0.3, 2.8, -0.3, 2.8, 0.05, lambda x, y: -1.7, ROAD)
    # Carve the single cell containing (1.25, 1.25) by cell bounds.
    lo, hi = 1.25, 1.30
    hole = (xy[:, 0] >= lo) & (xy[:, 0] < hi) & (xy[:, 1] >= lo) & (xy[:, 1] < hi)
    assert hole.sum() >= 1
    z = z.copy()
    z[hole] = -1.9
    _update(g, xy, z, p, m)
    q = g.query_point(1.26, 1.26)
    assert q["flags"] & F_DEPRESSION
    assert q["cost"] >= 170


# ------------------------------------------------------- evidence / obstacle
def test_isolated_cell_has_unknown_slope_not_zero():
    g = FoveatedGrid("spec")
    _update(g, [[5.0, 5.0]], [-1.7], _probs(ROAD), [False])
    q = g.query_point(5.0, 5.0)
    assert q["slope_rad"] is None  # insufficient neighbors: unknown, not 0
    assert not (q["flags"] & F_SLOPE)


def test_vertical_obstacle_does_not_become_terrain():
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(1.0, 2.0, 1.0, 2.0, 0.05, lambda x, y: -1.7, ROAD)
    n = len(xy)
    # Building evidence dominates each cell 3:1 so the obstacle is dominant.
    # Wall height 0.0 m over ground -1.7 m: 1.7 m clearance is an overhang but
    # NOT passable-under (needs 2.5 m), so the obstacle dominates cost.
    pb = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pb[:, BUILDING] = 1.0
    xy2 = np.vstack([xy, xy, xy, xy])
    z2 = np.concatenate([z, np.full(3 * n, 0.0, dtype=np.float32)])
    p2 = np.vstack([p, pb, pb, pb])
    m2 = np.zeros(4 * n, bool)
    _update(g, xy2, z2, p2, m2)
    q = g.query_point(1.5, 1.5)
    assert q["ground"] == pytest.approx(-1.7, abs=0.02)  # roof is not terrain
    assert q["z_max"] == pytest.approx(0.0, abs=0.05)
    assert q["cost"] == 254  # obstacle dominates: lethal


def test_vehicle_over_road_keeps_ground_and_overhang():
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(1.0, 2.0, 1.0, 2.0, 0.05, lambda x, y: -1.7, ROAD)
    n = len(xy)
    pv = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pv[:, VEHICLE] = 1.0
    xy2 = np.vstack([xy, xy])
    z2 = np.concatenate([z, np.full(n, 0.5, dtype=np.float32)])
    p2 = np.vstack([p, pv])
    m2 = np.concatenate([m, np.zeros(n, bool)])
    _update(g, xy2, z2, p2, m2)
    q = g.query_point(1.5, 1.5)
    assert q["flags"] & F_OVERHANG
    assert q["ground"] == pytest.approx(-1.7, abs=0.02)
    assert q["clear"] == pytest.approx(2.2, abs=0.05)


def test_dynamic_vehicle_excluded_from_static_terrain():
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(1.0, 2.0, 1.0, 2.0, 0.05, lambda x, y: -1.7, ROAD)
    _update(g, xy, z, p, m)
    before = g.query_point(1.5, 1.5)
    assert before["cost"] == 0
    n = len(xy)
    pv = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pv[:, VEHICLE] = 1.0
    _update(g, xy, np.full(n, 0.5, dtype=np.float32), pv, np.ones(n, bool))
    q = g.query_point(1.5, 1.5)
    assert q["dynamic"] is True
    assert q["ground"] == pytest.approx(-1.7, abs=0.02)  # static terrain intact


def test_dynamic_person_blocks_traversability_temporarily():
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(1.0, 2.0, 1.0, 2.0, 0.05, lambda x, y: -1.7, ROAD)
    _update(g, xy, z, p, m)
    assert g.is_traversable(1.5, 1.5) is True
    pp = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    pp[0, PERSON] = 1.0
    _update(g, [[1.5, 1.5]], [0.0], pp, [True])
    assert g.is_traversable(1.5, 1.5) is False


# ------------------------------------------------------------------ unknown
def test_unknown_cell_is_not_safe():
    g = FoveatedGrid("spec")
    g.update(np.zeros((0, 2)), np.zeros(0, dtype=np.float32),
             np.zeros((0, NUM_CLASSES), dtype=np.float32), np.zeros(0, bool), (0.0, 0.0))
    q = g.query_point(3.0, 3.0)
    assert q["state"] == "UNKNOWN"
    assert q["cost"] == 255
    assert q["ground"] is None
    assert q["slope_rad"] is None
    assert q["rough"] is None
    assert q["is_traversable"] is False


def test_cost_bounds_and_unknown_only_values():
    rng = np.random.default_rng(3)
    g = FoveatedGrid("spec")
    for _ in range(5):
        n = 200
        xy = rng.uniform(-30, 30, (n, 2))
        z = rng.normal(-1.0, 1.0, n).astype(np.float32)
        p = rng.random((n, NUM_CLASSES)).astype(np.float32)
        p /= p.sum(axis=1, keepdims=True)
        _update(g, xy, z, p, rng.random(n) < 0.1)
    for s in g.state:
        c = np.asarray(s.cost)
        assert ((c <= 254) | (c == 255)).all()


@pytest.mark.parametrize("cls_id", list(range(NUM_CLASSES)))
def test_semantic_cost_policy_matches_configuration(cls_id):
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(1.0, 1.3, 1.0, 1.3, 0.05, lambda x, y: -1.7, cls_id)
    _update(g, xy, z, p, m)
    q = g.query_point(1.1, 1.1)
    assert q["cost"] == DEFAULT_CLASS_COSTS[cls_id]


def test_empty_and_single_point_frames():
    g = FoveatedGrid("spec")
    g.update(np.zeros((0, 2)), np.zeros(0, dtype=np.float32),
             np.zeros((0, NUM_CLASSES), dtype=np.float32), np.zeros(0, bool), (0.0, 0.0))
    rep = g.terrain_report()
    assert rep["total"]["known_ground"] == 0
    assert rep["total"]["mean_slope_rad"] is None
    g.update(np.array([[1.0, 1.0]]), np.array([-1.7], dtype=np.float32),
             _probs(ROAD), np.array([False]), (0.0, 0.0))
    assert g.query_point(1.0, 1.0)["slope_rad"] is None


def test_nan_coordinates_rejected_by_contract():
    with pytest.raises(NumericalConsistencyError):
        LiDARFrame(
            pts=np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
            intensity=np.ones(1, dtype=np.float32),
            ring=np.zeros(1, dtype=np.int16),
            pose=np.eye(4, dtype=np.float64),
            timestamp=0.0,
            frame_id="bad",
            source_id="test",
        )


def test_extreme_coordinates_do_not_destabilize():
    g = FoveatedGrid("spec")
    n = 3
    xy = np.array([[1e6, 0.0], [-1e6, 0.0], [0.0, 1e6]])
    z = np.full(n, -1.7, dtype=np.float32)
    p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    p[:, ROAD] = 1.0
    _update(g, xy, z, p, np.zeros(n, bool))
    assert g.last_diagnostics["filtered_points"] == n


# ---------------------------------------------------------------- tier-aware
def test_coarse_tier_preserves_physical_slope():
    a = 0.1
    expected = math.atan(a)
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(20.0, 30.0, -1.0, 1.0, 0.25, lambda x, y: -1.7 + a * x, ROAD)
    _update(g, xy, z, p, m)
    q = g.query_point(25.0, 0.0)
    assert q["tier"] == 1
    assert q["slope_rad"] == pytest.approx(expected, abs=0.03)


def test_tier_boundary_cells_stay_consistent():
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(9.0, 11.0, -1.0, 1.0, 0.05, lambda x, y: -1.7 + 0.05 * x, ROAD)
    _update(g, xy, z, p, m)
    q_in = g.query_point(9.9, 0.0)
    q_out = g.query_point(10.5, 0.0)
    assert q_in["slope_rad"] == pytest.approx(math.atan(0.05), abs=0.03)
    assert q_out["slope_rad"] == pytest.approx(math.atan(0.05), abs=0.03)


# ------------------------------------------------------------------- parity
def test_numpy_torch_terrain_parity():
    scenarios = [
        _patch(1.0, 4.0, 1.0, 3.0, 0.05, lambda x, y: -1.7 + 0.3 * x, ROAD),
        _patch(1.0, 2.0, 1.0, 2.0, 0.05, lambda x, y: -1.7, SIDEWALK),
    ]
    for xy, z, p, m in scenarios:
        gn = FoveatedGrid("spec")
        gt = TorchFoveatedGrid("spec", device="cpu")
        _update(gn, xy, z, p, m)
        _update(gt, xy, z, p, m)
        for t in range(len(gn.tiers)):
            np.testing.assert_array_equal(
                np.asarray(gn.state[t].cost), gt.state[t].cost.cpu().numpy())
            np.testing.assert_array_equal(
                np.asarray(gn.state[t].flags), gt.state[t].flags.cpu().numpy())
        qn = gn.query_point(2.0, 2.0)
        qt = gt.query_point(2.0, 2.0)
        assert qn["slope_rad"] == pytest.approx(qt["slope_rad"], abs=1e-4)
        assert qn["cost"] == qt["cost"]
        rn = gn.terrain_report()
        rt = gt.terrain_report()
        assert rn == rt


def test_roughness_mip_up_parity_within_tolerance():
    rng = np.random.default_rng(11)
    n = 400
    xy = rng.uniform(9.0, 11.0, (n, 2))
    z = (rng.normal(0.0, 0.08, n) - 1.7).astype(np.float32)
    p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    p[:, ROAD] = 1.0
    m = np.zeros(n, bool)
    gn = FoveatedGrid("spec")
    gt = TorchFoveatedGrid("spec", device="cpu")
    _update(gn, xy, z, p, m)
    _update(gt, xy, z, p, m)
    r_n = np.asarray(gn.state[1].rough, dtype=np.float64)
    r_t = gt.state[1].rough.float().cpu().numpy().astype(np.float64)
    mask = np.isfinite(r_n) & np.isfinite(r_t)
    assert mask.sum() > 0
    np.testing.assert_allclose(r_n[mask], r_t[mask], atol=1e-2)
    assert (np.asarray(gn.state[1].flags) == gt.state[1].flags.cpu().numpy()).all()


def test_slope_flag_matches_exposed_slope_fuzz():
    rng = np.random.default_rng(5)
    thresh = TerrainConfig().slope_threshold_rad
    for grid in (FoveatedGrid("spec"), TorchFoveatedGrid("spec", device="cpu")):
        n = 300
        xy = rng.uniform(0.0, 5.0, (n, 2))
        z = (-1.7 + 0.2 * xy[:, 0] + rng.normal(0, 0.02, n)).astype(np.float32)
        p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        p[:, ROAD] = 1.0
        _update(grid, xy, z, p, np.zeros(n, bool))
        for _ in range(20):
            qx, qy = rng.uniform(0.5, 4.5, 2)
            q = grid.query_point(float(qx), float(qy))
            if q.get("tier", -1) < 0 or q["ground"] is None:
                continue
            flagged = bool(q["flags"] & F_SLOPE)
            assert flagged == (q["slope_rad"] is not None and q["slope_rad"] > thresh)


# ------------------------------------------------------------------ runtime
def test_runtime_end_to_end_terrain_snapshot_query():
    from foveamap.runtime.runtime import FoveaMapRuntime
    from foveamap.runtime.perception import ClassicalFallbackBackend
    import torch as _torch
    cfg = FoveaMapConfig()
    rt = FoveaMapRuntime(
        config=cfg,
        perception_backend=ClassicalFallbackBackend(device=_torch.device("cpu")),
    )
    n = 200
    xs = np.linspace(1.0, 3.0, n)
    pts = np.column_stack([xs, np.full(n, 1.0), np.full(n, -1.7)]).astype(np.float32)
    frame = LiDARFrame(
        pts=pts,
        intensity=np.ones(n, dtype=np.float32),
        ring=np.zeros(n, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64),
        timestamp=0.0,
        frame_id="t7",
        source_id="test",
    )
    snap = rt.process(frame)
    q = snap.query_point(2.0, 1.0)
    assert q["slope_rad"] is not None
    assert abs(q["slope_rad"]) < 0.05
    assert q["cost"] != 255
    rep = rt.grid.terrain_report()
    assert rep["total"]["known_ground"] > 0
    assert rep["total"]["traversable_cells"] > 0


def test_custom_slope_threshold_changes_flag_behavior():
    a = 0.3
    xy, z, p, m = _patch(1.0, 4.0, 1.0, 3.0, 0.05, lambda x, y: -1.7 + a * x, ROAD)
    g_lo = FoveatedGrid("spec")
    _update(g_lo, xy, z, p, m)
    assert g_lo.query_point(2.5, 2.0)["flags"] & F_SLOPE
    g_hi = FoveatedGrid(
        "spec", terrain_config=TerrainConfig(slope_threshold_rad=0.5, slope_critical_rad=0.6))
    _update(g_hi, xy, z, p, m)
    q = g_hi.query_point(2.5, 2.0)
    assert not (q["flags"] & F_SLOPE)
    assert q["slope_rad"] == pytest.approx(math.atan(a), abs=0.02)


# ------------------------------------------------------------------ report
def test_terrain_report_aggregates_and_matches_grids():
    g = FoveatedGrid("spec")
    xy, z, p, m = _patch(1.0, 3.0, 1.0, 3.0, 0.05, lambda x, y: -1.7, ROAD)
    _update(g, xy, z, p, m)
    rep = g.terrain_report()
    total = rep["total"]
    assert total["tiers"] == 2
    assert total["traversable_cells"] == total["known_ground"]
    assert total["unknown_cost_cells"] == total["total_cells"] - total["known_ground"]
    assert total["mean_slope_rad"] == pytest.approx(0.0, abs=1e-6)
    gt = TorchFoveatedGrid("spec", device="cpu")
    _update(gt, xy, z, p, m)
    assert gt.terrain_report() == rep
