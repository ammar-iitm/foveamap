"""Public API contract tests: grid, Torch grid, snapshot, and terrain report
must never disagree on traversability, state, cost, or dynamics.

Semantic matrix (authoritative policy, TerrainConfig.traversable_cost_max):
  UNKNOWN / OUT_OF_BOUNDS -> traversable False
  dynamic occupancy       -> traversable False
  cost >= threshold       -> traversable False (lethal included)
  known static, low cost  -> traversable True
  STALE                   -> follows the same cost rule (penalty priced in)
"""
import numpy as np
import pytest

from foveamap.grid import FoveatedGrid, UNKNOWN, F_STEP, F_DEPRESSION
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.core.config import TerrainConfig
from foveamap.core.contracts import MapSnapshot
from foveamap.core.ontology import (
    NUM_CLASSES, ROAD, SIDEWALK, TERRAIN, VEGETATION, BUILDING, POLE,
    VEHICLE, PERSON,
)


def _probs(cls_id, n=1):
    p = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    p[:, cls_id] = 1.0
    return p


def _flat(cls_id, x0=1.0, x1=2.0, y0=1.0, y1=2.0, z=-1.7, step=0.05, moving=False):
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    n = len(xy)
    return xy, np.full(n, z, dtype=np.float32), _probs(cls_id, n), np.full(n, moving, dtype=bool)


def _snap(grid):
    return MapSnapshot(
        timestamp=0.0,
        frame_id="contract",
        ego_pose=np.eye(4, dtype=np.float64),
        origins=tuple(tuple(int(c) for c in o) for o in grid.origins),
        tier_states=tuple(grid.snapshot()),
        dynamic_cells=(),
        dynamic_tracks=tuple(grid.temporal_snapshot()),
        temporal_metadata=dict(grid.temporal_stats()),
        metadata={
            "traversable_cost_max": grid._trav_max(),
            "stale_age_threshold": int(grid.terrain.stale_age_threshold),
        },
    )


def _check_all(grid, snap, x, y, *, state, cost, dynamic, trav):
    """Every public API agrees on one world state."""
    qn = grid.query_point(x, y)
    assert qn["state"] == state, f"grid state {qn['state']} != {state}"
    assert qn["cost"] == cost
    assert qn["dynamic"] is dynamic
    assert qn["is_traversable"] is trav
    assert grid.is_traversable(x, y) is trav
    qs = snap.query_point(x, y)
    assert qs["state"] == state, f"snapshot state {qs['state']} != {state}"
    assert qs["cost"] == cost
    assert qs["dynamic"] is dynamic
    assert snap.is_traversable(x, y) is trav


@pytest.fixture(params=["numpy", "torch"])
def grid(request):
    if request.param == "numpy":
        return FoveatedGrid("spec")
    return TorchFoveatedGrid("spec", device="cpu")


def test_contract_road(grid):
    xy, z, p, m = _flat(ROAD)
    grid.update(xy, z, p, m, (0.0, 0.0))
    _check_all(grid, _snap(grid), 1.5, 1.5,
               state="OBSERVED_STATIC", cost=0, dynamic=False, trav=True)


def test_contract_lethal_building_never_traversable(grid):
    # Wall-height obstacle dominating 3:1 (see phase 7 obstacle semantics).
    xy, z, p, m = _flat(ROAD)
    n = len(xy)
    pb = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pb[:, BUILDING] = 1.0
    grid.update(np.vstack([xy, xy, xy, xy]),
                np.concatenate([z, np.full(3 * n, 0.0, dtype=np.float32)]),
                np.vstack([p, pb, pb, pb]), np.zeros(4 * n, bool), (0.0, 0.0))
    _check_all(grid, _snap(grid), 1.5, 1.5,
               state="OBSERVED_STATIC", cost=254, dynamic=False, trav=False)


@pytest.mark.parametrize("cls_id,cost", [(SIDEWALK, 110), (TERRAIN, 150), (VEGETATION, 254), (POLE, 254)])
def test_contract_semantic_costs(grid, cls_id, cost):
    xy, z, p, m = _flat(cls_id)
    grid.update(xy, z, p, m, (0.0, 0.0))
    trav = cost < 180
    _check_all(grid, _snap(grid), 1.5, 1.5,
               state="OBSERVED_STATIC", cost=cost, dynamic=False, trav=trav)


def test_contract_custom_mid_cost_agrees_everywhere():
    # Cost 160 sits between the old 150 default and the 180 boundary: every
    # API must agree it is traversable under the unified policy.
    priors = [200] * 256
    priors[ROAD] = 160
    cfg = TerrainConfig(cost_priors=tuple(priors))
    for make in (lambda: FoveatedGrid("spec", terrain_config=cfg),
                 lambda: TorchFoveatedGrid("spec", device="cpu", terrain_config=cfg)):
        g = make()
        xy, z, p, m = _flat(ROAD)
        g.update(xy, z, p, m, (0.0, 0.0))
        _check_all(g, _snap(g), 1.5, 1.5,
                   state="OBSERVED_STATIC", cost=160, dynamic=False, trav=True)
        assert g.is_traversable(1.5, 1.5, max_cost=160) is False  # explicit override stays exclusive


def test_contract_high_cost_rejected_everywhere():
    priors = [200] * 256
    priors[ROAD] = 190
    cfg = TerrainConfig(cost_priors=tuple(priors))
    for make in (lambda: FoveatedGrid("spec", terrain_config=cfg),
                 lambda: TorchFoveatedGrid("spec", device="cpu", terrain_config=cfg)):
        g = make()
        xy, z, p, m = _flat(ROAD)
        g.update(xy, z, p, m, (0.0, 0.0))
        _check_all(g, _snap(g), 1.5, 1.5,
                   state="OBSERVED_STATIC", cost=190, dynamic=False, trav=False)


def test_contract_unknown_and_oob(grid):
    grid.update(np.zeros((0, 2)), np.zeros(0, dtype=np.float32),
                np.zeros((0, NUM_CLASSES), dtype=np.float32), np.zeros(0, bool), (0.0, 0.0))
    _check_all(grid, _snap(grid), 50.0, 50.0,
               state="UNKNOWN", cost=255, dynamic=False, trav=False)
    assert grid.is_traversable(9999.0, 9999.0) is False
    assert _snap(grid).is_traversable(9999.0, 9999.0) is False


def test_contract_dynamic_vehicle_and_person(grid):
    for cls_id in (VEHICLE, PERSON):
        grid.reset()
        xy, z, p, m = _flat(ROAD)
        grid.update(xy, z, p, m, (0.0, 0.0))
        pv = _probs(cls_id)
        grid.update(np.array([[1.5, 1.5]]), np.array([0.0], dtype=np.float32), pv, np.array([True]), (0.0, 0.0))
        grid.update(np.array([[1.5, 1.5]]), np.array([0.0], dtype=np.float32), pv, np.array([True]), (0.0, 0.0))
        q = grid.query_point(1.5, 1.5)
        assert q["dynamic"] is True
        assert q["is_traversable"] is False
        assert grid.is_traversable(1.5, 1.5) is False
        assert _snap(grid).is_traversable(1.5, 1.5) is False


def test_contract_stale_boundary_exact():
    # Threshold 3: age 2 fresh, age 3 stale (unified >= semantics + cost +20).
    cfg = TerrainConfig(stale_age_threshold=3)
    for make in (lambda: FoveatedGrid("spec", terrain_config=cfg),
                 lambda: TorchFoveatedGrid("spec", device="cpu", terrain_config=cfg)):
        g = make()
        xy, z, p, m = _flat(ROAD)
        g.update(xy, z, p, m, (0.0, 0.0))
        e = (np.zeros((0, 2)), np.zeros(0, dtype=np.float32),
             np.zeros((0, NUM_CLASSES), dtype=np.float32), np.zeros(0, bool))
        g.update(*e, (0.0, 0.0))
        g.update(*e, (0.0, 0.0))
        assert g.query_point(1.5, 1.5)["state"] == "OBSERVED_STATIC"
        g.update(*e, (0.0, 0.0))
        for api_q in (g.query_point(1.5, 1.5), _snap(g).query_point(1.5, 1.5)):
            assert api_q["state"] == "STALE"
            assert api_q["cost"] == 20  # road prior 0 + stale penalty at age == threshold
        assert g.is_traversable(1.5, 1.5) is True  # stale follows cost rule
        assert _snap(g).is_traversable(1.5, 1.5) is True


def test_contract_step_pothole_rough_block_everywhere(grid):
    xy, z, p, m = _flat(ROAD, x0=1.0, x1=2.0)
    xy2, z2, p2, m2 = _flat(ROAD, x0=2.0, x1=3.0)
    z2 = z2 + 0.5
    grid.update(np.vstack([xy, xy2]), np.concatenate([z, z2]),
                np.vstack([p, p2]), np.concatenate([m, m2]), (0.0, 0.0))
    q = grid.query_point(1.97, 1.5)
    assert q["cost"] >= 180
    assert grid.is_traversable(1.97, 1.5) is False
    assert _snap(grid).is_traversable(1.97, 1.5) is False


def test_contract_slope_and_clearance_units(grid):
    xy, z, p, m = _flat(BUILDING)
    n = len(xy)
    pr = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    pr[:, ROAD] = 1.0
    zb = z + 1.7  # wall rising 1.7 m above the road surface
    grid.update(np.vstack([xy, xy, xy, xy]),
                np.concatenate([zb, np.full(3 * n, -1.7, dtype=np.float32)]),
                np.vstack([p, pr, pr, pr]), np.zeros(4 * n, bool), (0.0, 0.0))
    # Ground is road (dominant 3:1); building evidence gives 1.7 m clearance.
    qg = grid.query_point(1.5, 1.5)
    qs = _snap(grid).query_point(1.5, 1.5)
    assert qg["clear"] == pytest.approx(1.7, abs=0.05)
    assert qs["clearance"] == pytest.approx(qg["clear"], abs=1e-9)  # same metres unit
    snap = _snap(grid)
    assert snap.is_traversable(1.5, 1.5, clearance_req=2.0) is False
    assert snap.is_traversable(1.5, 1.5, clearance_req=1.0) is True


def test_terrain_report_obeys_unified_policy():
    # Dynamic-occupied cells with low static cost must not inflate
    # traversable_cells; stale counting uses unified >= boundary.
    for make in (lambda: FoveatedGrid("spec"),
                 lambda: TorchFoveatedGrid("spec", device="cpu")):
        g = make()
        xy, z, p, m = _flat(ROAD, x0=1.0, x1=3.0, y0=1.0, y1=3.0)
        g.update(xy, z, p, m, (0.0, 0.0))
        pv = _probs(VEHICLE)
        g.update(np.array([[1.5, 1.5]]), np.array([0.0], dtype=np.float32), pv, np.array([True]), (0.0, 0.0))
        g.update(np.array([[1.5, 1.5]]), np.array([0.0], dtype=np.float32), pv, np.array([True]), (0.0, 0.0))
        rep = g.terrain_report()["total"]
        assert rep["dynamic_cells"] >= 1
        # Independent recomputation of the unified rule over live tiers.
        expect_trav = 0
        for t, s in zip(g.tiers, g.state):
            cost = np.asarray(s.cost if not hasattr(s.cost, "cpu") else s.cost.cpu().numpy())
            dyn = np.asarray(s.dynamic_mask if not hasattr(s.dynamic_mask, "cpu")
                             else s.dynamic_mask.cpu().numpy(), dtype=bool)
            expect_trav += int(((cost != 255) & ~dyn & (cost < 180)).sum())
        assert rep["traversable_cells"] == expect_trav
        assert rep["traversable_cells"] + rep["nontraversable_cells"] + rep["unknown_cost_cells"] == rep["total_cells"]
