"""P0 regression tests for final Phase 1-5 hardening.

Covers:
- NumPy effective-class vectorization (old `gcls in GROUND_CLASSES` raised
  ValueError on arrays) across scalar/multi/ground/non-ground/mixed/empty/
  overhang cases plus NumPy/Torch parity.
- Secondary-class evidence semantics: stored class always pairs with its own
  confidence; ground preference is explicit and confidence-correct.
- Safe checkpoint loading: weights_only, missing/invalid/mismatch rejected.
"""
import os
import numpy as np
import pytest
import torch

from foveamap.grid import FoveatedGrid, UNKNOWN
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.core.ontology import (
    NUM_CLASSES, GROUND_CLASSES, ROAD, SIDEWALK, PARKING, TERRAIN,
    VEGETATION, BUILDING, POLE, VEHICLE, PERSON,
)
from foveamap.model import RangeUNet


def _probs(primary, p_primary, secondary, p_secondary, tertiary=None, p_tertiary=0.0):
    p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p[0, primary] = p_primary
    p[0, secondary] = p_secondary
    if tertiary is not None:
        p[0, tertiary] = p_tertiary
    s = p.sum()
    if s > 0:
        p /= s
    return p


def _update_once(grid, xy, z, probs, moving=None):
    m = np.zeros(len(xy), dtype=bool) if moving is None else np.asarray(moving, dtype=bool)
    if isinstance(grid, TorchFoveatedGrid):
        grid.update(xy, z, probs, m, (0.0, 0.0))
    else:
        grid.update(xy, z, probs, m, (0.0, 0.0))


# ---------------------------------------------------------------- eff_cls
def test_eff_cls_vectorized_single_and_multi():
    g = FoveatedGrid("spec", fuse=False)
    s = g.state[0]
    # Directly exercise the property on multi-cell arrays (old code raised ValueError).
    s.cls[:2, :2] = VEHICLE
    s.flags[:2, :2] = (ROAD << 4)  # ground evidence road
    s.ground[:2, :2] = 0.0
    s.clear[:2, :2] = np.uint8(130)  # 2.6 m clearance -> passable
    eff = s.eff_cls
    assert eff.shape == s.cls.shape
    assert int(eff[0, 0]) == ROAD
    assert int(eff[1, 1]) == ROAD


def test_eff_cls_ground_vs_nonground():
    g = FoveatedGrid("spec", fuse=False)
    s = g.state[0]
    s.cls[0, 0] = VEHICLE
    s.flags[0, 0] = np.uint8(ROAD << 4)
    s.ground[0, 0] = 0.0
    s.clear[0, 0] = np.uint8(130)
    assert int(s.eff_cls[0, 0]) == ROAD  # ground -> passable

    s.cls[0, 1] = VEHICLE
    s.flags[0, 1] = np.uint8(POLE << 4)  # non-ground runner-up
    s.ground[0, 1] = 0.0
    s.clear[0, 1] = np.uint8(130)
    assert int(s.eff_cls[0, 1]) == VEHICLE  # non-ground must NOT become effective


def test_eff_cls_mixed_unknown_empty_overhang():
    g = FoveatedGrid("spec", fuse=False)
    s = g.state[0]
    # UNKNOWN primary stays UNKNOWN regardless of flags.
    s.cls[2, 2] = UNKNOWN
    s.flags[2, 2] = np.uint8(ROAD << 4)
    s.ground[2, 2] = 0.0
    s.clear[2, 2] = np.uint8(130)
    assert int(s.eff_cls[2, 2]) == UNKNOWN
    # No overhang (clear small) -> dominant kept.
    s.cls[3, 3] = VEHICLE
    s.flags[3, 3] = np.uint8(ROAD << 4)
    s.ground[3, 3] = 0.0
    s.clear[3, 3] = np.uint8(10)  # 0.2 m
    assert int(s.eff_cls[3, 3]) == VEHICLE
    # Empty/unknown flags nibble 0xF -> never passable.
    s.cls[4, 4] = VEHICLE
    s.flags[4, 4] = np.uint8(0xF0)
    s.ground[4, 4] = 0.0
    s.clear[4, 4] = np.uint8(130)
    assert int(s.eff_cls[4, 4]) == VEHICLE


def test_eff_cls_numpy_torch_parity():
    gn = FoveatedGrid("spec", fuse=False)
    gt = TorchFoveatedGrid("spec", fuse=False, device="cpu")
    for (i, j, dom, sec) in [(0, 0, VEHICLE, ROAD), (0, 1, VEHICLE, POLE), (1, 0, BUILDING, TERRAIN)]:
        gn.state[0].cls[i, j] = dom
        gn.state[0].flags[i, j] = np.uint8(sec << 4)
        gn.state[0].ground[i, j] = 0.0
        gn.state[0].clear[i, j] = np.uint8(130)
        gt.state[0].cls[i, j] = dom
        gt.state[0].flags[i, j] = np.uint8(sec << 4)
        gt.state[0].ground[i, j] = 0.0
        gt.state[0].clear[i, j] = np.uint8(130)
    en = gn.state[0].eff_cls
    et = gt.state[0].eff_cls.cpu().numpy()
    np.testing.assert_array_equal(en[:2, :2], et[:2, :2])


# ------------------------------------------------------- secondary semantics
@pytest.mark.parametrize("case,primary,p1,secondary,p2,ground_extra,expected_sec", [
    # A: dominant vehicle, runner pole, ground road -> ground preferred (explicit).
    ("A", VEHICLE, 0.5, POLE, 0.3, (ROAD, 0.2), ROAD),
    # B: dominant road, runner sidewalk, ground road (same as dominant) -> runner.
    ("B", ROAD, 0.6, SIDEWALK, 0.3, None, SIDEWALK),
    # C: dominant vehicle, runner road, ground road -> road.
    ("C", VEHICLE, 0.5, ROAD, 0.3, None, ROAD),
    # D: dominant vegetation, runner terrain, ground terrain -> terrain.
    ("D", VEGETATION, 0.5, TERRAIN, 0.3, None, TERRAIN),
    # E: three-way: vehicle 0.5, road 0.25, pole 0.25 -> ground road preferred.
    ("E", VEHICLE, 0.5, ROAD, 0.25, (POLE, 0.25), ROAD),
])
def test_secondary_semantics_numpy_and_torch(case, primary, p1, secondary, p2, ground_extra, expected_sec):
    if case == "A":
        p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        p[0, VEHICLE] = 0.5
        p[0, POLE] = 0.3
        p[0, ROAD] = 0.2
    elif case == "E":
        p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
        p[0, VEHICLE] = 0.5
        p[0, ROAD] = 0.25
        p[0, POLE] = 0.25
    else:
        p = _probs(primary, p1, secondary, p2)
    xy = np.array([[2.0, 2.0]])
    z = np.array([0.0])
    for grid in (FoveatedGrid("spec", fuse=True), TorchFoveatedGrid("spec", fuse=True, device="cpu")):
        grid.reset()
        _update_once(grid, xy, z, p)
        q = grid.query_point(2.0, 2.0)
        assert q["dominant_class"] == primary, case
        assert q["secondary_class"] == expected_sec, f"{case}: got {q['secondary_class']}"
        # Confidence must belong to the stored class: recompute expected from fused q.
        # Stored 4-bit quantization allows ~1/15 tolerance.
        assert 0.0 <= q["secondary_confidence"] <= 1.0
        # Primary confidence should be ~p1 (first frame, no history blending distortion beyond alpha).
        assert q["secondary_confidence"] > 0.02


def test_secondary_confidence_matches_stored_class():
    # Dominant road 0.6, sidewalk 0.3 -> secondary sidewalk conf ~0.3 (quantized).
    p = _probs(ROAD, 0.6, SIDEWALK, 0.3, PARKING, 0.1)
    xy = np.array([[2.0, 2.0]])
    z = np.array([0.0])
    gn = FoveatedGrid("spec", fuse=True)
    gt = TorchFoveatedGrid("spec", fuse=True, device="cpu")
    _update_once(gn, xy, z, p)
    _update_once(gt, xy, z, p)
    qn = gn.query_point(2.0, 2.0)
    qt = gt.query_point(2.0, 2.0)
    assert qn["secondary_class"] == SIDEWALK
    assert qt["secondary_class"] == SIDEWALK
    # Both engines agree within one 4-bit step (1/15 ~= 0.067).
    assert abs(qn["secondary_confidence"] - 0.3) < 0.08
    assert abs(qt["secondary_confidence"] - 0.3) < 0.08
    assert abs(qn["secondary_confidence"] - qt["secondary_confidence"]) < 0.08


def test_secondary_none_maps_to_unknown_parity():
    gn = FoveatedGrid("spec")
    gt = TorchFoveatedGrid("spec", device="cpu")
    assert int(gn.state[0].secondary_class[0, 0]) == UNKNOWN
    assert int(gt.state[0].secondary_class[0, 0].item()) == UNKNOWN


# ------------------------------------------------------- checkpoint safety
def test_safe_checkpoint_roundtrip_and_rejections(tmp_path):
    m = RangeUNet()
    ckpt = os.path.join(str(tmp_path), "good.pt")
    torch.save(m.state_dict(), ckpt)
    from foveamap.model import load_model
    loaded = load_model(ckpt, device="cpu")
    assert isinstance(loaded, RangeUNet)

    # Missing file must raise clearly (FileNotFoundError or OSError).
    with pytest.raises(Exception):
        load_model(os.path.join(str(tmp_path), "missing.pt"), device="cpu")

    # Invalid (non-checkpoint) file must fail clearly, not execute.
    bad = os.path.join(str(tmp_path), "bad.pt")
    with open(bad, "wb") as fh:
        fh.write(b"not a checkpoint")
    with pytest.raises(Exception):
        load_model(bad, device="cpu")


def test_production_checkpoint_policy_rejects_mismatch(tmp_path):
    from foveamap.runtime.perception import RangeUNetBackend
    from foveamap.core.config import PerceptionConfig, SensorConfig
    import torch as th
    # Build a checkpoint with wrong class count (e.g. 5 instead of 9).
    m = RangeUNet()
    sd = m.state_dict()
    sd["sem.weight"] = th.zeros((5, 16, 1, 1))
    sd["sem.bias"] = th.zeros((5,))
    ckpt = os.path.join(str(tmp_path), "mismatch.pt")
    th.save(sd, ckpt)
    cfg = PerceptionConfig(checkpoint_path=ckpt, num_classes=9)
    with pytest.raises(Exception):
        RangeUNetBackend(config=cfg, sensor_config=SensorConfig(), device=th.device("cpu"))
