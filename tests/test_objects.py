"""Object extraction: boxes, classes, motion, tier handling and matching."""
import numpy as np

from foveamap.grid import FoveatedGrid, TierLayers
from foveamap.objects import extract_objects, match_objects
from foveamap.sim import POLE, VEHICLE, PERSON


def _empty_grid(ego=(0.3, -0.2)):
    g = FoveatedGrid("spec")
    g.origins = g.window_origins(np.asarray(ego))
    g.state = [TierLayers(t.n) for t in g.tiers]
    for s in g.state:
        s.eff_cls = s.cls.copy()
    return g


def _paint(g, k, inside, cls, z=1.5):
    """Mark the tier-k cells whose centres satisfy inside(x, y) as observed `cls` this frame."""
    cen = g.cell_centres(k)
    m = inside(cen[..., 0], cen[..., 1])
    s = g.state[k]
    s.cls[m] = s.eff_cls[m] = cls
    s.age[m] = 0
    s.z_max[m] = z
    return m


def _rect(cx, cy, length, width, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return lambda x, y: (np.abs((x - cx) * c + (y - cy) * s) < length / 2) & \
                        (np.abs(-(x - cx) * s + (y - cy) * c) < width / 2)


def test_boxes_classes_and_motion():
    g = _empty_grid()
    _paint(g, 0, _rect(3.0, 2.0, 4.0, 1.8, 0.5), VEHICLE)                 # parked car, fine tier
    _paint(g, 0, lambda x, y: np.hypot(x + 4, y - 4) < 0.08, POLE, z=3.0)
    _paint(g, 1, _rect(40.0, 10.0, 4.5, 2.0, 0.0), VEHICLE)               # far car, coarse tier
    _paint(g, 1, _rect(3.0, 2.0, 4.0, 1.8, 0.5), VEHICLE)                 # coarse copy under the fine box: ignored
    cen = g.cell_centres(0)
    pi, pj = np.nonzero(np.hypot(cen[..., 0] - 6, cen[..., 1] + 3) < 0.25)
    dyn = [dict(i=pi, j=pj, cls=np.full(len(pi), PERSON)), dict(i=np.zeros(0, int), j=np.zeros(0, int), cls=np.zeros(0, int))]

    objs = extract_objects(g, g.state, dyn)
    by = {(o["cls"], round(o["x"])): o for o in objs}
    assert len(objs) == 4, objs

    car = by[(VEHICLE, 3)]
    assert abs(car["x"] - 3.0) < 0.1 and abs(car["y"] - 2.0) < 0.1
    assert abs(car["length"] - 4.0) < 0.15 and abs(car["width"] - 1.8) < 0.15
    assert abs(car["yaw"] - 0.5) < 0.05
    assert not car["moving"] and abs(car["z_top"] - 1.5) < 1e-3

    far = by[(VEHICLE, 40)]
    assert abs(far["length"] - 4.5) <= 0.5 and abs(far["yaw"]) < 1e-6      # 0.5 m cells

    assert by[(POLE, -4)]["length"] < 0.25
    person = by[(PERSON, 6)]
    assert person["moving"] and person["z_top"] is None                     # dynamic cells carry no height


def test_stale_cells_are_left_out():
    g = _empty_grid()
    m = _paint(g, 0, _rect(3.0, 2.0, 4.0, 1.8, 0.0), VEHICLE)
    g.state[0].age[m] = 5
    assert extract_objects(g, g.state) == []
    assert len(extract_objects(g, g.state, max_age=5)) == 1


def test_matching_counts():
    car = dict(cls=VEHICLE, x=5.0, y=0.0, moving=False, n_cells=40)
    gt = [car, dict(cls=PERSON, x=8.0, y=1.0, moving=True, n_cells=6), dict(cls=VEHICLE, x=60.0, y=0.0, moving=False, n_cells=9)]
    pred = [dict(car, x=5.8), dict(car, x=3.0, n_cells=4),                  # the car, and a fragment of it
            dict(cls=PERSON, x=8.0, y=2.5, moving=True, n_cells=5),
            dict(cls=POLE, x=2.0, y=2.0, moving=False, n_cells=3)]
    counts, agree, pairs, diag = match_objects(pred, gt, (0.0, 0.0))
    assert counts[VEHICLE] == [1, 1, 0]          # the car 60 m away is out of range
    assert counts[PERSON] == [0, 1, 1]           # 1.5 m off: no match
    assert counts[POLE] == [0, 1, 0]
    assert (agree, pairs) == (1, 1)
    assert diag[VEHICLE] == dict(tp_cells=[40], fp_cells=[4], fn_cells=[], fp_fragment=[True], fp_other_class=[False])
    assert diag[PERSON]["fp_fragment"] == [True] and diag[PERSON]["fn_cells"] == [6]
    assert diag[POLE]["fp_fragment"] == [False] and diag[POLE]["fp_other_class"] == [False]
