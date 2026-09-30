"""The nuScenes loader must undo every transform the mock applies:
rotated/shifted world, rotated sensor mount, shuffled laser ids, lidarseg
label ids, and moving flags recovered from boxes + attributes."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from mock_nuscenes import write_mock, G  # noqa: E402
from foveamap.nuscenes import NuScenesLite, nuscenes_info  # noqa: E402
from foveamap.frames import world_points, make_features, prev_in_ego  # noqa: E402
from foveamap.sim import PARKING, ROAD, POLE, BUILDING  # noqa: E402


@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("nusc"))
    truth = write_mock(root, scenes=(("scene-0103", 5),), n_sweeps=15)
    return root, truth


def test_frames_match_simulator(mock):
    root, truth = mock
    nusc = NuScenesLite(root, verbose=False)
    assert nusc.scene_names("mini_val") == ["scene-0103"]
    assert nusc.scene_names("mini_train") == []
    info = nuscenes_info(nusc.n_rings())
    assert info.n_rows == 64
    toks = nusc.keyframe_tokens("scene-0103")
    assert len(toks) == 2
    for tk in toks:
        f = nusc.frame(tk, info)
        tr = truth[tk]
        assert len(f["pts"]) == len(tr["xyz"])
        # ego-frame geometry recovered through the rotated sensor mount
        assert np.abs(f["pts"] - tr["xyz"]).max() < 2e-3
        np.testing.assert_allclose(f["sensor"], [0, 0, 1.73], atol=1e-6)
        # world pose: 35 deg yaw + shift
        pw = world_points(f)
        sim_world = tr["xyz"].astype(np.float64) + np.array([tr["ego"][0], tr["ego"][1], 0])
        assert np.abs(pw - G(sim_world)).max() < 2e-3
        assert abs(np.degrees(np.arctan2(f["pose"][1, 0], f["pose"][0, 0])) - 35.0) < 1e-6
        # shuffled laser ids sorted back into top-to-bottom rows
        assert np.array_equal(f["ring"], tr["beam"])
        # labels: parking -> drivable surface, pole -> manmade (nuScenes has no such classes)
        exp = tr["label"].astype(int).copy()
        exp[exp == PARKING] = ROAD
        exp[exp == POLE] = BUILDING
        assert np.array_equal(f["label"], exp)
        # moving flags from annotation boxes + attributes
        agree = np.mean(f["moving"] == tr["moving"])
        assert agree > 0.999, agree
        assert f["moving"].sum() > 0
        # previous sweeps exist for the motion cue and re-project near the current frame
        assert all(p is not None for p in f["prev"])
        feats, idx, _, _ = make_features(f, info, prev_in_ego(f))
        valid = idx >= 0
        static = valid & ~np.isin(np.where(valid, f["label"][np.where(valid, idx, 0)], -1), (7, 8))
        assert np.median(feats[6][static]) < 0.05        # static world: small residual
