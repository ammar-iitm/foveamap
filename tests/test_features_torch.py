"""Parity of the PyTorch range-image features against frames.make_features."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foveamap import features_torch as FT  # noqa: E402
from foveamap.frames import SIM_INFO, DatasetInfo, make_features, prev_in_ego  # noqa: E402
from test_grid_torch import DEVICES  # noqa: E402


def test_interp_matches_numpy():
    r = np.random.default_rng(0)
    xp = np.sort(r.uniform(-30, 10, 32))
    fp = r.normal(size=32)
    x = np.concatenate([r.uniform(-40, 20, 1000), xp, [xp[0] - 1, xp[-1] + 1]])
    got = FT.interp(*(torch.from_numpy(v) for v in (x, xp, fp))).numpy()
    np.testing.assert_allclose(got, np.interp(x, xp, fp), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_rows", [64, 32])          # simulator, and nuScenes-sized image
def test_features_parity(drive, device, n_rows):
    frames, _ = drive
    info = SIM_INFO if n_rows == 64 else DatasetInfo("half", 32, 1024)
    for t, f in enumerate(frames):
        if n_rows == 32:
            f = dict(f, ring=(f["ring"] // 2).astype(np.int16))
        prev = prev_in_ego(f)
        if n_rows == 32:
            prev = [None if p is None else (p[0], p[1] // 2) for p in prev]
        a, ia, ra, ca = make_features(f, info, prev)
        b, ib, rb, cb = (x.cpu().numpy() for x in FT.make_features(f, info, prev, device))
        np.testing.assert_array_equal(ra, rb)
        np.testing.assert_array_equal(ca, cb)
        # a pixel may pick a different point only when two points tie on range (NumPy's order is arbitrary)
        tie = ia != ib
        assert tie.sum() <= 5 and np.all(a[0][tie] == b[0][tie]) and np.all((ia[tie] >= 0) & (ib[tie] >= 0))
        np.testing.assert_allclose(b[:6][:, ~tie], a[:6][:, ~tie], rtol=1e-6, atol=1e-6)
        # motion residuals: an older point exactly between two rows may land on the other row
        bad = np.abs(a[6:] - b[6:]) > 1e-4
        assert bad.mean() <= 2e-4, (t, bad.sum())
        if t >= 2:
            assert (a[6:] > 0).sum() > 1000       # the residual channels are really exercised
