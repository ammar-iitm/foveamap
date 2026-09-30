"""Parity of the PyTorch grid engine against the NumPy reference."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from foveamap.grid import FoveatedGrid  # noqa: E402
from foveamap.grid_torch import TorchFoveatedGrid, stats_to_numpy  # noqa: E402
from foveamap.sim import NUM_CLASSES, PERSON, VEHICLE  # noqa: E402
from test_grid import _rand  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) \
    + (["mps"] if torch.backends.mps.is_available() else [])
EGO = np.array([3.3, -2.5])


def _soft(n=50000, seed=1, ego=EGO):
    """Soft class probabilities, some moving points, ground heights with spread."""
    xy, _, _, _ = _rand(n, seed, tuple(ego))
    r = np.random.default_rng(seed)
    xy[n // 2:] = r.uniform(-4, 4, (n - n // 2, 2)) + ego      # dense near the ego: many points per fine cell
    logits = r.normal(0, 2, (n, NUM_CLASSES))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    z = 1.5 + r.normal(0, 0.3, n)
    cls = p.argmax(1)
    moving = np.isin(cls, (VEHICLE, PERSON)) & (r.random(n) < 0.5)
    return xy, z, p, moving


def _check_stats(ref, got, tol):
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert a.keys() == b.keys()
        np.testing.assert_array_equal(a["key"], b["key"])
        np.testing.assert_array_equal(a["n_pts"], b["n_pts"])
        assert a["n_in"] == b["n_in"]
        for k in ("n_static", "n_dyn", "n_dyn_person"):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        for k in ("z_min", "z_max", "ground", "rough", "zmin_ng", "p_static", "p_all"):
            np.testing.assert_allclose(b[k], a[k], rtol=tol, atol=tol, err_msg=k)   # nan/inf must match


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("profile", ["spec", "graded"])
@pytest.mark.parametrize("data", [_rand, _soft])
def test_bin_points_parity(device, profile, data):
    xy, z, p, m = data()
    ref_g, g = FoveatedGrid(profile), TorchFoveatedGrid(profile, device=device)
    org = ref_g.window_origins(EGO)
    _check_stats(ref_g.bin_points(xy, z, p, m, org), stats_to_numpy(g.bin_points(xy, z, p, m, org)), 1e-5)


def test_bin_points_parity_float64():
    xy, z, p, m = _soft()
    ref_g, g = FoveatedGrid(), TorchFoveatedGrid(device="cpu", dtype=torch.float64)
    org = ref_g.window_origins(EGO)
    _check_stats(ref_g.bin_points(xy, z, p, m, org), stats_to_numpy(g.bin_points(xy, z, p, m, org)), 1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_bin_points_tensor_input_and_empty(device):
    g, ref_g = TorchFoveatedGrid(device=device), FoveatedGrid()
    org = ref_g.window_origins(EGO)
    xy, z, p, m = _rand(5000)
    got = g.bin_points(torch.from_numpy(xy), torch.from_numpy(z), torch.from_numpy(p), torch.from_numpy(m), org)
    _check_stats(ref_g.bin_points(xy, z, p, m, org), stats_to_numpy(got), 1e-5)
    empty = stats_to_numpy(g.bin_points(np.zeros((0, 2)), np.zeros(0), np.zeros((0, NUM_CLASSES)),
                                        np.zeros(0, bool), org))
    assert all(s["n_in"] == 0 and len(s["key"]) == 0 and s["p_all"].shape == (0, NUM_CLASSES) for s in empty)
