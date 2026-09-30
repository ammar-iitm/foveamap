"""Parity of the PyTorch grid engine against the NumPy reference."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.ndimage import uniform_filter  # noqa: E402
from foveamap.grid import FoveatedGrid, DRIVABLE, DEPRESSION_THRESH, DEPRESSION_WIN  # noqa: E402
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


def _drive(frames=6, n=40000, seed=2):
    """A short drive: ego moves ~1.3 m/frame (window scrolls), dense structured ground near the ego."""
    r = np.random.default_rng(seed)
    for f in range(frames):
        ego = np.array([1.3 * f + 0.37, 0.6 * f - 0.21])
        xy, z, p, m = _soft(n, seed + f, ego)
        near = np.abs(xy - ego).max(1) < 9
        rel = xy - ego
        z[near] = 0.02 * rel[near, 0] + np.where(rel[near, 1] > 3, 0.15, 0.0) + r.normal(0, 0.01, near.sum())
        hole = near & (np.hypot(rel[:, 0] - 2, rel[:, 1]) < 0.4)
        z[hole] -= 0.12                                     # pothole
        over = near & (np.abs(rel[:, 0] + 3) < 1) & (r.random(n) < 0.3)
        z[over] += 3.0                                      # overhang (e.g. a sign gantry)
        yield xy, z, p, m, ego


def _depression_ties(t, s, eps=1e-6):
    """Cells whose pothole test sits within eps of its threshold in the NumPy reference.

    The local-mean filter (scipy vs avg_pool2d) rounds differently in the last
    float32 bit, so the depression flag of these cells is a coin toss either way.
    """
    gz = s.ground.astype(np.float32)
    drv = DRIVABLE[s.eff_cls] & np.isfinite(gz)
    win = max(3, int(round(DEPRESSION_WIN / t.cell)) | 1)
    num = uniform_filter(np.where(drv, gz, 0.0), win, mode="constant")
    den = uniform_filter(drv.astype(np.float32), win, mode="constant")
    ref = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)
    return drv & (np.abs(gz - (ref - DEPRESSION_THRESH)) <= eps)


def _state_mismatch(ref_s, got_s, f16_tol, ignore=None):
    """Fraction of cells whose layers differ (f16 heights compared with a tolerance)."""
    bad = np.zeros(ref_s.cls.shape, bool)
    for f in ("count", "cls", "conf", "flags", "clear", "cost", "age", "eff_cls"):
        bad |= getattr(ref_s, f) != getattr(got_s, f)
    for f in ("z_min", "z_max", "ground", "rough"):
        a, b = getattr(ref_s, f).astype(np.float64), getattr(got_s, f).astype(np.float64)
        bad |= ~(np.isclose(a, b, rtol=f16_tol, atol=f16_tol, equal_nan=True))
    if ignore is not None:
        bad &= ~ignore
    return bad.mean(), bad.sum()


def _run_parity(device, dtype, profile, fuse, max_frac, f16_tol, skip_ties=False):
    ref_g = FoveatedGrid(profile, fuse=fuse)
    g = TorchFoveatedGrid(profile, fuse=fuse, device=device, dtype=dtype)
    for xy, z, p, m, ego in _drive():
        ref_dyn, _ = ref_g.update(xy, z, p, m, ego)
        dyn, _ = g.update(xy, z, p, m, ego)
        assert all(np.array_equal(a, b) for a, b in zip(ref_g.origins, g.origins))
        for k, (rs, ts, rd, td) in enumerate(zip(ref_g.state, g.state, ref_dyn, dyn)):
            ties = _depression_ties(ref_g.tiers[k], rs) if skip_ties else None
            frac, cnt = _state_mismatch(rs, ts.to_numpy(), f16_tol, ties)
            assert frac <= max_frac, (k, frac, cnt)
            for key in ("i", "j", "cls"):
                np.testing.assert_array_equal(rd[key], td[key].cpu().numpy())
    return ref_g, g


@pytest.mark.parametrize("profile", ["spec", "graded"])
@pytest.mark.parametrize("fuse", [True, False])
def test_update_parity_float64_exact(profile, fuse):
    ref_g, _ = _run_parity("cpu", torch.float64, profile, fuse, max_frac=0.0, f16_tol=0.0, skip_ties=True)
    s0 = ref_g.state[0]         # the drive really exercises the derived layers
    assert all(((s0.flags & fl) > 0).sum() > 10 for fl in (1 << 1, 1 << 2, 1 << 3))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fuse", [True, False])
def test_update_parity_float32(device, fuse):
    # float32 can move an f16 height by one ulp or tip a threshold; allow a handful of cells in 160k
    _run_parity(device, torch.float32, "spec", fuse, max_frac=1e-4, f16_tol=2e-3)


def test_torch_state_is_16_bytes_per_cell():
    g = TorchFoveatedGrid("spec", device="cpu")
    assert g.nbytes == 16 * g.n_cells == 16 * 320000
