"""Grid-engine invariants from the PRD acceptance criteria."""
import numpy as np
from foveamap.grid import FoveatedGrid
from foveamap.core.ontology import NUM_CLASSES


def _rand(n=50000, seed=0, ego=(3.3, -2.5)):
    r = np.random.default_rng(seed)
    xy = r.uniform(-99.9, 99.9, (n, 2)) + np.array(ego)
    # add points exactly on tier and cell boundaries
    xy[:200, 0] = 10.0
    xy[200:400, 1] = -10.0 + ego[1]
    xy[400:600] = np.round(xy[400:600] / 0.05) * 0.05
    z = r.normal(0, 0.1, n)
    p = np.eye(NUM_CLASSES)[r.integers(0, NUM_CLASSES, n)]
    return xy, z, p, np.zeros(n, bool)


def test_no_point_lost_and_nesting():
    for prof in ("spec", "graded"):
        g = FoveatedGrid(prof, fuse=False)
        xy, z, p, m = _rand()
        ego = np.array([3.3, -2.5])
        org = g.window_origins(ego)
        st = g.bin_points(xy, z, p, m, org)
        # outer tier holds every point in its window
        f = g.fine_index(xy) // g.tiers[-1].ratio - org[-1]
        inside = ((f >= 0) & (f < g.tiers[-1].n)).all(1)
        assert st[-1]["n_pts"].sum() == inside.sum()
        # each finer tier's points aggregate exactly into the coarser tier's cells
        g.origins = org
        for k in range(1, len(g.tiers)):
            fine, coarse = st[k - 1], st[k]
            r = g.tiers[k].ratio // g.tiers[k - 1].ratio
            nf, nc = g.tiers[k - 1].n, g.tiers[k].n
            fi, fj = fine["key"] // nf + org[k - 1][0], fine["key"] % nf + org[k - 1][1]
            ck = (fi // r - org[k][0]) * nc + (fj // r - org[k][1])
            sums = {}
            for c, n in zip(ck, fine["n_pts"]):
                sums[c] = sums.get(c, 0) + n
            coarse_map = dict(zip(coarse["key"], coarse["n_pts"]))
            for c, n in sums.items():
                assert coarse_map[c] == n, (prof, k, c, n, coarse_map[c])
            # every coarse cell under the fine box is fully covered by fine cells
            mask = g.inner_mask(k).ravel()
            under = coarse["key"][mask[coarse["key"]]]
            assert set(under) == set(sums), prof


def test_scroll_keeps_world_alignment():
    g = FoveatedGrid("spec", fuse=True)
    xy = np.array([[5.02, 1.03]])
    z = np.array([0.4])
    p = np.eye(NUM_CLASSES)[[5]]
    g.update(xy, z, p, np.zeros(1, bool), (0.0, 0.0))
    s0 = g.state[0]
    i0, j0 = np.argwhere(s0.count > 0)[0]
    g.update(np.zeros((0, 2)), np.zeros(0), np.zeros((0, NUM_CLASSES)), np.zeros(0, bool), (1.7, 0.2))
    s1 = g.state[0]
    i1, j1 = np.argwhere(s1.count > 0)[0]
    c = g.cell_centres(0)[i1, j1]
    assert abs(c[0] - 5.025) < 1e-6 and abs(c[1] - 1.025) < 1e-6, c
    assert s1.age[i1, j1] == 1


def test_memory_is_16_bytes_per_cell():
    g = FoveatedGrid("spec")
    assert g.nbytes == 16 * g.n_cells == 16 * 320000
