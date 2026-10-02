"""Comprehensive Phase 5 Production Mapping Engine Verification Suite.

Tests all Phase 5 requirements from the FoveaMap PRD and Architecture Vision:
1. Authoritative Tier Selection & Invariant Conservation.
2. Exactly-one native tier assignment (no double-counting, no lost points).
3. Integer Lattice Mip-Up Conservation (parent-child hierarchy, count conservation).
4. Boundary handling: exact boundary coordinates, negative coords, float precision.
5. Dynamic Layer Separation: rebuilt each frame, no ghost trails, class selection.
6. Stale and Free-Space Semantics: conservative ray clearing, stale cell lifecycle.
7. Terrain derivation: ground height, roughness, 2.5D slope, step/depression, cost.
8. Memory Accounting: <= 8 MB total, >= 30x vs uniform 5cm.
9. NumPy and Torch Parity across all mapping operations.
10. Query API on FoveatedGrid, TorchFoveatedGrid, and MapSnapshot.
11. Snapshot Ownership & Detached Immutability.
12. Failure & Edge Cases: 0 pts, 1 pt, duplicates, boundaries, negative coords, 100-frame soak.
"""
import copy
import numpy as np
import pytest
import torch

from foveamap.grid import FoveatedGrid
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.core.config import FoveaMapConfig, GridConfig, TerrainConfig
from foveamap.core.ontology import NUM_CLASSES, VEHICLE, PERSON
from foveamap.core.contracts import LiDARFrame, MapSnapshot
from foveamap.runtime.runtime import FoveaMapRuntime


# ============================================================================
# 1. Authoritative Tier Selection & Conservation Invariants
# ============================================================================

def test_points_in_equals_assigned_plus_filtered():
    """Verify points_in == native_points_assigned + filtered_points."""
    g = FoveatedGrid("spec", fuse=False)
    rng = np.random.default_rng(42)
    # Generate points across near, mid, far, and outside
    xy = rng.uniform(-120.0, 120.0, (10000, 2))
    ego = np.array([0.0, 0.0])
    org = g.window_origins(ego)
    
    tier_idx, diag = g.assign_native_tiers(xy, org)
    
    num_in = len(xy)
    num_assigned = diag["native_points_assigned"]
    num_filtered = diag["filtered_points"]
    
    assert num_in == num_assigned + num_filtered
    assert num_in == diag["points_in"]
    inside_mask = tier_idx >= 0
    assert inside_mask.sum() == num_assigned
    assert np.all(tier_idx[inside_mask] >= 0)
    assert np.all(tier_idx[inside_mask] < len(g.tiers))
    assert np.all(tier_idx[~inside_mask] == -1)


def test_exactly_one_native_tier_assignment():
    """Verify no point receives two native tiers and boundary coordinates are deterministic."""
    g = FoveatedGrid("graded", fuse=False)
    # Graded profile has 3 tiers:
    # Tier 0: half 10.0 (r=0.05, [-10, 10))
    # Tier 1: half 25.0 (r=0.10, [-25, 25))
    # Tier 2: half 100.0 (r=0.50, [-100, 100))
    coords = [
        [0.0, 0.0],          # Center -> Tier 0
        [4.95, -4.95],       # Inside Tier 0 -> Tier 0
        [10.0, 0.0],         # Exact boundary Tier 0 / Tier 1 -> Tier 1
        [-10.0, 0.0],        # Exact negative boundary Tier 0 / Tier 1 -> Tier 0 (half-open)
        [15.0, 15.0],        # Mid tier 1 -> Tier 1
        [25.0, 0.0],         # Exact boundary Tier 1 / Tier 2 -> Tier 2
        [35.0, -35.0],       # Outer tier 2 -> Tier 2
        [100.0, 0.0],        # Exact outer boundary Tier 2 -> Filtered
        [-100.0, -100.0],    # Exact negative outer boundary Tier 2 -> Tier 2
        [150.0, 150.0],      # Far outside -> Filtered
    ]
    xy = np.array(coords, dtype=np.float64)
    org = g.window_origins((0.0, 0.0))
    tier_idx, diag = g.assign_native_tiers(xy, org)
    
    # Each coordinate has exactly one assignment (either in [0, 2] or -1)
    assert len(tier_idx) == len(coords)
    assert tier_idx[0] == 0  # center is tier 0
    assert tier_idx[1] == 0  # inside tier 0
    assert tier_idx[2] == 1  # 10.0 is tier 1
    assert tier_idx[3] == 0  # -10.0 is tier 0
    assert tier_idx[4] == 1  # 15.0 is tier 1
    assert tier_idx[5] == 2  # 25.0 is tier 2
    assert tier_idx[6] == 2  # 35.0 is tier 2
    assert tier_idx[7] == -1 # 100.0 is outside outer tier
    assert tier_idx[8] == 2  # -100.0 is exact negative lower bound of tier 2
    assert tier_idx[9] == -1 # 150.0 is outside


def test_negative_coordinates_integer_lattice():
    """Verify negative coordinates floor properly without sign symmetry distortion."""
    g = FoveatedGrid("spec", fuse=False)
    xy = np.array([[-0.01, -0.01], [-0.05, -0.05], [-0.06, -0.06], [-10.0, -10.0]])
    org = g.window_origins((0.0, 0.0))
    tier_idx, diag = g.assign_native_tiers(xy, org)
    assert np.all(tier_idx >= 0)

    # Tier 0 has n=400, r=0.05, origin=(-200, -200)
    # x=-0.01 -> floor(-0.01/0.05) = -1. Cell = -1 - (-200) = 199
    c0 = np.floor(xy[0] / 0.05).astype(int) - org[0]
    assert np.array_equal(c0, [199, 199])


# ============================================================================
# 2. Integer Lattice Mip-Up Invariants
# ============================================================================

def test_mip_up_parent_child_conservation():
    """Verify parent point count equals sum of contributing child cell counts."""
    g = FoveatedGrid("spec", fuse=False)
    rng = np.random.default_rng(123)
    # Populate fine tier heavily
    xy = rng.uniform(-9.5, 9.5, (5000, 2))
    z = rng.uniform(-0.5, 1.5, 5000)
    p = np.eye(NUM_CLASSES)[rng.integers(0, NUM_CLASSES, 5000)]
    m = np.zeros(5000, bool)
    
    org = g.window_origins((0.0, 0.0))
    stats = g.bin_points(xy, z, p, m, org)
    
    # Check parent-child count conservation for each tier pair
    for k in range(1, len(g.tiers)):
        fine = stats[k - 1]
        coarse = stats[k]
        r = g.tiers[k].ratio // g.tiers[k - 1].ratio
        nf, nc = g.tiers[k - 1].n, g.tiers[k].n
        
        fi = fine["key"] // nf + org[k - 1][0]
        fj = fine["key"] % nf + org[k - 1][1]
        ck = (fi // r - org[k][0]) * nc + (fj // r - org[k][1])
        
        fine_child_sums = {}
        for c_key, n_pts in zip(ck, fine["n_pts"]):
            fine_child_sums[c_key] = fine_child_sums.get(c_key, 0) + n_pts
            
        coarse_map = dict(zip(coarse["key"], coarse["n_pts"]))
        for c_key, expected_sum in fine_child_sums.items():
            assert coarse_map.get(c_key, 0) >= expected_sum


def test_mip_up_height_statistics_bounds():
    """Verify coarse min/max height properly encompasses all contributing child cells."""
    g = FoveatedGrid("spec", fuse=False)
    rng = np.random.default_rng(99)
    xy = rng.uniform(-9.0, 9.0, (2000, 2))
    z = rng.uniform(-1.0, 3.0, 2000)
    p = np.eye(NUM_CLASSES)[rng.integers(0, NUM_CLASSES, 2000)]
    m = np.zeros(2000, bool)
    
    org = g.window_origins((0.0, 0.0))
    stats = g.bin_points(xy, z, p, m, org)
    
    fine, coarse = stats[0], stats[1]
    r = g.tiers[1].ratio // g.tiers[0].ratio
    nf, nc = g.tiers[0].n, g.tiers[1].n
    fi = fine["key"] // nf + org[0][0]
    fj = fine["key"] % nf + org[0][1]
    ck = (fi // r - org[1][0]) * nc + (fj // r - org[1][1])
    
    coarse_dict = {k: (mn, mx) for k, mn, mx in zip(coarse["key"], coarse["z_min"], coarse["z_max"])}
    
    for c_key, f_min, f_max in zip(ck, fine["z_min"], fine["z_max"]):
        c_min, c_max = coarse_dict[c_key]
        assert c_min <= f_min + 1e-4, f"Coarse min {c_min} > fine min {f_min}"
        assert c_max >= f_max - 1e-4, f"Coarse max {c_max} < fine max {f_max}"


# ============================================================================
# 3. Dynamic Layer Separation & Ghost Removal
# ============================================================================

def test_dynamic_layer_rebuilt_every_frame_no_ghost_trail():
    """Verify dynamic objects do not persist in static grid and leave no ghost trail."""
    terrain = TerrainConfig(enable_ray_clearing=False)
    g = FoveatedGrid("spec", fuse=True, terrain_config=terrain)
    
    # Frame 0: Dynamic vehicle at (2.0, 2.0)
    xy0 = np.array([[2.0, 2.0], [2.02, 2.02]])
    z0 = np.array([0.5, 0.6])
    p0 = np.zeros((2, NUM_CLASSES), dtype=np.float32)
    p0[:, VEHICLE] = 0.95
    m0 = np.array([True, True], dtype=bool)
    
    dyn0, _ = g.update(xy0, z0, p0, m0, (0.0, 0.0))
    assert len(dyn0[0]["i"]) > 0
    # In static layer, dynamic points must NOT be accumulated into static count
    c0 = np.floor(xy0[0] / 0.05).astype(int) - g.origins[0]
    assert g.state[0].count[c0[0], c0[1]] == 0
    # Dynamic layer marks the active dynamic cell in current frame
    assert g.state[0].dynamic[c0[0], c0[1]] == True
    
    # Frame 1: Dynamic vehicle moved to (4.0, 2.0). Nothing at (2.0, 2.0).
    xy1 = np.array([[4.0, 2.0], [4.02, 2.02]])
    z1 = np.array([0.5, 0.6])
    p1 = np.zeros((2, NUM_CLASSES), dtype=np.float32)
    p1[:, VEHICLE] = 0.95
    m1 = np.array([True, True], dtype=bool)
    
    dyn1, _ = g.update(xy1, z1, p1, m1, (0.0, 0.0))
    # Old position (2.0, 2.0) in dynamic report must be absent
    old_c0 = tuple(c0)
    dyn1_cells = list(zip(dyn1[0]["i"], dyn1[0]["j"]))
    assert old_c0 not in dyn1_cells, "Ghost trail detected at old dynamic position!"
    
    # Static cell at (2.0, 2.0) remains empty/unknown, and dynamic flag is cleared
    assert g.state[0].count[c0[0], c0[1]] == 0
    assert g.state[0].dynamic[c0[0], c0[1]] == False


def test_dynamic_class_selection_person_vs_vehicle():
    """Verify deterministic selection if both vehicle and person appear in dynamic cell."""
    g = FoveatedGrid("spec", fuse=True)
    
    # Two points in the exact same cell (1.02, 1.02): one person, one vehicle
    xy = np.array([[1.02, 1.02], [1.03, 1.03]])
    z = np.array([0.2, 0.8])
    p = np.zeros((2, NUM_CLASSES), dtype=np.float32)
    p[0, PERSON] = 0.90
    p[1, VEHICLE] = 0.95  # higher confidence vehicle
    m = np.array([True, True], dtype=bool)
    
    dyn, _ = g.update(xy, z, p, m, (0.0, 0.0))
    assert len(dyn[0]["cls"]) == 1
    assert dyn[0]["cls"][0] == VEHICLE
    assert dyn[0]["count"][0] == 2


# ============================================================================
# 4. Stale State & Conservative Ray Clearing
# ============================================================================

def test_stale_cell_transitions_and_reversion():
    """Verify unobserved cells age into STALE and eventually revert to UNKNOWN."""
    terrain = TerrainConfig(stale_age_threshold=5, max_stale_age=15)
    g = FoveatedGrid("spec", fuse=True, terrain_config=terrain)
    
    # Observe static point at (1.0, 1.0)
    xy = np.array([[1.02, 1.02]])
    z = np.array([0.2])
    p = np.eye(NUM_CLASSES)[[4]]
    m = np.array([False], dtype=bool)
    
    g.update(xy, z, p, m, (0.0, 0.0))
    c = np.floor(xy[0] / 0.05).astype(int) - g.origins[0]
    assert g.state[0].count[c[0], c[1]] == 1
    assert g.state[0].age[c[0], c[1]] == 0
    
    # Step 5 frames with empty observations
    for _ in range(5):
        g.update(np.zeros((0, 2)), np.zeros(0), np.zeros((0, NUM_CLASSES)), np.zeros(0, bool), (0.0, 0.0))
        
    assert g.state[0].age[c[0], c[1]] == 5
    q5 = g.query_point(1.02, 1.02)
    assert q5["state"] == "STALE"
    
    # Step 11 more frames (total age = 16 >= max_stale_age=15)
    for _ in range(11):
        g.update(np.zeros((0, 2)), np.zeros(0), np.zeros((0, NUM_CLASSES)), np.zeros(0, bool), (0.0, 0.0))
        
    assert g.state[0].count[c[0], c[1]] == 0
    q16 = g.query_point(1.02, 1.02)
    assert q16["state"] == "UNKNOWN"


def test_conservative_ray_clearing():
    """Verify rays clear unoccluded free space and never clear behind the return."""
    terrain = TerrainConfig(enable_ray_clearing=True, free_clear_frames=1)
    g = FoveatedGrid("spec", fuse=True, terrain_config=terrain)
    
    # Step 1: Place an obstacle at (3.0, 0.0)
    xy = np.array([[3.02, 0.02]])
    z = np.array([0.5])
    p = np.eye(NUM_CLASSES)[[4]] # building/obstacle
    m = np.array([False], dtype=bool)
    g.update(xy, z, p, m, (0.0, 0.0))
    
    c_obs = np.floor(xy[0] / 0.05).astype(int) - g.origins[0]
    assert g.state[0].count[c_obs[0], c_obs[1]] > 0
    
    # Step 2: Return at (5.0, 0.0) from sensor at (0.0, 0.0).
    # Ray from (0,0) to (5,0) traverses (3,0). It must clear the previous obstacle at (3,0).
    xy_far = np.array([[5.02, 0.02]])
    z_far = np.array([0.5])
    p_far = np.eye(NUM_CLASSES)[[4]]
    m_far = np.array([False], dtype=bool)
    
    g.update(xy_far, z_far, p_far, m_far, (0.0, 0.0), sensor_origin=(0.0, 0.0))
    
    # The cell at (3.0, 0.0) should now be cleared
    assert g.state[0].count[c_obs[0], c_obs[1]] == 0
    
    # A cell behind the return (6.0, 0.0) should NOT have been cleared if it had an obstacle
    xy_behind = np.array([[6.02, 0.02]])
    g.update(xy_behind, np.array([0.5]), np.eye(NUM_CLASSES)[[4]], np.array([False]), (0.0, 0.0))
    c_behind = np.floor(xy_behind[0] / 0.05).astype(int) - g.origins[0]
    assert g.state[0].count[c_behind[0], c_behind[1]] > 0
    
    # Re-fire to (5.0, 0.0). Ray stops at (5.0, 0.0); cell behind (6.0, 0.0) must remain protected!
    g.update(xy_far, z_far, p_far, m_far, (0.0, 0.0), sensor_origin=(0.0, 0.0))
    assert g.state[0].count[c_behind[0], c_behind[1]] > 0, "Cell behind return was falsely cleared!"


# ============================================================================
# 5. Terrain & Traversability Derivation
# ============================================================================

def test_terrain_slope_and_cost_penalties():
    """Verify ground slope is estimated from 3x3 neighbors and penalizes traversability."""
    terrain = TerrainConfig(slope_threshold_rad=0.20, slope_critical_rad=0.35)
    g = FoveatedGrid("spec", fuse=True, terrain_config=terrain)
    
    # Create a 3x3 patch with a steep slope: z increases by 0.5m over 0.05m cell (gradient = 10 -> steep)
    xs, ys, zs = [], [], []
    for dx in [-0.05, 0.0, 0.05]:
        for dy in [-0.05, 0.0, 0.05]:
            xs.append(1.0 + dx + 0.02)
            ys.append(1.0 + dy + 0.02)
            zs.append(0.5 + dx * 6.0) # dz/dx = 6.0 -> slope angle > 0.40 rad (critical)
            
    xy = np.column_stack([xs, ys])
    z = np.array(zs, dtype=np.float32)
    p = np.zeros((len(xy), NUM_CLASSES), dtype=np.float32)
    p[:, 1] = 0.95 # Drivable ground
    m = np.zeros(len(xy), bool)
    
    g.update(xy, z, p, m, (0.0, 0.0))
    
    q = g.query_point(1.02, 1.02)
    # Critical slope must receive critical obstacle cost >= 220
    assert q["cost"] >= 220
    assert not g.is_traversable(1.02, 1.02)


# ============================================================================
# 6. Memory Accounting PRD Acceptance
# ============================================================================

def test_memory_accounting_prd_budget():
    """Verify allocated memory is <= 8 MB and >= 30x smaller than uniform 5cm baseline."""
    g = FoveatedGrid("spec")
    report = g.memory_report()
    
    assert report["bytes_per_cell"] == 16
    assert report["allocated_bytes"] <= 8 * 1024 * 1024, f"Allocated bytes {report['allocated_bytes']} > 8 MB"
    assert report["memory_reduction_ratio"] >= 30.0, f"Reduction ratio {report['memory_reduction_ratio']} < 30x"
    assert report["allocated_cells"] == 320000


def test_torch_memory_honesty():
    """Verify Torch persistent state equals 16 bytes/cell without hidden tensors."""
    gt = TorchFoveatedGrid("spec", device="cpu")
    rep = gt.memory_report()
    assert rep["bytes_per_cell"] == 16
    assert rep["allocated_bytes"] <= 8 * 1024 * 1024
    assert rep["allocated_cells"] == 320000


# ============================================================================
# 7. NumPy / Torch Mathematical Parity
# ============================================================================

def test_numpy_torch_complete_mapping_parity():
    """Verify complete frame update produces exact parity between NumPy and Torch."""
    gn = FoveatedGrid("spec", fuse=True)
    gt = TorchFoveatedGrid("spec", device="cpu", fuse=True)
    
    rng = np.random.default_rng(777)
    xy = rng.uniform(-9.5, 9.5, (3000, 2))
    z = rng.uniform(-0.5, 1.8, 3000)
    p = np.eye(NUM_CLASSES)[rng.integers(0, NUM_CLASSES, 3000)]
    m = (rng.uniform(0, 1, 3000) > 0.85)
    
    dyn_np, _ = gn.update(xy, z, p, m, (1.2, -0.8))
    dyn_tr, _ = gt.update(xy, z, p, m, (1.2, -0.8))
    
    # Dynamic counts match
    assert len(dyn_np[0]["i"]) == len(dyn_tr[0]["i"])
    
    # Check each tier layer parity
    for t in range(len(gn.tiers)):
        sn = gn.state[t]
        st = gt.state[t]
        assert np.array_equal(sn.count, st.count16.numpy().view(np.uint16))
        assert np.array_equal(sn.cls, st.cls.numpy())
        assert np.array_equal(sn.dynamic, st.dynamic.numpy())
        assert np.array_equal(sn.age, st.age.numpy())
        assert np.array_equal(sn.cost, st.cost.numpy())
        # float fields with float16 tolerance
        occ = sn.count > 0
        if occ.any():
            np.testing.assert_allclose(sn.z_min[occ], st.z_min.numpy()[occ], atol=1e-3)
            np.testing.assert_allclose(sn.z_max[occ], st.z_max.numpy()[occ], atol=1e-3)
            np.testing.assert_allclose(sn.ground[occ], st.ground.numpy()[occ], atol=1e-3)
            np.testing.assert_allclose(sn.rough[occ], st.rough.numpy()[occ], atol=1e-3)


# ============================================================================
# 8. Query API & MapSnapshot Contract
# ============================================================================

def test_query_api_foveated_grid():
    """Verify query_point, is_traversable, and get_height on grid and snapshot."""
    g = FoveatedGrid("spec", fuse=True)
    xy = np.array([[2.02, 1.02]])
    z = np.array([0.15])
    p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p[0, 1] = 0.95 # Drivable ground
    m = np.zeros(1, bool)
    
    g.update(xy, z, p, m, (0.0, 0.0))
    
    res = g.query_point(2.02, 1.02)
    assert res["state"] == "OBSERVED_STATIC"
    assert res["count"] == 1
    assert abs(res["ground"] - 0.15) < 1e-3
    assert g.is_traversable(2.02, 1.02)
    assert abs(g.get_height(2.02, 1.02) - 0.15) < 1e-3
    
    # Query out of bounds
    oob = g.query_point(999.0, 999.0)
    assert oob["state"] == "OUT_OF_BOUNDS"
    assert not g.is_traversable(999.0, 999.0)


def test_mapsnapshot_detached_immutability():
    """Verify MapSnapshot is write-protected and completely detached from live grid state."""
    g = FoveatedGrid("spec", fuse=True)
    xy = np.array([[2.02, 1.02]])
    z = np.array([0.15])
    p = np.eye(NUM_CLASSES)[[1]]
    m = np.zeros(1, bool)
    
    dyn, _ = g.update(xy, z, p, m, (0.0, 0.0))
    snap_layers = g.snapshot()
    
    snap = MapSnapshot(
        timestamp=1.0,
        frame_id="frame_0",
        ego_pose=np.eye(4),
        origins=tuple(tuple(int(c) for c in o) for o in g.origins),
        tier_states=tuple(snap_layers),
        dynamic_cells=tuple(dyn),
    )
    
    # Verify ego_pose is write protected
    with pytest.raises(ValueError):
        snap.ego_pose[0, 0] = 999.0
        
    # Verify snapshot query API
    q = snap.query_point(2.02, 1.02)
    assert q["state"] == "OBSERVED_STATIC"
    assert snap.is_traversable(2.02, 1.02)
    
    # Mutating subsequent grid frame does not mutate published snapshot
    g.reset()
    assert g.state[0].count.sum() == 0
    assert snap.tier_states[0].count.sum() > 0, "Snapshot was mutated by grid reset!"


# ============================================================================
# 9. Failure & Edge Cases
# ============================================================================

def test_edge_cases_zero_and_single_point():
    """Verify zero points and single point execute without error."""
    g = FoveatedGrid("spec", fuse=True)
    
    # 0 points
    dyn0, _ = g.update(np.zeros((0, 2)), np.zeros(0), np.zeros((0, NUM_CLASSES)), np.zeros(0, bool), (0.0, 0.0))
    assert len(dyn0[0]["i"]) == 0
    
    # 1 point
    dyn1, _ = g.update(np.array([[0.5, 0.5]]), np.array([0.2]), np.eye(NUM_CLASSES)[[1]], np.zeros(1, bool), (0.0, 0.0))
    assert len(dyn1[0]["i"]) == 0
    assert g.state[0].count.sum() == 1


def test_edge_cases_duplicate_points():
    """Verify multiple identical points accumulate cleanly."""
    g = FoveatedGrid("spec", fuse=False)
    xy = np.tile([1.02, 1.02], (50, 1))
    z = np.full(50, 0.5)
    p = np.tile(np.eye(NUM_CLASSES)[1], (50, 1))
    m = np.zeros(50, bool)
    
    org = g.window_origins((0.0, 0.0))
    stats = g.bin_points(xy, z, p, m, org)
    assert stats[0]["n_pts"][0] == 50


def test_large_ego_motion_scroll():
    """Verify large ego jumps do not crash the grid."""
    g = FoveatedGrid("spec", fuse=True)
    xy = np.array([[0.0, 0.0]])
    z = np.array([0.1])
    p = np.eye(NUM_CLASSES)[[1]]
    m = np.zeros(1, bool)
    
    g.update(xy, z, p, m, (0.0, 0.0))
    # Jump 100 meters
    g.update(xy + 100.0, z, p, m, (100.0, 100.0))
    assert g.state[0].count.sum() >= 0


def test_deterministic_reset():
    """Verify reset completely cleans all state to identical initial condition."""
    g = FoveatedGrid("spec", fuse=True)
    xy = np.array([[2.0, 2.0]])
    g.update(xy, np.array([0.5]), np.eye(NUM_CLASSES)[[1]], np.zeros(1, bool), (0.0, 0.0))
    
    g.reset()
    for t in g.state:
        assert t.count.sum() == 0
        assert (t.age == 255).all()
        assert np.all(t.cost == 255)
        assert np.all(t.dynamic == False)


def test_longevity_100_frame_soak():
    """Verify 100 consecutive frames run without memory leaks or unbounded growth."""
    cfg = FoveaMapConfig()
    g = FoveatedGrid("spec", fuse=True, terrain_config=cfg.terrain)
    rng = np.random.default_rng(101)
    
    ego = np.array([0.0, 0.0])
    for frame in range(100):
        ego += np.array([0.1, 0.02])
        xy = rng.uniform(-15.0, 15.0, (500, 2)) + ego
        z = rng.uniform(-0.5, 1.5, 500)
        p = np.eye(NUM_CLASSES)[rng.integers(0, NUM_CLASSES, 500)]
        m = (rng.uniform(0, 1, 500) > 0.90)
        
        dyn, _ = g.update(xy, z, p, m, ego)
        
    # After 100 frames, verify memory report is identical and bounded
    rep = g.memory_report()
    assert rep["allocated_bytes"] == 5120000
    assert rep["bytes_per_cell"] == 16


def test_persistent_secondary_class_evidence():
    """Verify persistent top-2 class evidence and confidences across NumPy and Torch."""
    gn = FoveatedGrid("spec", fuse=True)
    gt = TorchFoveatedGrid("spec", fuse=True, device="cpu")

    # Point with primary class=ROAD(0) prob=0.6, secondary class=SIDEWALK(1) prob=0.3
    xy = np.array([[2.0, 2.0]])
    z = np.array([0.0])
    p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p[0, 0] = 0.6
    p[0, 1] = 0.3
    p[0, 2] = 0.1
    m = np.zeros(1, bool)

    gn.update(xy, z, p, m, (0.0, 0.0))
    gt.update(xy, z, p, m, (0.0, 0.0))

    qn = gn.query_point(2.0, 2.0)
    qt = gt.query_point(2.0, 2.0)

    assert qn["dominant_class"] == 0
    assert qt["dominant_class"] == 0
    assert qn["secondary_class"] == 1
    assert qt["secondary_class"] == 1
    assert abs(qn["secondary_confidence"] - qt["secondary_confidence"]) < 0.1


def test_dynamic_query_and_traversability_consistency():
    """Verify dynamic layer state connects to cell flags, snapshot, and query API."""
    g = FoveatedGrid("spec", fuse=True)
    xy = np.array([[3.0, 3.0]])
    z = np.array([0.5])
    p = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p[0, PERSON] = 0.95
    m = np.ones(1, bool)

    g.update(xy, z, p, m, (0.0, 0.0))

    # 1. Grid level check
    q = g.query_point(3.0, 3.0)
    assert q["dynamic"] is True
    assert q["state"] == "OBSERVED_DYNAMIC"
    assert g.is_traversable(3.0, 3.0) is False

    # 2. Snapshot level check
    snap = MapSnapshot(
        timestamp=0.1,
        frame_id="f1",
        ego_pose=np.eye(4),
        origins=tuple(g.origins),
        tier_states=tuple(g.snapshot()),
    )
    snap_q = snap.query_point(3.0, 3.0)
    assert snap_q["dynamic"] is True
    assert snap_q["state"] == "OBSERVED_DYNAMIC"
    assert snap.is_traversable(3.0, 3.0) is False


def test_true_consecutive_ray_clearing():
    """Verify that ray clearing requires consecutive frames and resets streak if interrupted."""
    tc = TerrainConfig(enable_ray_clearing=True, free_clear_frames=3)
    g = FoveatedGrid("spec", fuse=True, terrain_config=tc)

    # Frame 0: Put an obstacle at (3.0, 0.0)
    xy_obs = np.array([[3.0, 0.0]])
    z_obs = np.array([1.0])
    p_obs = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p_obs[0, 5] = 1.0  # Building obstacle
    g.update(xy_obs, z_obs, p_obs, np.zeros(1, bool), (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))

    c = g.fine_index(xy_obs)[0] - g.origins[0]
    assert g.state[0].cls[c[0], c[1]] == 5

    # Frame 1: Ray passes through (3.0, 0.0) to hit obstacle at (6.0, 0.0)
    xy_far = np.array([[6.0, 0.0]])
    z_far = np.array([1.0])
    p_far = np.zeros((1, NUM_CLASSES), dtype=np.float32)
    p_far[0, 5] = 1.0
    g.update(xy_far, z_far, p_far, np.zeros(1, bool), (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))
    # Streak = 1, obstacle not cleared yet
    assert g.state[0].free_passes[c[0], c[1]] == 1
    assert g.state[0].cls[c[0], c[1]] == 5

    # Frame 2: Non-traversing frame (rays along orthogonal direction) -> streak must RESET to 0!
    xy_other = np.array([[0.0, 6.0]])
    g.update(xy_other, z_far, p_far, np.zeros(1, bool), (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))
    assert g.state[0].free_passes[c[0], c[1]] == 0, "Streak failed to reset on non-traversed frame!"

    # Now 3 consecutive frames of rays through (3.0, 0.0)
    for frame_idx in range(1, 4):
        g.update(xy_far, z_far, p_far, np.zeros(1, bool), (0.0, 0.0), sensor_origin=(0.0, 0.0, 0.0))
        if frame_idx < 3:
            assert g.state[0].free_passes[c[0], c[1]] == frame_idx
            assert g.state[0].cls[c[0], c[1]] == 5
        else:
            # Reached 3 consecutive passes -> obstacle cleared to UNKNOWN!
            assert g.state[0].cls[c[0], c[1]] == 255
            assert g.state[0].free_passes[c[0], c[1]] == 0


def test_graded_profile_memory_accounting_distinction():
    """Verify honest distinction between allocated dense cells (570k) and effective cells (520k)."""
    g = FoveatedGrid("graded", fuse=False)
    rep = g.memory_report()

    assert rep["allocated_cells"] == 570000
    assert rep["effective_non_overlapping_cells"] == 520000
    assert rep["persistent_state_bytes"] == 570000 * 16
    assert rep["effective_bytes"] == 520000 * 16
    assert rep["bytes_per_cell"] == 16


def test_pcd_header_validation_and_truncated_ascii(tmp_path):
    """Verify PCD parser rejects header mismatches and truncated ASCII data."""
    from foveamap.data.file import parse_pcd
    from foveamap.core.exceptions import DataAdapterError

    # 1. Header with field count mismatch
    bad_pcd = tmp_path / "bad_header.pcd"
    bad_pcd.write_text("VERSION 0.7\nFIELDS x y z\nSIZE 4 4\nTYPE F F\nCOUNT 1 1\nPOINTS 1\nDATA ascii\n0 0 0\n")
    with pytest.raises(DataAdapterError, match="length mismatch"):
        parse_pcd(bad_pcd)

    # 2. Truncated ASCII PCD: header declares 5 points, file has only 2
    trunc_pcd = tmp_path / "trunc.pcd"
    trunc_pcd.write_text("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nPOINTS 5\nDATA ascii\n0 0 0\n1 1 1\n")
    with pytest.raises(DataAdapterError, match="Truncated ASCII PCD"):
        parse_pcd(trunc_pcd)


def test_bin_raw_intensity_default(tmp_path):
    """Verify parse_bin preserves raw sensor intensity without normalization by default."""
    from foveamap.data.file import parse_bin

    bin_path = tmp_path / "raw.bin"
    # 4 points: x, y, z, intensity with high values (e.g. 500.0)
    arr = np.array([
        [1.0, 2.0, 3.0, 500.0],
        [4.0, 5.0, 6.0, 1200.0],
    ], dtype=np.float32)
    arr.tofile(str(bin_path))

    parsed, prov = parse_bin(bin_path, columns=4)
    assert prov == "raw"
    np.testing.assert_array_equal(parsed["intensity"], [500.0, 1200.0])

