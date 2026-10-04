"""Adversarial and Stress Hardening Test Suite (Phase 11 Engineering Gate).

Covers edge cases, numerical anomalies, boundary behaviors, and lifecycle stress:
1. Empty point clouds (0 points) in Grid, TorchGrid, Runtime, and Pipeline.
2. Numerical anomaly rejection (NaN, Inf, negative intensity, mismatched shapes) in contracts.
3. Extreme out-of-bounds coordinates ($10^6$ m) and single-point sweeps.
4. Tier boundary lattice parity between NumPy and PyTorch engines.
5. Runtime lifecycle stress: 15+ repeated reset-process cycles, memory stability.
"""
import numpy as np
import pytest
import torch

from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import ContractError, NumericalConsistencyError
from foveamap.core.ontology import NUM_CLASSES, ROAD, VEHICLE
from foveamap.grid import FoveatedGrid
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.core.config import FoveaMapConfig
from foveamap.runtime import FoveaMapRuntime


class TestAdversarialEmptyInputs:
    """Verifies that 0-point inputs never crash any layer of the system."""

    def test_empty_lidar_frame_contract(self):
        empty_frame = LiDARFrame(
            pts=np.zeros((0, 3), dtype=np.float32),
            intensity=np.zeros((0,), dtype=np.float32),
            ring=np.zeros((0,), dtype=np.int16),
            pose=np.eye(4, dtype=np.float64),
            frame_id="empty_frame_0",
        )
        assert empty_frame.num_points == 0
        assert not empty_frame.has_annotations

    def test_empty_update_numpy_and_torch_parity(self):
        gn = FoveatedGrid("spec", fuse=True)
        gt = TorchFoveatedGrid("spec", fuse=True, device="cpu")

        xy_0 = np.zeros((0, 2), dtype=np.float32)
        z_0 = np.zeros((0,), dtype=np.float32)
        p_0 = np.zeros((0, NUM_CLASSES), dtype=np.float32)
        m_0 = np.zeros((0,), dtype=bool)

        dyn_n, stats_n = gn.update(xy_0, z_0, p_0, m_0, (0.0, 0.0))
        dyn_t, stats_t = gt.update(
            torch.from_numpy(xy_0),
            torch.from_numpy(z_0),
            torch.from_numpy(p_0),
            torch.from_numpy(m_0),
            (0.0, 0.0),
        )

        assert len(dyn_n) == len(gn.tiers)
        assert len(dyn_t) == len(gt.tiers)
        for sn, st in zip(stats_n, stats_t):
            assert sn["n_in"] == 0
            assert st["n_in"] == 0

    def test_empty_frame_in_runtime_produces_safe_snapshot(self):
        runtime = FoveaMapRuntime(config=FoveaMapConfig.cpu_dev())
        empty_frame = LiDARFrame(
            pts=np.zeros((0, 3), dtype=np.float32),
            intensity=np.zeros((0,), dtype=np.float32),
            ring=np.zeros((0,), dtype=np.int16),
            pose=np.eye(4, dtype=np.float64),
            frame_id="empty_0",
        )

        snap = runtime.process(empty_frame)
        assert snap is not None
        assert snap.metadata.get("empty_frame") is True
        assert snap.metadata.get("dropped_frames") == 1
        assert runtime.frame_count == 1

        metrics = runtime.get_metrics()
        assert metrics["dropped_frames"] == 1


class TestAdversarialNumericalAnomalies:
    """Verifies strict rejection of numerical corruptions and graceful handling of extreme coordinates."""

    def test_nan_coordinates_rejected_by_contract(self):
        pts = np.array([[1.0, 2.0, np.nan]], dtype=np.float32)
        with pytest.raises(NumericalConsistencyError, match="NaN or Inf"):
            LiDARFrame(
                pts=pts,
                intensity=np.ones((1,), dtype=np.float32),
                ring=np.zeros((1,), dtype=np.int16),
                pose=np.eye(4, dtype=np.float64),
            )

    def test_inf_coordinates_rejected_by_contract(self):
        pts = np.array([[np.inf, 2.0, 0.0]], dtype=np.float32)
        with pytest.raises(NumericalConsistencyError, match="NaN or Inf"):
            LiDARFrame(
                pts=pts,
                intensity=np.ones((1,), dtype=np.float32),
                ring=np.zeros((1,), dtype=np.int16),
                pose=np.eye(4, dtype=np.float64),
            )

    def test_negative_intensity_rejected_by_contract(self):
        pts = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
        with pytest.raises(NumericalConsistencyError, match="negative values"):
            LiDARFrame(
                pts=pts,
                intensity=np.array([-0.5], dtype=np.float32),
                ring=np.zeros((1,), dtype=np.int16),
                pose=np.eye(4, dtype=np.float64),
            )

    def test_mismatched_shape_rejected_by_contract(self):
        pts = np.array([[1.0, 2.0, 0.0], [2.0, 3.0, 0.0]], dtype=np.float32)
        with pytest.raises(ContractError, match="LiDARFrame.intensity must be"):
            LiDARFrame(
                pts=pts,
                intensity=np.ones((1,), dtype=np.float32),  # shape 1 vs points 2
                ring=np.zeros((2,), dtype=np.int16),
                pose=np.eye(4, dtype=np.float64),
            )

    def test_extreme_out_of_bounds_coordinates_handled_safely(self):
        gn = FoveatedGrid("spec", fuse=False)
        gt = TorchFoveatedGrid("spec", fuse=False, device="cpu")

        xy_ext = np.array([[1e6, -1e6], [-50000.0, 20000.0], [1.0, 1.0]], dtype=np.float32)
        z_ext = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        p_ext = np.repeat(np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32), 3, axis=0)
        m_ext = np.zeros(3, dtype=bool)

        dyn_n, stats_n = gn.update(xy_ext, z_ext, p_ext, m_ext, (0.0, 0.0))
        dyn_t, stats_t = gt.update(
            torch.from_numpy(xy_ext),
            torch.from_numpy(z_ext),
            torch.from_numpy(p_ext),
            torch.from_numpy(m_ext),
            (0.0, 0.0),
        )

        # Only the (1.0, 1.0) point should be binned in the finest tier
        assert stats_n[0]["n_in"] == 1
        assert stats_t[0]["n_in"] == 1
        # The extreme points (> 1000m) are safely dropped by the outer window without crash
        assert stats_n[-1]["n_in"] == 1
        assert stats_t[-1]["n_in"] == 1


class TestAdversarialBoundaryParity:
    """Verifies exact tier boundary lattice behaviors between NumPy and PyTorch."""

    def test_exact_tier_boundary_points(self):
        gn = FoveatedGrid("spec", fuse=False)
        gt = TorchFoveatedGrid("spec", fuse=False, device="cpu")

        # Boundary coordinates: tier 0 half-extent is 10.0m, tier 1 is 20.0m
        pts = np.array([
            [10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 10.0],
            [0.0, -10.0],
            [20.0, 0.0],
            [-20.0, 0.0],
            [9.999, 9.999],
            [10.001, 10.001],
            [19.999, 19.999],
            [20.001, 20.001],
        ], dtype=np.float32)

        z = np.zeros(len(pts), dtype=np.float32)
        p = np.repeat(np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32), len(pts), axis=0)
        m = np.zeros(len(pts), dtype=bool)

        _, stats_n = gn.update(pts, z, p, m, (0.0, 0.0))
        _, stats_t = gt.update(
            torch.from_numpy(pts),
            torch.from_numpy(z),
            torch.from_numpy(p),
            torch.from_numpy(m),
            (0.0, 0.0),
        )

        counts_n = [int(s["n_in"]) for s in stats_n]
        counts_t = [int(s["n_in"]) for s in stats_t]
        assert counts_n == counts_t

    def test_single_point_sweep(self):
        gn = FoveatedGrid("spec", fuse=False)
        gt = TorchFoveatedGrid("spec", fuse=False, device="cpu")

        # Cell interior coordinates (away from exact multiples of 0.05m cell boundary)
        xy_1 = np.array([[2.52, -3.18]], dtype=np.float32)
        z_1 = np.array([0.15], dtype=np.float32)
        p_1 = np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32)
        m_1 = np.zeros(1, dtype=bool)

        gn.update(xy_1, z_1, p_1, m_1, (0.0, 0.0))
        gt.update(
            torch.from_numpy(xy_1),
            torch.from_numpy(z_1),
            torch.from_numpy(p_1),
            torch.from_numpy(m_1),
            (0.0, 0.0),
        )

        qn = gn.query_point(2.52, -3.18)
        qt = gt.query_point(2.52, -3.18)

        assert qn is not None and qt is not None
        assert qn["dominant_class"] == ROAD
        assert qt["dominant_class"] == ROAD
        assert qn["ground"] == pytest.approx(0.15, abs=1e-3)
        assert qt["ground"] == pytest.approx(0.15, abs=1e-3)


class TestAdversarialRuntimeLifecycle:
    """Stress tests runtime lifecycle transitions and reset memory hygiene."""

    def test_repeated_runtime_resets_are_idempotent(self):
        runtime = FoveaMapRuntime(config=FoveaMapConfig.cpu_dev())

        pts = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float32)
        frame = LiDARFrame(
            pts=pts,
            intensity=np.ones(2, dtype=np.float32),
            ring=np.zeros(2, dtype=np.int16),
            pose=np.eye(4, dtype=np.float64),
            frame_id="frame_0",
        )

        for cycle in range(15):
            runtime.process(frame)
            assert runtime.frame_count == 1
            snap = runtime.snapshot()
            assert snap is not None

            # Reset
            runtime.reset()
            assert runtime.frame_count == 0
            assert runtime.snapshot() is None
            assert runtime._dropped_frames == 0
            metrics = runtime.get_metrics()
            assert metrics["fps"] == 0.0
            assert metrics["dropped_frames"] == 0


class TestAdversarialTemporalModel:
    """Verifies temporal model robustness against timestamp anomalies and index consistency."""

    def test_temporal_timestamp_discontinuity_and_backward_jump(self):
        from foveamap.temporal import DynamicWorldModel, DynamicObservation

        model = DynamicWorldModel()
        ob1 = [DynamicObservation(tier=0, i=10, j=10, x=1.0, y=1.0, cls=VEHICLE)]
        model.update(ob1, frame_idx=0, timestamp=1.0)
        assert len(model) == 1

        # Frame 1: valid 0.1s dt, moving 0.1m in x
        ob2 = [DynamicObservation(tier=0, i=10, j=10, x=1.1, y=1.0, cls=VEHICLE)]
        model.update(ob2, frame_idx=1, timestamp=1.1)
        track = list(model._tracks.values())[0]
        assert track.has_velocity
        assert track.vx == pytest.approx(1.0, abs=1e-2)

        # Frame 2: timestamp jumps backward (e.g. out-of-order packet / bag loop)
        ob3 = [DynamicObservation(tier=0, i=10, j=10, x=1.2, y=1.0, cls=VEHICLE)]
        model.update(ob3, frame_idx=2, timestamp=0.5)
        track = list(model._tracks.values())[0]
        assert not track.has_velocity
        assert track.vx == 0.0
        assert track.vy == 0.0

        # Frame 3: large timestamp jump (>5.0s discontinuity)
        ob4 = [DynamicObservation(tier=0, i=10, j=10, x=1.3, y=1.0, cls=VEHICLE)]
        model.update(ob4, frame_idx=3, timestamp=100.0)
        track = list(model._tracks.values())[0]
        assert not track.has_velocity
        assert track.vx == 0.0
        assert track.vy == 0.0

        # Frame 4: non-finite timestamp (NaN)
        ob5 = [DynamicObservation(tier=0, i=10, j=10, x=1.4, y=1.0, cls=VEHICLE)]
        model.update(ob5, frame_idx=4, timestamp=float("nan"))
        track = list(model._tracks.values())[0]
        assert track.last_seen == 4

    def test_temporal_cell_index_consistency(self):
        from foveamap.temporal import DynamicWorldModel, DynamicObservation

        model = DynamicWorldModel()
        ob1 = [DynamicObservation(tier=0, i=10, j=10, x=1.0, y=1.0, cls=VEHICLE)]
        model.update(ob1, frame_idx=0, timestamp=0.0)

        # O(1) lookup
        assert model.lookup(0, 10, 10) is not None
        assert model.lookup(0, 10, 11) is None
        assert model.lookup(1, 10, 10) is None

        # Track moves to cell (12, 10)
        ob2 = [DynamicObservation(tier=0, i=12, j=10, x=1.2, y=1.0, cls=VEHICLE)]
        model.update(ob2, frame_idx=1, timestamp=0.1)
        assert model.lookup(0, 10, 10) is None
        assert model.lookup(0, 12, 10) is not None

        # Window scrolls: origin advances by (2, 0)
        model.on_scroll(deltas=[(2, 0)], tier_n=[100])
        assert model.lookup(0, 12, 10) is None
        assert model.lookup(0, 10, 10) is not None

        # Reset cleans cell index completely
        model.reset()
        assert model.lookup(0, 10, 10) is None
        assert len(model._cell_index) == 0


class TestAdversarialIngestion:
    """Verifies file ingestion resilience to outliers and format ambiguity."""

    def test_outlier_robust_intensity_normalization(self):
        from foveamap.data.file import normalize_intensity

        rng = np.random.default_rng(42)
        base = rng.uniform(0.0, 100.0, size=1000).astype(np.float32)
        # Add single extreme retroreflector outlier (glint)
        base[0] = 50000.0

        normed, prov = normalize_intensity(base, mode="auto")
        # Ensure single glint does not trigger 16-bit scaling (dividing by 65535 gives ~0.0007)
        assert prov == "auto_scaled_8bit"
        median_val = float(np.median(normed[1:]))
        assert median_val > 0.15, f"Intensity collapsed by outlier: median is {median_val}"
        assert np.all(normed >= 0.0)
        assert np.all(normed <= 1.0)

    def test_parse_bin_column_ambiguity_resolution(self, tmp_path):
        from foveamap.data.file import parse_bin

        # 20 floats is divisible by both 4 (5 points) and 5 (4 points)
        data = np.arange(20, dtype=np.float32)
        bin_file = tmp_path / "test_ambiguous.bin"
        data.tofile(str(bin_file))

        # Default resolution: standard automotive 4 columns
        res_4, prov_4 = parse_bin(bin_file)
        pts_4 = res_4["pts"]
        intens_4 = res_4["intensity"]
        assert pts_4.shape == (5, 3)
        assert len(intens_4) == 5

        # Explicit 5 columns
        res_5, prov_5 = parse_bin(bin_file, columns=5)
        pts_5 = res_5["pts"]
        intens_5 = res_5["intensity"]
        assert pts_5.shape == (4, 3)
        assert len(intens_5) == 4

