"""Regression test suite for Phase 11 Pre-Deployment Hardening (Bugs #1 to #5).

Covers:
1. CUDA Device Canonicalization (Bug #1)
2. Packed Confidence Interpretation (Bug #2)
3. Ground Class Persistence Under Occlusion (Bug #3)
4. Clearance / Traversability Semantics (Bug #4)
5. Checkpoint Safety / Untrained Model Path (Bug #5)
"""

from __future__ import annotations

import os
from unittest.mock import patch
import numpy as np
import pytest
import torch

from foveamap.core.confidence import (
    pack_confidence_np,
    unpack_primary_confidence_np,
    unpack_secondary_confidence_np,
    pack_confidence_torch,
    unpack_primary_confidence_torch,
    unpack_secondary_confidence_torch,
)
from foveamap.core.config import FoveaMapConfig
from foveamap.core.ontology import NUM_CLASSES, ROAD, BUILDING, VEHICLE
from foveamap.grid import UNKNOWN
from foveamap.core.contracts import MapSnapshot
from foveamap.core.exceptions import (
    ConfigurationError,
    ContractError,
)
from foveamap.runtime.perception import (
    CheckpointNotFoundError,
    DevicePerceptionResult,
)
from foveamap.grid import FoveatedGrid
from foveamap.grid_torch import TorchFoveatedGrid
from foveamap.pipeline import FoveaMapPipeline, run_benchmark
from foveamap.runtime.device import canonicalize_device, devices_match


# ==============================================================================
# BUG #1: CUDA DEVICE CANONICALIZATION REGRESSION TESTS
# ==============================================================================

class TestBug1CUDADeviceCanonicalization:
    """Tests canonicalize_device and devices_match to prevent rejecting valid CUDA devices."""

    def test_cpu_vs_cpu(self):
        d1 = torch.device("cpu")
        d2 = torch.device("cpu")
        assert canonicalize_device(d1) == torch.device("cpu")
        assert devices_match(d1, d2)
        assert devices_match("cpu", "cpu")

    def test_cuda_vs_cuda_0_mocked(self):
        """Even if CUDA is not available or device index is None, 'cuda' matches 'cuda:0'."""
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.current_device", return_value=0):
            c_cuda = canonicalize_device("cuda")
            c_cuda0 = canonicalize_device("cuda:0")
            assert c_cuda == torch.device("cuda:0")
            assert c_cuda0 == torch.device("cuda:0")
            assert devices_match("cuda", "cuda:0")
            assert devices_match(torch.device("cuda"), torch.device("cuda:0"))

    def test_cuda_n_vs_cuda_n(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert devices_match("cuda:1", "cuda:1")
            assert devices_match(torch.device("cuda:2"), torch.device("cuda:2"))
            assert not devices_match("cuda:0", "cuda:1")

    def test_cpu_vs_cuda_mismatch(self):
        assert not devices_match("cpu", "cuda")
        assert not devices_match("cpu", "cuda:0")
        assert not devices_match(torch.device("cpu"), torch.device("cuda:0"))

    def test_device_perception_result_validation(self):
        """DevicePerceptionResult should accept matching logical devices and reject mismatches."""
        n_points = 10
        class_probs = torch.zeros((n_points, 19), dtype=torch.float32, device="cpu")
        moving_probs = torch.zeros((n_points,), dtype=torch.float32, device="cpu")
        semantic_preds = torch.zeros((n_points,), dtype=torch.int64, device="cpu")
        is_moving = torch.zeros((n_points,), dtype=torch.bool, device="cpu")
        confidence = torch.ones((n_points,), dtype=torch.float32, device="cpu")

        # Valid CPU result
        res = DevicePerceptionResult(
            class_probabilities=class_probs,
            moving_probabilities=moving_probs,
            semantic_predictions=semantic_preds,
            is_moving=is_moving,
            confidence=confidence,
            device=torch.device("cpu"),
        )
        assert res.device == torch.device("cpu")

        # Mismatched tensor device raises ContractError
        with pytest.raises(ContractError, match="mismatch with result device"):
            DevicePerceptionResult(
                class_probabilities=class_probs,
                moving_probabilities=moving_probs,
                semantic_predictions=semantic_preds,
                is_moving=is_moving,
                confidence=confidence,
                device=torch.device("cuda:0"),
            )

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Hardware CUDA not available")
    def test_real_cuda_canonicalization(self):
        dev_cuda = torch.device("cuda")
        dev_cuda0 = torch.device("cuda:0")
        assert devices_match(dev_cuda, dev_cuda0)


# ==============================================================================
# BUG #2: PACKED CONFIDENCE INTERPRETATION REGRESSION TESTS
# ==============================================================================

class TestBug2PackedConfidenceInterpretation:
    """Verifies packed confidence representation, pack/unpack helpers, and semantic consumption."""

    def test_pack_unpack_bounds_numpy(self):
        # Min
        b_min = pack_confidence_np(0.0, 0.0)
        assert b_min == 0
        assert unpack_primary_confidence_np(b_min) == 0.0
        assert unpack_secondary_confidence_np(b_min) == 0.0

        # Max
        b_max = pack_confidence_np(1.0, 1.0)
        assert b_max == 0xFF
        assert unpack_primary_confidence_np(b_max) == 1.0
        assert unpack_secondary_confidence_np(b_max) == 1.0

        # Mixed
        # primary 1.0 (nibble 15 = 0xF0), secondary 0.0 (nibble 0 = 0x00) -> 0xF0 (240)
        b_mixed1 = pack_confidence_np(1.0, 0.0)
        assert b_mixed1 == 0xF0
        assert unpack_primary_confidence_np(b_mixed1) == 1.0
        assert unpack_secondary_confidence_np(b_mixed1) == 0.0

        # primary 0.0, secondary 1.0 -> 0x0F (15)
        b_mixed2 = pack_confidence_np(0.0, 1.0)
        assert b_mixed2 == 0x0F
        assert unpack_primary_confidence_np(b_mixed2) == 0.0
        assert unpack_secondary_confidence_np(b_mixed2) == 1.0

    def test_pack_unpack_bounds_torch(self):
        p = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        s = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        packed = pack_confidence_torch(p, s)
        expected = torch.tensor([0x00, 0xFF, 0xF0, 0x0F], dtype=torch.uint8)
        assert torch.equal(packed, expected)

        unpacked_p = unpack_primary_confidence_torch(packed)
        unpacked_s = unpack_secondary_confidence_torch(packed)
        assert torch.allclose(unpacked_p, p)
        assert torch.allclose(unpacked_s, s)

    def test_numpy_torch_confidence_parity(self):
        p_vals = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        s_vals = np.linspace(1.0, 0.0, 16, dtype=np.float32)

        packed_np = pack_confidence_np(p_vals, s_vals)
        packed_th = pack_confidence_torch(torch.from_numpy(p_vals), torch.from_numpy(s_vals))
        assert np.array_equal(packed_np, packed_th.numpy())

        up_np = unpack_primary_confidence_np(packed_np)
        up_th = unpack_primary_confidence_torch(packed_th).numpy()
        assert np.allclose(up_np, up_th)

    def test_query_point_confidence_contract(self):
        """MapSnapshot and Grid query_point must return normalized confidence in [0.0, 1.0]."""
        grid = FoveatedGrid("spec")
        xy = np.array([[1.0, 1.0]], dtype=np.float32)
        z = np.array([0.0], dtype=np.float32)
        probs = np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32)
        moving = np.zeros(1, dtype=bool)
        grid.update(xy, z, probs, moving, (0.0, 0.0))

        res_grid = grid.query_point(1.0, 1.0)
        assert res_grid is not None
        assert res_grid["conf"] > 0
        assert res_grid["primary_confidence"] == pytest.approx(1.0, abs=0.1)
        assert res_grid["confidence"] == pytest.approx(1.0, abs=0.1)

        snap = MapSnapshot(
            timestamp=1.0,
            frame_id="f1",
            ego_pose=np.eye(4),
            origins=tuple(grid.origins),
            tier_states=tuple(grid.snapshot()),
        )
        res_snap = snap.query_point(1.0, 1.0)
        assert res_snap is not None
        assert res_snap["conf"] > 0
        assert res_snap["primary_confidence"] == pytest.approx(1.0, abs=0.1)
        assert res_snap["confidence"] == pytest.approx(1.0, abs=0.1)


# ==============================================================================
# BUG #3: GROUND CLASS PERSISTENCE UNDER OCCLUSION REGRESSION TESTS
# ==============================================================================

class TestBug3GroundClassPersistence:
    """Verifies that ground semantics persist under transient occlusion (e.g. overhangs, dynamic obstacles)."""

    def test_ground_persists_in_secondary_under_occlusion_numpy(self):
        grid = FoveatedGrid("spec", fuse=True)

        # Frame 1: Road (Ground) observed at (2.0, 0.0) with elevation 0.0
        xy_road = np.array([[2.0, 0.0], [2.05, 0.05], [1.95, -0.05]], dtype=np.float32)
        z_road = np.zeros(3, dtype=np.float32)
        p_road = np.repeat(np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32), 3, axis=0)
        m_road = np.zeros(3, dtype=bool)
        grid.update(xy_road, z_road, p_road, m_road, (0.0, 0.0))

        cell_f1 = grid.query_point(2.0, 0.0)
        assert cell_f1 is not None
        assert cell_f1["primary_class"] == ROAD

        # Subsequent frames: Vehicle / Overhang observed higher up at (2.0, 0.0) with elevation 2.5m
        # No ground return in these frames (clearance overhang or persistent obstacle)
        xy_obs = np.array([[2.0, 0.0], [2.05, 0.05]], dtype=np.float32)
        z_obs = np.full(2, 2.5, dtype=np.float32)
        p_obs = np.repeat(np.eye(NUM_CLASSES)[[VEHICLE]].astype(np.float32), 2, axis=0)
        m_obs = np.zeros(2, dtype=bool)
        # 2 frames to flip Bayesian EMA (alpha=0.3)
        grid.update(xy_obs, z_obs, p_obs, m_obs, (0.0, 0.0))
        grid.update(xy_obs, z_obs, p_obs, m_obs, (0.0, 0.0))

        cell_f2 = grid.query_point(2.0, 0.0)
        assert cell_f2 is not None
        # Primary is now the vehicle/obstacle
        assert cell_f2["primary_class"] == VEHICLE
        # BUT prior ground road is preserved in secondary evidence
        assert cell_f2["secondary_class"] == ROAD
        assert cell_f2["ground_elev"] == pytest.approx(0.0, abs=0.05)

    def test_ground_persists_under_occlusion_torch_parity(self):
        grid_np = FoveatedGrid("spec", fuse=True)
        grid_th = TorchFoveatedGrid("spec", fuse=True, device="cpu")

        xy_f1 = np.array([[3.0, 1.0], [3.05, 1.05]], dtype=np.float32)
        z_f1 = np.zeros(2, dtype=np.float32)
        p_f1 = np.repeat(np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32), 2, axis=0)
        m_f1 = np.zeros(2, dtype=bool)

        grid_np.update(xy_f1, z_f1, p_f1, m_f1, (0.0, 0.0))
        grid_th.update(
            torch.from_numpy(xy_f1),
            torch.from_numpy(z_f1),
            torch.from_numpy(p_f1),
            torch.from_numpy(m_f1),
            (0.0, 0.0),
        )

        xy_f2 = np.array([[3.0, 1.0]], dtype=np.float32)
        z_f2 = np.array([3.0], dtype=np.float32)
        p_f2 = np.eye(NUM_CLASSES)[[BUILDING]].astype(np.float32)
        m_f2 = np.zeros(1, dtype=bool)

        for _ in range(2):
            grid_np.update(xy_f2, z_f2, p_f2, m_f2, (0.0, 0.0))
            grid_th.update(
                torch.from_numpy(xy_f2),
                torch.from_numpy(z_f2),
                torch.from_numpy(p_f2),
                torch.from_numpy(m_f2),
                (0.0, 0.0),
            )

        q_np = grid_np.query_point(3.0, 1.0)
        q_th = grid_th.query_point(3.0, 1.0)

        assert q_np["primary_class"] == BUILDING
        assert q_th["primary_class"] == BUILDING
        assert q_np["secondary_class"] == ROAD
        assert q_th["secondary_class"] == ROAD

    def test_reset_clears_ground_persistence(self):
        grid = FoveatedGrid("spec")
        xy = np.array([[2.0, 0.0]], dtype=np.float32)
        z = np.zeros(1, dtype=np.float32)
        p = np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32)
        m = np.zeros(1, dtype=bool)
        grid.update(xy, z, p, m, (0.0, 0.0))
        grid.reset()
        cell = grid.query_point(2.0, 0.0)
        assert cell["is_unknown"] is True
        assert cell.get("primary_class", UNKNOWN) == UNKNOWN


# ==============================================================================
# BUG #4: CLEARANCE / TRAVERSABILITY SEMANTICS REGRESSION TESTS
# ==============================================================================

class TestBug4ClearanceTraversabilitySemantics:
    """Verifies that clearance=None means unlimited headroom (open road PASS) rather than FAIL."""

    def test_clearance_semantics_in_snapshot(self):
        grid = FoveatedGrid("spec")

        xy = np.array([
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
        ], dtype=np.float32)
        z = np.zeros(3, dtype=np.float32)
        probs = np.repeat(np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32), 3, axis=0)
        moving = np.zeros(3, dtype=bool)
        grid.update(xy, z, probs, moving, (0.0, 0.0))

        # Set specific clearances on cells in tier 0:
        # Cell at (1.0, 1.0) -> clearance = 3.0m (3.0 / 0.02 = 150)
        # Cell at (1.5, 1.5) -> clearance = 1.5m (1.5 / 0.02 = 75)
        # Cell at (0.5, 0.5) -> remains UNKNOWN (no overhead return -> clearance is None)
        c1 = grid.query_point(1.0, 1.0)["cell_coord"]
        grid.state[0].clear[c1[0], c1[1]] = int(3.0 / 0.02)

        c2 = grid.query_point(1.5, 1.5)["cell_coord"]
        grid.state[0].clear[c2[0], c2[1]] = int(1.5 / 0.02)

        snap = MapSnapshot(
            timestamp=1.0,
            frame_id="f1",
            ego_pose=np.eye(4),
            origins=tuple(grid.origins),
            tier_states=tuple(grid.snapshot()),
        )

        assert snap.query_point(0.5, 0.5)["clearance"] is None
        assert snap.query_point(1.0, 1.0)["clearance"] == pytest.approx(3.0, abs=0.05)
        assert snap.query_point(1.5, 1.5)["clearance"] == pytest.approx(1.5, abs=0.05)

        # Test Case 1: Open road passes 2m requirement
        assert snap.is_traversable(0.5, 0.5, clearance_req=2.0) is True

        # Test Case 2: High bridge (3m) passes 2m requirement
        assert snap.is_traversable(1.0, 1.0, clearance_req=2.0) is True

        # Test Case 3: Low bridge (1.5m) fails 2m requirement
        assert snap.is_traversable(1.5, 1.5, clearance_req=2.0) is False

        # Live grid parity
        assert grid.is_traversable(0.5, 0.5, clearance_req=2.0) is True
        assert grid.is_traversable(1.0, 1.0, clearance_req=2.0) is True
        assert grid.is_traversable(1.5, 1.5, clearance_req=2.0) is False

    def test_grid_and_torch_clearance_traversability_parity(self):
        grid_np = FoveatedGrid("spec")
        grid_th = TorchFoveatedGrid("spec", device="cpu")

        # Open road point: Ground returns only, no overhead returns -> clearance is NaN / None
        xy = np.array([
            [1.0, 0.0],
            [1.05, 0.05],
            [0.95, -0.05],
        ], dtype=np.float32)
        z = np.zeros(3, dtype=np.float32)
        probs = np.repeat(np.eye(NUM_CLASSES)[[ROAD]].astype(np.float32), 3, axis=0)
        moving = np.zeros(3, dtype=bool)

        grid_np.update(xy, z, probs, moving, (0.0, 0.0))
        grid_th.update(
            torch.from_numpy(xy),
            torch.from_numpy(z),
            torch.from_numpy(probs),
            torch.from_numpy(moving),
            (0.0, 0.0),
        )

        # Both must pass traversability for required clearance 2.5m
        assert grid_np.is_traversable(1.0, 0.0, clearance_req=2.5) is True
        assert grid_th.is_traversable(1.0, 0.0, clearance_req=2.5) is True


# ==============================================================================
# BUG #5: CHECKPOINT SAFETY / UNTRAINED MODEL PATH REGRESSION TESTS
# ==============================================================================

class TestBug5CheckpointSafety:
    """Verifies that missing or invalid checkpoints fail safely unless explicit allow_untrained=True."""

    def test_pipeline_missing_checkpoint_raises_configuration_error(self):
        with pytest.raises(ConfigurationError, match="requires a valid checkpoint path"):
            FoveaMapPipeline(grid="numpy", device="cpu", ckpt=None, allow_untrained=False)

    def test_pipeline_nonexistent_checkpoint_raises_checkpoint_not_found(self):
        with pytest.raises(CheckpointNotFoundError, match="Perception checkpoint file not found"):
            FoveaMapPipeline(
                grid="numpy",
                device="cpu",
                ckpt="/nonexistent/checkpoint_path.pt",
                allow_untrained=False,
            )

    def test_pipeline_explicit_allow_untrained_succeeds_with_warning(self):
        with pytest.warns(UserWarning, match="UNTRAINED / RANDOM WEIGHTS"):
            pipeline = FoveaMapPipeline(
                grid="numpy",
                device="cpu",
                ckpt=None,
                allow_untrained=True,
            )
            assert pipeline is not None

    def test_benchmark_safety_rejects_missing_checkpoint(self, tmp_path):
        from foveamap.frames import SIM_INFO
        with pytest.raises(ConfigurationError, match="requires a valid checkpoint path"):
            run_benchmark(frames=[], ckpt=None, info=SIM_INFO, out_dir=str(tmp_path), allow_untrained=False)

    def test_benchmark_explicit_allow_untrained_succeeds(self, tmp_path):
        from foveamap.frames import SIM_INFO
        from foveamap.data import create_source
        source = create_source("simulator", n_steps=1, seed=42)
        frames = list(source)
        with pytest.warns(UserWarning, match="UNTRAINED / RANDOM WEIGHTS"):
            summary, per_frame = run_benchmark(frames=frames, ckpt=None, info=SIM_INFO, out_dir=str(tmp_path), allow_untrained=True)
            assert summary["frames"] == 1
            assert "fps" in summary
            assert "latency_ms" in summary
            assert len(per_frame) == 1
