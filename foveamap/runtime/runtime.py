"""Core runtime orchestrator for FoveaMap.

Coordinates LiDAR frame ingestion, preprocessing, perception inference,
foveated 2.5D projection, temporal fusion, and canonical MapSnapshot publication.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.contracts import LiDARFrame, PerceptionResult, MapSnapshot
from ..core.config import FoveaMapConfig
from ..core.exceptions import ContractError, PerceptionError, MappingError
from .device import DeviceContext, resolve_device, sync_device
from .perception import PerceptionBackend, RangeUNetBackend
from ..grid import FoveatedGrid
from ..grid_torch import TorchFoveatedGrid
from ..frames import transform


class FoveaMapRuntime:
    """Production runtime orchestrating LiDAR ingest -> perception -> grid mapping -> snapshot."""

    def __init__(
        self,
        config: FoveaMapConfig | None = None,
        perception_backend: PerceptionBackend | None = None,
    ) -> None:
        self.config = config if config is not None else FoveaMapConfig()
        self.device_ctx: DeviceContext = resolve_device(self.config.runtime, self.config.perception)

        if perception_backend is not None:
            self.perception = perception_backend
        else:
            self.perception = RangeUNetBackend(
                config=self.config.perception,
                sensor_config=self.config.sensor,
                device=self.device_ctx.device,
                features_engine=self.config.runtime.features_engine,
            )

        if self.config.runtime.grid_engine == "torch":
            self.grid = TorchFoveatedGrid(
                profile=self.config.grid,
                fuse=self.config.grid.fuse,
                device=self.device_ctx.device,
                terrain_config=self.config.terrain,
            )
        else:
            self.grid = FoveatedGrid(
                profile=self.config.grid,
                fuse=self.config.grid.fuse,
                terrain_config=self.config.terrain,
            )

        self.last_snapshot: MapSnapshot | None = None
        self.last_perception: PerceptionResult | None = None
        self.last_timing: dict[str, float] = {}
        self.frame_count: int = 0

    def process(self, frame: LiDARFrame | dict[str, Any]) -> MapSnapshot:
        """Process a single LiDAR frame and publish a canonical MapSnapshot.

        Accepts either a canonical LiDARFrame or a legacy frame dictionary.
        """
        if isinstance(frame, dict):
            try:
                canonical_frame = LiDARFrame.from_legacy_dict(frame)
            except Exception as exc:
                raise ContractError(f"Failed to convert legacy dict to LiDARFrame: {exc}") from exc
        elif isinstance(frame, LiDARFrame):
            canonical_frame = frame
        else:
            raise ContractError(f"Expected LiDARFrame or legacy dict, got {type(frame).__name__}")

        t_start = time.perf_counter()
        timing: dict[str, float] = {}

        # 1. Perception inference
        t0 = time.perf_counter()
        try:
            perception_result = self.perception.predict(canonical_frame)
        except Exception as exc:
            raise PerceptionError(f"Perception stage failed: {exc}") from exc
        sync_device(self.device_ctx.device)
        timing["perception"] = time.perf_counter() - t0
        self.last_perception = perception_result

        # 2. Transform points to world coordinates
        t0 = time.perf_counter()
        pts_world = transform(canonical_frame.pose, canonical_frame.pts.astype(np.float64))
        ego_xy = canonical_frame.pose[:2, 3]

        # 3. Binning into foveated grid
        origins = self.grid.window_origins(ego_xy)
        try:
            stats = self.grid.bin_points(
                pts_world[:, :2],
                pts_world[:, 2],
                perception_result.class_probabilities,
                perception_result.is_moving,
                origins,
            )
        except Exception as exc:
            raise MappingError(f"Grid binning stage failed: {exc}") from exc
        sync_device(self.device_ctx.device)
        timing["projection"] = time.perf_counter() - t0

        # 4. Temporal fusion & traversability derivation
        t0 = time.perf_counter()
        try:
            dyn = self.grid.fuse_stats(stats, origins)
        except Exception as exc:
            raise MappingError(f"Grid fusion stage failed: {exc}") from exc
        sync_device(self.device_ctx.device)
        timing["fusion"] = time.perf_counter() - t0
        timing["total"] = time.perf_counter() - t_start
        self.last_timing = timing

        # 5. Publication of MapSnapshot contract
        tier_states = tuple(self.grid.snapshot())
        snapshot = MapSnapshot(
            timestamp=float(canonical_frame.timestamp),
            frame_id=str(canonical_frame.frame_id),
            ego_pose=canonical_frame.pose.copy(),
            origins=tuple(tuple(int(c) for c in o) for o in origins),
            tier_states=tier_states,
            dynamic_cells=tuple(dyn) if dyn is not None else (),
            metadata={
                "frame_count": self.frame_count,
                "grid_engine": self.config.runtime.grid_engine,
                "device": str(self.device_ctx.device),
                "timing": timing if self.config.runtime.enable_profiling else {},
            },
        )
        self.last_snapshot = snapshot
        self.frame_count += 1
        return snapshot

    def step(self, frame: LiDARFrame | dict[str, Any]) -> MapSnapshot:
        """Alias for process(frame)."""
        return self.process(frame)

    def snapshot(self) -> MapSnapshot | None:
        """Return the most recently generated MapSnapshot."""
        return self.last_snapshot

    def reset(self) -> None:
        """Reset internal grid state, perception history, and frame counters."""
        if self.config.runtime.grid_engine == "torch":
            self.grid = TorchFoveatedGrid(
                profile=self.config.grid,
                fuse=self.config.grid.fuse,
                device=self.device_ctx.device,
                terrain_config=self.config.terrain,
            )
        else:
            self.grid = FoveatedGrid(
                profile=self.config.grid,
                fuse=self.config.grid.fuse,
                terrain_config=self.config.terrain,
            )
        self.perception.reset()
        self.last_snapshot = None
        self.last_perception = None
        self.last_timing = {}
        self.frame_count = 0
