"""Core runtime orchestrator for FoveaMap.

Coordinates LiDAR frame ingestion, preprocessing, perception inference,
foveated 2.5D projection, temporal fusion, and canonical MapSnapshot publication.
Maintains full device-resident execution for Torch/CUDA without GPU->CPU->GPU round-trips.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from ..core.contracts import LiDARFrame, PerceptionResult, MapSnapshot
from ..core.config import FoveaMapConfig
from ..core.exceptions import ContractError, PerceptionError, MappingError
from .device import DeviceContext, resolve_device, sync_device
from .perception import (
    PerceptionBackend,
    RangeUNetBackend,
    DevicePerceptionResult,
    create_perception_backend,
)
from ..grid import FoveatedGrid
from ..grid_torch import TorchFoveatedGrid
from ..frames import transform
from ..data.preprocess import LiDARPreprocessor


class FoveaMapRuntime:
    """Production runtime orchestrating LiDAR ingest -> perception -> grid mapping -> snapshot."""

    def __init__(
        self,
        config: FoveaMapConfig | None = None,
        perception_backend: PerceptionBackend | None = None,
        preprocessor: LiDARPreprocessor | None = None,
    ) -> None:
        self.config = config if config is not None else FoveaMapConfig()
        self.device_ctx: DeviceContext = resolve_device(self.config.runtime, self.config.perception)

        if preprocessor is not None:
            self.preprocessor = preprocessor
        elif getattr(self.config, "preprocess", None) is not None and self.config.preprocess.enabled:
            self.preprocessor = LiDARPreprocessor(self.config.preprocess)
        else:
            self.preprocessor = None

        if perception_backend is not None:
            self.perception = perception_backend
        else:
            self.perception = create_perception_backend(
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
                dynamic_config=self.config.dynamic,
            )
        else:
            self.grid = FoveatedGrid(
                profile=self.config.grid,
                fuse=self.config.grid.fuse,
                terrain_config=self.config.terrain,
                dynamic_config=self.config.dynamic,
            )

        self.last_snapshot: MapSnapshot | None = None
        self._last_perception: PerceptionResult | None = None
        self._last_device_perception: DevicePerceptionResult | None = None
        self.last_timing: dict[str, float] = {}
        self.frame_count: int = 0
        self._lifecycle: str = "INITIALIZED"
        self._start_time: float | None = None
        self._frame_times: list[float] = []
        self._dropped_frames: int = 0

    @property
    def lifecycle(self) -> str:
        """Current product lifecycle state: INITIALIZED, CONFIGURED, ACTIVE, or STOPPED."""
        return self._lifecycle

    def configure(self) -> str:
        """Configure runtime and verify perception readiness."""
        if self._lifecycle == "STOPPED":
            self.reset()
        backend = getattr(self, "perception", None)
        if backend is not None and hasattr(backend, "is_ready") and not backend.is_ready:
            raise PerceptionError("Perception backend is not ready to configure")
        self._lifecycle = "CONFIGURED"
        return self._lifecycle

    def start(self) -> str:
        """Activate the runtime session for processing frames."""
        if self._lifecycle == "STOPPED":
            self.reset()
        self._lifecycle = "ACTIVE"
        if self._start_time is None:
            self._start_time = time.monotonic()
        return self._lifecycle

    def stop(self) -> str:
        """Deactivate processing without destroying existing grid map state."""
        self._lifecycle = "STOPPED"
        return self._lifecycle

    @property
    def last_perception(self) -> PerceptionResult | None:
        """Return canonical host PerceptionResult from the latest frame."""
        if self._last_perception is not None:
            return self._last_perception
        if self._last_device_perception is not None:
            self._last_perception = self._last_device_perception.to_host()
            return self._last_perception
        return None

    @property
    def last_device_perception(self) -> DevicePerceptionResult | None:
        """Return device-resident perception result (tensors) if Torch engine was active."""
        return self._last_device_perception

    def process(self, frame: LiDARFrame | dict[str, Any]) -> MapSnapshot:
        """Process a single LiDAR frame and publish a canonical MapSnapshot.

        Accepts either a canonical LiDARFrame or a legacy frame dictionary.
        """
        if self._lifecycle == "STOPPED":
            raise ContractError("Cannot process frame while runtime is in STOPPED state. Call start() or reset() first.")
        if self._lifecycle in ("INITIALIZED", "CONFIGURED"):
            self._lifecycle = "ACTIVE"
            if self._start_time is None:
                self._start_time = time.monotonic()

        if isinstance(frame, dict):
            try:
                canonical_frame = LiDARFrame.from_legacy_dict(frame)
            except Exception as exc:
                raise ContractError(f"Failed to convert legacy dict to LiDARFrame: {exc}") from exc
        elif isinstance(frame, LiDARFrame):
            canonical_frame = frame
        else:
            raise ContractError(f"Expected LiDARFrame or legacy dict, got {type(frame).__name__}")

        if self.preprocessor is not None:
            canonical_frame = self.preprocessor.process(canonical_frame)

        if canonical_frame.num_points == 0:
            # Deterministic, safe empty frame handling (Part 13)
            self._dropped_frames += 1
            origins = self.grid.window_origins(canonical_frame.pose[:2, 3])
            snapshot = MapSnapshot(
                timestamp=float(canonical_frame.timestamp),
                frame_id=str(canonical_frame.frame_id),
                ego_pose=canonical_frame.pose.copy(),
                origins=tuple(tuple(int(c) for c in o) for o in origins),
                tier_states=tuple(self.grid.snapshot()),
                dynamic_cells=(),
                dynamic_tracks=tuple(self.grid.temporal_snapshot()),
                temporal_metadata=dict(self.grid.temporal_stats()),
                metadata={
                    "empty_frame": True,
                    "frame_count": self.frame_count,
                    "grid_engine": self.config.runtime.grid_engine,
                    "device": str(self.device_ctx.device),
                    "dropped_frames": self._dropped_frames,
                    "points": 0,
                    "tier_configs": [
                        {"cell_size_m": float(t.cell), "half_extent_m": float(t.half), "n": int(t.n)}
                        for t in self.grid.tiers
                    ],
                },
            )
            self.last_snapshot = snapshot
            self.frame_count += 1
            return snapshot

        t_start = time.perf_counter()
        timing: dict[str, float] = {}

        on_dev = (self.config.runtime.grid_engine == "torch")
        dev_math = on_dev and (self.config.runtime.features_engine == "torch") and (self.device_ctx.device.type != "mps")

        profile = self.config.runtime.enable_profiling
        dev = self.device_ctx.device
        t0 = time.perf_counter()
        if on_dev:
            # 1. Device-resident perception (no CPU host conversion)
            dev_perception = self.perception.predict_device(
                canonical_frame, dev_math=dev_math, profiling=profile
            )
            if profile:
                sync_device(dev)
                timing["perception"] = time.perf_counter() - t0
            self._last_device_perception = dev_perception
            self._last_perception = None  # Lazily converted to host if requested

            # 2. Transform points to world coordinates (stays on device if dev_math)
            if profile:
                t0 = time.perf_counter()
            if dev_perception.pts_world is not None:
                pw = dev_perception.pts_world
            else:
                pw = transform(canonical_frame.pose, canonical_frame.pts.astype(np.float64))
            ego_xy = canonical_frame.pose[:2, 3]

            # 3. Binning into Torch foveated grid (consumed directly on device)
            origins = self.grid.window_origins(ego_xy)
            try:
                stats = self.grid.bin_points(
                    pw[:, :2],
                    pw[:, 2],
                    dev_perception.class_probabilities,
                    dev_perception.is_moving,
                    origins,
                )
            except Exception as exc:
                raise MappingError(f"Grid binning stage failed: {exc}") from exc
            if profile:
                sync_device(dev)
                timing["projection"] = time.perf_counter() - t0

            # 4. Temporal fusion & traversability derivation (on device)
            # Phase 6: static fusion -> dynamic lifecycle -> terrain ->
            # coherent world-state snapshot (see docs/DYNAMIC_WORLD_MODEL.md).
            if profile:
                t0 = time.perf_counter()
            try:
                dyn = self.grid.fuse_stats(
                    stats, origins, sensor_origin=ego_xy,
                    timestamp=float(canonical_frame.timestamp),
                )
            except Exception as exc:
                raise MappingError(f"Grid fusion stage failed: {exc}") from exc
            if profile:
                sync_device(dev)
                timing["fusion"] = time.perf_counter() - t0
        else:
            # CPU NumPy execution path
            host_perception = self.perception.predict(canonical_frame)
            if profile:
                timing["perception"] = time.perf_counter() - t0
            self._last_perception = host_perception
            self._last_device_perception = None

            if profile:
                t0 = time.perf_counter()
            pw = transform(canonical_frame.pose, canonical_frame.pts.astype(np.float64))
            ego_xy = canonical_frame.pose[:2, 3]

            origins = self.grid.window_origins(ego_xy)
            try:
                stats = self.grid.bin_points(
                    pw[:, :2],
                    pw[:, 2],
                    host_perception.class_probabilities,
                    host_perception.is_moving,
                    origins,
                )
            except Exception as exc:
                raise MappingError(f"Grid binning stage failed: {exc}") from exc
            if profile:
                timing["projection"] = time.perf_counter() - t0

            if profile:
                t0 = time.perf_counter()
            try:
                dyn = self.grid.fuse_stats(
                    stats, origins, sensor_origin=ego_xy,
                    timestamp=float(canonical_frame.timestamp),
                )
            except Exception as exc:
                raise MappingError(f"Grid fusion stage failed: {exc}") from exc
            if profile:
                timing["fusion"] = time.perf_counter() - t0

        if profile:
            if on_dev:
                sync_device(dev)
            timing["total"] = time.perf_counter() - t_start
        self.last_timing = timing

        # 5. Publication of canonical MapSnapshot (intentional host copy of grid state)
        # Phase 6: the snapshot is a coherent world state — static layers plus
        # the bounded temporal dynamic lifecycle, published atomically after
        # all per-frame updates complete.
        tier_states = tuple(self.grid.snapshot())
        temporal_tracks = tuple(self.grid.temporal_snapshot())
        temporal_stats = dict(self.grid.temporal_stats())
        snapshot = MapSnapshot(
            timestamp=float(canonical_frame.timestamp),
            frame_id=str(canonical_frame.frame_id),
            ego_pose=canonical_frame.pose.copy(),
            origins=tuple(tuple(int(c) for c in o) for o in origins),
            tier_states=tier_states,
            dynamic_cells=tuple(dyn) if dyn is not None else (),
            dynamic_tracks=temporal_tracks,
            temporal_metadata=temporal_stats,
            metadata={
                "frame_count": self.frame_count,
                "grid_engine": self.config.runtime.grid_engine,
                "device": str(self.device_ctx.device),
                "device_resident": on_dev,
                # Authoritative tier geometry so MapSnapshot.query_point cannot
                # misinterpret this snapshot under another profile.
                "tier_configs": [
                    {"cell_size_m": float(t.cell), "half_extent_m": float(t.half), "n": int(t.n)}
                    for t in self.grid.tiers
                ],
                "profile_name": getattr(self.config.grid, "profile_name", ""),
                "stale_age_threshold": int(self.config.terrain.stale_age_threshold),
                "max_stale_age": int(self.config.terrain.max_stale_age),
                "traversable_cost_max": int(self.config.terrain.traversable_cost_max),
                "dynamic_config": {
                    "activation_frames": int(self.config.dynamic.activation_frames),
                    "missing_tolerance_frames": int(self.config.dynamic.missing_tolerance_frames),
                    "stale_frames": int(self.config.dynamic.stale_frames),
                    "confidence_threshold": float(self.config.dynamic.confidence_threshold),
                    "correspondence_distance_m": float(self.config.dynamic.correspondence_distance_m),
                    "max_tracks": int(self.config.dynamic.max_tracks),
                },
                "timing": timing if self.config.runtime.enable_profiling else {},
                "points": canonical_frame.num_points,
            },
        )
        self.last_snapshot = snapshot
        self.frame_count += 1
        self._frame_times.append(time.monotonic())
        if len(self._frame_times) > 60:
            del self._frame_times[: len(self._frame_times) - 60]
        return snapshot

    def step(self, frame: LiDARFrame | dict[str, Any]) -> MapSnapshot:
        """Alias for process(frame)."""
        return self.process(frame)

    def process_source(self, source: Any, max_frames: int | None = None) -> Iterator[MapSnapshot]:
        """Sequentially process frames from a LiDARSource and yield published MapSnapshots."""
        for i, frame in enumerate(source):
            if max_frames is not None and i >= max_frames:
                break
            yield self.process(frame)

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
                dynamic_config=self.config.dynamic,
            )
        else:
            self.grid = FoveatedGrid(
                profile=self.config.grid,
                fuse=self.config.grid.fuse,
                terrain_config=self.config.terrain,
                dynamic_config=self.config.dynamic,
            )
        self.perception.reset()
        self.last_snapshot = None
        self._last_perception = None
        self._last_device_perception = None
        self.last_timing = {}
        self.frame_count = 0
        self._lifecycle = "CONFIGURED"
        self._start_time = None
        self._frame_times.clear()
        self._dropped_frames = 0

    def get_metrics(self) -> dict[str, Any]:
        """Collect authoritative runtime metrics from actual execution state."""
        now = time.monotonic()
        fps = 0.0
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            fps = round((len(self._frame_times) - 1) / span, 2) if span > 0 else 0.0
        uptime = round((now - self._start_time), 2) if self._start_time else 0.0

        timing_ms = {k: round(float(v) * 1000.0, 3) for k, v in self.last_timing.items()}

        mem: dict[str, Any] = {}
        if hasattr(self.grid, "memory_report"):
            try:
                mem = self.grid.memory_report()
            except Exception:
                mem = {}

        cuda_mem: dict[str, Any] = {}
        if self.device_ctx.use_cuda and torch.cuda.is_available():
            try:
                cuda_mem = {
                    "allocated_mb": round(torch.cuda.memory_allocated(self.device_ctx.device) / (1024 * 1024), 2),
                    "reserved_mb": round(torch.cuda.memory_reserved(self.device_ctx.device) / (1024 * 1024), 2),
                    "max_allocated_mb": round(torch.cuda.max_memory_allocated(self.device_ctx.device) / (1024 * 1024), 2),
                }
            except Exception:
                cuda_mem = {}

        temporal_stats: dict[str, Any] = {}
        if hasattr(self.grid, "temporal_stats"):
            try:
                temporal_stats = dict(self.grid.temporal_stats())
            except Exception:
                temporal_stats = {}

        last_points = self.last_snapshot.metadata.get("points", 0) if self.last_snapshot else 0
        backend_name = str(getattr(self.perception, "name", None) or type(self.perception).__name__)

        return {
            "lifecycle": self._lifecycle,
            "uptime_s": uptime,
            "frames_processed": self.frame_count,
            "frames_dropped": self._dropped_frames,
            "fps": fps,
            "last_points": last_points,
            "backend": backend_name,
            "device": str(self.device_ctx.device),
            "grid_engine": self.config.runtime.grid_engine,
            "features_engine": self.config.runtime.features_engine,
            "timing_ms": timing_ms,
            "memory": {
                "allocated_bytes": mem.get("allocated_bytes", 0),
                "uniform_baseline_bytes": mem.get("uniform_baseline_bytes", 0),
                "reduction_ratio": mem.get("reduction_ratio", 0.0),
                "under_8mb_target": mem.get("under_8mb_target", True),
                "cuda": cuda_mem,
            },
            "temporal": temporal_stats,
        }

