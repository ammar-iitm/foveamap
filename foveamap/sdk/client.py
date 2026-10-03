"""Public session client (Phase 9).

``FoveaMap`` owns the lifecycle around an (optionally injected)
:class:`FoveaMapRuntime` and exposes only frozen public types. All mapping
semantics come from the core; this module adds lifecycle, ownership,
health, metrics plumbing, and serialization — no mapping logic.
"""
from __future__ import annotations

import threading
import time
from dataclasses import fields as _dc_fields
from typing import Any, Sequence

from foveamap.core.config import FoveaMapConfig
from foveamap.core.contracts import LiDARFrame, MapSnapshot
from foveamap.core.exceptions import ConfigurationError, FoveaMapError
from foveamap.runtime.runtime import FoveaMapRuntime

from . import API_VERSION
from .errors import SDKConfigError, SDKError, SDKLifecycleError, SDKQueryError
from .types import HealthReport, QueryResult, RuntimeMetrics, RuntimeStatus, SnapshotView

CREATED = "CREATED"
CONFIGURED = "CONFIGURED"
ACTIVE = "ACTIVE"
INACTIVE = "INACTIVE"
SHUTDOWN = "SHUTDOWN"

_DEGRADED_AFTER_FAILURES = 3


class FoveaMap:
    """Stable public entry point for one mapping session."""

    api_version = API_VERSION

    def __init__(
        self,
        config: FoveaMapConfig | dict[str, Any] | None = None,
        *,
        runtime: FoveaMapRuntime | None = None,
        perception_backend: Any | None = None,
    ) -> None:
        if isinstance(config, dict):
            config = self._config_from_dict(config)
        self._config = config if config is not None else FoveaMapConfig()
        if not isinstance(self._config, FoveaMapConfig):
            raise SDKConfigError(f"config must be FoveaMapConfig or dict, got {type(config).__name__}")
        self._owns_runtime = runtime is None
        self._runtime = runtime if runtime is not None else FoveaMapRuntime(
            config=self._config, perception_backend=perception_backend)
        self._lock = threading.RLock()
        self._state = CREATED
        self._snapshot: MapSnapshot | None = None
        self._received = 0
        self._processed = 0
        self._dropped = 0
        self._consec_failures = 0
        self._last_error: str | None = None
        self._last_frame_id = ""
        self._last_timestamp: float | None = None
        self._fps_times: list[float] = []
        self._stage_totals: dict[str, float] = {}
        self._stage_counts: dict[str, int] = {}

    # ------------------------------------------------------------ config API
    @staticmethod
    def _config_from_dict(raw: dict[str, Any]) -> FoveaMapConfig:
        """Strict dict -> FoveaMapConfig (unknown keys rejected)."""
        if not isinstance(raw, dict):
            raise SDKConfigError(f"config dict required, got {type(raw).__name__}")
        sections = ("sensor", "grid", "perception", "terrain", "runtime", "preprocess", "dynamic")
        unknown = sorted(set(raw) - set(sections))
        if unknown:
            raise SDKConfigError(f"Unknown config sections: {unknown}")
        base = FoveaMapConfig()
        kwargs: dict[str, Any] = {}
        for name in sections:
            current = getattr(base, name)
            value = raw.get(name, None)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise SDKConfigError(f"Section {name!r} must be a dict")
            allowed = {f.name for f in _dc_fields(current)}
            bad = sorted(set(value) - allowed)
            if bad:
                raise SDKConfigError(f"Unknown {name} keys: {bad}")
            try:
                kwargs[name] = type(current)(
                    **{**{f.name: getattr(current, f.name) for f in _dc_fields(current)}, **value})
            except (TypeError, ValueError, ConfigurationError) as exc:
                raise SDKConfigError(f"Invalid {name} values: {exc}") from exc
        try:
            return FoveaMapConfig(**{**{s: getattr(base, s) for s in sections}, **kwargs})
        except (TypeError, ValueError) as exc:
            raise SDKConfigError(f"Invalid configuration values: {exc}") from exc

    @property
    def config(self) -> FoveaMapConfig:
        return self._config

    @property
    def lifecycle(self) -> str:
        return self._state

    # --------------------------------------------------------------- lifecycle
    def _require(self, *allowed: str, op: str) -> None:
        if self._state not in allowed:
            raise SDKLifecycleError(f"{op} illegal from {self._state}; allowed: {list(allowed)}")

    def configure(self) -> RuntimeStatus:
        with self._lock:
            self._require(CREATED, INACTIVE, op="configure")
            backend = getattr(self._runtime, "perception", None)
            if backend is None or not getattr(backend, "is_ready", False):
                raise SDKLifecycleError("Perception backend not ready; cannot configure")
            self._state = CONFIGURED
            return self.status()

    def start(self) -> RuntimeStatus:
        """Activate a configured session (processing allowed after this)."""
        with self._lock:
            self._require(CONFIGURED, op="start")
            self._state = ACTIVE
            return self.status()

    def stop(self) -> RuntimeStatus:
        """Deactivate frame acceptance without destroying state."""
        with self._lock:
            if self._state == ACTIVE:
                self._state = INACTIVE
            return self.status()

    def close(self) -> None:
        """Shut down; releases owned runtime resources (never foreign ones)."""
        with self._lock:
            if self._owns_runtime and self._runtime is not None:
                try:
                    self._runtime.reset()
                except FoveaMapError:
                    pass
            self._runtime = None  # type: ignore[assignment]
            self._snapshot = None
            self._state = SHUTDOWN

    # -------------------------------------------------------------- processing
    def process(self, frame: LiDARFrame | dict[str, Any]) -> SnapshotView:
        """Ingest one frame and return a detached snapshot view (ACTIVE only)."""
        with self._lock:
            self._require(ACTIVE, op="process")
            self._received += 1
            t0 = time.perf_counter()
            try:
                snapshot = self._runtime.process(frame)
            except FoveaMapError as exc:
                self._consec_failures += 1
                self._dropped += 1
                self._last_error = f"{type(exc).__name__}: {exc}"[:300]
                raise
            except Exception as exc:
                self._consec_failures += 1
                self._dropped += 1
                self._last_error = f"{type(exc).__name__}: {exc}"[:300]
                raise SDKError(f"Frame processing failed: {exc}") from exc
            dt = time.perf_counter() - t0
            self._processed += 1
            self._consec_failures = 0
            self._last_error = None
            self._snapshot = snapshot
            self._last_frame_id = str(snapshot.frame_id)
            self._last_timestamp = float(snapshot.timestamp)
            self._fps_times.append(time.monotonic())
            if len(self._fps_times) > 60:
                del self._fps_times[: len(self._fps_times) - 60]
            self._stage_totals["sdk_process_s"] = self._stage_totals.get("sdk_process_s", 0.0) + dt
            self._stage_counts["sdk_process_s"] = self._stage_counts.get("sdk_process_s", 0) + 1
            return SnapshotView(snapshot)

    def reset(self) -> RuntimeStatus:
        """Clear map/temporal/counter state; config and backend preserved."""
        with self._lock:
            self._require(CONFIGURED, ACTIVE, INACTIVE, op="reset")
            self._runtime.reset()
            self._snapshot = None
            self._received = 0
            self._processed = 0
            self._dropped = 0
            self._consec_failures = 0
            self._last_error = None
            self._last_frame_id = ""
            self._last_timestamp = None
            self._fps_times.clear()
            self._stage_totals.clear()
            self._stage_counts.clear()
            return self.status()

    # ----------------------------------------------------------------- query
    def _view(self) -> SnapshotView:
        snap = self._snapshot
        if snap is None:
            raise SDKQueryError("No snapshot yet; process a frame first")
        if isinstance(snap, SnapshotView):
            return snap
        return SnapshotView(snap)

    def query_point(self, x: float, y: float) -> QueryResult:
        with self._lock:
            return self._view().query_point(float(x), float(y))

    def query_points(self, xs: Sequence[float], ys: Sequence[float], *, limit: int = 4096) -> list[QueryResult]:
        with self._lock:
            return self._view().query_points(xs, ys, limit=limit)

    def query_ray(self, x: float, y: float, theta_rad: float, *,
                  step_m: float = 0.5, max_steps: int = 512) -> list[QueryResult]:
        """Sample the authoritative snapshot along a ray (delegates per sample)."""
        with self._lock:
            return self._view().query_ray(float(x), float(y), float(theta_rad),
                                          step_m=step_m, max_steps=max_steps)

    def export_numpy(self) -> dict[str, Any]:
        """Detached NumPy copy of tier arrays (mutating it affects nothing)."""
        with self._lock:
            return self._view().export_numpy()

    def is_traversable(self, x: float, y: float, clearance_req: float = 0.0) -> bool:
        with self._lock:
            return self._view().is_traversable(float(x), float(y), float(clearance_req))

    def snapshot(self) -> SnapshotView:
        """Current detached snapshot view (raises if none yet)."""
        with self._lock:
            return self._view()

    # ----------------------------------------------------------- health/metrics
    def _backend_name(self) -> str:
        runtime = self._runtime
        backend = getattr(runtime, "perception", None) if runtime is not None else None
        if backend is None:
            return "none"
        return str(getattr(backend, "name", None) or type(backend).__name__)

    def health(self) -> HealthReport:
        with self._lock:
            if self._state == SHUTDOWN or self._runtime is None:
                return HealthReport(status="unavailable", reasons=("shutdown",))
            reasons: list[str] = []
            backend = self._backend_name().lower()
            if "classical" in backend or "fallback" in backend or "heuristic" in backend:
                reasons.append("fallback_perception_backend")
            if self._consec_failures >= _DEGRADED_AFTER_FAILURES:
                reasons.append(f"repeated_frame_failures:{self._consec_failures}")
            if self._last_error:
                reasons.append(f"last_error:{self._last_error}")
            if self._state != ACTIVE:
                reasons.append(f"lifecycle_{self._state.lower()}")
            if reasons:
                return HealthReport(status="degraded", reasons=tuple(reasons))
            return HealthReport(status="healthy", reasons=())

    def status(self) -> RuntimeStatus:
        with self._lock:
            health = self.health()
            device = str(getattr(getattr(self._runtime, "device_ctx", None), "device", "unknown"))
            try:
                grid_engine = str(self._runtime.config.runtime.grid_engine)
            except AttributeError:
                grid_engine = "unknown"
            return RuntimeStatus(
                lifecycle=self._state,
                healthy=health.status == "healthy",
                health=health.status,
                health_reasons=health.reasons,
                device=device,
                grid_engine=grid_engine,
                backend=self._backend_name(),
                frames_received=self._received,
                frames_processed=self._processed,
                frames_dropped=self._dropped,
                last_frame_id=self._last_frame_id,
                last_timestamp=self._last_timestamp,
                last_error=self._last_error,
                owns_runtime=self._owns_runtime,
            )

    def metrics(self) -> RuntimeMetrics:
        with self._lock:
            fps = 0.0
            if len(self._fps_times) >= 2:
                span = self._fps_times[-1] - self._fps_times[0]
                fps = (len(self._fps_times) - 1) / span if span > 0 else 0.0
            stage_ms = {k: round(v / max(self._stage_counts.get(k, 1), 1) * 1000.0, 3)
                        for k, v in self._stage_totals.items()}
            timing = dict(getattr(self._runtime, "last_timing", {}) or {})
            for k, v in timing.items():
                try:
                    stage_ms[f"runtime_{k}_ms"] = round(float(v) * 1000.0, 3)
                except (TypeError, ValueError):
                    continue
            mem: dict[str, Any] = {}
            tracks = 0
            try:
                grid = getattr(self._runtime, "grid", None)
                if grid is not None and hasattr(grid, "memory_report"):
                    rep = grid.memory_report()
                    mem = {k: rep[k] for k in ("allocated_bytes", "auxiliary_bytes",
                                              "total_allocated_bytes", "under_8mb_target") if k in rep}
                temporal = getattr(grid, "temporal_stats", None)
                if callable(temporal):
                    tracks = int(temporal().get("n_tracks", 0))
            except FoveaMapError:
                pass
            device = str(getattr(getattr(self._runtime, "device_ctx", None), "device", "unknown"))
            return RuntimeMetrics(
                frames_received=self._received,
                frames_processed=self._processed,
                frames_dropped=self._dropped,
                fps=round(fps, 2),
                stage_latency_ms=stage_ms,
                map_memory=mem,
                temporal_tracks=tracks,
                device=device,
                cuda_note=("cuda_device_selected" if "cuda" in device else "cuda_unavailable_cpu_execution"),
            )
