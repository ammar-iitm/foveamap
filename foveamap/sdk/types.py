"""Immutable public result types (Phase 9).

Every type is a frozen dataclass holding plain values only — no NumPy arrays,
no tensors, no live grid references. ``to_dict()`` output is JSON-compatible
and carries the SDK ``API_VERSION`` so future refactors cannot silently break
consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
import math

import numpy as np

from . import API_VERSION
from .errors import SDKQueryError


def _clean(value: Any) -> Any:
    """Make a value JSON-safe (tuples -> lists; NaN/Inf -> None)."""
    if isinstance(value, tuple):
        return [_clean(v) for v in value]
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    try:
        import numpy as _np  # noqa: F401
        if isinstance(value, _np.generic):
            return _clean(value.item())
        if isinstance(value, _np.ndarray):
            return _clean(value.tolist())
    except ImportError:
        pass
    return value


@dataclass(frozen=True)
class QueryResult:
    """Point query answered by the authoritative core implementation."""

    x: float
    y: float
    tier: int
    cell_size_m: float | None
    state: str
    dominant_class: int | None
    confidence: float | None
    ground_m: float | None
    roughness_m: float | None
    slope_rad: float | None
    clearance_m: float | None
    cost: int | None
    dynamic: bool
    dynamic_state: str | None
    traversable: bool
    age_frames: int | None

    @classmethod
    def from_snapshot_query(
        cls, x: float, y: float, raw: dict[str, Any], *, traversable: bool | None = None,
    ) -> QueryResult:
        """Translate a core ``query_point`` dict without reimplementing policy.

        Core snapshot queries carry no traversability flag, so callers pass
        the result of the authoritative ``snapshot.is_traversable(x, y)``;
        grid-level dicts already contain ``is_traversable`` and need no
        override. ``traversable`` must never silently default.
        """
        try:
            res = raw.get("resolution", raw.get("cell_size_m"))
            if traversable is None:
                if "is_traversable" not in raw:
                    raise SDKQueryError("Query dict has no traversability flag and none was supplied")
                traversable = bool(raw["is_traversable"])
            return cls(
                x=float(x),
                y=float(y),
                tier=int(raw.get("tier", -1)),
                cell_size_m=None if res is None else float(res),
                state=str(raw.get("state", "UNKNOWN")),
                dominant_class=raw.get("dominant_class"),
                confidence=raw.get("confidence"),
                ground_m=raw.get("ground"),
                roughness_m=raw.get("roughness"),
                slope_rad=raw.get("slope_rad"),
                clearance_m=raw.get("clearance", raw.get("clear")),
                cost=raw.get("cost"),
                dynamic=bool(raw.get("dynamic", False)),
                dynamic_state=raw.get("dynamic_state"),
                traversable=bool(traversable),
                age_frames=raw.get("age"),
            )
        except (TypeError, ValueError) as exc:
            raise SDKQueryError(f"Cannot translate core query result: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {"api_version": API_VERSION, **{k: _clean(v) for k, v in self.__dict__.items()}}


@dataclass(frozen=True)
class SnapshotView:
    """Read-only handle to one detached world state.

    Holds the core snapshot by reference (it is already detached from the
    live grid) but exposes no mutators and no internal arrays. Queries
    delegate to the snapshot's authoritative implementation.
    """

    _snapshot: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._snapshot is None:
            from .errors import SDKQueryError as _Q
            raise _Q("SnapshotView requires a snapshot")

    @property
    def timestamp(self) -> float:
        return float(self._snapshot.timestamp)

    @property
    def frame_id(self) -> str:
        return str(self._snapshot.frame_id)

    @property
    def num_tiers(self) -> int:
        return len(self._snapshot.tier_states)

    @property
    def origins(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(a), int(b)) for a, b in self._snapshot.origins)

    def query_point(self, x: float, y: float) -> QueryResult:
        x, y = float(x), float(y)
        raw = self._snapshot.query_point(x, y)
        return QueryResult.from_snapshot_query(
            x, y, raw, traversable=bool(self._snapshot.is_traversable(x, y)))

    def query_points(self, xs: Sequence[float], ys: Sequence[float], *, limit: int = 4096) -> list[QueryResult]:
        xs = list(xs)
        ys = list(ys)
        if len(xs) != len(ys):
            raise SDKQueryError(f"xs/ys length mismatch ({len(xs)} != {len(ys)})")
        if len(xs) > limit:
            raise SDKQueryError(f"Batch limited to {limit} points, got {len(xs)}")
        return [self.query_point(float(x), float(y)) for x, y in zip(xs, ys)]

    def query_ray(
        self,
        x: float,
        y: float,
        theta_rad: float,
        *,
        step_m: float = 0.5,
        max_steps: int = 512,
    ) -> list[QueryResult]:
        """Sample the authoritative snapshot query along a ray (no new mapping logic).

        Each sample delegates to ``query_point``; the SDK adds only the ray
        walk itself. Bounded by ``max_steps`` (1..4096) with positive step.
        """
        if not (np.isfinite(float(x)) and np.isfinite(float(y)) and np.isfinite(float(theta_rad))):
            raise SDKQueryError("Ray origin and heading must be finite")
        if not (float(step_m) > 0):
            raise SDKQueryError(f"step_m must be positive, got {step_m}")
        if not (1 <= int(max_steps) <= 4096):
            raise SDKQueryError(f"max_steps must be in [1, 4096], got {max_steps}")
        dx, dy = math.cos(float(theta_rad)), math.sin(float(theta_rad))
        return [
            self.query_point(float(x) + k * float(step_m) * dx, float(y) + k * float(step_m) * dy)
            for k in range(int(max_steps))
        ]

    def export_numpy(self) -> dict[str, Any]:
        """Detached NumPy copy of tier arrays for offline/export consumers.

        Returns ``{"api_version", "tiers": [...], "origins": [...]}`` where
        each tier dict holds fresh array copies (mutating them cannot affect
        the live map or this view). Read-only semantic content; sizes follow
        the configured foveated profile.
        """
        import numpy as _np

        tiers: list[dict[str, Any]] = []
        for tier in self._snapshot.tier_states:
            entry: dict[str, Any] = {"cell_m": float(getattr(tier, "cell", getattr(tier, "r", 0.0)) or 0.0)}
            for name in ("count", "cls", "conf", "flags", "clear", "cost", "age"):
                entry[name] = np.array(getattr(tier, name), copy=True)
            for name in ("z_min", "z_max", "ground", "rough"):
                entry[name] = np.array(getattr(tier, name), dtype=np.float32, copy=True)
            entry["dynamic"] = np.array(tier.dynamic, dtype=bool, copy=True)
            tiers.append(entry)
        return {"api_version": API_VERSION, "tiers": tiers,
                "origins": [list(o) for o in self.origins]}

    def is_traversable(self, x: float, y: float, clearance_req: float = 0.0) -> bool:
        return bool(self._snapshot.is_traversable(float(x), float(y), float(clearance_req)))

    def to_dict(self) -> dict[str, Any]:
        meta = self._snapshot.metadata if isinstance(self._snapshot.metadata, dict) else {}
        return {
            "api_version": API_VERSION,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "num_tiers": self.num_tiers,
            "origins": [list(o) for o in self.origins],
            "temporal": _clean(dict(getattr(self._snapshot, "temporal_metadata", {}) or {})),
            "metadata": _clean({k: v for k, v in meta.items() if k != "timing"}),
        }


@dataclass(frozen=True)
class RuntimeStatus:
    """Lifecycle + transport state (plain values)."""

    lifecycle: str
    healthy: bool
    health: str
    health_reasons: tuple[str, ...]
    device: str
    grid_engine: str
    backend: str
    frames_received: int
    frames_processed: int
    frames_dropped: int
    last_frame_id: str
    last_timestamp: float | None
    last_error: str | None
    owns_runtime: bool

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["health_reasons"] = list(d["health_reasons"])
        return {"api_version": API_VERSION, **{k: _clean(v) for k, v in d.items()}}


@dataclass(frozen=True)
class RuntimeMetrics:
    """Measured metrics only (no invented numbers)."""

    frames_received: int
    frames_processed: int
    frames_dropped: int
    fps: float
    stage_latency_ms: dict[str, float]
    map_memory: dict[str, Any]
    temporal_tracks: int
    device: str
    cuda_note: str

    def to_dict(self) -> dict[str, Any]:
        return {"api_version": API_VERSION, **{k: _clean(v) for k, v in self.__dict__.items()}}


@dataclass(frozen=True)
class HealthReport:
    """Lightweight health answer."""

    status: str  # "healthy" | "degraded" | "unavailable"
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"api_version": API_VERSION, "status": self.status, "reasons": list(self.reasons)}
