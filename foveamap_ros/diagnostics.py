"""Operational metrics aggregation (stage-separated, cheap by design).

Stages: INPUT / PERCEPTION / MAPPING / DYNAMIC / TERRAIN / PUBLICATION /
SYSTEM. The aggregator only accumulates scalars and small plain-value dicts;
per-frame cost is O(1). GPU fields report honest device strings (e.g. "cpu"
or "cuda (unverified)") — never fabricated measurements.
"""
from __future__ import annotations

import time
from typing import Any


class MetricsAggregator:
    """Bounded counters + latency statistics for /foveamap/metrics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.errors: dict[str, int] = {}
        self._lat: dict[str, list[float]] = {}
        self._last_frame_end: float | None = None
        self._fps_times: list[float] = []
        self.last_report: dict[str, Any] = {}

    def note_received(self) -> None:
        self.frames_received += 1

    def note_dropped(self, reason: str) -> None:
        self.frames_dropped += 1
        self.errors[f"dropped_{reason}"] = self.errors.get(f"dropped_{reason}", 0) + 1

    def note_error(self, stage: str, kind: str) -> None:
        key = f"{stage}_{kind}"
        self.errors[key] = self.errors.get(key, 0) + 1

    def observe_latency(self, stage: str, seconds: float) -> None:
        buf = self._lat.setdefault(stage, [])
        buf.append(float(seconds))
        if len(buf) > 120:  # bounded rolling window
            del buf[: len(buf) - 120]

    def _percentile(self, values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        k = (len(ordered) - 1) * (pct / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
        frac = k - lo
        return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

    def note_frame_done(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.frames_processed += 1
        self._fps_times.append(now)
        if len(self._fps_times) > 60:
            del self._fps_times[: len(self._fps_times) - 60]

    @property
    def fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        span = self._fps_times[-1] - self._fps_times[0]
        if span <= 0:
            return 0.0
        return (len(self._fps_times) - 1) / span

    def report(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Plain-data metrics payload (JSON-serializable)."""
        stages = {stage: {"mean_ms": round(sum(v) / len(v) * 1000.0, 3),
                          "p95_ms": round(self._percentile(v, 95) * 1000.0, 3),
                          "count": len(v)}
                  for stage, v in self._lat.items()}
        payload = {
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "fps": round(self.fps, 2),
            "stage_latency": stages,
            "errors": dict(self.errors),
        }
        if extra:
            payload.update(extra)
        self.last_report = payload
        return dict(payload)
