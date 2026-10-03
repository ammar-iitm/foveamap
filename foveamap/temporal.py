"""Temporal dynamic world model (Phase 6).

Evolves FoveaMap from a spatial foveated map into a temporally aware world
representation. The per-frame dynamic observation masks owned by the grid
engines (`dynamic_mask`, rebuilt fresh every frame) remain the *observation*
layer; this module is the *temporal* layer that turns frame-isolated dynamic
evidence into a bounded, deterministic lifecycle:

    UNKNOWN -> OBSERVED -> ACTIVE_DYNAMIC -> TEMPORARILY_MISSING -> STALE -> REMOVED

Design rules:
- One engine-agnostic implementation shared by the NumPy and Torch grid
  engines (exact parity by construction). Torch grids feed it sparse
  host-side dynamic cell indices once per frame; this is an intentional,
  documented sparse host boundary over tens of cells, never per-point data.
- All correspondence happens in world coordinates on post-scroll grid
  indices, so ego motion is already compensated by the foveated window
  scroll. Raw sensor-frame indices are never compared across frames.
- Bounded: at most ``max_tracks`` live tracks; deterministic eviction of the
  stalest tracks when exceeded. No per-frame history is retained.
- Static grid arrays are never written here; dynamic evidence never becomes
  static geometry. Query/traversability precedence (dynamic over static) is
  applied read-only at query time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from .core.config import DynamicConfig

# Lifecycle states (plain strings for snapshot/query serializability).
UNKNOWN = "UNKNOWN"
OBSERVED = "OBSERVED"
ACTIVE_DYNAMIC = "ACTIVE_DYNAMIC"
TEMPORARILY_MISSING = "TEMPORARILY_MISSING"
STALE = "STALE"
REMOVED = "REMOVED"

# Occupied states: the spatial region counts as dynamically occupied
# (blocks traversability, reports dynamic=True).
OCCUPIED_STATES = (OBSERVED, ACTIVE_DYNAMIC, TEMPORARILY_MISSING)

# Coarse spatial hash bucket width (m). The match neighborhood ring is derived
# from the configured correspondence distance (see _match_ring), so any
# validated correspondence_distance_m is fully covered by construction.
_BUCKET_M = 2.0

# Documented per-track memory estimate (bytes) used by memory_report().
TRACK_BYTES_ESTIMATE = 192


@dataclass
class DynamicObservation:
    """Single-frame dynamic evidence for one grid cell (world frame)."""

    tier: int
    i: int
    j: int
    x: float  # world x (m) of the observed cell centre
    y: float  # world y (m) of the observed cell centre
    cls: int  # semantic class id (vehicle/person in practice)
    conf: float = 1.0  # per-observation evidence weight in [0, 1] (plumbed
    # for future motion-probability weighting; lifecycle confidence itself is
    # purely temporal persistence evidence, see DynamicWorldModel.update)
    count: int = 1  # dynamic points binned into this cell this frame


@dataclass
class DynamicTrack:
    """Bounded temporal state for one dynamically occupied cell region."""

    tid: int
    tier: int
    i: int
    j: int
    x: float
    y: float
    cls: int
    conf: float
    count: int
    hits: int  # consecutive observed frames
    missing: int  # consecutive unobserved frames
    first_seen: int  # frame index of creation
    last_seen: int  # frame index of last observation
    last_time: float  # timestamp of last observation
    vx: float = 0.0  # world-frame velocity x (m/s), valid if has_velocity
    vy: float = 0.0  # world-frame velocity y (m/s), valid if has_velocity
    has_velocity: bool = False


def _bucket(x: float, y: float) -> tuple[int, int]:
    return (math.floor(x / _BUCKET_M), math.floor(y / _BUCKET_M))


def _match_ring(correspondence_distance_m: float) -> int:
    """Bucket ring radius covering the correspondence distance.

    A track up to ``correspondence_distance_m`` away can sit at most
    ``ceil(distance / _BUCKET_M)`` buckets from the observation bucket, so the
    searched ``(2*ring+1)^2`` neighborhood always contains every candidate.
    """
    return max(1, math.ceil(float(correspondence_distance_m) / _BUCKET_M))


class DynamicWorldModel:
    """Deterministic, bounded temporal dynamic state shared by both engines."""

    def __init__(self, config: DynamicConfig | None = None) -> None:
        self.config = config if config is not None else DynamicConfig()
        self._tracks: dict[int, DynamicTrack] = {}
        self._buckets: dict[tuple[int, int], set[int]] = {}
        self._next_tid = 0
        self.frame_index = -1
        self.last_timestamp: float | None = None
        self.total_expired = 0
        self.total_evicted = 0
        self._last_stats: dict[str, Any] = {
            "n_observed": 0,
            "n_provisional": 0,
            "n_active": 0,
            "n_missing": 0,
            "n_stale": 0,
            "n_expired": 0,
            "n_evicted": 0,
            "n_tracks": 0,
        }

    # ------------------------------------------------------------- lifecycle
    @staticmethod
    def derive_state(track: DynamicTrack, config: DynamicConfig) -> str:
        """Pure function mapping track counters to lifecycle state."""
        if track.missing > config.stale_frames:
            return REMOVED
        if track.missing > config.missing_tolerance_frames:
            return STALE
        if track.missing > 0:
            return TEMPORARILY_MISSING
        if track.hits >= config.activation_frames and track.conf >= config.confidence_threshold:
            return ACTIVE_DYNAMIC
        return OBSERVED

    def state_of(self, track: DynamicTrack) -> str:
        return self.derive_state(track, self.config)

    @staticmethod
    def is_occupied_state(state: str) -> bool:
        return state in OCCUPIED_STATES

    # ---------------------------------------------------------------- update
    def update(
        self,
        observations: Sequence[DynamicObservation],
        frame_idx: int | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """Advance one frame. Returns per-frame statistics (plain values)."""
        cfg = self.config
        if frame_idx is None:
            frame_idx = self.frame_index + 1
        self.frame_index = int(frame_idx)
        ts = float(timestamp) if timestamp is not None else float(frame_idx) * 0.1
        dt_frame = 0.1
        if self.last_timestamp is not None:
            dt_frame = ts - self.last_timestamp
        self.last_timestamp = ts

        seen: set[int] = set()
        n_observed = 0
        # Deterministic order: finest tier first so the track anchors at the
        # finest reporting tier; coarser mip-up duplicates of the same object
        # match the same track (cross-tier, world proximity) but must not
        # double-count hits within one frame (last_seen guard below).
        for ob in sorted(observations, key=lambda o: (o.tier, o.i, o.j)):
            match = self._match(ob)
            if match is None:
                tid = self._next_tid
                self._next_tid += 1
                # Initial confidence is the configured prior: confidence is
                # purely temporal persistence evidence (repeated observation),
                # because the per-point moving gate already decided that this
                # cell carries dynamic evidence this frame.
                track = DynamicTrack(
                    tid=tid,
                    tier=int(ob.tier),
                    i=int(ob.i),
                    j=int(ob.j),
                    x=float(ob.x),
                    y=float(ob.y),
                    cls=int(ob.cls),
                    conf=float(cfg.initial_confidence),
                    count=int(ob.count),
                    hits=1,
                    missing=0,
                    first_seen=int(frame_idx),
                    last_seen=int(frame_idx),
                    last_time=ts,
                )
                self._tracks[tid] = track
                self._add_bucket(track)
            else:
                track = match
                if track.last_seen == int(frame_idx):
                    # Same-frame mip-up duplicate from a coarser tier: the
                    # track already counted this frame; the finest-tier anchor
                    # (processed first) is kept.
                    seen.add(track.tid)
                    continue
                self._move_bucket(track, float(ob.x), float(ob.y), int(ob.tier), int(ob.i), int(ob.j))
                if dt_frame > 1e-6:
                    track.vx = (float(ob.x) - track.x) / dt_frame
                    track.vy = (float(ob.y) - track.y) / dt_frame
                    track.has_velocity = True
                track.x = float(ob.x)
                track.y = float(ob.y)
                track.cls = int(ob.cls)
                track.conf = min(1.0, track.conf + cfg.hit_increment)
                track.count = int(ob.count)
                track.hits += 1
                track.missing = 0
                track.last_seen = int(frame_idx)
                track.last_time = ts
            seen.add(track.tid)
            n_observed += 1

        # Decay unobserved tracks; expire beyond stale_frames.
        expired: list[int] = []
        for tid, track in self._tracks.items():
            if tid in seen:
                continue
            track.missing += 1
            track.hits = 0
            track.conf *= cfg.decay_factor
            if track.missing > cfg.stale_frames:
                expired.append(tid)
        for tid in expired:
            self._remove(tid)
        self.total_expired += len(expired)

        # Deterministic eviction when bounded capacity is exceeded.
        evicted = 0
        while len(self._tracks) > cfg.max_tracks:
            victim = min(
                self._tracks.values(),
                key=lambda t: (-t.missing, t.last_seen, t.tid),
            )
            # min() with negated missing selects largest missing; ties break
            # by oldest last_seen then smallest tid -> fully deterministic.
            self._remove(victim.tid)
            evicted += 1
        self.total_evicted += evicted

        # State counters use exact lifecycle names: n_provisional counts
        # OBSERVED tracks, n_active counts ACTIVE_DYNAMIC tracks only.
        n_provisional = n_active = n_missing = n_stale = 0
        for track in self._tracks.values():
            st = self.state_of(track)
            if st == ACTIVE_DYNAMIC:
                n_active += 1
            elif st == OBSERVED:
                n_provisional += 1
            elif st == TEMPORARILY_MISSING:
                n_missing += 1
            elif st == STALE:
                n_stale += 1
        self._last_stats = {
            "n_observed": n_observed,
            "n_provisional": n_provisional,
            "n_active": n_active,
            "n_missing": n_missing,
            "n_stale": n_stale,
            "n_expired": len(expired),
            "n_evicted": evicted,
            "n_tracks": len(self._tracks),
        }
        return dict(self._last_stats)

    # ---------------------------------------------------------- correspondence
    def _match(self, ob: DynamicObservation) -> DynamicTrack | None:
        """Nearest world-frame track within correspondence distance.

        Cross-tier by design: mip-up reports one object in several tiers. Any
        tier may match; determinism comes from (distance, tier, tid) ordering:
        nearest first, then finest tier, then smallest track id.
        """
        cfg = self.config
        dyn_classes = set(int(c) for c in cfg.dynamic_classes)
        bx, by = _bucket(ob.x, ob.y)
        best: DynamicTrack | None = None
        best_key: tuple[float, int, int] | None = None
        max_d2 = cfg.correspondence_distance_m * cfg.correspondence_distance_m
        ring = _match_ring(cfg.correspondence_distance_m)
        for dbx in range(-ring, ring + 1):
            for dby in range(-ring, ring + 1):
                for tid in self._buckets.get((bx + dbx, by + dby), ()):
                    track = self._tracks.get(tid)
                    if track is None:
                        continue
                    if not (int(ob.cls) == int(track.cls) or (int(ob.cls) in dyn_classes and int(track.cls) in dyn_classes)):
                        continue
                    dx = float(ob.x) - track.x
                    dy = float(ob.y) - track.y
                    d2 = dx * dx + dy * dy
                    if d2 > max_d2:
                        continue
                    key = (d2, int(track.tier), int(track.tid))
                    if best_key is None or key < best_key:
                        best_key = key
                        best = track
        return best

    # ---------------------------------------------------------------- buckets
    def _add_bucket(self, track: DynamicTrack) -> None:
        key = _bucket(track.x, track.y)
        self._buckets.setdefault(key, set()).add(track.tid)

    def _move_bucket(self, track: DynamicTrack, x: float, y: float, tier: int, i: int, j: int) -> None:
        old_key = _bucket(track.x, track.y)
        new_key = _bucket(x, y)
        track.tier, track.i, track.j = int(tier), int(i), int(j)
        if old_key == new_key:
            return
        old_set = self._buckets.get(old_key)
        if old_set is not None:
            old_set.discard(track.tid)
            if not old_set:
                del self._buckets[old_key]
        self._buckets.setdefault(new_key, set()).add(track.tid)

    def _remove(self, tid: int) -> None:
        track = self._tracks.pop(tid, None)
        if track is None:
            return
        key = _bucket(track.x, track.y)
        bucket = self._buckets.get(key)
        if bucket is not None:
            bucket.discard(tid)
            if not bucket:
                del self._buckets[key]

    # ------------------------------------------------------------------ scroll
    def on_scroll(self, deltas: Sequence[tuple[int, int]], tier_n: Sequence[int]) -> None:
        """Shift track grid coordinates with the foveated window scroll.

        Mirrors ``TierLayers.shifted``: a world-fixed cell at old (i, j) moves
        to (i - d0, j - d1) where d = new_origin - old_origin. Tracks leaving
        every tier window are dropped (they will re-activate on re-observation;
        window eviction is distinct from lifecycle expiry and is not counted in
        ``total_expired``).
        """
        for track in list(self._tracks.values()):
            k = int(track.tier)
            if k < 0 or k >= len(deltas):
                self._remove(track.tid)
                continue
            d0, d1 = int(deltas[k][0]), int(deltas[k][1])
            ni, nj = int(track.i) - d0, int(track.j) - d1
            n = int(tier_n[k])
            if 0 <= ni < n and 0 <= nj < n:
                track.i, track.j = ni, nj
            else:
                self._remove(track.tid)

    # ------------------------------------------------------------------- query
    def lookup(self, tier: int, i: int, j: int) -> DynamicTrack | None:
        """Exact-cell track lookup for query enrichment (no allocation).

        When several tracks share a cell (possible only across incompatible
        semantics), an occupied track is preferred; ties follow insertion
        order, so results stay deterministic.
        """
        fallback: DynamicTrack | None = None
        for track in self._tracks.values():
            if int(track.tier) == int(tier) and int(track.i) == int(i) and int(track.j) == int(j):
                if self.state_of(track) in OCCUPIED_STATES:
                    return track
                if fallback is None:
                    fallback = track
        return fallback

    def query_enrichment(self, tier: int, i: int, j: int) -> dict[str, Any] | None:
        """Lifecycle enrichment for a queried cell, or None when no track."""
        track = self.lookup(tier, i, j)
        if track is None:
            return None
        state = self.state_of(track)
        return {
            "dynamic_state": state,
            "dynamic_confidence": float(track.conf),
            "dynamic_age_frames": int(self.frame_index - track.last_seen) if self.frame_index >= 0 else 0,
            "dynamic_hits": int(track.hits),
            "dynamic_missing": int(track.missing),
            "dynamic_occupied": bool(state in OCCUPIED_STATES),
            "velocity": [float(track.vx), float(track.vy)] if track.has_velocity else None,
            "track_id": int(track.tid),
        }

    # ---------------------------------------------------------------- snapshot
    def snapshot_tracks(self) -> tuple[dict[str, Any], ...]:
        """Detached, serializable copy of live tracks sorted by track id."""
        out: list[dict[str, Any]] = []
        for track in sorted(self._tracks.values(), key=lambda t: t.tid):
            out.append(
                {
                    "track_id": int(track.tid),
                    "tier": int(track.tier),
                    "cell": (int(track.i), int(track.j)),
                    "world_xy": [float(track.x), float(track.y)],
                    "cls": int(track.cls),
                    "confidence": float(track.conf),
                    "count": int(track.count),
                    "state": self.state_of(track),
                    "hits": int(track.hits),
                    "missing": int(track.missing),
                    "first_seen": int(track.first_seen),
                    "last_seen": int(track.last_seen),
                    "velocity": [float(track.vx), float(track.vy)] if track.has_velocity else None,
                }
            )
        return tuple(out)

    def stats(self) -> dict[str, Any]:
        """Last-frame statistics plus lifetime counters (plain values)."""
        return dict(
            self._last_stats,
            frame_index=int(self.frame_index),
            total_expired=int(self.total_expired),
            total_evicted=int(self.total_evicted),
        )

    def memory_report(self) -> dict[str, Any]:
        """Bounded-state memory accounting (documented estimate)."""
        n = len(self._tracks)
        return {
            "temporal_tracks": n,
            "temporal_tracks_capacity": int(self.config.max_tracks),
            "temporal_bytes_estimate": n * TRACK_BYTES_ESTIMATE,
            "temporal_bytes_per_track_estimate": TRACK_BYTES_ESTIMATE,
            "temporal_total_expired": int(self.total_expired),
            "temporal_total_evicted": int(self.total_evicted),
        }

    def reset(self) -> None:
        """Remove all temporal state (no ghost state may survive reset)."""
        self._tracks.clear()
        self._buckets.clear()
        self._next_tid = 0
        self.frame_index = -1
        self.last_timestamp = None
        self.total_expired = 0
        self.total_evicted = 0
        self._last_stats = {
            "n_observed": 0,
            "n_provisional": 0,
            "n_active": 0,
            "n_missing": 0,
            "n_stale": 0,
            "n_expired": 0,
            "n_evicted": 0,
            "n_tracks": 0,
        }

    def __len__(self) -> int:
        return len(self._tracks)
