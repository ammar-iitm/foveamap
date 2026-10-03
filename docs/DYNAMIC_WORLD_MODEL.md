# FoveaMap — Dynamic World Model & Temporal Environment Intelligence (Phase 6)

## 1. Architecture

Phase 6 evolves FoveaMap from a spatial foveated map into a temporally aware
world representation. The data flow is now explicit in code:

```text
LiDAR Frame t
      |
      v
Preprocess -> Perception (class probs + is_moving, world-frame points)
      |
      +------------+------------+
      |                         |
      v                         v
STATIC EVIDENCE           DYNAMIC EVIDENCE (per-frame dynamic_mask/dyn dicts,
      |                   rebuilt fresh every frame, never fused to static)
      v                         |
Persistent Map                  v
(static EMA layers)   Temporal Dynamic Map (DynamicWorldModel:
      |               bounded tracks + lifecycle, world coordinates)
      +------------+------------+
                   |
                   v
           Temporal Fusion (fuse_stats order:
            scroll -> static -> lifecycle -> terrain -> ray clearing)
                   |
                   v
             World State t
              /    |    \
         STATIC DYNAMIC UNKNOWN/STALE
              \    |    /
                   v
              MapSnapshot (static layers + dynamic_tracks, published atomically)
```

Module: `foveamap/temporal.py` (`DynamicWorldModel`, `DynamicTrack`,
`DynamicObservation`, lifecycle constants). One engine-agnostic
implementation is shared by `FoveatedGrid` and `TorchFoveatedGrid` (exact
parity by construction). Configuration: `DynamicConfig` in
`foveamap/core/config.py`, exposed as `FoveaMapConfig.dynamic`.

## 2. State lifecycle

```text
UNKNOWN -> OBSERVED -> ACTIVE_DYNAMIC -> TEMPORARILY_MISSING -> STALE -> REMOVED
```

- `OBSERVED`: seen in fewer than `activation_frames` consecutive frames.
  Provisional but occupied: reports `dynamic=True`, blocks traversability.
- `ACTIVE_DYNAMIC`: `hits >= activation_frames` **and**
  `conf >= confidence_threshold`. Confirmed occupant.
- `TEMPORARILY_MISSING`: unobserved for `1..missing_tolerance_frames` frames.
  Still occupied (conservative: the object may be briefly occluded).
- `STALE`: missing for `(tolerance, stale_frames]`. Released: `dynamic=False`,
  traversability falls back to static geometry.
- `REMOVED`: missing beyond `stale_frames`. Track deleted; cells report
  `UNKNOWN` (or static state if geometry exists).

Defaults: `activation_frames=2`, `missing_tolerance_frames=2`,
`stale_frames=4`, `confidence_threshold=0.5`. A lone noisy moving point
therefore never becomes `ACTIVE_DYNAMIC`, but is still visible as
`OBSERVED` while present.

Legacy query strings are preserved: `OBSERVED`/`ACTIVE_DYNAMIC` report
`"OBSERVED_DYNAMIC"`/`"ACTIVE_DYNAMIC"`; `"TEMPORARILY_MISSING"` is new.
Queries additionally expose fine-grained `dynamic_state`,
`dynamic_confidence`, `dynamic_age_frames`, and `velocity`.

## 3. Coordinate frames & ego-motion awareness

- Correspondence operates exclusively on **world coordinates** and
  **post-scroll grid indices**. The foveated window scroll (`shifted`)
  compensates ego motion for all persistent state; the temporal model
  mirrors the scroll via `on_scroll(deltas)` (same `(i - d0, j - d1)`
  mapping, out-of-window tracks dropped).
- Raw sensor-frame point indices are never compared across frames
  (index N in frame t is unrelated to index N in frame t+1).
- Each dynamic observation carries the world position of its cell centre;
  tracks store world `(x, y)` and grid `(tier, i, j)`.

## 4. Temporal correspondence

- Observations are matched to tracks by world proximity
  (`correspondence_distance_m`, default 1.5 m) plus semantic compatibility
  (same class, or both in `dynamic_classes`).
- Matching is **cross-tier**: integer-lattice mip-up reports one physical
  object in several tiers, and all of those observations feed a single track
  anchored at the finest reporting tier. Same-frame duplicates never
  double-count (one hit per track per frame).
- Lookup is a 2 m spatial hash whose searched ring radius is derived from the
  configured `correspondence_distance_m` (`ceil(distance / 2 m)`, so any
  validated distance is fully covered by construction); per-frame cost is
  `O(dynamic cells)`, never `O(map)` or `O(history)`.
- Velocity is optional evidence derived from consecutive world positions
  (`(x_t - x_{t-1}) / dt`, timestamps from the frame; 10 Hz deterministic
  fallback). It is published for consumers, never required for association,
  and never destabilizes the core.

## 5. Static/dynamic separation & conflict resolution

Precedence rule (read-only at query time; arrays are never mixed):

> **Current dynamic evidence overrides static geometry for query and
> traversability, but dynamic observations never overwrite persistent
> static arrays.**

Mechanisms (all pre-existing Phase 5, preserved):

- Binned moving points are excluded from static accumulation
  (`n_static`/`p_static`/heights use `~moving` only).
- `dynamic_mask` is rebuilt fresh every frame (zero ghost trails in static).
- Lifecycle tracks live in a separate bounded store.

Consequences: a vehicle temporarily on a road leaves the static `road`
class, ground height, and cost untouched; while present the cell reports
dynamic + not traversable; after expiry the road is traversable again with
its original cost. A previously static-looking cell is never permanently
blocked by a transient occupant.

## 6. Confidence & decay

`dynamic_confidence` (query field and track `confidence`) means **temporal
persistence confidence** — how consistently the region has been observed
dynamic — and NOT neural perception confidence. Perception confidence already
gated the observation itself (the per-point moving decision upstream); the
lifecycle layer only counts repetitions. Conflating the two would let a
single high-probability misclassification activate a track.

- New track: `initial_confidence` (0.5).
- Observed frame: `conf = min(1.0, conf + hit_increment)` (0.3).
- Missing frame: `conf *= decay_factor` (0.75); consecutive-hit counter resets.
- Promotion additionally requires `conf >= confidence_threshold`.

No learned temporal model; simple, deterministic, configurable.

## 7. Ray-clearing interaction

Phase 5 guards are intact and extended by construction:

- Clearing decisions use the **frame-isolated observation mask**, not
  lifecycle state: an `ACTIVE` cell (observed this frame) is never cleared;
  a `MISSING` cell (no observation) is clearable by valid free-space rays,
  so a departed vehicle correctly yields to free space.
- Ground classes, static obstacles behind returns, occluded regions, and
  `UNKNOWN` handling are unchanged. `TorchFoveatedGrid._clear_rays` mirrors
  the NumPy guards; only its ray-geometry sampling loop is host-side
  (explicit opt-in boundary — clearing is disabled by default and never runs
  in the default Torch hot path), while streak updates and obstacle clearing
  are device-resident masked tensor ops with no per-cell host sync.

## 8. Query & snapshot semantics

Publication order per frame: preprocess -> perception -> static fusion ->
dynamic lifecycle -> terrain/traversability -> snapshot. The snapshot is
built only after all updates complete, so it always represents one coherent
logical frame.

- `grid.query_point` / `TorchFoveatedGrid.query_point`: `dynamic` is
  `mask OR lifecycle-occupied`; new keys `dynamic_state`,
  `dynamic_confidence`, `dynamic_age_frames`, `velocity` (all
  backward-compatible additions).
- `MapSnapshot` gains `dynamic_tracks` (detached plain dicts) and
  `temporal_metadata` (frame stats); its `query_point` applies the same
  precedence from the detached track index. Manual snapshots without tracks
  behave exactly as in Phase 5.
- Traversability: `ACTIVE`/`MISSING`/`OBSERVED` block (`is_traversable`
  False); `STALE`/removed fall back to static cost, so cleared road becomes
  traversable again without any static-cost mutation.
- Ownership follows Phase 5 rules: detached snapshot-owned data; live-grid
  mutation cannot affect a published snapshot.

## 9. Configuration

All Phase 6 thresholds live in `DynamicConfig` (validated; impossible values
rejected, e.g. `stale_frames <= missing_tolerance_frames`, `max_tracks < 1`):

`activation_frames`, `missing_tolerance_frames`, `stale_frames`,
`confidence_threshold`, `initial_confidence`, `hit_increment`,
`decay_factor`, `correspondence_distance_m`, `dynamic_classes`,
`max_tracks` (default 20000).

## 10. Memory bounds

- Live tracks `<= max_tracks` (deterministic eviction: stalest first, then
  oldest, then smallest id). No per-frame history retained.
- `memory_report()` adds `temporal_tracks`, `temporal_tracks_capacity`,
  `temporal_bytes_estimate` (`tracks x 192 B` documented estimate),
  `temporal_stats`. PRD grid-layer keys (`allocated_bytes == 5120000`,
  `under_8mb_target`) are unchanged; temporal state is reported as separate
  runtime overhead.

## 11. NumPy / Torch behavior

- Single shared `DynamicWorldModel`; Torch grids feed it sparse host-side
  dynamic cell indices once per frame (intentional documented boundary over
  tens of cells, never per-point tensors, no hot-path sync).
- Integer lifecycle/semantic state is exact across engines; confidence and
  velocity are float-identical in practice (same arithmetic order).
- CUDA has no dedicated temporal kernels; GPU validation is structural on
  CPU (device-resident masks stay on-device; lifecycle is sparse host
  bookkeeping). Physical CUDA verification remains deferred to Colab/Kaggle.

## 12. Limitations & non-goals (deliberate)

- No multi-object tracker: no persistent object IDs across long occlusions
  beyond the missing/stale window, no data association across classes, no
  motion prediction. The question answered is "is this region currently
  occupied by a dynamic entity", not MOT.
- No SLAM/localization/planning/fusion changes; no ROS2/SDK/TensorRT/ONNX.
- Velocity is first-order finite-difference evidence only (no filtering).
- Fast objects crossing tier windows re-anchor (track tier follows the
  finest reporting tier); identity across a tier-boundary crossing is kept
  by world proximity while within `correspondence_distance_m`.
- Real-world dynamic detection accuracy is not claimed here (no dataset
  evaluation in Phase 6 scope); correctness is established by deterministic
  synthetic/golden scenarios plus soak and parity tests.
