# FoveaMap — Python SDK (Phase 9)

## 1. Architecture

```text
consumer -> foveamap.sdk.FoveaMap -> FoveaMapRuntime -> core
```

The SDK is an adapter: it owns lifecycle, ownership, health, metrics, and
serialization. Mapping, traversability, lifecycle, and query policy all live
in the core and are never reimplemented here.

## 2. Installation / import

```python
from foveamap.sdk import FoveaMap, API_VERSION  # API_VERSION == "1"
```

Works on CPU without ROS 2 or CUDA. `foveamap_ros` and the SDK are
independent adapters over the same runtime.

## 3. Basic usage

```python
mapper = FoveaMap(perception_backend=backend)  # or FoveaMap(config=...)
mapper.configure()
mapper.start()
view = mapper.process(frame)        # LiDARFrame or legacy dict
result = view.query_point(x, y)     # or mapper.query_point(x, y)
mapper.metrics()
mapper.reset()
mapper.close()
```

## 4. Configuration

`FoveaMapConfig` directly, or a strict dict (`{"terrain": {...}, ...}` —
unknown sections/keys rejected with `SDKConfigError`). No second hierarchy;
dicts convert immediately into canonical dataclasses.

## 5. Lifecycle

`CREATED -> CONFIGURED -> ACTIVE ⇄ INACTIVE -> SHUTDOWN` (`configure`,
`start`, `stop`, `close`). `process`/`reset` outside their states raise
`SDKLifecycleError`. `reset()` clears map/temporal/counters/snapshot but
keeps configuration and backend.

## 6. Processing

`process(frame)` accepts `LiDARFrame` (primary) or legacy dicts (thin
pass-through to the runtime). Core `FoveaMapError`s propagate unchanged;
unexpected failures become `SDKError` with the cause chained. Failed frames
count as dropped; the session stays usable.

## 7. Querying

`query_point(x, y)`, `query_points(xs, ys)` (bounded, default limit 4096),
`query_ray(x, y, theta, step_m, max_steps)` (bounded ray sampling that
delegates every sample to the snapshot — no new mapping logic),
`is_traversable(x, y, clearance_req=0.0)` and `export_numpy()` (detached
tier-array copies for offline consumers) — all answered from the latest
detached snapshot via the snapshot's authoritative implementation
(including its traversability policy). `QueryResult` is frozen and carries
tier, state, class, confidence, ground, roughness, slope, clearance, cost,
dynamic state, traversability, and age.

## 8. Snapshots

`snapshot()` returns a `SnapshotView`: read-only, no arrays, no mutators,
backed by the already-detached core snapshot. Later frames cannot mutate a
published view. `to_dict()` is JSON-safe and version-stamped.

## 9. Health / status

`health()` → `healthy` / `degraded` / `unavailable` with explicit reasons
(`fallback_perception_backend`, `repeated_frame_failures:N`,
`last_error:…`, non-ACTIVE lifecycle). `status()` adds lifecycle, device,
engines, backend, counters, last frame/error, and ownership.

## 10. Metrics

`metrics()` reports measured values only: SDK FPS, per-stage means,
runtime timings (when the runtime collected them), map memory
(`allocated_bytes` etc.), temporal track count, device plus an honest CUDA
note (`cuda_unavailable_cpu_execution` on CPU).

## 11. Errors

`SDKError` (base), `SDKLifecycleError`, `SDKQueryError`, `SDKConfigError`
(also a `ConfigurationError` for core-compat). Core errors pass through
unchanged; unexpected errors chain their cause. Nothing is silently
swallowed: failures are raised and counted.

## 12. Reset

See §5: state cleared, config/backend preserved, deterministic clean state
verified by tests.

## 13. Ownership

`FoveaMap()` creates and owns its runtime (`owns_runtime=True`; `close()`
resets it). `FoveaMap(runtime=rt)` borrows (`owns_runtime=False`;
`close()` never touches the foreign runtime). Documented and tested.

## 14. Concurrency

One `RLock` guards `process`/`reset`/lifecycle transitions (single-writer
model). Queries read the latest detached snapshot reference and are safe
against concurrent processing. No thread pools, no async; GPU ops are never
made concurrent by this layer.

## 15. Serialization

`to_dict()` on every public type is JSON-compatible (`api_version`
included; NaN/Inf → None; tuples → lists; NumPy scalars converted). No
pickle anywhere; no tensors/arrays serialized.

## 16. Performance

Measured on CPU (classical backend, numpy grid, 2000-pt frames):
raw runtime ≈186 ms/frame vs SDK ≈195 ms/frame (≈8.5 ms / 4.6% wrapper
overhead: view wrap + bookkeeping). Single query ≈14 ms (two core query
evaluations: state dict + authoritative traversability; documented cost of
not duplicating policy). `SnapshotView.to_dict()` is sub-millisecond.
Batch queries are capped (default 4096) to bound the cost. No GPU numbers
(CUDA unavailable here).

## 17. Limitations

- Queries reflect the latest snapshot only (no live-grid reads).
- Batch query cost scales with core query cost (see §16).
- HTTP surface (if used) is loopback-only; see `docs/API.md`.
- No authentication, no persistence, no streaming deltas (out of scope).

## 18. Versioning

`API_VERSION = "1"`, stamped on every payload. Breaking field changes later
require a version bump.
