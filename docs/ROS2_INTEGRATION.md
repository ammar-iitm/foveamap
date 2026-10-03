# FoveaMap — ROS 2 / System Integration (Phase 8)

## 1. Architecture

The ROS 2 layer is an adapter, not the core. All computation stays in
`foveamap` (usable without ROS 2 installed; core modules never import this
package, and importing it never requires `rclpy`):

```text
sensor_msgs/PointCloud2 ──> foveamap_ros/pointcloud.py ──╮
tf2 ──> foveamap_ros/frames.py (TransformProvider) ─────╰─> LiDARFrame
                                                              │
                                              FoveaMapRuntime.process()
                                                              │
                                              MapSnapshot (detached, atomic)
                                                        ┌─────┴──────┐
                                                        ▼            ▼
                                              messages.py      diagnostics.py
                                              /grid  ●●●●●●●●●  /metrics
                                              /points_labeled
```

Package layout (`foveamap_ros/`, plus `msg/*.msg` + `package.xml` for the
future colcon message package): `pointcloud`, `frames`, `config`, `qos`,
`messages`, `diagnostics`, `node` (`FoveaMapNodeCore` ROS-independent +
guarded `FoveaMapRosNode` wrapper).

## 2. Topics

| Topic | Type (near term) | Content |
|---|---|---|
| `/foveamap/grid` | custom `GridTile[]` (see `msg/`) / JSON payload | per-tier decimated cell rows: tier, centre xy, class, cost, ground, dynamic, traversability |
| `/foveamap/points_labeled` | `sensor_msgs/PointCloud2` (+ `label`, `confidence`, `moving`) | perception-annotated points, deterministic stride |
| `/foveamap/metrics` | custom `FoveaMapMetrics` / JSON payload | counters, latencies, device, temporal stats |

Only public snapshot/query data is published; internal `TierLayers` never
leaves the process. Complete snapshots are published for correctness
(bandwidth measured per message); deltas are explicitly future work.

## 3. Message formats

- Grid rows omit empty unknown cells and are stride-decimated
  (`max_cells_per_tier`, default 20000); `serialized_size_bytes` is measured
  on every payload for bandwidth observability.
- Labeled points reuse the binary PointCloud2 layout (no ROS dependency to
  build); header stamp/frame preserved from the snapshot.
- `.msg` sources live in `foveamap_ros/msg/` for the colcon build.

## 4. Input contract

`sensor_msgs/PointCloud2` with float `x/y/z` (values preserved exactly, no
hidden transforms). `intensity` (any standard name/storage) is normalized
with the core `normalize_intensity` policy; when absent, ones are used and
provenance records `default_ones_missing_field`. `ring` (any standard name)
zero-fills with `ring_available=False` when absent — ring IDs are never
invented. A float `time` channel is parsed into per-point offsets when
present (validated downstream; absent means normal processing with
provenance, never fabricated de-skew). Binary layout honors field offsets,
counts, `point_step`, and `row_step` (padded/shuffled/organized clouds). Little-endian only (explicit rejection otherwise); truncated,
mismatched, duplicate-field, NaN/Inf clouds raise `DataAdapterError`.

## 5. Output contract

Timestamps/frame IDs come from the snapshot (never re-stamped). Grid rows
carry tier, cell centre, class, cost, ground, dynamic flag, and
traversability computed with the authoritative configured threshold.
Points carry class/confidence/moving from the perception result that
produced the snapshot. Metrics carry stage latencies plus temporal stats.

## 6. TF assumptions

`sensor_frame` (cloud `frame_id`) → `base_frame` (default `base_link`,
runtime ego convention) → `world_frame` (default `map`, runtime pose).
Same-frame clouds need no transform (recorded, not assumed elsewhere).
Otherwise a timestamped transform is REQUIRED: missing → typed
`MissingTransformError` (frame dropped + counted), never silent identity;
stale (> `max_tf_age_s`, default 0.2 s) → `StaleTransformError` unless
`allow_stale_tf` (then recorded). Output frames keep the cloud stamp; frames
are never mixed across timestamps.
The runtime pose (`LiDARFrame.pose`, ego/base → world) is resolved from the
*same* cloud timestamp via a `(world_frame, base_frame)` lookup under the
identical stale policy. Identity pose is legal only for explicitly configured
same-frame operation (`world_frame == base_frame`, e.g. local-map startup),
recorded as `identity_same_frame_operation`; with no provider and differing
frames the frame fails loudly instead of drifting on a false identity.

## 7. Coordinate frames

Ego convention inherited from `LiDARFrame` (X forward, Y left, Z up, metres).
TF math is quaternion+translation → SE(3), applied in float64.

## 8. QoS

Sensor-data-like best-effort/volatile for input and snapshot topics
(stale data is worse than dropped data; depth 5 in / 1 out), reliable +
transient-local depth 1 for metrics (late joiners see counters). `keep_all`
is rejected by validation. See `qos.py`; convertible via `to_rclpy()`.

## 9. Configuration

Flat ROS params → `FoveaMapConfig` + `RosIOConfig` (`config.from_ros_params`;
unknown keys rejected, everything validated by existing dataclasses).
`perception.backend_type: classical` selects the explicit deterministic
fallback; RangeUNet requires a real checkpoint (missing file fails startup,
never silent untrained inference). No duplicate map semantics exist in this
package.

## 10. Lifecycle

`CREATED → CONFIGURED → ACTIVE ⇄ INACTIVE → SHUTDOWN`. `configure()`
validates the perception backend readiness; only `ACTIVE` accepts frames;
`shutdown()` drains the queue. Illegal transitions raise.

## 11. Failure handling

Typed failures per stage (`INPUT_*`, `RUNTIME_*` counters): malformed/empty/
bad-frame clouds, missing/stale TF, perception/mapping exceptions. Failed
frames return before `runtime.process`, so partial map updates are
impossible and the previous map is kept. No broad silent swallowing (a
last-resort `except Exception` counts `unexpected_*` and still drops safely).

## 12. Backpressure

Bounded `max_queue` (default 2) with `oldest`/`newest` drop policy; drops are
counted by reason (`dropped_backpressure_*`, `dropped_inactive`); queue depth
is published. No unbounded backlog, no stale-data processing.

## 13. Diagnostics

`MetricsAggregator`: received/processed/dropped, bounded 120-sample stage
latencies (INPUT/RUNTIME/PUBLICATION + perception internals when profiling),
rolling FPS, error counters, queue depth, grid payload bytes, temporal stats.
Device is reported honestly (`cpu`, or CUDA string when selected — never
measured GPU numbers on CPU).

## 14. Testing

`tests/test_phase8_ros.py`: 31 ROS-independent tests (conversion incl.
provenance/endianness/malformed/empty, TF math/missing/stale, config
mapping/rejection, QoS bounds, lifecycle, queue bounds + both drop
policies, full numpy + torch-CPU cycles, determinism, snapshot isolation,
serialization roundtrip, bandwidth metadata) plus 2 rclpy-guarded tests that
skip with an explicit reason without ROS 2.

## 15. Known limitations

- rclpy integration is structural only here (no ROS 2 on this machine);
  topic wiring against a real `rclpy` executor happens on the robot/Colab.
- Complete-snapshot publication (no deltas yet); bandwidth scales with
  observed cells × `max_cells_per_tier` cap.
- No executor/threading model beyond synchronous `spin_once` (queue-ready
  for a bounded worker).
- Classical backend in tests is a geometric stand-in, not learned
  perception; no accuracy claims.
- No CUDA measurement on this host; GPU validation stays deferred.
