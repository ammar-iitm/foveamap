# FoveaMap Architecture (as implemented)

Reference for contributors. Every claim below is traceable to source; planned
work is noted as such and not described as built.

## 1. Data flow

```
sweep ──► features (8 ch range image)
        ──► range-image U-Net (9 classes + moving head)
        ──► projection (GPU-or-CPU point transforms)
        ──► foveated grid (tier select, scatter-reduce, mip-up)
        ──► temporal world (dynamic lifecycle) + terrain/traversability
        ──► publish: SDK / HTTP / ROS 2 / dashboard
```

Entry points: `foveamap/cli.py` (product CLI), `foveamap/runtime/runtime.py`
(`FoveaMapRuntime`: `configure → start → process(frame) → … → reset/stop`),
`foveamap/sdk/client.py` (`FoveaMap`), `foveamap/sdk/http.py`
(`FoveaMapHttpServer`), `foveamap_ros/node.py` (ROS 2 adapter).

## 2. Module map

| Area | Location |
|---|---|
| Contracts, config, ontology, errors | `foveamap/core/{contracts,config,ontology,exceptions}.py` |
| Frames, features | `foveamap/frames.py`, `foveamap/features_torch.py` |
| Perception (range U-Net + backends) | `foveamap/model.py`, `foveamap/runtime/perception.py` |
| Foveated grid (NumPy / Torch parity) | `foveamap/grid.py`, `foveamap/grid_torch.py` |
| Temporal world model | `foveamap/temporal.py` |
| Terrain / traversability | `foveamap/terrain.py` |
| Pipeline + benchmark harness | `foveamap/pipeline.py`, `foveamap/benchmarks/` |
| Data sources (sim, KITTI, nuScenes, files) | `foveamap/data/`, `foveamap/{sim,nuscenes,semantickitti,kitti}.py` |
| SDK + HTTP | `foveamap/sdk/{client,http,types,errors}.py` |
| CLI | `foveamap/cli.py` |
| ROS 2 adapter | `foveamap_ros/` |
| Benchmarks, scripts | `benchmarks/`, `scripts/` |
| Tests | `tests/` (34 modules) |

## 3. Contracts and frames

- `LiDARFrame` (adapter → pipeline): ego-frame points in meters
  (X-forward / Y-left / Z-up), `pose` SE(3) ego→world, `sensor_origin`
  `[0,0,1.73]`, `timestamp` seconds, optional `label`/`moving`/`time_offsets`
  and `prev_sweeps` motion context. `validate()` enforces dtypes, shapes,
  and finite values.
- `PerceptionResult` (perception → mapping): per-point class distribution
  over 9 ontology ids + moving probability, with device-aware
  (`DevicePerceptionResult`) and host forms.
- `MapSnapshot` (mapping → consumers): copied ego pose, per-tier origins and
  cell layers, detached dynamic cells/tracks, `query_point` /
  `is_traversable` as the authoritative 2.5D read API.

## 4. Coordinate frames and units

- Points: ego frame, meters, X-forward / Y-left / Z-up (ISO 8855 orientation).
- Poses: 4×4 float64 SE(3), ego→world; world is the first scan's ego frame
  for cached datasets.
- Grid: world-aligned scrolling windows snapped to the coarse lattice;
  tier-0 cells 5 cm (±10 m), outer tier 50 cm (±100 m) in the `spec` preset.
- Time: seconds (float); per-point `time_offsets` bounded to ±600 s.
- Elevation/clearance/roughness: meters; slope: radians; cost: unitless
  0–254 with `traversable_cost_max=180`.

## 5. Tier mathematics

Tiers nest exactly via integer fine indices with floor-division: coarse cells
are integer multiples of the base resolution and extents, windows snap to the
coarse lattice and scroll with the vehicle, and every point is assigned to
exactly one tier (0 points lost at boundaries — asserted every frame by
`test_golden_frame.py` and scrolling-drive tests). Each cell is a 16-byte
structure-of-arrays record holding fused class, ground height, roughness,
overhang clearance, curb-step/pothole flags, and traversability cost.
`TorchFoveatedGrid` keeps the same layout and is parity-tested cell-by-cell
against `FoveatedGrid`.

## 6. Perception

`model.py` implements the range-image U-Net (9-class + moving/static heads,
FP16-capable, class masking for datasets missing classes). Backends
(`classical`, `heuristic`, `range_unet`, …) are registered in
`runtime/perception.py`; the default checkpoint is
`checkpoints/range_unet.pt` when present, otherwise an explicit
development-only untrained mode (`allow_untrained=True`, loud warning).
Device selection flows through `RuntimeConfig(device)` and
`FoveaMapConfig.{cpu_dev,gpu_dev,benchmark,demo,ros2}()`.

## 7. Temporal world model

Bounded lifecycle `OBSERVED → ACTIVE → MISSING → STALE → REMOVED` over
frame-isolated dynamic evidence, shared by both grid engines (see
`docs/DYNAMIC_WORLD_MODEL.md`). Per-cell world-frame velocity evidence is
published; there is deliberately no multi-object tracker. Free-space ray
clearing (`TerrainConfig.enable_ray_clearing`) treats dynamic occupancy as
temporary blockage without erasing static structure.

## 8. Terrain and traversability

`terrain.py` fuses class priors, ground height, roughness, overhang
clearance, curb-step and pothole flags into a per-cell cost
(`DEFAULT_CLASS_COSTS`, max 180 traversable). Snapshot `query_point`
returns the authoritative semantic/elevation/cost answer; `is_traversable`
additionally enforces the clearance requirement and dynamic exclusion.

## 9. Runtime lifecycle

`FoveaMapRuntime.configure() → start() → process(frame) [per frame] →
get_metrics() → reset() → stop()`. `process` accepts a `LiDARFrame` (or
legacy dict) and returns a detached `MapSnapshot`; `last_timing` exposes
per-stage milliseconds (perception/projection/fusion/…). `process_source`
streams a source; `FoveaMapConfig.from_env()` + profiles select
device/engines without code changes.

## 10. SDK / HTTP / ROS 2 boundaries

- SDK (`sdk/client.py`, `sdk/types.py`): `SnapshotView.query_point(s)`,
  `query_ray`, `export_numpy`, with frozen plain-value result types and
  `SDKQueryError` on misuse.
- HTTP (`sdk/http.py`): loopback-only by default; strict validation, no
  stack-trace leakage; `/health /status /metrics /map/query /map/snapshot`.
- ROS 2 (`foveamap_ros/`): PointCloud2 conversion, TF/world-pose policy,
  lifecycle (`configure/activate/deactivate/shutdown`), bounded queues, QoS
  profiles, diagnostics. Physical ROS 2 execution is Linux-only and
  unverified on Windows hosts.

## 11. Invariants contributors must not break

1. Zero points lost at tier boundaries; exact integer nesting.
2. Ontology ids 0–8 stable (`core/ontology.py` is the single source).
3. Units/frames: meters, seconds, ego X-fwd/Y-left/Z-up, SE(3) poses.
4. Snapshot detachment: consumers must not observe later mutation.
5. NumPy/Torch grid parity (layout + scrolling + fusion).
6. Untrained inference stays development-only and loud.
7. HTTP stays strict/loopback-safe; errors stay actionable.
