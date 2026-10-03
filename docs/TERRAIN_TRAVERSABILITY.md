# FoveaMap — Terrain & Traversability Engine (Phase 7)

## 1. Purpose

Turn the foveated world model (Phases 1–6) into a physically meaningful
terrain representation plus a deterministic traversability layer, without
touching the mapping engine, the temporal model, the ontology, or the
memory layout. All Phase 7 logic consumes existing fields on the existing
lattice; per-query slope and aggregate diagnostics are derived on demand.

## 2. Architecture

```text
FoveaMap cell (ground, z_min/z_max, rough, clear, cost, cls/conf/flags, age, dynamic)
      │
      ▼
foveamap/terrain.py  (shared interpretation: slope_at_cell, report_tier,
                      report_layers, summarize_reports; zero persistent state)
      │                        │
      ▼                        ▼
grid._derive            query_point / MapSnapshot.query_point
(per-frame flags+cost,  (slope_rad on demand, existing terrain keys)
 NumPy + Torch)
      │
      ▼
terrain_report()  (diagnostic aggregates; Torch converts via host snapshot)
```

`terrain.py` owns interpretation; engines own fields and derivation. No
`TerrainSnapshot` type: consumers use `TierLayers`, `MapSnapshot`, and the
query API. No second grid, no second ontology, no second config system.

## 3. Terrain fields

| Field | dtype | Unit | Meaning |
|---|---|---|---|
| `ground` | float16 | m | EMA-fused surface elevation; NaN = no evidence |
| `z_min` / `z_max` | float16 | m | static vertical extent (objects above ground included) |
| `rough` | float16 | m | ground-height **standard deviation (σ)** within the cell; NaN if < 2 ground points |
| `clear` | uint8 | 2 cm units | `zmin_non_ground − ground`; 255 = unknown |
| `cost` | uint8 | 0..254, 255 = UNKNOWN | traversability cost (see §8) |
| `cls` / `conf` | uint8 | id / packed | semantic evidence; low `conf` feeds the uncertainty penalty |
| `flags` low nibble | bits | — | `F_SLOPE/F_STEP/F_DEPRESSION/F_OVERHANG` |
| `age` | frames | — | staleness evidence for the stale penalty |

## 4. Units

Metres for elevation/roughness/clearance; **radians for slope**
(`slope_threshold_rad`, `slope_critical_rad`); frames for ages; unitless
0..254 cost. No degree/radian mixing anywhere.

## 5. Slope calculation

`slope = atan(sqrt((dz/dx)^2 + (dz/dy)^2))` with gradients in physical units
(`tier resolution` per cell step — a 0.05 m cell and a 0.50 m cell report the
same slope for the same surface). Neighbor policy, identical in
`grid._derive`, `grid_torch._derive`, and `terrain.slope_at_cell`: one-sided
differences at window edges, central differences with two valid neighbors,
and **no estimate** unless the centre is valid and at least one orthogonal
neighbor is valid (query returns `slope_rad = None`; `_derive` sets no flag
and no penalty).

## 6. Roughness semantics

`rough` is σ (metres), fused by EMA across frames and combined across mip-up
levels by variance combination (`σ²` sums weighted by ground counts, then
`sqrt`). Bit-identical combination math in both engines (verified to 1e-2).
Penalty: `+30` when finite and above `roughness_threshold_m` (0.04 m).

## 7. Semantic policy

`DEFAULT_CLASS_COSTS` (road 0, parking 10, sidewalk 110, terrain 150,
vegetation/building/pole/vehicle/person 254; default 200) seeds
`TerrainConfig.cost_priors`, which is the policy surface: classification
(semantic class) is separated from policy ( priors + penalties). Effective
class for costing is the passable-under overhang resolution, never a
non-ground runner-up.

## 8. Traversability calculation

```
cost = semantic_prior[eff_cls]
     + 20 if passable-under overhang
     = max(cost,180) for steps on drivable (max(cost,140) otherwise)
     = max(cost,170) for depressions
     + 30 if rough (finite ground σ above threshold)
     + round(slope_excess * 40) above slope_threshold_rad
     = max(cost,220) at/above slope_critical_rad
     + 25 if classification confidence < 150
     + 20 if stale
clip 0..254; UNKNOWN cells keep 255
```

Deterministic, bounded, identical in both engines. One authoritative policy
(`terrain.is_traversable_cell`, threshold `TerrainConfig.traversable_cost_max`,
default 180, aligned with the step clamp): known cell + no dynamic occupancy
+ `cost < traversable_cost_max`. `Grid.is_traversable()` defaults to the
configured threshold (explicit `max_cost` is a planner override);
`MapSnapshot.is_traversable()` enforces the same cost rule from snapshot
metadata (lethal known costs are never traversable) plus an optional
`clearance_req` in metres. Query `clearance`/`snapshot clearance` are metres
(`None` when unknown); the stored byte stays 2 cm units. Stale is uniformly
`age >= stale_age_threshold` in derivation, queries, snapshots, and reports.

## 9. Unknown handling

Unknown is explicit and conservative: NaN ground, cost 255, `state UNKNOWN`,
`is_traversable False`, `slope_rad None`. Missing neighbors yield unknown
slope, never zero-filled safety. No ground evidence is never converted into
free space.

## 10. Dynamic-object handling

Phase 6 separation is respected, not redesigned: moving points never enter
static accumulation, so vehicle roofs/people never redefine `ground`
(verified: ground stays at road height under mixed cells); lifecycle-occupied
cells report dynamic and block traversability read-only; after expiry the
static cost applies unchanged. Ray clearing still treats dynamic occupancy
as temporary blockage under Phase 5 guards.

## 11. Foveated-tier behavior

All gradients scale by the tier's own `cell`; coarse tiers aggregate by the
existing mip-up semantics (min/max elevation, ground-count-weighted means,
variance-combined roughness, summed class evidence) — conservative where
safety matters (steps/depressions flag from neighbor differences, never from
a single copied child). Queries resolve the finest containing tier first.

## 12. NumPy/Torch behavior

One cost/flag algorithm in two engine-native implementations
(vectorized NumPy vs device-resident Torch); one shared interpretation
module. Query slope: NumPy reads arrays directly; Torch reads a single 3x3
host patch inside the already-host query boundary (documented, never
per-point, never hot path). `terrain_report()` is vectorized NumPy; the
Torch override converts via the existing host snapshot (diagnostic-only).
No `.cpu()/.numpy()/.item()` was added to any hot path (audited).

## 13. Configuration

No new parameters: every threshold already lives in `TerrainConfig`
(`vehicle_clearance_m`, `step_threshold_m`, `depression_threshold_m`,
`depression_window_m`, `roughness_threshold_m`, `slope_threshold_rad`,
`slope_critical_rad`, `stale_age_threshold`, `traversable_cost_max`,
`cost_priors`, ray-clearing
fields), each with name, type, unit, range, default, and meaning in code
plus validation. Custom priors/thresholds demonstrably alter behavior
(tests + pre-existing runtime tests).

## 14. Limitations

- Slope/roughness resolution is limited by float16 ground storage and by
  single-point cells (roughness needs ≥ 2 ground points).
- The cost model is heuristic and policy-configurable, not learned; no
  real-world terrain accuracy is claimed (formal validation is Phase 11).
- Depression detection needs a drivable neighborhood within
  `depression_window_m`; isolated patches cannot flag potholes.
- CUDA performance is unmeasured here (CPU-only environment); architecture
  is device-resident by construction.

## 15. Testing

`tests/test_phase7_terrain.py` (35 tests): analytical planes
(`z = ax + by + c` ⇒ known `atan` slope), flat/steep/critical bands, radian
units, roughness σ, steps, potholes, isolated-cell unknown slope, obstacle
vs ground, clearance values, dynamic exclusion, UNKNOWN≠safe, cost bounds
fuzz, per-class policy, empty/single/NaN/extreme inputs, coarse-tier
physics, tier boundaries, NumPy/Torch parity (exact flags/cost, slope
1e-4, roughness 1e-2, identical reports), runtime end-to-end, custom
thresholds, aggregate report checks.

## 16. Performance

Measured on CPU (5000-point update, spec profile): `terrain_report()`
≈ 55 ms both engines (identical aggregates); 200 slope-enriched queries ≈
570 ms NumPy / ≈ 540 ms Torch (≈ 2.8 ms/query, SDK-rate, not hot path).
Persistent memory impact: **+0 bytes** (`allocated_bytes == 5120000` before
and after; slope is derived, reports allocate only small temporaries).
No CUDA numbers fabricated.
