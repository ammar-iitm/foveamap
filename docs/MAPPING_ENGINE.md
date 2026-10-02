# FoveaMap Production Foveated 2.5D Mapping Engine (Phase 5)

## 1. Architectural Overview & Foundations

FoveaMap turns classified 3D LiDAR point sweeps into an adaptive, multi-tier, variable-resolution 2.5D elevation and traversability representation.

The canonical execution flow is:

```text
Raw Classified Points (pw, z, P, moving)
              ↓
Authoritative Single Native Tier Assignment (fine_index // ratio - origin)
              ↓
Native Tier Cell Accumulation (scatter-reduce & unique keys)
              ↓
Integer Lattice Mip-Up Aggregation (fine child cells → coarse parent cells)
              ↓
Temporal Static Integration (EMA blending & stale lifecycle)
              ↓
Dynamic Observation Layer (isolated per-frame rebuild, zero ghost trails)
              ↓
2.5D Terrain & Cost Derivation (slope, step edges, depressions, clearance)
              ↓
Published MapSnapshot Contract (detached snapshot-owned data)
```

Both reference **NumPy** (`foveamap.grid.FoveatedGrid`) and device **PyTorch** (`foveamap.grid_torch.TorchFoveatedGrid`) implementations enforce exact numerical and structural equivalence across CPU and GPU devices.

---

## 2. Integer Lattice & Tier Semantics

FoveaMap defines a nested spatial hierarchy where every coarse cell cleanly subdivides into an exact integer number of fine cells.

### Configuration & Presets
Profiles are authoritatively managed by `foveamap.core.config.GridConfig`:
* **`spec`**: 
  - Tier 0: $\text{cell} = 0.05\,\text{m}$, $\text{half} = 10.0\,\text{m}$ ($400 \times 400$ cells, $20\,\text{m}$ extent)
  - Tier 1: $\text{cell} = 0.50\,\text{m}$, $\text{half} = 100.0\,\text{m}$ ($400 \times 400$ cells, $200\,\text{m}$ extent)
  - Total allocated cells: $320,000$ cells
* **`graded`**:
  - Tier 0: $\text{cell} = 0.05\,\text{m}$, $\text{half} = 10.0\,\text{m}$ ($400 \times 400$ cells)
  - Tier 1: $\text{cell} = 0.10\,\text{m}$, $\text{half} = 25.0\,\text{m}$ ($500 \times 500$ cells)
  - Tier 2: $\text{cell} = 0.50\,\text{m}$, $\text{half} = 100.0\,\text{m}$ ($400 \times 400$ cells)

### Window Snapping & Scrolling
Tier windows track ego motion while remaining snapped to the coarse integer lattice:
$$\text{ego\_coarse} = \lfloor \mathbf{x}_{\text{ego}} / c_{\text{coarse}} \rfloor$$
$$\text{origin}_k = (\text{ego\_coarse} - \text{half\_coarse}_k) \times (c_{\text{coarse}} / c_k)$$
This guarantees:
1. No sub-cell drifting or coordinate jittering during continuous vehicle motion.
2. Fine and coarse cell boundary alignment is mathematically preserved across arbitrary ego trajectories.

---

## 3. Authoritative Single Native Tier Assignment

Points entering the mapping pipeline are assigned to **exactly one native tier**:
1. Points within Tier 0's bounds are assigned exclusively to Tier 0.
2. Points within Tier 1's bounds but outside Tier 0 are assigned to Tier 1.
3. Points outside all tier boundaries are marked as filtered ($-1$).

### Mathematical Invariants
For any incoming point set $\mathcal{P}_{\text{in}}$:
$$\text{points\_in} = \text{native\_points\_assigned} + \text{filtered\_points}$$
* $\forall p \in \mathcal{P}_{\text{assigned}}$, $p$ belongs to exactly one native tier $k \in [0, K-1]$.
* No point is ever accumulated into multiple native tiers.
* Coordinates at exact boundaries ($x = 10.0\,\text{m}$, negative coords $x = -10.0\,\text{m}$) are handled deterministically via half-open intervals $[0, n_k)$.

---

## 4. Integer Lattice Mip-Up Aggregation

Coarser tiers aggregate their representation from fine child cells via deterministic integer coordinate transformation without floating-point roundtrips:

$$\mathbf{i}_{\text{parent}} = \lfloor (\mathbf{i}_{\text{child}} + \mathbf{o}_{\text{child}}) / r_{c\to p} \rfloor - \mathbf{o}_{\text{parent}}$$
$$\mathbf{j}_{\text{parent}} = \lfloor (\mathbf{j}_{\text{child}} + \mathbf{o}_{\text{child}}) / r_{c\to p} \rfloor - \mathbf{o}_{\text{parent}}$$

### Conservation Rules
* **Point Count**: $N_{\text{parent}} = \sum_{c \in \text{children}} n_c$.
* **Elevation Bounds**:
  $$z_{\text{min, parent}} = \min_{c} z_{\text{min}, c}, \quad z_{\text{max, parent}} = \max_{c} z_{\text{max}, c}$$
* **Ground Mean**: Weighted average over ground-contributing children:
  $$\bar{g}_{\text{parent}} = \frac{\sum_{c} n_{g, c} \cdot g_c}{\sum_{c} n_{g, c}}$$
* **Combined Roughness Variance (Two-Pass Numerically Stable)**:
  To eliminate catastrophic cancellation in float32/float16, combined variance is derived via:
  $$\sigma_{\text{parent}}^2 = \frac{\sum_{c} n_{g, c} \cdot \left(\sigma_c^2 + (g_c - \bar{g}_{\text{parent}})^2\right)}{\sum_c n_{g, c}}$$
* **Class Evidence**: Vector sums of point class distributions are preserved identically.

---

## 5. Dynamic Layer Separation

Static and dynamic observations are strictly partitioned:

* **Static Layer**:
  - Persists across sweeps via temporal EMA blending.
  - Represents non-moving obstacles, roads, terrain, and structures.
* **Dynamic Layer**:
  - Rebuilt completely from scratch every single frame from current returns where $\text{moving} = \text{True}$.
  - Never accumulates temporally into the persistent static grid.
  - Zero ghost trails: moving objects leave completely empty/unknown cells behind when they move.
  - Published explicitly as structured dynamic cells:
    $$\{\text{tier}, \text{cell\_coord}, \text{class}, \text{confidence}, \text{point\_count}, z_{\text{min}}, z_{\text{max}}\}$$
  - Deterministic tie-breaking: selects between `PERSON` and `VEHICLE` based on maximum evidence counter.

---

## 6. Stale Lifecycle & Conservative Ray Clearing

Cells in FoveaMap follow an explicit lifecycle:
$$\text{UNKNOWN} \xrightarrow{\text{point return}} \text{OBSERVED} \xrightarrow[\text{age} \ge 20]{\text{unobserved}} \text{STALE} \xrightarrow[\text{age} \ge 100]{\text{timeout}} \text{UNKNOWN}$$

### Conservative 2.5D Ray Clearing
When `enable_ray_clearing` is enabled:
* LiDAR beams from the sensor origin to observed obstacle returns traverse free-space cells before the return.
* Cells traversed by beams increment a conservative `free_passes` counter.
* If a cell has no observation in the current frame and receives $\ge \text{free\_clear\_frames}$ (default 3) **consecutive** frames, its previous static obstacle occupancy is cleared back to `UNKNOWN`. Any frame without traversal resets the streak.
* **Safety Invariants**: cells behind the obstacle return are never cleared; ground classes are never cleared; dynamic-obstacle cells are never cleared; occluded regions are never cleared.
* Both engines enforce identical guards: NumPy (`FoveatedGrid._clear_rays`) and Torch (`TorchFoveatedGrid._clear_rays` device override with explicit host-boundary ray sampling).

---

## 7. 2.5D Terrain & Traversability Derivation

Every cell computes a comprehensive 2.5D traversability cost in $[0, 255]$ incorporating:

1. **Semantic Cost Prior**: Baseline cost per class (`road` = 10, `sidewalk` = 30, `vegetation` = 90, `building` = 254).
2. **2.5D Ground Slope Gradient**:
   Central and forward/backward finite differences from $3 \times 3$ ground-valid neighbors:
   $$\nabla g = \sqrt{\left(\frac{\partial g}{\partial x}\right)^2 + \left(\frac{\partial g}{\partial y}\right)^2}, \quad \theta_{\text{slope}} = \arctan(\nabla g)$$
   - Slopes exceeding $\theta_{\text{thresh}} = 0.25\,\text{rad}$ add a proportional penalty.
   - Slopes exceeding $\theta_{\text{crit}} = 0.40\,\text{rad}$ clamp cost to $\ge 220$ (critical obstacle).
3. **Step Edges**: Height discontinuities $\Delta z > 0.15\,\text{m}$ between adjacent cells clamp cost to non-traversable ($\ge 180$).
4. **Depressions / Potholes**: Deviations below a local window filter clamp cost to $\ge 170$.
5. **Overhang Clearance**: Cells with $z_{\text{min\_ng}} - g \ge 2.2\,\text{m}$ (`vehicle_clearance_m`) are flagged `passable_under` and take ground-class traversability cost.
6. **Confidence & Stale Penalties**: Low-confidence cells ($< 150$) and stale cells ($> 20$ frames) receive additive penalties.

---

## 8. Memory Accounting & Validation

### Measured Persistent State: 16 Bytes / Cell
Each tier allocates a compact Structure-of-Arrays (SoA) layout:

| Field | Data Type | Bytes | Description |
| :--- | :--- | :---: | :--- |
| `count` | `uint16` | 2 | Static point observation count |
| `z_min` | `float16` | 2 | Minimum elevation |
| `z_max` | `float16` | 2 | Maximum elevation |
| `ground` | `float16` | 2 | Estimated ground elevation |
| `rough` | `float16` | 2 | Ground surface roughness ($\sigma$) |
| `cls` | `uint8` | 1 | Dominant semantic class (255=UNKNOWN) |
| `conf` | `uint8` | 1 | Packed confidence: upper nibble primary conf, lower nibble secondary conf (each /15) |
| `flags` | `uint8` | 1 | Low nibble terrain flags (overhang/step/slope/depression); high nibble secondary-evidence class (0xF=none, else 0..8) |
| `clear` | `uint8` | 1 | Overhang clearance in 2 cm units |
| `cost` | `uint8` | 1 | Traversability cost $[0, 255]$ |
| `age` | `uint8` | 1 | Frames since last direct observation |
| **Total** | | **16** | **Bytes per persistent cell** |

### Secondary-evidence semantics (flags high nibble + conf low nibble)

The pair stores the most relevant non-dominant class **with its own matching
confidence** (never mixed). A distinct ground class
(road/sidewalk/parking/terrain) is preferred when confidently present
(overhang/underpass case); otherwise the fused runner-up (top-2) is stored.
`secondary_class` maps the `0xF` sentinel to `UNKNOWN` (255) for parity
between NumPy, Torch, and the query API. Effective class
(`eff_cls`, passable-under) uses the stored class **only when it is a ground
class** and clearance >= `vehicle_clearance_m` (2.5 m); a non-ground
runner-up (pole/vegetation/building) never becomes effective.

### Memory Reduction Benchmark vs. Uniform 5 cm Baseline

| Metric | Spec Profile (2 Tiers) | Uniform 5 cm Baseline | Reduction |
| :--- | :---: | :---: | :---: |
| **Extent** | $\pm 100\,\text{m}$ | $\pm 100\,\text{m}$ | Identical coverage |
| **Cell Dimension** | $400 \times 400 \times 2$ | $4000 \times 4000$ | — |
| **Allocated Cells** | **320,000** | 16,000,000 | **50.0× fewer cells** |
| **Persistent Bytes** | **5.12 MB** (5,120,000 B) | 256.0 MB | **50.0× reduction** |
| **Target Budget** | **$\le 8.0\,\text{MB}$ (PASSED)** | — | **PRD Met** |

---

## 9. Hardware & Parity Verification

### NumPy / PyTorch Parity
* Exact integer parity on `count`, `cls`, `flags`, `cost`, `age`, and `dynamic` masks.
* Absolute error $\le 10^{-3}$ on `z_min`, `z_max`, `ground`, and `rough` due to float16/float32 precision.

### CUDA Verification Status
* **Explicit Status**: `CUDA NOT VERIFIED — NO CUDA DEVICE`
* Physical test execution ran on host Intel/AMD CPU. PyTorch device tensors and conditional CUDA paths remain preserved for execution on NVIDIA T4/Jetson systems.
