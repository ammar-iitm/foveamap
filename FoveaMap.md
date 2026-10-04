# FoveaMap

## Adaptive Variable-Resolution 2.5D Semantic LiDAR Mapping

**Project:** FoveaMap\
**Repository:** `ammar-iitm/foveamap`\
**Development branch:** `dev`\
**Current stage:** Phase 11 complete → Pre-Deployment Engineering Gate
passed → Phase 12 deployment not yet started\
**Document purpose:** Product, problem, architecture, engineering,
validation, and roadmap reference

------------------------------------------------------------------------

# 1. Executive Summary

FoveaMap is an **adaptive variable-resolution 2.5D semantic LiDAR
mapping system** designed for autonomous and intelligent ground-vehicle
perception.

The central idea is simple:

> **Keep high spatial resolution where perception and navigation
> decisions are most sensitive, and progressively reduce resolution
> where the LiDAR itself provides less useful spatial detail.**

Instead of representing a large environment with a uniformly fine grid,
FoveaMap creates a **foveated map** inspired by biological vision:

-   approximately **5 cm resolution in the near field**
-   progressively coarser tiers farther away
-   approximately **50 cm resolution at 100 m**
-   shared integer-aligned geometry between tiers
-   semantic information
-   elevation and terrain information
-   static/dynamic separation
-   temporal persistence and clearing
-   traversability reasoning
-   query/export interfaces
-   ROS2 and SDK boundaries
-   dashboard and benchmark infrastructure

The original product requirement targets a roughly **50× reduction in
map memory** compared with a uniform 5 cm grid over the same coverage
area, while preserving safety-critical near-field detail.

The current implementation has progressed from a prototype-style
pipeline into a modular system with:

``` text
LiDAR Source
     │
     ▼
LiDARFrame Contract
     │
     ▼
Data Preprocessing
     │
     ▼
Perception
     │
     ▼
Temporal / Dynamic World Model
     │
     ▼
Foveated 2.5D Grid
     │
     ├── Terrain
     ├── Traversability
     ├── Static Layer
     └── Dynamic Layer
     │
     ▼
MapSnapshot / Query APIs
     │
     ├── Python SDK
     ├── HTTP Boundary
     ├── ROS2
     ├── Dashboard
     └── Benchmark / Metrics
```

The most recent engineering hardening pass performed a whole-project
adversarial audit, fixed the known correctness defects plus additional
hidden defects, and completed the full automated regression suite with:

-   **435 tests collected**
-   **430 passed**
-   **0 failed**
-   **5 cleanly skipped**
-   CLI end-to-end validation passed
-   engineering sign-off: **PASS**
-   Phase 12 deployment: **not started**

The five skips are environment-related, principally the absence of
physical CUDA and native ROS2 binaries on the Windows development host.
Those environments still require deployment-environment validation.

------------------------------------------------------------------------

# 2. The Problem

## 2.1 Why LiDAR mapping is difficult

Autonomous vehicles need to understand their surroundings continuously.

A LiDAR sensor can provide extremely rich 3D information, including:

-   road surfaces
-   curbs
-   potholes
-   sidewalks
-   buildings
-   poles
-   vegetation
-   vehicles
-   pedestrians
-   overhangs
-   elevation changes

A modern LiDAR sweep can contain tens or hundreds of thousands of
points.

The problem is not simply collecting the points. The difficult part is
turning them into a representation that is:

1.  spatially useful,
2.  semantically meaningful,
3.  temporally stable,
4.  computationally affordable,
5.  memory efficient,
6.  queryable by downstream systems,
7.  suitable for real-time operation.

## 2.2 Problem with a uniform high-resolution grid

Suppose a system represents a 200 m × 200 m region at 5 cm resolution.

That requires:

``` text
4000 × 4000 = 16,000,000 cells
```

A uniformly fine grid therefore allocates high resolution even in
regions where:

-   LiDAR returns are sparse,
-   exact centimetre-scale geometry is unnecessary,
-   the vehicle will not immediately make a decision,
-   many cells remain empty or noisy.

The result is unnecessary memory consumption and computational work.

## 2.3 Problem with a coarse grid

The opposite strategy is to use a large cell size everywhere.

That saves memory but loses information.

A coarse grid can blur or hide:

-   a 10--12 cm curb,
-   a pothole,
-   a small step,
-   a low obstacle,
-   a narrow free-space boundary,
-   an overhang,
-   a pedestrian-sized obstacle.

For autonomous navigation, the near field is often exactly where high
precision matters most.

## 2.4 Problem with ordinary 2D occupancy maps

A purely 2D occupancy representation loses vertical information.

For example, these situations can look similar in a 2D occupancy map:

``` text
flat road
raised curb
pothole
low-hanging branch
bridge/overhang
```

A 2.5D representation preserves useful elevation information while
remaining substantially cheaper than a full dense 3D voxel
representation.

## 2.5 Problem with dynamic environments

Vehicles and pedestrians move.

If their observations are simply accumulated into a persistent map, the
map can develop **ghost obstacles**:

``` text
Frame 1: vehicle at A
Frame 2: vehicle moves to B
Frame 3: vehicle is at C

Naive accumulation:
A + B + C all remain occupied
```

FoveaMap therefore separates persistent static information from
transient dynamic information and provides temporal lifecycle logic for
clearing and stale state.

## 2.6 Problem with perception semantics

Geometry alone does not answer:

> Can the vehicle drive here?

The map needs to distinguish semantic categories and terrain properties.

FoveaMap's canonical semantic ontology is:

    ID Class
  ---- ------------
     0 ROAD
     1 SIDEWALK
     2 PARKING
     3 TERRAIN
     4 VEGETATION
     5 BUILDING
     6 POLE
     7 VEHICLE
     8 PERSON

Semantic class, elevation, slope, roughness, clearance, dynamic state,
and confidence can then participate in traversability reasoning.

------------------------------------------------------------------------

# 3. FoveaMap Solution

FoveaMap addresses the above problems by combining several ideas into
one system.

## 3.1 Foveated spatial representation

The map is divided into concentric resolution tiers.

Conceptually:

``` text
                FAR FIELD
        ┌───────────────────────┐
        │      coarse tier      │
        │   ~50 cm resolution   │
        │                       │
        │    ┌─────────────┐    │
        │    │ medium tier │    │
        │    │             │    │
        │    │ ┌─────────┐ │    │
        │    │ │  FOVEA  │ │    │
        │    │ │  5 cm   │ │    │
        │    │ └─────────┘ │    │
        │    └─────────────┘    │
        └───────────────────────┘
```

The exact configuration is controlled through centralized grid/tier
configuration rather than scattered constants.

## 3.2 Exact tier nesting

The tiers use an integer-aligned lattice.

A key design requirement is that the coarser grid is not an
independently positioned approximation of the fine grid.

Instead:

``` text
fine grid
    │
    │ exact integer relationship
    ▼
coarse grid
```

This prevents seams, drift, and ambiguous boundary ownership.

The implementation uses native-tier assignment and integer-lattice
aggregation/mip-up.

## 3.3 Native-tier point assignment

Each point is assigned to its appropriate native tier based on distance.

The system does not simply duplicate every point into every tier.

Instead:

``` text
point
  │
  ├── near → fine native tier
  │
  ├── middle → intermediate tier
  │
  └── far → coarse native tier
```

Coarser representations are derived from lower-resolution aggregation
where appropriate.

This provides conservation and avoids systematic double counting.

## 3.4 2.5D instead of dense 3D

Each grid cell stores a compact representation of observed vertical
structure rather than a full voxel volume.

This provides:

-   position in the horizontal plane,
-   elevation statistics,
-   semantic state,
-   terrain information,
-   dynamic state,
-   confidence,
-   temporal information,

without allocating a full 3D voxel volume.

## 3.5 Static and dynamic world representation

The map is logically separated into:

``` text
             World Model
                 │
       ┌─────────┴─────────┐
       │                   │
   Static Layer        Dynamic Layer
       │                   │
persistent terrain     transient agents
and environment        vehicles/persons
```

Static information is accumulated over time.

Dynamic information has bounded temporal behavior and can be
rebuilt/cleared as objects move.

This is essential for preventing ghost obstacles.

## 3.6 Terrain and traversability

FoveaMap does not treat semantic class as the only source of
navigability.

Traversability can combine:

-   semantic class,
-   slope,
-   roughness,
-   elevation/step characteristics,
-   clearance,
-   dynamic occupancy,
-   terrain policy.

The resulting map can answer questions such as:

``` text
Is this cell traversable?
What is the elevation?
What semantic class is here?
Is there an obstacle?
Is there sufficient overhead clearance?
```

## 3.7 Temporal fusion

The map is updated continuously rather than rebuilt from scratch for
every conceptual world state.

Temporal mechanisms provide:

-   confidence accumulation,
-   static persistence,
-   dynamic clearing,
-   stale state,
-   unknown state,
-   bounded lifecycle,
-   reset/recovery behavior.

The temporal state is explicitly bounded so history does not grow
without limit.

------------------------------------------------------------------------

# 4. Product Requirements

The original product requirements define five major goals.

## G1 --- Terrain analysis

Represent terrain and determine whether observed regions are:

-   drivable,
-   non-drivable ground,
-   unknown.

Also preserve useful elevation and roughness information.

## G2 --- Object understanding

Segment the scene into semantic categories and distinguish:

-   static obstacles,
-   dynamic agents,
-   moving objects.

## G3 --- Foveated representation

Provide:

-   approximately 5 cm near-field resolution,
-   approximately 50 cm far-field resolution,
-   approximately 100 m mapping range,
-   exact tier alignment,
-   no intentional point loss,
-   no systematic double counting.

## G4 --- Real-time operation

The product target is operation at sensor rate, with an original target
of approximately:

``` text
20 FPS
p95 latency ≤ 50 ms
```

These are product targets, not claims that every current environment has
already physically demonstrated them.

## G5 --- Measured proof

The benchmark system is intended to report:

-   end-to-end latency,
-   stage latency,
-   memory,
-   map compression,
-   accuracy by distance,
-   comparison against uniform baselines.

------------------------------------------------------------------------

# 5. Non-Goals

The current v1 scope intentionally does not attempt to become a complete
autonomous-driving stack.

FoveaMap does **not** itself perform:

-   path planning,
-   vehicle control,
-   actuation,
-   global SLAM,
-   loop closure,
-   persistent object-ID tracking and trajectory prediction,
-   camera/radar fusion in the v1 LiDAR-only architecture,
-   automotive safety certification such as ISO 26262.

The intended boundary is:

``` text
Sensors / perception
        ↓
    FoveaMap
        ↓
planner / downstream consumer
```

------------------------------------------------------------------------

# 6. System Architecture

## 6.1 High-level architecture

``` text
┌───────────────────────────────────────────────────────────────────┐
│                         SENSOR / DATA SOURCES                     │
│                                                                   │
│ KITTI .bin | nuScenes | PCD | NPY/NPZ | Simulator | ROS2         │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                             │
│                                                                   │
│ LiDARSource adapters                                               │
│ LiDARFrame validation                                              │
│ finite/range/intensity handling                                    │
│ pose + sensor metadata                                             │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                           RUNTIME                                  │
│                                                                   │
│ FoveaMapRuntime                                                    │
│ DeviceContext / device policy                                     │
│ lifecycle / reset / metrics                                       │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                          PERCEPTION                                │
│                                                                   │
│ PerceptionBackend interface                                        │
│   ├── RangeUNet                                                    │
│   ├── classical / heuristic paths                                 │
│   └── other pluggable backends                                     │
│                                                                   │
│ semantic classes + confidence + motion information                │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                     TEMPORAL WORLD MODEL                           │
│                                                                   │
│ static accumulation | dynamic state | clearing | stale lifecycle │
│ bounded temporal state                                             │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│                    FOVEATED 2.5D GRID                              │
│                                                                   │
│ native-tier assignment                                             │
│ integer lattice                                                     │
│ exact tier nesting                                                 │
│ mip-up / aggregation                                               │
│ compact cell state                                                 │
└──────────────────────┬──────────────────────────┬─────────────────┘
                       │                          │
                       ▼                          ▼
              ┌────────────────┐        ┌────────────────────┐
              │ Terrain /       │        │ Semantic / Dynamic │
              │ Traversability  │        │ Map Layers         │
              └────────┬───────┘        └──────────┬─────────┘
                       │                           │
                       └────────────┬──────────────┘
                                    ▼
┌───────────────────────────────────────────────────────────────────┐
│                       MAP SNAPSHOT / QUERY                         │
│                                                                   │
│ point query | batch query | ray query | traversability | export  │
└───────────────┬──────────────────┬────────────────┬───────────────┘
                │                  │                │
                ▼                  ▼                ▼
             Python SDK          HTTP             ROS2
                │
                └──────────────► Dashboard / Applications
```

------------------------------------------------------------------------

# 7. Core Contracts

A major architectural improvement over the earlier prototype is the use
of explicit typed contracts.

## 7.1 LiDARFrame

`LiDARFrame` is the canonical representation of an incoming sweep.

It carries validated information such as:

-   point coordinates,
-   intensity,
-   pose,
-   sensor origin,
-   source ID,
-   annotations,
-   previous sweeps where available.

Validation rejects malformed data rather than allowing invalid state to
propagate through the system.

The hardening pass also preserved mapping-style compatibility for legacy
consumers where required.

## 7.2 PerceptionResult

`PerceptionResult` carries the output of a perception backend.

It validates:

-   shape,
-   semantic class bounds,
-   probability/confidence bounds,
-   point correspondence/index information.

## 7.3 MapSnapshot

`MapSnapshot` is the externally consumable representation of the current
mapped state.

It provides a stable boundary between internal mutable runtime state and
downstream consumers.

## 7.4 Configuration

Configuration is centralized into typed structures covering areas such
as:

``` text
TierConfig
GridConfig
SensorConfig
PerceptionConfig
TerrainConfig
RuntimeConfig
FoveaMapConfig
```

This prevents important system behavior from being hidden in unrelated
implementation files.

------------------------------------------------------------------------

# 8. Device and Runtime Architecture

FoveaMap separates execution policy from the mapping logic.

The runtime contains:

-   device resolution,
-   lifecycle management,
-   perception execution,
-   mapping integration,
-   reset behavior,
-   metrics collection.

The design supports CPU and Torch-based execution and is intended to
support CUDA execution when a compatible accelerator is available.

A critical design principle is:

> The core mapping policy should not have to know whether perception is
> executing on CPU or GPU.

The hardening work also introduced canonical device comparison so
logically equivalent CUDA devices such as `cuda` and `cuda:0` are
handled consistently.

------------------------------------------------------------------------

# 9. Perception Architecture

Perception is represented through a backend abstraction.

Conceptually:

``` text
             PerceptionBackend
                    │
        ┌───────────┴───────────┐
        │                       │
    RangeUNet              Classical /
                           Heuristic
```

The architecture permits future model replacement without rewriting the
map engine.

## Checkpoint safety

A critical product-safety rule is:

> Production perception must not silently use an untrained random
> checkpoint.

The legacy pipeline was hardened so missing checkpoints require explicit
opt-in for untrained development/testing behavior.

This prevents a system from appearing operational while producing
meaningless learned predictions.

------------------------------------------------------------------------

# 10. Semantic Representation

The canonical ontology is centralized rather than duplicated across
subsystems.

``` text
0 ROAD
1 SIDEWALK
2 PARKING
3 TERRAIN
4 VEGETATION
5 BUILDING
6 POLE
7 VEHICLE
8 PERSON
```

Semantic information influences:

-   map cell class,
-   dynamic classification,
-   traversability,
-   terrain policy,
-   visualization,
-   queries,
-   benchmark evaluation.

The architecture deliberately avoids allowing different subsystems to
silently invent incompatible class IDs.

------------------------------------------------------------------------

# 11. Foveated Grid Architecture

The grid is one of the most important technical components of FoveaMap.

## 11.1 Persistent cell representation

The mapping implementation uses a compact fixed-size cell
representation.

The validated design uses:

``` text
16 bytes / persistent cell
```

The exact field packing is an implementation constraint and must not be
casually changed because the memory budget depends on it.

## 11.2 Confidence packing

Confidence values use packed 4-bit nibbles.

A dedicated confidence module centralizes:

-   packing,
-   unpacking,
-   primary confidence access,
-   secondary confidence access.

This avoids interpreting a packed byte incorrectly as a raw 0--255
confidence value.

## 11.3 Memory target

The reference foveated grid uses approximately:

``` text
320,000 cells
× 16 bytes/cell
≈ 5.12 MB
≈ 4.88 MiB
```

The corresponding uniform 5 cm reference is approximately:

``` text
16,000,000 cells
× 16 bytes/cell
≈ 256 MB
```

Therefore:

``` text
256 MB / 5.12 MB ≈ 50×
```

This is the origin of the project's 50× memory-reduction target.

------------------------------------------------------------------------

# 12. Ground Persistence and Overhang Handling

One subtle mapping problem is that the highest observed point is not
always the ground.

For example:

``` text
       bridge / branch
          █████
             ↓
        LiDAR returns
             ↓
──────────── ROAD ────────────
```

If a cell simply replaces its ground state with the elevated obstacle, a
temporary overhang can incorrectly make a drivable road appear blocked.

FoveaMap therefore maintains ground/elevation semantics carefully across
observations and temporal updates.

The hardening pass fixed a confirmed ground-persistence issue and also
corrected a related lockout behavior in both NumPy and Torch mapping
paths.

------------------------------------------------------------------------

# 13. Clearance Semantics

Clearance has an important distinction:

``` text
clearance = finite value
```

means there is a measured/known overhead limitation.

Whereas:

``` text
clearance = None
```

means no finite overhead restriction is currently known.

Therefore `None` must not automatically be interpreted as:

``` text
insufficient clearance
```

The corrected semantics are:

-   finite clearance → compare against the requested requirement;
-   no finite restriction → treat as open/unlimited unless another rule
    blocks traversal.

This matters for roads with open sky versus roads under bridges or other
overhead structures.

------------------------------------------------------------------------

# 14. Temporal Dynamic World Model

The dynamic subsystem prevents the persistent map from becoming a
history of every object ever seen.

Conceptually:

``` text
Frame t
  ↓
observe dynamic object
  ↓
dynamic state

Frame t+1
  ↓
object moved
  ↓
new dynamic location

old location
  ↓
ray/temporal clearing
  ↓
free / stale / unknown
```

The system uses bounded temporal state and deterministic lifecycle
transitions.

The design intentionally avoids unbounded history.

------------------------------------------------------------------------

# 15. Ray Clearing and Free Space

A major challenge in LiDAR mapping is distinguishing:

``` text
not observed
```

from:

``` text
observed free
```

FoveaMap uses ray-based clearing and temporal evidence to update free
space.

The implementation includes consecutive-frame behavior so a single noisy
observation does not necessarily erase persistent information
immediately.

This supports more stable dynamic-world behavior.

------------------------------------------------------------------------

# 16. Terrain and Traversability

Terrain analysis combines geometric and semantic information.

Relevant features include:

-   ground/elevation,
-   slope,
-   roughness,
-   step/depression characteristics,
-   overhead clearance,
-   semantic class,
-   dynamic state.

A conceptual traversability policy is:

``` text
semantic cost
      +
terrain geometry
      +
dynamic state
      +
clearance
      ↓
traversability decision
```

The system therefore does not reduce navigation suitability to a single
semantic label.

------------------------------------------------------------------------

# 17. Data Ingestion

The architecture provides a common sensor/data abstraction around
different input sources.

The project has support or adapters for sources including:

-   KITTI `.bin`
-   PCD
-   NPY/NPZ
-   nuScenes
-   simulator-generated data
-   ROS2 PointCloud2

The canonical output is a validated `LiDARFrame`.

The system also supports intensity normalization and malformed-input
rejection.

A production design goal is that downstream mapping code should not need
to know whether the frame originated from KITTI, nuScenes, a simulator,
a file, or ROS2.

------------------------------------------------------------------------

# 18. ROS2 Integration

ROS2 is treated as an integration boundary rather than as a hard
dependency of the core engine.

The ROS2 architecture includes concepts for:

-   PointCloud2 input,
-   TF handling,
-   QoS,
-   lifecycle,
-   bounded queues,
-   diagnostics,
-   ROS message definitions.

This keeps the core FoveaMap engine independently testable.

### Important validation status

The Windows development host does not have native ROS2 `rclpy` binaries,
so some ROS2 tests are cleanly skipped in the host environment.

Therefore:

> ROS2 integration should receive final physical deployment validation
> in the actual ROS2/container environment during Phase 12.

The architecture is implemented and tested through available mechanisms,
but absence of native ROS2 on the development laptop means that
host-level execution is not equivalent to a full target-robot ROS2
deployment test.

------------------------------------------------------------------------

# 19. SDK Architecture

The SDK follows a deliberately thin architecture:

``` text
Python Application
       │
       ▼
 FoveaMap SDK
       │
       ▼
 FoveaMapRuntime
       │
       ▼
      Core
```

The SDK does not reimplement mapping or traversability policy.

It exposes the existing engine through stable operations such as:

-   initialization,
-   processing,
-   runtime status,
-   map queries,
-   snapshots,
-   ray queries,
-   export,
-   shutdown.

This avoids semantic divergence between the internal engine and external
applications.

------------------------------------------------------------------------

# 20. HTTP Boundary

The HTTP layer is designed as a thin external boundary around the
runtime.

The goal is to provide structured access without copying the complete
mapping implementation into a web service.

Important product concerns include:

-   bounded payload sizes,
-   bounded point counts,
-   explicit lifecycle,
-   controlled binding,
-   structured errors,
-   reuse of the same core runtime.

The HTTP layer should remain an adapter, not become a second map engine.

------------------------------------------------------------------------

# 21. Dashboard and Visualization

The product includes a dashboard/replay visualization layer.

The visual design requirements emphasize:

-   map as the primary visual element,
-   visible fovea/tier boundary,
-   semantic visualization,
-   live KPI/metrics information,
-   comparison against baseline/target values,
-   visualization decoupled from the critical processing path.

The dashboard is intended for:

-   demonstrations,
-   debugging,
-   replay,
-   engineering validation,
-   product communication.

It is not allowed to become a performance bottleneck in the mapping
pipeline.

------------------------------------------------------------------------

# 22. Benchmark Architecture

Benchmarking is treated as a first-class engineering subsystem.

Metrics include:

-   end-to-end latency,
-   per-stage latency,
-   frame rate,
-   memory,
-   map cell count,
-   foveated versus uniform representation,
-   accuracy where ground truth is available.

The benchmark architecture is designed to make product claims measurable
instead of relying only on architectural estimates.

Reference baselines include:

-   uniform 5 cm 2.5D,
-   uniform 5 cm voxel-style representation,
-   uniform 2D/coarse representations where appropriate.

------------------------------------------------------------------------

# 23. Development Phases

The project was developed as a sequence of verifiable engineering
phases.

## Phase 0 --- Problem and Repository Audit

Established:

-   problem definition,
-   target architecture,
-   baseline behavior,
-   requirements,
-   development roadmap.

## Phase 1 --- Core Foundation

Established:

-   typed contracts,
-   ontology,
-   configuration,
-   exceptions,
-   package structure.

## Phase 2 --- Runtime

Established:

-   device abstraction,
-   runtime lifecycle,
-   perception integration,
-   device-aware execution,
-   reset behavior.

## Phase 3 --- Data

Established:

-   canonical LiDAR frame,
-   dataset adapters,
-   PCD/BIN/NPY-style ingestion,
-   preprocessing,
-   validation.

## Phase 4 --- Perception

Established:

-   perception backend abstraction,
-   RangeUNet integration,
-   confidence,
-   checkpoint policy,
-   temporal/perception hardening.

## Phase 5 --- Foveated Mapping

Established:

-   variable-resolution grid,
-   exact tier relationship,
-   compact cells,
-   static/dynamic information,
-   queries,
-   memory accounting.

## Phase 6 --- Dynamic World

Established:

-   temporal state,
-   dynamic/static separation,
-   lifecycle,
-   stale state,
-   clearing,
-   reset.

## Phase 7 --- Terrain and Traversability

Established:

-   terrain features,
-   slope,
-   roughness,
-   clearance,
-   semantic costs,
-   traversability.

## Phase 8 --- ROS2

Established:

-   ROS2 adapters,
-   PointCloud2 integration,
-   TF/QoS/lifecycle concepts,
-   bounded queues,
-   diagnostics.

## Phase 9 --- SDK/API

Established:

-   public Python SDK,
-   HTTP boundary,
-   runtime queries,
-   snapshots,
-   export interfaces.

## Phase 9.5 / 9.6 --- Closure and Cross-Phase Hardening

Focused on:

-   consistency,
-   interface correctness,
-   requirements closure,
-   pre-hardware readiness,
-   regression analysis.

## Phase 10 --- Hardware/CUDA Validation

The project has historical CUDA/T4 validation artifacts, including:

-   CUDA execution,
-   FP32/FP16 comparison,
-   long-running soak testing,
-   GPU memory observations,
-   performance profiling.

However, the current development host has no physical NVIDIA GPU, so
native CUDA must still be validated in the target deployment
environment.

## Phase 10.5 --- GPU Performance Hardening

Focused on performance bottlenecks and GPU-oriented optimization
targets.

## Phase 11 --- Productization and Verification

Phase 11 established:

-   deployment profiles,
-   CLI,
-   product metrics,
-   baseline comparisons,
-   Docker support,
-   documentation,
-   product tests,
-   deployment-oriented configuration.

## Pre-Deployment Hardening Gate

The latest pass went beyond simply accepting the Phase 11 report.

The codebase was inspected and subjected to:

-   known-bug remediation,
-   adversarial testing,
-   cross-implementation checks,
-   malformed-input tests,
-   boundary tests,
-   lifecycle tests,
-   numerical parity tests,
-   regression testing,
-   full-suite execution,
-   CLI validation.

The result was:

**ENGINEERING SIGN-OFF: PASS**

## Phase 12 --- Deployment

Phase 12 has **not started yet**.

Its purpose is deployment/release hardening and target-environment
validation.

------------------------------------------------------------------------

# 24. Bugs Found and Fixed During Final Hardening

The final hardening pass is important because it demonstrated why
reports alone should not be trusted without source and execution
verification.

## Bug 1 --- CUDA device mismatch

### Problem

PyTorch can distinguish:

``` text
cuda
cuda:0
```

in device equality even when they refer to the same current physical
accelerator.

### Risk

Valid tensors could be incorrectly rejected as being on the wrong
device.

### Fix

Introduced canonical device normalization and logical device matching.

------------------------------------------------------------------------

## Bug 2 --- Packed confidence interpretation

### Problem

Confidence values were packed into a byte using two 4-bit nibbles.

Some consumers could incorrectly treat the entire byte as one confidence
value.

### Risk

Incorrect confidence-dependent behavior in:

-   semantic blending,
-   low-confidence handling,
-   cost decisions,
-   SDK consumers.

### Fix

Centralized packing/unpacking and named confidence accessors.

------------------------------------------------------------------------

## Bug 3 --- Ground persistence under temporary occlusion

### Problem

An elevated object could incorrectly replace the ground interpretation
of a cell.

### Risk

A drivable road under an overhang could become incorrectly classified as
blocked.

### Fix

Corrected ground persistence and associated mapping lockout behavior in
both NumPy and Torch implementations.

------------------------------------------------------------------------

## Bug 4 --- Clearance semantics

### Problem

`clearance=None` was incorrectly treated as a failure when a finite
clearance requirement was queried.

### Risk

Open-sky roads could be incorrectly marked non-traversable.

### Fix

`None` now represents absence of a finite overhead restriction; finite
values are compared against the requested requirement.

------------------------------------------------------------------------

## Bug 5 --- Missing checkpoint safety

### Problem

A legacy pipeline path could instantiate an untrained model when no
checkpoint was supplied.

### Risk

A user could accidentally run a system that looked operational but was
not using a trained model.

### Fix

Missing checkpoint now requires explicit `allow_untrained=True` behavior
for development/testing.

------------------------------------------------------------------------

# 25. Additional Hidden Defects Found

The adversarial audit found additional defects beyond the five known
issues.

## Hidden defect A --- Torch secondary-class sentinel

A secondary-class sentinel value of `15` could be used as an index into
a 9-class ground mask.

This was fixed by safe sentinel handling.

## Hidden defect B --- Empty benchmark

An empty benchmark result could cause:

``` text
per_frame[0]
```

to raise an `IndexError`.

The benchmark now handles empty results safely.

## Hidden defect C --- LiDARFrame legacy mapping compatibility

Some older code expected dictionary-like access to `LiDARFrame`.

Mapping protocol support was added where required.

## Hidden defect D --- Torch boundary precision

A floating-point boundary case produced NumPy/Torch differences near
exact grid boundaries.

The Torch query path was corrected to use appropriate float64 arithmetic
for the relevant boundary calculation.

------------------------------------------------------------------------

# 26. Final Validation

The latest complete regression run reported:

``` text
Total tests:   435
Passed:        430
Failed:        0
Skipped:       5
```

The five skips were clean environment-related skips rather than test
failures.

## CLI validation

The following were successfully executed:

``` text
foveamap info
foveamap demo --frames 3
```

The demo processed:

``` text
3 frames
62,500 points/frame
```

A representative map query returned:

``` text
Tier:         Tier 0
Resolution:   5 cm
Semantic:     road (0)
Elevation:    0.00 m
Traversable:  True
```

The measured reference grid reported approximately:

``` text
Foveated memory: 4.88 MB
Memory target:   ≤ 8 MB
Reduction:       50.0×
```

These values support the structural memory-efficiency invariant.

------------------------------------------------------------------------

# 27. Environment Truth

The latest engineering validation was performed primarily on:

``` text
OS:       Windows 11 x86_64
Python:   3.13.5
PyTorch:  2.7.0+cpu
```

The development machine does not have a physical NVIDIA GPU.

Therefore:

### Directly validated

-   Python/core system
-   NumPy implementation
-   CPU Torch implementation
-   contracts
-   runtime behavior available on host
-   mapping
-   temporal logic
-   terrain
-   SDK
-   HTTP
-   CLI
-   adversarial tests
-   regression suite

### Requires target-environment validation

-   physical CUDA execution on the deployment GPU
-   target GPU performance
-   native ROS2 runtime behavior
-   actual deployment-network behavior
-   final production container/runtime environment

Historical project work also contains Tesla T4 CUDA validation
artifacts, including FP32/FP16 agreement and a 1,000-frame soak test.
Those artifacts are valuable evidence, but they should not be confused
with current physical execution on the Windows development machine.

------------------------------------------------------------------------

# 28. Engineering Design Principles

FoveaMap follows several principles that should remain invariant as the
project evolves.

## 28.1 One source of truth

Ontology, configuration, traversability policy, and dynamic semantics
should be centralized.

Avoid duplicate definitions.

## 28.2 Thin adapters

ROS2, HTTP, SDK, and dashboard layers should call the core engine.

They should not implement independent mapping logic.

## 28.3 Explicit contracts

Bad data should fail at boundaries.

Do not allow malformed state to silently propagate.

## 28.4 Deterministic geometry

Tier geometry must remain integer-aligned and deterministic.

Avoid ad-hoc resampling between tiers.

## 28.5 Device-aware execution

Do not introduce unnecessary:

``` text
GPU → CPU → GPU
```

transfers into the critical path.

## 28.6 Bounded state

Temporal buffers, queues, and track stores must have explicit bounds.

## 28.7 Metrics-first engineering

Important product claims must have measured evidence.

## 28.8 Safety over convenience

A system must not silently:

-   use an untrained model,
-   accept malformed point data,
-   interpret unknown clearance as blocked,
-   confuse dynamic objects with persistent terrain,
-   produce incompatible semantic IDs.

------------------------------------------------------------------------

# 29. Current Repository Architecture

A representative organization is:

``` text
foveamap/
├── foveamap/
│   ├── core/
│   │   ├── contracts.py
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   └── confidence.py
│   │
│   ├── runtime/
│   │   ├── device.py
│   │   ├── perception.py
│   │   └── runtime.py
│   │
│   ├── data/
│   │   ├── preprocessing
│   │   ├── dataset adapters
│   │   └── source abstractions
│   │
│   ├── perception/
│   │   └── model/backend implementations
│   │
│   ├── mapping/
│   │   ├── grid.py
│   │   ├── grid_torch.py
│   │   └── temporal/dynamic state
│   │
│   ├── terrain/
│   │   └── terrain/traversability logic
│   │
│   ├── sdk/
│   │   └── public API / HTTP
│   │
│   └── benchmarks/
│
├── foveamap_ros/
│   ├── PointCloud2 integration
│   ├── TF
│   ├── QoS
│   ├── lifecycle
│   └── diagnostics
│
├── tests/
├── benchmarks/
├── dashboard/
├── docs/
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

The exact tree may evolve; this section documents the architectural
separation rather than requiring every filename to remain unchanged.

------------------------------------------------------------------------

# 30. What Makes FoveaMap Different

The key contribution is not simply:

> "Use a smaller grid."

The system combines several constraints simultaneously:

``` text
variable resolution
        +
exact tier nesting
        +
semantic perception
        +
2.5D elevation
        +
static/dynamic separation
        +
temporal fusion
        +
terrain reasoning
        +
compact representation
        +
runtime/SDK/ROS2 boundaries
        +
benchmark-driven validation
```

A conventional occupancy grid can be memory efficient but lose vertical
and semantic information.

A dense point cloud preserves information but is expensive to query and
maintain.

A dense voxel map preserves 3D structure but can become extremely
expensive at useful spatial resolutions.

FoveaMap is designed as a middle representation:

> **retain the information required for perception and navigation while
> allocating precision according to spatial importance.**

------------------------------------------------------------------------

# 31. Reference Data Flow

A complete frame conceptually follows:

``` text
Raw LiDAR sweep
      │
      ▼
Validate / normalize
      │
      ▼
LiDARFrame
      │
      ▼
Perception backend
      │
      ├── semantic class
      ├── confidence
      └── motion/dynamic information
      │
      ▼
Temporal world model
      │
      ├── static accumulation
      ├── dynamic state
      └── clearing / stale state
      │
      ▼
Native foveated tier assignment
      │
      ▼
2.5D cell update
      │
      ├── elevation
      ├── semantic state
      ├── confidence
      ├── dynamic state
      └── temporal metadata
      │
      ▼
Terrain / traversability
      │
      ▼
MapSnapshot
      │
      ├── query
      ├── ray query
      ├── export
      ├── SDK
      ├── HTTP
      └── ROS2/dashboard
```

------------------------------------------------------------------------

# 32. Performance and Memory Targets

The main product-level targets are:

  Metric                            Target
  ----------------------------- ----------
  Near-field resolution             \~5 cm
  Far-field resolution             \~50 cm
  Mapping range                    \~100 m
  Persistent grid target            ≤ 8 MB
  Reference reduction                ≥ 30×
  Stretch/reference reduction        \~50×
  Target sustained FPS            ≥ 20 FPS
  Target p95 latency               ≤ 50 ms
  Hard latency limit              ≤ 100 ms
  Hard FPS limit                  ≥ 10 FPS

These targets should always be reported separately from measured values.

A benchmark result is not allowed to silently become a product
guarantee.

------------------------------------------------------------------------

# 33. Accuracy and Validation Philosophy

The system is intended to evaluate perception and map quality by
distance bands.

Important evaluation dimensions include:

-   semantic accuracy,
-   drivable-region accuracy,
-   moving-object accuracy,
-   terrain accuracy,
-   map conservation,
-   NumPy/Torch parity,
-   memory,
-   latency.

The architecture also recognizes that perception quality and mapping
quality are different measurements.

For example:

``` text
good semantic segmentation
≠
good temporal map

good map compression
≠
good semantic accuracy
```

Therefore the benchmark system should preserve separate metrics rather
than collapsing everything into a single score.

------------------------------------------------------------------------

# 34. Known Scope Deviations and Limitations

The project documentation has identified several areas where the
implementation does not perfectly equal every original PRD ambition.

These should remain explicit rather than being hidden.

## Per-point de-skew

The original PRD describes de-skewing when per-point timestamps are
available.

The current implementation does not claim full production-grade
geometric de-skewing for every input source.

## Perception backbone

The PRD discusses sparse 3D perception as a preferred architecture and
range-image approaches as alternatives.

The implemented system uses a pluggable backend design with RangeUNet
and classical/heuristic paths rather than claiming that every proposed
model architecture has been implemented.

## Full class histogram

The compact map representation emphasizes dominant and secondary
semantic evidence rather than retaining an unlimited full per-cell class
histogram.

This is a deliberate representation trade-off and should be documented
whenever comparing implementation against the original FR-9 requirement.

## ROS2 export details

The architecture supports ROS2 integration, but some richer
export/integration paths may remain deployment-specific rather than
being assumed complete merely because an adapter exists.

## Final physical hardware validation

A software test suite running on CPU cannot substitute for target-GPU
validation.

The final deployment gate must therefore include real hardware.

------------------------------------------------------------------------

# 35. Phase 12 Direction

Phase 12 is the next stage, but it should not be confused with the
engineering sign-off that has already occurred.

The current state is:

``` text
Development
    │
    ▼
Phase 11 Productization
    │
    ▼
Pre-Deployment Hardening
    │
    ▼
Engineering Sign-Off: PASS
    │
    ▼
>>> PHASE 12 DEPLOYMENT <<<
```

Phase 12 should concentrate on deployment rather than reopening already
validated core architecture without evidence.

Priority areas include:

1.  target GPU/container deployment,
2.  physical CUDA execution,
3.  target ROS2 environment,
4.  real dataset replay at deployment scale,
5.  final performance measurements,
6.  resource limits,
7.  deployment observability,
8.  reproducible build,
9.  configuration/secrets handling,
10. release documentation,
11. failure/recovery testing,
12. final acceptance criteria.

------------------------------------------------------------------------

# 36. Engineering Status Summary

## Core product

**Status: implemented and hardened**

The core architecture is modular and has been subjected to extensive
regression testing.

## Foveated mapping

**Status: implemented and validated structurally**

The reference grid demonstrates the intended memory reduction and exact
tier geometry.

## Dynamic world

**Status: implemented and hardened**

Static/dynamic separation, temporal lifecycle, clearing, and reset
behavior are part of the architecture.

## Terrain/traversability

**Status: implemented and hardened**

Terrain features and traversability policy are integrated with semantic
and dynamic state.

## SDK/API

**Status: implemented**

The SDK is designed as a thin boundary over the runtime.

## ROS2

**Status: implemented as an integration layer; target-environment
validation remains**

Native ROS2 execution is environment-dependent and must be validated
during deployment.

## CUDA

**Status: CUDA-capable architecture with historical T4 evidence; target
hardware validation remains**

The current Windows development host has no physical NVIDIA GPU.

## Productization

**Status: Phase 11 complete**

CLI, profiles, metrics, Docker, tests, and documentation were added.

## Engineering hardening

**Status: PASS**

Latest full test suite:

``` text
430 passed
0 failed
5 skipped
```

## Deployment

**Status: NOT STARTED**

Phase 12 is the next stage.

------------------------------------------------------------------------

# 37. Final Project Definition

FoveaMap can be summarized as:

> **A compact, semantically aware, temporally maintained,
> variable-resolution 2.5D LiDAR world model that spends spatial
> precision where it matters most and exposes the resulting map through
> production-oriented runtime, SDK, ROS2, visualization, and
> benchmarking interfaces.**

The fundamental architecture is:

``` text
                    FOVEAMAP
                       │
                       ▼
                 Raw LiDAR Data
                       │
                       ▼
                Validated LiDARFrame
                       │
                       ▼
                  Perception
             semantic + dynamic
                       │
                       ▼
             Temporal World Model
                       │
                       ▼
              Foveated 2.5D Grid
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
     Static         Dynamic        Terrain
      Map             Map        / Traversability
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                  MapSnapshot
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
       SDK            ROS2           HTTP
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                Applications /
                Dashboard /
                Planner
```

The central engineering invariant is:

> **High fidelity near the vehicle, controlled information loss at
> distance, exact spatial alignment between tiers, and explicit
> semantics throughout the pipeline.**

That is the core of FoveaMap.

------------------------------------------------------------------------

# 38. Source-of-Truth Documents

This document consolidates the project understanding from the project's
existing specifications and engineering records.

Important source documents include:

-   `docs/FoveaMap_PRD.pdf`
-   `docs/FoveaMap_Architecture_Vision.pdf`
-   `docs/FoveaMap_Visual_Design.pdf`
-   `docs/PHASE11_PRODUCTIZATION.md`
-   project test suites under `tests/`
-   benchmark modules under `benchmarks/`
-   the latest pre-deployment hardening and engineering sign-off report

For implementation questions, the **actual source code and tests remain
authoritative over narrative reports**.

------------------------------------------------------------------------

# 39. Engineering Rule for Future Development

Before declaring a future phase complete:

``` text
1. Read the requirements.
2. Inspect the actual source.
3. Inspect the actual tests.
4. Run the relevant tests.
5. Test boundary cases.
6. Test subsystem interactions.
7. Check performance/memory claims with measurements.
8. Check the deployment environment.
9. Fix confirmed defects.
10. Run the full regression suite.
11. Only then issue engineering sign-off.
```

A generated status report is evidence, not authority.

**FoveaMap should continue to be developed under this rule.**
