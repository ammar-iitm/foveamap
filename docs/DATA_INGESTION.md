# FoveaMap — Data Ingestion & Preprocessing Architecture

## 1. Architectural Overview

FoveaMap Phase 3 establishes an immutable, deterministic, source-independent boundary between heterogeneous LiDAR inputs and the perception/mapping runtime:

```text
       Raw LiDAR Inputs
  (SemanticKITTI / nuScenes / PCD / BIN / NPY / Simulator)
                    │
                    ▼
            ┌───────────────┐
            │  LiDARSource  │   (Source Adapter / Ingestion)
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  LiDARFrame   │   (Canonical Typed Contract)
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ Preprocessor  │   (Deterministic Geometric Filtering)
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │ FoveaMapRuntime│   (Phase 2 Perception & Mapping)
            └───────────────┘
```

This ensures downstream components (RangeUNet, projection, foveated grid, temporal fusion) are entirely decoupled from dataset-specific formats, disk layouts, and calibration idiosyncrasies.

---

## 2. Canonical Contract: `LiDARFrame`

The canonical representation across the entire system is `foveamap.core.contracts.LiDARFrame`.

### Invariants & Coordinates
- **Coordinate Convention**: ISO 8855 right-handed ego vehicle frame:
  - $+X$: Forward (metres)
  - $+Y$: Left (metres)
  - $+Z$: Up (metres)
- **Point Coordinates (`pts`)**: `(N, 3)` `float32`, strictly finite (no `NaN` or `Inf`).
- **Intensity / Remission (`intensity`)**: `(N,)` `float32`, strictly finite, normalized to $[0, 1]$ or explicit scale.
- **Beam Index / Ring (`ring`)**: `(N,)` integer (`int16`), laser beam index / range image row (0 = uppermost beam).
- **Ego Pose (`pose`)**: `(4, 4)` `float64` $SE(3)$ transformation matrix from ego to world frame. Valid homogeneous bottom row `[0, 0, 0, 1]`.
- **Sensor Origin (`sensor_origin`)**: `(3,)` `float32` LiDAR optical center in the ego vehicle coordinate frame.
- **Timestamp (`timestamp`)**: `float` seconds (monotonically non-decreasing across a sweep sequence).
- **Source Identification (`source_id`, `frame_id`)**: Strings for provenance and diagnostic tracing.
- **Optional Annotations (`label`, `moving`)**: `(N,)` `int8` / `(N,)` `bool`. Ground truth is strictly isolated and can be removed via `frame.without_annotations()`.

---

## 3. Semantics & Provenance Policies

### 3.1 Intensity Semantics
Arbitrary point clouds must not be silently clipped. `normalize_intensity(intensity, mode)` supports:
- **`auto`** (default):
  - Values in $(255, 65535] \rightarrow$ divided by $65535.0$ (16-bit sensor).
  - Values in $(1.5, 255] \rightarrow$ divided by $255.0$ (8-bit sensor).
  - Values in $[0, 1.0] \rightarrow$ preserved as unit range.
  - Values in $(1.0, 1.5] \rightarrow$ clipped to $[0, 1]$ (retroreflectors).
- **`scale_255`**: Explicit 8-bit sensor scaling ($I / 255.0$).
- **`scale_65535`**: Explicit 16-bit sensor scaling ($I / 65535.0$).
- **`clip`**: Direct clipping to $[0, 1]$.
- **`raw`**: Leaves intensity untouched as raw sensor values.

### 3.2 Sensor Origin Semantics
- **Calibrated Dataset (SemanticKITTI, nuScenes)**: Uses dataset-calibrated sensor mounting transformation with provenance `dataset_calibrated_mount`.
- **User Supplied**: If `sensor_origin` is explicitly passed to a file or source adapter, it is used with provenance `user_supplied`.
- **Arbitrary Files (`.pcd`, `.bin`, `.npy`)**: Defaults to `[0.0, 0.0, 0.0]` (optical center) with provenance `default_lidar_center`. FoveaMap **never** implicitly assumes arbitrary external files are mounted at vehicle height $1.73\,\text{m}$.

### 3.3 Timestamp Semantics & Provenance
Frame metadata explicitly distinguishes:
- **`recorded_sensor_epoch`**: Actual hardware clock timestamps (e.g. nuScenes microsecond timestamps).
- **`recorded_sensor_times_txt`**: External recorded timestamp log (e.g. KITTI `times.txt`).
- **`derived_from_scan_id_10hz`**: Reconstructed from scan rate (e.g. KITTI $10\,\text{Hz}$ scan index).
- **`derived_from_sequence_index`**: Chronological file sequence ordering ($i / \text{hz}$).
- **`synthetic`**: Pure simulator time steps.

---

## 4. Point Cloud File Parsers

File parsers in `foveamap.data.file` reject corrupt data with clear `DataAdapterError`:
- **PCD (`.pcd`)**:
  - Supports ASCII and binary PCD with structured types (`F`, `I`, `U`) and heterogeneous field sizes (`SIZE 4 4 4 4 2`).
  - Rejects binary compressed (LZF) PCD with explicit instruction.
  - Validates `x`, `y`, `z` fields exist and file contains expected byte size.
- **BIN (`.bin`)**:
  - Rejects files where byte length is not a multiple of 4 bytes (`float32`).
  - Automatically identifies 3-, 4-, or 5-float point formats, or accepts explicit `columns`.
- **NPY / NPZ (`.npy`, `.npz`)**:
  - Supports 2D arrays (`(N, >=3)`) and `.npz` archives with `pts`, `intensity`, `ring` keys.

---

## 5. Preprocessing: `LiDARPreprocessor`

Deterministic, non-mutating point cloud filtering governed by `PreprocessConfig`:
1. **Finite Coordinate Check**: Rejects `NaN`/`Inf` or drops invalid points if `remove_invalid=True`.
2. **Radial Range Filtering**: Filters points outside $[\text{min\_range\_m}, \text{max\_range\_m}]$ relative to `sensor_origin`.
3. **Ego Self-Hit Removal**: Filters points where $\sqrt{(x - x_0)^2 + (y - y_0)^2} \le \text{self\_hit\_radius\_m}$.
4. **Z-Bounds Filtering**: Optional minimum and maximum vehicle $Z$ elevation cuts.

The input `LiDARFrame` is **strictly immutable** and unmodified by the preprocessor.

---

## 6. How to Add a New LiDAR Sensor or Dataset

Adding a new sensor or dataset requires subclassing `LiDARSource` and registering it in `foveamap.data.factory`:

```python
import numpy as np
from typing import Iterator
from foveamap.core.contracts import LiDARFrame
from foveamap.core.config import SensorConfig
from foveamap.data import LiDARSource, register_source

class OusterOS1Source(LiDARSource):
    """Adapter for Ouster OS1-64 LiDAR packet stream or pcap extract."""

    def __init__(self, file_path: str, hz: float = 10.0) -> None:
        self.file_path = file_path
        self.hz = hz
        self._idx = 0
        # Load raw packets or scans
        self._scans = self._load_scans()

    def _load_scans(self):
        # Implementation-specific loading...
        return []

    def __len__(self) -> int:
        return len(self._scans)

    def __iter__(self) -> Iterator[LiDARFrame]:
        while self._idx < len(self._scans):
            yield self[self._idx]
            self._idx += 1

    def __getitem__(self, index: int) -> LiDARFrame:
        scan = self._scans[index]
        return LiDARFrame(
            pts=scan["xyz"],                       # (N, 3) float32
            intensity=scan["intensity"],           # (N,) float32
            ring=scan["beam_id"],                  # (N,) int16
            pose=scan.get("pose", np.eye(4)),      # (4, 4) float64
            sensor_origin=np.array([0.0, 0.0, 1.8], dtype=np.float32),
            timestamp=float(scan["timestamp_sec"]),
            frame_id=f"os1_{index:06d}",
            source_id=self.source_id,
            metadata={"timestamp_provenance": "recorded_sensor_epoch"},
        )

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        return "ouster/os1_64"

    @property
    def sensor_config(self) -> SensorConfig:
        return SensorConfig(
            name="os1_64",
            n_rows=64,
            n_cols=1024,
            fov_up_deg=22.5,
            fov_down_deg=-22.5,
            hz=self.hz,
            source_description="Ouster OS1-64 64-beam LiDAR",
        )

# Register into the global factory:
register_source("ouster", OusterOS1Source)
register_source("os1", OusterOS1Source)
```

Now any caller can instantiate the source seamlessly:

```python
from foveamap.data import create_source
source = create_source("ouster", file_path="/data/drive.pcap")
```

---

## 7. Legacy Compatibility Boundary

Legacy files (`foveamap/frames.py`, `semantickitti.py`, `nuscenes.py`) are preserved for training recipes and historical benchmarking:
- `LiDARFrame.to_legacy_dict()` converts canonical frames into the dictionary format expected by legacy code.
- `LiDARFrame.from_legacy_dict(d)` converts legacy dictionaries into canonical `LiDARFrame` instances.
- Both `FoveaMapRuntime.process()` and `FoveaMapPipeline.step()` natively accept canonical `LiDARFrame` and legacy dictionaries.
