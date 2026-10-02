# FoveaMap — Perception Architecture & Model Integration

## 1. Architectural Overview

Phase 4 establishes an independent, decoupled perception subsystem that transforms canonical `LiDARFrame` sweeps into validated `PerceptionResult` outputs. RangeUNet is an implementation of this interface—it is not the architecture itself.

```text
                  LiDARFrame (Canonical Input)
                              │
                              ▼
                      LiDARPreprocessor
                              │
                              ▼
                      Feature Extraction
                (Range Image + LMNet Motion Cues)
                              │
                              ▼
                    ┌───────────────────┐
                    │ PerceptionBackend │ (Abstract Interface)
                    └─────────┬─────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
    RangeUNetBackend               ClassicalFallbackBackend
 (Range-image U-Net Neural Net)   (Heuristic Geometric Baseline)
            │                                   │
            └─────────────────┬─────────────────┘
                              │
                              ▼
                      PerceptionResult
            (Host: NumPy / Device: DevicePerceptionResult)
                              │
                              ▼
                      FoveaMapRuntime
              (Projection & Multi-Tier Mapping)
```

The mapping subsystem consumes `PerceptionResult` (or `DevicePerceptionResult` on GPU) and does not depend on internal RangeUNet neural layers or channel layouts.

---

## 2. Formal Perception Backend Interface

All perception models implement the `foveamap.runtime.PerceptionBackend` abstract base class:

```python
class PerceptionBackend(ABC):
    @abstractmethod
    def predict(
        self,
        frame: LiDARFrame,
        *,
        device: DeviceContext | None = None,
    ) -> PerceptionResult:
        """Run inference and return host-resident canonical PerceptionResult."""
        ...

    @abstractmethod
    def predict_device(
        self,
        frame: LiDARFrame,
        dev_math: bool = False,
    ) -> DevicePerceptionResult:
        """Run inference retaining tensors on device for zero-copy GPU mapping."""
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...
```

### Registered Implementations
- **`RangeUNetBackend`**: High-accuracy range-image U-Net predicting 9 semantic classes and dynamic object residuals.
- **`ClassicalFallbackBackend`**: Zero-weight geometric heuristic classifier for low-power fallback and testing environments.

---

## 3. Model Input Contract (RangeUNet)

RangeUNet processes a 2D cylindrical range image projection of shape `[B, 8, H, W]`:
- **Batch Dimension ($B$)**: 1 during online streaming inference.
- **Spatial Height ($H$)**: Number of sensor beam rows (e.g. 64 for Velodyne HDL-64E, 32 for nuScenes).
- **Spatial Width ($W$)**: Number of horizontal azimuth columns (1024 columns standard, $0.35^\circ$ angular resolution).
- **Input Channels (8)**:
  1. `channel 0`: Radial range normalized by $50\,\text{m}$ ($r / 50.0$, where $r = \sqrt{x^2 + y^2 + z^2}$).
  2. `channel 1`: Ego forward coordinate normalized by $50\,\text{m}$ ($x / 50.0$).
  3. `channel 2`: Ego left coordinate normalized by $50\,\text{m}$ ($y / 50.0$).
  4. `channel 3`: Ego up coordinate normalized by $3\,\text{m}$ ($z / 3.0$).
  5. `channel 4`: Calibrated intensity/remission in $[0, 1]$.
  6. `channel 5`: Binary validity mask ($1.0$ for valid return, $0.0$ for empty/no-return).
  7. `channel 6`: Motion residual against sweep at $t - 0.1\,\text{s}$ ($\min(5 \cdot |\Delta r| / r, 1.0)$).
  8. `channel 7`: Motion residual against sweep at $t - 0.2\,\text{s}$ ($\min(5 \cdot |\Delta r| / r, 1.0)$).

### Pixel Projection & Collisions
- **Azimuth Column**: $\text{col} = \text{round}((180.0 - \text{azimuth}) / 360.0 \cdot W) \pmod W$.
- **Beam Row**: Directly indexed by sensor `ring` ($0 = \text{top beam}$).
- **Multiple Points per Pixel**: **Nearest point wins** (smallest radial distance $r$; ties broken by highest point ID).
- **Empty / No-Return Pixels**: Zero across all channels, validity mask $= 0.0$.

---

## 4. Model Output Contract (RangeUNet)

RangeUNet produces dual output heads:
1. **Semantic Head**: `[B, 9, H, W]` logits $\rightarrow$ `softmax` over class dimension $\rightarrow$ per-pixel probabilities $[0, 1]$.
   - Inactive classes are masked with $-10^4$ before softmax.
2. **Motion Head**: `[B, 1, H, W]` logits $\rightarrow$ `sigmoid` $\rightarrow$ per-pixel probability of being dynamic $[0, 1]$.

### Per-Point Gather & Motion Classification
Every LiDAR point gathers predictions from its projected pixel coordinate $(r, c)$:
- **Point Semantic Class**: `cls = argmax(P, dim=1)` ($0 \le \text{cls} < 9$).
- **Point Confidence**: Max class probability $\max_c P_c \in [0, 1]$.
- **Motion Decision**: Dynamic (`is_moving = True`) if and only if:
  $$\text{moving\_probability} > \text{confidence\_threshold} \quad \land \quad \text{cls} \in \{\text{VEHICLE}, \text{PERSON}\}$$

---

## 5. Canonical Perception Result Contracts

### `PerceptionResult` (Host / NumPy)
Frozen dataclass defining system-wide boundary:
- `class_probabilities`: `(N, C)` `float32` in $[0, 1]$.
- `moving_probabilities`: `(N,)` `float32` in $[0, 1]$.
- `semantic_predictions`: `(N,)` `int64` with $0 \le \text{class\_id} < C$.
- `is_moving`: `(N,)` `bool`.
- `confidence`: `(N,)` `float32` in $[0, 1]$ (accessible via `.point_confidence`).
- `point_indices`: `tuple[np.ndarray, np.ndarray]` $(row, col)$ mapping points to range-image pixels.
- `metadata`: Execution device, model name, and profiling timings.

### `DevicePerceptionResult` (Device / PyTorch)
Zero-copy container keeping tensors on active hardware (`device`):
- Consumed directly by `TorchFoveatedGrid` on GPU without intermediate CPU round-trips.
- `.to_host()` method produces a validated `PerceptionResult` copy when CPU consumers request it.

---

## 6. Device-Resident Execution & Precision

- **Device Support**: `cpu`, `cuda`, `mps` (Apple Silicon), `auto`.
- **Zero-Copy Pipeline**:
  - `LiDARFrame -> Feature Extraction (GPU) -> RangeUNet (GPU) -> Point Projection (GPU) -> Grid Binning (GPU)`.
  - CPU host transfer only happens on snapshot publication.
- **Precision**:
  - `fp16=True`: When running on CUDA, activates `torch.autocast(device_type="cuda", dtype=torch.float16)`.
  - When running on CPU, autocast is automatically disabled to guarantee full float32 numerical precision.
  - Class indices and point indices remain strictly 64-bit integer types (`int64`).

---

## 7. Checkpoint Management & Inference Lifecycle

1. **One-Time Initialization**: Model weights are loaded once during `__init__` via `torch.load(..., map_location="cpu", weights_only=True)` and transferred to target device.
2. **Missing Checkpoint**: Raises `PerceptionError` with clear diagnostic path.
3. **Inference Mode**: Inference strictly executes within `@torch.inference_mode()` with `model.eval()`. No autograd graphs are retained.
4. **Finiteness Validation**: All model outputs are checked for `NaN`/`Inf`; non-finite activations immediately raise `PerceptionError`.

---

## 8. Latency Profiling Instrumentation

Every inference pass measures fine-grained stage timings reported in `metadata["timing"]`:
- `feature_ms`: Time spent building range-image features and motion residuals.
- `inference_ms`: Time spent executing neural network forward pass.
- `postprocess_ms`: Time spent gathering point predictions and applying motion logic.
- `total_ms`: End-to-end perception latency.

---

## 9. Known Limitations
- **CUDA Physical Verification**: Conducted on CPU development host; CUDA execution must be physically verified in a dedicated GPU environment (e.g. 12-GB Colab runner).
- **Batch Size**: Optimized for single-sweep streaming inference ($B=1$).
