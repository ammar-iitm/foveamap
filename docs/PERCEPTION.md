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

## 2. Formal Semantic Ontology (Single Source of Truth)

All perception and mapping components reference a single authoritative semantic taxonomy defined in `foveamap.core.ontology`:

- **Authoritative Constant**: `foveamap.core.ontology.NUM_CLASSES = 9`
- **Canonical Class IDs**:
  - `0`: `ROAD`
  - `1`: `SIDEWALK`
  - `2`: `PARKING`
  - `3`: `TERRAIN`
  - `4`: `VEGETATION`
  - `5`: `BUILDING`
  - `6`: `POLE`
  - `7`: `VEHICLE`
  - `8`: `PERSON`
- **Ontology Decoupling**: Dataset loaders (e.g. `SemanticKITTI`, `nuScenes`, `sim`) map dataset-specific labels into this canonical ontology. The perception backend does not couple to any dataset-specific ontology. Incompatible `num_classes` configurations are strictly rejected at initialization time with `ConfigurationError`.

---

## 3. Formal Perception Backend Interface

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

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal temporal history."""
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...
```

### Registered Implementations
- **`RangeUNetBackend`**: High-accuracy range-image U-Net predicting 9 semantic classes and dynamic motion probabilities. Requires a trained checkpoint file.
- **`ClassicalFallbackBackend`**: Zero-weight geometric heuristic classifier for low-power fallback, degraded operation, and headless testing environments. Must be explicitly requested (`backend_type="classical"`). It is never silently substituted for a missing RangeUNet checkpoint.

---

## 4. Device Temporal State & Zero Round-Trip Pipeline

To eliminate per-frame `GPU -> CPU -> GPU` temporal round-trips:

- **`DeviceTemporalState` & `DeviceTemporalSweep`**:
  - Sweep history is retained strictly on the active PyTorch device as `pts_world` and `ring` tensors.
  - History is bounded to 2 previous sweeps (`max_sweeps=2`).
  - In `RangeUNetBackend.predict_device(frame)`, previous sweeps are transformed into current ego coordinates on the device using `torch.linalg.inv(pose_dev)` in double precision.
  - No CPU memory allocation or tensor round-trip occurs in the device hot path.
  - Deterministic and resettable via `backend.reset()`.
  - Longevity verified across 100+ streaming frames without unbounded memory accumulation.

---

## 5. Model Input & Output Contracts (RangeUNet)

### RangeUNet Input Tensor
Processes a 2D cylindrical range image projection of shape `[B, 8, H, W]`:
- **Channels**:
  1. `channel 0`: Radial range normalized by $50\,\text{m}$ ($r / 50.0$).
  2. `channel 1`: Ego forward coordinate normalized by $50\,\text{m}$ ($x / 50.0$).
  3. `channel 2`: Ego left coordinate normalized by $50\,\text{m}$ ($y / 50.0$).
  4. `channel 3`: Ego up coordinate normalized by $3\,\text{m}$ ($z / 3.0$).
  5. `channel 4`: Calibrated intensity/remission in $[0, 1]$.
  6. `channel 5`: Binary validity mask ($1.0$ for valid return, $0.0$ for empty/no-return).
  7. `channel 6`: Motion residual against sweep at $t - 0.1\,\text{s}$ ($\min(5 \cdot |\Delta r| / r, 1.0)$).
  8. `channel 7`: Motion residual against sweep at $t - 0.2\,\text{s}$ ($\min(5 \cdot |\Delta r| / r, 1.0)$).

### RangeUNet Dual Heads
1. **Semantic Head**: `[B, 9, H, W]` logits $\rightarrow$ `softmax` over class dimension $\rightarrow$ per-pixel probabilities $[0, 1]$.
2. **Motion Head**: `[B, 1, H, W]` logits $\rightarrow$ `sigmoid` $\rightarrow$ per-pixel dynamic probability $[0, 1]$.

Points gather predictions from their projected $(row, col)$ pixel coordinates. Points are declared dynamic if:
$$\text{moving\_probability} > \text{threshold} \quad \land \quad \text{class} \in \{\text{VEHICLE}, \text{PERSON}\}$$

---

## 6. Hardened Perception Result Contracts

### `PerceptionResult` (Host / NumPy)
Strictly validated frozen contract:
- `class_probabilities`: `(N, C)` `float32` in $[0, 1]$.
- `moving_probabilities`: `(N,)` `float32` in $[0, 1]$.
- `semantic_predictions`: `(N,)` `int64` with $0 \le \text{class\_id} < C$.
- `is_moving`: `(N,)` `bool`.
- `confidence`: `(N,)` `float32` in $[0, 1]$ (accessible via `.point_confidence`).
- `point_indices`: `tuple[np.ndarray, np.ndarray]` $(row, col)$ with non-negative coordinates.
- Contract violations raise `ContractError` or `NumericalConsistencyError`.

### `DevicePerceptionResult` (Device / PyTorch)
On-device zero-copy tensor container:
- Validates tensor types, device matching, shapes, finiteness, and bounds via device reductions without CPU synchronization.
- `.to_host()` converts tensors to a validated host `PerceptionResult` when requested.

---

## 7. Checkpoint Safety Policy

1. **Mandatory Checkpoint**: `backend_type="range_unet"` requires a valid checkpoint path. If `checkpoint_path is None`, initialization raises `ConfigurationError`. If the file does not exist, initialization raises `CheckpointNotFoundError`.
2. **No Silent Random Inference**: Randomly initialized neural network weights are prohibited in production pipelines.
3. **Opt-in Untrained Mode**: An explicit flag `allow_untrained=True` is provided solely for headless unit testing.
4. **Lifecycle**: Weights are loaded once during initialization. Model parameters are frozen in `model.eval()`, and all inference runs under `@torch.inference_mode()`.

---

## 8. Hardware & CUDA Verification Status

- **Host Testing**: All unit, integration, and longevity tests pass on CPU.
- **CUDA Verification**: **NOT PHYSICALLY VERIFIED — CUDA UNAVAILABLE** on the current Windows development machine. Real CUDA execution and CPU/CUDA FP16 numerical parity tests are conditionally guarded and must be physically verified in a dedicated GPU environment (e.g. 12-GB Colab runner).

---

## 9. Known Limitations
- **CUDA Physical Verification**: Conducted on CPU development host; CUDA execution must be physically verified in a dedicated GPU environment (e.g. 12-GB Colab runner).
- **Batch Size**: Optimized for single-sweep streaming inference ($B=1$).
