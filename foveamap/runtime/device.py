"""Execution device resolution and runtime context for FoveaMap.

Provides hardware discovery and device selection for CPU and CUDA targets
without hardcoded platform assumptions or external cloud dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import torch

from ..core.config import RuntimeConfig, PerceptionConfig
from ..core.exceptions import ConfigurationError


@dataclass(frozen=True)
class DeviceContext:
    """Resolved computing context for runtime execution."""
    device: torch.device
    device_type: str
    grid_engine: str
    features_engine: str
    use_cuda: bool
    fp16_enabled: bool
    description: str


def canonical_device(device: torch.device | str) -> torch.device:
    """The device with its index filled in, as tensors report it.

    ``torch.device("cuda") != torch.device("cuda:0")`` even though tensors
    created on the first spells their device ``cuda:0``; the same holds for
    ``mps``. Comparing canonical devices avoids false mismatches.
    """
    d = torch.device(device)
    if d.index is None:
        if d.type == "cuda":
            try:
                return torch.device("cuda", torch.cuda.current_device())
            except (AssertionError, RuntimeError):     # no usable CUDA (e.g. CPU-only build): the default is 0
                return torch.device("cuda", 0)
        if d.type == "mps":
            return torch.device("mps", 0)
    return d


def resolve_device(
    runtime_config: RuntimeConfig,
    perception_config: PerceptionConfig | None = None,
) -> DeviceContext:
    """Resolve computing device and execution parameters from runtime configuration.

    Args:
        runtime_config: Hardware and engine configuration.
        perception_config: Optional perception configuration for fp16 policy.

    Returns:
        DeviceContext with resolved torch.device and hardware capabilities.

    Raises:
        ConfigurationError: If CUDA is explicitly requested but unavailable.
    """
    requested = (runtime_config.device or "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            use_cuda = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            use_cuda = False
        else:
            device = torch.device("cpu")
            use_cuda = False
    elif requested == "cpu":
        device = torch.device("cpu")
        use_cuda = False
    elif requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ConfigurationError(
                f"CUDA device {runtime_config.device!r} requested, but "
                "torch.cuda.is_available() is False in the current environment."
            )
        try:
            device = torch.device(runtime_config.device)
        except Exception as exc:
            raise ConfigurationError(f"Invalid CUDA device specification {runtime_config.device!r}: {exc}") from exc
        use_cuda = True
    else:
        try:
            device = torch.device(runtime_config.device)
        except Exception as exc:
            raise ConfigurationError(f"Unsupported device specification {runtime_config.device!r}: {exc}") from exc
        use_cuda = (device.type == "cuda")
        if use_cuda and not torch.cuda.is_available():
            raise ConfigurationError(f"CUDA device requested ({device}), but torch.cuda is not available.")

    # Configure CPU thread count if running on CPU
    if not use_cuda and not torch.cuda.is_available():
        cpu_cores = max(1, os.cpu_count() or 1)
        torch.set_num_threads(cpu_cores)

    # Determine mixed-precision policy
    fp16 = bool(perception_config.fp16 and use_cuda) if perception_config is not None else False

    # Build human-readable hardware descriptor
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "Unknown GPU"
        desc = f"CUDA GPU ({gpu_name}) - FP16 inference {'enabled' if fp16 else 'disabled'}"
    elif device.type == "mps":
        desc = "Apple Silicon MPS (Metal Performance Shaders)"
    else:
        desc = f"CPU ({os.cpu_count() or 1} vCPU threads)"

    device = canonical_device(device)
    return DeviceContext(
        device=device,
        device_type=device.type,
        grid_engine=runtime_config.grid_engine,
        features_engine=runtime_config.features_engine,
        use_cuda=use_cuda,
        fp16_enabled=fp16,
        description=desc,
    )


def sync_device(device: torch.device) -> None:
    """Synchronize device execution stream if running on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
