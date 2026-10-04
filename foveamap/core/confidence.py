"""Authoritative confidence packing and unpacking for FoveaMap.

Defines the exact 1-byte cell representation for class confidence:
- Upper 4 bits (conf >> 4): Primary class confidence (0..15 -> 0.0..1.0)
- Lower 4 bits (conf & 0x0F): Secondary class confidence (0..15 -> 0.0..1.0)

Provides unified pack, unpack, and accessor functions with exact mathematical
parity between NumPy and PyTorch implementations.
"""
from __future__ import annotations

import numpy as np
import torch


def pack_confidence_np(
    primary: float | np.ndarray,
    secondary: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Pack primary and secondary confidence into uint8 byte(s)."""
    p = np.clip(np.round(np.asarray(primary, dtype=np.float32) * 15.0), 0, 15).astype(np.uint8)
    s = np.clip(np.round(np.asarray(secondary, dtype=np.float32) * 15.0), 0, 15).astype(np.uint8)
    return (p << 4) | (s & 0x0F)


def unpack_primary_confidence_np(packed: int | np.ndarray) -> np.ndarray:
    """Extract primary confidence in [0.0, 1.0] from packed uint8 byte(s)."""
    return (np.asarray(packed, dtype=np.uint8) >> 4).astype(np.float32) / 15.0


def unpack_secondary_confidence_np(packed: int | np.ndarray) -> np.ndarray:
    """Extract secondary confidence in [0.0, 1.0] from packed uint8 byte(s)."""
    return (np.asarray(packed, dtype=np.uint8) & 0x0F).astype(np.float32) / 15.0


def pack_confidence_torch(
    primary: torch.Tensor,
    secondary: torch.Tensor,
) -> torch.Tensor:
    """Pack primary and secondary confidence tensors on device into torch.uint8."""
    p = (primary * 15.0).round().clamp(0, 15).to(torch.uint8)
    s = (secondary * 15.0).round().clamp(0, 15).to(torch.uint8)
    return (p << 4) | (s & 0x0F)


def unpack_primary_confidence_torch(packed: torch.Tensor) -> torch.Tensor:
    """Extract primary confidence in [0.0, 1.0] from packed torch.uint8 tensor."""
    return (packed >> 4).float() / 15.0


def unpack_secondary_confidence_torch(packed: torch.Tensor) -> torch.Tensor:
    """Extract secondary confidence in [0.0, 1.0] from packed torch.uint8 tensor."""
    return (packed & 0x0F).float() / 15.0
