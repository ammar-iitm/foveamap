"""File-based LiDAR data sources and parsers (.bin, .pcd, .npy).

Provides deterministic readers for individual point cloud files and sequences of files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Any
import numpy as np

from .source import LiDARSource
from ..core.contracts import LiDARFrame
from ..core.config import SensorConfig
from ..core.exceptions import DataAdapterError


def parse_pcd(path: str | Path) -> dict[str, np.ndarray]:
    """Lightweight, self-contained PCD parser for ASCII and binary PCD files.

    Returns dict containing 'pts' (N, 3) float32, and optional 'intensity', 'ring'.
    """
    path_str = str(path)
    if not os.path.isfile(path_str):
        raise DataAdapterError(f"PCD file not found: {path_str}")

    header = {}
    data_offset = 0

    with open(path_str, "rb") as fh:
        while True:
            line_bytes = fh.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("ascii", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            key = parts[0].upper()
            header[key] = parts[1:]

            if key == "DATA":
                data_offset = fh.tell()
                break

    if "DATA" not in header:
        raise DataAdapterError(f"Malformed PCD file (no DATA section found): {path_str}")

    data_type = header["DATA"][0].lower()
    num_points = int(header.get("POINTS", [header.get("WIDTH", [0])[0]])[0])
    fields = [f.lower() for f in header.get("FIELDS", ["x", "y", "z"])]

    if "x" not in fields or "y" not in fields or "z" not in fields:
        raise DataAdapterError(f"PCD file must contain x, y, and z fields: {fields}")

    x_idx = fields.index("x")
    y_idx = fields.index("y")
    z_idx = fields.index("z")
    inten_idx = fields.index("intensity") if "intensity" in fields else -1
    ring_idx = fields.index("ring") if "ring" in fields else (fields.index("laser") if "laser" in fields else -1)

    if data_type == "ascii":
        try:
            with open(path_str, "r", encoding="ascii", errors="ignore") as fh:
                fh.seek(data_offset)
                data = np.loadtxt(fh, dtype=np.float32)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
        except Exception as exc:
            raise DataAdapterError(f"Failed to parse ASCII PCD points from {path_str}: {exc}") from exc
    elif data_type == "binary":
        try:
            with open(path_str, "rb") as fh:
                fh.seek(data_offset)
                data = np.fromfile(fh, dtype=np.float32).reshape(num_points, -1)
        except Exception as exc:
            raise DataAdapterError(f"Failed to parse binary PCD points from {path_str}: {exc}") from exc
    else:
        raise DataAdapterError(f"Unsupported PCD data type: '{data_type}' (expected ascii or binary)")

    pts = np.stack([data[:, x_idx], data[:, y_idx], data[:, z_idx]], axis=1).astype(np.float32)
    intensity = data[:, inten_idx].astype(np.float32) if inten_idx >= 0 else np.ones(len(pts), dtype=np.float32)
    ring = data[:, ring_idx].astype(np.int16) if ring_idx >= 0 else np.zeros(len(pts), dtype=np.int16)

    return {"pts": pts, "intensity": intensity, "ring": ring}


def parse_bin(path: str | Path, columns: int | None = None) -> dict[str, np.ndarray]:
    """Parse raw binary point cloud (.bin).

    Expects float32 records of 4 floats (x, y, z, intensity) or 5 floats (x, y, z, intensity, ring).
    """
    path_str = str(path)
    if not os.path.isfile(path_str):
        raise DataAdapterError(f"Binary LiDAR file not found: {path_str}")

    try:
        raw = np.fromfile(path_str, dtype=np.float32)
    except Exception as exc:
        raise DataAdapterError(f"Failed to read binary file {path_str}: {exc}") from exc

    if columns is not None:
        if len(raw) % columns != 0:
            raise DataAdapterError(f"Binary file {path_str} has {len(raw)} floats, not divisible by {columns}")
        data = raw.reshape(-1, columns)
    elif path_str.endswith(".pcd.bin") and len(raw) % 5 == 0:
        data = raw.reshape(-1, 5)
    elif len(raw) % 4 == 0:
        data = raw.reshape(-1, 4)
    elif len(raw) % 5 == 0:
        data = raw.reshape(-1, 5)
    elif len(raw) % 3 == 0:
        data = raw.reshape(-1, 3)
    else:
        raise DataAdapterError(
            f"Binary file {path_str} has invalid byte length {len(raw)*4}; not divisible by 3, 4, or 5 float32 items"
        )

    pts = data[:, :3].astype(np.float32)
    if data.shape[1] >= 4:
        intensity = np.clip(data[:, 3], 0.0, 1.0).astype(np.float32)
    else:
        intensity = np.ones(len(pts), dtype=np.float32)

    if data.shape[1] >= 5:
        ring = data[:, 4].astype(np.int16)
    else:
        ring = np.zeros(len(pts), dtype=np.int16)

    return {"pts": pts, "intensity": intensity, "ring": ring}


def parse_npy(path: str | Path) -> dict[str, np.ndarray]:
    """Parse NumPy .npy or .npz point cloud file."""
    path_str = str(path)
    if not os.path.isfile(path_str):
        raise DataAdapterError(f"NumPy file not found: {path_str}")

    try:
        data = np.load(path_str, allow_pickle=False)
    except Exception as exc:
        raise DataAdapterError(f"Failed to load NumPy file {path_str}: {exc}") from exc

    if isinstance(data, np.lib.npyio.NpzFile):
        if "pts" in data:
            pts = data["pts"].astype(np.float32)
        elif "xyz" in data:
            pts = data["xyz"].astype(np.float32)
        else:
            raise DataAdapterError(f"NPZ archive {path_str} does not contain 'pts' or 'xyz' array")

        intensity = data["intensity"].astype(np.float32) if "intensity" in data else np.ones(len(pts), dtype=np.float32)
        ring = data["ring"].astype(np.int16) if "ring" in data else np.zeros(len(pts), dtype=np.int16)
    else:
        # Standard .npy array
        if data.ndim != 2 or data.shape[1] < 3:
            raise DataAdapterError(f"NPY file {path_str} must have shape (N, >=3); got {data.shape}")
        pts = data[:, :3].astype(np.float32)
        intensity = data[:, 3].astype(np.float32) if data.shape[1] >= 4 else np.ones(len(pts), dtype=np.float32)
        ring = data[:, 4].astype(np.int16) if data.shape[1] >= 5 else np.zeros(len(pts), dtype=np.int16)

    return {"pts": pts, "intensity": intensity, "ring": ring}


class FileLiDARSource(LiDARSource):
    """Source that loads a single point cloud file (.pcd, .bin, or .npy)."""

    def __init__(
        self,
        file_path: str | Path,
        sensor_origin: np.ndarray | None = None,
        columns: int | None = None,
    ) -> None:
        self.file_path = str(file_path)
        self.sensor_origin = sensor_origin if sensor_origin is not None else np.array([0.0, 0.0, 1.73], dtype=np.float32)
        self.columns = columns
        self._frame: LiDARFrame | None = None
        self._idx = 0
        self._load()

    def _load(self) -> None:
        ext = os.path.splitext(self.file_path)[1].lower()
        stem = Path(self.file_path).stem

        if ext == ".pcd":
            parsed = parse_pcd(self.file_path)
        elif ext == ".bin":
            parsed = parse_bin(self.file_path, columns=self.columns)
        elif ext in (".npy", ".npz"):
            parsed = parse_npy(self.file_path)
        else:
            raise DataAdapterError(f"Unsupported point cloud file extension '{ext}' for {self.file_path}")

        self._frame = LiDARFrame(
            pts=parsed["pts"],
            intensity=parsed["intensity"],
            ring=parsed["ring"],
            pose=np.eye(4, dtype=np.float64),
            sensor_origin=self.sensor_origin,
            timestamp=0.0,
            frame_id=stem,
            source_id=self.source_id,
            metadata={"path": self.file_path},
        )

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[LiDARFrame]:
        if self._idx == 0 and self._frame is not None:
            self._idx += 1
            yield self._frame

    def __getitem__(self, index: int) -> LiDARFrame:
        if index != 0 or self._frame is None:
            raise IndexError(f"Single-file source index {index} out of range [0, 1)")
        return self._frame

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        return f"file/{Path(self.file_path).name}"


class BinFileSource(FileLiDARSource):
    """LiDAR source for raw binary .bin files."""
    pass


class PcdFileSource(FileLiDARSource):
    """LiDAR source for Point Cloud Data .pcd files."""
    pass


class NpyFileSource(FileLiDARSource):
    """LiDAR source for NumPy .npy / .npz point cloud files."""
    pass


class FileSequenceSource(LiDARSource):
    """Source that iterates over an ordered sequence of point cloud files in a directory."""

    def __init__(
        self,
        directory_or_files: str | Path | list[str] | list[Path],
        pattern: str = "*.bin",
        hz: float = 10.0,
        sensor_origin: np.ndarray | None = None,
    ) -> None:
        self.hz = hz
        self.sensor_origin = sensor_origin if sensor_origin is not None else np.array([0.0, 0.0, 1.73], dtype=np.float32)
        self._idx = 0

        if isinstance(directory_or_files, (list, tuple)):
            self.file_paths = [str(p) for p in directory_or_files]
        else:
            dir_path = Path(directory_or_files)
            if not dir_path.is_dir():
                raise DataAdapterError(f"Directory not found: {directory_or_files}")
            self.file_paths = sorted([str(p) for p in dir_path.glob(pattern)])

        if not self.file_paths:
            raise DataAdapterError(
                f"No files matching '{pattern}' found in {directory_or_files}"
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __iter__(self) -> Iterator[LiDARFrame]:
        while self._idx < len(self.file_paths):
            frame = self[self._idx]
            self._idx += 1
            yield frame

    def __getitem__(self, index: int) -> LiDARFrame:
        if index < 0 or index >= len(self.file_paths):
            raise IndexError(f"Sequence index {index} out of range [0, {len(self.file_paths)})")

        file_p = self.file_paths[index]
        source = FileLiDARSource(file_p, sensor_origin=self.sensor_origin)
        frame = source[0]

        # Assign chronological timestamp and frame_id based on sequence position
        dt = 1.0 / self.hz if self.hz > 0 else 0.1
        return LiDARFrame(
            pts=frame.pts,
            intensity=frame.intensity,
            ring=frame.ring,
            pose=frame.pose,
            sensor_origin=frame.sensor_origin,
            timestamp=float(index * dt),
            frame_id=Path(file_p).stem,
            source_id=self.source_id,
            metadata=dict(frame.metadata, sequence_index=index),
        )

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        if self.file_paths:
            parent = Path(self.file_paths[0]).parent.name
            return f"sequence/{parent}"
        return "sequence/empty"
