"""File-based LiDAR data sources and parsers (.bin, .pcd, .npy).

Provides deterministic readers for individual point cloud files and sequences of files,
with explicit intensity normalization, sensor origin policies, and timestamp provenance.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Any, Sequence
import numpy as np

from .source import LiDARSource
from ..core.contracts import LiDARFrame
from ..core.config import SensorConfig
from ..core.exceptions import DataAdapterError


_PCD_TYPE_MAP: dict[tuple[str, int], type] = {
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
    ("F", 4): np.float32,
    ("F", 8): np.float64,
}


def normalize_intensity(intensity: np.ndarray, mode: str = "auto") -> tuple[np.ndarray, str]:
    """Normalize LiDAR intensity/remission values according to an explicit, deterministic policy.

    Modes:
        - 'auto': inspects data values:
            * if max > 255.0 and max <= 65535.0 -> scales by 1/65535.0 to [0, 1]
            * if max > 1.5 and max <= 255.0 -> scales by 1/255.0 to [0, 1]
            * if max <= 1.0 and min >= 0.0 -> preserved as unit range [0, 1]
            * if max <= 1.5 -> clipped to [0, 1] (handles retroreflectors)
        - 'scale_255': divides by 255.0 and clips to [0, 1] (standard 8-bit sensors)
        - 'scale_65535': divides by 65535.0 and clips to [0, 1] (standard 16-bit sensors)
        - 'clip': clips directly to [0, 1]
        - 'raw' / 'none': returns raw float32 array untouched
    """
    inten = np.asarray(intensity, dtype=np.float32)
    if len(inten) == 0:
        return inten, "empty"

    mode_lower = mode.strip().lower()
    if mode_lower in ("raw", "none"):
        return inten, "raw"

    if mode_lower == "scale_255":
        return np.clip(inten / 255.0, 0.0, 1.0).astype(np.float32), "scale_255"

    if mode_lower == "scale_65535":
        return np.clip(inten / 65535.0, 0.0, 1.0).astype(np.float32), "scale_65535"

    if mode_lower == "clip":
        return np.clip(inten, 0.0, 1.0).astype(np.float32), "clip"

    if mode_lower == "auto":
        valid = np.isfinite(inten)
        if not np.any(valid):
            return np.zeros_like(inten), "auto_all_nan"
        valid_vals = inten[valid]
        thresh_val = float(np.percentile(valid_vals, 99.5)) if len(valid_vals) >= 100 else float(np.max(valid_vals))
        if thresh_val > 255.0:
            return np.clip(inten / 65535.0, 0.0, 1.0).astype(np.float32), "auto_scaled_16bit"
        elif thresh_val > 1.5:
            return np.clip(inten / 255.0, 0.0, 1.0).astype(np.float32), "auto_scaled_8bit"
        elif thresh_val > 1.0:
            return np.clip(inten, 0.0, 1.0).astype(np.float32), "auto_clipped_retroreflector"
        else:
            return np.clip(inten, 0.0, 1.0).astype(np.float32), "auto_unit_range"

    raise DataAdapterError(
        f"Unknown intensity normalization mode '{mode}'. Choose from 'auto', 'scale_255', 'scale_65535', 'clip', 'raw'."
    )


def parse_pcd(path: str | Path, normalize_intensity_mode: str = "auto") -> tuple[dict[str, np.ndarray], str]:
    """Robust PCD parser supporting ASCII and binary PCD files with arbitrary field types.

    Returns:
        (parsed_dict, normalization_provenance) where parsed_dict contains 'pts', 'intensity', 'ring'.
    """
    path_str = str(path)
    if not os.path.isfile(path_str):
        raise DataAdapterError(f"PCD file not found: {path_str}")

    header: dict[str, list[str]] = {}
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
    if data_type == "binary_compressed":
        raise DataAdapterError(
            f"Binary compressed (LZF) PCD is not supported for {path_str}; save as standard binary or ASCII PCD"
        )
    if data_type not in ("ascii", "binary"):
        raise DataAdapterError(f"Unsupported PCD data type '{data_type}' in {path_str}; expected 'ascii' or 'binary'")

    num_points = int(header.get("POINTS", [header.get("WIDTH", [0])[0]])[0])
    fields = [f.lower() for f in header.get("FIELDS", ["x", "y", "z"])]
    sizes = [int(s) for s in header.get("SIZE", ["4"] * len(fields))]
    types = [t.upper() for t in header.get("TYPE", ["F"] * len(fields))]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]

    if len(fields) != len(sizes) or len(fields) != len(types) or len(fields) != len(counts):
        raise DataAdapterError(
            f"Malformed PCD header in {path_str}: fields ({len(fields)}), sizes ({len(sizes)}), "
            f"types ({len(types)}), and counts ({len(counts)}) length mismatch"
        )

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
                if data.size == 0 and num_points > 0:
                    raise DataAdapterError(f"ASCII PCD file {path_str} has no readable point data")
                if data.ndim == 1 and data.size > 0:
                    data = data.reshape(1, -1)
                elif data.ndim == 1 and data.size == 0:
                    data = np.empty((0, len(fields)), dtype=np.float32)
                if num_points > 0 and len(data) != num_points:
                    raise DataAdapterError(
                        f"Truncated ASCII PCD file {path_str}: expected {num_points} points, got {len(data)}"
                    )
        except Exception as exc:
            if isinstance(exc, DataAdapterError):
                raise
            raise DataAdapterError(f"Failed to parse ASCII PCD points from {path_str}: {exc}") from exc

        if len(data) == 0:
            pts = np.empty((0, 3), dtype=np.float32)
            intensity = np.empty((0,), dtype=np.float32)
            ring = np.empty((0,), dtype=np.int16)
            norm_prov = "empty"
        else:
            pts = np.stack([data[:, x_idx], data[:, y_idx], data[:, z_idx]], axis=1).astype(np.float32)
            raw_inten = data[:, inten_idx].astype(np.float32) if inten_idx >= 0 else np.ones(len(pts), dtype=np.float32)
            ring = data[:, ring_idx].astype(np.int16) if ring_idx >= 0 else np.zeros(len(pts), dtype=np.int16)
            intensity, norm_prov = normalize_intensity(raw_inten, mode=normalize_intensity_mode)

    elif data_type == "binary":
        dtype_fields = []
        for f, s, t, c in zip(fields, sizes, types, counts):
            key = (t.upper(), s)
            if key not in _PCD_TYPE_MAP:
                raise DataAdapterError(f"Unsupported PCD field '{f}' with TYPE {t} and SIZE {s} in {path_str}")
            elem_dtype = _PCD_TYPE_MAP[key]
            if c == 1:
                dtype_fields.append((f, elem_dtype))
            else:
                dtype_fields.append((f, elem_dtype, (c,)))

        struct_dtype = np.dtype(dtype_fields)
        expected_bytes = struct_dtype.itemsize * num_points

        with open(path_str, "rb") as fh:
            fh.seek(data_offset)
            raw_bytes = fh.read()

        if len(raw_bytes) < expected_bytes:
            raise DataAdapterError(
                f"Truncated binary PCD file {path_str}: expected {expected_bytes} bytes for {num_points} points, got {len(raw_bytes)}"
            )

        if num_points == 0:
            pts = np.empty((0, 3), dtype=np.float32)
            intensity = np.empty((0,), dtype=np.float32)
            ring = np.empty((0,), dtype=np.int16)
            norm_prov = "empty"
        else:
            record = np.frombuffer(raw_bytes[:expected_bytes], dtype=struct_dtype, count=num_points)
            pts = np.stack([
                record["x"].astype(np.float32),
                record["y"].astype(np.float32),
                record["z"].astype(np.float32),
            ], axis=1)

            if inten_idx >= 0:
                raw_inten = record[fields[inten_idx]].astype(np.float32)
            else:
                raw_inten = np.ones(len(pts), dtype=np.float32)

            if ring_idx >= 0:
                ring = record[fields[ring_idx]].astype(np.int16)
            else:
                ring = np.zeros(len(pts), dtype=np.int16)

            intensity, norm_prov = normalize_intensity(raw_inten, mode=normalize_intensity_mode)

    return {"pts": pts, "intensity": intensity, "ring": ring}, norm_prov


def parse_bin(
    path: str | Path,
    columns: int | None = None,
    normalize_intensity_mode: str = "raw",
) -> tuple[dict[str, np.ndarray], str]:
    """Parse raw binary point cloud (.bin).

    Expects float32 records of 3 floats (x, y, z), 4 floats (x, y, z, intensity)
    or 5 floats (x, y, z, intensity, ring).
    """
    path_str = str(path)
    if not os.path.isfile(path_str):
        raise DataAdapterError(f"Binary LiDAR file not found: {path_str}")

    file_size = os.path.getsize(path_str)
    if file_size % 4 != 0:
        raise DataAdapterError(
            f"Binary file {path_str} has invalid byte length {file_size}; not divisible by 4 (float32 size)"
        )

    try:
        raw = np.fromfile(path_str, dtype=np.float32)
    except Exception as exc:
        raise DataAdapterError(f"Failed to read binary file {path_str}: {exc}") from exc

    if len(raw) == 0:
        return {
            "pts": np.empty((0, 3), dtype=np.float32),
            "intensity": np.empty((0,), dtype=np.float32),
            "ring": np.empty((0,), dtype=np.int16),
        }, "empty"

    if columns is not None:
        if len(raw) % columns != 0:
            raise DataAdapterError(
                f"Binary file {path_str} has {len(raw)} floats, not divisible by specified column count {columns}"
            )
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
        raw_inten = data[:, 3].astype(np.float32)
    else:
        raw_inten = np.ones(len(pts), dtype=np.float32)

    if data.shape[1] >= 5:
        ring = data[:, 4].astype(np.int16)
    else:
        ring = np.zeros(len(pts), dtype=np.int16)

    intensity, norm_prov = normalize_intensity(raw_inten, mode=normalize_intensity_mode)
    return {"pts": pts, "intensity": intensity, "ring": ring}, norm_prov


def parse_npy(
    path: str | Path,
    normalize_intensity_mode: str = "auto",
) -> tuple[dict[str, np.ndarray], str]:
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

        raw_inten = data["intensity"].astype(np.float32) if "intensity" in data else np.ones(len(pts), dtype=np.float32)
        ring = data["ring"].astype(np.int16) if "ring" in data else np.zeros(len(pts), dtype=np.int16)
    else:
        # Standard .npy array
        if data.ndim != 2 or data.shape[1] < 3:
            raise DataAdapterError(f"NPY file {path_str} must have shape (N, >=3); got {data.shape}")
        pts = data[:, :3].astype(np.float32)
        raw_inten = data[:, 3].astype(np.float32) if data.shape[1] >= 4 else np.ones(len(pts), dtype=np.float32)
        ring = data[:, 4].astype(np.int16) if data.shape[1] >= 5 else np.zeros(len(pts), dtype=np.int16)

    intensity, norm_prov = normalize_intensity(raw_inten, mode=normalize_intensity_mode)
    return {"pts": pts, "intensity": intensity, "ring": ring}, norm_prov


class FileLiDARSource(LiDARSource):
    """Source that loads a single point cloud file (.pcd, .bin, or .npy).

    Sensor Origin Policy:
        - If explicitly provided via `sensor_origin`, that 3D vector is used.
        - Otherwise, defaults to [0.0, 0.0, 0.0] (the LiDAR optical center).
          Arbitrary files are NOT implicitly assumed to be vehicle-mounted at 1.73m.
    """

    def __init__(
        self,
        file_path: str | Path,
        sensor_origin: np.ndarray | Sequence[float] | None = None,
        columns: int | None = None,
        intensity_normalization: str = "auto",
    ) -> None:
        self.file_path = str(file_path)
        self.columns = columns
        self.intensity_normalization = intensity_normalization

        if sensor_origin is not None:
            self.sensor_origin = np.asarray(sensor_origin, dtype=np.float32)
            self._origin_provenance = "user_supplied"
        else:
            self.sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._origin_provenance = "default_lidar_center"

        self._frame: LiDARFrame | None = None
        self._idx = 0
        self._load()

    def _load(self) -> None:
        ext = os.path.splitext(self.file_path)[1].lower()
        stem = Path(self.file_path).stem

        if ext == ".pcd":
            parsed, inten_prov = parse_pcd(self.file_path, normalize_intensity_mode=self.intensity_normalization)
        elif ext == ".bin":
            parsed, inten_prov = parse_bin(
                self.file_path, columns=self.columns, normalize_intensity_mode=self.intensity_normalization
            )
        elif ext in (".npy", ".npz"):
            parsed, inten_prov = parse_npy(self.file_path, normalize_intensity_mode=self.intensity_normalization)
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
            metadata={
                "path": self.file_path,
                "sensor_origin_provenance": self._origin_provenance,
                "intensity_normalization": inten_prov,
                "timestamp_provenance": "unspecified_single_frame",
            },
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
        sensor_origin: np.ndarray | Sequence[float] | None = None,
        intensity_normalization: str = "auto",
        columns: int | None = None,
    ) -> None:
        self.hz = hz
        self.intensity_normalization = intensity_normalization
        self.columns = columns

        if sensor_origin is not None:
            self.sensor_origin = np.asarray(sensor_origin, dtype=np.float32)
            self._origin_provenance = "user_supplied"
        else:
            self.sensor_origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self._origin_provenance = "default_lidar_center"

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
        source = FileLiDARSource(
            file_p,
            sensor_origin=self.sensor_origin,
            columns=self.columns,
            intensity_normalization=self.intensity_normalization,
        )
        frame = source[0]

        dt = 1.0 / self.hz if self.hz > 0 else 0.1
        meta = dict(frame.metadata)
        meta["sequence_index"] = index
        meta["timestamp_provenance"] = "derived_from_sequence_index"
        meta["sensor_origin_provenance"] = self._origin_provenance

        return LiDARFrame(
            pts=frame.pts,
            intensity=frame.intensity,
            ring=frame.ring,
            pose=frame.pose,
            sensor_origin=frame.sensor_origin,
            timestamp=float(index * dt),
            frame_id=Path(file_p).stem,
            source_id=self.source_id,
            metadata=meta,
        )

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        if self.file_paths:
            parent = Path(self.file_paths[0]).parent.name
            return f"sequence/{parent}"
        return "sequence/empty"
