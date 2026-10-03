"""PointCloud2 <-> FoveaMap conversion (ROS 2 independent).

Operates on duck-typed messages: any object with ``fields``, ``height``,
``width``, ``point_step``, ``row_step``, ``data``, ``is_dense`` and
``header`` (with ``stamp.sec``, ``stamp.nanosec``, ``frame_id``) attributes
works — including real ``sensor_msgs.msg.PointCloud2`` when rclpy is present
and the lightweight :class:`RosPointCloud2` mirror used in tests.

Input policy (mirrors :mod:`foveamap.data.file` conventions):
- ``x``, ``y``, ``z`` must exist as 4- or 8-byte floats. Values are preserved
  bit-for-bit through float32/float64 conversion; no hidden transforms.
- ``intensity`` (any of ``intensity``/``reflectivity``/``amplitude`` names,
  float or unsigned-int storage) is normalized with the existing
  :func:`foveamap.data.file.normalize_intensity` policy. When absent, ones
  are used (the documented core default) and provenance records it.
- ``ring`` (any of ``ring``/``laser``/``channel`` names, integer or float
  storage) defaults to zeros with ``ring_available=False`` when absent; ring
  IDs are never invented beyond the zero-fill convention.
- Only little-endian clouds are supported (ROS 2 tier-1 platforms are
  little-endian); big-endian input is rejected explicitly, never silently
  byte-swapped.
- Malformed clouds (bad steps, truncated data, missing xyz, NaN/Inf in xyz
  when ``reject_invalid=True``) raise :class:`DataAdapterError`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import DataAdapterError
from foveamap.data.file import normalize_intensity

# sensor_msgs/PointField datatype constants (rep-0145 layout).
INT8 = 1
UINT8 = 2
INT16 = 3
UINT16 = 4
INT32 = 5
UINT32 = 6
FLOAT32 = 7
FLOAT64 = 8

_DTYPE_OF = {
    INT8: ("b", np.int8),
    UINT8: ("B", np.uint8),
    INT16: ("h", np.int16),
    UINT16: ("H", np.uint16),
    INT32: ("i", np.int32),
    UINT32: ("I", np.uint32),
    FLOAT32: ("f", np.float32),
    FLOAT64: ("d", np.float64),
}

INTENSITY_NAMES = ("intensity", "reflectivity", "amplitude")
RING_NAMES = ("ring", "laser", "channel")
TIME_NAMES = ("time", "timestamp", "t", "stamps")


@dataclass
class RosStamp:
    sec: int = 0
    nanosec: int = 0

    def to_seconds(self) -> float:
        return float(self.sec) + float(self.nanosec) * 1e-9


@dataclass
class RosHeader:
    stamp: RosStamp = field(default_factory=RosStamp)
    frame_id: str = ""


@dataclass
class RosPointField:
    name: str
    offset: int
    datatype: int
    count: int = 1


@dataclass
class RosPointCloud2:
    """Minimal structural mirror of sensor_msgs/PointCloud2 for tests/tools."""

    fields: list[RosPointField] = field(default_factory=list)
    height: int = 1
    width: int = 0
    point_step: int = 0
    row_step: int = 0
    data: bytes = b""
    is_dense: bool = True
    header: RosHeader = field(default_factory=RosHeader)
    is_bigendian: bool = False


def stamp_to_seconds(stamp: Any) -> float:
    """Convert a ROS stamp (sec/nanosec attributes) to float seconds."""
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataAdapterError(f"Invalid ROS timestamp {stamp!r}: {exc}") from exc


def _field_map(msg: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in msg.fields:
        out[str(getattr(f, "name", "")).lower()] = f
    return out


def _read_channel(raw: np.ndarray, spec: Any, n_pts: int) -> np.ndarray:
    """Extract one logical channel from the raw structured record array."""
    dtype_id = int(getattr(spec, "datatype", 0))
    count = int(getattr(spec, "count", 1))
    if dtype_id not in _DTYPE_OF:
        raise DataAdapterError(f"Unsupported PointCloud2 datatype {dtype_id} for field {spec.name!r}")
    fmt, np_dtype = _DTYPE_OF[dtype_id]
    name = str(getattr(spec, "name", ""))
    if count == 1:
        col = raw[name].astype(np_dtype)
    else:
        col = np.asarray(raw[name]).reshape(n_pts, count)[:, 0].astype(np_dtype)
    return np.ascontiguousarray(col)


def cloud_to_arrays(
    msg: Any,
    *,
    reject_invalid: bool = True,
    intensity_mode: str = "auto",
) -> dict[str, Any]:
    """Parse PointCloud2 payload into ``pts``/``intensity``/``ring`` arrays.

    Returns a dict with ``pts`` (N,3 float32), ``intensity`` (N float32),
    ``ring`` (N int16), ``header`` info, and provenance flags. Never invents
    geometry; numerical xyz values are preserved exactly.
    """
    try:
        height = int(msg.height)
        width = int(msg.width)
        point_step = int(msg.point_step)
        row_step = int(msg.row_step)
        data = bytes(msg.data)
        bigendian = bool(getattr(msg, "is_bigendian", False))
        header = msg.header
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataAdapterError(f"Malformed PointCloud2 message: {exc}") from exc
    if bigendian:
        raise DataAdapterError("Big-endian PointCloud2 is not supported; only little-endian clouds are accepted")
    if height < 0 or width < 0 or point_step <= 0:
        raise DataAdapterError(f"Invalid PointCloud2 geometry: height={height} width={width} point_step={point_step}")
    n_pts = height * width
    if n_pts == 0:
        return {
            "pts": np.zeros((0, 3), dtype=np.float32),
            "intensity": np.zeros((0,), dtype=np.float32),
            "ring": np.zeros((0,), dtype=np.int16),
            "time_offsets": None,
            "time_provenance": "empty",
            "intensity_provenance": "empty",
            "ring_available": False,
            "timestamp": stamp_to_seconds(header.stamp),
            "frame_id": str(getattr(header, "frame_id", "")),
        }
    if row_step < width * point_step:
        raise DataAdapterError(f"PointCloud2 row_step {row_step} smaller than width*point_step {width * point_step}")
    if len(data) < height * row_step:
        raise DataAdapterError(
            f"Truncated PointCloud2: {len(data)} bytes < {height * row_step} expected "
            f"({height}x{width}, step {point_step})"
        )

    fields = _field_map(msg)
    for axis in ("x", "y", "z"):
        if axis not in fields:
            raise DataAdapterError(f"PointCloud2 missing required '{axis}' field; have {sorted(fields)}")
        if int(fields[axis].datatype) not in (FLOAT32, FLOAT64):
            raise DataAdapterError(f"PointCloud2 '{axis}' must be float32/float64, got datatype {fields[axis].datatype}")

    names = [str(getattr(f, "name", "")) for f in msg.fields]
    if len(set(names)) != len(names):
        raise DataAdapterError(f"PointCloud2 has duplicate field names: {names}")
    if any(not nm for nm in names):
        raise DataAdapterError("PointCloud2 has an empty field name")
    # Honor the wire layout exactly: offsets and point_step define placement,
    # so padded / shuffled / non-contiguous Velodyne/Ouster layouts parse
    # correctly instead of being misread as packed sequential fields.
    layout: list[tuple[str, Any, int, int, int]] = []  # name, dtype, offset, count, size
    for f in msg.fields:
        dtype_id = int(getattr(f, "datatype", 0))
        count = int(getattr(f, "count", 1))
        offset = int(getattr(f, "offset", -1))
        if dtype_id not in _DTYPE_OF or count < 1:
            raise DataAdapterError(f"Unsupported PointCloud2 field {f.name!r}: datatype={dtype_id} count={count}")
        if offset < 0:
            raise DataAdapterError(f"PointCloud2 field {f.name!r} has invalid offset {offset}")
        _, np_dtype = _DTYPE_OF[dtype_id]
        size = int(np.dtype(np_dtype).itemsize) * count
        if offset + size > point_step:
            raise DataAdapterError(
                f"PointCloud2 field {f.name!r} spans bytes [{offset}, {offset + size}) "
                f"beyond point_step {point_step}"
            )
        layout.append((str(getattr(f, "name", "")), np_dtype, offset, count, size))
    layout.sort(key=lambda e: e[2])
    for (_, _, off_a, _, size_a), (_, _, off_b, _, _) in zip(layout, layout[1:]):
        if off_b < off_a + size_a:
            raise DataAdapterError(
                f"PointCloud2 fields overlap at byte offset {off_b} "
                f"(previous field ends at {off_a + size_a})"
            )
    descr = {
        "names": [e[0] for e in layout],
        "formats": [(e[1], (e[3],)) if e[3] > 1 else e[1] for e in layout],
        "offsets": [e[2] for e in layout],
        "itemsize": point_step,
    }
    rows = []
    try:
        record_dtype = np.dtype(descr)
        for r in range(height):
            base = r * row_step
            if base + width * point_step > len(data):
                raise DataAdapterError(
                    f"Truncated PointCloud2 row {r}: need {width * point_step} bytes at offset {base}, "
                    f"have {len(data) - base}"
                )
            rows.append(np.frombuffer(data, dtype=record_dtype, count=width, offset=base))
        record = np.stack(rows) if rows else np.zeros((height, width), dtype=record_dtype)
    except DataAdapterError:
        raise
    except (ValueError, TypeError) as exc:
        raise DataAdapterError(f"PointCloud2 buffer does not match field layout: {exc}") from exc
    flat = record.reshape(n_pts)

    pts = np.column_stack([
        np.asarray(flat["x"], dtype=np.float64),
        np.asarray(flat["y"], dtype=np.float64),
        np.asarray(flat["z"], dtype=np.float64),
    ]).astype(np.float32)

    inten_name = next((k for k in INTENSITY_NAMES if k in fields), None)
    if inten_name is None:
        intensity = np.ones(n_pts, dtype=np.float32)
        inten_prov: str = "default_ones_missing_field"
    else:
        raw_inten = _read_channel(flat, fields[inten_name], n_pts).astype(np.float32)
        intensity, inten_prov = normalize_intensity(raw_inten, mode=intensity_mode)

    ring_name = next((k for k in RING_NAMES if k in fields), None)
    if ring_name is None:
        ring = np.zeros(n_pts, dtype=np.int16)
        ring_available = False
    else:
        ring = _read_channel(flat, fields[ring_name], n_pts).astype(np.int16)
        ring_available = True

    # Per-point time offsets (seconds, relative to the header stamp): parsed
    # when a time channel exists, validated by the LiDARFrame contract
    # downstream. Absent means no per-point timing (normal processing +
    # provenance, never fabricated de-skew).
    time_name = next((k for k in TIME_NAMES if k in fields), None)
    if time_name is None:
        time_offsets = None
        time_prov = "absent_no_per_point_timing"
    else:
        if int(fields[time_name].datatype) not in (FLOAT32, FLOAT64):
            raise DataAdapterError(
                f"PointCloud2 time channel must be float32/float64, got datatype {fields[time_name].datatype}"
            )
        time_offsets = _read_channel(flat, fields[time_name], n_pts).astype(np.float32)
        time_prov = "ros_time_channel"

    if reject_invalid and n_pts:
        if not np.all(np.isfinite(pts)):
            raise DataAdapterError("PointCloud2 xyz contains NaN or Inf (rejected; set reject_invalid=False to filter upstream)")
        if not np.all(np.isfinite(intensity)):
            raise DataAdapterError("PointCloud2 intensity contains NaN or Inf")
        if time_offsets is not None and not np.all(np.isfinite(time_offsets)):
            raise DataAdapterError("PointCloud2 time channel contains NaN or Inf")

    return {
        "pts": pts,
        "intensity": intensity.astype(np.float32),
        "ring": ring,
        "time_offsets": time_offsets,
        "time_provenance": time_prov,
        "intensity_provenance": inten_prov,
        "ring_available": ring_available,
        "timestamp": stamp_to_seconds(header.stamp),
        "frame_id": str(getattr(header, "frame_id", "")),
    }


def cloud_to_lidar_frame(
    msg: Any,
    *,
    pose: np.ndarray | None = None,
    sensor_origin: Sequence[float] | np.ndarray | None = None,
    source_id: str = "ros/pointcloud",
    reject_invalid: bool = True,
    intensity_mode: str = "auto",
) -> LiDARFrame:
    """Convert PointCloud2 into the canonical :class:`LiDARFrame` contract.

    Points are expected in the vehicle ego frame already (see
    :mod:`foveamap_ros.frames` for the TF path); ``pose`` is the ego-to-world
    SE(3) transform. No coordinate values are altered here.
    """
    arrays = cloud_to_arrays(msg, reject_invalid=reject_invalid, intensity_mode=intensity_mode)
    n = len(arrays["pts"])
    pose_m = np.eye(4, dtype=np.float64) if pose is None else np.asarray(pose, dtype=np.float64)
    origin = (
        np.asarray(sensor_origin, dtype=np.float32).reshape(3)
        if sensor_origin is not None
        else np.array([0.0, 0.0, 1.73], dtype=np.float32)
    )
    return LiDARFrame(
        pts=arrays["pts"],
        intensity=arrays["intensity"],
        ring=arrays["ring"],
        pose=pose_m,
        sensor_origin=origin,
        timestamp=float(arrays["timestamp"]),
        frame_id=str(arrays["frame_id"]) or f"ros_frame_{n}pts",
        source_id=source_id,
        time_offsets=arrays.get("time_offsets"),
        metadata={
            "intensity_provenance": arrays["intensity_provenance"],
            "ring_available": bool(arrays["ring_available"]),
            "timestamp_provenance": "ros_header_stamp",
            "time_provenance": arrays.get("time_provenance", "absent_no_per_point_timing"),
            "origin_provenance": "user_supplied" if sensor_origin is not None else "default_vehicle_mount",
        },
    )


def build_pointcloud2(
    xyz: np.ndarray,
    extra: dict[str, tuple[np.ndarray, int]] | None = None,
    *,
    frame_id: str = "",
    stamp_sec: int = 0,
    stamp_nanosec: int = 0,
) -> RosPointCloud2:
    """Serialize (N,3) float32 points + scalar extra channels to PointCloud2.

    ``extra`` maps field name -> (values (N,) array, datatype constant).
    Used for the ``/foveamap/points_labeled`` payload without requiring ROS.
    """
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    names = ["x", "y", "z"]
    dtypes = [np.float32, np.float32, np.float32]
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    for key, (vals, dtype_id) in (extra or {}).items():
        if dtype_id not in _DTYPE_OF:
            raise DataAdapterError(f"Unsupported extra field datatype {dtype_id} for {key!r}")
        vals = np.asarray(vals).reshape(n).astype(_DTYPE_OF[dtype_id][1])
        names.append(str(key))
        dtypes.append(_DTYPE_OF[dtype_id][1])
        cols.append(vals)
    dtype = np.dtype([(nm, dt) for nm, dt in zip(names, dtypes)])
    rec = np.empty(n, dtype=dtype)
    for nm, col in zip(names, cols):
        rec[nm] = col
    step = int(dtype.itemsize)
    fields = [RosPointField(name=nm, offset=int(dtype.fields[nm][1]), datatype=dt, count=1)
              for nm, dt in zip(names, [FLOAT32, FLOAT32, FLOAT32] + [d for _, d in (extra or {}).values()])]
    return RosPointCloud2(
        fields=fields,
        height=1,
        width=n,
        point_step=step,
        row_step=step * n,
        data=rec.tobytes(),
        is_dense=True,
        header=RosHeader(stamp=RosStamp(sec=int(stamp_sec), nanosec=int(stamp_nanosec)), frame_id=str(frame_id)),
    )


def ros_stamp_from_seconds(timestamp: float) -> RosStamp:
    """Split float seconds into (sec, nanosec) without precision surprises."""
    sec = int(np.floor(float(timestamp)))
    nanosec = int(round((float(timestamp) - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return RosStamp(sec=sec, nanosec=max(0, nanosec))
