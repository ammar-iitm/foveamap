"""SemanticKITTI data source adapter for FoveaMap.

Converts SemanticKITTI sequences, poses, and calibrations into canonical LiDARFrame contracts.
"""
from __future__ import annotations

import os
from typing import Iterator, Any
import numpy as np

from .source import LiDARSource
from ..core.contracts import LiDARFrame
from ..core.config import SensorConfig
from ..core.exceptions import DataAdapterError
from ..semantickitti import SemanticKITTI, kitti_info, selected_scans
from ..sim import SENSOR_H, N_BEAMS, FOV_UP, FOV_DOWN


class SemanticKITTISource(LiDARSource):
    """LiDARSource adapter for SemanticKITTI odometry and segmentation sequences."""

    def __init__(
        self,
        root: str,
        sequence: str = "08",
        stride: int = 1,
        start: int = 0,
        remove_close: float = 0.0,
    ) -> None:
        self.root = root
        self.sequence = sequence
        self.stride = stride
        self.start = start
        self.remove_close = remove_close
        self._idx = 0

        self._ds = SemanticKITTI(root)
        self._seq_dir = self._ds.seq_dir(sequence)
        if not os.path.isdir(self._seq_dir):
            raise DataAdapterError(
                f"SemanticKITTI sequence directory not found: {self._seq_dir}"
            )

        vel_dir = os.path.join(self._seq_dir, "velodyne")
        if not os.path.isdir(vel_dir):
            raise DataAdapterError(
                f"SemanticKITTI velodyne directory missing: {vel_dir}"
            )

        try:
            self.poses = self._ds.poses_ego(sequence)
        except Exception as exc:
            raise DataAdapterError(
                f"Failed to load poses and calibration for sequence {sequence}: {exc}"
            ) from exc

        # Check for recorded timestamps file (times.txt) in KITTI sequence dir
        times_file = os.path.join(self._seq_dir, "times.txt")
        if os.path.isfile(times_file):
            try:
                self.timestamps = np.loadtxt(times_file, dtype=np.float64)
                self._timestamp_provenance = "recorded_sensor_times_txt"
            except Exception:
                self.timestamps = None
                self._timestamp_provenance = "derived_from_scan_id_10hz"
        else:
            self.timestamps = None
            self._timestamp_provenance = "derived_from_scan_id_10hz"

        # Find total scans
        bin_files = sorted([f for f in os.listdir(vel_dir) if f.endswith(".bin")])
        if not bin_files:
            raise DataAdapterError(f"No .bin point cloud files found in {vel_dir}")

        self.scans = selected_scans(len(bin_files), stride=stride, start=start)
        self.info = kitti_info()

    def __len__(self) -> int:
        return len(self.scans)

    def __iter__(self) -> Iterator[LiDARFrame]:
        while self._idx < len(self.scans):
            frame = self[self._idx]
            self._idx += 1
            yield frame

    def __getitem__(self, index: int) -> LiDARFrame:
        if index < 0 or index >= len(self.scans):
            raise IndexError(f"Scan index {index} out of range [0, {len(self.scans)})")

        scan_id = self.scans[index]
        scan_bin_path = self._ds.scan_path(self.sequence, scan_id)
        if not os.path.isfile(scan_bin_path):
            raise DataAdapterError(f"SemanticKITTI scan file missing: {scan_bin_path}")

        try:
            raw_dict = self._ds.frame(
                self.sequence,
                scan_id,
                self.poses,
                self.info,
                remove_close=self.remove_close,
            )
            frame = LiDARFrame.from_legacy_dict(raw_dict)

            if self.timestamps is not None and scan_id < len(self.timestamps):
                ts = float(self.timestamps[scan_id])
            else:
                ts = float(scan_id * 0.1)

            meta = dict(frame.metadata, sequence=self.sequence, scan_id=scan_id)
            meta["timestamp_provenance"] = self._timestamp_provenance
            meta["sensor_origin_provenance"] = "dataset_calibrated_mount"

            return LiDARFrame(
                pts=frame.pts,
                intensity=frame.intensity,
                ring=frame.ring,
                pose=frame.pose,
                sensor_origin=frame.sensor_origin,
                timestamp=ts,
                frame_id=f"{self.sequence}_{scan_id:06d}",
                source_id=self.source_id,
                label=frame.label,
                moving=frame.moving,
                prev_sweeps=frame.prev_sweeps,
                metadata=meta,
            )
        except Exception as exc:
            if isinstance(exc, DataAdapterError):
                raise
            raise DataAdapterError(
                f"Failed to read SemanticKITTI frame {scan_id} in sequence {self.sequence}: {exc}"
            ) from exc

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        return f"semantickitti/{self.sequence}"

    @property
    def sensor_config(self) -> SensorConfig:
        return SensorConfig(
            name="hdl64e",
            n_rows=N_BEAMS,
            n_cols=1024,
            fov_up_deg=FOV_UP,
            fov_down_deg=FOV_DOWN,
            sensor_height_m=SENSOR_H,
            hz=10.0,
            source_description=f"Velodyne HDL-64E from SemanticKITTI sequence {self.sequence}",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "sequence": self.sequence,
            "total_scans": len(self.scans),
            "stride": self.stride,
        }
