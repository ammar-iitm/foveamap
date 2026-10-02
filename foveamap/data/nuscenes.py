"""nuScenes data source adapter for FoveaMap.

Converts nuScenes scenes, calibration, and lidarseg labels into canonical LiDARFrame contracts.
"""
from __future__ import annotations

import os
from typing import Iterator, Any
import numpy as np

from .source import LiDARSource
from ..core.contracts import LiDARFrame
from ..core.config import SensorConfig
from ..core.exceptions import DataAdapterError
from ..nuscenes import NuScenesLite, nuscenes_info


class NuScenesSource(LiDARSource):
    """LiDARSource adapter for nuScenes keyframe scenes and lidarseg."""

    def __init__(
        self,
        root: str,
        version: str = "v1.0-mini",
        scene_name: str | None = None,
        split: str | None = None,
        remove_close: float = 0.0,
    ) -> None:
        self.root = root
        self.version = version
        self.split = split
        self.remove_close = remove_close
        self._idx = 0

        try:
            self._nusc = NuScenesLite(root, version=version, verbose=False)
        except Exception as exc:
            raise DataAdapterError(
                f"Failed to initialize nuScenes reader at {root} (version {version}): {exc}"
            ) from exc

        available_scenes = self._nusc.scene_names(split=split)
        if not available_scenes:
            raise DataAdapterError(
                f"No scenes found for nuScenes split '{split}' at {root}"
            )

        if scene_name is None:
            self.scene_name = available_scenes[0]
        else:
            if scene_name not in available_scenes:
                raise DataAdapterError(
                    f"Scene '{scene_name}' not found among available scenes: {available_scenes}"
                )
            self.scene_name = scene_name

        self.tokens = self._nusc.keyframe_tokens(self.scene_name)
        if not self.tokens:
            raise DataAdapterError(
                f"No keyframe LiDAR_TOP tokens found for scene '{self.scene_name}'"
            )

        n_rings = self._nusc.n_rings()
        self.info = nuscenes_info(n_rows=n_rings)

    def __len__(self) -> int:
        return len(self.tokens)

    def __iter__(self) -> Iterator[LiDARFrame]:
        while self._idx < len(self.tokens):
            frame = self[self._idx]
            self._idx += 1
            yield frame

    def __getitem__(self, index: int) -> LiDARFrame:
        if index < 0 or index >= len(self.tokens):
            raise IndexError(f"Keyframe index {index} out of range [0, {len(self.tokens)})")

        sd_token = self.tokens[index]
        try:
            raw_dict = self._nusc.frame(
                sd_token,
                self.info,
                remove_close=self.remove_close,
            )
            frame = LiDARFrame.from_legacy_dict(raw_dict)
            meta = dict(frame.metadata, scene=self.scene_name)
            meta["timestamp_provenance"] = "recorded_sensor_epoch"
            meta["sensor_origin_provenance"] = "dataset_calibrated_mount"

            return LiDARFrame(
                pts=frame.pts,
                intensity=frame.intensity,
                ring=frame.ring,
                pose=frame.pose,
                sensor_origin=frame.sensor_origin,
                timestamp=frame.timestamp,
                frame_id=sd_token,
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
                f"Failed to load nuScenes keyframe {sd_token} in scene {self.scene_name}: {exc}"
            ) from exc

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        return f"nuscenes/{self.scene_name}"

    @property
    def sensor_config(self) -> SensorConfig:
        return SensorConfig(
            name="nuscenes_32",
            n_rows=self.info.n_rows,
            n_cols=1024,
            hz=2.0,
            source_description=f"nuScenes 32-beam LiDAR keyframe ({self.scene_name})",
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "version": self.version,
            "scene_name": self.scene_name,
            "keyframe_count": len(self.tokens),
        }
