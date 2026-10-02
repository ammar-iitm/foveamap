"""Simulator data source adapter for FoveaMap.

Wraps synthetic drives and simulated LiDAR sequences into canonical LiDARFrame contracts.
"""
from __future__ import annotations

from typing import Iterator, Any
import numpy as np

from .source import LiDARSource
from ..core.contracts import LiDARFrame
from ..core.config import SensorConfig
from ..core.exceptions import DataAdapterError
from ..frames import sim_frames
from ..sim import SENSOR_H, simulate_sequence


def _sequence_to_dict(seq: dict[str, Any]) -> dict[str, Any]:
    F = seq["frames"]
    return {
        "range": np.stack([f["range"] for f in F]),
        "xyz": np.stack([f["xyz"] for f in F]),
        "intensity": np.stack([f["intensity"] for f in F]),
        "valid": np.stack([f["valid"] for f in F]),
        "label": np.stack([f["label"] for f in F]),
        "moving": np.stack([f["moving"] for f in F]),
        "ego": np.stack([f["ego"] for f in F]),
        "potholes": np.asarray(seq["scene"].potholes, np.float32),
        "cross_x": np.float32(seq["scene"].cross_x),
    }


class SimulatorSource(LiDARSource):
    """LiDAR source that replays or generates simulated sequences."""

    def __init__(
        self,
        path_or_seq: str | dict[str, Any] | None = None,
        n_steps: int = 15,
        seed: int = 42,
    ) -> None:
        self._path_or_seq = path_or_seq
        self._n_steps = n_steps
        self._seed = seed
        self._idx = 0
        self._frames: list[LiDARFrame] = []
        self._truth: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self._path_or_seq is None:
                seq = simulate_sequence(seed=self._seed, n_frames=self._n_steps)
                seq_dict = _sequence_to_dict(seq)
                raw_frames, self._truth = sim_frames(seq_dict)
            else:
                raw_frames, self._truth = sim_frames(self._path_or_seq)

            self._frames = []
            for i, rf in enumerate(raw_frames):
                frame = LiDARFrame.from_legacy_dict(rf)
                t_val = float(rf.get("meta", {}).get("t", i * 0.1))
                frame_meta = dict(frame.metadata)
                frame_meta["timestamp_provenance"] = "synthetic"
                frame_meta["sensor_origin_provenance"] = "simulated_model_mount"
                frame_meta["evaluation_truth"] = self._truth

                frame = LiDARFrame(
                    pts=frame.pts,
                    intensity=frame.intensity,
                    ring=frame.ring,
                    pose=frame.pose,
                    sensor_origin=frame.sensor_origin,
                    timestamp=t_val,
                    frame_id=f"sim_{i:04d}",
                    source_id="simulator",
                    label=frame.label,
                    moving=frame.moving,
                    prev_sweeps=frame.prev_sweeps,
                    metadata=frame_meta,
                )
                self._frames.append(frame)
        except Exception as exc:
            if isinstance(exc, DataAdapterError):
                raise
            raise DataAdapterError(f"Failed to load or generate simulator frames: {exc}") from exc

    def __iter__(self) -> Iterator[LiDARFrame]:
        for frame in self._frames[self._idx:]:
            self._idx += 1
            yield frame

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> LiDARFrame:
        if index < 0 or index >= len(self._frames):
            raise IndexError(f"Frame index {index} out of range [0, {len(self._frames)})")
        return self._frames[index]

    def reset(self) -> None:
        self._idx = 0

    @property
    def source_id(self) -> str:
        return "simulator"

    @property
    def sensor_config(self) -> SensorConfig:
        return SensorConfig(
            name="sim",
            n_rows=64,
            n_cols=1024,
            sensor_height_m=SENSOR_H,
            hz=10.0,
            source_description="Simulated 64-beam Velodyne LiDAR",
        )

    @property
    def evaluation_truth(self) -> dict[str, Any]:
        """Ground truth scene features (potholes, crosswalk, trajectories) for evaluation."""
        return self._truth

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._truth, frame_count=len(self._frames))
