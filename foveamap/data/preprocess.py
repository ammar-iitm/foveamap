"""Deterministic point cloud preprocessing for LiDARFrame contracts.

Performs geometric filtering, validity checks, range thresholds, and self-hit removal
strictly before perception or mapping stages without altering input frames.
"""
from __future__ import annotations

import numpy as np

from ..core.contracts import LiDARFrame
from ..core.config import PreprocessConfig
from ..core.exceptions import DataAdapterError, NumericalConsistencyError


class LiDARPreprocessor:
    """Configurable, deterministic point cloud preprocessor."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def process(self, frame: LiDARFrame) -> LiDARFrame:
        """Process a LiDARFrame and return a clean, filtered LiDARFrame copy.

        The original frame is strictly unmutated.
        """
        if not self.config.enabled:
            return frame

        pts = frame.pts
        n_orig = len(pts)

        # 1. Finite coordinate check
        finite_mask = np.all(np.isfinite(pts), axis=1) & np.isfinite(frame.intensity)
        if not np.all(finite_mask):
            if not self.config.remove_invalid:
                raise NumericalConsistencyError(
                    f"Frame {frame.frame_id} from {frame.source_id} contains non-finite points"
                )
            keep_mask = finite_mask
        else:
            keep_mask = np.ones(n_orig, dtype=bool)

        # 2. Radial range filtering from sensor origin
        sensor_xyz = frame.sensor_origin
        d_sensor = pts - sensor_xyz
        radial_dist = np.linalg.norm(d_sensor, axis=1)
        range_mask = (radial_dist >= self.config.min_range_m) & (radial_dist <= self.config.max_range_m)
        keep_mask &= range_mask

        # 3. Ego-frame self-hit filtering (xy proximity to sensor origin)
        if self.config.remove_self_hits and self.config.self_hit_radius_m > 0:
            d_xy = np.hypot(pts[:, 0] - sensor_xyz[0], pts[:, 1] - sensor_xyz[1])
            self_hit_mask = d_xy > self.config.self_hit_radius_m
            keep_mask &= self_hit_mask

        # 4. Optional Z bounds in ego frame
        if self.config.z_min_m is not None:
            keep_mask &= (pts[:, 2] >= self.config.z_min_m)
        if self.config.z_max_m is not None:
            keep_mask &= (pts[:, 2] <= self.config.z_max_m)

        # Extract filtered arrays
        pts_filt = pts[keep_mask].astype(np.float32)
        inten_filt = frame.intensity[keep_mask].astype(np.float32)
        ring_filt = frame.ring[keep_mask].astype(np.int16)
        label_filt = frame.label[keep_mask].astype(np.int8) if frame.label is not None else None
        moving_filt = frame.moving[keep_mask].astype(bool) if frame.moving is not None else None

        # Build updated metadata
        meta = dict(frame.metadata)
        meta["preprocessing"] = {
            "original_points": n_orig,
            "retained_points": len(pts_filt),
            "filtered_points": n_orig - len(pts_filt),
        }

        return LiDARFrame(
            pts=pts_filt,
            intensity=inten_filt,
            ring=ring_filt,
            pose=frame.pose.copy(),
            sensor_origin=frame.sensor_origin.copy(),
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            source_id=frame.source_id,
            label=label_filt,
            moving=moving_filt,
            prev_sweeps=frame.prev_sweeps,
            metadata=meta,
        )
