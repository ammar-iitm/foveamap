"""Write a simulated drive in the exact SemanticKITTI file layout.

Used to test the SemanticKITTI loader without the real data. As in KITTI,
points are stored in the Lidar frame, poses are cam0 poses (y down, z forward)
relative to the first scan, and a realistic velodyne -> cam0 calibration sits
in between, so the loader has to undo all three to recover world points.
"""
import os

import numpy as np

from foveamap.sim import SENSOR_H

# FoveaMap class -> a SemanticKITTI raw id (road, sidewalk, parking, terrain, vegetation, building, pole, vehicle, person)
CLASS_TO_RAW = np.array([40, 48, 44, 72, 70, 50, 80, 10, 30], np.uint32)
MOVING_RAW = {7: 252, 8: 254}                     # moving car, moving person

# velodyne -> cam0 (KITTI-like: x_cam = -y_velo, y_cam = -z_velo, z_cam = x_velo, plus an offset)
TR = np.array([[0.0, -1.0, 0.0, -0.004],
               [0.0, 0.0, -1.0, -0.076],
               [1.0, 0.0, 0.0, -0.272],
               [0.0, 0.0, 0.0, 1.0]])


def write_mock(root, frames, seq="08"):
    """frames: simulator Frames (sim_frames output). Returns the world offset (sim world of scan 0's ego)."""
    d = os.path.join(root, "dataset", "sequences", seq)
    os.makedirs(os.path.join(d, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    S = np.eye(4)
    S[2, 3] = SENSOR_H
    T0 = frames[0]["pose"]
    poses = []
    for i, f in enumerate(frames):
        pts_velo = f["pts"] - np.array([0, 0, SENSOR_H], np.float32)
        order = np.random.default_rng(i).permutation(len(pts_velo))      # KITTI order carries no ring id
        scan = np.c_[pts_velo, f["inten"]].astype(np.float32)[order]
        raw = np.where(f["label"] >= 0, CLASS_TO_RAW[np.clip(f["label"], 0, 8)], 0).astype(np.uint32)
        for c, r in MOVING_RAW.items():
            raw[(f["label"] == c) & f["moving"]] = r
        raw = raw | (np.uint32(7) << 16)                                 # instance ids live in the high bits
        scan.tofile(os.path.join(d, "velodyne", f"{i:06d}.bin"))
        raw[order].tofile(os.path.join(d, "labels", f"{i:06d}.label"))
        T_wv = np.linalg.inv(S) @ np.linalg.inv(T0) @ f["pose"] @ S       # velodyne -> world (velodyne of scan 0)
        poses.append((TR @ T_wv @ np.linalg.inv(TR))[:3, :4])            # as a cam0 pose
    np.savetxt(os.path.join(d, "poses.txt"), np.array(poses).reshape(len(poses), 12), fmt="%.9e")
    with open(os.path.join(d, "calib.txt"), "w") as fh:
        fh.write("P0: " + " ".join(["0"] * 12) + "\n")
        fh.write("Tr: " + " ".join(f"{v:.9e}" for v in TR[:3, :4].ravel()) + "\n")
    return T0
