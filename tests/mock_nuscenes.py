"""Write simulator drives in the exact nuScenes + lidarseg file layout.

Used to test the nuScenes loader without the real data. The mock is made
deliberately awkward: the world is rotated 35 deg and shifted, the Lidar is
mounted rotated -90 deg in the ego frame, laser ids are shuffled, and moving
flags must be recovered from annotation boxes + attributes.
"""
import json
import os
import sys
import uuid

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from foveamap.sim import (simulate_sequence, VEHICLE, PERSON, SENSOR_H, N_BEAMS)  # noqa: E402
from foveamap.nuscenes import LIDARSEG_NAMES  # noqa: E402

SIM_TO_NUSC = np.array([24, 26, 24, 27, 30, 28, 28, 17, 2])     # sim class -> lidarseg index
YAW = np.deg2rad(35.0)
SHIFT = np.array([400.0, 1200.0, 0.0])
SENSOR_YAW = np.deg2rad(-90.0)
RING_PERM = np.argsort(np.r_[np.arange(0, N_BEAMS, 2), np.arange(1, N_BEAMS, 2)])   # interleaved laser ids


def tok():
    return uuid.uuid4().hex


def quat_yaw(a):
    return [float(np.cos(a / 2)), 0.0, 0.0, float(np.sin(a / 2))]


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def G(p):
    return p @ Rz(YAW).T + SHIFT


def boxes_at(scene, t):
    """(instance, category, center_sim, size[w,l,h], moving, vxy) for vehicles and people at time t."""
    out = {}
    for b in scene.boxes:
        x0, y0, z0, x1, y1, z1, cls, inst, vx, vy = b
        if int(cls) != VEHICLE:
            continue
        x0, x1 = x0 + vx * t, x1 + vx * t
        y0, y1 = y0 + vy * t, y1 + vy * t
        if inst in out:
            a = out[inst]
            x0, y0, z0 = min(x0, a[0]), min(y0, a[1]), min(z0, a[2])
            x1, y1, z1 = max(x1, a[3]), max(y1, a[4]), max(z1, a[5])
        out[inst] = [x0, y0, z0, x1, y1, z1, abs(vx) + abs(vy) > 0.3]
    res = []
    for inst, (x0, y0, z0, x1, y1, z1, mv) in out.items():
        res.append((int(inst), "vehicle.car", np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]),
                    [y1 - y0, x1 - x0, z1 - z0], mv))
    for c in scene.cyls:
        x, y, r, z0, z1, cls, inst, vx, vy = c
        if int(cls) != PERSON:
            continue
        res.append((int(inst), "human.pedestrian.adult", np.array([x + vx * t, y + vy * t, (z0 + z1) / 2]),
                    [2 * r, 2 * r, z1 - z0], abs(vx) + abs(vy) > 0.3))
    return res


def write_mock(root, scenes=(("scene-0061", 11), ("scene-0553", 12), ("scene-0103", 13)), n_sweeps=35, version="v1.0-mini"):
    tdir = os.path.join(root, version)
    for d in (tdir, os.path.join(root, "samples/LIDAR_TOP"), os.path.join(root, "sweeps/LIDAR_TOP"),
              os.path.join(root, "lidarseg", version)):
        os.makedirs(d, exist_ok=True)
    T = {k: [] for k in ["scene", "sample", "sample_data", "calibrated_sensor", "sensor", "ego_pose", "category",
                         "attribute", "instance", "sample_annotation", "lidarseg", "log"]}
    cat_tok = {n: tok() for n in LIDARSEG_NAMES}
    T["category"] = [dict(token=cat_tok[n], name=n, description="", index=i) for i, n in enumerate(LIDARSEG_NAMES)]
    attr_tok = {n: tok() for n in ["vehicle.moving", "vehicle.parked", "pedestrian.moving", "pedestrian.standing"]}
    T["attribute"] = [dict(token=t, name=n, description="") for n, t in attr_tok.items()]
    sensor_tok, cs_tok, log_tok = tok(), tok(), tok()
    T["sensor"] = [dict(token=sensor_tok, channel="LIDAR_TOP", modality="lidar")]
    T["calibrated_sensor"] = [dict(token=cs_tok, sensor_token=sensor_tok, translation=[0.0, 0.0, SENSOR_H],
                                   rotation=quat_yaw(SENSOR_YAW), camera_intrinsic=[])]
    T["log"] = [dict(token=log_tok, logfile="mock", vehicle="sim", date_captured="2026-09-30", location="sim")]
    truth = {}
    ts0 = 1_600_000_000_000_000
    for name, seed in scenes:
        seq = simulate_sequence(seed, n_frames=n_sweeps, dt=0.05)
        scene_tok = tok()
        key_idx = list(range(4, n_sweeps, 10))
        sample_toks = [tok() for _ in key_idx]
        inst_tok = {}
        prev_sd = ""
        sd_toks = [tok() for _ in range(n_sweeps)]
        for k, f in enumerate(seq["frames"]):
            ts = ts0 + k * 50_000
            ego = f["ego"].astype(np.float64)
            ep_tok = tok()
            T["ego_pose"].append(dict(token=ep_tok, timestamp=ts, rotation=quat_yaw(YAW),
                                      translation=list(map(float, G(ego[None])[0]))))
            v = f["valid"].reshape(-1)
            p_e = f["xyz"].reshape(-1, 3)[v].astype(np.float64)
            p_s = (p_e - np.array([0, 0, SENSOR_H])) @ Rz(SENSOR_YAW)          # = R^T (p - t)
            beam = np.nonzero(f["valid"])[0]
            raw = np.c_[p_s, f["intensity"].reshape(-1)[v] * 255, RING_PERM[beam]].astype(np.float32)
            is_key = k in key_idx
            folder = "samples" if is_key else "sweeps"
            fn = f"{folder}/LIDAR_TOP/{name}_{k:03d}.pcd.bin"
            raw.tofile(os.path.join(root, fn))
            si = min(range(len(key_idx)), key=lambda i: abs(key_idx[i] - k))
            T["sample_data"].append(dict(token=sd_toks[k], sample_token=sample_toks[si], ego_pose_token=ep_tok,
                                         calibrated_sensor_token=cs_tok, timestamp=ts, fileformat="pcd",
                                         is_key_frame=is_key, height=0, width=0, filename=fn,
                                         prev=prev_sd, next=sd_toks[k + 1] if k + 1 < n_sweeps else ""))
            prev_sd = sd_toks[k]
            if is_key:
                i = key_idx.index(k)
                T["sample"].append(dict(token=sample_toks[i], timestamp=ts, scene_token=scene_tok,
                                        prev=sample_toks[i - 1] if i else "",
                                        next=sample_toks[i + 1] if i + 1 < len(key_idx) else ""))
                lab = SIM_TO_NUSC[f["label"].reshape(-1)[v].astype(int)].astype(np.uint8)
                lfn = f"lidarseg/{version}/{sd_toks[k]}_lidarseg.bin"
                lab.tofile(os.path.join(root, lfn))
                T["lidarseg"].append(dict(token=tok(), sample_data_token=sd_toks[k], filename=lfn))
                t_now = k * 0.05
                for inst, cat, c, wlh, mv in boxes_at(seq["scene"], t_now):
                    if np.hypot(*(c[:2] - ego[:2])) > 70:
                        continue
                    if inst not in inst_tok:
                        inst_tok[inst] = tok()
                        T["instance"].append(dict(token=inst_tok[inst], category_token=cat_tok[cat]))
                    attr = ("vehicle." if cat.startswith("vehicle") else "pedestrian.") + \
                           ("moving" if mv else ("parked" if cat.startswith("vehicle") else "standing"))
                    T["sample_annotation"].append(dict(
                        token=tok(), sample_token=sample_toks[i], instance_token=inst_tok[inst],
                        attribute_tokens=[attr_tok[attr]], visibility_token="", prev="", next="",
                        translation=list(map(float, G(c[None])[0])), size=list(map(float, wlh)),
                        rotation=quat_yaw(YAW), num_lidar_pts=0, num_radar_pts=0))
                truth[sd_toks[k]] = dict(xyz=p_e.astype(np.float32), label=f["label"].reshape(-1)[v],
                                         moving=f["moving"].reshape(-1)[v], beam=beam, ego=ego)
        T["scene"].append(dict(token=scene_tok, name=name, description="mock", log_token=log_tok,
                               nbr_samples=len(key_idx), first_sample_token=sample_toks[0],
                               last_sample_token=sample_toks[-1]))
    for k, rows in T.items():
        with open(os.path.join(tdir, k + ".json"), "w") as fh:
            json.dump(rows, fh)
    return truth


if __name__ == "__main__":
    import pickle
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock_nuscenes"
    truth = write_mock(root)
    with open(os.path.join(root, "truth.pkl"), "wb") as fh:
        pickle.dump(truth, fh)
    print("mock written to", root, "keyframes:", len(truth))
