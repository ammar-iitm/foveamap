"""nuScenes (+ nuScenes-lidarseg) loader -> FoveaMap Frames.

Reads the nuScenes JSON tables directly, so the official devkit is optional.
Works with v1.0-mini (10 scenes, ~4 GB) and v1.0-trainval.

Expected layout (what the official tarballs extract to):
    <dataroot>/v1.0-mini/{scene,sample,sample_data,calibrated_sensor,sensor,
                          ego_pose,category,lidarseg,sample_annotation,
                          instance,attribute}.json
    <dataroot>/samples/LIDAR_TOP/*.pcd.bin      keyframes, 2 Hz, labelled
    <dataroot>/sweeps/LIDAR_TOP/*.pcd.bin       intermediate sweeps, 20 Hz
    <dataroot>/lidarseg/v1.0-mini/*_lidarseg.bin

Each .pcd.bin holds float32 (x, y, z, intensity, ring) in the LIDAR_TOP frame.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np

from .sim import (NUM_CLASSES, ROAD, SIDEWALK, PARKING, TERRAIN, VEGETATION, BUILDING,
                  POLE, VEHICLE, PERSON)
from .frames import DatasetInfo, transform

# nuScenes-lidarseg category name -> FoveaMap class. nuScenes has no separate
# pole/sign class (they are inside static.manmade), so the 'pole' slot holds
# barriers and traffic cones, and there is no 'parking' class.
NAME_TO_CLASS = {
    "flat.driveable_surface": ROAD,
    "flat.sidewalk": SIDEWALK,
    "flat.other": SIDEWALK,
    "flat.terrain": TERRAIN,
    "static.vegetation": VEGETATION,
    "static.manmade": BUILDING,
    "static.other": BUILDING,
    "static_object.bicycle_rack": BUILDING,
    "movable_object.debris": BUILDING,
    "movable_object.pushable_pullable": BUILDING,
    "movable_object.barrier": POLE,
    "movable_object.trafficcone": POLE,
}
IGNORE_NAMES = {"noise", "vehicle.ego"}
MOVING_ATTRS = {"vehicle.moving", "pedestrian.moving", "cycle.with_rider"}

# the 32 lidarseg classes in index order (used only if category.json lacks 'index')
LIDARSEG_NAMES = [
    "noise", "animal", "human.pedestrian.adult", "human.pedestrian.child",
    "human.pedestrian.construction_worker", "human.pedestrian.personal_mobility",
    "human.pedestrian.police_officer", "human.pedestrian.stroller", "human.pedestrian.wheelchair",
    "movable_object.barrier", "movable_object.debris", "movable_object.pushable_pullable",
    "movable_object.trafficcone", "static_object.bicycle_rack", "vehicle.bicycle", "vehicle.bus.bendy",
    "vehicle.bus.rigid", "vehicle.car", "vehicle.construction", "vehicle.emergency.ambulance",
    "vehicle.emergency.police", "vehicle.motorcycle", "vehicle.trailer", "vehicle.truck",
    "flat.driveable_surface", "flat.other", "flat.sidewalk", "flat.terrain", "static.manmade",
    "static.other", "static.vegetation", "vehicle.ego",
]

MINI_SPLITS = {
    "mini_train": ["scene-0061", "scene-0553", "scene-0655", "scene-0757", "scene-0796",
                   "scene-1077", "scene-1094", "scene-1100"],
    "mini_val": ["scene-0103", "scene-0916"],
}

NUSC_CLASS_NAMES = ["drivable surface", "sidewalk / other flat", "(unused)", "terrain", "vegetation",
                    "manmade / static", "barrier / cone", "vehicle", "person"]


def nuscenes_info(n_rows=32, source=None):
    active = np.ones(NUM_CLASSES, bool)
    active[PARKING] = False
    return DatasetInfo("nuscenes", n_rows, 1024, class_names=list(NUSC_CLASS_NAMES), active=active,
                       source=source or f"nuScenes {n_rows}-beam Lidar, 2 Hz labelled keyframes", hz=2.0)


def map_name(name: str) -> int:
    if name in IGNORE_NAMES:
        return -1
    if name in NAME_TO_CLASS:
        return NAME_TO_CLASS[name]
    if name.startswith("vehicle."):
        return VEHICLE
    if name.startswith("human.") or name == "animal":
        return PERSON
    return -1


def quat_to_R(q):
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_T(translation, rotation):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(rotation)
    T[:3, 3] = translation
    return T


class NuScenesLite:
    def __init__(self, dataroot, version="v1.0-mini", verbose=True):
        self.root, self.version = dataroot, version
        tdir = os.path.join(dataroot, version)
        if not os.path.isdir(tdir):
            raise FileNotFoundError(f"{tdir} not found. Extract the nuScenes tarball into {dataroot}.")

        def load(name, required=True):
            p = os.path.join(tdir, name + ".json")
            if not os.path.exists(p):
                if required:
                    raise FileNotFoundError(p)
                return []
            with open(p) as fh:
                return json.load(fh)

        by = lambda rows: {r["token"]: r for r in rows}  # noqa: E731
        self.scene = load("scene")
        self.sample = by(load("sample"))
        self.sample_data = by(load("sample_data"))
        self.calib = by(load("calibrated_sensor"))
        self.sensor = by(load("sensor"))
        self.ego_pose = by(load("ego_pose"))
        self.category = load("category")
        self.attribute = by(load("attribute", False))
        self.instance = by(load("instance", False))
        anns = load("sample_annotation", False)
        lidarseg = load("lidarseg", False)
        if not lidarseg:
            raise FileNotFoundError(f"{tdir}/lidarseg.json missing: extract nuScenes-lidarseg into {dataroot}.")
        self.lidarseg = {r["sample_data_token"]: r for r in lidarseg}
        self.ann_by_sample = {}
        self.ann = by(anns)
        for a in anns:
            self.ann_by_sample.setdefault(a["sample_token"], []).append(a)
        # lidarseg index -> FoveaMap class
        self.lut = np.full(256, -1, np.int8)
        if all("index" in c for c in self.category):
            for c in self.category:
                self.lut[c["index"]] = map_name(c["name"])
        else:
            for i, n in enumerate(LIDARSEG_NAMES):
                self.lut[i] = map_name(n)
        self.cat_of_instance = {}
        cat_by_token = {c["token"]: c["name"] for c in self.category}
        for tok, inst in self.instance.items():
            self.cat_of_instance[tok] = cat_by_token.get(inst["category_token"], "")
        # keyframe LIDAR_TOP sample_data per sample
        self.lidar_of_sample = {}
        for sd in self.sample_data.values():
            ch = self.sensor[self.calib[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            sd["_channel"] = ch
            if ch == "LIDAR_TOP" and sd.get("is_key_frame"):
                self.lidar_of_sample[sd["sample_token"]] = sd["token"]
        if verbose:
            print(f"nuScenes {version}: {len(self.scene)} scenes, {len(self.sample)} samples, "
                  f"{len(self.lidarseg)} lidarseg files")

    # ---------------------------------------------------------------- tables
    def scene_names(self, split=None):
        names = [s["name"] for s in self.scene]
        if split is None:
            return names
        if split in MINI_SPLITS:
            return [n for n in MINI_SPLITS[split] if n in names]
        try:
            from nuscenes.utils.splits import create_splits_scenes
            return [n for n in create_splits_scenes()[split] if n in names]
        except ImportError as e:
            raise ImportError("pip install nuscenes-devkit for trainval splits") from e

    def keyframe_tokens(self, scene_name):
        sc = next(s for s in self.scene if s["name"] == scene_name)
        out, tok = [], sc["first_sample_token"]
        while tok:
            if tok in self.lidar_of_sample:
                out.append(self.lidar_of_sample[tok])
            tok = self.sample[tok]["next"]
        return out

    def sensor_to_ego(self, sd):
        c = self.calib[sd["calibrated_sensor_token"]]
        return make_T(c["translation"], c["rotation"])

    def ego_to_world(self, sd):
        e = self.ego_pose[sd["ego_pose_token"]]
        return make_T(e["translation"], e["rotation"])

    def load_bin(self, sd):
        raw = np.fromfile(os.path.join(self.root, sd["filename"]), dtype=np.float32)
        return raw.reshape(-1, 5)

    # ---------------------------------------------------------------- frames
    def ring_order(self, pts_sensor, ring, n_rows):
        """Map laser id -> range-image row, top beam first (sorted by mean elevation)."""
        elev = np.degrees(np.arctan2(pts_sensor[:, 2], np.hypot(pts_sensor[:, 0], pts_sensor[:, 1])))
        ids = np.arange(n_rows)
        mean = np.array([elev[ring == i].mean() if np.any(ring == i) else -90 + i * 1e-3 for i in ids])
        order = np.argsort(-mean)
        lut = np.zeros(max(n_rows, int(ring.max()) + 1), np.int16)
        lut[order] = np.arange(n_rows)
        return lut

    def moving_mask(self, sample_token, pts_ego, T_ew):
        """Points inside boxes of moving objects (attribute or speed > 0.5 m/s)."""
        mov = np.zeros(len(pts_ego), bool)
        T_we = np.linalg.inv(T_ew)
        for a in self.ann_by_sample.get(sample_token, []):
            names = {self.attribute[t]["name"] for t in a.get("attribute_tokens", []) if t in self.attribute}
            moving = bool(names & MOVING_ATTRS)
            if not moving:
                v = self.velocity(a)
                moving = v is not None and v > 0.5
            if not moving:
                continue
            Tb = T_we @ make_T(a["translation"], a["rotation"])      # box -> ego
            local = transform(np.linalg.inv(Tb), pts_ego)
            w, l, h = a["size"]
            tol = 0.1
            inside = (np.abs(local[:, 0]) <= l / 2 + tol) & (np.abs(local[:, 1]) <= w / 2 + tol) & (np.abs(local[:, 2]) <= h / 2 + tol)
            mov |= inside
        return mov

    def velocity(self, a):
        prev, nxt = a.get("prev", ""), a.get("next", "")
        if not prev and not nxt:
            return None
        a0 = self.ann[prev] if prev else a
        a1 = self.ann[nxt] if nxt else a
        dt = (self.sample[a1["sample_token"]]["timestamp"] - self.sample[a0["sample_token"]]["timestamp"]) / 1e6
        if dt <= 0:
            return None
        d = np.subtract(a1["translation"], a0["translation"])[:2]
        return float(np.linalg.norm(d) / dt)

    def frame(self, sd_token, info: DatasetInfo, prev_steps=(2, 4), remove_close=1.0):
        """One labelled keyframe as a FoveaMap Frame. prev_steps counts 20 Hz
        sweeps back: 2 -> 0.1 s, 4 -> 0.2 s (the motion cue the model expects)."""
        sd = self.sample_data[sd_token]
        raw = self.load_bin(sd)
        ring_raw = raw[:, 4].astype(np.int64)
        lab_raw = np.fromfile(os.path.join(self.root, self.lidarseg[sd_token]["filename"]), dtype=np.uint8)
        assert len(lab_raw) == len(raw), f"label/point count mismatch for {sd_token}"
        keep = np.hypot(raw[:, 0], raw[:, 1]) > remove_close
        raw, ring_raw, lab_raw = raw[keep], ring_raw[keep], lab_raw[keep]
        lut_ring = self.ring_order(raw[:, :3], ring_raw, info.n_rows)
        T_se, T_ew = self.sensor_to_ego(sd), self.ego_to_world(sd)
        pts = transform(T_se, raw[:, :3].astype(np.float64)).astype(np.float32)
        inten = raw[:, 3].astype(np.float32)
        inten = inten / 255.0 if inten.max() > 1.5 else inten
        label = self.lut[lab_raw]
        moving = self.moving_mask(sd["sample_token"], pts, T_ew) & np.isin(label, (VEHICLE, PERSON))
        prev = []
        for k in prev_steps:
            p = sd
            for _ in range(k):
                p = self.sample_data[p["prev"]] if p.get("prev") else None
                if p is None:
                    break
            if p is None:
                prev.append(None)
                continue
            r = self.load_bin(p)
            r = r[np.hypot(r[:, 0], r[:, 1]) > remove_close]
            pw = transform(self.ego_to_world(p) @ self.sensor_to_ego(p), r[:, :3].astype(np.float64))
            rr = r[:, 4].astype(np.int64)
            prev.append((pw.astype(np.float32), lut_ring[np.clip(rr, 0, len(lut_ring) - 1)].astype(np.int16)))
        return dict(
            pts=pts, inten=inten.astype(np.float32),
            ring=lut_ring[np.clip(ring_raw, 0, len(lut_ring) - 1)].astype(np.int16),
            label=label.astype(np.int8), moving=moving, pose=T_ew,
            sensor=T_se[:3, 3].astype(np.float32), prev=prev,
            meta=dict(token=sd_token, timestamp=sd["timestamp"], sample=sd["sample_token"]),
        )

    def n_rings(self):
        """Number of lasers, read from the first LIDAR_TOP keyframe (32 for nuScenes)."""
        tok = next(iter(self.lidar_of_sample.values()))
        return int(self.load_bin(self.sample_data[tok])[:, 4].max()) + 1

    def scene_frames(self, scene_name, info=None):
        info = info or nuscenes_info(self.n_rings())
        return [self.frame(t, info) for t in self.keyframe_tokens(scene_name)]


def build_cache(dataroot, out_dir, version="v1.0-mini", splits=("mini_train", "mini_val")):
    """Convert scenes to cached Frame lists (one pickle per scene)."""
    nusc = NuScenesLite(dataroot, version)
    info = nuscenes_info(nusc.n_rings())
    os.makedirs(out_dir, exist_ok=True)
    index = {"n_rows": info.n_rows, "version": version}
    for split in splits:
        index[split] = []
        for name in nusc.scene_names(split):
            path = os.path.join(out_dir, f"{name}.pkl")
            if not os.path.exists(path):
                frames = nusc.scene_frames(name, info)
                with open(path, "wb") as fh:
                    pickle.dump(frames, fh, protocol=4)
                n = len(frames)
            else:
                n = -1
            index[split].append(name)
            print(f"  {split}: {name} ({n if n >= 0 else 'cached'} frames)", flush=True)
    with open(os.path.join(out_dir, "index.json"), "w") as fh:
        json.dump(index, fh)
    return index


def cached_info(out_dir):
    with open(os.path.join(out_dir, "index.json")) as fh:
        return nuscenes_info(json.load(fh)["n_rows"])


def load_cached(out_dir, split):
    with open(os.path.join(out_dir, "index.json")) as fh:
        index = json.load(fh)
    frames = []
    for name in index[split]:
        with open(os.path.join(out_dir, f"{name}.pkl"), "rb") as fh:
            frames.extend(pickle.load(fh))
    return frames


def load_scene(out_dir, name):
    with open(os.path.join(out_dir, f"{name}.pkl"), "rb") as fh:
        return pickle.load(fh)
