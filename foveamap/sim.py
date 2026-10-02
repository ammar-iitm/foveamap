"""Procedural street scenes + a simulated 64-beam spinning Lidar.

Why a simulator: it gives exact per-point labels (class, moving flag) with
zero download, and we control the things the PRD cares about: 15 cm curbs,
potholes, overhanging structures, parked vs moving vehicles, pedestrians.

Everything is built from three primitive types so ray casting stays
vectorised in NumPy:
  * axis-aligned boxes   (sidewalks, terrain slabs, buildings, walls, cars, signs, gantry)
  * vertical cylinders   (poles, trunks, pedestrians)
  * spheres              (tree canopies)
plus a ground plane z = 0 (road) with circular potholes.

Coordinates: world frame, x along the main road, z up, metres.
The ego vehicle drives along +x in the right lane (y = -2.5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# ----------------------------------------------------------------------------
# Taxonomy (canonical 9 classes defined in core.ontology)
# ----------------------------------------------------------------------------
from .core.ontology import (
    CANONICAL_CLASSES,
    ROAD,
    SIDEWALK,
    PARKING,
    TERRAIN,
    VEGETATION,
    BUILDING,
    POLE,
    VEHICLE,
    PERSON,
    NUM_CLASSES,
    GROUND_CLASSES,
    DRIVABLE_CLASSES,
    DYNAMIC_CLASSES,
)
CLASSES = list(CANONICAL_CLASSES)

# base reflectivity per class (for the intensity channel)
REFLECT = np.array([0.15, 0.30, 0.20, 0.35, 0.45, 0.35, 0.40, 0.55, 0.30], np.float32)

# ----------------------------------------------------------------------------
# Sensor model (HDL-64E-like)
# ----------------------------------------------------------------------------
N_BEAMS = 64
N_AZ = 1024
FOV_UP, FOV_DOWN = 2.0, -24.9           # degrees
SENSOR_H = 1.73                          # m above ground
MAX_RANGE = 100.0
RANGE_NOISE = 0.02                       # m, 1 sigma
DROPOUT = 0.03


def beam_directions():
    """Unit ray directions, shape (N_BEAMS, N_AZ, 3). Row 0 = top beam."""
    elev = np.deg2rad(np.linspace(FOV_UP, FOV_DOWN, N_BEAMS))
    az = np.deg2rad(np.linspace(180.0, -180.0, N_AZ, endpoint=False))  # col 0 = behind-left, sweeps clockwise
    ce, se = np.cos(elev)[:, None], np.sin(elev)[:, None]
    d = np.stack([ce * np.cos(az)[None, :], ce * np.sin(az)[None, :], np.broadcast_to(se, (N_BEAMS, N_AZ))], -1)
    return d.astype(np.float32)


DIRS = beam_directions()


# ----------------------------------------------------------------------------
# Scene description
# ----------------------------------------------------------------------------
@dataclass
class Scene:
    boxes: list = field(default_factory=list)      # [xmin,ymin,zmin,xmax,ymax,zmax,cls,inst,vx,vy]
    cyls: list = field(default_factory=list)       # [cx,cy,r,z0,z1,cls,inst,vx,vy]
    spheres: list = field(default_factory=list)    # [cx,cy,cz,r,cls,inst]
    potholes: list = field(default_factory=list)   # [cx,cy,r,depth]
    markings: list = field(default_factory=list)   # [xmin,ymin,xmax,ymax]
    cross_x: float = 0.0
    n_inst: int = 0

    def inst(self):
        self.n_inst += 1
        return self.n_inst


ROAD_HALF = 7.0          # road |y| < 7 (2 lanes + parking strips)
CURB_H = 0.15
EGO_Y = -2.5


def _not_in_cross(x0, x1, cx, half=7.5):
    """Split [x0,x1] so it avoids the cross street [cx-half, cx+half]."""
    out = []
    if x1 <= cx - half or x0 >= cx + half:
        return [(x0, x1)]
    if x0 < cx - half:
        out.append((x0, cx - half))
    if x1 > cx + half:
        out.append((cx + half, x1))
    return out


def make_scene(seed: int, x_lo: float, x_hi: float, ego_speed: float, n_frames: int, dt: float) -> Scene:
    rng = np.random.default_rng(seed)
    s = Scene()
    ego_x0 = x_lo + 100.0
    travel = ego_speed * dt * n_frames
    s.cross_x = cx = ego_x0 + rng.uniform(20.0, travel + 40.0)

    def box(x0, y0, z0, x1, y1, z1, cls, inst=None, v=(0.0, 0.0)):
        s.boxes.append([x0, y0, z0, x1, y1, z1, cls, inst if inst is not None else s.inst(), v[0], v[1]])

    def cyl(x, y, r, z0, z1, cls, inst=None, v=(0.0, 0.0)):
        s.cyls.append([x, y, r, z0, z1, cls, inst if inst is not None else s.inst(), v[0], v[1]])

    # --- sidewalks (raised 15 cm) and roadside strips ------------------------
    for side in (-1, 1):
        for a, b in _not_in_cross(x_lo, x_hi, cx):
            y0, y1 = sorted((side * ROAD_HALF, side * 10.0))
            box(a, y0, -0.01, b, y1, CURB_H, SIDEWALK)
            # beyond the sidewalk: terrain or parking lots in 15-35 m chunks
            x = a
            while x < b:
                w = min(rng.uniform(15, 35), b - x)
                ya, yb = sorted((side * 10.0, side * 60.0))
                if rng.random() < 0.3:
                    box(x, ya, -0.01, x + w, yb, 0.02, PARKING)
                else:
                    box(x, ya, -0.01, x + w, yb, CURB_H + rng.uniform(-0.03, 0.06), TERRAIN)
                x += w

    # --- buildings, walls/fences ---------------------------------------------
    for side in (-1, 1):
        x = x_lo
        while x < x_hi:
            w = rng.uniform(10, 30)
            gap = rng.uniform(3, 15)
            for a, b in _not_in_cross(x, x + w, cx, 9):
                d0 = rng.uniform(16, 26)
                d1 = d0 + rng.uniform(8, 20)
                ya, yb = sorted((side * d0, side * d1))
                box(a, ya, 0, b, yb, rng.uniform(5, 18), BUILDING)
            x += w + gap
        x = x_lo
        while x < x_hi:
            if rng.random() < 0.35:
                w = rng.uniform(5, 15)
                for a, b in _not_in_cross(x, x + w, cx, 8):
                    yy = side * 10.3
                    box(a, min(yy, yy + side * 0.2), 0, b, max(yy, yy + side * 0.2), rng.uniform(0.9, 2.0), BUILDING)
            x += rng.uniform(10, 25)

    # --- poles, signs, trees --------------------------------------------------
    for side in (-1, 1):
        x = x_lo + rng.uniform(0, 10)
        while x < x_hi:
            if abs(x - cx) > 8:
                inst = s.inst()
                yy = side * 7.6
                cyl(x, yy, 0.1, 0, rng.uniform(5, 7), POLE, inst)
                if rng.random() < 0.5:
                    box(x - 0.3, yy - 0.03, 2.2, x + 0.3, yy + 0.03, 2.9, POLE, inst)
            x += rng.uniform(18, 30)
        x = x_lo + rng.uniform(0, 8)
        while x < x_hi:
            if abs(x - cx) > 9:
                inst = s.inst()
                yy = side * rng.uniform(8.8, 11.5)
                cyl(x, yy, 0.18, 0, 3.4, VEGETATION, inst)
                r = rng.uniform(1.6, 2.4)
                s.spheres.append([x, yy, 3.3 + r, r, VEGETATION, inst])
            x += rng.uniform(9, 20)
        for _ in range(int((x_hi - x_lo) / 12)):   # bushes on terrain
            x = rng.uniform(x_lo, x_hi)
            if abs(x - cx) > 9:
                yy = side * rng.uniform(11, 15)
                box(x, yy, 0, x + rng.uniform(0.8, 2.5), yy + side * rng.uniform(0.8, 1.5), rng.uniform(0.5, 1.2), VEGETATION)

    # --- an overhead gantry across the road (overhang test) -------------------
    gx = rng.uniform(15, 45)
    inst = s.inst()
    cyl(gx, -7.8, 0.2, 0, 5.6, POLE, inst)
    cyl(gx, 7.8, 0.2, 0, 5.6, POLE, inst)
    box(gx - 0.4, -7.8, 4.9, gx + 0.4, 7.8, 5.6, POLE, inst)

    # --- potholes and lane markings ------------------------------------------
    for _ in range(rng.integers(6, 11)):
        s.potholes.append([rng.uniform(x_lo + 100, x_lo + 100 + 60) if rng.random() < 0.6 else rng.uniform(x_lo, x_hi),
                           rng.uniform(-4.5, 4.5), rng.uniform(0.25, 0.6), rng.uniform(0.06, 0.12)])
    x = x_lo
    while x < x_hi:
        if abs(x - cx) > 8:
            s.markings.append([x, -0.08, x + 3, 0.08])
        x += 6
    for yy in (-5.0, 5.0):
        for a, b in _not_in_cross(x_lo, x_hi, cx, 8):
            s.markings.append([a, yy - 0.08, b, yy + 0.08])

    # --- vehicles --------------------------------------------------------------
    def car(x, y, v, heading_x=True):
        inst = s.inst()
        L, W = (rng.uniform(4.0, 4.9), rng.uniform(1.7, 1.9))
        if not heading_x:
            L, W = W, L
        box(x - L / 2, y - W / 2, 0.25, x + L / 2, y + W / 2, 1.05, VEHICLE, inst, v)
        cl, cw = (L * 0.55, W * 0.9) if heading_x else (L * 0.9, W * 0.55)
        box(x - cl / 2, y - cw / 2, 1.05, x + cl / 2, y + cw / 2, rng.uniform(1.4, 1.6), VEHICLE, inst, v)

    x = x_lo
    while x < x_hi:                                     # parked cars
        for side in (-1, 1):
            if rng.random() < 0.35 and abs(x - cx) > 9:
                car(x, side * 6.0, (0.0, 0.0))
        x += 6.0
    for _ in range(rng.integers(1, 3)):                 # same lane / same direction, ahead
        car(ego_x0 + rng.uniform(22, 60), EGO_Y, (ego_speed + rng.uniform(-1.5, 3.0), 0.0))
    for _ in range(rng.integers(0, 2)):                 # behind
        car(ego_x0 - rng.uniform(15, 40), EGO_Y, (ego_speed + rng.uniform(-1, 1), 0.0))
    for _ in range(rng.integers(2, 5)):                 # oncoming
        car(ego_x0 + rng.uniform(-10, 110), 2.5, (-rng.uniform(7, 13), 0.0))
    for _ in range(rng.integers(1, 3)):                 # cross traffic
        car(cx + rng.choice([-3.0, 3.0]), rng.uniform(-60, 60), (0.0, rng.choice([-1, 1]) * rng.uniform(5, 10)), heading_x=False)

    # --- pedestrians ------------------------------------------------------------
    for _ in range(rng.integers(6, 12)):
        side = rng.choice([-1, 1])
        x = ego_x0 + rng.uniform(-20, 90)
        moving = rng.random() < 0.6
        v = (rng.choice([-1, 1]) * rng.uniform(0.9, 1.7), 0.0) if moving else (0.0, 0.0)
        cyl(x, side * rng.uniform(7.8, 9.6), rng.uniform(0.25, 0.32), 0.15, rng.uniform(1.6, 1.9), PERSON, None, v)
    for _ in range(rng.integers(1, 3)):                 # crossing the road
        x = ego_x0 + rng.uniform(25, 70)
        cyl(x, rng.uniform(-6, 6), 0.3, 0.0, 1.75, PERSON, None, (0.0, rng.choice([-1, 1]) * rng.uniform(1.0, 1.6)))
    return s


# ----------------------------------------------------------------------------
# Ray casting
# ----------------------------------------------------------------------------
def _boxes_at(scene: Scene, t: float):
    b = np.asarray(scene.boxes, np.float32).copy()
    b[:, [0, 3]] += b[:, 8:9] * t
    b[:, [1, 4]] += b[:, 9:10] * t
    return b


def _cyls_at(scene: Scene, t: float):
    c = np.asarray(scene.cyls, np.float32).copy()
    c[:, 0] += c[:, 7] * t
    c[:, 1] += c[:, 8] * t
    return c


def cast(scene: Scene, origin: np.ndarray, t_now: float, rng: np.random.Generator):
    """Return per-ray range (inf = no return), class, instance, moving flag."""
    O = origin.astype(np.float32)
    D = DIRS.reshape(-1, 3)
    n = D.shape[0]
    best = np.full(n, np.inf, np.float32)
    cls = np.full(n, -1, np.int16)
    inst = np.zeros(n, np.int32)
    mov = np.zeros(n, bool)

    # ground plane z = 0 (road) + potholes
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(D[:, 2] < -1e-6, -O[2] / D[:, 2], np.inf).astype(np.float32)
    hx, hy = O[0] + tg * D[:, 0], O[1] + tg * D[:, 1]
    for px, py, pr, pd in scene.potholes:
        inside = (hx - px) ** 2 + (hy - py) ** 2 < pr * pr
        if inside.any():
            tg[inside] = (-pd - O[2]) / D[inside, 2]
    best[:] = tg
    cls[np.isfinite(tg)] = ROAD

    # boxes (slab test), culled to MAX_RANGE + margin
    B = _boxes_at(scene, t_now)
    cxb, cyb = (B[:, 0] + B[:, 3]) / 2, (B[:, 1] + B[:, 4]) / 2
    rad = np.hypot(B[:, 3] - B[:, 0], B[:, 4] - B[:, 1]) / 2
    B = B[np.hypot(cxb - O[0], cyb - O[1]) - rad < MAX_RANGE + 5]
    invD = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
    CH = 4096
    for i in range(0, n, CH):
        iv = invD[i:i + CH, None, :]                     # (c,1,3)
        t1 = (B[None, :, 0:3] - O) * iv
        t2 = (B[None, :, 3:6] - O) * iv
        tmin = np.minimum(t1, t2).max(-1)
        tmax = np.maximum(t1, t2).min(-1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
        tb = np.where(hit, tmin, np.inf)
        j = tb.argmin(1)
        tj = tb[np.arange(tb.shape[0]), j]
        sl = slice(i, i + CH)
        closer = tj < best[sl]
        idx = np.nonzero(closer)[0] + i
        best[idx] = tj[closer]
        cls[idx] = B[j[closer], 6].astype(np.int16)
        inst[idx] = B[j[closer], 7].astype(np.int32)
        mov[idx] = np.abs(B[j[closer], 8]) + np.abs(B[j[closer], 9]) > 0.3

    # vertical cylinders
    C = _cyls_at(scene, t_now)
    C = C[np.hypot(C[:, 0] - O[0], C[:, 1] - O[1]) < MAX_RANGE + 5]
    a = D[:, 0] ** 2 + D[:, 1] ** 2
    for i in range(0, n, CH):
        d = D[i:i + CH, None, :]
        ocx = O[0] - C[None, :, 0]
        ocy = O[1] - C[None, :, 1]
        aa = a[i:i + CH, None]
        b = 2 * (d[..., 0] * ocx + d[..., 1] * ocy)
        c = ocx ** 2 + ocy ** 2 - C[None, :, 2] ** 2
        disc = b * b - 4 * aa * c
        with np.errstate(invalid="ignore", divide="ignore"):
            tc = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * aa)
        z = O[2] + tc * d[..., 2]
        ok = (disc > 0) & (tc > 0) & (z >= C[None, :, 3]) & (z <= C[None, :, 4])
        tc = np.where(ok, tc, np.inf)
        j = tc.argmin(1)
        tj = tc[np.arange(tc.shape[0]), j]
        closer = tj < best[i:i + CH]
        idx = np.nonzero(closer)[0] + i
        best[idx] = tj[closer]
        cls[idx] = C[j[closer], 5].astype(np.int16)
        inst[idx] = C[j[closer], 6].astype(np.int32)
        mov[idx] = np.abs(C[j[closer], 7]) + np.abs(C[j[closer], 8]) > 0.3

    # spheres (canopies)
    if scene.spheres:
        S = np.asarray(scene.spheres, np.float32)
        S = S[np.hypot(S[:, 0] - O[0], S[:, 1] - O[1]) < MAX_RANGE + 5]
        for i in range(0, n, CH):
            d = D[i:i + CH, None, :]
            oc = O[None, None, :] - S[None, :, 0:3]
            b = 2 * (d * oc).sum(-1)
            c = (oc ** 2).sum(-1) - S[None, :, 3] ** 2
            disc = b * b - 4 * c
            with np.errstate(invalid="ignore"):
                ts = (-b - np.sqrt(np.maximum(disc, 0))) / 2
            # canopies are porous: ~35% of rays pass through the outer shell
            porous = rng.random(ts.shape) < 0.35
            ok = (disc > 0) & (ts > 0) & ~porous
            ts = np.where(ok, ts, np.inf)
            j = ts.argmin(1)
            tj = ts[np.arange(ts.shape[0]), j]
            closer = tj < best[i:i + CH]
            idx = np.nonzero(closer)[0] + i
            best[idx] = tj[closer]
            cls[idx] = S[j[closer], 4].astype(np.int16)
            inst[idx] = S[j[closer], 5].astype(np.int32)
            mov[idx] = False
    return best, cls, inst, mov


def simulate_sequence(seed: int, n_frames: int = 60, ego_speed: float = 8.0, dt: float = 0.1):
    """Simulate a drive. Returns dict of arrays, one entry per frame.

    Per frame we keep the organised range image layout (64 x 1024) because
    the segmentation network works on it directly.
    """
    rng = np.random.default_rng(seed + 1000)
    travel = ego_speed * dt * n_frames
    x_lo, x_hi = -100.0, travel + 100.0 + 110.0
    # shift so ego starts at x = 0 (x_lo + 100 in make_scene)
    scene = make_scene(seed, x_lo - 0.0, x_hi, ego_speed, n_frames, dt)
    # make_scene places ego_x0 at x_lo + 100 == 0
    frames = []
    for k in range(n_frames):
        t = k * dt
        ego = np.array([ego_speed * t, EGO_Y, 0.0], np.float32)
        origin = ego + np.array([0, 0, SENSOR_H], np.float32)
        rng_, cls, inst, mov = cast(scene, origin, t, rng)
        rng_ = rng_ + rng.normal(0, RANGE_NOISE, rng_.shape).astype(np.float32)
        valid = np.isfinite(rng_) & (rng_ < MAX_RANGE) & (rng.random(rng_.shape) > DROPOUT) & (cls >= 0)
        pts_w = origin + DIRS.reshape(-1, 3) * np.where(valid, rng_, 0)[:, None]
        # intensity: class reflectivity, lane markings bright, range falloff, noise
        inten = REFLECT[np.clip(cls, 0, 8)].copy()
        on_road = (cls == ROAD) & valid
        if scene.markings:
            M = np.asarray(scene.markings, np.float32)
            px, py = pts_w[:, 0], pts_w[:, 1]
            for (a0, b0, a1, b1) in M:
                m = on_road & (px >= a0) & (px <= a1) & (py >= b0) & (py <= b1)
                inten[m] = 0.85
        inten = np.clip(inten * (1 - 0.3 * np.clip(rng_ / MAX_RANGE, 0, 1)) + rng.normal(0, 0.04, inten.shape), 0, 1)
        frames.append(dict(
            range=np.where(valid, rng_, 0).reshape(N_BEAMS, N_AZ).astype(np.float32),
            xyz=(pts_w - np.array([ego[0], ego[1], 0], np.float32)).reshape(N_BEAMS, N_AZ, 3).astype(np.float32),
            intensity=inten.reshape(N_BEAMS, N_AZ).astype(np.float32),
            valid=valid.reshape(N_BEAMS, N_AZ),
            label=np.where(valid, cls, -1).reshape(N_BEAMS, N_AZ).astype(np.int8),
            moving=(mov & valid).reshape(N_BEAMS, N_AZ),
            ego=ego,
        ))
    return dict(frames=frames, scene=scene)


def save_sequence(path: str, seq: dict):
    F = seq["frames"]
    np.savez_compressed(
        path,
        range=np.stack([f["range"] for f in F]),
        xyz=np.stack([f["xyz"] for f in F]).astype(np.float16),
        intensity=np.stack([f["intensity"] for f in F]).astype(np.float16),
        valid=np.stack([f["valid"] for f in F]),
        label=np.stack([f["label"] for f in F]),
        moving=np.stack([f["moving"] for f in F]),
        ego=np.stack([f["ego"] for f in F]),
        potholes=np.asarray(seq["scene"].potholes, np.float32),
        cross_x=np.float32(seq["scene"].cross_x),
    )


def load_sequence(path: str):
    z = np.load(path)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    import time
    t0 = time.time()
    seq = simulate_sequence(0, n_frames=2)
    f = seq["frames"][0]
    print("sim 2 frames: %.2fs" % (time.time() - t0), "valid pts:", f["valid"].sum())
    lab = f["label"][f["valid"]]
    for i, c in enumerate(CLASSES):
        print(f"  {c:10s} {np.sum(lab == i):6d}")
    print("moving pts:", f["moving"].sum())
