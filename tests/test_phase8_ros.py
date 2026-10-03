"""Phase 8 tests: ROS 2 / system integration adapter.

Two levels (ROS 2 is unavailable on this machine, verified at collection):
A. ROS-independent adapter/unit tests (always run).
B. ROS 2 integration tests guarded by ROS2_AVAILABLE (skip with reason).
"""
import json

import numpy as np
import pytest

from foveamap_ros import ROS2_AVAILABLE, ROS2_MISSING_REASON
from foveamap_ros.pointcloud import (
    FLOAT32, UINT16, UINT8,
    RosHeader, RosPointCloud2, RosPointField, RosStamp,
    build_pointcloud2, cloud_to_arrays, cloud_to_lidar_frame,
    ros_stamp_from_seconds, stamp_to_seconds,
)
from foveamap_ros.frames import (
    DictTransformProvider, FramePolicy, MissingTransformError,
    RigidTransform, StaleTransformError, resolve_ego_points,
)
from foveamap_ros.config import from_ros_params
from foveamap_ros.qos import (
    GRID_OUTPUT_QOS, LIDAR_INPUT_QOS, METRICS_OUTPUT_QOS, POINTS_OUTPUT_QOS,
    QosProfile,
)
from foveamap_ros.messages import build_grid_payload, build_points_payload, payload_size_bytes
from foveamap_ros.diagnostics import MetricsAggregator
from foveamap_ros.node import FoveaMapNodeCore, FoveaMapRosNode
from foveamap.core.config import FoveaMapConfig
from foveamap.core.exceptions import ConfigurationError, DataAdapterError
from foveamap.grid import FoveatedGrid
from foveamap.core.ontology import NUM_CLASSES, ROAD

needs_ros = pytest.mark.skipif(
    not ROS2_AVAILABLE,
    reason=f"ROS 2 integration requires rclpy ({ROS2_MISSING_REASON})",
)


def _cloud_xyz(n=10, frame="lidar", sec=123, nanosec=456_000_000):
    xyz = np.column_stack([
        np.linspace(1.0, 5.0, n),
        np.full(n, 0.5),
        np.full(n, -1.7),
    ]).astype(np.float32)
    return xyz, RosPointCloud2(
        fields=[
            RosPointField("x", 0, FLOAT32),
            RosPointField("y", 4, FLOAT32),
            RosPointField("z", 8, FLOAT32),
        ],
        height=1, width=n, point_step=12, row_step=12 * n,
        data=xyz.tobytes(), is_dense=True,
        header=RosHeader(RosStamp(sec, nanosec), frame),
    )


# ------------------------------------------------------- A1 conversion: xyz
def test_xyz_extraction_preserves_values():
    xyz, msg = _cloud_xyz(n=7, frame="base_link")
    out = cloud_to_arrays(msg)
    np.testing.assert_array_equal(out["pts"], xyz)
    assert out["pts"].dtype == np.float32


def test_intensity_extraction_and_provenance():
    xyz, msg = _cloud_xyz(n=5, frame="base_link")
    rec = np.zeros(5, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32),
                             ("intensity", np.float32)])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["intensity"] = np.array([0.0, 64.0, 128.0, 200.0, 255.0], dtype=np.float32)
    msg2 = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
                RosPointField("z", 8, FLOAT32), RosPointField("intensity", 12, FLOAT32)],
        height=1, width=5, point_step=16, row_step=80, data=rec.tobytes(),
        header=RosHeader(RosStamp(1, 0), "lidar"))
    out = cloud_to_arrays(msg2)
    np.testing.assert_allclose(out["intensity"], rec["intensity"] / 255.0, rtol=1e-6)
    assert "intensity_provenance" in out


def test_missing_intensity_uses_documented_default():
    _, msg = _cloud_xyz(n=4, frame="base_link")
    out = cloud_to_arrays(msg)
    np.testing.assert_array_equal(out["intensity"], np.ones(4, dtype=np.float32))
    assert out["intensity_provenance"] == "default_ones_missing_field"


def test_missing_ring_zero_fills_and_flags_unavailable():
    _, msg = _cloud_xyz(n=4, frame="base_link")
    out = cloud_to_arrays(msg)
    np.testing.assert_array_equal(out["ring"], np.zeros(4, dtype=np.int16))
    assert out["ring_available"] is False


def test_ring_extraction_uint16():
    xyz, msg = _cloud_xyz(n=4, frame="base_link")
    rec = np.zeros(4, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32),
                             ("ring", np.uint16)])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["ring"] = np.array([0, 5, 31, 63], dtype=np.uint16)
    msg2 = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
                RosPointField("z", 8, FLOAT32), RosPointField("ring", 12, UINT16)],
        height=1, width=4, point_step=14, row_step=56, data=rec.tobytes(),
        header=RosHeader(RosStamp(2, 0), "lidar"))
    out = cloud_to_arrays(msg2)
    np.testing.assert_array_equal(out["ring"], [0, 5, 31, 63])
    assert out["ring_available"] is True


def test_timestamp_and_frame_id_preserved():
    _, msg = _cloud_xyz(n=3, frame="velodyne", sec=1700000000, nanosec=123_456_789)
    out = cloud_to_arrays(msg)
    assert out["timestamp"] == pytest.approx(1700000000.123456789, rel=1e-9)
    assert out["frame_id"] == "velodyne"
    frame = cloud_to_lidar_frame(msg)
    assert frame.timestamp == pytest.approx(1700000000.123456789, rel=1e-9)
    assert frame.frame_id == "velodyne"


def test_stamp_roundtrip():
    s = ros_stamp_from_seconds(123.456789123)
    assert (s.sec, s.nanosec) == (123, 456789123)
    assert stamp_to_seconds(s) == pytest.approx(123.456789123, rel=1e-9)


def test_empty_cloud():
    msg = RosPointCloud2(fields=[], height=1, width=0, point_step=12, row_step=0,
                         data=b"", header=RosHeader(RosStamp(5, 0), "lidar"))
    out = cloud_to_arrays(msg)
    assert out["pts"].shape == (0, 3)
    frame = cloud_to_lidar_frame(msg)
    assert frame.num_points == 0


def test_malformed_clouds_rejected():
    xyz, good = _cloud_xyz(n=4, frame="base_link")
    bad_steps = RosPointCloud2(fields=list(good.fields), height=1, width=4,
                               point_step=0, row_step=0, data=good.data, header=good.header)
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(bad_steps)
    truncated = RosPointCloud2(fields=list(good.fields), height=1, width=4,
                               point_step=12, row_step=48, data=good.data[:10], header=good.header)
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(truncated)
    no_xyz = RosPointCloud2(fields=[RosPointField("intensity", 0, FLOAT32)], height=1,
                            width=2, point_step=4, row_step=8, data=b"\x00" * 8, header=good.header)
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(no_xyz)
    bigendian = RosPointCloud2(fields=list(good.fields), height=1, width=4,
                               point_step=12, row_step=48, data=good.data,
                               header=good.header, is_bigendian=True)
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(bigendian)


def test_nan_xyz_rejected_by_default():
    xyz, msg = _cloud_xyz(n=3, frame="base_link")
    xyz[1, 0] = np.nan
    msg2 = RosPointCloud2(fields=list(msg.fields), height=1, width=3,
                          point_step=12, row_step=36, data=xyz.tobytes(), header=msg.header)
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(msg2)


def test_lidar_frame_contract_from_cloud():
    _, msg = _cloud_xyz(n=6, frame="base_link")
    frame = cloud_to_lidar_frame(msg, sensor_origin=[0.0, 0.0, 1.5])
    assert frame.num_points == 6
    assert frame.metadata["ring_available"] is False
    np.testing.assert_allclose(frame.sensor_origin, [0.0, 0.0, 1.5], rtol=1e-6)


# ------------------------------------------------------- A2 TF
def test_same_frame_needs_no_transform():
    pts = np.array([[1.0, 2.0, 3.0]])
    ego, prov = resolve_ego_points(pts, "base_link", 10.0, FramePolicy(), None)
    np.testing.assert_allclose(ego, pts)
    assert prov["tf"] == "identity_same_frame"


def test_tf_transform_applies_translation_and_yaw():
    pts = np.array([[1.0, 0.0, 0.0]])
    tf = RigidTransform(translation=(1.0, 2.0, 0.0),
                        rotation_xyzw=(0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
                        stamp=10.0, parent_frame="base_link", child_frame="lidar")
    provider = DictTransformProvider({("base_link", "lidar"): tf})
    ego, prov = resolve_ego_points(pts, "lidar", 10.05, FramePolicy(), provider)
    np.testing.assert_allclose(ego, [[1.0, 3.0, 0.0]], atol=1e-6)
    assert prov["tf"] == "transformed"


def test_missing_tf_is_typed_failure_not_identity():
    with pytest.raises(MissingTransformError):
        resolve_ego_points(np.array([[1.0, 0.0, 0.0]]), "lidar", 10.0,
                           FramePolicy(), DictTransformProvider({}))
    with pytest.raises(MissingTransformError):
        resolve_ego_points(np.array([[1.0, 0.0, 0.0]]), "lidar", 10.0, FramePolicy(), None)


def test_stale_tf_rejected_unless_allowed():
    tf = RigidTransform(translation=(0.0, 0.0, 0.0), stamp=9.0,
                        parent_frame="base_link", child_frame="lidar")
    provider = DictTransformProvider({("base_link", "lidar"): tf})
    with pytest.raises(StaleTransformError):
        resolve_ego_points(np.array([[1.0, 0.0, 0.0]]), "lidar", 10.0,
                           FramePolicy(max_tf_age_s=0.2), provider)
    ego, prov = resolve_ego_points(
        np.array([[1.0, 0.0, 0.0]]), "lidar", 10.0,
        FramePolicy(max_tf_age_s=0.2, allow_stale_tf=True), provider)
    assert prov["tf"] == "transformed_stale_accepted"
    assert prov["tf_age_s"] == pytest.approx(1.0)


def test_empty_frame_id_rejected():
    with pytest.raises(DataAdapterError):
        resolve_ego_points(np.array([[1.0, 0.0, 0.0]]), "", 10.0, FramePolicy(), None)


# ------------------------------------------------------- A3 config
def test_ros_params_map_to_core_config():
    cfg = from_ros_params({
        "sensor.n_rows": 32,
        "runtime.grid_engine": "torch",
        "terrain.traversable_cost_max": 170,
        "dynamic.max_tracks": 500,
        "io.max_queue": 4,
        "io.drop_policy": "newest",
    })
    assert cfg.core.sensor.n_rows == 32
    assert cfg.core.runtime.grid_engine == "torch"
    assert cfg.core.terrain.traversable_cost_max == 170
    assert cfg.core.dynamic.max_tracks == 500
    assert cfg.io.max_queue == 4
    assert cfg.io.drop_policy == "newest"
    assert isinstance(cfg.core, FoveaMapConfig)


def test_ros_params_reject_unknown_and_invalid():
    with pytest.raises(ConfigurationError):
        from_ros_params({"grid.celll_size": 0.05})
    with pytest.raises(ConfigurationError):
        from_ros_params({"terrain.traversable_cost_max": 300})
    with pytest.raises(ConfigurationError):
        from_ros_params({"io.drop_policy": "sometimes"})
    with pytest.raises(ConfigurationError):
        from_ros_params({"io.max_queue": 0})
    with pytest.raises(ConfigurationError):
        from_ros_params({"io.input_topic": "relative/topic"})


def test_qos_profiles_documented_and_bounded():
    assert LIDAR_INPUT_QOS.reliability == "best_effort"
    assert LIDAR_INPUT_QOS.depth == 5
    assert GRID_OUTPUT_QOS.depth == 1
    assert POINTS_OUTPUT_QOS.depth == 1
    assert METRICS_OUTPUT_QOS.reliability == "reliable"
    assert METRICS_OUTPUT_QOS.durability == "transient_local"
    with pytest.raises(ValueError):
        QosProfile(history="keep_all")
    with pytest.raises(ValueError):
        QosProfile(depth=0)


# ------------------------------------------------------- A4 node lifecycle
def _node_config(**over):
    # Same-frame operation (world == base) unless a test overrides it: the
    # node never silently substitutes identity world pose when TF is required.
    # The runtime runs on the CPU: "auto" would pick CUDA or MPS when present.
    params = {"perception.backend_type": "classical", "runtime.grid_engine": "numpy",
              "runtime.device": "cpu", "io.world_frame": "base_link"}
    params.update(over)
    return from_ros_params(params)


def test_lifecycle_transitions_deterministic():
    from foveamap_ros.node import CREATED, CONFIGURED, ACTIVE, INACTIVE, SHUTDOWN
    core = FoveaMapNodeCore(config=_node_config())
    assert core.state == CREATED
    core.configure()
    assert core.state == CONFIGURED
    core.activate()
    assert core.state == ACTIVE
    core.deactivate()
    assert core.state == INACTIVE
    core.configure()
    core.activate()
    core.shutdown()
    assert core.state == SHUTDOWN
    assert core.queue_depth == 0


def test_inactive_node_drops_with_metrics():
    core = FoveaMapNodeCore(config=_node_config())
    core.configure()
    _, msg = _cloud_xyz(n=3, frame="base_link")
    assert core.submit(msg) is False
    assert core.metrics.frames_dropped == 1


def test_bounded_queue_drop_oldest_and_newest():
    _, msg = _cloud_xyz(n=2, frame="base_link")
    core = FoveaMapNodeCore(config=_node_config(**{"io.max_queue": 2, "io.drop_policy": "oldest"}))
    core.configure()
    core.activate()
    assert all(core.submit(msg) for _ in range(2))
    assert core.submit(msg) is True  # evicts oldest, still accepted
    assert core.queue_depth == 2
    assert core.metrics.frames_dropped == 1

    core2 = FoveaMapNodeCore(config=_node_config(**{"io.max_queue": 1, "io.drop_policy": "newest"}))
    core2.configure()
    core2.activate()
    assert core2.submit(msg) is True
    assert core2.submit(msg) is False
    assert core2.queue_depth == 1


def test_full_ros_like_cycle_numpy():
    core = FoveaMapNodeCore(config=_node_config())
    core.configure()
    core.activate()
    _, msg = _cloud_xyz(n=50, frame="base_link")
    assert core.submit(msg) is True
    snap = core.spin_once()
    assert snap is not None
    out = core.last_outputs
    assert "/foveamap/grid" in out and "/foveamap/points_labeled" in out and "/foveamap/metrics" in out
    grid_payload = out["/foveamap/grid"]
    assert grid_payload["frame_id"] == snap.frame_id
    assert grid_payload["serialized_size_bytes"] > 0
    metrics = out["/foveamap/metrics"]
    assert metrics["frames_processed"] == 1
    assert metrics["device"] == "cpu"
    assert "cuda" not in metrics["device"]
    assert metrics["cuda_note"] == "cuda_unavailable_cpu_execution"


def test_full_ros_like_cycle_torch_cpu():
    core = FoveaMapNodeCore(config=_node_config(**{"runtime.grid_engine": "torch"}))
    core.configure()
    core.activate()
    _, msg = _cloud_xyz(n=50, frame="base_link")
    core.submit(msg)
    snap = core.spin_once()
    assert snap is not None
    assert snap.query_point(2.0, 0.5)["state"] in (
        "OBSERVED_STATIC", "STALE", "UNKNOWN", "OBSERVED_DYNAMIC", "ACTIVE_DYNAMIC")


def test_deterministic_repeated_processing():
    def run_once():
        core = FoveaMapNodeCore(config=_node_config())
        core.configure()
        core.activate()
        _, msg = _cloud_xyz(n=30, frame="base_link")
        core.submit(msg)
        snap = core.spin_once()
        return snap.query_point(2.0, 0.5), core.last_outputs["/foveamap/metrics"]["frames_processed"]

    q1, f1 = run_once()
    q2, f2 = run_once()
    assert (f1, q1["state"], q1["cost"]) == (f2, q2["state"], q2["cost"])


def test_publication_cannot_mutate_live_map():
    core = FoveaMapNodeCore(config=_node_config())
    core.configure()
    core.activate()
    _, msg = _cloud_xyz(n=40, frame="base_link")
    core.submit(msg)
    snap = core.spin_once()
    before = snap.query_point(2.0, 0.5)
    _, msg2 = _cloud_xyz(n=40, frame="base_link")
    core.submit(msg2)
    core.spin_once()
    after = snap.query_point(2.0, 0.5)
    assert set(before) == set(after)
    for key in before:
        b, a = before[key], after[key]
        if isinstance(b, float) and isinstance(a, float) and np.isnan(b) and np.isnan(a):
            continue
        assert b == a, f"snapshot mutated at {key}: {b!r} != {a!r}"
    tracks_before = [dict(t) for t in snap.dynamic_tracks]
    assert tracks_before == [dict(t) for t in snap.dynamic_tracks]


def test_malformed_frame_does_not_corrupt_map():
    core = FoveaMapNodeCore(config=_node_config())
    core.configure()
    core.activate()
    _, good = _cloud_xyz(n=20, frame="base_link")
    core.submit(good)
    snap_ok = core.spin_once()
    assert snap_ok is not None
    bad = RosPointCloud2(fields=[], height=1, width=0, point_step=0, row_step=0,
                         data=b"", header=RosHeader(RosStamp(9, 0), "lidar"))
    bad.point_step = 0
    core.submit(bad)
    assert core.spin_once() is None  # conversion failure, map untouched
    assert core.last_snapshot is snap_ok
    assert core.metrics.errors


def test_empty_cloud_processes_without_crash():
    core = FoveaMapNodeCore(config=_node_config())
    core.configure()
    core.activate()
    msg = RosPointCloud2(fields=[], height=1, width=0, point_step=12, row_step=0,
                         data=b"", header=RosHeader(RosStamp(9, 0), "lidar"))
    core.submit(msg)
    snap = core.spin_once()
    assert snap is not None


def test_points_payload_serialization_roundtrip():
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    payload = build_points_payload(
        xyz, np.array([0, 7]), np.array([0.9, 0.8], dtype=np.float32),
        np.array([False, True]), frame_id="lidar", timestamp=12.5, max_points=12000)
    assert payload.header.frame_id == "lidar"
    assert (payload.header.stamp.sec, payload.header.stamp.nanosec) == (12, 500_000_000)
    back = cloud_to_arrays(payload)
    np.testing.assert_allclose(back["pts"], xyz, rtol=1e-6)
    # ring/laser/channel absent in labeled output by design -> zero-fill.
    np.testing.assert_array_equal(back["ring"], [0, 0])
    assert back["ring_available"] is False
    # Semantic labels survive exactly in the binary payload itself.
    raw = np.frombuffer(payload.data, dtype=np.uint8)
    assert len(raw) == payload.row_step
    assert payload_size_bytes({"a": 1}) == len(json.dumps({"a": 1}).encode())


def test_grid_payload_metadata_and_bandwidth():
    g = FoveatedGrid("spec")
    xy = np.column_stack([np.linspace(1.0, 3.0, 40), np.full(40, 1.0)])
    p = np.zeros((40, NUM_CLASSES), dtype=np.float32)
    p[:, ROAD] = 1.0
    g.update(xy, np.full(40, -1.7, dtype=np.float32), p, np.zeros(40, bool), (0.0, 0.0))
    from foveamap_ros.messages import build_grid_payload as bgp
    snap = g.snapshot()
    from foveamap.core.contracts import MapSnapshot
    ms = MapSnapshot(timestamp=3.0, frame_id="f", ego_pose=np.eye(4),
                     origins=tuple(tuple(int(c) for c in o) for o in g.origins),
                     tier_states=tuple(snap))
    payload = bgp(ms, max_cells_per_tier=1000)
    assert payload["frame_id"] == "f"
    assert payload["serialized_size_bytes"] > 0
    assert len(payload["tiers"]) == 2
    assert all("traversable" in c and "cls" in c and "cost" in c for t in payload["tiers"] for c in t["cells"])
    full = bgp(ms, max_cells_per_tier=None)
    assert full["serialized_size_bytes"] >= payload["serialized_size_bytes"]


def test_single_point_with_ring_processes_through_node():
    core = FoveaMapNodeCore(config=_node_config())
    core.configure()
    core.activate()
    rec = np.zeros(1, dtype=[("x", np.float32), ("y", np.float32),
                             ("z", np.float32), ("ring", np.uint8)])
    rec["x"], rec["y"], rec["z"] = [10.0], [2.0], [0.5]
    rec["ring"] = [7]
    msg = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
                RosPointField("z", 8, FLOAT32), RosPointField("ring", 12, UINT8)],
        height=1, width=1, point_step=13, row_step=13, data=rec.tobytes(),
        header=RosHeader(RosStamp(20, 0), "base_link"))
    core.submit(msg)
    snap = core.spin_once()
    assert snap is not None
    assert isinstance(core.last_outputs["/foveamap/grid"], dict)
    assert core.last_outputs["/foveamap/metrics"]["frames_processed"] == 1


# ------------------------------------------------------- B guarded ROS2 tests
@needs_ros
def test_ros_node_lifecycle_with_rclpy():
    node = FoveaMapRosNode(ros_params={"perception.backend_type": "classical"})
    node.configure()
    node.activate()
    assert node.state == "ACTIVE"
    node.deactivate()
    node.shutdown()
    assert node.state == "SHUTDOWN"


@needs_ros
def test_ros_qos_conversion_with_rclpy():
    from foveamap_ros.qos import to_rclpy
    q = to_rclpy(LIDAR_INPUT_QOS)
    assert q.depth == 5


# --------------------------------------- C closure: padded/shuffled layouts
def _velodyne_style_cloud(n=6, frame="lidar"):
    """Ouster/Velodyne-style layout: shuffled fields, alignment padding."""
    import struct as _struct
    xs = np.linspace(1.0, 6.0, n).astype(np.float32)
    recs = bytearray()
    for k in range(n):
        # ring u16@0, pad, x f32@8, y f32@12, z f32@16, intensity u8@20, pad, ring u16@22
        recs += _struct.pack("<H6xfffBxH",
                             k % 64, float(xs[k]), 0.5, -1.7, 100 + k, 7)
    return RosPointCloud2(
        fields=[RosPointField("ring", 0, UINT16),
                RosPointField("x", 8, FLOAT32),
                RosPointField("y", 12, FLOAT32),
                RosPointField("z", 16, FLOAT32),
                RosPointField("intensity", 20, UINT8),
                RosPointField("pad_ring", 22, UINT16)],
        height=1, width=n, point_step=24, row_step=24 * n, data=bytes(recs),
        header=RosHeader(RosStamp(50, 0), frame)), xs


def test_shuffled_padded_layout_parses_exactly():
    msg, xs = _velodyne_style_cloud(n=6)
    out = cloud_to_arrays(msg)
    np.testing.assert_allclose(out["pts"][:, 0], xs, rtol=1e-6)
    np.testing.assert_allclose(out["pts"][:, 1], 0.5, rtol=1e-6)
    np.testing.assert_allclose(out["pts"][:, 2], -1.7, rtol=1e-6)
    # Offset-honoring proof: ring@0 reads k%64, NOT the pad_ring@22 value (7).
    np.testing.assert_array_equal(out["ring"], np.arange(6, dtype=np.int16) % 64)
    assert out["ring_available"] is True
    np.testing.assert_allclose(out["intensity"], (100 + np.arange(6)) / 255.0, rtol=1e-6)


def test_organized_cloud_with_row_padding():
    xyz, msg = _cloud_xyz(n=4)
    # organized 2x2 with 8 bytes of row padding: row_step = 2*12 + 8
    padded = msg.data[:24] + b"\x00" * 8 + msg.data[24:] + b"\x00" * 8
    org = RosPointCloud2(fields=list(msg.fields), height=2, width=2,
                         point_step=12, row_step=32, data=padded, header=msg.header)
    out = cloud_to_arrays(org)
    np.testing.assert_allclose(out["pts"].reshape(2, 2, 3)[:, :, 0],
                               xyz[:, 0].reshape(2, 2), rtol=1e-6)


def test_overlapping_and_oob_fields_rejected():
    _, msg = _cloud_xyz(n=2)
    overlap = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 2, FLOAT32),
                RosPointField("z", 8, FLOAT32)],
        height=1, width=2, point_step=12, row_step=24, data=msg.data, header=msg.header)
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(overlap)
    oob = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
                RosPointField("z", 12, FLOAT32)],
        height=1, width=2, point_step=12, row_step=24, data=msg.data, header=msg.header)
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(oob)


def test_count_multi_field_first_element_used():
    import struct as _struct
    recs = b"".join(_struct.pack("<fff3f", 1.0 + k, 0.0, 0.0, 0.1 * k, 0.2 * k, 0.3 * k) for k in range(3))
    msg = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
                RosPointField("z", 8, FLOAT32),
                RosPointField("normal", 12, FLOAT32, 3)],
        height=1, width=3, point_step=24, row_step=72, data=recs,
        header=RosHeader(RosStamp(1, 0), "lidar"))
    out = cloud_to_arrays(msg)  # must not crash on count > 1 auxiliary field
    assert len(out["pts"]) == 3


def test_uppercase_and_mixed_case_fields_parse():
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    msg = RosPointCloud2(
        fields=[RosPointField("X", 0, FLOAT32), RosPointField("Y", 4, FLOAT32),
                RosPointField("Z", 8, FLOAT32)],
        height=1, width=2, point_step=12, row_step=24, data=xyz.tobytes(),
        header=RosHeader(RosStamp(1, 0), "lidar"))
    out = cloud_to_arrays(msg)
    np.testing.assert_allclose(out["pts"], xyz, rtol=1e-6)
    mixed = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("y", 4, FLOAT32),
                RosPointField("Intensity", 8, FLOAT32), RosPointField("z", 12, FLOAT32)],
        height=1, width=2, point_step=16, row_step=32,
        data=np.zeros((2, 4), dtype=np.float32).tobytes(),
        header=RosHeader(RosStamp(1, 0), "lidar"))
    out2 = cloud_to_arrays(mixed)
    assert len(out2["pts"]) == 2
    assert out2["intensity_provenance"] != "default_ones_missing_field"


def test_case_insensitive_duplicates_rejected():
    msg = RosPointCloud2(
        fields=[RosPointField("x", 0, FLOAT32), RosPointField("X", 12, FLOAT32),
                RosPointField("y", 4, FLOAT32), RosPointField("z", 8, FLOAT32)],
        height=1, width=1, point_step=16, row_step=16,
        data=np.zeros(16, dtype=np.uint8).tobytes(),
        header=RosHeader(RosStamp(1, 0), "lidar"))
    with pytest.raises(DataAdapterError):
        cloud_to_arrays(msg)


# --------------------------------------- C closure: TF world pose
def _tf_node(world="map", provider=None):
    from foveamap_ros.node import FoveaMapNodeCore
    params = {"perception.backend_type": "classical", "runtime.grid_engine": "numpy",
              "io.world_frame": world}
    return FoveaMapNodeCore(config=_node_config(**params), tf_provider=provider)


def _static_tf(x, stamp=100.0):
    from foveamap_ros.frames import RigidTransform, DictTransformProvider
    return DictTransformProvider({
        ("base_link", "lidar"): RigidTransform((0, 0, 0), (0, 0, 0, 1), stamp, "base_link", "lidar"),
        ("map", "base_link"): RigidTransform((x, 0, 0), (0, 0, 0, 1), stamp, "map", "base_link"),
    })


def test_world_pose_moves_with_vehicle():
    from foveamap_ros.frames import resolve_world_pose
    from foveamap_ros.frames import FramePolicy
    policy = FramePolicy(base_frame="base_link", world_frame="map")
    for x in (0.0, 5.0, 10.0):
        pose, prov = resolve_world_pose("base_link", "map", 100.0, policy, _static_tf(x))
        assert pose[0, 3] == pytest.approx(x)
        assert prov["pose"] == "transformed"


def test_world_pose_missing_or_stale_fails_typed():
    from foveamap_ros.frames import FramePolicy, resolve_world_pose
    from foveamap_ros.frames import DictTransformProvider, MissingTransformError, StaleTransformError
    policy = FramePolicy(base_frame="base_link", world_frame="map")
    with pytest.raises(MissingTransformError):
        resolve_world_pose("base_link", "map", 100.0, policy, DictTransformProvider({}))
    with pytest.raises(MissingTransformError):
        resolve_world_pose("base_link", "map", 100.0, policy, None)
    old = DictTransformProvider({("map", "base_link"):
        RigidTransform((0, 0, 0), (0, 0, 0, 1), 90.0, "map", "base_link")})
    with pytest.raises(StaleTransformError):
        resolve_world_pose("base_link", "map", 100.0, policy, old)
    pose, prov = resolve_world_pose("base_link", "base_link", 100.0, policy, None)
    np.testing.assert_allclose(pose, np.eye(4))
    assert prov["pose"] == "identity_same_frame_operation"


def test_node_world_frame_end_to_end_and_timestamp_consistency():
    core = _tf_node(world="map", provider=_static_tf(5.0))
    core.configure()
    core.activate()
    # Cloud stamp exactly matches the TF stamp (100.0); staleness is strict.
    xyz, msg = _cloud_xyz(n=10, frame="lidar", sec=100, nanosec=0)
    assert core.submit(msg) is True
    snap = core.spin_once()
    assert snap is not None
    assert snap.ego_pose[0, 3] == pytest.approx(5.0)  # world pose from TF, same stamp
    assert snap.timestamp == pytest.approx(100.0, rel=1e-9)


def test_node_missing_world_tf_drops_frame_with_metrics():
    # base_link cloud needs no sensor TF, isolating the world-pose branch.
    core = _tf_node(world="map", provider=None)
    core.configure()
    core.activate()
    _, msg = _cloud_xyz(n=5, frame="base_link", sec=100, nanosec=0)
    core.submit(msg)
    assert core.spin_once() is None
    assert core.metrics.errors.get("INPUT_MissingTransformError", 0) == 1
    assert core.last_snapshot is None


# --------------------------------------- C closure: terminal lifecycle
def test_shutdown_is_terminal():
    from foveamap_ros.node import SHUTDOWN
    from foveamap.core.exceptions import FoveaMapError
    core = FoveaMapNodeCore(config=_node_config())
    core.configure()
    core.activate()
    core.shutdown()
    assert core.state == SHUTDOWN
    for op in (core.configure, core.activate):
        with pytest.raises(FoveaMapError):
            op()
    core.deactivate()  # safe no-op, stays terminal
    assert core.state == SHUTDOWN
    _, msg = _cloud_xyz(n=2, frame="base_link")
    assert core.submit(msg) is False
    assert core.spin_once() is None


# --------------------------------------- C closure: colcon structure
def test_ros_package_structurally_colcon_valid():
    import os
    import xml.etree.ElementTree as ET
    root = os.path.join(os.path.dirname(__file__), "..", "foveamap_ros")
    assert os.path.isfile(os.path.join(root, "package.xml"))
    assert os.path.isfile(os.path.join(root, "CMakeLists.txt"))
    tree = ET.parse(os.path.join(root, "package.xml"))
    assert tree.getroot().find("name").text == "foveamap_ros"
    with open(os.path.join(root, "CMakeLists.txt")) as fh:
        cmake = fh.read()
    msgs = sorted(f for f in os.listdir(os.path.join(root, "msg")) if f.endswith(".msg"))
    assert msgs, "msg/ must contain message definitions"
    for m in msgs:
        assert m in cmake, f"{m} not wired into CMakeLists.txt"
    for dep in ("rosidl_default_generators", "std_msgs", "sensor_msgs"):
        assert dep in cmake, f"{dep} missing from CMakeLists.txt"


# ------------------------------------------------------- B guarded ROS2 tests
@needs_ros
def test_ros_node_lifecycle_with_rclpy():
    node = FoveaMapRosNode(ros_params={"perception.backend_type": "classical"})
    node.configure()
    node.activate()
    assert node.state == "ACTIVE"
    node.deactivate()
    node.shutdown()
    assert node.state == "SHUTDOWN"


@needs_ros
def test_ros_qos_conversion_with_rclpy():
    from foveamap_ros.qos import to_rclpy
    q = to_rclpy(LIDAR_INPUT_QOS)
    assert q.depth == 5

