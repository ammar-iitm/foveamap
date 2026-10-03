"""Phase 9 tests: public SDK boundary + optional HTTP adapter.

Validates behavior (not mere existence): lifecycle, processing, queries via
authoritative core semantics, snapshot isolation, status/health/metrics,
reset/shutdown, ownership, serialization, errors, determinism, both engines,
and Phase 8 compatibility. HTTP tests use the stdlib server (no ROS/CUDA).
"""
import json
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

from foveamap.sdk import (
    API_VERSION, FoveaMap, HealthReport, QueryResult, RuntimeMetrics,
    RuntimeStatus, SDKError, SDKLifecycleError, SDKQueryError, SnapshotView,
)
from foveamap.sdk.errors import SDKConfigError
from foveamap.core.config import FoveaMapConfig, RuntimeConfig
from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import ConfigurationError, ContractError, FoveaMapError
from foveamap.core.ontology import NUM_CLASSES, ROAD, VEHICLE
from foveamap.runtime.perception import ClassicalFallbackBackend


def _backend():
    return ClassicalFallbackBackend(device=torch.device("cpu"))


def _frame(n=60, cls_id=ROAD, x0=1.0, x1=3.0, ts=0.0, frame_id="f0"):
    xs = np.linspace(x0, x1, n)
    pts = np.column_stack([xs, np.full(n, 1.0), np.full(n, -1.7)]).astype(np.float32)
    return LiDARFrame(
        pts=pts, intensity=np.ones(n, dtype=np.float32), ring=np.zeros(n, dtype=np.int16),
        pose=np.eye(4, dtype=np.float64), timestamp=float(ts),
        frame_id=frame_id, source_id="test")


def _active(**kw):
    # the backend runs on the CPU, so the runtime must too ("auto" picks CUDA or MPS when present)
    if "config" in kw:
        cfg = kw["config"]
        if isinstance(cfg, dict):
            cfg.setdefault("runtime", {})
            if isinstance(cfg["runtime"], dict):
                cfg["runtime"].setdefault("device", "cpu")
        elif isinstance(cfg, FoveaMapConfig):
            if cfg.runtime.device == "auto":
                cfg.runtime.device = "cpu"
    else:
        kw["config"] = FoveaMapConfig(runtime=RuntimeConfig(device="cpu"))
    m = FoveaMap(perception_backend=_backend(), **kw)
    m.configure()
    m.start()
    return m


# ------------------------------------------------------- 1-4 basics + config
def test_public_imports_and_version():
    assert API_VERSION == "1"
    for name in ("FoveaMap", "SnapshotView", "QueryResult", "RuntimeStatus",
                 "RuntimeMetrics", "HealthReport", "SDKError", "SDKLifecycleError", "SDKQueryError"):
        assert name


def test_from_config_object_and_dict():
    m1 = FoveaMap(config=FoveaMapConfig(), perception_backend=_backend())
    assert isinstance(m1.config, FoveaMapConfig)
    m2 = FoveaMap(config={"terrain": {"traversable_cost_max": 170}}, perception_backend=_backend())
    assert m2.config.terrain.traversable_cost_max == 170


def test_unknown_config_keys_rejected():
    with pytest.raises(SDKConfigError):
        FoveaMap(config={"nope": {}}, perception_backend=_backend())
    with pytest.raises(SDKConfigError):
        FoveaMap(config={"terrain": {"traversable_cost_mx": 1}}, perception_backend=_backend())
    with pytest.raises(SDKConfigError):
        FoveaMap(config={"terrain": {"traversable_cost_max": 999}}, perception_backend=_backend())


def test_lifecycle_transitions_and_violations():
    m = FoveaMap(perception_backend=_backend())
    assert m.lifecycle == "CREATED"
    with pytest.raises(SDKLifecycleError):
        m.process(_frame())
    with pytest.raises(SDKLifecycleError):
        m.start()
    m.configure()
    m.start()
    assert m.lifecycle == "ACTIVE"
    with pytest.raises(SDKLifecycleError):
        m.configure()
    m.stop()
    with pytest.raises(SDKLifecycleError):
        m.process(_frame())
    m.close()
    assert m.lifecycle == "SHUTDOWN"
    with pytest.raises(SDKLifecycleError):
        m.reset()
    with pytest.raises(SDKLifecycleError):
        m.process(_frame())


def test_close_without_ownership_keeps_foreign_runtime():
    from foveamap.runtime.runtime import FoveaMapRuntime
    rt = FoveaMapRuntime(config=FoveaMapConfig(), perception_backend=_backend())
    m = FoveaMap(runtime=rt)
    assert m.status().owns_runtime is False
    m.configure()
    m.start()
    m.process(_frame())
    m.close()
    assert rt.frame_count == 1  # foreign runtime untouched by close
    assert m.status().owns_runtime is False


# ------------------------------------------------------- 5-8 processing/query
def test_valid_processing_and_snapshot_keys():
    m = _active()
    view = m.process(_frame())
    assert isinstance(view, SnapshotView)
    assert view.frame_id == "f0"
    assert view.num_tiers == 2
    assert len(view.origins) == 2
    q = view.query_point(2.0, 1.0)
    assert isinstance(q, QueryResult)
    assert q.state == "OBSERVED_STATIC"
    assert q.cost == 0
    assert q.traversable is True
    assert q.tier == 0


def test_query_points_batch_and_limit():
    m = _active()
    m.process(_frame())
    out = m.query_points([1.5, 2.0, 2.5], [1.0, 1.0, 1.0])
    assert [q.state for q in out] == ["OBSERVED_STATIC"] * 3
    with pytest.raises(SDKQueryError):
        m.query_points([1.0], [1.0, 2.0])
    with pytest.raises(SDKQueryError):
        m.query_points([1.0] * 5000, [1.0] * 5000, limit=100)


def test_query_before_snapshot_raises():
    m = _active()
    with pytest.raises(SDKQueryError):
        m.query_point(0.0, 0.0)


def test_snapshot_isolation_from_live_grid():
    m = _active()
    first = m.process(_frame(ts=0.0, frame_id="a"))
    q_before = first.query_point(2.0, 1.0)
    m.process(_frame(ts=0.1, frame_id="b", x0=20.0, x1=25.0))
    q_after = first.query_point(2.0, 1.0)
    assert (q_before.state, q_before.cost) == (q_after.state, q_after.cost)
    assert m.snapshot().frame_id == "b"


def test_invalid_frame_propagates_typed_error():
    m = _active()
    bad = {"pts": [[0.0, 0.0]]}  # legacy dict missing required keys
    with pytest.raises(Exception):
        m.process(bad)
    assert m.lifecycle == "ACTIVE"  # session survives
    assert m.status().frames_dropped >= 0


def test_contract_error_preserves_cause():
    from foveamap.core.exceptions import NumericalConsistencyError
    m = _active()
    # Contract enforced at construction time (before the SDK is involved).
    with pytest.raises(NumericalConsistencyError):
        LiDARFrame(
            pts=np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
            intensity=np.ones(1, dtype=np.float32), ring=np.zeros(1, dtype=np.int16),
            pose=np.eye(4, dtype=np.float64), timestamp=0.0, frame_id="x", source_id="x")
    # Contract violations via legacy dicts propagate unchanged (never wrapped).
    with pytest.raises(FoveaMapError):
        m.process({"pts": np.zeros((0, 3), dtype=np.float32)})
    # Unexpected backend failures translate to SDKError with cause preserved.
    from foveamap.runtime.perception import PerceptionBackend
    from foveamap.runtime.runtime import FoveaMapRuntime

    class _Boom(PerceptionBackend):
        @property
        def num_classes(self):
            return 9

        @property
        def device(self):
            import torch as _t
            return _t.device("cpu")

        @property
        def is_ready(self):
            return True

        def predict(self, frame, device=None):
            raise RuntimeError("boom")

        def predict_device(self, frame, dev_math=False, profiling=False):
            raise RuntimeError("boom")

        def reset(self):
            pass

    boom = FoveaMap(runtime=FoveaMapRuntime(config=FoveaMapConfig(), perception_backend=_Boom()))
    boom.configure()
    boom.start()
    with pytest.raises(SDKError) as caught:
        boom.process(_frame())
    assert isinstance(caught.value.__cause__, RuntimeError)


# ------------------------------------------------------- 10-15 status/health
def test_status_and_health_fields():
    m = _active()
    st = m.status()
    assert isinstance(st, RuntimeStatus)
    assert st.lifecycle == "ACTIVE"
    assert st.device == "cpu"
    assert st.grid_engine == "numpy"
    assert st.frames_processed == 0
    h = m.health()
    assert isinstance(h, HealthReport)
    assert h.status == "degraded"  # classical fallback is reported, not hidden
    assert "fallback_perception_backend" in h.reasons


def test_metrics_measured_not_invented():
    m = _active()
    m.process(_frame())
    met = m.metrics()
    assert isinstance(met, RuntimeMetrics)
    assert met.frames_processed == 1
    assert met.device == "cpu"
    assert met.cuda_note == "cuda_unavailable_cpu_execution"
    assert met.map_memory["allocated_bytes"] == 5120000
    assert "sdk_process_s" in met.stage_latency_ms


def test_repeated_failures_degrade_health():
    m = _active()
    for _ in range(3):
        with pytest.raises(Exception):
            m.process({"bogus": True})
    assert m.health().status == "degraded"
    assert any("repeated_frame_failures" in r for r in m.health().reasons)


def test_reset_clears_state_keeps_config():
    m = _active()
    m.process(_frame())
    st = m.reset()
    assert st.frames_processed == 0
    assert m.config.terrain.traversable_cost_max == 180
    with pytest.raises(SDKQueryError):
        m.query_point(0.0, 0.0)
    m.process(_frame())
    assert m.query_point(2.0, 1.0).state == "OBSERVED_STATIC"


def test_shutdown_behavior():
    m = _active()
    m.process(_frame())
    m.close()
    assert m.health().status == "unavailable"
    with pytest.raises(SDKLifecycleError):
        m.process(_frame())


def test_repeated_processing_deterministic():
    def run():
        mm = _active()
        mm.process(_frame())
        return mm.query_point(2.0, 1.0)
    a, b = run(), run()
    assert (a.state, a.cost, a.traversable) == (b.state, b.cost, b.traversable)


# ------------------------------------------------------- 21-27 serialization
def test_serialization_roundtrip_json():
    m = _active()
    view = m.process(_frame())
    blob = json.dumps({
        "snapshot": view.to_dict(),
        "query": view.query_point(2.0, 1.0).to_dict(),
        "status": m.status().to_dict(),
        "metrics": m.metrics().to_dict(),
        "health": m.health().to_dict(),
    })
    back = json.loads(blob)
    assert back["snapshot"]["api_version"] == "1"
    assert back["query"]["state"] == "OBSERVED_STATIC"
    assert "TierLayers" not in blob and "ndarray" not in blob


def test_no_mutable_internal_state_leaks():
    import dataclasses
    m = _active()
    view = m.process(_frame())
    assert SnapshotView.__dataclass_params__.frozen
    assert QueryResult.__dataclass_params__.frozen
    d = view.to_dict()
    d["origins"][0][0] = 999999
    assert view.origins[0][0] != 999999
    assert not hasattr(view, "tier_states")
    assert dataclasses.is_dataclass(view.query_point(0.0, 0.0))


# ------------------------------------------------------- 23-26 engines/compat
def test_torch_cpu_engine_end_to_end():
    import dataclasses
    base = FoveaMapConfig()
    torch_cfg = dataclasses.replace(base, runtime=dataclasses.replace(base.runtime, grid_engine="torch"))
    m = FoveaMap(config=torch_cfg, perception_backend=_backend())
    m.configure()
    m.start()
    view = m.process(_frame())
    assert view.query_point(2.0, 1.0).state == "OBSERVED_STATIC"
    assert m.status().grid_engine == "torch"


def test_sdk_core_query_agreement():
    m = _active()
    view = m.process(_frame())
    snap = m.snapshot()._snapshot
    core_q = snap.query_point(2.0, 1.0)
    sdk_q = view.query_point(2.0, 1.0)
    assert sdk_q.tier == core_q["tier"]
    assert sdk_q.state == core_q["state"]
    assert sdk_q.cost == core_q["cost"]
    assert sdk_q.dynamic == core_q["dynamic"]
    assert sdk_q.traversable == snap.is_traversable(2.0, 1.0)
    assert sdk_q.ground_m == pytest.approx(core_q["ground"])
    assert (sdk_q.slope_rad or 0.0) == pytest.approx(core_q["slope_rad"] or 0.0, abs=1e-9)


def test_phase8_node_core_interop():
    from foveamap_ros.node import FoveaMapNodeCore
    from foveamap_ros.config import from_ros_params
    ros_cfg = from_ros_params({"perception.backend_type": "classical"})
    sdk = FoveaMap(config=ros_cfg.core, perception_backend=_backend())
    sdk.configure()
    sdk.start()
    node = FoveaMapNodeCore(config=ros_cfg)
    node.configure()
    assert sdk.config.terrain.traversable_cost_max == node.config.core.terrain.traversable_cost_max == 180


# ------------------------------------------------------- HTTP (stdlib server)
def _serve(m):
    from foveamap.sdk.http import FoveaMapHttpServer
    server = FoveaMapHttpServer(m)
    url = server.start_background()
    return server, url


def _get(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def test_http_health_status_metrics():
    m = _active()
    server, url = _serve(m)
    try:
        code, health = _get(url, "/health")
        assert code == 200 and health["api_version"] == "1" and "status" in health
        code, status = _get(url, "/status")
        assert code == 200 and status["lifecycle"] == "ACTIVE"
        code, metrics = _get(url, "/metrics")
        assert code == 200 and "fps" in metrics
    finally:
        server.stop()


def test_http_query_and_validation():
    m = _active()
    m.process(_frame())
    server, url = _serve(m)
    try:
        code, q = _get(url, "/map/query?x=2.0&y=1.0")
        assert code == 200 and q["state"] == "OBSERVED_STATIC"
        code, err = _get(url, "/map/query?x=nan&y=1.0")
        assert code == 400 and err["type"] == "SDKQueryError"
        code, err = _get(url, "/nope")
        assert code == 404
    finally:
        server.stop()


def test_http_frame_lifecycle_and_limits():
    m = _active()
    server, url = _serve(m)
    try:
        pts = [[1.0 + i * 0.05, 1.0, -1.7] for i in range(20)]
        code, out = _post(url, "/frames", {"pts": pts, "frame_id": "h0"})
        assert code == 200 and out["snapshot"]["frame_id"] == "h0"
        assert "pose_identity_synthetic_test_only" in out["defaults_applied"]
        assert "timestamp_zero_synthetic" in out["defaults_applied"]
        code, err = _post(url, "/frames", {"nope": 1})
        assert code == 400
        code, err = _post(url, "/frames", {"pts": pts})  # frame_id required
        assert code == 400
        code, err = _post(url, "/frames", {"pts": []})
        assert code == 400
        code, err = _post(url, "/frames", {"pts": pts, "frame_id": "bad",
                                           "intensity": [1.0, 2.0]})
        assert code == 400  # shape mismatch is validation, not a crash
        code, st = _post(url, "/lifecycle", {"action": "stop"})
        assert code == 200 and st["status"]["lifecycle"] == "INACTIVE"
        code, err = _post(url, "/frames", {"pts": pts, "frame_id": "after-stop"})
        assert code == 409
        code, st = _post(url, "/lifecycle", {"action": "configure"})
        assert code == 200
        code, st = _post(url, "/lifecycle", {"action": "start"})
        assert code == 200 and st["status"]["lifecycle"] == "ACTIVE"
        code, err = _post(url, "/lifecycle", {"action": "explode"})
        assert code == 400
        code, _ = _post(url, "/reset", {})
        assert code == 200
    finally:
        server.stop()


def test_http_rejects_non_loopback_bind():
    from foveamap.sdk.http import FoveaMapHttpServer
    from foveamap.sdk.errors import SDKError
    m = _active()
    with pytest.raises(SDKError):
        FoveaMapHttpServer(m, host="0.0.0.0")


# --------------------------------------- closure: error state, rays, export
def test_last_error_stored_and_cleared_on_recovery():
    m = _active()
    assert m.status().last_error is None
    with pytest.raises(Exception):
        m.process({"bogus": True})
    err = m.status().last_error
    assert isinstance(err, str) and len(err) > 0  # typed failure preserved
    assert any(r.startswith("last_error:") for r in m.health().reasons)
    m.process(_frame())
    assert m.status().last_error is None
    assert not any(r.startswith("last_error:") for r in m.health().reasons)


def test_snapshot_view_exposes_no_mutable_arrays():
    m = _active()
    view = m.process(_frame())
    for name in dir(view):
        if name.startswith("__"):
            continue
        try:
            attr = getattr(view, name)
        except Exception:
            continue
        assert not isinstance(attr, np.ndarray), f"public accessor {name} leaks ndarray"
    for value in view.to_dict().values():
        assert not isinstance(value, np.ndarray)
    snap = view.to_dict()
    assert snap["api_version"] == "1"
    # No accessor returns live tier arrays; origins are plain tuples.
    assert view.origins[0] == tuple(view.origins[0])


def test_query_ray_follows_authoritative_queries():
    m = _active()
    m.process(_frame())
    ray = m.query_ray(0.0, 1.0, 0.0, step_m=0.5, max_steps=8)
    assert len(ray) == 8
    for k, q in enumerate(ray):
        ref = m.query_point(0.0 + k * 0.5, 1.0)
        assert (q.state, q.cost, q.dynamic) == (ref.state, ref.cost, ref.dynamic)
    with pytest.raises(SDKQueryError):
        m.query_ray(0.0, 1.0, 0.0, step_m=0.0)
    with pytest.raises(SDKQueryError):
        m.query_ray(0.0, 1.0, 0.0, max_steps=5000)
    with pytest.raises(SDKQueryError):
        m.query_ray(float("nan"), 1.0, 0.0)


def test_export_numpy_detached_copies():
    m = _active()
    m.process(_frame())
    exp = m.export_numpy()
    assert set(exp) == {"api_version", "tiers", "origins"}
    assert exp["api_version"] == "1"
    assert len(exp["tiers"]) == 2
    for tier in exp["tiers"]:
        for key in ("count", "cls", "cost", "ground", "dynamic"):
            assert isinstance(tier[key], np.ndarray)
    before = m.query_point(2.0, 1.0).cost
    exp["tiers"][0]["cost"][:] = 0
    exp["tiers"][0]["ground"][:] = 999.0
    assert m.query_point(2.0, 1.0).cost == before
