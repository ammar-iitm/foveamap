"""Minimal HTTP boundary over the SDK (Phase 9, stdlib only).

Justification: non-Python consumers need health/status/metrics/query/reset
without learning the Python API. This module adds no mapping logic and no
dependencies (``http.server`` + ``json`` only).

Safety rules enforced here:
- JSON bodies only, capped at ``MAX_BODY_BYTES`` (413 otherwise).
- Point arrays capped at ``MAX_POINTS_PER_REQUEST``.
- No filesystem access, no model loading, no pickle, no internal arrays.
- Errors map to typed JSON ``{error, type}`` (400 validation / 404 unknown /
  409 lifecycle / 500 unexpected); tracebacks never leak.
- Single-threaded server; the SDK lock serializes processing anyway.
- No shutdown endpoint: server lifetime is owned by the embedder via
  :meth:`FoveaMapHttpServer.stop`.

Endpoints:
    GET  /health, /status, /metrics, /map/query?x=&y=
    POST /frames   {"pts": [[x,y,z],...], ...}
    POST /reset
    POST /lifecycle {"action": "configure"|"start"|"stop"}
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from foveamap.core.contracts import LiDARFrame
from foveamap.core.exceptions import FoveaMapError

from . import API_VERSION
from .client import FoveaMap
from .errors import SDKError, SDKLifecycleError, SDKQueryError

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_POINTS_PER_REQUEST = 200_000


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    server: FoveaMapHttpServer  # type: ignore[assignment]

    def log_message(self, *args: Any) -> None:  # keep quiet by default
        pass

    def _send(self, code: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _send_file(self, file_path: Path, content_type: str) -> None:
        try:
            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self._fail(500, "InternalError", f"Failed to serve file: {type(exc).__name__}")

    def _fail(self, code: int, kind: str, message: str) -> None:
        self._send(code, {"api_version": API_VERSION, "error": str(message)[:500], "type": kind})

    def _read_json(self) -> Any:
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            raise SDKError("Invalid Content-Length")
        if size > MAX_BODY_BYTES:
            raise SDKError(f"Body too large ({size} > {MAX_BODY_BYTES})")
        raw = self.rfile.read(max(size, 0)) if size else b""
        if not raw:
            raise SDKError("Empty JSON body")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise SDKError(f"Invalid JSON body: {exc}") from exc

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            sdk = self.server.sdk
            if parsed.path == "/health":
                self._send(200, sdk.health().to_dict())
            elif parsed.path == "/status":
                self._send(200, sdk.status().to_dict())
            elif parsed.path == "/metrics":
                self._send(200, sdk.metrics().to_dict())
            elif parsed.path in ("/map/snapshot", "/snapshot"):
                try:
                    self._send(200, sdk.snapshot().to_dict())
                except SDKQueryError as exc:
                    self._fail(409, "SDKQueryError", str(exc))
            elif parsed.path == "/map/query":
                args = parse_qs(parsed.query)
                try:
                    x = float(args["x"][0])
                    y = float(args["y"][0])
                except (KeyError, IndexError, TypeError, ValueError):
                    return self._fail(400, "SDKQueryError", "query needs float x and y (?x=&y=)")
                if not (np.isfinite(x) and np.isfinite(y)):
                    return self._fail(400, "SDKQueryError", "x/y must be finite")
                try:
                    self._send(200, sdk.query_point(x, y).to_dict())
                except SDKQueryError as exc:
                    self._fail(409, "SDKQueryError", str(exc))
            elif getattr(self.server, "dashboard_dir", None) is not None:
                # Safe dashboard file serving with traversal prevention (Part 14)
                d_dir = Path(self.server.dashboard_dir).resolve()
                clean_path = parsed.path.lstrip("/")
                if clean_path in ("", "dashboard", "dashboard/"):
                    clean_path = "index.html"
                elif clean_path.startswith("dashboard/"):
                    clean_path = clean_path[len("dashboard/"):]
                target = (d_dir / clean_path).resolve()
                try:
                    target.relative_to(d_dir)
                except ValueError:
                    return self._fail(403, "Forbidden", "Path traversal forbidden")
                if target.is_file():
                    mime_types = {
                        ".html": "text/html; charset=utf-8",
                        ".css": "text/css; charset=utf-8",
                        ".js": "application/javascript; charset=utf-8",
                        ".json": "application/json; charset=utf-8",
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".txt": "text/plain; charset=utf-8",
                    }
                    content_type = mime_types.get(target.suffix.lower(), "application/octet-stream")
                    return self._send_file(target, content_type)
                else:
                    self._fail(404, "NotFound", f"static asset not found: {parsed.path!r}")
            else:
                self._fail(404, "NotFound", f"unknown endpoint {parsed.path!r}")
        except Exception as exc:  # never leak internals; never hang the server
            self._fail(500, "InternalError", f"{type(exc).__name__}")


    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            sdk = self.server.sdk
            if parsed.path == "/frames":
                try:
                    body = self._read_json()
                    frame, defaults = _frame_from_json(body)
                except SDKError as exc:
                    code = 413 if "Body too large" in str(exc) else 400
                    return self._fail(code, "SDKError", str(exc))
                except FoveaMapError as exc:
                    return self._fail(400, type(exc).__name__, str(exc))
                try:
                    view = sdk.process(frame)
                except SDKLifecycleError as exc:
                    return self._fail(409, "SDKLifecycleError", str(exc))
                except FoveaMapError as exc:
                    return self._fail(422, type(exc).__name__, str(exc))
                self._send(200, {"api_version": API_VERSION, "snapshot": view.to_dict(),
                                 "defaults_applied": defaults})
            elif parsed.path == "/reset":
                try:
                    sdk.reset()
                except SDKLifecycleError as exc:
                    return self._fail(409, "SDKLifecycleError", str(exc))
                self._send(200, {"api_version": API_VERSION, "status": sdk.status().to_dict()})
            elif parsed.path == "/lifecycle":
                try:
                    body = self._read_json()
                except SDKError as exc:
                    return self._fail(400, "SDKError", str(exc))
                action = body.get("action") if isinstance(body, dict) else None
                try:
                    if action == "configure":
                        sdk.configure()
                    elif action == "start":
                        sdk.start()
                    elif action == "stop":
                        sdk.stop()
                    else:
                        return self._fail(400, "SDKError", "action must be 'configure', 'start' or 'stop'")
                except SDKLifecycleError as exc:
                    return self._fail(409, "SDKLifecycleError", str(exc))
                self._send(200, {"api_version": API_VERSION, "status": sdk.status().to_dict()})
            else:
                self._fail(404, "NotFound", f"unknown endpoint {parsed.path!r}")
        except Exception as exc:
            self._fail(500, "InternalError", f"{type(exc).__name__}")


def _frame_from_json(body: Any) -> tuple[LiDARFrame, list[str]]:
    """Strict JSON -> (LiDARFrame, defaults_applied).

    External production-like input must carry its own identity: ``frame_id``
    is required. ``timestamp``/``pose``/``sensor_origin`` fall back to
    synthetic defaults ONLY for testing/replay, and every applied default is
    reported so consumers can never mistake synthetic metadata for measured
    data (no silent invention of physically meaningful values).
    """
    if not isinstance(body, dict):
        raise SDKError("Frame body must be a JSON object")
    if "pts" not in body:
        raise SDKError("Frame body requires 'pts' ([[x,y,z],...])")
    if "frame_id" not in body or not str(body["frame_id"]).strip():
        raise SDKError("Frame body requires a non-empty 'frame_id'")
    try:
        pts = np.asarray(body["pts"], dtype=np.float32).reshape(-1, 3)
    except (TypeError, ValueError) as exc:
        raise SDKError(f"'pts' must reshape to (N,3): {exc}") from exc
    if len(pts) == 0:
        raise SDKError("Empty point list rejected (send at least one point)")
    if len(pts) > MAX_POINTS_PER_REQUEST:
        raise SDKError(f"Too many points ({len(pts)} > {MAX_POINTS_PER_REQUEST})")
    if not np.all(np.isfinite(pts)):
        raise SDKError("pts must be finite")
    n = len(pts)
    try:
        intensity = np.asarray(body.get("intensity", np.ones(n, dtype=np.float32)), dtype=np.float32).reshape(n)
        ring = np.asarray(body.get("ring", np.zeros(n, dtype=np.int16)), dtype=np.int16).reshape(n)
    except (TypeError, ValueError) as exc:
        raise SDKError(f"'intensity'/'ring' must reshape to (N,): {exc}") from exc
    defaults: list[str] = []
    try:
        if "pose" in body:
            pose = np.asarray(body["pose"], dtype=np.float64).reshape(4, 4)
        else:
            pose = np.eye(4, dtype=np.float64)
            defaults.append("pose_identity_synthetic_test_only")
        if "sensor_origin" in body:
            origin = np.asarray(body["sensor_origin"], dtype=np.float32).reshape(3)
        else:
            origin = np.array([0.0, 0.0, 1.73], dtype=np.float32)
            defaults.append("sensor_origin_core_default_mount")
        if "timestamp" in body:
            timestamp = float(body["timestamp"])
        else:
            timestamp = 0.0
            defaults.append("timestamp_zero_synthetic")
    except (TypeError, ValueError) as exc:
        raise SDKError(f"Invalid frame metadata (pose/sensor_origin/timestamp): {exc}") from exc
    frame = LiDARFrame(
        pts=pts,
        intensity=intensity,
        ring=ring,
        pose=pose,
        sensor_origin=origin,
        timestamp=timestamp,
        frame_id=str(body["frame_id"]),
        source_id="sdk/http",
        metadata={"defaults_applied": list(defaults)},
    )
    return frame, defaults


class FoveaMapHttpServer:
    """Embeddable HTTP server bound to one SDK session (stdlib only)."""

    def __init__(
        self,
        sdk: FoveaMap,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        dashboard_dir: str | Path | None = None,
        allow_insecure_remote: bool = False,
    ) -> None:
        if not isinstance(sdk, FoveaMap):
            raise SDKError(f"HTTP server requires a FoveaMap session, got {type(sdk).__name__}")
        remote_allowed = allow_insecure_remote or os.environ.get("FOVEAMAP_ALLOW_INSECURE_REMOTE", "").strip().lower() in ("1", "true", "yes")
        if host not in ("127.0.0.1", "localhost", "::1"):
            if not remote_allowed:
                raise SDKError(
                    f"Refusing non-loopback bind {host!r} (explicit local-only policy). "
                    f"To enable external/container exposure, pass allow_insecure_remote=True or set FOVEAMAP_ALLOW_INSECURE_REMOTE=1."
                )
        self.sdk = sdk
        self.dashboard_dir = Path(dashboard_dir).resolve() if dashboard_dir is not None else None
        self._server = HTTPServer((host, int(port)), _Handler)
        self._server.sdk = sdk  # type: ignore[attr-defined]
        self._server.dashboard_dir = self.dashboard_dir  # type: ignore[attr-defined]
        self._thread: threading.Thread | None = None


    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start_background(self) -> str:
        """Serve in a daemon thread; returns the base URL."""
        if self._thread is not None:
            raise SDKError("Server already started")
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
                                        daemon=True)
        self._thread.start()
        return self.url

    def stop(self) -> None:
        """Stop serving (safe to call only after start_background)."""
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread = None
