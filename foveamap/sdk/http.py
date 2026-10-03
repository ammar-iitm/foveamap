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
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
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
        self.end_headers()
        self.wfile.write(body)

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
                    frame = _frame_from_json(body)
                except SDKError as exc:
                    return self._fail(400, "SDKError", str(exc))
                except FoveaMapError as exc:
                    return self._fail(400, type(exc).__name__, str(exc))
                try:
                    view = sdk.process(frame)
                except SDKLifecycleError as exc:
                    return self._fail(409, "SDKLifecycleError", str(exc))
                except FoveaMapError as exc:
                    return self._fail(422, type(exc).__name__, str(exc))
                self._send(200, {"api_version": API_VERSION, "snapshot": view.to_dict()})
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


def _frame_from_json(body: Any) -> LiDARFrame:
    """Strict JSON -> LiDARFrame (no silent defaults beyond documented ones)."""
    if not isinstance(body, dict):
        raise SDKError("Frame body must be a JSON object")
    if "pts" not in body:
        raise SDKError("Frame body requires 'pts' ([[x,y,z],...])")
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
    intensity = np.asarray(body.get("intensity", np.ones(n, dtype=np.float32)), dtype=np.float32).reshape(n)
    ring = np.asarray(body.get("ring", np.zeros(n, dtype=np.int16)), dtype=np.int16).reshape(n)
    pose = np.asarray(body.get("pose", np.eye(4)), dtype=np.float64).reshape(4, 4)
    origin = np.asarray(body.get("sensor_origin", [0.0, 0.0, 1.73]), dtype=np.float32).reshape(3)
    return LiDARFrame(
        pts=pts,
        intensity=intensity,
        ring=ring,
        pose=pose,
        sensor_origin=origin,
        timestamp=float(body.get("timestamp", 0.0)),
        frame_id=str(body.get("frame_id", "")),
        source_id="sdk/http",
    )


class FoveaMapHttpServer:
    """Embeddable HTTP server bound to one SDK session (stdlib only)."""

    def __init__(self, sdk: FoveaMap, *, host: str = "127.0.0.1", port: int = 0) -> None:
        if not isinstance(sdk, FoveaMap):
            raise SDKError(f"HTTP server requires a FoveaMap session, got {type(sdk).__name__}")
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise SDKError(f"Refusing non-loopback bind {host!r} (explicit local-only policy)")
        self.sdk = sdk
        self._server = HTTPServer((host, int(port)), _Handler)
        self._server.sdk = sdk  # type: ignore[attr-defined]
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
