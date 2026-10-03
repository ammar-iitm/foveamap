# FoveaMap — HTTP API (Phase 9, stdlib only)

Thin boundary over the SDK (`foveamap.sdk.http`, no dependencies):
`HTTP → SDK → Runtime → Core`. Loopback-only bind enforced
(`127.0.0.1`/`localhost`/`::1`); anything else is refused at construction.

## Endpoints

| Method | Path | Body | Success | Errors |
|---|---|---|---|---|
| GET | `/health` | — | 200 `HealthReport` | 500 |
| GET | `/status` | — | 200 `RuntimeStatus` | 500 |
| GET | `/metrics` | — | 200 `RuntimeMetrics` | 500 |
| GET | `/map/query?x=&y=` | — | 200 `QueryResult` | 400 bad coords, 409 no snapshot |
| POST | `/frames` | `{"pts": [[x,y,z]..], "intensity"?, "ring"?, "pose"?, "sensor_origin"?, "timestamp"?, "frame_id"?}` | 200 snapshot dict | 400 validation, 409 inactive, 422 mapping failure |
| POST | `/reset` | — | 200 status | 409 illegal state |
| POST | `/lifecycle` | `{"action": "configure"|"start"|"stop"}` | 200 status | 400 bad action, 409 illegal transition |

No shutdown endpoint (server lifetime belongs to the embedder via `stop()`).

## Limits

- JSON bodies capped at 8 MiB (413-style 400 `Body too large`).
- Point arrays capped at 200,000 points; empty lists rejected.
- Coordinates must be finite; timestamps/poses validated by `LiDARFrame`.

## Errors

Typed JSON `{api_version, error, type}` with HTTP codes; tracebacks never
leave the server. Nothing is deserialized with pickle; no filesystem, model,
or code-execution surface exists.

## Testing

Covered in `tests/test_phase9_sdk.py` against a live stdlib server:
endpoints, validation, size/point limits, error mapping, lifecycle
transitions, reset, and bind refusal.
