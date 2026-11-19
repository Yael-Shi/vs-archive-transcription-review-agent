"""Length-prefixed JSON-RPC frames for the Codex app-tools pipe."""

from __future__ import annotations

import json
from typing import Any

MAX_FRAME_BYTES = 8 * 1024 * 1024


class FrameError(ValueError):
    """A frame is too large or is not one JSON value."""


def encode_frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError("frame is too large")
    return len(body).to_bytes(4, "little") + body


def decode_frame(buffer: bytes) -> tuple[dict[str, Any], bytes]:
    """Return one JSON object and the unconsumed tail."""

    if len(buffer) < 4:
        raise FrameError("frame header is incomplete")
    length = int.from_bytes(buffer[:4], "little")
    if length > MAX_FRAME_BYTES:
        raise FrameError("frame is too large")
    if len(buffer) < 4 + length:
        raise FrameError("frame body is incomplete")
    try:
        value = json.loads(buffer[4 : 4 + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError("frame is not JSON") from exc
    if not isinstance(value, dict):
        raise FrameError("frame JSON is not an object")
    return value, buffer[4 + length :]
