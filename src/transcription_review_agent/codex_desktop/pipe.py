"""Windows named-pipe calls for the running Codex Desktop app.

The PowerShell helper is hidden and does not activate Desktop. Tests inject
``call`` instead of launching that helper.
"""

from __future__ import annotations

import base64
import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from transcription_review_agent.codex_desktop.focus import parse_observation
from transcription_review_agent.codex_desktop.framing import MAX_FRAME_BYTES, FrameError
from transcription_review_agent.codex_desktop.models import FocusObservation

ASSETS = Path(__file__).resolve().parent / "assets"
HELPER = ASSETS / "codex-pipe.ps1"
_CODEX_EXE = "chatgpt.exe"
_PACKAGE_MARK = "\\windowsapps\\openai.codex_"


class PipeError(RuntimeError):
    """The helper failed before a JSON-RPC result was available."""


@dataclass(frozen=True, slots=True)
class PipeEndpoint:
    name: str
    pid: int
    executable: str


@dataclass(frozen=True, slots=True)
class PipeCall:
    response: dict[str, Any]
    focus: FocusObservation


class CodexTransport(Protocol):
    def call(
        self, method: str, params: dict[str, Any], *, timeout_s: float
    ) -> PipeCall: ...


def trusted_desktop_owner(executable: str) -> bool:
    normalized = executable.replace("/", "\\").casefold()
    return normalized.endswith("\\" + _CODEX_EXE) and _PACKAGE_MARK in normalized


def rpc_body(method: str, params: dict[str, Any], rpc_id: str) -> bytes:
    """UTF-8 JSON-RPC object. The id is this call, not a Desktop turn."""

    body = json.dumps(
        {"id": rpc_id, "jsonrpc": "2.0", "method": method, "params": params},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError("frame is too large")
    return body


def parse_response_bytes(
    raw: bytes, *, expected_id: str | None = None
) -> dict[str, Any]:
    """Parse one IPC response exactly as the pipe returned it.

    When ``expected_id`` is set, the JSON-RPC ``id`` must be that exact string.
    A missing, numeric, or different id is not a result for this call.
    """

    if len(raw) > MAX_FRAME_BYTES:
        raise FrameError("frame is too large")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError("frame is not JSON") from exc
    if not isinstance(value, dict):
        raise FrameError("frame JSON is not an object")
    if expected_id is not None:
        response_id = value.get("id")
        if not isinstance(response_id, str) or response_id != expected_id:
            raise FrameError("JSON-RPC response id does not match the request")
    return value


def select_endpoint(candidates: list[PipeEndpoint]) -> PipeEndpoint:
    trusted = [item for item in candidates if trusted_desktop_owner(item.executable)]
    if len(trusted) != 1:
        raise PipeError("expected one Codex Desktop app-tools pipe")
    return trusted[0]


class PowershellCodexTransport:
    """One helper process per call. The helper never shows a window."""

    def __init__(self, *, powershell: str = "powershell.exe") -> None:
        self.powershell = powershell

    def discover(self) -> list[PipeEndpoint]:
        payload = self._run({"action": "discover"}, timeout_s=20)
        raw = payload.get("pipes")
        if not isinstance(raw, list):
            raise PipeError("pipe discovery returned no list")
        found: list[PipeEndpoint] = []
        for item in raw:
            if not isinstance(item, dict):
                raise PipeError("pipe discovery entry is invalid")
            name = item.get("name")
            pid = item.get("pid")
            executable = item.get("executable")
            if (
                not isinstance(name, str)
                or not isinstance(pid, int)
                or not isinstance(executable, str)
            ):
                raise PipeError("pipe discovery entry is invalid")
            found.append(PipeEndpoint(name=name, pid=pid, executable=executable))
        return found

    def call(
        self, method: str, params: dict[str, Any], *, timeout_s: float
    ) -> PipeCall:
        rpc_id = uuid.uuid4().hex
        frame = rpc_body(method, params, rpc_id)
        payload = self._run(
            {
                "action": "call",
                "frameBase64": base64.b64encode(frame).decode("ascii"),
                "timeoutMs": int(timeout_s * 1000),
            },
            timeout_s=timeout_s + 15,
        )
        encoded = payload.get("responseBase64")
        if not isinstance(encoded, str):
            raise PipeError("pipe call returned no response bytes")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise PipeError("pipe response bytes are not base64") from exc
        response = parse_response_bytes(raw, expected_id=rpc_id)
        return PipeCall(
            response=response, focus=parse_observation(payload.get("focus"))
        )

    def _run(self, request: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        if not HELPER.is_file():
            raise PipeError("Codex pipe helper is missing")
        command = [
            self.powershell,
            "-NoProfile",
            "-NonInteractive",
            "-WindowStyle",
            "Hidden",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(HELPER),
        ]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PipeError("Codex pipe helper timed out") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise PipeError(detail or "Codex pipe helper failed")
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise PipeError("Codex pipe helper returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise PipeError("Codex pipe helper returned a non-object")
        if payload.get("ok") is not True:
            message = payload.get("error")
            raise PipeError(
                message if isinstance(message, str) else "pipe helper error"
            )
        return payload
