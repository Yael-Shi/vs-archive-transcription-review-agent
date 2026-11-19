"""Foreground identity for one IPC call.

The helper records the foreground window. It does not call
``SetForegroundWindow``, ``ShowWindow``, or send input.
"""

from __future__ import annotations

from transcription_review_agent.codex_desktop.models import (
    FocusObservation,
    ForegroundWindow,
)


def is_codex_desktop(executable: str) -> bool:
    normalized = executable.replace("/", "\\").casefold()
    return normalized.endswith("\\chatgpt.exe") and "\\openai.codex_" in normalized


def focus_verdict(observation: FocusObservation) -> str:
    """Return preserved, codex_activated, or foreground_changed."""

    if observation.preserved:
        return "preserved"
    if is_codex_desktop(observation.after.executable) and (
        observation.after.hwnd != observation.before.hwnd
    ):
        return "codex_activated"
    return "foreground_changed"


def parse_window(payload: object) -> ForegroundWindow:
    if not isinstance(payload, dict):
        raise ValueError("foreground window is not an object")
    hwnd = payload.get("hwnd")
    pid = payload.get("pid")
    executable = payload.get("executable")
    if not isinstance(hwnd, int) or not isinstance(pid, int):
        raise ValueError("foreground window identity is incomplete")
    if not isinstance(executable, str):
        raise ValueError("foreground executable is missing")
    return ForegroundWindow(hwnd=hwnd, pid=pid, executable=executable)


def parse_observation(payload: object) -> FocusObservation:
    if not isinstance(payload, dict):
        raise ValueError("focus observation is not an object")
    return FocusObservation(
        before=parse_window(payload.get("before")),
        after=parse_window(payload.get("after")),
    )
