"""Checks that a launch can only use the isolated Grok profile."""

from __future__ import annotations

import os
import re
from pathlib import Path

ISOLATED_MARKER = "grok-bot-isolated-pilot"
_PERMISSION = re.compile(r'"localToolPermission"\s*:\s*"(always|ask|never)"')


class ProfileError(Exception):
    """The isolated profile is missing, or a path belongs to the normal install."""


def read_permission(text: str) -> str:
    found = _PERMISSION.findall(text)
    if len(found) != 1:
        return "unreadable"
    return found[0]


def assert_isolated_profile(
    profile_dir: Path,
    *,
    marker_text: str,
    pilot_settings: str,
    live_settings: str,
) -> None:
    """Refuse the normal profile, a bad marker, or a permission other than never."""

    normalized = str(profile_dir).replace("/", "\\").lower()
    if "\\appdata\\roaming\\grok bot" in normalized or "\\.grokbot" in normalized:
        raise ProfileError("refusing the normal Grok profile")
    if "grok bot isolated pilot" not in normalized:
        raise ProfileError("profile is not the isolated pilot")
    if marker_text.strip() != ISOLATED_MARKER:
        raise ProfileError("isolated marker mismatch")
    if read_permission(pilot_settings) != "never":
        raise ProfileError("isolated local execution is not never")
    if read_permission(live_settings) != "ask":
        raise ProfileError("normal local execution permission changed")


def local_read_path(path: Path) -> Path:
    """Map a Windows path so a WSL caller can read the same profile files.

    The PowerShell host still receives the original Windows path.
    """

    raw = str(path)
    if os.name == "nt":
        return path
    match = re.fullmatch(r"([A-Za-z]):[\\/](.*)", raw)
    if match is None:
        return path
    tail = match.group(2).replace("\\", "/")
    return Path("/mnt") / match.group(1).lower() / tail


def load_profile_text(path: Path) -> str:
    return local_read_path(path).read_text(encoding="utf-8")
