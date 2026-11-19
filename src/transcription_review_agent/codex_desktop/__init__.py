"""Codex Desktop IPC adapter. It does not use the Grok profile or the Codex CLI."""

from transcription_review_agent.codex_desktop.client import CodexDesktopClient
from transcription_review_agent.codex_desktop.models import (
    CodexRequest,
    CodexResult,
    CodexStatus,
)

__all__ = [
    "CodexDesktopClient",
    "CodexRequest",
    "CodexResult",
    "CodexStatus",
]
