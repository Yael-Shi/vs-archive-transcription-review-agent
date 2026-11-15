"""Isolated Grok Bot Desktop adapter.

This package is not used by the review pipeline. It sends at most one message
to a predeclared bot and records agent-owned resources locally. It does not
delete a remote bot.
"""

from transcription_review_agent.grok_desktop.adapter import GrokDesktopAdapter
from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)

__all__ = [
    "ExchangeRequest",
    "ExchangeResult",
    "ExchangeStatus",
    "GrokDesktopAdapter",
]
