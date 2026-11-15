"""Requests and results for one Grok Desktop exchange."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ExchangeStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    UNCERTAIN = "uncertain"
    ERROR = "error"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ExchangeRequest:
    """One send to a bot the caller already identified.

    ``request_id`` must occur inside ``message``. It is the correlation tag
    and the idempotency key. A repeated id is never submitted again.
    """

    agent_id: str
    bot_name: str
    message: str
    request_id: str
    source_document_id: str | None = None


@dataclass(frozen=True)
class ExchangeResult:
    status: ExchangeStatus
    sent: bool
    request_id: str
    agent_id: str
    reply: str = ""
    raw_reply: str = ""
    reply_preview: str = ""
    reason: str = ""
    states_seen: tuple[tuple[str, ...], ...] = ()
    replayed: bool = False
    detail: str = ""
    cleanup: str = "unrecorded"
    baseline: str = "unrecorded"

    def to_public_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "sent": self.sent,
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "reply": self.reply,
            "raw_reply": self.raw_reply,
            "reply_preview": self.reply_preview,
            "reason": self.reason,
            "states_seen": [list(item) for item in self.states_seen],
            "replayed": self.replayed,
            "detail": self.detail,
            "cleanup": self.cleanup,
            "baseline": self.baseline,
        }
