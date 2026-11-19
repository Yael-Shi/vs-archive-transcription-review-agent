"""One Codex Desktop turn. This is not a review-ledger record."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CodexStatus(str, Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    UNCERTAIN = "uncertain"
    ERROR = "error"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ForegroundWindow:
    hwnd: int
    pid: int
    executable: str


@dataclass(frozen=True, slots=True)
class FocusObservation:
    before: ForegroundWindow
    after: ForegroundWindow

    @property
    def preserved(self) -> bool:
        return self.before.hwnd == self.after.hwnd and self.before.pid == self.after.pid


@dataclass(frozen=True, slots=True)
class CodexRequest:
    """One message for one existing Desktop thread.

    ``request_id`` must occur inside ``message``. It is the correlation tag
    and the idempotency key.
    """

    thread_id: str
    request_id: str
    message: str
    image_sha256: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodexResult:
    status: CodexStatus
    sent: bool
    request_id: str
    thread_id: str
    reply: str = ""
    raw_reply: str = ""
    turn_id: str = ""
    dispatch_turn_id: str = ""
    reason: str = ""
    replayed: bool = False
    focus_preserved: bool | None = None

    def to_public_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "sent": self.sent,
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            "reply": self.reply,
            "raw_reply": self.raw_reply,
            "turn_id": self.turn_id,
            "dispatch_turn_id": self.dispatch_turn_id,
            "reason": self.reason,
            "replayed": self.replayed,
            "focus_preserved": self.focus_preserved,
        }
