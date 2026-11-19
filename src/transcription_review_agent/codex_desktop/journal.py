"""Exactly-once record for Codex Desktop sends.

This file remembers one transport attempt. It is not a review ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from transcription_review_agent.codex_desktop.models import (
    CodexRequest,
    CodexResult,
    CodexStatus,
)


class JournalError(RuntimeError):
    """The attempt file cannot be updated safely."""


@dataclass(frozen=True, slots=True)
class StoredAttempt:
    fingerprint: str
    result: CodexResult


def request_fingerprint(request: CodexRequest) -> str:
    body: dict[str, object] = {
        "message": request.message,
        "request_id": request.request_id,
        "thread_id": request.thread_id,
    }
    if request.image_sha256:
        body["image_sha256"] = list(request.image_sha256)
    encoded = json.dumps(
        body, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class AttemptJournal:
    def __init__(self, path: Path) -> None:
        self.path = path

    def reserve(self, request: CodexRequest) -> CodexResult | None:
        """Store an uncertain attempt before the pipe write.

        The same fingerprint returns the stored result and must not be sent
        again. A different message under that id is an error and is not sent.
        """

        fingerprint = request_fingerprint(request)
        with self._locked():
            attempts = self._read()
            existing = attempts.get(request.request_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    return _copy(
                        existing.result,
                        status=CodexStatus.ERROR,
                        sent=False,
                        reason="request_id_reused",
                        replayed=True,
                    )
                return _copy(existing.result, replayed=True)
            placeholder = CodexResult(
                status=CodexStatus.UNCERTAIN,
                sent=True,
                request_id=request.request_id,
                thread_id=request.thread_id,
                reason="attempt_started",
            )
            attempts[request.request_id] = StoredAttempt(fingerprint, placeholder)
            self._write(attempts)
            return None

    def finish(self, result: CodexResult) -> None:
        with self._locked():
            attempts = self._read()
            existing = attempts.get(result.request_id)
            if existing is None:
                raise JournalError("attempt was not reserved")
            if (
                result.request_id != existing.result.request_id
                or result.thread_id != existing.result.thread_id
            ):
                raise JournalError("attempt identity changed")
            attempts[result.request_id] = StoredAttempt(existing.fingerprint, result)
            self._write(attempts)

    def _read(self) -> dict[str, StoredAttempt]:
        if not self.path.is_file():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw = payload.get("attempts")
        if not isinstance(raw, dict):
            raise JournalError("attempt journal is invalid")
        found: dict[str, StoredAttempt] = {}
        for key, item in raw.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                raise JournalError("attempt journal is invalid")
            fingerprint = item.get("fingerprint")
            result = item.get("result")
            if not isinstance(fingerprint, str) or not isinstance(result, dict):
                raise JournalError("attempt journal is invalid")
            found[key] = StoredAttempt(fingerprint, _result(result))
        return found

    def _write(self, attempts: dict[str, StoredAttempt]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "attempts": {
                key: {
                    "fingerprint": item.fingerprint,
                    "result": item.result.to_public_dict(),
                }
                for key, item in attempts.items()
            }
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(body, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 5
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise JournalError("attempt journal is locked") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)


def _result(payload: dict[str, object]) -> CodexResult:
    status = payload.get("status")
    sent = payload.get("sent")
    request_id = payload.get("request_id")
    thread_id = payload.get("thread_id")
    if not isinstance(status, str) or not isinstance(sent, bool):
        raise JournalError("stored attempt is invalid")
    if not isinstance(request_id, str) or not isinstance(thread_id, str):
        raise JournalError("stored attempt is invalid")
    focus = payload.get("focus_preserved")
    if focus is not None and not isinstance(focus, bool):
        raise JournalError("stored attempt is invalid")
    return CodexResult(
        status=CodexStatus(status),
        sent=sent,
        request_id=request_id,
        thread_id=thread_id,
        reply=_text(payload.get("reply")),
        raw_reply=_text(payload.get("raw_reply")),
        turn_id=_text(payload.get("turn_id")),
        dispatch_turn_id=_text(payload.get("dispatch_turn_id")),
        reason=_text(payload.get("reason")),
        replayed=payload.get("replayed") is True,
        focus_preserved=focus,
    )


def _text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise JournalError("stored attempt is invalid")
    return value


def _copy(
    result: CodexResult,
    *,
    status: CodexStatus | None = None,
    sent: bool | None = None,
    reason: str | None = None,
    replayed: bool | None = None,
) -> CodexResult:
    return CodexResult(
        status=result.status if status is None else status,
        sent=result.sent if sent is None else sent,
        request_id=result.request_id,
        thread_id=result.thread_id,
        reply=result.reply,
        raw_reply=result.raw_reply,
        turn_id=result.turn_id,
        dispatch_turn_id=result.dispatch_turn_id,
        reason=result.reason if reason is None else reason,
        replayed=result.replayed if replayed is None else replayed,
        focus_preserved=result.focus_preserved,
    )
