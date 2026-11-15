"""Persistent registry of resources and send attempts created by this adapter.

Historical or manually created bots are not added by a send. Ownership is
recorded only through ``register_owned``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import IO

from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)


class RegistryError(Exception):
    """The registry file or a requested record is inconsistent."""


class DuplicateResourceError(RegistryError):
    """This local id or provider id is already registered."""


class ResourceKind(str, Enum):
    GROK_BOT = "grok_bot"
    GROK_CHAT = "grok_chat"
    CODEX_EXEC = "codex_exec"


class ResourceState(str, Enum):
    ACTIVE = "ACTIVE"
    RETAINED = "RETAINED"
    ELIGIBLE_FOR_CLEANUP = "ELIGIBLE_FOR_CLEANUP"
    DELETED = "DELETED"
    FAILED_CREATE = "FAILED_CREATE"
    FAILED_SEND = "FAILED_SEND"
    FAILED_DELETE = "FAILED_DELETE"


@dataclass(frozen=True)
class ResourceRecord:
    local_id: str
    provider: str
    kind: ResourceKind
    provider_id: str
    source_document_id: str | None
    state: ResourceState
    ownership_evidence: str
    outputs_verified: bool = False
    retain_reason: str = ""

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> ResourceRecord:
        source = payload.get("source_document_id")
        if source is not None and not isinstance(source, str):
            raise RegistryError("invalid source_document_id")
        try:
            kind = ResourceKind(_text(payload, "kind"))
            state = ResourceState(_text(payload, "state"))
        except ValueError as exc:
            raise RegistryError("invalid resource enum") from exc
        verified = payload.get("outputs_verified", False)
        if not isinstance(verified, bool):
            raise RegistryError("invalid outputs_verified")
        provider = _text(payload, "provider")
        if provider not in {"grok", "codex"}:
            raise RegistryError("unknown provider")
        return cls(
            local_id=_text(payload, "local_id"),
            provider=provider,
            kind=kind,
            provider_id=_text(payload, "provider_id"),
            source_document_id=source,
            state=state,
            ownership_evidence=_text(payload, "ownership_evidence"),
            outputs_verified=verified,
            retain_reason=_text(payload, "retain_reason", empty=True),
        )


@dataclass(frozen=True)
class StoredAttempt:
    fingerprint: str
    result: ExchangeResult


def request_fingerprint(request: ExchangeRequest) -> str:
    """Identity of one send. The request id is stored beside this, not inside it."""

    body = {
        "agent_id": request.agent_id,
        "bot_name": " ".join(request.bot_name.split()),
        "message": request.message,
        "source_document_id": request.source_document_id,
    }
    encoded = json.dumps(
        body, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ResourceRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._resources: dict[str, ResourceRecord] = {}
        self._attempts: dict[str, StoredAttempt] = {}
        if path.is_file():
            self._resources, self._attempts = self._read_disk()

    def register_owned(
        self,
        *,
        local_id: str,
        provider: str,
        kind: ResourceKind,
        provider_id: str,
        ownership_evidence: str,
        source_document_id: str | None = None,
    ) -> ResourceRecord:
        if provider not in {"grok", "codex"}:
            raise RegistryError("unknown provider")
        if not local_id or not provider_id or not ownership_evidence.strip():
            raise RegistryError("ownership evidence is required")
        with self._locked():
            self._reload()
            if local_id in self._resources:
                raise DuplicateResourceError(local_id)
            for record in self._resources.values():
                if (
                    record.provider == provider
                    and record.kind is kind
                    and record.provider_id == provider_id
                ):
                    raise DuplicateResourceError(provider_id)
            record = ResourceRecord(
                local_id=local_id,
                provider=provider,
                kind=kind,
                provider_id=provider_id,
                source_document_id=source_document_id,
                state=ResourceState.ACTIVE,
                ownership_evidence=ownership_evidence.strip(),
            )
            self._resources[local_id] = record
            self._commit()
            return record

    def get(self, local_id: str) -> ResourceRecord:
        try:
            return self._resources[local_id]
        except KeyError as exc:
            raise RegistryError(f"unknown resource {local_id}") from exc

    def replace(self, record: ResourceRecord) -> ResourceRecord:
        """Store one legal lifecycle step. Identity and ownership stay put."""

        with self._locked():
            self._reload()
            current = self._resources.get(record.local_id)
            if current is None:
                raise RegistryError(f"unknown resource {record.local_id}")
            _require_legal_update(current, record)
            self._resources[record.local_id] = record
            self._commit()
            return record

    def begin_attempt(self, request: ExchangeRequest) -> ExchangeResult | None:
        """Reserve ``request_id`` before any CDP call.

        The same fingerprint replays the stored result and does not send.
        A different message, bot, or document under that id is rejected and
        the stored attempt is left unchanged.
        """

        fingerprint = request_fingerprint(request)
        with self._locked():
            self._reload()
            existing = self._attempts.get(request.request_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    return ExchangeResult(
                        status=ExchangeStatus.ERROR,
                        sent=False,
                        request_id=request.request_id,
                        agent_id=request.agent_id,
                        reason="request_id_reused",
                        replayed=True,
                    )
                stored = existing.result
                return ExchangeResult(
                    status=stored.status,
                    sent=stored.sent,
                    request_id=stored.request_id,
                    agent_id=stored.agent_id,
                    reply=stored.reply,
                    raw_reply=stored.raw_reply,
                    reply_preview=stored.reply_preview,
                    reason=stored.reason or "replayed",
                    states_seen=stored.states_seen,
                    replayed=True,
                    detail=stored.detail,
                    cleanup=stored.cleanup,
                    baseline=stored.baseline,
                )
            placeholder = ExchangeResult(
                status=ExchangeStatus.UNCERTAIN,
                sent=True,
                request_id=request.request_id,
                agent_id=request.agent_id,
                reason="attempt_already_started",
            )
            self._attempts[request.request_id] = StoredAttempt(fingerprint, placeholder)
            self._commit()
            return None

    def finish_attempt(self, result: ExchangeResult) -> None:
        with self._locked():
            self._reload()
            existing = self._attempts.get(result.request_id)
            if existing is None:
                raise RegistryError("attempt was not reserved")
            if (
                result.request_id != existing.result.request_id
                or result.agent_id != existing.result.agent_id
            ):
                raise RegistryError("attempt identity changed")
            self._attempts[result.request_id] = StoredAttempt(
                existing.fingerprint, result
            )
            self._commit()

    def _reload(self) -> None:
        if self.path.is_file():
            self._resources, self._attempts = self._read_disk()
        else:
            self._resources = {}
            self._attempts = {}

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        handle = open(lock_path, "a+b")
        try:
            _lock_handle(handle)
            yield
        finally:
            _unlock_handle(handle)
            handle.close()

    def _read_disk(self) -> tuple[dict[str, ResourceRecord], dict[str, StoredAttempt]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError("registry corrupt") from exc
        if not isinstance(payload, dict):
            raise RegistryError("registry shape")
        resources = payload.get("resources")
        attempts = payload.get("attempts")
        if not isinstance(resources, list) or not isinstance(attempts, list):
            raise RegistryError("registry shape")
        loaded_resources: dict[str, ResourceRecord] = {}
        provider_ids: set[tuple[str, str, str]] = set()
        for item in resources:
            if not isinstance(item, dict):
                raise RegistryError("registry corrupt")
            record = ResourceRecord.from_json(item)
            if record.local_id in loaded_resources:
                raise RegistryError("duplicate local id")
            provider_key = (record.provider, record.kind.value, record.provider_id)
            if provider_key in provider_ids:
                raise RegistryError("duplicate provider id")
            provider_ids.add(provider_key)
            loaded_resources[record.local_id] = record
        loaded_attempts: dict[str, StoredAttempt] = {}
        for item in attempts:
            if not isinstance(item, dict):
                raise RegistryError("registry corrupt")
            fingerprint = item.get("fingerprint")
            result_payload = item.get("result")
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(char not in "0123456789abcdef" for char in fingerprint)
                or not isinstance(result_payload, dict)
            ):
                raise RegistryError("attempt fingerprint missing")
            result = _result_from_json(result_payload)
            if result.request_id in loaded_attempts:
                raise RegistryError("duplicate request id")
            loaded_attempts[result.request_id] = StoredAttempt(fingerprint, result)
        return loaded_resources, loaded_attempts

    def _commit(self) -> None:
        try:
            self._save()
        except RegistryError:
            self._reload()
            raise

    def _save(self) -> None:
        payload = {
            "resources": [record.to_json() for record in self._resources.values()],
            "attempts": [
                {
                    "fingerprint": attempt.fingerprint,
                    "result": _result_to_json(attempt.result),
                }
                for attempt in self._attempts.values()
            ],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        try:
            _write_complete(temporary, data)
        except RegistryError:
            _discard(temporary)
            raise
        try:
            os.replace(temporary, self.path)
        except OSError as exc:
            _discard(temporary)
            raise RegistryError("registry replace failed") from exc
        try:
            _fsync_directory(self.path.parent)
        except OSError as exc:
            raise RegistryError("registry directory sync failed") from exc


def _write_complete(path: Path, data: bytes) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    except OSError as exc:
        raise RegistryError("registry write failed") from exc
    try:
        remaining = data
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise RegistryError("registry write incomplete")
            remaining = remaining[written:]
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise RegistryError("registry write failed") from exc
    except OSError as exc:
        raise RegistryError("registry write failed") from exc
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    handle = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _discard(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        return


def _lock_handle(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        if handle.read(1) != b"\0":
            handle.seek(0)
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_handle(handle: IO[bytes]) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _optional_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise RegistryError(f"invalid {key}")
    return value


def _cleanup_text(payload: dict[str, object]) -> str:
    if "cleanup" not in payload:
        return "unrecorded"
    value = payload.get("cleanup")
    if value not in {"closed", "uncertain", "failed", "unrecorded"}:
        raise RegistryError("invalid cleanup")
    return str(value)


def _baseline_text(payload: dict[str, object]) -> str:
    if "baseline" not in payload:
        return "unrecorded"
    value = payload.get("baseline")
    if value not in {"intact", "failed", "unrecorded"}:
        raise RegistryError("invalid baseline")
    return str(value)


def _text(payload: dict[str, object], key: str, *, empty: bool = False) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise RegistryError(f"invalid {key}")
    return value


def _require_legal_update(current: ResourceRecord, proposed: ResourceRecord) -> None:
    identity = (
        current.local_id,
        current.provider,
        current.kind,
        current.provider_id,
        current.source_document_id,
        current.ownership_evidence,
    )
    proposed_identity = (
        proposed.local_id,
        proposed.provider,
        proposed.kind,
        proposed.provider_id,
        proposed.source_document_id,
        proposed.ownership_evidence,
    )
    if identity != proposed_identity:
        raise RegistryError("resource identity is immutable")
    if proposed == current:
        return
    if (
        current.state is ResourceState.ACTIVE
        and proposed.state is ResourceState.ACTIVE
        and proposed.outputs_verified
        and not current.outputs_verified
        and proposed.retain_reason == current.retain_reason
    ):
        return
    from transcription_review_agent.grok_desktop.lifecycle import (
        LifecycleError,
        apply_transition,
    )

    try:
        stepped = apply_transition(
            current, proposed.state, retain_reason=proposed.retain_reason
        )
    except LifecycleError as exc:
        raise RegistryError("state change is not allowed") from exc
    if stepped != proposed:
        raise RegistryError("state change does not match the lifecycle")


def _result_to_json(result: ExchangeResult) -> dict[str, object]:
    return result.to_public_dict()


def _result_from_json(payload: dict[str, object]) -> ExchangeResult:
    sent = payload.get("sent")
    if not isinstance(sent, bool):
        raise RegistryError("invalid sent")
    raw_states = payload.get("states_seen")
    if not isinstance(raw_states, list):
        raise RegistryError("invalid states_seen")
    states: list[tuple[str, ...]] = []
    for item in raw_states:
        if not isinstance(item, list) or not all(
            isinstance(part, str) for part in item
        ):
            raise RegistryError("invalid states_seen")
        states.append(tuple(item))
    try:
        status = ExchangeStatus(_text(payload, "status"))
    except ValueError as exc:
        raise RegistryError("invalid status") from exc
    return ExchangeResult(
        status=status,
        sent=sent,
        request_id=_text(payload, "request_id"),
        agent_id=_text(payload, "agent_id"),
        reply=_text(payload, "reply", empty=True),
        raw_reply=_text(payload, "raw_reply", empty=True),
        reply_preview=_text(payload, "reply_preview", empty=True),
        reason=_text(payload, "reason", empty=True),
        states_seen=tuple(states),
        replayed=False,
        detail=_optional_text(payload, "detail"),
        cleanup=_cleanup_text(payload),
        baseline=_baseline_text(payload),
    )
