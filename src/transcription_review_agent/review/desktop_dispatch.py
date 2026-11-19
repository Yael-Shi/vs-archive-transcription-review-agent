"""Send one registered desktop task through a provider sender, then install the reply.

This is the neutral half of automatic desktop dispatch. It names no desktop
application. A sender (see ``desktop_orchestration``) turns a published task
into one message for its own interface and performs the one send. This module
then

* checks the sealed image bytes before and after the send,
* refuses a message that names the other provider's task or the archive text,
* preserves the raw reply bytes before any schema decision,
* writes a provenance record beside them,
* copies the unmodified reply into the queue output slot only when the
  existing first-pass or challenge check accepts it.

The controller transitions stay in ``ReviewController`` and ``document_run``.
Nothing here writes the review ledger, repairs JSON, or resends. The transport
journals (Codex attempt journal, Grok resource registry) are the exactly-once
record for a send. A raw reply that is already preserved for a request id is
reused and the sender is not called again.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.desktop_result_intake import _place_exact
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPolicyError,
)
from transcription_review_agent.review.models import CropRecord, WorkRequest
from transcription_review_agent.review.paths import stage_dir_name

PROVENANCE_SCHEMA = "desktop-dispatch-provenance-v1"
ATTEMPT_SCHEMA = "desktop-dispatch-attempt-v1"
VERIFIED_SCHEMA = "desktop-dispatch-post-send-verified-v1"
BLOCKED_SCHEMA = "desktop-dispatch-integrity-blocked-v1"
ReplyStatus = Literal["reply", "uncertain", "not_sent", "blocked"]
Outcome = Literal[
    "installed",
    "invalid_reply",
    "uncertain",
    "not_sent",
    "blocked",
    "integrity_blocked",
]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_FIRST_PASS_JSON = "Write one JSON file. Dispatch is manual. No API is assumed.\n"
_FIRST_PASS_WRITE = re.compile(
    r"Write the model transcription to this Windows file and to no other path:\n"
    r"[^\n]+\n"
)
_FIRST_PASS_INDEX = (
    "Before transcribing, confirm that the parent directory contains "
    "desktop_queue.json.\n"
    "That index is the publication marker. If it is missing, or this "
    "directory is under .partial, stop and write nothing.\n"
)
_CHALLENGE_WRITE = re.compile(
    r"Write one JSON file and no other path:\n[^\n]+\n"
    r"If challenge_queue\.json is missing from the parent directory, "
    r"stop and write nothing\. Do not open the other provider directory\.\n"
)


def transport_request_id(task_id: str) -> str:
    """Correlation id for one send. It is derived from the exact task id."""

    return task_id.replace(":", ".")


@dataclass(frozen=True, slots=True)
class DispatchTask:
    """One published provider task and the facts a send is checked against."""

    request: WorkRequest
    task_dir: Path
    windows_task_dir: str
    output_path: Path
    packet_dir: Path
    windows_packet_dir: str
    evidence_sha256: tuple[str, ...]
    forbidden_texts: tuple[str, ...]

    @property
    def provider(self) -> str:
        return self.request.provider

    @property
    def stage(self) -> str:
        return self.request.stage

    @property
    def round_index(self) -> int:
        return self.request.round_index

    @property
    def request_id(self) -> str:
        return transport_request_id(self.request.task_id)


@dataclass(frozen=True, slots=True)
class PreparedSend:
    """The exact message and files a sender will submit. Nothing is sent yet."""

    request_id: str
    message: str
    attachments: tuple[Path, ...]
    mechanism: str


@dataclass(frozen=True, slots=True)
class DispatchReply:
    """What a sender observed. ``raw_reply`` is the model text, unchanged."""

    status: ReplyStatus
    raw_reply: str
    reason: str
    transport: Mapping[str, object]


class DesktopSender(Protocol):
    def prepare(self, task: DispatchTask) -> PreparedSend: ...

    def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply: ...


@dataclass(frozen=True, slots=True)
class DispatchRecord:
    provider: str
    stage: str
    round_index: int
    task_id: str
    request_id: str
    outcome: Outcome
    reason: str | None
    raw_sha256: str | None
    raw_path: str | None
    installed_path: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "installed_path": self.installed_path,
            "outcome": self.outcome,
            "provider": self.provider,
            "raw_path": self.raw_path,
            "raw_sha256": self.raw_sha256,
            "reason": self.reason,
            "request_id": self.request_id,
            "round_index": self.round_index,
            "stage": self.stage,
            "task_id": self.task_id,
        }


def dispatch_task(
    sender: DesktopSender,
    task: DispatchTask,
    *,
    preservation_dir: Path,
    validate: Callable[[bytes, WorkRequest], str | None],
) -> DispatchRecord:
    """Send once, preserve the raw reply, and install it only when it validates."""

    paths = _preservation_paths(preservation_dir, task)
    if os.path.lexists(paths.blocked):
        return _record(
            task,
            "integrity_blocked",
            _blocked_reason(paths.blocked),
            raw_path=paths.raw if os.path.lexists(paths.raw) else None,
        )
    raw_present = _require_regular_or_absent(paths.raw, "preserved reply")
    try:
        _require_evidence(task)
    except ReviewIntegrityError as exc:
        if raw_present:
            _block(paths, task, str(exc))
        raise
    prepared = sender.prepare(task)
    _check_prepared(task, prepared)
    base = _base_record(task, prepared)
    if raw_present:
        raw = paths.raw.read_bytes()
        _settle_provenance(paths, base, raw)
        unverified = _unverified_reason(paths, base, raw)
        if unverified is not None:
            # The first execution never finished its post-send evidence check.
            # A restored image cannot show that it was safe, so nothing is
            # installed and nothing is sent again.
            _block(paths, task, unverified, raw)
            return _record(
                task,
                "integrity_blocked",
                f"integrity blocked: {unverified}",
                raw_path=paths.raw,
                raw=raw,
            )
    else:
        _publish_or_verify(paths.attempt, {**base, "schema_version": ATTEMPT_SCHEMA})
        reply = sender.send(task, prepared)
        if reply.status != "reply":
            return _record(task, reply.status, reply.reason or reply.status)
        raw = reply.raw_reply.encode("utf-8")
        _place_exact(paths.raw, raw)
        _publish_provenance(paths.provenance, base, raw, dict(reply.transport))
    raw_path = paths.raw
    try:
        _require_evidence(task)
    except ReviewIntegrityError as exc:
        # The reply stays as audit evidence. It is never installed or reused.
        _block(paths, task, str(exc), raw)
        raise
    if not os.path.lexists(paths.verified):
        _write_new(paths.verified, _verified_body(base, raw))
    failure = validate(raw, task.request)
    if failure is not None:
        return _record(
            task,
            "invalid_reply",
            failure,
            raw_path=raw_path,
            raw=raw,
        )
    _place_exact(task.output_path, raw)
    return _record(
        task,
        "installed",
        None,
        raw_path=raw_path,
        raw=raw,
        installed=task.output_path,
    )


def reply_delivery_prompt(prompt: str, stage: str) -> str:
    """Turn "write one file" into "reply with one JSON object".

    Only the delivery lines change. Review instructions are untouched. Each
    delivery line must match exactly once, so a prompt that drifts from the
    published wording is refused instead of silently adapted.
    """

    if stage == "first_pass":
        text = _swap_once(
            prompt,
            _FIRST_PASS_JSON,
            "Return exactly one JSON object as the final answer and no other text. "
            "Dispatch is manual. No API is assumed.\n",
        )
        text = _swap_pattern(
            text,
            _FIRST_PASS_WRITE,
            "Do not write any file. The final answer is the model transcription.\n",
        )
        return _swap_once(text, _FIRST_PASS_INDEX, "")
    if stage == "challenge":
        return _swap_pattern(
            prompt,
            _CHALLENGE_WRITE,
            "Do not write any file. The final answer is that one JSON object "
            "and no other text.\n"
            "Do not open the other provider directory.\n",
        )
    raise ReviewPolicyError("desktop dispatch covers first_pass and challenge")


def identity_values(request: WorkRequest) -> dict[str, object]:
    """Identity fields the reply must copy. These come from the work request."""

    values: dict[str, object] = {
        "schema_version": "review-provider-output-v1",
        "provider": request.provider,
        "provider_interface": request.provider_interface,
        "task_id": request.task_id,
        "job_id": request.job_id,
        "batch_id": request.batch_id,
        "document_id": request.document_id,
        "attempt": request.attempt,
        "source_bundle_sha256": request.source_bundle_sha256,
        "stage": request.stage,
    }
    if request.stage == "challenge":
        values["round_index"] = request.round_index
    return values


def challenge_evidence_hashes(task_dir: Path) -> tuple[str, ...]:
    """Page and crop hashes the published challenge names, sorted."""

    recorded: set[str] = set()
    crops = task_dir / "input" / "crops"
    if crops.is_dir():
        for path in sorted(crops.glob("*.json")):
            try:
                record = CropRecord.model_validate_json(path.read_bytes())
            except ValidationError as exc:
                raise ReviewIntegrityError(
                    "sealed crop manifest does not match the schema"
                ) from exc
            recorded.add(record.page_sha256)
            if record.crop_sha256 is not None:
                recorded.add(record.crop_sha256)
    disputes = task_dir / "input" / "disputes.json"
    if disputes.is_file():
        try:
            payload = json.loads(disputes.read_bytes())
        except json.JSONDecodeError as exc:
            raise ReviewIntegrityError("sealed dispute manifest is not JSON") from exc
        groups = payload.get("groups") if isinstance(payload, dict) else None
        for group in groups if isinstance(groups, list) else []:
            if isinstance(group, dict):
                for key in ("page_sha256", "crop_sha256"):
                    value = group.get(key)
                    if isinstance(value, str):
                        recorded.add(value)
    if not recorded:
        raise ReviewIntegrityError("sealed task manifest does not record an image hash")
    return tuple(sorted(recorded))


def present_image_hashes(task: DispatchTask) -> tuple[str, ...]:
    """Hashes of the image files a provider would be shown, sorted and unique."""

    found: set[str] = set()
    for path in _image_files(task.task_dir):
        if path.is_symlink():
            raise ReviewIntegrityError("refusing a symlinked sealed image")
        found.add(sha256_bytes(path.read_bytes()))
    return tuple(sorted(found))


def _image_files(task_dir: Path) -> list[Path]:
    paths = [
        path
        for path in task_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    inputs = task_dir / "input"
    if inputs.is_dir():
        paths.extend(
            path
            for path in inputs.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
    return sorted(paths)


def _require_evidence(task: DispatchTask) -> None:
    if present_image_hashes(task) != tuple(sorted(set(task.evidence_sha256))):
        raise ReviewIntegrityError("sealed image bytes do not match the published task")


def _check_prepared(task: DispatchTask, prepared: PreparedSend) -> None:
    if prepared.request_id != task.request_id:
        raise ReviewIntegrityError("send request id does not match the task id")
    if not prepared.message.strip() or prepared.request_id not in prepared.message:
        raise ReviewIntegrityError("send message does not carry its request id")
    for text in task.forbidden_texts:
        if text and text in prepared.message:
            raise ReviewIntegrityError(
                "send message names the other provider or the archive transcription"
            )
    roots = [task.task_dir.resolve(), task.packet_dir.resolve()]
    for path in prepared.attachments:
        if path.is_symlink():
            raise ReviewIntegrityError("refusing a symlinked attachment")
        resolved = path.resolve()
        if not resolved.is_file() or not any(
            root == resolved.parent or root in resolved.parents for root in roots
        ):
            raise ReviewIntegrityError("attachment is outside this provider's task")
        if resolved.suffix.lower() in _IMAGE_SUFFIXES and (
            sha256_bytes(resolved.read_bytes()) not in task.evidence_sha256
        ):
            raise ReviewIntegrityError("attached image is not sealed task evidence")


@dataclass(frozen=True, slots=True)
class _Paths:
    raw: Path
    provenance: Path
    attempt: Path
    verified: Path
    blocked: Path


def _preservation_paths(preservation_dir: Path, task: DispatchTask) -> _Paths:
    directory = (
        preservation_dir
        / "dispatch"
        / stage_dir_name(task.stage, task.round_index)
        / task.provider
    )
    stem = directory / task.request_id
    return _Paths(
        raw=stem.with_name(f"{task.request_id}.raw.txt"),
        provenance=stem.with_name(f"{task.request_id}.provenance.json"),
        attempt=stem.with_name(f"{task.request_id}.attempt.json"),
        verified=stem.with_name(f"{task.request_id}.verified.json"),
        blocked=stem.with_name(f"{task.request_id}.integrity_blocked.json"),
    )


def _base_record(task: DispatchTask, prepared: PreparedSend) -> dict[str, object]:
    """Identity, message, and evidence a send is bound to. No reply fields."""

    request = task.request
    return {
        "attachment_sha256": [
            sha256_bytes(item.read_bytes()) for item in prepared.attachments
        ],
        "attempt": request.attempt,
        "batch_id": request.batch_id,
        "document_id": request.document_id,
        "evidence_sha256": sorted(set(task.evidence_sha256)),
        "job_id": request.job_id,
        "mechanism": prepared.mechanism,
        "message_sha256": sha256_bytes(prepared.message.encode("utf-8")),
        "provider": request.provider,
        "provider_interface": request.provider_interface,
        "request_id": prepared.request_id,
        "round_index": request.round_index,
        "schema_version": PROVENANCE_SCHEMA,
        "source_bundle_sha256": request.source_bundle_sha256,
        "stage": request.stage,
        "task_id": request.task_id,
    }


def _require_regular_or_absent(path: Path, label: str) -> bool:
    """True for a regular file, False when absent. A link or other node is refused."""

    if not os.path.lexists(path):
        return False
    if path.is_symlink() or not path.is_file():
        raise ReviewIntegrityError(f"{label} is not a regular file")
    return True


def _read_record(path: Path, label: str) -> dict[str, object]:
    _require_regular_or_absent(path, label)
    try:
        loaded = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReviewIntegrityError(f"{label} is not readable JSON") from exc
    if not isinstance(loaded, dict):
        raise ReviewIntegrityError(f"{label} is not an object")
    return loaded


def _write_new(path: Path, body: Mapping[str, object]) -> None:
    """Publish a record that does not exist yet. An existing one is never replaced."""

    if os.path.lexists(path):
        raise ReviewIntegrityError("refusing to replace an existing dispatch record")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _publish_or_verify(path: Path, body: dict[str, object]) -> None:
    """Write the attempt record once. A later call must match it exactly."""

    if os.path.lexists(path):
        if _read_record(path, "dispatch attempt record") != body:
            raise ReviewIntegrityError(
                "dispatch attempt record does not match this request"
            )
        return
    _write_new(path, body)


def _provenance_body(
    base: dict[str, object], raw: bytes, transport: dict[str, object]
) -> dict[str, object]:
    return {
        **base,
        "raw_reply_bytes": len(raw),
        "raw_reply_sha256": sha256_bytes(raw),
        "transport": transport,
    }


def _verify_provenance(
    loaded: dict[str, object], base: dict[str, object], raw: bytes
) -> None:
    expected = _provenance_body(base, raw, {})
    loaded = dict(loaded)
    transport = loaded.pop("transport", None)
    loaded.pop("recovered_after_interruption", None)
    expected.pop("transport")
    if not isinstance(transport, dict) or loaded != expected:
        raise ReviewIntegrityError(
            "dispatch provenance does not match the request, message, evidence, "
            "and preserved reply"
        )


def _publish_provenance(
    path: Path, base: dict[str, object], raw: bytes, transport: dict[str, object]
) -> None:
    if os.path.lexists(path):
        _verify_provenance(_read_record(path, "dispatch provenance"), base, raw)
        return
    _write_new(path, _provenance_body(base, raw, transport))


def _settle_provenance(paths: _Paths, base: dict[str, object], raw: bytes) -> None:
    """Check provenance before a preserved reply is reused.

    Missing provenance is accepted only after an interruption between the
    reply and its provenance, and only when the attempt record written before
    the send matches this exact request. The reply is not sent again.
    """

    if os.path.lexists(paths.provenance):
        _verify_provenance(
            _read_record(paths.provenance, "dispatch provenance"), base, raw
        )
        return
    if not os.path.lexists(paths.attempt):
        raise ReviewIntegrityError(
            "preserved reply has no provenance and no attempt record"
        )
    attempt = _read_record(paths.attempt, "dispatch attempt record")
    if attempt != {**base, "schema_version": ATTEMPT_SCHEMA}:
        raise ReviewIntegrityError(
            "dispatch attempt record does not match this request"
        )
    body = _provenance_body(base, raw, {"recovered_after_interruption": True})
    body["recovered_after_interruption"] = True
    _write_new(paths.provenance, body)


def _verified_body(base: dict[str, object], raw: bytes) -> dict[str, object]:
    """Written only after the post-send evidence check passed for this reply."""

    return {
        **base,
        "raw_reply_sha256": sha256_bytes(raw),
        "schema_version": VERIFIED_SCHEMA,
        "post_send_evidence_verified": True,
    }


def _unverified_reason(
    paths: _Paths, base: dict[str, object], raw: bytes
) -> str | None:
    """None only when durable state shows the post-send check completed."""

    if not os.path.lexists(paths.verified):
        return "post-send evidence verification did not complete for this reply"
    try:
        loaded = _read_record(paths.verified, "post-send verification record")
    except ReviewIntegrityError as exc:
        return f"post-send verification record is unusable: {exc}"
    if loaded != _verified_body(base, raw):
        return "post-send verification record does not match this reply"
    return None


def _block(
    paths: _Paths, task: DispatchTask, reason: str, raw: bytes | None = None
) -> None:
    """Durably mark this attempt. The preserved reply is never installed."""

    if os.path.lexists(paths.blocked):
        return
    if raw is None and os.path.lexists(paths.raw):
        raw = paths.raw.read_bytes()
    _write_new(
        paths.blocked,
        {
            "evidence_sha256": sorted(set(task.evidence_sha256)),
            "raw_reply_sha256": None if raw is None else sha256_bytes(raw),
            "reason": reason,
            "request_id": task.request_id,
            "schema_version": BLOCKED_SCHEMA,
            "task_id": task.request.task_id,
        },
    )


def _blocked_reason(path: Path) -> str:
    try:
        loaded = _read_record(path, "integrity marker")
    except ReviewIntegrityError:
        return "integrity marker is present and unreadable"
    reason = loaded.get("reason")
    return (
        f"integrity blocked: {reason}"
        if isinstance(reason, str)
        else "integrity blocked"
    )


def _record(
    task: DispatchTask,
    outcome: Outcome,
    reason: str | None,
    *,
    raw_path: Path | None = None,
    raw: bytes | None = None,
    installed: Path | None = None,
) -> DispatchRecord:
    return DispatchRecord(
        provider=task.provider,
        stage=task.stage,
        round_index=task.round_index,
        task_id=task.request.task_id,
        request_id=task.request_id,
        outcome=outcome,
        reason=reason,
        raw_sha256=None if raw is None else sha256_bytes(raw),
        raw_path=None if raw_path is None else str(raw_path),
        installed_path=None if installed is None else str(installed),
    )


def _swap_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ReviewIntegrityError(
            "task prompt does not match the desktop dispatch adapter"
        )
    return text.replace(old, new)


def _swap_pattern(text: str, pattern: re.Pattern[str], new: str) -> str:
    adapted, count = pattern.subn(lambda _match: new, text)
    if count != 1:
        raise ReviewIntegrityError(
            "task prompt does not match the desktop dispatch adapter"
        )
    return adapted
