"""Build one image-grounded Grok task without launching the desktop app.

The composer message keeps the existing 2,000-character limit. A longer
prompt is attached whole as ``prompt.txt`` and is not shortened. A published
challenge that lists disputes is different: its model-facing text is a chat
packet, because the isolated profile cannot read or write local files. The
sealed images are file attachments. A path named in the message is not an
attachment.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)
from transcription_review_agent.grok_desktop.profile import local_read_path
from transcription_review_agent.grok_desktop.transcript import validate_request
from transcription_review_agent.review.desktop_result_intake import _assess
from transcription_review_agent.review.document_run import _challenge_problem
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewUsageError,
)
from transcription_review_agent.review.models import WorkRequest

_MAX_MESSAGE = 2000
_MAX_ATTACHMENTS = 8
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
_PACKET_NAME = "challenge_packet.txt"
_ALREADY_SUBMITTED = "batch-000903.1.challenge.r1.grok.a1"
_FILESYSTEM_FACING = re.compile(
    r"[A-Za-z]:\\|write one json file|write nothing|working directory|"
    r"\bshell\b|\bpowershell\b|\bcmd\.exe\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class GrokImageDispatch:
    """One prepared send. Creating it does not open Grok."""

    request_id: str
    message: str
    attachments: tuple[Path, ...]
    prompt_attached: bool
    stage: str
    output_name: str
    packet_attached: bool = False
    packet_sha256: str = ""

    def exchange_request(self, *, agent_id: str, bot_name: str) -> ExchangeRequest:
        request = ExchangeRequest(
            agent_id=agent_id,
            bot_name=bot_name,
            message=self.message,
            request_id=self.request_id,
            attachments=tuple(str(path) for path in self.attachments),
        )
        problem = validate_request(
            agent_id=request.agent_id,
            bot_name=request.bot_name,
            message=request.message,
            request_id=request.request_id,
        )
        if problem:
            raise ReviewUsageError(problem)
        return request


@dataclass(frozen=True, slots=True)
class RawReplyPreservation:
    """The received bytes, saved before any schema decision."""

    raw_path: Path
    raw_sha256: str
    output_path: Path | None
    accepted: bool
    failure: str | None


def plan_grok_image_dispatch(
    task_dir: Path,
    *,
    packet_dir: Path | None = None,
    transport_request_id: str | None = None,
) -> GrokImageDispatch:
    """Read one published Grok task directory and choose its attachments."""

    task_dir = task_dir.resolve()
    request = _work_request(task_dir)
    if request.provider != "grok":
        raise ReviewUsageError("grok image dispatch requires the grok task")
    if request.stage == "challenge" and (task_dir / "input" / "disputes.json").is_file():
        return _plan_chat_challenge(
            task_dir, request, packet_dir, transport_request_id
        )
    prompt_path = _contained_file(task_dir, task_dir / "prompt.txt")
    prompt = prompt_path.read_text(encoding="utf-8")
    if _foreign_text(prompt):
        raise ReviewIntegrityError("task prompt contains another provider path")
    request_id = _correlation_id(request.task_id)
    message, prompt_attached = _composer_message(prompt, request_id)
    images = _task_images(task_dir)
    if not images:
        raise ReviewUsageError("grok image dispatch requires a sealed image")
    attachments = list(images)
    if prompt_attached:
        attachments.append(prompt_path)
    return GrokImageDispatch(
        request_id=request_id,
        message=message,
        attachments=tuple(attachments),
        prompt_attached=prompt_attached,
        stage=request.stage,
        output_name=_output_name(request),
    )


def challenge_review_status(
    result: ExchangeResult, preservation: RawReplyPreservation | None
) -> str:
    """Separate a delivered chat turn from an accepted challenge file.

    ``success`` on the exchange means the turn's own completion signals
    produced one reply. It does not mean that reply is the review result.
    """

    delivered = result.status is ExchangeStatus.SUCCESS and result.sent
    accepted = preservation is not None and preservation.accepted
    if accepted and not delivered:
        raise ReviewIntegrityError("review acceptance requires a delivered reply")
    if accepted:
        return "accepted"
    if delivered:
        return "transport_only"
    if result.sent:
        return "delivery_unconfirmed"
    return "not_sent"


def preserve_grok_reply(
    task_dir: Path,
    raw_reply: str,
    preservation_dir: Path,
    *,
    transport_request_id: str | None = None,
) -> RawReplyPreservation:
    """Save the raw reply first, then accept only an unmodified valid file.

    Invalid JSON is kept as raw evidence and is not rewritten into the
    provider output. An existing different output is left unchanged.
    ``transport_request_id`` names the raw file for a chat-native challenge
    whose transport id is not the published task id. It does not change the
    published task or the JSON identity.
    """

    task_dir = task_dir.resolve()
    request = _work_request(task_dir)
    if transport_request_id is None:
        plan = plan_grok_image_dispatch(task_dir)
        request_id = plan.request_id
        output_name = plan.output_name
    else:
        if transport_request_id == _ALREADY_SUBMITTED:
            raise ReviewUsageError(
                "request batch-000903.1.challenge.r1.grok.a1 was already submitted"
            )
        problem = validate_request(
            agent_id="00000000-0000-4000-8000-000000000000",
            bot_name="pilot",
            message=f"request_id: {transport_request_id}",
            request_id=transport_request_id,
        )
        if problem:
            raise ReviewUsageError("challenge transport request id is not usable")
        request_id = transport_request_id
        output_name = _output_name(request)
    raw = raw_reply.encode("utf-8")
    destination = preservation_dir / request.provider / f"{request_id}.raw.txt"
    _create_exact(destination, raw)
    failure = _reply_problem(raw, request)
    if failure is not None:
        return RawReplyPreservation(
            raw_path=destination,
            raw_sha256=sha256_bytes(raw),
            output_path=None,
            accepted=False,
            failure=failure,
        )
    output = task_dir / "output" / output_name
    _create_exact(output, raw)
    return RawReplyPreservation(
        raw_path=destination,
        raw_sha256=sha256_bytes(raw),
        output_path=output,
        accepted=True,
        failure=None,
    )


def _plan_chat_challenge(
    task_dir: Path,
    request: WorkRequest,
    packet_dir: Path | None,
    transport_request_id: str | None,
) -> GrokImageDispatch:
    """Deliver one challenge in the chat without a filesystem task."""

    derived = _correlation_id(request.task_id)
    request_id = transport_request_id or derived
    if request_id == _ALREADY_SUBMITTED:
        raise ReviewUsageError(
            "request batch-000903.1.challenge.r1.grok.a1 was already submitted"
        )
    problem = validate_request(
        agent_id="00000000-0000-4000-8000-000000000000",
        bot_name="pilot",
        message=f"request_id: {request_id}",
        request_id=request_id,
    )
    if problem:
        raise ReviewUsageError("challenge transport request id is not usable")
    instructions = _chat_instructions(task_dir, request.provider)
    disputes = _contained_file(task_dir, task_dir / "input" / "disputes.json")
    schema = _contained_file(task_dir, task_dir / "output_schema.json")
    example = _contained_file(task_dir, task_dir / "output_example.json")
    images = _task_images(task_dir)
    if not images:
        raise ReviewUsageError("grok image dispatch requires a sealed image")
    packet = _challenge_packet(
        request_id=request_id,
        request=request,
        instructions=instructions,
        disputes=disputes.read_text(encoding="utf-8"),
        crops=_crop_records(task_dir),
        schema=_chat_schema(schema.read_text(encoding="utf-8")),
        example=example.read_text(encoding="utf-8"),
    )
    _require_chat_text(packet, "challenge packet")
    if _foreign_text(packet):
        raise ReviewIntegrityError("challenge packet contains another provider path")
    message, packet_path, digest = _chat_message(
        task_dir, packet, request_id, packet_dir
    )
    attachments = list(images)
    if packet_path is not None:
        attachments.append(packet_path)
    if len(attachments) > _MAX_ATTACHMENTS:
        raise ReviewUsageError("grok challenge has more attachments than the composer allows")
    return GrokImageDispatch(
        request_id=request_id,
        message=message,
        attachments=tuple(attachments),
        prompt_attached=False,
        stage=request.stage,
        output_name=_output_name(request),
        packet_attached=packet_path is not None,
        packet_sha256=digest,
    )


def _chat_instructions(task_dir: Path, provider: str) -> str:
    path = _contained_file(task_dir, task_dir / f"instructions_{provider}.txt")
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    text = text.replace(
        "Write one JSON file. Dispatch is manual. No API is assumed.\n",
        "Return exactly one JSON object in the chat and no other text. "
        "Dispatch is manual. No API is assumed.\n",
    )
    text = text.replace("work_request.json", "the identity values in this packet")
    text = text.replace("output_example.json", "the shape template in this packet")
    if "work_request.json" in text or "output_example.json" in text:
        raise ReviewIntegrityError("challenge instructions still name a task file")
    _require_chat_text(text, "challenge instructions")
    for anchor in (
        "Candidate readings are anonymous.",
        "Cite visual evidence.",
        "Do not change a reading only to force agreement.",
    ):
        if anchor not in text:
            raise ReviewIntegrityError("challenge instructions lost a review rule")
    return text


def _chat_schema(text: str) -> str:
    text = text.replace("work_request.json", "the identity values in this packet")
    return text.replace("output_example.json", "the shape template in this packet")


def _challenge_packet(
    *,
    request_id: str,
    request: WorkRequest,
    instructions: str,
    disputes: str,
    crops: str,
    schema: str,
    example: str,
) -> str:
    identity = {
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
        "round_index": request.round_index,
        "forbidden_inputs": list(request.forbidden_inputs),
        "page_ids": list(request.page_ids),
    }
    return (
        "Chat reply contract:\n"
        "Return exactly one JSON object in the chat and no other text.\n"
        "Do not write a preamble, a progress message, or a Markdown code fence.\n"
        "Do not read local filesystem paths. Do not run a command. "
        "Do not write an output file.\n"
        "The page and crop images are the other attachments. Use those images.\n"
        f"The request id is {request_id}. Do not change the identity values below.\n"
        "\nReview rules:\n"
        f"{instructions}"
        "\nIdentity values:\n"
        f"{json.dumps(identity, indent=2, sort_keys=True)}\n"
        "\nAnonymous disputes and evidence:\n"
        f"{disputes if disputes.endswith(chr(10)) else disputes + chr(10)}"
        "\nCrop window records:\n"
        f"{crops if crops.endswith(chr(10)) else crops + chr(10)}"
        "\nOutput schema:\n"
        f"{schema if schema.endswith(chr(10)) else schema + chr(10)}"
        "\nShape template, not a reading:\n"
        f"{example if example.endswith(chr(10)) else example + chr(10)}"
    )


def _crop_records(task_dir: Path) -> str:
    crops = task_dir / "input" / "crops"
    if not crops.exists():
        return ""
    if not crops.is_dir() or crops.is_symlink():
        raise ReviewIntegrityError("challenge crops are not a real directory")
    parts: list[str] = []
    for path in sorted(crops.iterdir()):
        if path.suffix.lower() == ".json":
            parts.append(_contained_file(task_dir, path).read_text(encoding="utf-8"))
    return "\n".join(parts)


def _chat_message(
    task_dir: Path, packet: str, request_id: str, packet_dir: Path | None
) -> tuple[str, Path | None, str]:
    digest = hashlib.sha256(packet.encode("utf-8")).hexdigest()
    if request_id not in packet:
        raise ReviewIntegrityError("challenge packet lost its request id")
    if len(packet) <= _MAX_MESSAGE:
        return packet, None, digest
    if packet_dir is None:
        raise ReviewUsageError(
            "challenge evidence does not fit the composer and no packet directory was given"
        )
    directory = packet_dir.resolve()
    if directory == task_dir or directory.is_relative_to(task_dir):
        raise ReviewIntegrityError(
            "challenge packet must not be written into the published task"
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _PACKET_NAME
    _create_exact(path, packet.encode("utf-8"))
    envelope = (
        f"request_id: {request_id}\n"
        "The complete challenge is the attached file challenge_packet.txt. "
        f"Its sha256 is {digest} and its length is {len(packet.encode('utf-8'))} bytes. "
        "Read that attached file in full. It has not been shortened.\n"
        "The other attachments are the sealed page and crop images.\n"
        "Do not read local filesystem paths, run a command, or write a file.\n"
        "Reply with exactly one JSON object and no other text. "
        "No preamble, progress message, or Markdown code fence.\n"
    )
    _require_chat_text(envelope, "challenge envelope")
    if len(envelope) > _MAX_MESSAGE or request_id not in envelope:
        raise ReviewUsageError("grok composer envelope exceeds the message limit")
    return envelope, path, digest


def _require_chat_text(text: str, label: str) -> None:
    if _FILESYSTEM_FACING.search(text):
        raise ReviewIntegrityError(f"{label} still tells the model to use the filesystem")


def _composer_message(prompt: str, request_id: str) -> tuple[str, bool]:
    """Return the composer text and whether the whole prompt is a file.

    The prompt itself is never cut down to fit the composer.
    """

    if request_id in prompt:
        candidate = prompt if prompt.endswith("\n") else prompt + "\n"
    else:
        candidate = prompt.rstrip() + f"\nrequest_id: {request_id}\n"
    if len(candidate) <= _MAX_MESSAGE:
        return candidate, False
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    envelope = (
        f"request_id: {request_id}\n"
        "The complete task instructions are the attached file prompt.txt. "
        f"Its sha256 is {digest} and its length is {len(prompt.encode('utf-8'))} bytes. "
        "Read that attached file in full. It has not been shortened.\n"
        "The other attachments are the sealed source images. "
        "Reply with only the JSON object required by the attached instructions.\n"
    )
    if len(envelope) > _MAX_MESSAGE or request_id not in envelope:
        raise ReviewUsageError("grok composer envelope exceeds the message limit")
    return envelope, True


def _correlation_id(task_id: str) -> str:
    request_id = task_id.replace(":", ".")
    problem = validate_request(
        agent_id="00000000-0000-4000-8000-000000000000",
        bot_name="pilot",
        message=f"request_id: {request_id}",
        request_id=request_id,
    )
    if problem:
        raise ReviewUsageError("task id cannot make a grok request id")
    return request_id


def _task_images(task_dir: Path) -> tuple[Path, ...]:
    found: list[Path] = []
    for name in ("original.png", "original.jpg", "original.jpeg"):
        path = task_dir / name
        if path.exists():
            found.append(_contained_file(task_dir, path))
    page = task_dir / "input" / "page_001.png"
    if page.exists():
        found.append(_contained_file(task_dir, page))
    crops = task_dir / "input" / "crops"
    if crops.exists():
        if not crops.is_dir() or crops.is_symlink():
            raise ReviewIntegrityError("challenge crops are not a real directory")
        for path in sorted(crops.iterdir()):
            if path.suffix.lower() in _IMAGE_SUFFIXES:
                found.append(_contained_file(task_dir, path))
    return tuple(found)


def _work_request(task_dir: Path) -> WorkRequest:
    path = _contained_file(task_dir, task_dir / "work_request.json")
    try:
        return WorkRequest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ReviewUsageError("work request is not readable") from exc


def _contained_file(root: Path, path: Path) -> Path:
    if path.is_symlink():
        raise ReviewIntegrityError("refusing a symlinked task file")
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ReviewIntegrityError("task file is outside the provider directory")
    return resolved


def _foreign_text(prompt: str) -> bool:
    lowered = prompt.replace("\\", "/").lower()
    return "/gpt/" in lowered or lowered.endswith("/gpt")


def _output_name(request: WorkRequest) -> str:
    if request.stage == "first_pass":
        return "transcription_grok.json"
    if request.stage == "challenge":
        return "challenge_grok.json"
    raise ReviewUsageError("grok image dispatch covers first_pass and challenge")


def _reply_problem(raw: bytes, request: WorkRequest) -> str | None:
    if request.stage == "challenge":
        return _challenge_problem(raw, request)
    if request.stage != "first_pass":
        return "grok image dispatch covers first_pass and challenge"
    _schema, _identity, failure = _assess(raw, request)
    return failure


def _create_exact(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ReviewIntegrityError("refusing to follow a result symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ReviewIntegrityError("refusing to change an existing provider result")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ReviewIntegrityError(
                "refusing to change an existing provider result"
            ) from None
        return
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)


def dispatch_digest(plan: GrokImageDispatch) -> str:
    """Fingerprint the prepared bytes. This does not contact the desktop app."""

    body = {
        "attachments": [
            sha256_bytes(local_read_path(path).read_bytes())
            for path in plan.attachments
        ],
        "message_sha256": hashlib.sha256(plan.message.encode("utf-8")).hexdigest(),
        "request_id": plan.request_id,
    }
    return hashlib.sha256(
        json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
