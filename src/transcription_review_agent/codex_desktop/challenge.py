"""Prepare one GPT challenge for Codex Desktop and hand a real reply to the controller.

Planning does not open the pipe. Installing a reply does not send a turn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.desktop_result_intake import _place_exact
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewUsageError,
)
from transcription_review_agent.codex_desktop.models import CodexRequest
from transcription_review_agent.review.models import (
    ChallengeOutput,
    CropRecord,
    WorkRequest,
)
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.provider_output_contract import (
    template_placeholder_problem,
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True, slots=True)
class CodexChallengePlan:
    """Files the bound Desktop thread can read. Creating this does not send."""

    thread_id: str
    transport_request_id: str
    task_id: str
    message: str
    workspace: Path
    images: tuple[Path, ...]
    image_sha256: tuple[str, ...]
    output_file: Path
    task_dir: Path


@dataclass(frozen=True, slots=True)
class ChallengeInstall:
    raw_path: Path
    raw_sha256: str
    review_output_path: Path | None
    accepted: bool
    failure: str | None


def plan_codex_challenge(
    task_dir: Path,
    *,
    thread_id: str,
    workspace: Path,
    transport_request_id: str,
) -> CodexChallengePlan:
    """Bind the published GPT task to one workspace that already contains its images."""

    task_dir = task_dir.resolve()
    workspace = _workspace(workspace)
    request = WorkRequest.model_validate_json(
        (task_dir / "work_request.json").read_text(encoding="utf-8")
    )
    if request.provider != "gpt" or request.stage != "challenge":
        raise ReviewUsageError(
            "Codex challenge planning requires the gpt challenge task"
        )
    if not thread_id.strip() or not transport_request_id.strip():
        raise ReviewUsageError("thread id and transport request id are required")
    if transport_request_id not in (task_dir / "prompt.txt").read_text(
        encoding="utf-8"
    ):
        prompt = (task_dir / "prompt.txt").read_text(encoding="utf-8").rstrip() + (
            f"\n\ntransport_request_id: {transport_request_id}\n"
        )
    else:
        prompt = (task_dir / "prompt.txt").read_text(encoding="utf-8")
    images = _images(task_dir)
    if not images:
        raise ReviewUsageError("the challenge task has no sealed image")
    outside = [path for path in images if not _inside(workspace, path)]
    if outside:
        raise ReviewIntegrityError(
            "sealed images are outside the Codex thread workspace"
        )
    recorded = _recorded_image_hashes(task_dir)
    current = tuple(sha256_bytes(path.read_bytes()) for path in images)
    if set(current) != recorded:
        raise ReviewIntegrityError("sealed image bytes do not match the task manifest")
    output_file = task_dir / "output" / "challenge_gpt.json"
    return CodexChallengePlan(
        thread_id=thread_id,
        transport_request_id=transport_request_id,
        task_id=request.task_id,
        message=prompt if prompt.endswith("\n") else prompt + "\n",
        workspace=workspace,
        images=tuple(images),
        image_sha256=current,
        output_file=output_file,
        task_dir=task_dir,
    )


def verify_sealed_images(plan: CodexChallengePlan) -> tuple[str, ...]:
    """Re-read the bound page and crop against the sealed task manifest.

    The hashes stored on the plan are the bytes observed at planning. Dispatch
    also requires those bytes to still equal the crop record and dispute
    manifest, so a later snapshot cannot replace the published hashes.
    """

    current = tuple(sha256_bytes(path.read_bytes()) for path in plan.images)
    if current != plan.image_sha256:
        raise ReviewIntegrityError("sealed image bytes changed after planning")
    if set(current) != _recorded_image_hashes(plan.task_dir):
        raise ReviewIntegrityError("sealed image bytes do not match the task manifest")
    return current


def challenge_desktop_request(plan: CodexChallengePlan) -> CodexRequest:
    """Build the one send, with the current image hashes bound to the attempt."""

    verify_sealed_images(plan)
    return CodexRequest(
        thread_id=plan.thread_id,
        request_id=plan.transport_request_id,
        message=plan.message,
        image_sha256=plan.image_sha256,
    )


def install_challenge_reply(
    *,
    reviews_dir: Path,
    work_request: WorkRequest,
    raw: bytes,
    preservation_dir: Path,
    disputes: dict[str, object] | None = None,
) -> ChallengeInstall:
    """Preserve the raw reply, then copy it only when the challenge schema accepts it."""

    if work_request.provider != "gpt" or work_request.stage != "challenge":
        raise ReviewUsageError("challenge install requires the gpt challenge request")
    raw_path = preservation_dir / f"{_safe_name(work_request.task_id)}.raw"
    _place_exact(raw_path, raw)
    review_path = (
        document_root(reviews_dir, work_request.batch_id, work_request.document_id)
        / work_request.output_relative_path
    )
    failure = _failure(raw, work_request, disputes)
    if failure is not None:
        return ChallengeInstall(
            raw_path=raw_path,
            raw_sha256=sha256_bytes(raw),
            review_output_path=None,
            accepted=False,
            failure=failure,
        )
    _place_exact(review_path, raw)
    return ChallengeInstall(
        raw_path=raw_path,
        raw_sha256=sha256_bytes(raw),
        review_output_path=review_path,
        accepted=True,
        failure=None,
    )


def accept_installed_challenge(
    controller: ReviewController,
    install: ChallengeInstall,
    *,
    batch_id: str,
    document_id: int,
    round_index: int,
) -> object:
    """Record one already validated file through the existing accept transition."""

    if not install.accepted or install.review_output_path is None:
        raise ReviewUsageError("refusing to accept an invalid Codex challenge reply")
    return controller.accept(
        batch_id,
        document_id,
        "gpt",
        stage="challenge",
        round_index=round_index,
    )


def _failure(
    raw: bytes, request: WorkRequest, disputes: dict[str, object] | None
) -> str | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "provider output is not JSON"
    try:
        model = ChallengeOutput.model_validate(payload)
    except ValidationError:
        return "provider output does not match the schema"
    if (
        model.provider != request.provider
        or model.provider_interface != request.provider_interface
        or model.task_id != request.task_id
        or model.job_id != request.job_id
        or model.batch_id != request.batch_id
        or model.document_id != request.document_id
        or model.attempt != request.attempt
        or model.source_bundle_sha256 != request.source_bundle_sha256
        or model.stage != request.stage
        or model.round_index != request.round_index
    ):
        return "provider output does not match the published task"
    problem = template_placeholder_problem(model.model_dump(mode="json"))
    if problem is not None:
        return problem
    if disputes is not None:
        return _decision_problem(model, disputes)
    return None


def _decision_problem(
    model: ChallengeOutput, disputes: dict[str, object]
) -> str | None:
    required = disputes.get("required_decision_ids")
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        return "published disputes do not name the required decisions"
    got = [item.dispute_id for item in model.decisions]
    if got != required:
        return "challenge decisions do not match the published disputes"
    if disputes.get("challenge_contract") != "EVIDENCE_BOUND_V1":
        return None
    refs = _published_refs(disputes)
    for decision in model.decisions:
        if decision.evidence_ref is None or decision.evidence_assessment is None:
            return "evidence-bound challenge decision is missing its evidence"
        key = (
            decision.evidence_ref.kind,
            decision.evidence_ref.evidence_id,
            decision.evidence_ref.page_id,
        )
        allowed = refs.get(decision.dispute_id, ())
        if key not in allowed:
            return "evidence reference was not published for this dispute"
    return None


def _published_refs(
    disputes: dict[str, object],
) -> dict[str, tuple[tuple[str, str, str], ...]]:
    groups = disputes.get("groups")
    found: dict[str, list[tuple[str, str, str]]] = {}
    if not isinstance(groups, list):
        return {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        ids = group.get("dispute_ids")
        refs = group.get("evidence_refs")
        if not isinstance(ids, list) or not isinstance(refs, list):
            continue
        parsed: list[tuple[str, str, str]] = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            kind = ref.get("kind")
            evidence_id = ref.get("evidence_id")
            page_id = ref.get("page_id")
            if (
                isinstance(kind, str)
                and isinstance(evidence_id, str)
                and isinstance(page_id, str)
            ):
                parsed.append((kind, evidence_id, page_id))
        for dispute_id in ids:
            if isinstance(dispute_id, str):
                found[dispute_id] = parsed
    return {key: tuple(value) for key, value in found.items()}


def _recorded_image_hashes(task_dir: Path) -> set[str]:
    """Page and crop hashes named by the published task, not by the files."""

    recorded: set[str] = set()
    crops = task_dir / "input" / "crops"
    if crops.is_dir():
        for path in sorted(crops.glob("*.json")):
            try:
                record = CropRecord.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as exc:
                raise ReviewIntegrityError(
                    "sealed crop manifest does not match the schema"
                ) from exc
            recorded.add(record.page_sha256)
            if record.crop_sha256 is not None:
                recorded.add(record.crop_sha256)
    disputes = task_dir / "input" / "disputes.json"
    if disputes.is_file():
        try:
            payload = json.loads(disputes.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewIntegrityError("sealed dispute manifest is not JSON") from exc
        groups = payload.get("groups") if isinstance(payload, dict) else None
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict):
                    continue
                for key in ("page_sha256", "crop_sha256"):
                    value = group.get(key)
                    if isinstance(value, str):
                        recorded.add(value)
    if not recorded:
        raise ReviewIntegrityError("sealed task manifest does not record an image hash")
    return recorded


def _images(task_dir: Path) -> list[Path]:
    root = task_dir / "input"
    if not root.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
    )


def _workspace(workspace: Path) -> Path:
    text = str(workspace)
    if len(text) >= 2 and text[1] == ":":
        return Path(text)
    return workspace.resolve()


def _inside(workspace: Path, path: Path) -> bool:
    root = _windows_form(workspace).casefold().rstrip("\\")
    child = _windows_form(path).casefold()
    return child == root or child.startswith(root + "\\")


def _windows_form(path: Path) -> str:
    text = str(path)
    mount = "/mnt/"
    if (
        text.startswith(mount)
        and len(text) > len(mount) + 2
        and text[len(mount) + 1] == "/"
    ):
        drive = text[len(mount)].upper()
        rest = text[len(mount) + 2 :].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text.replace("/", "\\")


def _safe_name(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in task_id)
