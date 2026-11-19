"""Collect raw desktop-worker files into the existing review output paths.

The desktop apps write `output/transcription_{provider}.json` inside their own
task directories. This module copies those bytes, unchanged, into the
allowlisted review path that `ReviewController.accept` already reads. It does
not launch an app, call a model, repair JSON, or write the review ledger.

A human still opens one Codex Desktop or GPT Work session and one Grok Bot
Desktop session. Logical provider separation is two directories. It is not an
operating-system access boundary. Comparison, challenge, reconciliation, and
human verification stay in the existing review pipeline and start only after
both first-pass files have been accepted there.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.desktop_file_queue import (
    DesktopFileQueue,
    DesktopProviderTask,
    load_desktop_file_queue,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
)
from transcription_review_agent.review.models import FirstPassOutput, WorkRequest
from transcription_review_agent.review.paths import document_root, output_filename
from transcription_review_agent.review.provider_output_contract import (
    template_placeholder_problem,
)

ProviderState = Literal["missing", "invalid", "ready", "unchanged"]


@dataclass(frozen=True, slots=True)
class ProviderIntake:
    provider: Literal["gpt", "grok"]
    state: ProviderState
    raw_sha256: str | None
    preservation_path: Path | None
    review_output_path: Path
    schema_valid: bool
    identity_matches: bool
    failure: str | None


@dataclass(frozen=True, slots=True)
class DesktopIntakeReport:
    source_sha256: str
    gpt: ProviderIntake
    grok: ProviderIntake

    @property
    def both_ready_for_accept(self) -> bool:
        """True when both review files match their raw desktop bytes and validate.

        This does not record ledger acceptance and does not compare the readings.
        """

        return self.gpt.state in {"ready", "unchanged"} and self.grok.state in {
            "ready",
            "unchanged",
        }


def collect_desktop_first_pass(
    *,
    queue_root: Path | str,
    sealed_document_dir: Path,
    work_requests: tuple[WorkRequest, WorkRequest] | list[WorkRequest],
    reviews_dir: Path,
    preservation_dir: Path,
    mount_root: Path | None = None,
) -> DesktopIntakeReport:
    """Preserve each present desktop result and hand off a valid file once.

    A missing provider stays missing. Invalid bytes are preserved and are not
    copied into the review tree. An existing different file is left in place.
    """

    requests = _requests(work_requests)
    queue = load_desktop_file_queue(
        queue_root,
        sealed_document_dir,
        work_requests,
        reviews_dir=reviews_dir,
        mount_root=mount_root,
    )
    _reject_overlap(preservation_dir, queue)
    _reject_overlap(reviews_dir, queue)
    return DesktopIntakeReport(
        source_sha256=queue.source_sha256,
        gpt=_collect_provider(
            queue.gpt,
            request=requests["gpt"],
            queue=queue,
            reviews_dir=reviews_dir,
            preservation_dir=preservation_dir,
        ),
        grok=_collect_provider(
            queue.grok,
            request=requests["grok"],
            queue=queue,
            reviews_dir=reviews_dir,
            preservation_dir=preservation_dir,
        ),
    )


def _requests(
    work_requests: tuple[WorkRequest, WorkRequest] | list[WorkRequest],
) -> dict[str, WorkRequest]:
    found: dict[str, WorkRequest] = {
        request.provider: request for request in work_requests
    }
    if set(found) != {"gpt", "grok"}:
        raise ReviewIntegrityError(
            "desktop intake requires the gpt and grok work requests"
        )
    return found


def _collect_provider(
    task: DesktopProviderTask,
    *,
    request: WorkRequest,
    queue: DesktopFileQueue,
    reviews_dir: Path,
    preservation_dir: Path,
) -> ProviderIntake:
    review_path = _review_output(reviews_dir, queue, request)
    if not task.result_present:
        return ProviderIntake(
            provider=task.provider,
            state="missing",
            raw_sha256=None,
            preservation_path=None,
            review_output_path=review_path,
            schema_valid=False,
            identity_matches=False,
            failure="provider output is missing",
        )
    raw = task.result_path.read_bytes()
    preserved = preservation_dir / task.provider / task.result_path.name
    _place_exact(preserved, raw)
    schema_valid, identity_matches, failure = _assess(raw, request)
    if failure is not None:
        return ProviderIntake(
            provider=task.provider,
            state="invalid",
            raw_sha256=sha256_bytes(raw),
            preservation_path=preserved,
            review_output_path=review_path,
            schema_valid=schema_valid,
            identity_matches=identity_matches,
            failure=failure,
        )
    created = _place_exact(review_path, raw)
    return ProviderIntake(
        provider=task.provider,
        state="ready" if created else "unchanged",
        raw_sha256=sha256_bytes(raw),
        preservation_path=preserved,
        review_output_path=review_path,
        schema_valid=True,
        identity_matches=True,
        failure=None,
    )


def _review_output(
    reviews_dir: Path, queue: DesktopFileQueue, request: WorkRequest
) -> Path:
    filename = output_filename("first_pass", request.provider)
    if Path(request.output_relative_path).name != filename:
        raise ReviewIntegrityError("work request output is not the first-pass file")
    path = (
        document_root(reviews_dir, queue.batch_id, queue.document_id)
        / request.output_relative_path
    )
    if path.name != filename:
        raise ReviewIntegrityError("review output path is not allowlisted")
    return path


def _assess(raw: bytes, request: WorkRequest) -> tuple[bool, bool, str | None]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, False, "provider output is not JSON"
    try:
        model = FirstPassOutput.model_validate(payload)
    except ValidationError:
        return False, False, "provider output does not match the schema"
    identity_matches = (
        model.provider == request.provider
        and model.provider_interface == request.provider_interface
        and model.task_id == request.task_id
        and model.job_id == request.job_id
        and model.batch_id == request.batch_id
        and model.document_id == request.document_id
        and model.attempt == request.attempt
        and model.source_bundle_sha256 == request.source_bundle_sha256
        and model.stage == request.stage
        and [page.page_id for page in model.pages] == list(request.page_ids)
    )
    if not identity_matches:
        return True, False, "provider output does not match the published task"
    problem = template_placeholder_problem(model.model_dump(mode="json"))
    if problem is not None:
        return False, True, problem
    return True, True, None


def _place_exact(path: Path, payload: bytes) -> bool:
    """Create `path` with these bytes, or keep it when the bytes already match."""

    if path.is_symlink():
        raise ReviewIntegrityError("refusing to follow a collected output symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ReviewIntegrityError("refusing to change an existing provider result")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ReviewIntegrityError(
                "refusing to change an existing provider result"
            ) from None
        return False
    view = memoryview(payload)
    try:
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ReviewIntegrityError(
                    "desktop result preservation was interrupted"
                )
            view = view[written:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    os.close(fd)
    return True


def _reject_overlap(candidate: Path, queue: DesktopFileQueue) -> None:
    for task in (queue.gpt, queue.grok):
        if (
            candidate == task.directory
            or task.directory in candidate.parents
            or candidate in task.directory.parents
        ):
            raise ReviewPathError("intake path overlaps a provider task directory")
