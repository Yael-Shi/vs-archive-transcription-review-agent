"""Connect the review ledger to the desktop file queue.

The controller registers both first-pass tasks. This module publishes those
same work requests into the desktop queue, then later asks the controller to
accept collected raw files. It does not call a model, launch a desktop app,
compare readings, open a challenge, or finalize a document.

A person opens one Codex Desktop or GPT Work session on the GPT task directory
and one Grok Bot Desktop session on the Grok task directory. Each app writes
only its own ``output/transcription_{provider}.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.desktop_file_queue import (
    DesktopFileQueue,
    publish_desktop_file_queue,
)
from transcription_review_agent.review.desktop_result_intake import (
    DesktopIntakeReport,
    collect_desktop_first_pass,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPolicyError,
    ReviewUsageError,
)
from transcription_review_agent.review.models import WorkRequest
from transcription_review_agent.review.paths import (
    archive_document_dir,
    document_root,
    stage_dir_name,
)

_PROVIDERS = ("gpt", "grok")


@dataclass(frozen=True, slots=True)
class DesktopRegistration:
    """Ledger-registered first-pass tasks and their published desktop folders."""

    batch_id: str
    document_id: int
    queue: DesktopFileQueue
    work_requests: tuple[WorkRequest, WorkRequest]


@dataclass(frozen=True, slots=True)
class DesktopAcceptance:
    """Collection plus the ledger status after existing accept calls."""

    intake: DesktopIntakeReport
    providers: dict[str, str]
    execution_state: str
    finalized: bool
    ledger_sequence: int
    comparison_ready: bool
    notes: dict[str, str]


def register_desktop_first_pass(
    controller: ReviewController,
    *,
    batch_id: str,
    document_id: int,
    queue_root: Path | str,
    mount_root: Path | None = None,
) -> DesktopRegistration:
    """Open the sealed page, register attempt 1, and publish the desktop queue.

    A provider that is already waiting or already accepted keeps its work
    request. A failed or invalid attempt is not replaced.
    """

    status = controller.open_sealed_document(batch_id, document_id)
    if status.page_ids != ("page_001",):
        raise ReviewPolicyError("desktop queue supports one sealed page")
    for provider in _PROVIDERS:
        state = status.providers[provider]
        if state == "NOT_STARTED":
            controller.publish(batch_id, document_id, provider, stage="first_pass")
        elif state not in {"AWAITING_MANUAL_DISPATCH", "SUCCEEDED"}:
            raise ReviewPolicyError(
                f"{provider} is {state}; desktop registration does not open "
                "another attempt"
            )
    requests = load_registered_work_requests(controller, batch_id, document_id)
    queue = publish_desktop_file_queue(
        batches_dir=controller.batches_dir,
        batch_id=batch_id,
        document_id=document_id,
        queue_root=queue_root,
        work_requests=requests,
        reviews_dir=controller.reviews_dir,
        mount_root=mount_root,
    )
    return DesktopRegistration(
        batch_id=batch_id,
        document_id=document_id,
        queue=queue,
        work_requests=requests,
    )


def accept_desktop_first_pass(
    controller: ReviewController,
    *,
    batch_id: str,
    document_id: int,
    queue_root: Path | str,
    preservation_dir: Path,
    mount_root: Path | None = None,
) -> DesktopAcceptance:
    """Preserve raw desktop files and accept each valid waiting result once.

    A missing or invalid provider is reported and is not accepted. A provider
    that already succeeded is checked again through ``ReviewController.accept``.
    Comparison and finalization are not started.
    """

    requests = load_registered_work_requests(controller, batch_id, document_id)
    sealed = archive_document_dir(controller.batches_dir, batch_id, document_id)
    intake = collect_desktop_first_pass(
        queue_root=queue_root,
        sealed_document_dir=sealed,
        work_requests=requests,
        reviews_dir=controller.reviews_dir,
        preservation_dir=preservation_dir,
        mount_root=mount_root,
    )
    notes: dict[str, str] = {}
    for provider, collected in (("gpt", intake.gpt), ("grok", intake.grok)):
        state = controller.status(batch_id, document_id).providers[provider]
        if state == "SUCCEEDED" or (
            state == "AWAITING_MANUAL_DISPATCH"
            and collected.state in {"ready", "unchanged"}
        ):
            controller.accept(batch_id, document_id, provider, stage="first_pass")
            continue
        if state == "AWAITING_MANUAL_DISPATCH":
            notes[provider] = collected.failure or "provider output is not acceptable"
            continue
        notes[provider] = (
            f"{provider} is {state}; desktop acceptance does not open another attempt"
        )
    status = controller.status(batch_id, document_id)
    ready = (
        not status.finalized
        and status.providers["gpt"] == "SUCCEEDED"
        and status.providers["grok"] == "SUCCEEDED"
    )
    return DesktopAcceptance(
        intake=intake,
        providers=dict(status.providers),
        execution_state=status.execution_state,
        finalized=status.finalized,
        ledger_sequence=status.ledger_sequence,
        comparison_ready=ready,
        notes=notes,
    )


def load_registered_work_requests(
    controller: ReviewController,
    batch_id: str,
    document_id: int,
) -> tuple[WorkRequest, WorkRequest]:
    """Read the work requests the controller already wrote."""

    found: list[WorkRequest] = []
    root = document_root(controller.reviews_dir, batch_id, document_id)
    for provider in _PROVIDERS:
        path = (
            root
            / stage_dir_name("first_pass", 0)
            / "tasks"
            / provider
            / "work_request.json"
        )
        if not path.is_file():
            raise ReviewPolicyError(f"{provider} work has not been published")
        raw = path.read_bytes()
        request = WorkRequest.model_validate_json(raw)
        if canonical_json(request).encode("utf-8") != raw:
            raise ReviewIntegrityError(
                "registered work request bytes do not match the ledger task"
            )
        if (
            request.provider != provider
            or request.stage != "first_pass"
            or request.attempt != 1
            or request.page_ids != ["page_001"]
        ):
            raise ReviewIntegrityError(
                "registered work request is not an attempt-1 desktop task"
            )
        found.append(request)
    return (found[0], found[1])


def main(argv: list[str] | None = None) -> int:
    """Register, accept, or resume one document through the desktop queue."""

    parser = argparse.ArgumentParser(prog="desktop-e2e", add_help=True)
    parser.add_argument("operation", choices=("register", "accept", "advance"))
    parser.add_argument("--batches-dir", required=True, type=Path)
    parser.add_argument("--reviews-dir", required=True, type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--document-id", required=True, type=int)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--preservation-dir", type=Path)
    parser.add_argument("--mount-root", type=Path)
    parser.add_argument("--packet-dir", type=Path)
    parser.add_argument("--closure-mode")
    parser.add_argument("--reason")
    parser.add_argument("--finalize", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code in {0, None}:
            raise
        raise ReviewUsageError("desktop-e2e usage was not recognized") from exc
    if args.operation in {"accept", "advance"} and args.preservation_dir is None:
        raise ReviewUsageError(f"{args.operation} requires --preservation-dir")
    from transcription_review_agent.review.config import load_review_config

    controller = ReviewController(
        batches_dir=args.batches_dir,
        reviews_dir=args.reviews_dir,
        config=load_review_config(),
    )
    if args.operation == "advance":
        from transcription_review_agent.review.document_run import (
            advance_document,
            run_payload,
        )

        resumed = advance_document(
            controller,
            batch_id=args.batch_id,
            document_id=args.document_id,
            queue_root=args.queue_root,
            preservation_dir=args.preservation_dir,
            mount_root=args.mount_root,
            closure_mode=args.closure_mode,
            closure_reason=args.reason,
            finalize=args.finalize,
            packet_dir=args.packet_dir,
        )
        print(json.dumps(run_payload(resumed), sort_keys=True))
        return 0 if resumed.blocker != "challenge_result_not_acceptable" else 3
    if args.operation == "register":
        registered = register_desktop_first_pass(
            controller,
            batch_id=args.batch_id,
            document_id=args.document_id,
            queue_root=args.queue_root,
            mount_root=args.mount_root,
        )
        print(json.dumps(_registration_payload(registered), sort_keys=True))
        return 0
    accepted = accept_desktop_first_pass(
        controller,
        batch_id=args.batch_id,
        document_id=args.document_id,
        queue_root=args.queue_root,
        preservation_dir=args.preservation_dir,
        mount_root=args.mount_root,
    )
    print(json.dumps(_acceptance_payload(accepted), sort_keys=True))
    return 0 if accepted.comparison_ready else 3


if __name__ == "__main__":
    raise SystemExit(main())


def _registration_payload(registered: DesktopRegistration) -> dict[str, object]:
    queue = registered.queue
    return {
        "batch_id": registered.batch_id,
        "document_id": registered.document_id,
        "gpt_output": queue.gpt.windows_output_file,
        "gpt_task": queue.gpt.windows_directory,
        "grok_output": queue.grok.windows_output_file,
        "grok_task": queue.grok.windows_directory,
        "task_ids": {
            request.provider: request.task_id for request in registered.work_requests
        },
    }


def _acceptance_payload(accepted: DesktopAcceptance) -> dict[str, object]:
    return {
        "comparison_ready": accepted.comparison_ready,
        "execution_state": accepted.execution_state,
        "finalized": accepted.finalized,
        "gpt_state": accepted.intake.gpt.state,
        "grok_state": accepted.intake.grok.state,
        "ledger_sequence": accepted.ledger_sequence,
        "notes": accepted.notes,
        "providers": accepted.providers,
    }
