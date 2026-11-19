"""Resume one document from the review ledger through the existing stages.

Deterministic steps call ``ReviewController`` and the desktop intake that
already exists. This module does not call a model, invent a reading, close a
challenge, or finalize unless the operator passes an explicit closure. A
desktop result that is missing stops the run and leaves the published task
in place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.desktop_execution import (
    review_dispatch_blocked,
)
from transcription_review_agent.review.desktop_e2e import (
    accept_desktop_first_pass,
    register_desktop_first_pass,
)
from transcription_review_agent.review.desktop_file_queue import (
    resolve_desktop_queue_root,
)
from transcription_review_agent.review.desktop_result_intake import _place_exact
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewOutputRejected,
    ReviewPolicyError,
    ReviewUsageError,
)
from transcription_review_agent.review.human_verification_packet import (
    build_human_verification_packet,
    build_pending_review_packet,
)
from transcription_review_agent.review.models import ChallengeOutput, WorkRequest
from transcription_review_agent.review.runtime import (
    DocumentReviewState,
    ProviderRuntime,
)
from transcription_review_agent.review.paths import (
    document_root,
    output_filename,
    require_batch_id,
    stage_dir_name,
)
from transcription_review_agent.review.provider_output_contract import (
    template_placeholder_problem,
)
from transcription_review_agent.review.runtime import ProviderStatus

_PROVIDERS = ("gpt", "grok")


@dataclass(frozen=True, slots=True)
class DocumentRun:
    """One resume. ``blocker`` names the external action still required."""

    batch_id: str
    document_id: int
    stopped_at: str
    blocker: str | None
    operator_instruction: str | None
    execution_state: str
    finalized: bool
    ledger_sequence: int
    ledger_consistent: bool
    providers: dict[str, str]
    challenge_providers: dict[str, str]
    comparison_sha256: str | None
    disputes: tuple[dict[str, object], ...]
    challenge_reconciled: bool
    decisions: tuple[dict[str, object], ...]
    desktop_tasks: dict[str, str]
    packet_dir: str | None
    notes: dict[str, str]
    archive_transcription_used_as_reading: bool
    execution: tuple[dict[str, object], ...]


def advance_document(
    controller: ReviewController,
    *,
    batch_id: str,
    document_id: int,
    queue_root: Path | str,
    preservation_dir: Path,
    mount_root: Path | None = None,
    closure_mode: str | None = None,
    closure_reason: str | None = None,
    finalize: bool = False,
    packet_dir: Path | None = None,
) -> DocumentRun:
    """Accept, compare, publish, and collect using the ledger as it stands.

    A repeated call does not open another attempt and does not replace an
    accepted file. Challenge closure and finalization run only when this call
    passes them in explicitly, or when a closure is already on the ledger and
    ``finalize`` is set.
    """

    if (closure_mode is None) != (closure_reason is None):
        raise ReviewUsageError("closure requires both --closure-mode and --reason")
    raw_before = _queue_result_hashes(
        queue_root, batch_id, document_id, mount_root, "first_pass"
    )
    status = _ensure_first_pass(
        controller,
        batch_id=batch_id,
        document_id=document_id,
        queue_root=queue_root,
        preservation_dir=preservation_dir,
        mount_root=mount_root,
    )
    _require_same_hashes(
        raw_before,
        _queue_result_hashes(
            queue_root, batch_id, document_id, mount_root, "first_pass"
        ),
    )
    if status.providers.get("gpt") != "SUCCEEDED" or (
        status.providers.get("grok") != "SUCCEEDED"
    ):
        return _finish(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            stopped_at="waiting_for_first_pass",
            blocker="desktop_first_pass",
            operator_instruction=_first_pass_instruction(
                queue_root, batch_id, document_id, mount_root
            ),
            notes={},
            disputes=(),
            decisions=(),
            desktop_tasks={},
            packet_dir=None,
            archive_used=False,
        )
    comparison = controller.compare(batch_id, document_id)
    disputes: tuple[dict[str, object], ...] = tuple(
        {
            "dispute_id": item.dispute_id,
            "category": item.category.value,
            "substantive": item.substantive,
            "gpt_text": item.gpt_text,
            "grok_text": item.grok_text,
        }
        for item in comparison.disputes
    )
    archive_used = _archive_transcription_leaked(
        controller, batch_id, document_id, comparison_text=canonical_json(comparison)
    )
    if archive_used:
        raise ReviewIntegrityError(
            "comparison includes the sealed archive transcription as a reading"
        )
    substantive = [item for item in comparison.disputes if item.substantive]
    if substantive:
        _publish_missing_challenge(controller, batch_id, document_id)
        tasks = _mirror_challenge(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            queue_root=queue_root,
            mount_root=mount_root,
        )
        notes = _intake_challenge(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            queue_root=queue_root,
            preservation_dir=preservation_dir,
            mount_root=mount_root,
        )
        _require_same_hashes(
            raw_before,
            _queue_result_hashes(
                queue_root, batch_id, document_id, mount_root, "first_pass"
            ),
        )
        if _archive_transcription_leaked(controller, batch_id, document_id):
            raise ReviewIntegrityError(
                "challenge task includes the sealed archive transcription"
            )
    else:
        tasks = {}
        notes = {}
    _state, document = controller._require_document(batch_id, document_id)
    decisions: tuple[dict[str, object], ...] = tuple(
        {
            "dispute_id": item.dispute_id,
            "decision": item.decision,
            "agreed": item.agreed,
            "result_class": item.result_class,
            "gpt_decision": item.gpt_decision,
            "grok_decision": item.grok_decision,
            "remaining_uncertainty": item.remaining_uncertainty,
        }
        for item in document.reconciled_decisions
    )
    challenge = document.challenge_rounds[-1] if document.challenge_rounds else None
    waiting = challenge is not None and any(
        item.status is ProviderStatus.AWAITING_MANUAL_DISPATCH
        for item in challenge.providers
    )
    if waiting or (substantive and challenge is None):
        return _finish(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            stopped_at="waiting_for_challenge",
            blocker="desktop_challenge",
            operator_instruction=_challenge_instruction(tasks),
            notes=notes,
            disputes=disputes,
            decisions=decisions,
            desktop_tasks=tasks,
            packet_dir=None,
            archive_used=False,
        )
    if challenge is not None and not challenge.reconciled:
        return _finish(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            stopped_at="challenge_not_reconciled",
            blocker="challenge_result_not_acceptable",
            operator_instruction=None,
            notes=notes,
            disputes=disputes,
            decisions=decisions,
            desktop_tasks=tasks,
            packet_dir=None,
            archive_used=False,
        )
    if closure_mode is not None and closure_reason is not None:
        if not substantive:
            raise ReviewPolicyError("there is no substantive dispute to close")
        if document.challenge_closure is None:
            controller.close_challenge(
                batch_id,
                document_id,
                mode=closure_mode,
                reason=closure_reason,
            )
        finalize = True
    status = controller.status(batch_id, document_id)
    if not status.finalized and finalize:
        if status.challenge_closure is None and substantive:
            raise ReviewPolicyError(
                "finalization of substantive disputes needs an explicit challenge closure"
            )
        controller.finalize(batch_id, document_id)
        status = controller.status(batch_id, document_id)
    written = None
    packet_blocker = None
    if status.finalized and packet_dir is not None:
        try:
            build_human_verification_packet(
                [controller.reviews_dir],
                output_dir=packet_dir,
                batches_dir=controller.batches_dir,
                document_ids=(document_id,),
            )
        except ReviewUsageError as exc:
            if "phase-1 queue is empty" not in str(exc):
                raise
            notes = {**notes, "packet": str(exc)}
            packet_blocker = "human_packet_empty"
        else:
            written = str(packet_dir)
    if not status.finalized:
        blocker = "human_challenge_closure" if substantive else "human_finalize"
        pending = None
        if packet_dir is not None and challenge is not None and challenge.reconciled:
            build_pending_review_packet(
                controller,
                batch_id=batch_id,
                document_id=document_id,
                output_dir=packet_dir,
            )
            pending = str(packet_dir)
        return _finish(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            stopped_at="waiting_for_human",
            blocker=blocker,
            operator_instruction=_human_instruction(substantive=bool(substantive)),
            notes=notes,
            disputes=disputes,
            decisions=decisions,
            desktop_tasks=tasks,
            packet_dir=pending,
            archive_used=False,
        )
    if written is None:
        return _finish(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            stopped_at="finalized",
            blocker=packet_blocker
            if packet_blocker is not None
            else ("packet_dir_required" if packet_dir is None else None),
            operator_instruction=None,
            notes=notes,
            disputes=disputes,
            decisions=decisions,
            desktop_tasks=tasks,
            packet_dir=None,
            archive_used=False,
        )
    return _finish(
        controller,
        batch_id=batch_id,
        document_id=document_id,
        stopped_at="human_packet_written",
        blocker=None,
        operator_instruction=None,
        notes=notes,
        disputes=disputes,
        decisions=decisions,
        desktop_tasks=tasks,
        packet_dir=written,
        archive_used=False,
    )


def run_payload(result: DocumentRun) -> dict[str, object]:
    """JSON-ready view of one resume. It does not add a ledger event."""

    return {
        "archive_transcription_used_as_reading": (
            result.archive_transcription_used_as_reading
        ),
        "batch_id": result.batch_id,
        "blocker": result.blocker,
        "challenge_providers": result.challenge_providers,
        "challenge_reconciled": result.challenge_reconciled,
        "comparison_sha256": result.comparison_sha256,
        "decisions": list(result.decisions),
        "desktop_tasks": result.desktop_tasks,
        "disputes": list(result.disputes),
        "document_id": result.document_id,
        "execution": list(result.execution),
        "execution_state": result.execution_state,
        "finalized": result.finalized,
        "ledger_consistent": result.ledger_consistent,
        "ledger_sequence": result.ledger_sequence,
        "notes": result.notes,
        "operator_instruction": result.operator_instruction,
        "packet_dir": result.packet_dir,
        "providers": result.providers,
        "stopped_at": result.stopped_at,
    }


def _ensure_first_pass(
    controller: ReviewController,
    *,
    batch_id: str,
    document_id: int,
    queue_root: Path | str,
    preservation_dir: Path,
    mount_root: Path | None,
):
    status = (
        controller.status(batch_id, document_id)
        if _opened(controller, batch_id, document_id)
        else None
    )
    if status is None or any(
        status.providers[provider] == "NOT_STARTED" for provider in _PROVIDERS
    ):
        register_desktop_first_pass(
            controller,
            batch_id=batch_id,
            document_id=document_id,
            queue_root=queue_root,
            mount_root=mount_root,
        )
    accept_desktop_first_pass(
        controller,
        batch_id=batch_id,
        document_id=document_id,
        queue_root=queue_root,
        preservation_dir=preservation_dir,
        mount_root=mount_root,
    )
    return controller.status(batch_id, document_id)


def _opened(controller: ReviewController, batch_id: str, document_id: int) -> bool:
    try:
        controller.status(batch_id, document_id)
    except ReviewPolicyError:
        return False
    return True


def _publish_missing_challenge(
    controller: ReviewController, batch_id: str, document_id: int
) -> None:
    for provider in _PROVIDERS:
        _state, document = controller._require_document(batch_id, document_id)
        if document.challenge_closure is not None or document.finalized:
            return
        runtime = _challenge_runtime(document, provider)
        if runtime is None or runtime.status is ProviderStatus.NOT_STARTED:
            controller.publish(batch_id, document_id, provider, stage="challenge")


def _challenge_runtime(
    document: DocumentReviewState, provider: str
) -> ProviderRuntime | None:
    if not document.challenge_rounds:
        return None
    challenge = document.challenge_rounds[-1]
    return next(item for item in challenge.providers if item.provider == provider)


def _mirror_challenge(
    controller: ReviewController,
    *,
    batch_id: str,
    document_id: int,
    queue_root: Path | str,
    mount_root: Path | None,
) -> dict[str, str]:
    _state, document = controller._require_document(batch_id, document_id)
    if not document.challenge_rounds:
        return {}
    round_index = document.challenge_rounds[-1].round_index
    round_dir, windows_round = _challenge_location(
        queue_root,
        batch_id,
        document_id,
        round_index,
        mount_root,
    )
    round_dir.mkdir(parents=True, exist_ok=True)
    doc_root = document_root(controller.reviews_dir, batch_id, document_id)
    tasks: dict[str, str] = {}
    index_tasks: dict[str, str] = {}
    for provider in _PROVIDERS:
        source = (
            doc_root / stage_dir_name("challenge", round_index) / "tasks" / provider
        )
        if not source.is_dir():
            raise ReviewPolicyError(f"{provider} challenge task has not been published")
        target = round_dir / provider
        _copy_exact_tree(source, target)
        windows_task = f"{windows_round}\\{provider}"
        filename = output_filename("challenge", provider)
        windows_output = f"{windows_task}\\output\\{filename}"
        _place_exact(
            target / "prompt.txt",
            _provider_prompt(provider, windows_task, windows_output).encode("utf-8"),
        )
        (target / "output").mkdir(parents=True, exist_ok=True)
        tasks[f"{provider}_task"] = windows_task
        tasks[f"{provider}_output"] = windows_output
        index_tasks[provider] = windows_output
    payload = (
        json.dumps(
            {
                "batch_id": batch_id,
                "document_id": document_id,
                "gpt_output_file": index_tasks["gpt"],
                "grok_output_file": index_tasks["grok"],
                "round_index": round_index,
                "schema_version": "desktop-challenge-queue-v1",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _place_exact(round_dir / "challenge_queue.json", payload)
    return tasks


def _challenge_location(
    queue_root: Path | str,
    batch_id: str,
    document_id: int,
    round_index: int,
    mount_root: Path | None,
) -> tuple[Path, str]:
    """Challenge folders sit beside the first-pass document, not inside it.

    The first-pass queue rejects any extra name in its document directory.
    """

    location = resolve_desktop_queue_root(queue_root, mount_root=mount_root)
    safe_batch = require_batch_id(batch_id)
    name = f"document_{document_id}_challenge_round_{round_index:02d}"
    return (
        location.local_path / safe_batch / name,
        f"{location.windows_path}\\{safe_batch}\\{name}",
    )


def _provider_prompt(provider: str, windows_task: str, windows_output: str) -> str:
    if provider == "gpt":
        app = (
            "Use Codex Desktop in the existing transcription-prj environment. "
            "Open this task by the absolute paths below."
        )
    else:
        app = (
            "Use Grok Bot Desktop. Do not select a working directory. "
            "Keep Ask every time. Use only the absolute paths below."
        )
    return (
        f"{app}\n"
        "This is a visual challenge of anonymous candidate readings already listed "
        "in input\\disputes.json. It is not a new transcription of the whole page.\n"
        "Do not use an archive transcription, a website transcription, or the other "
        "provider's files. A completed first pass is not the correct reading.\n"
        "Do not change a reading only to force agreement. If the image does not "
        "support a candidate, return UNRESOLVED.\n"
        f"Read {windows_task}\\instructions_{provider}.txt, "
        f"{windows_task}\\work_request.json, {windows_task}\\input\\disputes.json, "
        f"and the page and crop files under {windows_task}\\input.\n"
        "Write one JSON file and no other path:\n"
        f"{windows_output}\n"
        "If challenge_queue.json is missing from the parent directory, stop and "
        "write nothing. Do not open the other provider directory.\n"
    )


def _intake_challenge(
    controller: ReviewController,
    *,
    batch_id: str,
    document_id: int,
    queue_root: Path | str,
    preservation_dir: Path,
    mount_root: Path | None,
) -> dict[str, str]:
    _state, document = controller._require_document(batch_id, document_id)
    if not document.challenge_rounds:
        return {}
    challenge = document.challenge_rounds[-1]
    round_dir, _windows = _challenge_location(
        queue_root,
        batch_id,
        document_id,
        challenge.round_index,
        mount_root,
    )
    doc_root = document_root(controller.reviews_dir, batch_id, document_id)
    notes: dict[str, str] = {}
    for provider in _PROVIDERS:
        runtime = next(
            item for item in challenge.providers if item.provider == provider
        )
        filename = output_filename("challenge", provider)
        desktop = round_dir / provider / "output" / filename
        relative = runtime.output_relative_path or (
            f"{stage_dir_name('challenge', challenge.round_index)}/output/{filename}"
        )
        review_path = doc_root / relative
        if runtime.status is ProviderStatus.SUCCEEDED:
            if desktop.is_file() and desktop.read_bytes() != review_path.read_bytes():
                raise ReviewIntegrityError(
                    f"{provider} desktop challenge file differs from the accepted result"
                )
            controller.accept(
                batch_id,
                document_id,
                provider,
                stage="challenge",
                round_index=challenge.round_index,
            )
            continue
        if runtime.status is not ProviderStatus.AWAITING_MANUAL_DISPATCH:
            notes[provider] = f"{provider} is {runtime.status.value}"
            continue
        if not desktop.is_file():
            notes[provider] = "challenge output is missing"
            continue
        raw = desktop.read_bytes()
        preserved = (
            preservation_dir
            / f"challenge_round_{challenge.round_index:02d}"
            / provider
            / filename
        )
        _place_exact(preserved, raw)
        request = WorkRequest.model_validate_json(
            (round_dir / provider / "work_request.json").read_bytes()
        )
        problem = _challenge_problem(raw, request)
        if problem is not None:
            notes[provider] = problem
            continue
        _place_exact(review_path, raw)
        try:
            controller.accept(
                batch_id,
                document_id,
                provider,
                stage="challenge",
                round_index=challenge.round_index,
            )
        except ReviewOutputRejected as exc:
            notes[provider] = str(exc)
    return notes


def _challenge_problem(raw: bytes, request: WorkRequest) -> str | None:
    try:
        model = ChallengeOutput.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return "challenge output does not match the schema"
    if (
        model.provider != request.provider
        or model.provider_interface != request.provider_interface
        or model.task_id != request.task_id
        or model.job_id != request.job_id
        or model.batch_id != request.batch_id
        or model.document_id != request.document_id
        or model.attempt != request.attempt
        or model.source_bundle_sha256 != request.source_bundle_sha256
        or model.stage != "challenge"
        or model.round_index != request.round_index
    ):
        return "challenge output does not match the published task"
    return template_placeholder_problem(model.model_dump(mode="json"))


def _copy_exact_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ReviewIntegrityError("refusing to copy a challenge task symlink")
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        _place_exact(destination, path.read_bytes())


def _queue_result_hashes(
    queue_root: Path | str,
    batch_id: str,
    document_id: int,
    mount_root: Path | None,
    stage: str,
) -> dict[str, str]:
    if stage != "first_pass":
        return {}
    location = resolve_desktop_queue_root(queue_root, mount_root=mount_root)
    document = (
        location.local_path / require_batch_id(batch_id) / f"document_{document_id}"
    )
    found: dict[str, str] = {}
    for provider in _PROVIDERS:
        path = document / provider / "output" / output_filename("first_pass", provider)
        if path.is_file():
            found[provider] = sha256_bytes(path.read_bytes())
    return found


def _require_same_hashes(before: dict[str, str], after: dict[str, str]) -> None:
    if before != after:
        raise ReviewIntegrityError(
            "desktop first-pass result bytes changed during resume"
        )


def _archive_transcription_leaked(
    controller: ReviewController,
    batch_id: str,
    document_id: int,
    *,
    comparison_text: str | None = None,
) -> bool:
    path = (
        controller.batches_dir
        / require_batch_id(batch_id)
        / "documents"
        / str(document_id)
        / "transcription"
        / "transcription.txt"
    )
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return False
    blob = comparison_text or ""
    doc_root = document_root(controller.reviews_dir, batch_id, document_id)
    comparison = doc_root / "comparison" / "comparison.json"
    if comparison.is_file():
        blob += comparison.read_text(encoding="utf-8")
    _state, document = controller._require_document(batch_id, document_id)
    for challenge in document.challenge_rounds:
        task_root = (
            doc_root / stage_dir_name("challenge", challenge.round_index) / "tasks"
        )
        for file in task_root.rglob("*"):
            if file.suffix.lower() not in {".json", ".txt", ".md"}:
                continue
            if file.is_file() and not file.is_symlink():
                blob += file.read_text(encoding="utf-8", errors="replace")
    return text in blob


def _first_pass_instruction(
    queue_root: Path | str,
    batch_id: str,
    document_id: int,
    mount_root: Path | None,
) -> str:
    location = resolve_desktop_queue_root(queue_root, mount_root=mount_root)
    root = (
        f"{location.windows_path}\\{require_batch_id(batch_id)}\\document_{document_id}"
    )
    return (
        "First-pass desktop results are still required. "
        f"Codex Desktop, transcription-prj, writes only {root}\\gpt\\output\\transcription_gpt.json. "
        "Grok Bot Desktop does not select a working directory and keeps Ask every time. "
        f"It writes only {root}\\grok\\output\\transcription_grok.json."
    )


def _challenge_instruction(tasks: dict[str, str]) -> str:
    gpt = tasks.get("gpt_output", "")
    grok = tasks.get("grok_output", "")
    return (
        "One challenge batch is ready. Do not edit the first-pass files. "
        "Codex Desktop, in the existing transcription-prj environment, reads the "
        f"gpt task directory and writes only {gpt}. "
        "Grok Bot Desktop does not select a working directory and keeps Ask every time. "
        f"It reads the grok task directory by absolute path and writes only {grok}. "
        "Each prompt.txt in that task states the same rule. "
        "Leave both challenge files for desktop-e2e advance to collect."
    )


def _human_instruction(*, substantive: bool) -> str:
    if substantive:
        return (
            "Both challenge results are accepted and reconciled. "
            "The run stops for an explicit challenge closure. "
            "Pass --packet-dir to write a pending human-review page. "
            "Pass --closure-mode and --reason to record closure, "
            "which also finalizes through the existing mechanism. "
            "A packet written after finalization is the completed verification packet. "
            "Neither step approves a reading."
        )
    return (
        "Comparison has no substantive dispute. "
        "Pass --finalize to publish the proposal through the existing mechanism. "
        "Pass --packet-dir when a completed review should become a verification packet."
    )


def _finish(
    controller: ReviewController,
    *,
    batch_id: str,
    document_id: int,
    stopped_at: str,
    blocker: str | None,
    operator_instruction: str | None,
    notes: dict[str, str],
    disputes: tuple[dict[str, object], ...],
    decisions: tuple[dict[str, object], ...],
    desktop_tasks: dict[str, str],
    packet_dir: str | None,
    archive_used: bool,
) -> DocumentRun:
    status = controller.status(batch_id, document_id)
    store = controller._store(batch_id)
    state = store.load()
    if state is None or state.applied_ledger_sequence != len(store.read_ledger()):
        raise ReviewIntegrityError("review ledger does not match its checkpoint")
    _state, document = controller._require_document(batch_id, document_id)
    challenge = document.challenge_rounds[-1] if document.challenge_rounds else None
    challenge_providers: dict[str, str] = {}
    if challenge is not None:
        challenge_providers = {
            item.provider: item.status.value for item in challenge.providers
        }
    return DocumentRun(
        batch_id=batch_id,
        document_id=document_id,
        stopped_at=stopped_at,
        blocker=blocker,
        operator_instruction=operator_instruction,
        execution_state=status.execution_state,
        finalized=status.finalized,
        ledger_sequence=status.ledger_sequence,
        ledger_consistent=True,
        providers=dict(status.providers),
        challenge_providers=challenge_providers,
        comparison_sha256=document.comparison_sha256,
        disputes=disputes,
        challenge_reconciled=bool(challenge and challenge.reconciled),
        decisions=decisions,
        desktop_tasks=desktop_tasks,
        packet_dir=packet_dir,
        notes=notes,
        archive_transcription_used_as_reading=archive_used,
        execution=_execution_report(stopped_at),
    )


def _execution_report(stopped_at: str) -> tuple[dict[str, object], ...]:
    stage = {
        "waiting_for_first_pass": "first_pass",
        "waiting_for_challenge": "challenge",
    }.get(stopped_at)
    if stage is None:
        return ()
    return tuple(item.to_dict() for item in review_dispatch_blocked(stage))
