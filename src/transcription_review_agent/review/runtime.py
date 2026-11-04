"""Append-only review ledger and the checkpoint it replays.

This ledger is the review workflow's history. It does not replace the
discovery ledger in ``state/``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Final, Literal, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from transcription_review_agent.bundling.layout import assert_safe_relative_path
from transcription_review_agent.constants import BATCH_ID_PATTERN, SHA256_PATTERN
from transcription_review_agent.review.errors import ReviewIntegrityError
from transcription_review_agent.review.models import (
    ChallengeClosureRecord,
    ChallengeDecision,
    ExecutionState,
    ProviderStatus,
    QualityLabel,
    ReconciledDecision,
    SourceEvidence,
    UnresolvedDispute,
)
from transcription_review_agent.review.paths import (
    batch_root,
    controller_dir,
    output_relative_path,
)

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
BatchId = Annotated[str, Field(pattern=BATCH_ID_PATTERN)]
ProviderName = Literal["gpt", "grok"]
StageName = Literal["first_pass", "challenge", "research"]
ChallengeContract = Literal["EVIDENCE_BOUND_V1"]
# One new attempt after INVALID_OUTPUT, per provider, per challenge round.
# The rejected attempt stays terminal. A second rejection is not recovered.
MAX_INVALID_OUTPUT_RECOVERIES: Final = 1

INITIAL_UPDATED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


class PageAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str
    reading_order: int = Field(ge=1)
    sha256: Sha256
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    mime_type: Literal["image/png", "image/jpeg"] = "image/png"
    original_filename: str = ""
    derived_png_sha256: Sha256 | None = None


class PageCoverageSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str
    coverage_complete: bool
    unresolved_text_present: bool


class ProviderRuntime(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    status: ProviderStatus = ProviderStatus.NOT_STARTED
    attempt: int = Field(default=0, ge=0)
    transient_failures: int = Field(default=0, ge=0)
    task_id: str | None = None
    job_id: str | None = None
    output_sha256: Sha256 | None = None
    output_relative_path: str | None = None
    coverage: list[PageCoverageSnapshot] = Field(default_factory=list)
    challenge_decisions: list[ChallengeDecision] = Field(default_factory=list)
    source_evidence: list[SourceEvidence] = Field(default_factory=list)
    # Absent on a checkpoint written before recovery existed. Replay of those
    # ledgers leaves this at zero, so the checkpoint still matches.
    invalid_output_recoveries: int = Field(default=0, ge=0)


class ChallengeRoundState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)
    evidence: str | None = None
    evidence_sha256: Sha256 | None = None
    evidence_relative_path: str | None = None
    providers: list[ProviderRuntime]
    reconciled: bool = False
    # A1. ``None`` is a legacy round (or one opened without an explicit
    # selection): it historically covers the full substantive set. A list is
    # an explicit selection, frozen when the first provider task was
    # published. ``selection_comparison_sha256`` binds that list to the
    # recorded comparison.
    selected_dispute_ids: list[str] | None = None
    selection_comparison_sha256: Sha256 | None = None
    # Absent means a historical round published before the evidence-bound
    # contract existed. New rounds always persist EVIDENCE_BOUND_V1.
    challenge_contract: ChallengeContract | None = None


class ReportVersion(BaseModel):
    """One immutable report snapshot. Later versions do not replace it."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    completion: Literal["interim_provisional", "completed"]
    quality: QualityLabel
    relative_dir: str
    artifact_sha256: dict[str, Sha256]

    @field_validator("relative_dir")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class DocumentReviewState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: int = Field(ge=1)
    language: Literal["en", "fr"]
    source_bundle_sha256: Sha256
    archive_transcription_sha256: Sha256
    pages: list[PageAsset]
    first_pass: list[ProviderRuntime]
    research: list[ProviderRuntime]
    challenge_rounds: list[ChallengeRoundState] = Field(default_factory=list)
    comparison_sha256: Sha256 | None = None
    dispute_ids: list[str] = Field(default_factory=list)
    substantive_dispute_ids: list[str] = Field(default_factory=list)
    reconciled_decisions: list[ReconciledDecision] = Field(default_factory=list)
    visual_readings_preserved: bool = False
    research_changes_diplomatic_text: bool = False
    finalized: bool = False
    # A4. Absent means the challenge stage was never explicitly closed. This
    # includes every legacy review: inactivity is never closure.
    challenge_closure: ChallengeClosureRecord | None = None
    quality: QualityLabel | None = None
    final_sha256: dict[str, Sha256] = Field(default_factory=dict)
    report_versions: list[ReportVersion] = Field(default_factory=list)
    estimated_work_units: int = Field(ge=0)
    provider_failure_count: int = Field(default=0, ge=0)
    execution_state: ExecutionState = ExecutionState.READY_FOR_PROVIDER


class ReviewBatchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    batch_id: BatchId
    applied_ledger_sequence: int = Field(default=0, ge=0)
    source_batch_manifest_sha256: Sha256 | None = None
    documents: list[DocumentReviewState] = Field(default_factory=list)
    updated_at: datetime = INITIAL_UPDATED_AT

    @field_validator("updated_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class ReviewLedgerBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    occurred_at: datetime
    batch_id: BatchId
    document_id: int = Field(ge=1)

    @field_validator("occurred_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value


class DocumentOpenedEntry(ReviewLedgerBase):
    event_type: Literal["document_opened"] = "document_opened"
    language: Literal["en", "fr"]
    source_bundle_sha256: Sha256
    source_batch_manifest_sha256: Sha256
    archive_transcription_sha256: Sha256
    pages: list[PageAsset] = Field(min_length=1)
    estimated_work_units: int = Field(ge=1)


class WorkPublishedEntry(ReviewLedgerBase):
    event_type: Literal["work_published"] = "work_published"
    provider: ProviderName
    stage: StageName
    round_index: int = Field(ge=0)
    attempt: int = Field(ge=1)
    task_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    output_relative_path: str
    evidence: str | None = None
    evidence_sha256: Sha256 | None = None
    evidence_relative_path: str | None = None
    # A1. Absent in legacy events, which mean the historical full substantive set.
    selected_dispute_ids: list[str] | None = None
    selection_comparison_sha256: Sha256 | None = None
    challenge_contract: ChallengeContract | None = None

    @field_validator("output_relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)

    @field_validator("evidence_relative_path")
    @classmethod
    def _evidence_relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return assert_safe_relative_path(value)


class ResultAcceptedEntry(ReviewLedgerBase):
    event_type: Literal["result_accepted"] = "result_accepted"
    provider: ProviderName
    stage: StageName
    round_index: int = Field(default=0, ge=0)
    output_sha256: Sha256
    output_relative_path: str
    coverage: list[PageCoverageSnapshot] = Field(default_factory=list)
    decisions: list[ChallengeDecision] = Field(default_factory=list)
    source_evidence: list[SourceEvidence] = Field(default_factory=list)

    @field_validator("output_relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class ResultRejectedEntry(ReviewLedgerBase):
    event_type: Literal["result_rejected"] = "result_rejected"
    provider: ProviderName
    stage: StageName
    round_index: int = Field(default=0, ge=0)
    reason: Literal["invalid_label", "schema", "stale", "mismatched", "duplicated"]
    preserve_success: bool = False


class ProviderFailureEntry(ReviewLedgerBase):
    event_type: Literal["provider_failure"] = "provider_failure"
    provider: ProviderName
    stage: StageName
    round_index: int = Field(default=0, ge=0)
    kind: Literal["transient", "quota"]


class ComparisonRecordedEntry(ReviewLedgerBase):
    event_type: Literal["comparison_recorded"] = "comparison_recorded"
    comparison_sha256: Sha256
    dispute_ids: list[str]
    substantive_dispute_ids: list[str]


class ChallengeReconciledEntry(ReviewLedgerBase):
    event_type: Literal["challenge_reconciled"] = "challenge_reconciled"
    round_index: int = Field(ge=1)
    decisions: list[ReconciledDecision]
    # A4. Absent in legacy events, which keep the historical merge. When set,
    # replay keeps the other rounds' decisions for the same dispute and never
    # lets a later round silently overwrite a resolved one.
    accumulates_history: bool = False


class ChallengeClosedEntry(ReviewLedgerBase):
    """The operator explicitly closed the challenge stage (A4). Append-only."""

    event_type: Literal["challenge_closed"] = "challenge_closed"
    closure: ChallengeClosureRecord


class ReviewProvisionalEntry(ReviewLedgerBase):
    """An interim report. It does not close the document."""

    event_type: Literal["review_provisional"] = "review_provisional"
    version: int = Field(ge=1)
    quality: QualityLabel
    relative_dir: str
    final_sha256: dict[str, Sha256]

    @field_validator("relative_dir")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class ReviewFinalizedEntry(ReviewLedgerBase):
    event_type: Literal["review_finalized"] = "review_finalized"
    quality: QualityLabel
    final_sha256: dict[str, Sha256]
    version: int = Field(default=0, ge=0)
    relative_dir: str | None = None

    @field_validator("relative_dir")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return assert_safe_relative_path(value)


ReviewLedgerEntry = Annotated[
    Union[
        DocumentOpenedEntry,
        WorkPublishedEntry,
        ResultAcceptedEntry,
        ResultRejectedEntry,
        ProviderFailureEntry,
        ComparisonRecordedEntry,
        ChallengeReconciledEntry,
        ChallengeClosedEntry,
        ReviewProvisionalEntry,
        ReviewFinalizedEntry,
    ],
    Field(discriminator="event_type"),
]

LEDGER_ADAPTER = TypeAdapter(ReviewLedgerEntry)


def initial_review_state(
    batch_id: str, now: datetime | None = None
) -> ReviewBatchState:
    return ReviewBatchState(
        batch_id=batch_id,
        updated_at=INITIAL_UPDATED_AT if now is None else now,
    )


def both_providers() -> list[ProviderRuntime]:
    return [ProviderRuntime(provider="gpt"), ProviderRuntime(provider="grok")]


def apply_review_entry(
    state: ReviewBatchState, entry: ReviewLedgerEntry
) -> ReviewBatchState:
    """Return the checkpoint after ``entry``. ``state`` is left unchanged."""

    if state.schema_version != 1:
        raise ReviewIntegrityError("unsupported review state schema")
    if entry.sequence != state.applied_ledger_sequence + 1:
        raise ReviewIntegrityError(
            f"review ledger sequence {entry.sequence} does not follow "
            f"{state.applied_ledger_sequence}"
        )
    if entry.batch_id != state.batch_id:
        raise ReviewIntegrityError(
            "review ledger batch_id does not match the checkpoint"
        )

    if isinstance(entry, DocumentOpenedEntry):
        updated = _apply_opened(state, entry)
    elif isinstance(entry, WorkPublishedEntry):
        updated = _apply_published(state, entry)
    elif isinstance(entry, ResultAcceptedEntry):
        updated = _apply_accepted(state, entry)
    elif isinstance(entry, ResultRejectedEntry):
        updated = _apply_rejected(state, entry)
    elif isinstance(entry, ProviderFailureEntry):
        updated = _apply_failure(state, entry)
    elif isinstance(entry, ComparisonRecordedEntry):
        updated = _apply_comparison(state, entry)
    elif isinstance(entry, ChallengeReconciledEntry):
        updated = _apply_reconciled(state, entry)
    elif isinstance(entry, ChallengeClosedEntry):
        updated = _apply_closed(state, entry)
    elif isinstance(entry, ReviewProvisionalEntry):
        updated = _apply_provisional(state, entry)
    elif isinstance(entry, ReviewFinalizedEntry):
        updated = _apply_finalized(state, entry)
    else:
        raise ReviewIntegrityError("unsupported review ledger entry")
    return updated.model_copy(
        update={
            "applied_ledger_sequence": entry.sequence,
            "updated_at": entry.occurred_at,
        }
    )


def recompute_execution(document: DocumentReviewState) -> ExecutionState:
    """Execution is independent of the quality label."""

    if document.finalized:
        return ExecutionState.COMPLETED
    statuses = [item.status for item in document.first_pass]
    waiting = list(document.first_pass) + list(document.research)
    for challenge in document.challenge_rounds:
        waiting.extend(challenge.providers)
    if any(status is ProviderStatus.QUOTA_BLOCKED for status in statuses):
        return ExecutionState.BLOCKED
    if any(
        item.status in {ProviderStatus.AWAITING_MANUAL_DISPATCH, ProviderStatus.RUNNING}
        for item in waiting
    ):
        return ExecutionState.WAITING_FOR_PROVIDER
    if any(status is ProviderStatus.RETRYABLE_FAILURE for status in statuses):
        return ExecutionState.IN_PROGRESS
    if any(status is ProviderStatus.SUCCEEDED for status in statuses):
        return ExecutionState.IN_PROGRESS
    if all(
        status in {ProviderStatus.FAILED, ProviderStatus.INVALID_OUTPUT}
        for status in statuses
    ):
        return ExecutionState.FAILED
    if all(status is ProviderStatus.NOT_STARTED for status in statuses):
        return ExecutionState.READY_FOR_PROVIDER
    return ExecutionState.IN_PROGRESS


def resolved_dispute_ids(document: DocumentReviewState) -> list[str]:
    """Substantive IDs that an earlier reconciliation decided (not ``UNRESOLVED``)."""

    return [
        item.dispute_id
        for item in document.reconciled_decisions
        if item.decision != "UNRESOLVED"
    ]


def round_selection(document: DocumentReviewState, round_index: int) -> list[str]:
    """The dispute IDs a round covers.

    A round without a recorded selection is a legacy round and covers the full
    substantive set, which is what the workflow did before selections existed.
    """

    challenge = _require_round(document, round_index)
    if challenge.selected_dispute_ids is None:
        return list(document.substantive_dispute_ids)
    return list(challenge.selected_dispute_ids)


def normalize_challenge_selection(
    selected: Sequence[str],
    *,
    substantive_ids: Sequence[str],
    resolved_ids: Sequence[str],
) -> list[str]:
    """Validate an explicit selection and return it in comparison order.

    Raises ``ValueError`` for an empty, duplicate, unknown (including stale or
    non-substantive), or already resolved ID. Order is canonicalized to the
    comparison order so one subset has one representation.
    """

    ids = list(selected)
    if not ids:
        raise ValueError("a challenge selection must not be empty")
    if any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("challenge selection IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("challenge selection contains a duplicate ID")
    known = set(substantive_ids)
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ValueError(
            "challenge selection has IDs that are not current substantive disputes: "
            + ", ".join(unknown)
        )
    already = sorted(set(ids) & set(resolved_ids))
    if already:
        raise ValueError(
            "challenge selection has IDs that are already resolved: "
            + ", ".join(already)
        )
    chosen = set(ids)
    return [item for item in substantive_ids if item in chosen]


OPEN_DISPATCH_STATUSES = frozenset(
    {
        ProviderStatus.AWAITING_MANUAL_DISPATCH,
        ProviderStatus.RUNNING,
        ProviderStatus.RETRYABLE_FAILURE,
    }
)
TERMINAL_FAILURE_STATUSES = frozenset(
    {
        ProviderStatus.FAILED,
        ProviderStatus.INVALID_OUTPUT,
        ProviderStatus.QUOTA_BLOCKED,
    }
)


def challenge_round_open(challenge: ChallengeRoundState) -> bool:
    """A provider of this round still has a live (or retryable) dispatch."""

    return any(item.status in OPEN_DISPATCH_STATUSES for item in challenge.providers)


def challenge_round_failed(challenge: ChallengeRoundState) -> bool:
    """A confirmed terminal provider failure with no live dispatch left.

    A round that is still open is never failed: its task is active and must be
    accepted or recorded as failed first. A reconciled round is never failed.
    """

    if challenge.reconciled or challenge_round_open(challenge):
        return False
    return any(item.status in TERMINAL_FAILURE_STATUSES for item in challenge.providers)


def accumulate_decision(
    earlier: ReconciledDecision,
    later: ReconciledDecision,
) -> ReconciledDecision:
    """Combine two rounds' decisions for one dispute without losing either.

    The standing record is returned; the other round's record is kept verbatim
    in ``other_round_decisions``.

    * Earlier ``UNRESOLVED``: the later record stands and keeps the earlier one.
    * Earlier resolved, later ``UNRESOLVED``: the earlier decision stands.
      Insufficient later evidence does not undo it, and the later record is kept.
    * Earlier resolved, later resolved to a different reading: the evidence
      contradicts itself, so the standing record is ``UNRESOLVED`` with basis
      ``CONTRADICTORY_ROUND_EVIDENCE``. No winner is chosen.
    * Earlier resolved, later the same reading: the later record stands.
    """

    earlier_flat = earlier.model_copy(update={"other_round_decisions": None})
    later_flat = later.model_copy(update={"other_round_decisions": None})
    history = list(earlier.other_round_decisions or [])
    earlier_resolved = earlier.decision != "UNRESOLVED"
    if earlier_resolved and later.decision == "UNRESOLVED":
        return earlier_flat.model_copy(
            update={"other_round_decisions": [*history, later_flat]}
        )
    contradicts = earlier_resolved and (
        (later.decision, later.third_reading)
        != (earlier.decision, earlier.third_reading)
    )
    if contradicts:
        return later_flat.model_copy(
            update={
                "decision": "UNRESOLVED",
                "third_reading": None,
                "agreed": False,
                "remaining_uncertainty": True,
                "result_class": "UNRESOLVED",
                "basis": "CONTRADICTORY_ROUND_EVIDENCE",
                "selected_reading": None,
                "other_round_decisions": [*history, earlier_flat, later_flat],
            }
        )
    return later_flat.model_copy(
        update={"other_round_decisions": [*history, earlier_flat]}
    )


def describe_unresolved(
    document: DocumentReviewState,
    failed_round_indexes: Sequence[int] = (),
) -> list[UnresolvedDispute]:
    """Every substantive dispute without a resolved decision, with its reason.

    Pure and deterministic; it never invents a decision. Order is the
    comparison order.
    """

    decisions = {item.dispute_id: item for item in document.reconciled_decisions}
    failed = set(failed_round_indexes)
    result: list[UnresolvedDispute] = []
    for dispute_id in document.substantive_dispute_ids:
        decision = decisions.get(dispute_id)
        if decision is not None and decision.decision != "UNRESOLVED":
            continue
        selected_in = [
            item.round_index
            for item in document.challenge_rounds
            if dispute_id in round_selection(document, item.round_index)
        ]
        status: Literal[
            "NOT_CHALLENGED", "NOT_SELECTED", "SELECTED_UNRESOLVED", "PROVIDER_FAILURE"
        ]
        if not document.challenge_rounds:
            status = "NOT_CHALLENGED"
        elif not selected_in:
            status = "NOT_SELECTED"
        elif selected_in[-1] in failed:
            status = "PROVIDER_FAILURE"
        else:
            status = "SELECTED_UNRESOLVED"
        result.append(
            UnresolvedDispute(
                dispute_id=dispute_id,
                status=status,
                selected_in_rounds=selected_in,
                basis=None if decision is None else decision.basis,
            )
        )
    return result


def challenge_closure_problem(
    document: DocumentReviewState,
    record: ChallengeClosureRecord,
) -> str | None:
    """Why ``record`` cannot close this document's challenge stage, or ``None``.

    This is the single rule used by the controller (as a policy error) and by
    ledger replay (as an integrity error). Closure is only ever explicit; it
    never closes an active dispatch.
    """

    if document.finalized:
        return "a finalized review cannot be closed"
    if document.challenge_closure is not None:
        return "the challenge stage is already closed"
    if document.comparison_sha256 is None:
        return "comparison must be recorded before the challenge stage is closed"
    if not document.substantive_dispute_ids:
        return "there are no substantive disputes to close"
    if record.comparison_sha256 != document.comparison_sha256:
        return "the closure is not bound to the recorded comparison"
    rounds = document.challenge_rounds
    if record.round_count != len(rounds):
        return "the closure round count does not match the recorded rounds"
    for challenge in rounds:
        if challenge_round_open(challenge):
            return (
                f"challenge round {challenge.round_index} has an open provider dispatch "
                "and cannot be closed; accept its result or record its failure first"
            )
        if not challenge.reconciled and all(
            item.status is ProviderStatus.SUCCEEDED for item in challenge.providers
        ):
            return (
                f"challenge round {challenge.round_index} has both results but is not "
                "reconciled yet"
            )
    unreconciled = [item.round_index for item in rounds if not item.reconciled]
    failed = list(record.failed_round_indexes)
    if record.mode == "PROVIDER_TERMINAL_FAILURE":
        if not failed or failed != unreconciled:
            return (
                "a provider-failure closure must list exactly the rounds that ended "
                "without a reconciled result"
            )
        for challenge in rounds:
            if challenge.round_index in failed and not challenge_round_failed(
                challenge
            ):
                return (
                    f"challenge round {challenge.round_index} has no confirmed terminal "
                    "provider failure"
                )
    else:
        if failed:
            return "only a provider-failure closure lists failed rounds"
        if unreconciled:
            return (
                f"challenge round {unreconciled[0]} ended without a reconciled result; "
                "close it with PROVIDER_TERMINAL_FAILURE"
            )
    if record.mode == "NO_CHALLENGE_REQUESTED":
        if rounds:
            return (
                "NO_CHALLENGE_REQUESTED applies only when no challenge round was opened"
            )
    elif not rounds:
        return f"{record.mode} requires at least one challenge round"
    if record.unresolved != describe_unresolved(document, failed):
        return "the closure's unresolved disputes do not match the recorded state"
    if record.mode in {"NO_NEW_INFORMATION", "ROUND_CAP"} and not record.unresolved:
        return f"{record.mode} requires an unresolved dispute"
    if record.mode == "ROUND_CAP" and (
        record.max_challenge_rounds is None
        or record.round_count < record.max_challenge_rounds
    ):
        return "the configured challenge round cap has not been reached"
    return None


def provider_runtime(
    document: DocumentReviewState,
    *,
    stage: str,
    provider: str,
    round_index: int,
) -> ProviderRuntime:
    for item in _providers(document, stage, round_index):
        if item.provider == provider:
            return item
    raise ReviewIntegrityError(f"{provider} is not on {stage}")


def parse_review_entry(data: dict[str, object]) -> ReviewLedgerEntry:
    try:
        return LEDGER_ADAPTER.validate_python(data)
    except ValueError as exc:
        raise ReviewIntegrityError(
            "review ledger entry does not match the schema"
        ) from exc


def dump_review_entry(entry: ReviewLedgerEntry) -> dict[str, object]:
    if isinstance(entry, BaseModel):
        return entry.model_dump(mode="json")
    raise TypeError("review ledger entry must be a pydantic model")


class ReviewStore:
    """Filesystem checkpoint plus JSONL ledger for one review batch."""

    def __init__(self, reviews_dir: Path, batch_id: str) -> None:
        self.reviews_dir = reviews_dir
        self.batch_id = batch_id
        root = controller_dir(reviews_dir, batch_id)
        self.ledger_path = root / "ledger.jsonl"
        self.checkpoint_path = root / "state.json"
        self.header_path = batch_root(reviews_dir, batch_id) / "manifest.json"

    def load(self) -> ReviewBatchState | None:
        entries = self.read_ledger()
        if not entries and not self.checkpoint_path.exists():
            return None
        try:
            replayed = _replay(self.batch_id, entries)
        except ReviewIntegrityError:
            raise
        except ValueError as exc:
            raise ReviewIntegrityError("review ledger replay failed") from exc
        if not self.checkpoint_path.exists():
            if entries:
                self.write_checkpoint(replayed)
            return replayed
        checkpoint = _read_checkpoint(self.checkpoint_path)
        if checkpoint.applied_ledger_sequence > len(entries):
            raise ReviewIntegrityError("review checkpoint is ahead of the ledger")
        prefix = _replay(self.batch_id, entries[: checkpoint.applied_ledger_sequence])
        if checkpoint != prefix:
            raise ReviewIntegrityError("review checkpoint does not match ledger replay")
        if checkpoint.applied_ledger_sequence < len(entries):
            self.write_checkpoint(replayed)
            return replayed
        return checkpoint

    def commit(self, entry: ReviewLedgerEntry) -> ReviewBatchState:
        current = self.load()
        if current is None:
            current = initial_review_state(self.batch_id)
        updated = apply_review_entry(current, entry)
        self.append_ledger_line(entry)
        self.write_checkpoint(updated)
        self.write_header(updated)
        return updated

    def read_ledger(self) -> list[ReviewLedgerEntry]:
        if not self.ledger_path.exists():
            return []
        text = self.ledger_path.read_text(encoding="utf-8")
        if text == "":
            return []
        if not text.endswith("\n"):
            raise ReviewIntegrityError("review ledger must end with a newline")
        entries: list[ReviewLedgerEntry] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise ReviewIntegrityError(f"blank review ledger line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewIntegrityError(
                    f"review ledger line {line_number} is not JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise ReviewIntegrityError(
                    f"review ledger line {line_number} is not an object"
                )
            entry = parse_review_entry(payload)
            if entry.sequence != line_number:
                raise ReviewIntegrityError(
                    f"review sequence {entry.sequence} is not contiguous"
                )
            entries.append(entry)
        return entries

    def append_ledger_line(self, entry: ReviewLedgerEntry) -> None:
        existing = self.read_ledger()
        expected = len(existing) + 1
        if entry.sequence != expected:
            raise ReviewIntegrityError(
                f"refusing to append review sequence {entry.sequence}; next is {expected}"
            )
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            dump_review_entry(entry), separators=(",", ":"), sort_keys=True
        )
        payload = (line + "\n").encode("utf-8")
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        fd = os.open(self.ledger_path, flags, 0o644)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)

    def write_checkpoint(self, state: ReviewBatchState) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.checkpoint_path,
            (state.model_dump_json(indent=2) + "\n").encode("utf-8"),
        )

    def write_header(self, state: ReviewBatchState) -> None:
        from transcription_review_agent.review.models import ReviewBatchHeader

        if state.source_batch_manifest_sha256 is None:
            return
        header = ReviewBatchHeader(
            batch_id=state.batch_id,
            source_batch_manifest_sha256=state.source_batch_manifest_sha256,
            document_ids=[document.document_id for document in state.documents],
        )
        self.header_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.header_path, (header.model_dump_json(indent=2) + "\n").encode("utf-8")
        )


def find_document(state: ReviewBatchState, document_id: int) -> DocumentReviewState:
    for document in state.documents:
        if document.document_id == document_id:
            return document
    raise ReviewIntegrityError(f"document {document_id} is not open for review")


def _replay(batch_id: str, entries: list[ReviewLedgerEntry]) -> ReviewBatchState:
    state = initial_review_state(batch_id)
    for entry in entries:
        state = apply_review_entry(state, entry)
    return state


def _apply_opened(
    state: ReviewBatchState, entry: DocumentOpenedEntry
) -> ReviewBatchState:
    if any(document.document_id == entry.document_id for document in state.documents):
        raise ReviewIntegrityError("document review is already open")
    if state.source_batch_manifest_sha256 not in {
        None,
        entry.source_batch_manifest_sha256,
    }:
        raise ReviewIntegrityError("source batch manifest hash changed")
    orders = [page.reading_order for page in entry.pages]
    if orders != list(range(1, len(entry.pages) + 1)):
        raise ReviewIntegrityError("page reading order must be 1..n")
    document = DocumentReviewState(
        document_id=entry.document_id,
        language=entry.language,
        source_bundle_sha256=entry.source_bundle_sha256,
        archive_transcription_sha256=entry.archive_transcription_sha256,
        pages=list(entry.pages),
        first_pass=both_providers(),
        research=both_providers(),
        estimated_work_units=entry.estimated_work_units,
        execution_state=ExecutionState.READY_FOR_PROVIDER,
    )
    return state.model_copy(
        update={
            "source_batch_manifest_sha256": entry.source_batch_manifest_sha256,
            "documents": [*state.documents, document],
        }
    )


def _recovered_attempt(
    document: DocumentReviewState,
    runtime: ProviderRuntime,
    entry: WorkPublishedEntry,
) -> int:
    """Allow one new challenge attempt after INVALID_OUTPUT.

    The rejected attempt is unchanged: its rejection event stays in the ledger,
    and this publication must carry a new task id, job id, and output path.
    Returns the recovery count to store on the new attempt.
    """

    if entry.stage != "challenge":
        raise ReviewIntegrityError("only a challenge attempt can be recovered")
    if document.finalized:
        raise ReviewIntegrityError(
            "a finalized review cannot recover a challenge attempt"
        )
    challenge = _require_round(document, entry.round_index)
    if challenge.reconciled:
        raise ReviewIntegrityError("a reconciled challenge round cannot be recovered")
    if runtime.invalid_output_recoveries >= MAX_INVALID_OUTPUT_RECOVERIES:
        raise ReviewIntegrityError("invalid-output recovery limit is already spent")
    if entry.attempt != runtime.attempt + 1:
        raise ReviewIntegrityError("published attempt does not follow the provider")
    expected_task = (
        f"{entry.batch_id}:{entry.document_id}:{entry.stage}:"
        f"r{entry.round_index}:{entry.provider}:a{entry.attempt}"
    )
    if entry.task_id != expected_task or entry.job_id != f"job:{expected_task}":
        raise ReviewIntegrityError(
            "a recovered attempt must use its own task and job id"
        )
    if (
        entry.task_id == runtime.task_id
        or entry.job_id == runtime.job_id
        or entry.output_relative_path == runtime.output_relative_path
    ):
        raise ReviewIntegrityError(
            "a recovered attempt must not reuse the rejected attempt identity"
        )
    expected_output = output_relative_path(
        entry.stage, entry.provider, entry.round_index, entry.attempt
    )
    if entry.output_relative_path != expected_output:
        raise ReviewIntegrityError("a recovered attempt must use its own output path")
    return runtime.invalid_output_recoveries + 1


def _apply_published(
    state: ReviewBatchState, entry: WorkPublishedEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    document = _ensure_challenge_round(document, entry)
    runtime = provider_runtime(
        document,
        stage=entry.stage,
        provider=entry.provider,
        round_index=entry.round_index,
    )
    if (
        runtime.status is ProviderStatus.AWAITING_MANUAL_DISPATCH
        and runtime.task_id == entry.task_id
    ):
        raise ReviewIntegrityError("duplicate work publication")
    recoveries = runtime.invalid_output_recoveries
    if runtime.status is ProviderStatus.INVALID_OUTPUT:
        recoveries = _recovered_attempt(document, runtime, entry)
    elif runtime.status not in {
        ProviderStatus.NOT_STARTED,
        ProviderStatus.RETRYABLE_FAILURE,
    }:
        raise ReviewIntegrityError("work cannot be published from this provider status")
    else:
        expected = (
            1 if runtime.status is ProviderStatus.NOT_STARTED else runtime.attempt + 1
        )
        if entry.attempt != expected:
            raise ReviewIntegrityError("published attempt does not follow the provider")
        if runtime.transient_failures > 1:
            raise ReviewIntegrityError("transient retry cap is already spent")
    updated = runtime.model_copy(
        update={
            "status": ProviderStatus.AWAITING_MANUAL_DISPATCH,
            "attempt": entry.attempt,
            "task_id": entry.task_id,
            "job_id": entry.job_id,
            "output_relative_path": entry.output_relative_path,
            "invalid_output_recoveries": recoveries,
        }
    )
    document = _replace_provider(document, entry.stage, entry.round_index, updated)
    return _replace_document(state, _with_execution(document))


def _apply_accepted(
    state: ReviewBatchState, entry: ResultAcceptedEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    runtime = provider_runtime(
        document,
        stage=entry.stage,
        provider=entry.provider,
        round_index=entry.round_index,
    )
    if runtime.status is not ProviderStatus.AWAITING_MANUAL_DISPATCH:
        raise ReviewIntegrityError("only a dispatched provider result can be accepted")
    if entry.stage == "first_pass":
        page_ids = [page.page_id for page in document.pages]
        if [item.page_id for item in entry.coverage] != page_ids:
            raise ReviewIntegrityError("accepted coverage does not list every page")
    if entry.stage == "challenge":
        challenge = _require_round(document, entry.round_index)
        selected = challenge.selected_dispute_ids
        if selected is not None:
            got = [item.dispute_id for item in entry.decisions]
            if len(got) != len(set(got)) or set(got) != set(selected):
                raise ReviewIntegrityError(
                    "accepted challenge decisions must match the selected disputes exactly"
                )
        if challenge.challenge_contract == "EVIDENCE_BOUND_V1" and any(
            item.evidence_ref is None or item.evidence_assessment is None
            for item in entry.decisions
        ):
            raise ReviewIntegrityError(
                "an evidence-bound accepted result cannot omit evidence binding"
            )
    updated = runtime.model_copy(
        update={
            "status": ProviderStatus.SUCCEEDED,
            "output_sha256": entry.output_sha256,
            "output_relative_path": entry.output_relative_path,
            "coverage": list(entry.coverage),
            "challenge_decisions": list(entry.decisions),
            "source_evidence": list(entry.source_evidence),
        }
    )
    document = _replace_provider(document, entry.stage, entry.round_index, updated)
    if entry.stage == "first_pass":
        document = document.model_copy(update={"visual_readings_preserved": True})
    return _replace_document(state, _with_execution(document))


def _apply_rejected(
    state: ReviewBatchState, entry: ResultRejectedEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    runtime = provider_runtime(
        document,
        stage=entry.stage,
        provider=entry.provider,
        round_index=entry.round_index,
    )
    document = document.model_copy(
        update={"provider_failure_count": document.provider_failure_count + 1}
    )
    if entry.preserve_success:
        if runtime.status is not ProviderStatus.SUCCEEDED:
            raise ReviewIntegrityError(
                "a conflicting file cannot replace a missing success"
            )
        return _replace_document(state, _with_execution(document))
    if runtime.status is not ProviderStatus.AWAITING_MANUAL_DISPATCH:
        raise ReviewIntegrityError("invalid output requires a dispatched attempt")
    updated = runtime.model_copy(update={"status": ProviderStatus.INVALID_OUTPUT})
    document = _replace_provider(document, entry.stage, entry.round_index, updated)
    return _replace_document(state, _with_execution(document))


def _apply_failure(
    state: ReviewBatchState, entry: ProviderFailureEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    runtime = provider_runtime(
        document,
        stage=entry.stage,
        provider=entry.provider,
        round_index=entry.round_index,
    )
    if runtime.status not in {
        ProviderStatus.AWAITING_MANUAL_DISPATCH,
        ProviderStatus.RUNNING,
    }:
        raise ReviewIntegrityError("failure requires a dispatched attempt")
    document = document.model_copy(
        update={"provider_failure_count": document.provider_failure_count + 1}
    )
    if entry.kind == "quota":
        updated = runtime.model_copy(update={"status": ProviderStatus.QUOTA_BLOCKED})
    elif runtime.transient_failures >= 1:
        updated = runtime.model_copy(
            update={
                "status": ProviderStatus.FAILED,
                "transient_failures": runtime.transient_failures + 1,
            }
        )
    else:
        updated = runtime.model_copy(
            update={
                "status": ProviderStatus.RETRYABLE_FAILURE,
                "transient_failures": runtime.transient_failures + 1,
            }
        )
    document = _replace_provider(document, entry.stage, entry.round_index, updated)
    return _replace_document(state, _with_execution(document))


def _apply_comparison(
    state: ReviewBatchState, entry: ComparisonRecordedEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    if any(item.status is not ProviderStatus.SUCCEEDED for item in document.first_pass):
        raise ReviewIntegrityError("comparison requires both provider results")
    if document.comparison_sha256 not in {None, entry.comparison_sha256}:
        raise ReviewIntegrityError("comparison hash changed")
    document = document.model_copy(
        update={
            "comparison_sha256": entry.comparison_sha256,
            "dispute_ids": list(entry.dispute_ids),
            "substantive_dispute_ids": list(entry.substantive_dispute_ids),
            "visual_readings_preserved": True,
        }
    )
    return _replace_document(state, _with_execution(document))


def _apply_reconciled(
    state: ReviewBatchState, entry: ChallengeReconciledEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    challenge = _require_round(document, entry.round_index)
    if challenge.reconciled:
        raise ReviewIntegrityError("challenge round is already reconciled")
    if any(item.status is not ProviderStatus.SUCCEEDED for item in challenge.providers):
        raise ReviewIntegrityError("reconciliation requires both challenge results")
    decisions = list(entry.decisions)
    if challenge.challenge_contract == "EVIDENCE_BOUND_V1" and any(
        item.result_class is None for item in decisions
    ):
        raise ReviewIntegrityError(
            "an evidence-bound round cannot replay legacy reconciliation"
        )
    if challenge.selected_dispute_ids is not None:
        # Explicit selection: decide exactly the selected IDs, keep every
        # earlier decision for other IDs, and never invent one for the rest.
        got = [item.dispute_id for item in entry.decisions]
        if len(got) != len(set(got)) or set(got) != set(challenge.selected_dispute_ids):
            raise ReviewIntegrityError(
                "reconciliation must decide exactly the selected disputes"
            )
        new_ids = set(got)
        if entry.accumulates_history:
            earlier = {item.dispute_id: item for item in document.reconciled_decisions}
            decisions = [
                accumulate_decision(earlier[item.dispute_id], item)
                if item.dispute_id in earlier
                else item
                for item in decisions
            ]
        kept = [
            item
            for item in document.reconciled_decisions
            if item.dispute_id not in new_ids
        ]
        merged = {item.dispute_id: item for item in [*kept, *decisions]}
        decisions = [
            merged[dispute_id]
            for dispute_id in document.substantive_dispute_ids
            if dispute_id in merged
        ]
    rounds = [
        challenge.model_copy(update={"reconciled": True})
        if item.round_index == entry.round_index
        else item
        for item in document.challenge_rounds
    ]
    document = document.model_copy(
        update={
            "challenge_rounds": rounds,
            "reconciled_decisions": decisions,
        }
    )
    return _replace_document(state, _with_execution(document))


def _apply_closed(
    state: ReviewBatchState, entry: ChallengeClosedEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    problem = challenge_closure_problem(document, entry.closure)
    if problem is not None:
        raise ReviewIntegrityError(problem)
    document = document.model_copy(update={"challenge_closure": entry.closure})
    return _replace_document(state, _with_execution(document))


def _version_dir(version: int) -> str:
    return f"final/versions/v{version:03d}"


def _append_version(
    document: DocumentReviewState,
    *,
    version: int,
    completion: Literal["interim_provisional", "completed"],
    quality: QualityLabel,
    relative_dir: str,
    artifact_sha256: dict[str, str],
) -> list[ReportVersion]:
    if version != len(document.report_versions) + 1:
        raise ReviewIntegrityError("report versions must be contiguous")
    if relative_dir != _version_dir(version):
        raise ReviewIntegrityError("report version path is not allowlisted")
    return [
        *document.report_versions,
        ReportVersion(
            version=version,
            completion=completion,
            quality=quality,
            relative_dir=relative_dir,
            artifact_sha256=dict(artifact_sha256),
        ),
    ]


def _apply_provisional(
    state: ReviewBatchState, entry: ReviewProvisionalEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    if document.finalized:
        raise ReviewIntegrityError(
            "a finalized review cannot gain a provisional report"
        )
    if document.research_changes_diplomatic_text:
        raise ReviewIntegrityError("research must not change diplomatic text")
    document = document.model_copy(
        update={
            "quality": entry.quality,
            "report_versions": _append_version(
                document,
                version=entry.version,
                completion="interim_provisional",
                quality=entry.quality,
                relative_dir=entry.relative_dir,
                artifact_sha256=entry.final_sha256,
            ),
        }
    )
    return _replace_document(state, _with_execution(document))


def _apply_finalized(
    state: ReviewBatchState, entry: ReviewFinalizedEntry
) -> ReviewBatchState:
    document = find_document(state, entry.document_id)
    if document.finalized:
        raise ReviewIntegrityError("review is already finalized")
    if document.research_changes_diplomatic_text:
        raise ReviewIntegrityError("research must not change diplomatic text")
    versions = list(document.report_versions)
    if entry.version:
        if entry.relative_dir is None:
            raise ReviewIntegrityError("a versioned final report needs its directory")
        versions = _append_version(
            document,
            version=entry.version,
            completion="completed",
            quality=entry.quality,
            relative_dir=entry.relative_dir,
            artifact_sha256=entry.final_sha256,
        )
    document = document.model_copy(
        update={
            "finalized": True,
            "quality": entry.quality,
            "final_sha256": dict(entry.final_sha256),
            "report_versions": versions,
            "execution_state": ExecutionState.COMPLETED,
        }
    )
    return _replace_document(state, document)


def _ensure_challenge_round(
    document: DocumentReviewState,
    entry: WorkPublishedEntry,
) -> DocumentReviewState:
    if entry.stage != "challenge":
        if entry.round_index != 0 or entry.evidence is not None:
            raise ReviewIntegrityError("only a challenge publication carries a round")
        if (
            entry.selected_dispute_ids is not None
            or entry.selection_comparison_sha256 is not None
            or entry.challenge_contract is not None
        ):
            raise ReviewIntegrityError(
                "only a challenge publication carries a selection or challenge contract"
            )
        return document
    if entry.round_index < 1:
        raise ReviewIntegrityError("challenge rounds start at 1")
    if document.challenge_closure is not None:
        raise ReviewIntegrityError("the challenge stage is closed")
    existing = [item.round_index for item in document.challenge_rounds]
    if entry.round_index in existing:
        current = _require_round(document, entry.round_index)
        if (
            entry.evidence != current.evidence
            or entry.evidence_sha256 != current.evidence_sha256
        ):
            raise ReviewIntegrityError("challenge evidence changed")
        if (
            entry.selected_dispute_ids != current.selected_dispute_ids
            or entry.selection_comparison_sha256 != current.selection_comparison_sha256
            or entry.challenge_contract != current.challenge_contract
        ):
            raise ReviewIntegrityError(
                "challenge selection and contract are frozen at the first provider publication"
            )
        return document
    selection = _validated_event_selection(document, entry)
    if entry.round_index != len(existing) + 1:
        raise ReviewIntegrityError("challenge rounds must be contiguous")
    if entry.round_index == 1 and (
        entry.evidence is not None or entry.evidence_sha256 is not None
    ):
        raise ReviewIntegrityError("the first challenge round has no extra evidence")
    if entry.round_index > 1 and (not entry.evidence or entry.evidence_sha256 is None):
        raise ReviewIntegrityError(
            "a later challenge round requires a new evidence file"
        )
    created = ChallengeRoundState(
        round_index=entry.round_index,
        evidence=entry.evidence,
        evidence_sha256=entry.evidence_sha256,
        evidence_relative_path=entry.evidence_relative_path,
        providers=both_providers(),
        selected_dispute_ids=selection,
        selection_comparison_sha256=(
            None if selection is None else entry.selection_comparison_sha256
        ),
        challenge_contract=entry.challenge_contract,
    )
    return document.model_copy(
        update={"challenge_rounds": [*document.challenge_rounds, created]}
    )


def _validated_event_selection(
    document: DocumentReviewState,
    entry: WorkPublishedEntry,
) -> list[str] | None:
    """Check a first-publication selection against the recorded comparison.

    A legacy event has no selection and replays as the full substantive set.
    """

    if entry.selected_dispute_ids is None:
        if entry.selection_comparison_sha256 is not None:
            raise ReviewIntegrityError(
                "a comparison binding needs a challenge selection"
            )
        return None
    if document.comparison_sha256 is None:
        raise ReviewIntegrityError(
            "a challenge selection requires a recorded comparison"
        )
    if entry.selection_comparison_sha256 != document.comparison_sha256:
        raise ReviewIntegrityError(
            "challenge selection is not bound to the recorded comparison"
        )
    try:
        canonical = normalize_challenge_selection(
            entry.selected_dispute_ids,
            substantive_ids=document.substantive_dispute_ids,
            resolved_ids=resolved_dispute_ids(document),
        )
    except ValueError as exc:
        raise ReviewIntegrityError(str(exc)) from exc
    if canonical != list(entry.selected_dispute_ids):
        raise ReviewIntegrityError("challenge selection is not in comparison order")
    return canonical


def _providers(
    document: DocumentReviewState,
    stage: str,
    round_index: int,
) -> list[ProviderRuntime]:
    if stage == "first_pass":
        return document.first_pass
    if stage == "research":
        return document.research
    if stage == "challenge":
        return _require_round(document, round_index).providers
    raise ReviewIntegrityError(f"unknown stage {stage}")


def _require_round(
    document: DocumentReviewState, round_index: int
) -> ChallengeRoundState:
    for item in document.challenge_rounds:
        if item.round_index == round_index:
            return item
    raise ReviewIntegrityError(f"challenge round {round_index} is not open")


def _replace_provider(
    document: DocumentReviewState,
    stage: str,
    round_index: int,
    updated: ProviderRuntime,
) -> DocumentReviewState:
    def replace(items: list[ProviderRuntime]) -> list[ProviderRuntime]:
        return [
            updated if item.provider == updated.provider else item for item in items
        ]

    if stage == "first_pass":
        return document.model_copy(update={"first_pass": replace(document.first_pass)})
    if stage == "research":
        return document.model_copy(update={"research": replace(document.research)})
    rounds = []
    for item in document.challenge_rounds:
        if item.round_index == round_index:
            rounds.append(
                item.model_copy(update={"providers": replace(item.providers)})
            )
        else:
            rounds.append(item)
    return document.model_copy(update={"challenge_rounds": rounds})


def _replace_document(
    state: ReviewBatchState,
    document: DocumentReviewState,
) -> ReviewBatchState:
    documents = [
        document if item.document_id == document.document_id else item
        for item in state.documents
    ]
    return state.model_copy(update={"documents": documents})


def _with_execution(document: DocumentReviewState) -> DocumentReviewState:
    return document.model_copy(
        update={"execution_state": recompute_execution(document)}
    )


def _read_checkpoint(path: Path) -> ReviewBatchState:
    try:
        return ReviewBatchState.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ReviewIntegrityError(
            "review checkpoint does not match the schema"
        ) from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
    fd = os.open(temporary, flags, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
