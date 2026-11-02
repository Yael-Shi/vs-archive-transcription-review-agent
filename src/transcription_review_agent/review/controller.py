"""File-based review controller.

The controller is the only writer of review state. Provider files are data.
Dispatch stays manual: this module never calls a model API and never types
into a desktop application.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_serializer,
    model_validator,
)

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.config import FORBIDDEN_WRITE_ROOTS
from transcription_review_agent.bundling.manifest import DocumentBundleManifest
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.bundling.verify import (
    verify_batch_bundle,
    verify_document_bundle,
)
from transcription_review_agent.review.assemble import (
    build_report,
    classify_quality,
    coverage_gap_ids,
    hebrew_summary,
    render_transcription,
)
from transcription_review_agent.review.adjudicate import (
    DisputeEvidenceScope,
    EvidenceBindingError,
    check_evidence_binding,
    reconcile_challenge,
)
from transcription_review_agent.review.challenge_groups import (
    ChallengeGroup,
    group_challenge_disputes,
    required_decision_ids,
)
from transcription_review_agent.review.compare import (
    challengeable_disputes,
    compare_transcriptions,
)
from transcription_review_agent.review.config import ReviewConfig, load_review_config
from transcription_review_agent.review.crop import (
    assert_crop_contains_target,
    build_crop,
    materialize_crop,
    page_bbox,
    png_dimensions,
)
from transcription_review_agent.review.jpegutil import decode_jpeg, jpeg_dimensions
from transcription_review_agent.review.pngutil import encode_png
from transcription_review_agent.review.evidence import collect_source_evidence
from transcription_review_agent.review.first_pass_contract import (
    first_pass_instruction_note,
    first_pass_schema_text,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewLimitError,
    ReviewOutputRejected,
    ReviewPathError,
    ReviewPolicyError,
    ReviewResultMissing,
    ReviewUsageError,
)
from transcription_review_agent.review.models import (
    BBox,
    ChallengeClosureRecord,
    ChallengeOutput,
    ChallengeSelectionRecord,
    ComparisonFile,
    CropRecord,
    Dispute,
    DisputeCategory,
    EvidenceKind,
    EvidenceReference,
    FirstPassOutput,
    Identification,
    ManifestArtifact,
    ManifestProvider,
    NEW_EVIDENCE_KINDS,
    PageTranscription,
    ReconciledDecision,
    ResearchOutput,
    ReviewManifest,
    SourceEvidence,
    SourceManifest,
    SourcePageRecord,
    TextRegion,
    WorkRequest,
    interface_for,
)
from transcription_review_agent.review.paths import (
    archive_document_dir,
    assert_allowlisted_output_dir,
    contained,
    document_root,
    output_filename,
    output_relative_path,
    reviews_root,
    stage_dir_name,
)
from transcription_review_agent.review.runtime import (
    ChallengeClosedEntry,
    ChallengeContract,
    ChallengeReconciledEntry,
    ChallengeRoundState,
    ComparisonRecordedEntry,
    DocumentOpenedEntry,
    DocumentReviewState,
    PageAsset,
    PageCoverageSnapshot,
    ProviderFailureEntry,
    ProviderRuntime,
    ProviderStatus,
    ResultAcceptedEntry,
    ResultRejectedEntry,
    ReviewBatchState,
    ReviewFinalizedEntry,
    ReviewProvisionalEntry,
    ReviewStore,
    WorkPublishedEntry,
    challenge_closure_problem,
    challenge_round_failed,
    describe_unresolved,
    find_document,
    normalize_challenge_selection,
    provider_runtime,
    resolved_dispute_ids,
    round_selection,
)

Clock = Callable[[], datetime]
CLOSURE_MODES: tuple[str, ...] = (
    "NO_CHALLENGE_REQUESTED",
    "SELECTED_ROUNDS_COMPLETE",
    "NO_NEW_INFORMATION",
    "ROUND_CAP",
    "PROVIDER_TERMINAL_FAILURE",
)
FINAL_NAMES = (
    "proposed_transcription.txt",
    "review_report.json",
    "review_manifest.json",
    "review_summary.md",
)
StageName = Literal["first_pass", "challenge", "research"]
ProviderFile = FirstPassOutput | ChallengeOutput | ResearchOutput


@dataclass(frozen=True, slots=True)
class DocumentStatus:
    document_id: int
    execution_state: str
    quality: str | None
    providers: dict[str, str]
    page_ids: tuple[str, ...]
    finalized: bool
    provider_failure_count: int
    estimated_work_units: int
    visual_readings_preserved: bool
    research_changes_diplomatic_text: bool
    challenge_rounds: int
    substantive_dispute_ids: tuple[str, ...]
    ledger_sequence: int
    challenge_closure: str | None = None


CHALLENGE_INPUT_V1: Final = "review-challenge-input-v1"
CHALLENGE_INPUT_V2: Final = "review-challenge-input-v2"
CHALLENGE_INPUT_V3: Final = "review-challenge-input-v3"
EVIDENCE_BOUND_CONTRACT: ChallengeContract = "EVIDENCE_BOUND_V1"
ChallengeLayout = Literal["UNGROUPED", "GROUPED"]


class AnonymousChallengeItem(BaseModel):
    """A disputed span with anonymous readings. Provider names are omitted.

    ``group_id`` is present only in the grouped layout (A2). Every item keeps
    its own dispute ID and needs its own decision, grouped or not.
    """

    model_config = ConfigDict(extra="forbid")

    dispute_id: str
    page_id: str
    crop_id: str
    group_id: str | None = None
    candidate_x: str
    candidate_y: str
    request_visual_evidence: Literal[True] = True

    @model_serializer(mode="wrap")
    def _omit_absent_group(self, handler):  # type: ignore[no-untyped-def]
        data = handler(self)
        if self.group_id is None:
            data.pop("group_id", None)
        return data


class AnonymousChallengeGroup(BaseModel):
    """Selected disputes that share one visual evidence window.

    Provider names never appear here. ``evidence_refs`` lists the references
    a decision on a member can cite: the group itself and its crop (or the
    labeled full-page fallback window).
    """

    model_config = ConfigDict(extra="forbid")

    group_id: str = Field(pattern=r"^grp_[0-9a-f]{16}$")
    page_id: str
    page_sha256: str
    basis: Literal["ALIGNED_SPAN", "PAGE_READING_ORDER", "UNLINKED"]
    dispute_ids: list[str] = Field(min_length=1)
    crop_id: str
    crop_sha256: str
    window_kind: Literal["TARGETED", "FULL_PAGE_FALLBACK"]
    evidence_refs: list[EvidenceReference] = Field(min_length=2, max_length=2)
    decisions_independent: Literal[True] = True


class AnonymousChallengeFile(BaseModel):
    """Immutable challenge input.

    V1 is historical ungrouped input, V2 is historical grouped input, and V3
    is grouped input with the evidence-bound A3 round contract.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "review-challenge-input-v1",
        "review-challenge-input-v2",
        "review-challenge-input-v3",
    ] = CHALLENGE_INPUT_V1
    disputes: list[AnonymousChallengeItem]
    groups: list[AnonymousChallengeGroup] | None = None
    required_decision_ids: list[str] | None = None
    challenge_contract: ChallengeContract | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_groups(self, handler):  # type: ignore[no-untyped-def]
        data = handler(self)
        for name in ("groups", "required_decision_ids", "challenge_contract"):
            if data.get(name) is None:
                data.pop(name, None)
        return data

    @model_validator(mode="after")
    def _consistent_layout(self) -> AnonymousChallengeFile:
        item_ids = [item.dispute_id for item in self.disputes]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("a dispute is listed twice")
        if self.schema_version == CHALLENGE_INPUT_V1:
            if (
                self.groups is not None
                or self.required_decision_ids is not None
                or self.challenge_contract is not None
                or any(item.group_id is not None for item in self.disputes)
            ):
                raise ValueError("an ungrouped challenge input carries no groups")
            return self
        if (
            self.schema_version == CHALLENGE_INPUT_V2
            and self.challenge_contract is not None
        ):
            raise ValueError(
                "a historical grouped input carries no evidence-bound contract"
            )
        if (
            self.schema_version == CHALLENGE_INPUT_V3
            and self.challenge_contract != EVIDENCE_BOUND_CONTRACT
        ):
            raise ValueError("an evidence-bound input must name its round contract")
        if self.groups is None or self.required_decision_ids is None:
            raise ValueError(
                "a grouped challenge input lists its groups and required decisions"
            )
        if self.required_decision_ids != item_ids:
            raise ValueError("required decisions must equal the listed disputes")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("a group is listed twice")
        grouped = [
            dispute_id for group in self.groups for dispute_id in group.dispute_ids
        ]
        if len(grouped) != len(set(grouped)) or set(grouped) != set(item_ids):
            raise ValueError("groups must cover each listed dispute exactly once")
        by_id = {item.dispute_id: item for item in self.disputes}
        for group in self.groups:
            if group.window_kind == "TARGETED":
                evidence_kind: EvidenceKind = "CROP"
            else:
                evidence_kind = "FULL_PAGE_FALLBACK"
            expected_refs = [
                EvidenceReference(
                    kind="GROUP", evidence_id=group.group_id, page_id=group.page_id
                ),
                EvidenceReference(
                    kind=evidence_kind,
                    evidence_id=group.crop_id,
                    page_id=group.page_id,
                ),
            ]
            if group.evidence_refs != expected_refs:
                raise ValueError("a group must list its exact group and crop evidence")
            for dispute_id in group.dispute_ids:
                item = by_id[dispute_id]
                if (
                    item.group_id != group.group_id
                    or item.crop_id != group.crop_id
                    or item.page_id != group.page_id
                ):
                    raise ValueError("a listed dispute disagrees with its group")
        return self


class ReviewController:
    """Prepare tasks and import provider files for sealed documents."""

    def __init__(
        self,
        *,
        batches_dir: Path,
        reviews_dir: Path,
        config: ReviewConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.batches_dir = reviews_root(batches_dir)
        self.reviews_dir = reviews_root(reviews_dir)
        self.config = load_review_config({}) if config is None else config
        self.clock = clock if clock is not None else _utc_now

    def open_sealed_document(self, batch_id: str, document_id: int) -> DocumentStatus:
        """Verify a sealed bundle, then copy page images into the review tree."""

        manifest, document_hash, batch_hash, transcription = self._sealed(
            batch_id, document_id
        )
        state = self._store(batch_id).load()
        if state is not None and _has_document(state, document_id):
            document = find_document(state, document_id)
            if document.source_bundle_sha256 != document_hash:
                raise ReviewIntegrityError("sealed document hash changed after intake")
            return self._status(state, document)
        if state is not None and len(state.documents) >= self.config.max_documents:
            raise ReviewLimitError("document limit for this review batch is reached")
        archive_dir = archive_document_dir(self.batches_dir, batch_id, document_id)
        pages, derived = _page_assets(manifest, archive_dir)
        if len(pages) > self.config.max_pages:
            raise ReviewLimitError("page limit for this review is reached")
        doc_root = document_root(self.reviews_dir, batch_id, document_id)
        _write_intake(doc_root, archive_dir, manifest, pages, derived, document_hash)
        _reject_transcription_leak(doc_root / "input", transcription)
        store = self._store(batch_id)
        current = store.load()
        sequence = 1 if current is None else current.applied_ledger_sequence + 1
        opened = store.commit(
            DocumentOpenedEntry(
                sequence=sequence,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                language=manifest.language,
                source_bundle_sha256=document_hash,
                source_batch_manifest_sha256=batch_hash,
                archive_transcription_sha256=manifest.transcription.sha256,
                pages=pages,
                estimated_work_units=len(pages) * 2,
            )
        )
        return self._status(opened, find_document(opened, document_id))

    def publish(
        self,
        batch_id: str,
        document_id: int,
        provider: str,
        *,
        stage: str = "first_pass",
        evidence: str | None = None,
        evidence_path: Path | None = None,
        selected_dispute_ids: Sequence[str] | None = None,
    ) -> Path:
        """Write a self-contained task bundle and a manual work request.

        ``selected_dispute_ids`` (challenge stage only) freezes an explicit
        subset of substantive disputes for the round at its first provider
        publication. Later publications of the same round reuse it; a different
        subset is refused. Without it the round covers the full substantive set.
        Every newly opened round is evidence-bound. Only a historical ledger
        round whose publication predates the contract keeps legacy behavior.
        """

        provider = _provider_name(provider)
        stage_name = _stage_name(stage)
        if stage_name != "challenge" and selected_dispute_ids is not None:
            raise ReviewPolicyError(
                "a dispute selection belongs only to a challenge round"
            )
        _state, document = self._require_document(batch_id, document_id)
        self._require_unchanged_source(batch_id, document)
        doc_root = document_root(self.reviews_dir, batch_id, document_id)
        if stage_name == "research":
            _require_research_allowed(document, doc_root)
        if stage_name == "challenge":
            _require_challenge_allowed(document)
        round_index = _round_for_publish(
            document,
            stage_name,
            provider,
            evidence,
            max_rounds=self.config.max_challenge_rounds,
        )
        existing_round = (
            next(
                (
                    item
                    for item in document.challenge_rounds
                    if item.round_index == round_index
                ),
                None,
            )
            if stage_name == "challenge"
            else None
        )
        challenge_contract: ChallengeContract | None = None
        if stage_name == "challenge":
            challenge_contract = (
                EVIDENCE_BOUND_CONTRACT
                if existing_round is None
                else existing_round.challenge_contract
            )
        comparison = (
            _load_comparison(doc_root, document) if stage_name == "challenge" else None
        )
        ledger_selection: list[str] | None = None
        selection_hash: str | None = None
        task_selection: list[str] | None = None
        if comparison is not None:
            ledger_selection, selection_hash, task_selection = _resolve_selection(
                document,
                comparison,
                existing_round,
                selected_dispute_ids,
                round_index=round_index,
            )
        evidence_sha256 = None
        evidence_relative = None
        if stage_name == "challenge":
            if existing_round is not None and evidence not in {
                None,
                existing_round.evidence,
            }:
                raise ReviewPolicyError("an open challenge round keeps its evidence")
            if existing_round is None and round_index > 1:
                evidence_sha256, evidence_relative = _validated_evidence(
                    document,
                    doc_root,
                    round_index,
                    evidence,
                    evidence_path,
                )
            if existing_round is not None:
                _published_challenge_scopes(
                    doc_root,
                    document,
                    round_index,
                    expected_contract=challenge_contract,
                )
        runtime = _runtime_or_new(document, stage_name, provider, round_index)
        if runtime.status is ProviderStatus.SUCCEEDED:
            raise ReviewPolicyError(
                "refusing to rerun a provider that already succeeded"
            )
        if runtime.status is ProviderStatus.QUOTA_BLOCKED:
            raise ReviewPolicyError("quota-blocked jobs are not retried")
        if runtime.status in {ProviderStatus.FAILED, ProviderStatus.INVALID_OUTPUT}:
            raise ReviewPolicyError("this provider attempt cannot be dispatched again")
        if runtime.status is ProviderStatus.AWAITING_MANUAL_DISPATCH:
            return _request_path(doc_root, stage_name, provider, round_index)
        if runtime.transient_failures > 1:
            raise ReviewPolicyError("the transient retry was already used")
        attempt = (
            1 if runtime.status is ProviderStatus.NOT_STARTED else runtime.attempt + 1
        )
        ledger_evidence = (
            None
            if stage_name != "challenge"
            else (existing_round.evidence if existing_round is not None else evidence)
        )
        if existing_round is not None:
            evidence_sha256 = existing_round.evidence_sha256
            evidence_relative = existing_round.evidence_relative_path
        request = _work_request(
            batch_id,
            document,
            provider=provider,
            stage=stage_name,
            round_index=round_index,
            attempt=attempt,
        )
        transcription = _archive_text(self.batches_dir, batch_id, document_id)
        gpt = (
            _load_output(doc_root, document, "gpt")
            if stage_name == "challenge"
            else None
        )
        grok = (
            _load_output(doc_root, document, "grok")
            if stage_name == "challenge"
            else None
        )
        if stage_name == "challenge" and existing_round is not None:
            _published_challenge_scopes(
                doc_root,
                document,
                round_index,
                expected_contract=challenge_contract,
            )
        _prepare_bundle(
            doc_root,
            document,
            request,
            transcription,
            comparison=comparison,
            gpt=gpt,
            grok=grok,
            challenge_selection=task_selection,
            challenge_contract=challenge_contract,
            challenge_already_published=existing_round is not None,
        )
        store = self._store(batch_id)
        current = store.load()
        assert current is not None
        store.commit(
            WorkPublishedEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                provider=provider,  # type: ignore[arg-type]
                stage=stage_name,  # type: ignore[arg-type]
                round_index=round_index,
                attempt=attempt,
                task_id=request.task_id,
                job_id=request.job_id,
                output_relative_path=request.output_relative_path,
                evidence=ledger_evidence,
                evidence_sha256=evidence_sha256,
                evidence_relative_path=evidence_relative,
                selected_dispute_ids=ledger_selection,
                selection_comparison_sha256=selection_hash,
                challenge_contract=challenge_contract,
            )
        )
        return _request_path(doc_root, stage_name, provider, round_index)

    def accept(
        self,
        batch_id: str,
        document_id: int,
        provider: str,
        *,
        stage: str = "first_pass",
        round_index: int | None = None,
        source_json: Path | None = None,
    ) -> DocumentStatus:
        """Validate one allowlisted provider file and record it once."""

        provider = _provider_name(provider)
        stage_name = _stage_name(stage)
        state, document = self._require_document(batch_id, document_id)
        self._require_unchanged_source(batch_id, document)
        if round_index is None:
            round_index = (
                document.challenge_rounds[-1].round_index
                if stage_name == "challenge"
                else 0
            )
        runtime = _runtime_or_new(document, stage_name, provider, round_index)
        doc_root = document_root(self.reviews_dir, batch_id, document_id)
        relative = runtime.output_relative_path or output_relative_path(
            stage_name, provider, round_index
        )
        output = contained(doc_root, doc_root / relative)
        if runtime.status is ProviderStatus.SUCCEEDED:
            try:
                _require_matching_success(runtime, output)
            except ReviewOutputRejected as exc:
                self._reject(
                    state,
                    document,
                    provider,
                    stage_name,
                    round_index,
                    exc.reason,
                    preserve=True,
                )
                raise
            if stage_name == "challenge":
                # Idempotent: a no-op unless an earlier crash left both results
                # accepted but the round unreconciled.
                self._reconcile_if_ready(batch_id, document_id, round_index)
            return self.status(batch_id, document_id)
        if (
            runtime.status is not ProviderStatus.AWAITING_MANUAL_DISPATCH
            or runtime.task_id is None
        ):
            raise ReviewPolicyError("provider work has not been published")
        try:
            assert_allowlisted_output_dir(output.parent)
        except ReviewPathError as exc:
            self._reject(
                state,
                document,
                provider,
                stage_name,
                round_index,
                "duplicated",
                preserve=False,
            )
            raise ReviewOutputRejected("duplicated", str(exc)) from exc
        if not output.is_file():
            raise ReviewResultMissing(f"missing provider output {output.name}")
        try:
            model = _parse_provider_file(
                output,
                batch_id=batch_id,
                provider=provider,
                stage=stage_name,
                document=document,
                runtime=runtime,
                round_index=round_index,
                doc_root=doc_root,
            )
        except ReviewOutputRejected as exc:
            self._reject(
                state,
                document,
                provider,
                stage_name,
                round_index,
                exc.reason,
                preserve=False,
            )
            raise
        coverage: list[PageCoverageSnapshot] = []
        decisions = []
        source_evidence: list[SourceEvidence] = []
        if isinstance(model, FirstPassOutput):
            coverage = [
                PageCoverageSnapshot(
                    page_id=page.page_id,
                    coverage_complete=page.coverage.coverage_complete,
                    unresolved_text_present=page.coverage.unresolved_text_present,
                )
                for page in model.pages
            ]
            if source_json is not None:
                source_evidence = _read_source_evidence(
                    source_json, output, provider, model
                )
        elif source_json is not None:
            raise ReviewUsageError("source evidence is recorded for a first-pass file")
        elif isinstance(model, ChallengeOutput):
            decisions = list(model.decisions)
        store = self._store(batch_id)
        current = store.load()
        assert current is not None
        updated = store.commit(
            ResultAcceptedEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                provider=provider,  # type: ignore[arg-type]
                stage=stage_name,  # type: ignore[arg-type]
                round_index=round_index,
                output_sha256=sha256_bytes(output.read_bytes()),
                output_relative_path=relative,
                coverage=coverage,
                decisions=decisions,
                source_evidence=source_evidence,
            )
        )
        if stage_name == "challenge":
            self._reconcile_if_ready(batch_id, document_id, round_index)
            reloaded = self._store(batch_id).load()
            assert reloaded is not None
            updated = reloaded
        return self._status(updated, find_document(updated, document_id))

    def record_failure(
        self,
        batch_id: str,
        document_id: int,
        provider: str,
        *,
        kind: str,
        stage: str = "first_pass",
    ) -> DocumentStatus:
        """Record a transient failure or a quota block. Quota is not retried."""

        if kind not in {"transient", "quota"}:
            raise ReviewPolicyError("failure kind must be transient or quota")
        provider = _provider_name(provider)
        stage_name = _stage_name(stage)
        state, document = self._require_document(batch_id, document_id)
        round_index = (
            document.challenge_rounds[-1].round_index
            if stage_name == "challenge"
            else 0
        )
        runtime = _runtime_or_new(document, stage_name, provider, round_index)
        if kind == "quota" and runtime.status is ProviderStatus.QUOTA_BLOCKED:
            return self._status(state, document)
        if runtime.status not in {
            ProviderStatus.AWAITING_MANUAL_DISPATCH,
            ProviderStatus.RUNNING,
        }:
            raise ReviewPolicyError("failure requires a dispatched attempt")
        store = self._store(batch_id)
        current = store.load()
        assert current is not None
        updated = store.commit(
            ProviderFailureEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                provider=provider,  # type: ignore[arg-type]
                stage=stage_name,  # type: ignore[arg-type]
                round_index=round_index,
                kind=kind,  # type: ignore[arg-type]
            )
        )
        return self._status(updated, find_document(updated, document_id))

    def compare(self, batch_id: str, document_id: int) -> ComparisonFile:
        """Align provider text. The controller does not pick a confident winner."""

        _state, document = self._require_document(batch_id, document_id)
        doc_root = document_root(self.reviews_dir, batch_id, document_id)
        gpt = _load_output(doc_root, document, "gpt")
        grok = _load_output(doc_root, document, "grok")
        if gpt is None or grok is None:
            raise ReviewPolicyError("comparison requires two provider results")
        comparison = compare_transcriptions(
            gpt,
            grok,
            gpt_sha256=_provider_hash(document, "gpt"),
            grok_sha256=_provider_hash(document, "grok"),
        )
        payload = canonical_json(comparison).encode("utf-8")
        path = doc_root / "comparison" / "comparison.json"
        _write_if_same(path, payload)
        digest = sha256_bytes(path.read_bytes())
        if document.comparison_sha256 == digest:
            return comparison
        store = self._store(batch_id)
        current = store.load()
        assert current is not None
        store.commit(
            ComparisonRecordedEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                comparison_sha256=digest,
                dispute_ids=[item.dispute_id for item in comparison.disputes],
                substantive_dispute_ids=[
                    item.dispute_id for item in challengeable_disputes(comparison)
                ],
            )
        )
        return comparison

    def finalize(
        self,
        batch_id: str,
        document_id: int,
        *,
        provisional: bool = False,
    ) -> dict[str, Path]:
        """Publish the four document outputs without touching the archive bundle."""

        state, document = self._require_document(batch_id, document_id)
        self._require_unchanged_source(batch_id, document)
        doc_root = document_root(self.reviews_dir, batch_id, document_id)
        if document.finalized:
            return _final_paths(doc_root)
        pending = _pending_opened_stage(document)
        if pending is not None:
            raise ReviewPolicyError(
                f"the opened {pending} stage is still pending and cannot be finalized"
            )
        if _dispatch_still_open(document):
            raise ReviewPolicyError("a provider is still awaiting manual dispatch")
        succeeded = [
            item.provider
            for item in document.first_pass
            if item.status is ProviderStatus.SUCCEEDED
        ]
        if not succeeded:
            raise ReviewPolicyError("neither provider produced a valid transcription")
        if len(succeeded) == 2 and document.comparison_sha256 is None:
            self.compare(batch_id, document_id)
            state, document = self._require_document(batch_id, document_id)
        gpt = _load_output(doc_root, document, "gpt") if "gpt" in succeeded else None
        grok = _load_output(doc_root, document, "grok") if "grok" in succeeded else None
        comparison = (
            _load_comparison(doc_root, document) if len(succeeded) == 2 else None
        )
        # A4: a completed proposal with substantive disputes needs an explicit
        # closure of the challenge stage. A reconciled round, or silence, is not one.
        unfinished_challenge = (
            bool(document.substantive_dispute_ids)
            and document.challenge_closure is None
        )
        if unfinished_challenge and not provisional:
            raise ReviewPolicyError(
                "substantive disputes need an explicitly closed challenge stage "
                "before a completed review"
            )
        identifications = _research_identifications(doc_root, document)
        text = render_transcription(
            page_ids=[page.page_id for page in document.pages],
            gpt=gpt,
            grok=grok,
            comparison=comparison,
            decisions=list(document.reconciled_decisions),
        )
        single = len(succeeded) == 1
        interim = single or provisional or unfinished_challenge
        completion = "interim_provisional" if interim else "completed"
        quality = classify_quality(
            single_provider=single,
            gpt=gpt,
            grok=grok,
            comparison=comparison,
            decisions=list(document.reconciled_decisions),
            challenge_closure=document.challenge_closure,
            interim=interim and not single,
        )
        report = build_report(
            batch_id=batch_id,
            document_id=document_id,
            quality=quality,
            page_ids=[page.page_id for page in document.pages],
            single_provider=single,
            succeeded_provider=succeeded[0] if single else None,
            comparison=comparison,
            decisions=list(document.reconciled_decisions),
            identifications=identifications,
            review_completion=completion,
            provider_failure_count=document.provider_failure_count,
            estimated_work_units=document.estimated_work_units,
            gpt=gpt,
            grok=grok,
            source_evidence=[
                item
                for runtime in document.first_pass
                for item in runtime.source_evidence
            ],
            challenge_selections=[
                ChallengeSelectionRecord(
                    round_index=item.round_index,
                    selected_dispute_ids=list(item.selected_dispute_ids),
                    comparison_sha256=item.selection_comparison_sha256,
                )
                for item in document.challenge_rounds
                if item.selected_dispute_ids is not None
                and item.selection_comparison_sha256 is not None
            ],
            challenge_closure=document.challenge_closure,
        )
        summary = hebrew_summary(
            batch_id=batch_id,
            document_id=document_id,
            language=document.language,
            page_count=len(document.pages),
            quality=quality,
            substantive_count=len(document.substantive_dispute_ids),
            coverage_gaps=coverage_gap_ids(gpt, grok),
            single_provider=single,
        )
        text_bytes = text.encode("utf-8")
        report_bytes = canonical_json(report).encode("utf-8")
        summary_bytes = summary.encode("utf-8")
        manifest = _manifest(
            state,
            document,
            text_sha=sha256_bytes(text_bytes),
            report_sha=sha256_bytes(report_bytes),
            summary_sha=sha256_bytes(summary_bytes),
        )
        payloads = {
            "proposed_transcription.txt": text_bytes,
            "review_report.json": report_bytes,
            "review_manifest.json": canonical_json(manifest).encode("utf-8"),
            "review_summary.md": summary_bytes,
        }
        hashes = {name: sha256_bytes(payload) for name, payload in payloads.items()}
        if document.report_versions:
            previous = document.report_versions[-1]
            if (
                previous.completion == completion
                and previous.quality == quality
                and previous.artifact_sha256 == hashes
            ):
                return _version_paths(doc_root, previous.relative_dir)
        _require_version_history(doc_root, document)
        store = self._store(batch_id)
        current = store.load()
        assert current is not None
        fresh = find_document(current, document_id)
        if fresh.finalized:
            return _final_paths(doc_root)
        if (
            fresh.report_versions
            and fresh.report_versions[-1].artifact_sha256 == hashes
        ):
            return _version_paths(doc_root, fresh.report_versions[-1].relative_dir)
        entry_version = len(fresh.report_versions) + 1
        entry_dir = _version_dir(entry_version)
        for name, payload in payloads.items():
            _write_if_same(doc_root / entry_dir / name, payload)
        if completion == "completed":
            for name, payload in payloads.items():
                _write_if_same(doc_root / "final" / name, payload)
        if completion == "completed":
            store.commit(
                ReviewFinalizedEntry(
                    sequence=current.applied_ledger_sequence + 1,
                    occurred_at=self.clock(),
                    batch_id=batch_id,
                    document_id=document_id,
                    quality=quality,
                    final_sha256=hashes,
                    version=entry_version,
                    relative_dir=entry_dir,
                )
            )
            return _final_paths(doc_root)
        store.commit(
            ReviewProvisionalEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                quality=quality,
                final_sha256=hashes,
                version=entry_version,
                relative_dir=entry_dir,
            )
        )
        return _version_paths(doc_root, entry_dir)

    def close_challenge(
        self,
        batch_id: str,
        document_id: int,
        *,
        mode: str,
        reason: str,
    ) -> DocumentStatus:
        """Explicitly close the challenge stage. Never inferred, never retried silently.

        ``mode`` says why the stage ends: ``NO_CHALLENGE_REQUESTED`` (zero
        rounds, after comparison), ``SELECTED_ROUNDS_COMPLETE``,
        ``NO_NEW_INFORMATION``, ``ROUND_CAP`` (the configured cap is reached) or
        ``PROVIDER_TERMINAL_FAILURE`` (a round ended without a result and has no
        live dispatch). ``reason`` is the operator's explicit, auditable text.

        A round with an open dispatch cannot be closed. Closing never decides a
        dispute: every substantive dispute without a resolved decision stays
        listed as unresolved and keeps both alternatives.
        """

        if mode not in CLOSURE_MODES:
            raise ReviewPolicyError("unknown challenge closure mode")
        if not isinstance(reason, str) or not reason.strip():
            raise ReviewPolicyError(
                "closing the challenge stage needs an explicit reason"
            )
        if len(reason) > 2000:
            raise ReviewPolicyError(
                "the closure reason is too long (at most 2000 characters)"
            )
        state, document = self._require_document(batch_id, document_id)
        self._require_unchanged_source(batch_id, document)
        existing = document.challenge_closure
        if existing is not None:
            if existing.mode == mode and existing.reason == reason:
                return self._status(state, document)
            raise ReviewPolicyError("the challenge stage is already closed")
        if document.finalized:
            raise ReviewPolicyError("a finalized review cannot be closed")
        doc_root = document_root(self.reviews_dir, batch_id, document_id)
        _load_comparison(doc_root, document)  # tamper check; raises on a changed file
        if document.comparison_sha256 is None:
            raise ReviewPolicyError(
                "comparison must be recorded before the challenge stage is closed"
            )
        failed = [
            item.round_index
            for item in document.challenge_rounds
            if challenge_round_failed(item)
        ]
        record = ChallengeClosureRecord(
            mode=mode,  # type: ignore[arg-type]
            reason=reason,
            comparison_sha256=document.comparison_sha256,
            round_count=len(document.challenge_rounds),
            failed_round_indexes=failed if mode == "PROVIDER_TERMINAL_FAILURE" else [],
            max_challenge_rounds=self.config.max_challenge_rounds,
            unresolved=describe_unresolved(
                document, failed if mode == "PROVIDER_TERMINAL_FAILURE" else []
            ),
        )
        problem = challenge_closure_problem(document, record)
        if problem is not None:
            raise ReviewPolicyError(problem)
        store = self._store(batch_id)
        current = store.load()
        assert current is not None
        updated = store.commit(
            ChallengeClosedEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                closure=record,
            )
        )
        return self._status(updated, find_document(updated, document_id))

    def status(self, batch_id: str, document_id: int) -> DocumentStatus:
        state, document = self._require_document(batch_id, document_id)
        return self._status(state, document)

    def _reconcile_if_ready(
        self, batch_id: str, document_id: int, round_index: int
    ) -> None:
        _state, document = self._require_document(batch_id, document_id)
        challenge = next(
            (
                item
                for item in document.challenge_rounds
                if item.round_index == round_index
            ),
            None,
        )
        if challenge is None or challenge.reconciled:
            return
        if any(
            item.status is not ProviderStatus.SUCCEEDED for item in challenge.providers
        ):
            return
        by_provider = {
            item.provider: item.challenge_decisions for item in challenge.providers
        }
        doc_root = document_root(self.reviews_dir, batch_id, document_id)
        _published, scopes = _published_challenge_scopes(
            doc_root,
            document,
            round_index,
            expected_contract=challenge.challenge_contract,
        )
        comparison = _load_comparison(doc_root, document)
        if comparison is None:
            raise ReviewIntegrityError("challenge reconciliation has no comparison")
        disputes = {item.dispute_id: item for item in comparison.disputes}
        reconciled: list[ReconciledDecision] = []
        for dispute_id in round_selection(document, round_index):
            left = next(
                item for item in by_provider["gpt"] if item.dispute_id == dispute_id
            )
            right = next(
                item for item in by_provider["grok"] if item.dispute_id == dispute_id
            )
            if challenge.challenge_contract == EVIDENCE_BOUND_CONTRACT:
                dispute = disputes.get(dispute_id)
                if dispute is None:
                    raise ReviewIntegrityError(
                        "challenged dispute is absent from the recorded comparison"
                    )
                try:
                    reconciled.append(
                        reconcile_challenge(dispute, left, right, scopes[dispute_id])
                    )
                except EvidenceBindingError as exc:
                    raise ReviewIntegrityError(
                        "accepted evidence-bound decision failed reconciliation binding"
                    ) from exc
                continue
            agreed = (
                left.decision == right.decision
                and left.third_reading == right.third_reading
            )
            evidence_differs = (
                left.visual_evidence.strip() != right.visual_evidence.strip()
            )
            reconciled.append(
                ReconciledDecision(
                    dispute_id=dispute_id,
                    decision=left.decision if agreed else "UNRESOLVED",
                    third_reading=left.third_reading if agreed else None,
                    agreed=agreed,
                    gpt_decision=left.decision,
                    grok_decision=right.decision,
                    gpt_visual_evidence=left.visual_evidence,
                    grok_visual_evidence=right.visual_evidence,
                    remaining_uncertainty=(
                        (not agreed)
                        or evidence_differs
                        or left.decision == "UNRESOLVED"
                    ),
                )
            )
        store = self._store(batch_id)
        current = store.load()
        assert current is not None
        store.commit(
            ChallengeReconciledEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=batch_id,
                document_id=document_id,
                round_index=round_index,
                decisions=reconciled,
                accumulates_history=challenge.selected_dispute_ids is not None,
            )
        )

    def _reject(
        self,
        state: ReviewBatchState,
        document: DocumentReviewState,
        provider: str,
        stage: str,
        round_index: int,
        reason: str,
        *,
        preserve: bool,
    ) -> None:
        if reason not in {
            "invalid_label",
            "schema",
            "stale",
            "mismatched",
            "duplicated",
        }:
            reason = "schema"
        store = self._store(state.batch_id)
        current = store.load()
        assert current is not None
        store.commit(
            ResultRejectedEntry(
                sequence=current.applied_ledger_sequence + 1,
                occurred_at=self.clock(),
                batch_id=state.batch_id,
                document_id=document.document_id,
                provider=provider,  # type: ignore[arg-type]
                stage=stage,  # type: ignore[arg-type]
                round_index=round_index,
                reason=reason,  # type: ignore[arg-type]
                preserve_success=preserve,
            )
        )

    def _require_document(
        self,
        batch_id: str,
        document_id: int,
    ) -> tuple[ReviewBatchState, DocumentReviewState]:
        state = self._store(batch_id).load()
        if state is None:
            raise ReviewPolicyError("no review batch is open")
        return state, find_document(state, document_id)

    def _require_unchanged_source(
        self, batch_id: str, document: DocumentReviewState
    ) -> None:
        document_dir = archive_document_dir(
            self.batches_dir, batch_id, document.document_id
        )
        verify_document_bundle(document_dir)
        digest = sha256_bytes((document_dir / "document_manifest.json").read_bytes())
        if digest != document.source_bundle_sha256:
            raise ReviewIntegrityError("source bundle hash changed before processing")
        doc_root = document_root(self.reviews_dir, batch_id, document.document_id)
        for page in document.pages:
            data = (doc_root / "input" / _original_name(page)).read_bytes()
            if sha256_bytes(data) != page.sha256:
                raise ReviewIntegrityError("review page bytes changed after intake")
            if page.derived_png_sha256 is not None:
                derived = (
                    doc_root / "input" / f"{page.page_id}.derived.png"
                ).read_bytes()
                if sha256_bytes(derived) != page.derived_png_sha256:
                    raise ReviewIntegrityError("derived page PNG changed after intake")

    def _sealed(
        self,
        batch_id: str,
        document_id: int,
    ) -> tuple[DocumentBundleManifest, str, str, str]:
        batch_dir = self.batches_dir / batch_id
        verify_batch_bundle(batch_dir)
        document_dir = archive_document_dir(self.batches_dir, batch_id, document_id)
        manifest = verify_document_bundle(document_dir)
        if manifest.document_id != document_id or manifest.batch_id != batch_id:
            raise ReviewIntegrityError(
                "sealed manifest does not match the requested document"
            )
        document_hash = sha256_bytes(
            (document_dir / "document_manifest.json").read_bytes()
        )
        batch_hash = sha256_bytes((batch_dir / "batch_manifest.json").read_bytes())
        transcription = (
            document_dir / "transcription" / "transcription.txt"
        ).read_text(encoding="utf-8")
        return manifest, document_hash, batch_hash, transcription

    def _store(self, batch_id: str) -> ReviewStore:
        return ReviewStore(self.reviews_dir, batch_id)

    def _status(
        self, state: ReviewBatchState, document: DocumentReviewState
    ) -> DocumentStatus:
        return DocumentStatus(
            document_id=document.document_id,
            execution_state=document.execution_state.value,
            quality=None if document.quality is None else document.quality.value,
            providers={
                item.provider: item.status.value for item in document.first_pass
            },
            page_ids=tuple(page.page_id for page in document.pages),
            finalized=document.finalized,
            provider_failure_count=document.provider_failure_count,
            estimated_work_units=document.estimated_work_units,
            visual_readings_preserved=document.visual_readings_preserved,
            research_changes_diplomatic_text=document.research_changes_diplomatic_text,
            challenge_rounds=len(document.challenge_rounds),
            substantive_dispute_ids=tuple(document.substantive_dispute_ids),
            ledger_sequence=state.applied_ledger_sequence,
            challenge_closure=(
                None
                if document.challenge_closure is None
                else document.challenge_closure.mode
            ),
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _provider_name(provider: str) -> str:
    if provider not in {"gpt", "grok"}:
        raise ReviewPathError("provider must be gpt or grok")
    return provider


def _stage_name(stage: str) -> StageName:
    if stage not in {"first_pass", "challenge", "research"}:
        raise ReviewPathError("unknown review stage")
    return stage  # type: ignore[return-value]


def _has_document(state: ReviewBatchState, document_id: int) -> bool:
    return any(document.document_id == document_id for document in state.documents)


def _page_assets(
    manifest: DocumentBundleManifest,
    document_dir: Path,
) -> tuple[list[PageAsset], dict[str, bytes]]:
    images = [
        source
        for source in sorted(manifest.sources, key=lambda item: item.order_index)
        if source.mime_type in {"image/png", "image/jpeg"}
    ]
    if not images:
        raise ReviewPolicyError("v1 review requires sealed PNG or JPEG page images")
    pages: list[PageAsset] = []
    derived: dict[str, bytes] = {}
    for index, source in enumerate(images, start=1):
        data = (document_dir / source.relative_path).read_bytes()
        if sha256_bytes(data) != source.sha256:
            raise ReviewIntegrityError("source bytes do not match the sealed manifest")
        page_id = f"page_{index:03d}"
        if source.mime_type == "image/png":
            width, height = png_dimensions(data)
            filename = f"{page_id}.png"
            derived_sha = None
        elif source.mime_type == "image/jpeg":
            try:
                width, height = jpeg_dimensions(data)
                decoded_width, decoded_height, rows, bpp = decode_jpeg(data)
                derived_png = encode_png(decoded_width, decoded_height, rows, bpp)
                derived_width, derived_height = png_dimensions(derived_png)
                repeated = encode_png(decoded_width, decoded_height, rows, bpp)
            except ValueError as exc:
                raise ReviewPolicyError(str(exc)) from exc
            if (derived_width, derived_height) != (width, height):
                raise ReviewPolicyError("derived PNG dimensions do not match the JPEG")
            if repeated != derived_png:
                raise ReviewPolicyError("derived PNG is not deterministic")
            filename = f"{page_id}.jpg"
            derived[page_id] = derived_png
            derived_sha = sha256_bytes(derived_png)
        else:
            raise ReviewPolicyError("v1 review requires sealed PNG or JPEG page images")
        pages.append(
            PageAsset(
                page_id=page_id,
                reading_order=index,
                sha256=source.sha256,
                width=width,
                height=height,
                mime_type=source.mime_type,  # type: ignore[arg-type]
                original_filename=filename,
                derived_png_sha256=derived_sha,
            )
        )
    return pages, derived


def _write_intake(
    doc_root: Path,
    archive_dir: Path,
    manifest: DocumentBundleManifest,
    pages: list[PageAsset],
    derived: dict[str, bytes],
    document_hash: str,
) -> None:
    source_manifest = SourceManifest(
        batch_id=manifest.batch_id,
        document_id=manifest.document_id,
        language=manifest.language,
        source_bundle_sha256=document_hash,
        pages=[
            SourcePageRecord(
                page_id=page.page_id,
                reading_order=page.reading_order,
                sha256=page.sha256,
                width=page.width,
                height=page.height,
                filename=_original_name(page),
                mime_type=page.mime_type,
                derived_png_filename=(
                    f"{page.page_id}.derived.png" if page.page_id in derived else None
                ),
                derived_png_sha256=page.derived_png_sha256,
            )
            for page in pages
        ],
    )
    _write_if_same(
        doc_root / "input" / "source_manifest.json",
        canonical_json(source_manifest).encode("utf-8"),
    )
    images = [
        source
        for source in sorted(manifest.sources, key=lambda item: item.order_index)
        if source.mime_type in {"image/png", "image/jpeg"}
    ]
    for page, source in zip(pages, images, strict=True):
        _write_if_same(
            doc_root / "input" / _original_name(page),
            (archive_dir / source.relative_path).read_bytes(),
        )
        if page.page_id in derived:
            _write_if_same(
                doc_root / "input" / f"{page.page_id}.derived.png",
                derived[page.page_id],
            )


def _work_request(
    batch_id: str,
    document: DocumentReviewState,
    *,
    provider: str,
    stage: StageName,
    round_index: int,
    attempt: int,
) -> WorkRequest:
    task_id = f"{batch_id}:{document.document_id}:{stage}:r{round_index}:{provider}:a{attempt}"
    return WorkRequest(
        provider=provider,  # type: ignore[arg-type]
        provider_interface=interface_for(provider),
        task_id=task_id,
        job_id=f"job:{task_id}",
        batch_id=batch_id,
        document_id=document.document_id,
        stage=stage,
        round_index=round_index,
        attempt=attempt,
        source_bundle_sha256=document.source_bundle_sha256,
        output_relative_path=output_relative_path(stage, provider, round_index),
        page_ids=[page.page_id for page in document.pages],
        forbidden_inputs=[
            "original_ocr_transcription",
            "archive_transcription_of_this_document",
            "other_provider_output",
            "previous_ai_interpretation",
            "web_search_results",
            "historical_identity_suggestions",
        ],
    )


def _round_for_publish(
    document: DocumentReviewState,
    stage: StageName,
    provider: str,
    evidence: str | None,
    *,
    max_rounds: int,
) -> int:
    if stage != "challenge":
        if evidence is not None:
            raise ReviewPolicyError("evidence belongs only to a later challenge round")
        return 0
    if not document.challenge_rounds:
        if evidence is not None:
            raise ReviewPolicyError(
                "the first challenge round does not take extra evidence"
            )
        return 1
    current = document.challenge_rounds[-1]
    try:
        runtime = provider_runtime(
            document,
            stage="challenge",
            provider=provider,
            round_index=current.round_index,
        )
    except ReviewIntegrityError:
        runtime = ProviderRuntime(provider=provider)  # type: ignore[arg-type]
    if runtime.status in {
        ProviderStatus.NOT_STARTED,
        ProviderStatus.RETRYABLE_FAILURE,
        ProviderStatus.AWAITING_MANUAL_DISPATCH,
    }:
        return current.round_index
    if evidence is None:
        raise ReviewPolicyError("no repeated challenge without new evidence")
    if evidence not in NEW_EVIDENCE_KINDS:
        raise ReviewPolicyError("new challenge evidence is not an allowed kind")
    if not current.reconciled:
        raise ReviewPolicyError(
            "the current challenge round is still open or ended without a result; "
            "another round needs it reconciled first"
        )
    if len(document.challenge_rounds) >= max_rounds:
        raise ReviewPolicyError("challenge round cap is reached")
    return current.round_index + 1


def _runtime_or_new(
    document: DocumentReviewState,
    stage: str,
    provider: str,
    round_index: int,
) -> ProviderRuntime:
    if stage == "challenge" and not any(
        item.round_index == round_index for item in document.challenge_rounds
    ):
        return ProviderRuntime(provider=provider)  # type: ignore[arg-type]
    return provider_runtime(
        document,
        stage=stage,
        provider=provider,
        round_index=round_index,
    )


def _require_challenge_allowed(document: DocumentReviewState) -> None:
    if document.challenge_closure is not None:
        raise ReviewPolicyError(
            "the challenge stage is closed; no further challenge round can be published"
        )
    if document.comparison_sha256 is None:
        raise ReviewPolicyError("comparison must be recorded before a challenge")
    if not document.substantive_dispute_ids:
        raise ReviewPolicyError("punctuation and coverage are not challenged")


def _resolve_selection(
    document: DocumentReviewState,
    comparison: ComparisonFile,
    existing_round: ChallengeRoundState | None,
    requested: Sequence[str] | None,
    *,
    round_index: int = 1,
) -> tuple[list[str] | None, str | None, list[str] | None]:
    """Decide the selection a challenge publication uses, before anything is written.

    Returns ``(ledger_selection, comparison_hash, task_selection)``. A ``None``
    ledger selection is a legacy full-set round; ``task_selection`` is ``None``
    for it too, so the task covers every substantive dispute.
    """

    if document.comparison_sha256 is None:
        raise ReviewPolicyError("comparison must be recorded before a challenge")
    current_ids = [item.dispute_id for item in challengeable_disputes(comparison)]
    if current_ids != list(document.substantive_dispute_ids):
        raise ReviewIntegrityError(
            "recorded substantive disputes do not match the comparison"
        )
    if existing_round is not None:
        frozen = existing_round.selected_dispute_ids
        if requested is not None:
            expected = list(current_ids) if frozen is None else list(frozen)
            if len(list(requested)) != len(set(requested)) or set(requested) != set(
                expected
            ):
                raise ReviewPolicyError(
                    "the challenge selection is frozen at the first provider publication"
                )
        return (
            None if frozen is None else list(frozen),
            existing_round.selection_comparison_sha256,
            None if frozen is None else list(frozen),
        )
    if requested is None:
        if round_index == 1:
            return None, None, None
        # A later round targets only disputes that are still unresolved.
        unresolved = [
            item
            for item in current_ids
            if item not in set(resolved_dispute_ids(document))
        ]
        if not unresolved:
            raise ReviewPolicyError(
                "no unresolved dispute remains for another challenge round"
            )
        return unresolved, document.comparison_sha256, unresolved
    try:
        chosen = normalize_challenge_selection(
            requested,
            substantive_ids=current_ids,
            resolved_ids=resolved_dispute_ids(document),
        )
    except ValueError as exc:
        raise ReviewPolicyError(str(exc)) from exc
    return chosen, document.comparison_sha256, chosen


def _require_research_allowed(document: DocumentReviewState, doc_root: Path) -> None:
    if not document.visual_readings_preserved:
        raise ReviewPolicyError("research starts after visual readings are preserved")
    comparison = _load_comparison(doc_root, document)
    if comparison is None:
        raise ReviewPolicyError("research starts after visual readings are preserved")
    if not any(
        item.category is DisputeCategory.NAME_OR_ENTITY for item in comparison.disputes
    ):
        raise ReviewPolicyError("research is only opened for material entity disputes")


def _prepare_bundle(
    doc_root: Path,
    document: DocumentReviewState,
    request: WorkRequest,
    transcription: str,
    *,
    comparison: ComparisonFile | None,
    gpt: FirstPassOutput | None,
    grok: FirstPassOutput | None,
    instruction_note: str = "",
    extra_files: dict[str, bytes] | None = None,
    challenge_selection: Sequence[str] | None = None,
    challenge_contract: ChallengeContract | None = None,
    challenge_already_published: bool = False,
) -> None:
    stage_root = doc_root / stage_dir_name(request.stage, request.round_index)
    task_dir = stage_root / "tasks" / request.provider
    instructions = _instructions(request.stage, challenge_contract=challenge_contract)
    layout: ChallengeLayout = "GROUPED"
    if request.stage == "challenge":
        layout = _challenge_layout(
            stage_root, already_published=challenge_already_published
        )
        if challenge_contract is not None and layout != "GROUPED":
            raise ReviewIntegrityError(
                "an evidence-bound round must use grouped evidence"
            )
        if layout == "GROUPED":
            instructions = f"{instructions}{_grouped_challenge_note()}"
    if instruction_note:
        instructions = f"{instructions}\n{instruction_note.strip()}\n"
    _write_if_same(
        stage_root / f"instructions_{request.provider}.txt",
        instructions.encode("utf-8"),
    )
    files = {
        f"instructions_{request.provider}.txt": instructions.encode("utf-8"),
        "work_request.json": canonical_json(request).encode("utf-8"),
        "output_schema.json": _schema_text(
            request.stage, challenge_contract=challenge_contract
        ).encode("utf-8"),
        "README.txt": _sandbox_note().encode("utf-8"),
    }
    if request.stage == "first_pass":
        files.update(_first_pass_allowlist(doc_root, document))
    elif request.stage == "challenge":
        if comparison is None or gpt is None or grok is None:
            raise ReviewPolicyError("a challenge requires both preserved readings")
        files.update(
            _write_challenge_input(
                doc_root,
                document,
                comparison,
                gpt,
                grok,
                request.round_index,
                selection=challenge_selection,
                layout=layout,
                challenge_contract=challenge_contract,
            )
        )
    else:
        _write_research_input(doc_root, gpt_and_grok=_both_outputs(doc_root, document))
        research_input = doc_root / "research" / "input" / "research_request.json"
        files["input/research_request.json"] = research_input.read_bytes()
    if extra_files:
        overlap = set(extra_files) & set(files)
        if overlap:
            raise ReviewIntegrityError(
                "local pilot file collides with the task allowlist"
            )
        files.update(extra_files)
    _write_allowlisted(task_dir, files)
    (stage_root / "output").mkdir(parents=True, exist_ok=True)
    _reject_transcription_leak(task_dir, transcription)


def _challenge_layout(
    stage_root: Path, *, already_published: bool = False
) -> ChallengeLayout:
    """Grouped for a new round; a round already published without groups stays ungrouped.

    A round that an earlier version published keeps its exact bytes, so a
    second provider (or a retry) of that round gets the same files.
    """

    path = stage_root / "input" / "disputes.json"
    if not path.exists():
        if already_published:
            raise ReviewIntegrityError("published challenge input is missing")
        return "GROUPED"
    try:
        published = AnonymousChallengeFile.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise ReviewIntegrityError("published challenge input is not readable") from exc
    return "UNGROUPED" if published.groups is None else "GROUPED"


def _grouped_challenge_note() -> str:
    return (
        "Disputes are listed in groups. The disputes of one group are in the same\n"
        "aligned span and share one crop (or the labeled full-page fallback).\n"
        "Every listed dispute still needs its own decision. A group is not a\n"
        "decision, and your decision for one dispute says nothing about another.\n"
    )


@dataclass(frozen=True, slots=True)
class _CropPlan:
    crop_id: str
    members: tuple[Dispute, ...]
    target: BBox
    window_kind: Literal["TARGETED", "FULL_PAGE_FALLBACK"]
    group: ChallengeGroup | None


def _write_challenge_input(
    doc_root: Path,
    document: DocumentReviewState,
    comparison: ComparisonFile,
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    round_index: int,
    *,
    selection: Sequence[str] | None = None,
    layout: ChallengeLayout = "GROUPED",
    challenge_contract: ChallengeContract | None = None,
) -> dict[str, bytes]:
    stage_root = doc_root / stage_dir_name("challenge", round_index)
    files: dict[str, bytes] = {}
    items: list[AnonymousChallengeItem] = []
    disputes = challengeable_disputes(comparison)
    if selection is not None:
        chosen = set(selection)
        disputes = [item for item in disputes if item.dispute_id in chosen]
        if len(disputes) != len(chosen):
            raise ReviewIntegrityError(
                "selected disputes are not all in the recorded comparison"
            )
    pages = {item.page_id: item for item in document.pages}

    def window_of(
        dispute: Dispute,
    ) -> tuple[BBox, Literal["TARGETED", "FULL_PAGE_FALLBACK"]]:
        page = pages[dispute.page_id]
        target, kind = _target_window(dispute, gpt, grok, page.width, page.height)
        return target, kind  # type: ignore[return-value]

    plans: list[_CropPlan] = []
    groups: list[ChallengeGroup] = []
    if layout == "GROUPED":
        groups = group_challenge_disputes(disputes, comparison.alignments, window_of)
        required = required_decision_ids(groups)
        if len(required) != len(set(required)) or set(required) != {
            item.dispute_id for item in disputes
        }:
            raise ReviewIntegrityError(
                "challenge groups do not cover the selected disputes"
            )
        by_id = {item.dispute_id: item for item in disputes}
        for group in groups:
            plans.append(
                _CropPlan(
                    group.crop_id,
                    tuple(by_id[dispute_id] for dispute_id in group.dispute_ids),
                    group.target,
                    group.window_kind,
                    group,
                )
            )
    else:
        for dispute in disputes:
            target, kind = window_of(dispute)
            plans.append(
                _CropPlan(f"crop_{dispute.dispute_id}", (dispute,), target, kind, None)
            )

    group_of: dict[str, ChallengeGroup] = {}
    records: list[AnonymousChallengeGroup] = []
    for plan in plans:
        first = plan.members[0]
        page = pages[first.page_id]
        original_bytes = (doc_root / "input" / _original_name(page)).read_bytes()
        pixel_bytes = (doc_root / "input" / _pixel_name(page)).read_bytes()
        shared = plan.group is not None and len(plan.members) > 1
        try:
            crop = build_crop(
                crop_id=plan.crop_id,
                dispute_id=None if shared else first.dispute_id,
                group_id=plan.group.group_id
                if shared and plan.group is not None
                else None,
                dispute_ids=[item.dispute_id for item in plan.members]
                if shared
                else None,
                page_id=first.page_id,
                page_sha256=page.sha256,
                page_width=page.width,
                page_height=page.height,
                target=plan.target,
                window_kind=plan.window_kind,
            )
            crop, crop_png = materialize_crop(crop, pixel_bytes)
            crop = crop.model_copy(
                update={
                    "source_mime_type": page.mime_type,
                    "derived_png_sha256": page.derived_png_sha256,
                }
            )
            assert_crop_contains_target(crop)
        except ValueError as exc:
            raise ReviewPolicyError(str(exc)) from exc
        crop_json = canonical_json(crop).encode("utf-8")
        _write_if_same(
            stage_root / "input" / "crops" / f"{crop.crop_id}.json", crop_json
        )
        _write_if_same(stage_root / "input" / "crops" / f"{crop.crop_id}.png", crop_png)
        _write_if_same(stage_root / "input" / _original_name(page), original_bytes)
        files[f"input/crops/{crop.crop_id}.json"] = crop_json
        files[f"input/crops/{crop.crop_id}.png"] = crop_png
        files[f"input/{_original_name(page)}"] = original_bytes
        if _pixel_name(page) != _original_name(page):
            _write_if_same(stage_root / "input" / _pixel_name(page), pixel_bytes)
            files[f"input/{_pixel_name(page)}"] = pixel_bytes
        if plan.group is not None:
            assert crop.crop_sha256 is not None
            records.append(
                AnonymousChallengeGroup(
                    group_id=plan.group.group_id,
                    page_id=plan.group.page_id,
                    page_sha256=page.sha256,
                    basis=plan.group.basis,
                    dispute_ids=list(plan.group.dispute_ids),
                    crop_id=crop.crop_id,
                    crop_sha256=crop.crop_sha256,
                    window_kind=plan.group.window_kind,
                    evidence_refs=[
                        EvidenceReference(
                            kind="GROUP",
                            evidence_id=plan.group.group_id,
                            page_id=plan.group.page_id,
                        ),
                        EvidenceReference(
                            kind=plan.group.evidence_kind,
                            evidence_id=crop.crop_id,
                            page_id=plan.group.page_id,
                        ),
                    ],
                )
            )
            for member in plan.members:
                group_of[member.dispute_id] = plan.group
    crop_of = {
        member.dispute_id: plan.crop_id for plan in plans for member in plan.members
    }
    for dispute in disputes:
        items.append(
            AnonymousChallengeItem(
                dispute_id=dispute.dispute_id,
                page_id=dispute.page_id,
                crop_id=crop_of[dispute.dispute_id],
                group_id=group_of[dispute.dispute_id].group_id
                if layout == "GROUPED"
                else None,
                candidate_x=dispute.gpt_text,
                candidate_y=dispute.grok_text,
            )
        )
    if layout == "GROUPED":
        challenge_input = AnonymousChallengeFile(
            schema_version=(
                CHALLENGE_INPUT_V3
                if challenge_contract == EVIDENCE_BOUND_CONTRACT
                else CHALLENGE_INPUT_V2
            ),
            disputes=items,
            groups=records,
            required_decision_ids=[item.dispute_id for item in items],
            challenge_contract=challenge_contract,
        )
    else:
        challenge_input = AnonymousChallengeFile(disputes=items)
    disputes_json = canonical_json(challenge_input).encode("utf-8")
    _write_if_same(stage_root / "input" / "disputes.json", disputes_json)
    files["input/disputes.json"] = disputes_json
    return files


def _write_research_input(
    doc_root: Path,
    *,
    gpt_and_grok: tuple[FirstPassOutput | None, FirstPassOutput | None],
) -> None:
    gpt, grok = gpt_and_grok
    readings = []
    for label, output in (("visual_a", gpt), ("visual_b", grok)):
        if output is None:
            continue
        readings.append(
            {
                "label": label,
                "pages": [
                    {
                        "page_id": page.page_id,
                        "diplomatic_transcription": page.diplomatic_transcription,
                    }
                    for page in output.pages
                ],
            }
        )
    payload = {
        "schema_version": "review-research-input-v1",
        "archive_transcription_included": False,
        "target_transcription_cannot_corroborate_itself": True,
        "diplomatic_spelling_is_fixed": True,
        "visual_readings": readings,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_if_same(doc_root / "research" / "input" / "research_request.json", encoded)


def _both_outputs(
    doc_root: Path,
    document: DocumentReviewState,
) -> tuple[FirstPassOutput | None, FirstPassOutput | None]:
    return (
        _load_output(doc_root, document, "gpt"),
        _load_output(doc_root, document, "grok"),
    )


def _target_window(
    dispute: Dispute,
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    width: int,
    height: int,
) -> tuple[BBox, str]:
    boxes: list[BBox] = []
    pairs = (
        (gpt, dispute.gpt_region_ids),
        (grok, dispute.grok_region_ids),
    )
    for output, region_ids in pairs:
        page = next(item for item in output.pages if item.page_id == dispute.page_id)
        for region_id in region_ids:
            region = _region(page, region_id)
            if region is not None and region.bbox is not None:
                boxes.append(region.bbox)
    if not boxes and dispute.region_id is not None:
        for output in (gpt, grok):
            page = next(
                item for item in output.pages if item.page_id == dispute.page_id
            )
            region = _region(page, dispute.region_id)
            if region is not None and region.bbox is not None:
                boxes.append(region.bbox)
    if not boxes:
        return page_bbox(width, height), "FULL_PAGE_FALLBACK"
    return _union(boxes), "TARGETED"


def _region(page: PageTranscription, region_id: str | None) -> TextRegion | None:
    if region_id is None:
        return None
    for region in page.regions:
        if region.region_id == region_id:
            return region
    return None


def _union(boxes: list[BBox]) -> BBox:
    x = min(box.x for box in boxes)
    y = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BBox(x=x, y=y, width=right - x, height=bottom - y)


def _instructions(
    stage: str, *, challenge_contract: ChallengeContract | None = None
) -> str:
    if stage == "first_pass":
        return (
            "Transcribe every page image in the given reading order.\n"
            "Return diplomatic transcription, region and line ids, unreadable spans,\n"
            "alternative readings, attention items, cancellations, insertions,\n"
            "marginal notes, and a coverage report.\n"
            "coverage_complete means every visible region was examined.\n"
            "unresolved_text_present may still be true when coverage is complete.\n"
            "Do not use an archive transcription, another model's output, or web search.\n"
            "Write one JSON file. Dispatch is manual. No API is assumed.\n"
            f"{first_pass_instruction_note()}"
        )
    if stage == "challenge":
        legacy = (
            "Inspect the page image and the crop window. The crop is existing pixels,\n"
            "not new detail. FULL_PAGE_FALLBACK means no region box was usable,\n"
            "so the window is the whole page and is not a targeted crop.\n"
            "Candidate readings are anonymous.\n"
            "Choose CANDIDATE_X, CANDIDATE_Y, THIRD_READING, or UNRESOLVED.\n"
            "Cite visual evidence. Do not change a reading only to force agreement.\n"
            "Write one JSON file. Dispatch is manual. No API is assumed.\n"
        )
        if challenge_contract is None:
            return legacy
        return (
            legacy
            + "For every decision, provide evidence_ref and evidence_assessment.\n"
            + "evidence_ref must exactly copy one supplied reference for that dispute's\n"
            + "group/crop/full-page fallback, including kind, evidence_id, and page_id.\n"
            + "evidence_assessment is SUPPORTS_DECISION, INSUFFICIENT, or UNREADABLE.\n"
            + "Use INSUFFICIENT or UNREADABLE when the cited evidence does not support\n"
            + "your selected reading; matching another decision is not visual support.\n"
        )
    return (
        "Use the preserved visual readings. Do not replace diplomatic spelling.\n"
        "The existing transcription of this document cannot corroborate itself.\n"
        "Record the hypothesis, supporting and opposing evidence, sources, links,\n"
        "whether the original image was inspected, limitations, and confidence.\n"
        "Write one JSON file. Dispatch is manual. No API is assumed.\n"
    )


def _schema_text(
    stage: str, *, challenge_contract: ChallengeContract | None = None
) -> str:
    if stage == "first_pass":
        return first_pass_schema_text()
    if stage == "challenge" and challenge_contract == EVIDENCE_BOUND_CONTRACT:
        return (
            json.dumps(
                {
                    "schema_version": "review-provider-output-v1",
                    "stage": "challenge",
                    "round_contract": EVIDENCE_BOUND_CONTRACT,
                    "note": "Return data only. Do not include commands.",
                    "decisions": {
                        "required_for_every_selected_dispute": True,
                        "fields": {
                            "dispute_id": "exact required decision ID",
                            "decision": [
                                "CANDIDATE_X",
                                "CANDIDATE_Y",
                                "THIRD_READING",
                                "UNRESOLVED",
                            ],
                            "third_reading": "required only for THIRD_READING; preserve exact text",
                            "visual_evidence": "non-empty evidence statement",
                            "evidence_ref": {
                                "kind": ["GROUP", "CROP", "FULL_PAGE_FALLBACK"],
                                "evidence_id": "exactly as published for this dispute",
                                "page_id": "exactly as published for this dispute",
                            },
                            "evidence_assessment": [
                                "SUPPORTS_DECISION",
                                "INSUFFICIENT",
                                "UNREADABLE",
                            ],
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return (
        json.dumps(
            {
                "schema_version": "review-provider-output-v1",
                "stage": stage,
                "note": "Return data only. Do not include commands.",
            },
            indent=2,
        )
        + "\n"
    )


def _read_source_evidence(
    source_json: Path,
    output: Path,
    provider: str,
    model: FirstPassOutput,
) -> list[SourceEvidence]:
    """Read unmapped values from the original provider JSON. The file is not written."""

    source = Path(source_json).expanduser().resolve()
    if source == Path(output).resolve():
        raise ReviewUsageError(
            "source evidence file must be the original, not the accepted output"
        )
    if not source.is_file():
        raise ReviewUsageError("source evidence file is missing")
    data = source.read_bytes()
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ReviewUsageError("source evidence file is not JSON") from exc
    if not isinstance(payload, dict):
        raise ReviewUsageError("source evidence file is not an object")
    labeled = payload.get("provider")
    if labeled is not None and labeled != provider:
        raise ReviewUsageError(
            "source evidence provider does not match the accepted output"
        )
    return collect_source_evidence(
        payload,
        model,
        original_sha256=sha256_bytes(data),
    )


def _parse_provider_file(
    path: Path,
    *,
    batch_id: str,
    provider: str,
    stage: str,
    document: DocumentReviewState,
    runtime: ProviderRuntime,
    round_index: int,
    doc_root: Path | None = None,
) -> ProviderFile:
    if path.name != output_filename(stage, provider):
        raise ReviewOutputRejected("mismatched", "output filename is not allowlisted")
    try:
        payload = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ReviewOutputRejected("schema", "provider output is not JSON") from exc
    if not isinstance(payload, dict):
        raise ReviewOutputRejected("schema", "provider output is not an object")
    if "provider" in payload or "provider_interface" in payload:
        if payload.get("provider") != provider or payload.get(
            "provider_interface"
        ) != interface_for(provider):
            raise ReviewOutputRejected(
                "invalid_label", "provider label does not match the file"
            )
    try:
        if stage == "first_pass":
            model: ProviderFile = FirstPassOutput.model_validate(payload)
        elif stage == "challenge":
            model = ChallengeOutput.model_validate(payload)
        elif stage == "research":
            model = ResearchOutput.model_validate(payload)
        else:
            raise KeyError(stage)
    except ValidationError as exc:
        missing = [
            ".".join(str(part) for part in error["loc"])
            for error in exc.errors()
            if error.get("type") == "missing"
        ]
        if missing:
            detail = "missing mandatory fields: " + ", ".join(missing)
        else:
            detail = "provider output does not match the schema"
        raise ReviewOutputRejected("schema", detail) from exc
    if (
        model.source_bundle_sha256 != document.source_bundle_sha256
        or model.attempt != runtime.attempt
    ):
        raise ReviewOutputRejected(
            "stale", "provider output does not match the current source or attempt"
        )
    if (
        model.task_id != runtime.task_id
        or model.job_id != runtime.job_id
        or model.batch_id != batch_id
        or model.document_id != document.document_id
    ):
        raise ReviewOutputRejected(
            "mismatched", "provider output does not match the published task"
        )
    if isinstance(model, FirstPassOutput):
        expected = [(page.page_id, page.reading_order) for page in document.pages]
        actual = [(page.page_id, page.reading_order) for page in model.pages]
        if actual != expected:
            raise ReviewOutputRejected(
                "mismatched", "pages do not match the sealed reading order"
            )
    if isinstance(model, ChallengeOutput):
        if doc_root is None:
            raise ReviewIntegrityError(
                "challenge validation requires the published task directory"
            )
        if model.round_index != round_index:
            raise ReviewOutputRejected(
                "mismatched", "challenge round does not match the task"
            )
        expected_ids = set(round_selection(document, round_index))
        got_ids = [item.dispute_id for item in model.decisions]
        if len(got_ids) != len(set(got_ids)) or set(got_ids) != expected_ids:
            raise ReviewOutputRejected(
                "mismatched", "challenge decisions do not match the disputes"
            )
        challenge = next(
            item
            for item in document.challenge_rounds
            if item.round_index == round_index
        )
        published, scopes = _published_challenge_scopes(
            doc_root,
            document,
            round_index,
            expected_contract=challenge.challenge_contract,
            provider=provider,
        )
        if challenge.challenge_contract == EVIDENCE_BOUND_CONTRACT:
            if published.schema_version != CHALLENGE_INPUT_V3:
                raise ReviewOutputRejected(
                    "mismatched",
                    "evidence-bound round input has the wrong contract version",
                )
            for decision in model.decisions:
                try:
                    check_evidence_binding(
                        decision, scopes[decision.dispute_id], provider=provider
                    )
                except EvidenceBindingError as exc:
                    reason = (
                        "schema"
                        if exc.code == "MISSING_EVIDENCE_BINDING"
                        else "mismatched"
                    )
                    raise ReviewOutputRejected(reason, str(exc)) from exc
        elif any(
            item.evidence_ref is not None or item.evidence_assessment is not None
            for item in model.decisions
        ):
            raise ReviewOutputRejected(
                "schema",
                "a historical legacy round cannot be silently upgraded to evidence-bound output",
            )
    return model


def _published_challenge_scopes(
    doc_root: Path,
    document: DocumentReviewState,
    round_index: int,
    *,
    expected_contract: ChallengeContract | None,
    provider: str | None = None,
) -> tuple[AnonymousChallengeFile, dict[str, DisputeEvidenceScope]]:
    """Validate immutable published evidence and return per-dispute scopes.

    The ledger contract determines behavior. Provider fields never select the
    legacy path. The published input and every referenced crop/page must still
    exist and agree with the recorded comparison and selection.
    """

    stage_root = doc_root / stage_dir_name("challenge", round_index)
    input_root = stage_root / "input"
    path = input_root / "disputes.json"
    if not path.is_file():
        raise ReviewIntegrityError("published challenge input is missing")
    try:
        published = AnonymousChallengeFile.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise ReviewIntegrityError("published challenge input is not readable") from exc
    if expected_contract == EVIDENCE_BOUND_CONTRACT:
        if (
            published.schema_version != CHALLENGE_INPUT_V3
            or published.challenge_contract != EVIDENCE_BOUND_CONTRACT
        ):
            raise ReviewIntegrityError(
                "published challenge input disagrees with the evidence-bound ledger contract"
            )
    elif (
        published.schema_version == CHALLENGE_INPUT_V3
        or published.challenge_contract is not None
    ):
        raise ReviewIntegrityError(
            "historical challenge input disagrees with its legacy ledger contract"
        )

    comparison = _load_comparison(doc_root, document)
    if comparison is None:
        raise ReviewIntegrityError("published challenge has no recorded comparison")
    expected_ids = round_selection(document, round_index)
    if [item.dispute_id for item in published.disputes] != expected_ids:
        raise ReviewIntegrityError(
            "published challenge disputes disagree with the frozen round"
        )
    disputes = {item.dispute_id: item for item in comparison.disputes}
    pages = {item.page_id: item for item in document.pages}
    for item in published.disputes:
        dispute = disputes.get(item.dispute_id)
        if (
            dispute is None
            or item.page_id != dispute.page_id
            or item.candidate_x != dispute.gpt_text
            or item.candidate_y != dispute.grok_text
        ):
            raise ReviewIntegrityError(
                "published challenge item disagrees with the recorded comparison"
            )

    scopes: dict[str, DisputeEvidenceScope] = {}
    groups = published.groups or []
    if groups:
        for group in groups:
            page = pages.get(group.page_id)
            if page is None or group.page_sha256 != page.sha256:
                raise ReviewIntegrityError(
                    "published challenge group has the wrong source page"
                )
            source = input_root / _original_name(page)
            if not source.is_file() or sha256_bytes(source.read_bytes()) != page.sha256:
                raise ReviewIntegrityError(
                    "published challenge source page is missing or changed"
                )
            crop_json = input_root / "crops" / f"{group.crop_id}.json"
            crop_png = input_root / "crops" / f"{group.crop_id}.png"
            if not crop_json.is_file() or not crop_png.is_file():
                raise ReviewIntegrityError("published challenge crop is incomplete")
            try:
                crop = CropRecord.model_validate_json(crop_json.read_bytes())
            except ValidationError as exc:
                raise ReviewIntegrityError(
                    "published challenge crop is not readable"
                ) from exc
            png_digest = sha256_bytes(crop_png.read_bytes())
            if (
                crop.crop_id != group.crop_id
                or crop.page_id != group.page_id
                or crop.page_sha256 != group.page_sha256
                or crop.window_kind != group.window_kind
                or crop.crop_sha256 != group.crop_sha256
                or png_digest != group.crop_sha256
            ):
                raise ReviewIntegrityError(
                    "published challenge crop disagrees with its group"
                )
            if len(group.dispute_ids) > 1:
                if (
                    crop.group_id != group.group_id
                    or crop.dispute_ids != group.dispute_ids
                    or crop.dispute_id is not None
                ):
                    raise ReviewIntegrityError(
                        "published shared crop has the wrong group membership"
                    )
            elif (
                crop.dispute_id != group.dispute_ids[0]
                or crop.group_id is not None
                or crop.dispute_ids is not None
            ):
                raise ReviewIntegrityError(
                    "published singleton crop has the wrong dispute owner"
                )
            allowed = frozenset(
                (reference.kind, reference.evidence_id)
                for reference in group.evidence_refs
            )
            for dispute_id in group.dispute_ids:
                scopes[dispute_id] = DisputeEvidenceScope(
                    dispute_id=dispute_id,
                    page_id=group.page_id,
                    allowed=allowed,
                )
    else:
        for item in published.disputes:
            crop_json = input_root / "crops" / f"{item.crop_id}.json"
            crop_png = input_root / "crops" / f"{item.crop_id}.png"
            if not crop_json.is_file() or not crop_png.is_file():
                raise ReviewIntegrityError("published legacy crop is incomplete")
            try:
                crop = CropRecord.model_validate_json(crop_json.read_bytes())
            except ValidationError as exc:
                raise ReviewIntegrityError(
                    "published legacy crop is not readable"
                ) from exc
            if (
                crop.dispute_id != item.dispute_id
                or crop.page_id != item.page_id
                or crop.crop_sha256 != sha256_bytes(crop_png.read_bytes())
            ):
                raise ReviewIntegrityError("published legacy crop changed")
            kind = "CROP" if crop.window_kind == "TARGETED" else "FULL_PAGE_FALLBACK"
            scopes[item.dispute_id] = DisputeEvidenceScope(
                dispute_id=item.dispute_id,
                page_id=item.page_id,
                allowed=frozenset({(kind, item.crop_id)}),
            )

    if set(scopes) != set(expected_ids):
        raise ReviewIntegrityError(
            "published evidence does not cover every challenged dispute"
        )
    challenge = next(
        item for item in document.challenge_rounds if item.round_index == round_index
    )
    providers = (
        [provider]
        if provider is not None
        else [item.provider for item in challenge.providers if item.task_id is not None]
    )
    for task_provider in providers:
        task_root = stage_root / "tasks" / task_provider
        stage_files = {
            item.relative_to(input_root).as_posix(): item.read_bytes()
            for item in input_root.rglob("*")
            if item.is_file()
        }
        task_input = task_root / "input"
        task_files = (
            {
                item.relative_to(task_input).as_posix(): item.read_bytes()
                for item in task_input.rglob("*")
                if item.is_file()
            }
            if task_input.is_dir()
            else {}
        )
        if task_files != stage_files:
            raise ReviewIntegrityError(
                "published provider task evidence is missing or inconsistent"
            )
        grouped_note = _grouped_challenge_note() if published.groups is not None else ""
        expected_instructions = (
            _instructions("challenge", challenge_contract=expected_contract)
            + grouped_note
        ).encode("utf-8")
        instruction = task_root / f"instructions_{task_provider}.txt"
        schema = task_root / "output_schema.json"
        request = task_root / "work_request.json"
        if (
            not instruction.is_file()
            or instruction.read_bytes() != expected_instructions
            or not schema.is_file()
            or schema.read_bytes()
            != _schema_text("challenge", challenge_contract=expected_contract).encode(
                "utf-8"
            )
            or not request.is_file()
        ):
            raise ReviewIntegrityError(
                "published provider task contract is missing or inconsistent"
            )
    return published, scopes


def _load_output(
    doc_root: Path,
    document: DocumentReviewState,
    provider: str,
) -> FirstPassOutput | None:
    runtime = provider_runtime(
        document, stage="first_pass", provider=provider, round_index=0
    )
    if runtime.status is not ProviderStatus.SUCCEEDED or runtime.output_sha256 is None:
        return None
    if runtime.output_relative_path is None:
        raise ReviewIntegrityError("accepted output is missing its path")
    path = contained(doc_root, doc_root / runtime.output_relative_path)
    data = path.read_bytes()
    if sha256_bytes(data) != runtime.output_sha256:
        raise ReviewIntegrityError("accepted provider output changed on disk")
    return FirstPassOutput.model_validate_json(data)


def _load_comparison(
    doc_root: Path, document: DocumentReviewState
) -> ComparisonFile | None:
    if document.comparison_sha256 is None:
        return None
    path = doc_root / "comparison" / "comparison.json"
    data = path.read_bytes()
    if sha256_bytes(data) != document.comparison_sha256:
        raise ReviewIntegrityError("comparison file changed after it was recorded")
    return ComparisonFile.model_validate_json(data)


def _provider_hash(document: DocumentReviewState, provider: str) -> str:
    runtime = provider_runtime(
        document, stage="first_pass", provider=provider, round_index=0
    )
    if runtime.output_sha256 is None:
        raise ReviewIntegrityError("provider output hash is missing")
    return runtime.output_sha256


def _research_identifications(
    doc_root: Path,
    document: DocumentReviewState,
) -> list[Identification]:
    found: list[Identification] = []
    for runtime in document.research:
        if (
            runtime.status is not ProviderStatus.SUCCEEDED
            or runtime.output_relative_path is None
        ):
            continue
        if runtime.output_sha256 is None:
            continue
        path = contained(doc_root, doc_root / runtime.output_relative_path)
        data = path.read_bytes()
        if sha256_bytes(data) != runtime.output_sha256:
            raise ReviewIntegrityError("accepted research output changed on disk")
        parsed = ResearchOutput.model_validate_json(data)
        found.extend(parsed.identifications)
    return found


def _manifest(
    state: ReviewBatchState,
    document: DocumentReviewState,
    *,
    text_sha: str,
    report_sha: str,
    summary_sha: str,
) -> ReviewManifest:
    providers: list[ManifestProvider] = []
    runtimes = list(document.first_pass) + list(document.research)
    for challenge in document.challenge_rounds:
        runtimes.extend(challenge.providers)
    for runtime in runtimes:
        if runtime.status is not ProviderStatus.SUCCEEDED:
            continue
        if (
            runtime.task_id is None
            or runtime.job_id is None
            or runtime.output_sha256 is None
        ):
            continue
        if any(runtime is item for item in document.research):
            stage = "research"
        elif any(
            runtime is provider_item
            for challenge in document.challenge_rounds
            for provider_item in challenge.providers
        ):
            stage = "challenge"
        else:
            stage = "first_pass"
        providers.append(
            ManifestProvider(
                provider=runtime.provider,
                provider_interface=interface_for(runtime.provider),
                stage=stage,
                task_id=runtime.task_id,
                job_id=runtime.job_id,
                output_sha256=runtime.output_sha256,
            )
        )
    if state.source_batch_manifest_sha256 is None:
        raise ReviewIntegrityError("review batch is missing its source hash")
    return ReviewManifest(
        batch_id=state.batch_id,
        document_id=document.document_id,
        source_bundle_sha256=document.source_bundle_sha256,
        source_batch_manifest_sha256=state.source_batch_manifest_sha256,
        providers=providers,
        comparison_sha256=document.comparison_sha256,
        artifacts=[
            ManifestArtifact(name="proposed_transcription.txt", sha256=text_sha),
            ManifestArtifact(name="review_report.json", sha256=report_sha),
            ManifestArtifact(name="review_summary.md", sha256=summary_sha),
        ],
    )


def _pending_opened_stage(document: DocumentReviewState) -> str | None:
    """An opened challenge or research stage that has not reached a terminal result."""

    terminal = {
        ProviderStatus.SUCCEEDED,
        ProviderStatus.FAILED,
        ProviderStatus.INVALID_OUTPUT,
        ProviderStatus.QUOTA_BLOCKED,
    }
    closure = document.challenge_closure
    failed_rounds = set() if closure is None else set(closure.failed_round_indexes)
    for challenge in document.challenge_rounds:
        if challenge.round_index in failed_rounds:
            # Explicitly closed after a confirmed terminal failure; the
            # closure was validated to have no live dispatch.
            continue
        if any(item.status not in terminal for item in challenge.providers):
            return "challenge"
        if not challenge.reconciled:
            return "challenge"
    for item in document.research:
        if item.status in {
            ProviderStatus.AWAITING_MANUAL_DISPATCH,
            ProviderStatus.RUNNING,
            ProviderStatus.RETRYABLE_FAILURE,
        }:
            return "research"
    return None


def _dispatch_still_open(document: DocumentReviewState) -> bool:
    runtimes = list(document.first_pass) + list(document.research)
    for challenge in document.challenge_rounds:
        runtimes.extend(challenge.providers)
    return any(
        item.status
        in {
            ProviderStatus.AWAITING_MANUAL_DISPATCH,
            ProviderStatus.RUNNING,
            ProviderStatus.RETRYABLE_FAILURE,
        }
        for item in runtimes
    )


def _require_matching_success(runtime: ProviderRuntime, output: Path) -> None:
    if not output.is_file():
        raise ReviewResultMissing(f"missing provider output {output.name}")
    digest = sha256_bytes(output.read_bytes())
    if digest != runtime.output_sha256:
        raise ReviewOutputRejected(
            "duplicated",
            "a later file conflicts with the accepted provider result",
        )


def _request_path(doc_root: Path, stage: str, provider: str, round_index: int) -> Path:
    return (
        doc_root
        / stage_dir_name(stage, round_index)
        / "tasks"
        / provider
        / "work_request.json"
    )


def _original_name(page: PageAsset) -> str:
    return page.original_filename or f"{page.page_id}.png"


def _pixel_name(page: PageAsset) -> str:
    if page.mime_type == "image/jpeg":
        return f"{page.page_id}.derived.png"
    return _original_name(page)


def _version_dir(version: int) -> str:
    return f"final/versions/v{version:03d}"


def _version_paths(doc_root: Path, relative: str) -> dict[str, Path]:
    paths = {name: doc_root / relative / name for name in FINAL_NAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ReviewIntegrityError("report version artifacts are incomplete")
    return paths


def _require_version_history(doc_root: Path, document: DocumentReviewState) -> None:
    for version in document.report_versions:
        directory = doc_root / version.relative_dir
        for name, digest in version.artifact_sha256.items():
            path = directory / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
                raise ReviewIntegrityError("an earlier report version changed on disk")


def _final_paths(doc_root: Path) -> dict[str, Path]:
    paths = {name: doc_root / "final" / name for name in FINAL_NAMES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ReviewIntegrityError("final artifacts are incomplete")
    return paths


def _archive_text(batches_dir: Path, batch_id: str, document_id: int) -> str:
    path = (
        archive_document_dir(batches_dir, batch_id, document_id)
        / "transcription"
        / "transcription.txt"
    )
    return path.read_text(encoding="utf-8")


def _sandbox_note() -> str:
    return (
        "This directory contains only the allowlisted files for one provider and one stage.\n"
        "It is a separate folder from the other provider and from earlier stages.\n"
        "Folder separation is not an operating-system sandbox.\n"
    )


def _first_pass_allowlist(
    doc_root: Path, document: DocumentReviewState
) -> dict[str, bytes]:
    """Copy only the sealed page images and the source manifest."""

    files = {
        "input/source_manifest.json": (
            doc_root / "input" / "source_manifest.json"
        ).read_bytes()
    }
    for page in document.pages:
        files[f"input/{_original_name(page)}"] = (
            doc_root / "input" / _original_name(page)
        ).read_bytes()
        if _pixel_name(page) != _original_name(page):
            files[f"input/{_pixel_name(page)}"] = (
                doc_root / "input" / _pixel_name(page)
            ).read_bytes()
    return files


def _write_allowlisted(task_dir: Path, files: dict[str, bytes]) -> None:
    for relative, payload in files.items():
        destination = task_dir / relative
        if relative == "work_request.json":
            _replace_file(destination, payload)
        else:
            _write_if_same(destination, payload)
    actual = {
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file()
    }
    if actual != set(files):
        raise ReviewIntegrityError("task bundle contains a file outside the allowlist")


def _validated_evidence(
    document: DocumentReviewState,
    doc_root: Path,
    round_index: int,
    evidence: str | None,
    evidence_path: Path | None,
) -> tuple[str, str]:
    if evidence not in NEW_EVIDENCE_KINDS:
        raise ReviewPolicyError(
            "a second challenge round requires a concrete new evidence kind"
        )
    if evidence_path is None:
        raise ReviewPolicyError(
            "a declared evidence category is not a new evidence file"
        )
    path = Path(evidence_path)
    if not path.is_file():
        raise ReviewPolicyError("new challenge evidence file does not exist")
    resolved = path.resolve()
    for root in FORBIDDEN_WRITE_ROOTS:
        forbidden = root.resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ReviewPolicyError(
                "new challenge evidence cannot be taken from the archive checkout"
            )
    data = path.read_bytes()
    if not data:
        raise ReviewPolicyError("new challenge evidence file is empty")
    digest = sha256_bytes(data)
    examined = {page.sha256 for page in document.pages}
    for challenge in document.challenge_rounds:
        if challenge.evidence_sha256 is not None:
            examined.add(challenge.evidence_sha256)
    if digest in examined:
        raise ReviewPolicyError(
            "new challenge evidence is not different from evidence already examined"
        )
    suffix = (
        path.suffix.lower()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        else ".bin"
    )
    relative = f"{stage_dir_name('challenge', round_index)}/evidence/evidence_{digest[:16]}{suffix}"
    _write_if_same(doc_root / relative, data)
    return digest, relative


def _reject_transcription_leak(root: Path, transcription: str) -> None:
    needle = transcription.encode("utf-8") if len(transcription) >= 12 else b""
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "transcription.txt":
            raise ReviewIntegrityError(
                "archive transcription file entered a provider bundle"
            )
        if needle and needle in path.read_bytes():
            raise ReviewIntegrityError(
                "archive transcription text entered a provider bundle"
            )


def _replace_file(path: Path, payload: bytes) -> None:
    """Replace a controller-owned request. Provider outputs are not replaced."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
    fd = os.open(temporary, flags, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _write_if_same(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ReviewIntegrityError(f"refusing to change {path.name}")
        return
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
