"""Read-only learning-candidate export for a completed transcription review.

The export copies recorded provider readings, challenge decisions, evidence
references, and reconciled spans. It does not approve a reading, invent a
coordinate, or write a learning ledger. Model agreement stays
``NOT_VERIFIED`` and is not eligible for authoritative learning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.layout import assert_safe_relative_path
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.bundling.verify import verify_batch_bundle
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.errors import (
    ReviewError,
    ReviewIntegrityError,
    ReviewPathError,
    ReviewUsageError,
)
from transcription_review_agent.review.models import (
    BBox,
    ChallengeOutput,
    ComparisonFile,
    CropRecord,
    Dispute,
    EvidenceReference,
    FirstPassOutput,
    PartialAgreementSpan,
    ProviderName,
    ReconciledDecision,
    ResearchOutput,
    ReviewManifest,
    ReviewReport,
    SourceManifest,
    SourcePageRecord,
)
from transcription_review_agent.review.paths import (
    contained,
    output_relative_path,
    reviews_root,
)
from transcription_review_agent.review.runtime import (
    DocumentReviewState,
    ReviewBatchState,
    ReviewStore,
    _read_checkpoint,
    _replay,
)

JSON_NAME = "learning_candidates.json"
MARKDOWN_NAME = "learning_candidates_review.md"
DEFAULT_OUTPUT_DIR = Path("/tmp/transcription_learning_candidates")

LearningCategory = Literal[
    "CORRECTION_CANDIDATE",
    "MODEL_SUPPORTED_READING",
    "UNRESOLVED_DISAGREEMENT",
    "REJECTED_OR_SUPERSEDED_READING",
]
ImageSpanStatus = Literal[
    "UNAVAILABLE",
    "RECORDED_REGION_BBOX",
    "RECORDED_TARGETED_WINDOW",
    "CONFLICTING_REGION_BBOX",
]
ReconciledStatus = Literal[
    "AUTOMATICALLY_SUPPORTED",
    "UNRESOLVED",
    "LEGACY_UNCLASSIFIED",
    "NOT_CHALLENGED",
    "PARTIAL_AGREEMENT",
]
CandidatePart = Literal["primary", "partial", "rejected", "superseded"]

_MARKDOWN_CATEGORIES = frozenset(
    {
        "WORDING",
        "NAME_OR_ENTITY",
        "DATE_OR_NUMBER",
        "CANCELLATION_OR_INSERTION",
        "MISSING_CONTENT",
        "READING_ORDER",
    }
)
_TASK_ID = re.compile(
    r"^(batch-[0-9]{6}):([0-9]+):(first_pass|challenge|research):r([0-9]+):(gpt|grok):a([0-9]+)$"
)
_LIMITATIONS: tuple[str, ...] = (
    "No human-verified transcription span is recorded. Every candidate is NOT_VERIFIED.",
    "Dual-model agreement is not human approval and is not eligible for authoritative learning.",
    "CORRECTION_CANDIDATE is not assigned. This export does not decide a correct transcription.",
    "Writer identity and handwriting style are not recorded and are not inferred.",
    "Research identifications are not copied. A historical hypothesis is not a handwriting label.",
    "Archive and website transcriptions are not ground truth and are not copied as readings.",
    "Exact image-to-text coordinates are copied only from a recorded region box or a targeted challenge window. A full-page fallback is not a text span.",
    "Character offsets are indexes into the recorded provider reading. They are not pixel positions.",
    "Non-substantive disputes that were not challenged are counted in the omission totals and are not exported as candidate records.",
    "This file is a read-only preview. It is not a learning ledger and it does not update the review.",
)


class TextBoundary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str


class TextSpanRecord(BaseModel):
    """Exact character bounds into the recorded readings, when the review stored them."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["DISPUTE_TEXT", "PARTIAL_AGREEMENT"]
    role: Literal["FULL", "ADOPTED", "UNRESOLVED"]
    gpt: TextBoundary
    grok: TextBoundary


class PageRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str
    reading_order: int = Field(ge=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    filename: str
    mime_type: Literal["image/png", "image/jpeg"]
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    derived_png_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class EvidenceRefRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    kind: str
    evidence_id: str
    page_id: str
    assessment: str | None = None
    window_kind: Literal["TARGETED", "FULL_PAGE_FALLBACK"] | None = None
    crop_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    exact_text_coordinates: Literal[False] = False


class LearningCandidate(BaseModel):
    """One recorded reading or span. Nothing here is a verified transcription."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^lc_[0-9a-f]{20}$")
    batch_id: str
    document_id: int = Field(ge=1)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    page_id: str
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    page_image_filename: str
    page_mime_type: Literal["image/png", "image/jpeg"]
    derived_png_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    region_id: str | None = None
    gpt_region_ids: list[str]
    grok_region_ids: list[str]
    dispute_id: str
    dispute_category: str
    learning_category: LearningCategory
    human_verification_status: Literal["NOT_VERIFIED"] = "NOT_VERIFIED"
    eligible_for_authoritative_learning: Literal[False] = False
    agreement_is_not_verification: Literal[True] = True
    model_supported: bool
    reading_role: Literal[
        "FULL_DISPUTE", "ADOPTED_SPAN", "UNRESOLVED_SPAN", "SUPERSEDED_READING"
    ]
    gpt_text: str
    grok_text: str
    dispute_gpt_text: str
    dispute_grok_text: str
    model_selected_reading: str | None = None
    superseded_text: str | None = None
    text_span: TextSpanRecord | None = None
    image_span_status: ImageSpanStatus
    image_span_limitation: str
    image_bbox: BBox | None = None
    reconciled_status: ReconciledStatus
    challenge_decision: str | None = None
    reconciliation_basis: str | None = None
    providers_agreed: bool | None = None
    gpt_decision: str | None = None
    grok_decision: str | None = None
    gpt_third_reading: str | None = None
    grok_third_reading: str | None = None
    evidence: list[EvidenceRefRecord] = Field(default_factory=list)
    partial_agreement_role: Literal["ADOPTED", "UNRESOLVED"] | None = None
    closure_status: str | None = None
    related_candidate_id: str | None = None
    comparison_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    review_report_sha256: str = Field(pattern=SHA256_PATTERN)
    gpt_output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    grok_output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _not_authoritative(self) -> LearningCandidate:
        if self.learning_category == "CORRECTION_CANDIDATE":
            raise ValueError(
                "a correction candidate requires a human approval record, which this export does not read"
            )
        if self.model_supported and self.human_verification_status != "NOT_VERIFIED":
            raise ValueError("model support is not human verification")
        return self


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    document_id: int = Field(ge=1)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    review_report_sha256: str = Field(pattern=SHA256_PATTERN)
    comparison_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    quality: str
    review_completion: Literal["completed"]
    sealed_bundle_checked: bool
    candidate_count: int = Field(ge=0)
    model_supported_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    rejected_or_superseded_count: int = Field(ge=0)
    pages: list[PageRef]


class OmissionCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nonsubstantive_punctuation: int = Field(default=0, ge=0)
    nonsubstantive_coverage_uncertainty: int = Field(default=0, ge=0)
    other_nonsubstantive: int = Field(default=0, ge=0)
    other_nonsubstantive_dispute_ids: list[str] = Field(default_factory=list)


class LearningCandidateExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["learning-candidate-export-v1"] = (
        "learning-candidate-export-v1"
    )
    export_role: Literal["READ_ONLY_PREVIEW"] = "READ_ONLY_PREVIEW"
    persistent_learning_ledger_written: Literal[False] = False
    authoritative_learning_eligible_count: Literal[0] = 0
    human_verified_count: Literal[0] = 0
    correction_candidate_count: Literal[0] = 0
    limitations: list[str]
    documents: list[DocumentSummary]
    candidates: list[LearningCandidate]
    omitted: OmissionCounts


@dataclass(frozen=True)
class _LoadedDocument:
    summary: DocumentSummary
    candidates: list[LearningCandidate]
    omitted: OmissionCounts


@dataclass(frozen=True)
class _ImageSpan:
    status: ImageSpanStatus
    limitation: str
    bbox: BBox | None


@dataclass(frozen=True)
class _Context:
    batch_id: str
    document_id: int
    source_bundle: str
    report_sha: str
    comparison_sha: str | None
    gpt_output_sha: str | None
    grok_output_sha: str | None
    pages: dict[str, SourcePageRecord]
    crops: dict[str, CropRecord]
    closure: dict[str, str]
    region_boxes: dict[tuple[str, str, str], BBox]


def export_learning_candidates(
    review_dirs: Sequence[Path],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batches_dir: Path | None = None,
    document_ids: Sequence[int] | None = None,
    dry_run: bool = False,
) -> LearningCandidateExport:
    """Build the export. Write the two output files only when ``dry_run`` is false.

    Integrity failures raise before any output file is created. Review files,
    sealed bundles, and ledgers are only read.
    """

    if not review_dirs:
        raise ReviewUsageError("at least one reviews directory is required")
    roots = [reviews_root(path) for path in review_dirs]
    sealed = reviews_root(batches_dir) if batches_dir is not None else None
    loaded = [
        _load_document(reviews, doc_root, sealed)
        for reviews, doc_root in _discover(roots, document_ids)
    ]
    export = _assemble(loaded)
    if not dry_run:
        _write_export(output_dir, export, roots, sealed)
    return export


def render_review_markdown(export: LearningCandidateExport) -> str:
    """Human-readable preview. Punctuation disputes stay in the omission count."""

    lines = [
        "# Learning candidate review",
        "",
        "Read-only preview of completed transcription reviews. "
        "No reading is approved. No learning ledger is written.",
        "",
        f"- Schema: `{export.schema_version}`",
        f"- Candidates: {len(export.candidates)}",
        "- Human verification: `NOT_VERIFIED` on every candidate",
        "- Eligible for authoritative learning: 0",
        "- Correction candidates: 0",
        f"- Punctuation disputes omitted: {export.omitted.nonsubstantive_punctuation}",
        f"- Coverage-uncertainty rows omitted: {export.omitted.nonsubstantive_coverage_uncertainty}",
        "- Non-substantive unchallenged disputes are counted and are not candidate records.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in export.limitations)
    lines.append("")
    by_doc: dict[tuple[str, int], list[LearningCandidate]] = {}
    for candidate in export.candidates:
        by_doc.setdefault((candidate.batch_id, candidate.document_id), []).append(
            candidate
        )
    for summary in export.documents:
        lines.extend(
            _document_markdown(
                summary, by_doc.get((summary.batch_id, summary.document_id), [])
            )
        )
    if export.omitted.other_nonsubstantive:
        lines.append(
            f"Other non-substantive disputes omitted from candidates: {export.omitted.other_nonsubstantive}."
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Export from the command line. A dry run and a preview write nothing."""

    parser = argparse.ArgumentParser(
        prog="export-learning-candidates",
        add_help=False,
    )
    parser.add_argument("--reviews-dir", action="append", required=True)
    parser.add_argument("--batches-dir")
    parser.add_argument("--document-id", action="append", type=int)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise ReviewUsageError(
            "export-learning-candidates usage was not recognized. Nothing was written."
        ) from exc
    export = export_learning_candidates(
        [Path(path) for path in args.reviews_dir],
        output_dir=Path(args.output_dir),
        batches_dir=Path(args.batches_dir) if args.batches_dir else None,
        document_ids=args.document_id,
        dry_run=args.dry_run or args.preview,
    )
    if args.preview:
        print(render_review_markdown(export), end="")
    if args.dry_run or args.preview:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "documents": len(export.documents),
                    "candidates": len(export.candidates),
                    "human_verified_count": export.human_verified_count,
                    "authoritative_learning_eligible_count": (
                        export.authoritative_learning_eligible_count
                    ),
                    "persistent_learning_ledger_written": (
                        export.persistent_learning_ledger_written
                    ),
                },
                sort_keys=True,
            )
        )
        return 0
    print(str(Path(args.output_dir)))
    return 0


def _discover(
    review_dirs: Sequence[Path], document_ids: Sequence[int] | None
) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    seen: set[tuple[str, int]] = set()
    for reviews in review_dirs:
        batches = sorted(
            path for path in reviews.glob("batch_batch-*") if path.is_dir()
        )
        for batch in batches:
            documents = batch / "documents"
            if not documents.is_dir():
                continue
            for doc_root in sorted(
                path for path in documents.glob("document_*") if path.is_dir()
            ):
                suffix = doc_root.name.removeprefix("document_")
                if not suffix.isdigit():
                    raise ReviewIntegrityError(
                        f"review directory {doc_root.name} is not a document id"
                    )
                document_id = int(suffix)
                if document_ids is not None and document_id not in document_ids:
                    continue
                batch_id = batch.name.removeprefix("batch_")
                key = (batch_id, document_id)
                if key in seen:
                    raise ReviewIntegrityError(
                        f"document {document_id} in {batch_id} is listed twice"
                    )
                seen.add(key)
                found.append((reviews, doc_root))
    if document_ids is not None:
        missing = [
            document_id
            for document_id in document_ids
            if document_id not in {item[1] for item in seen}
        ]
        if missing:
            raise ReviewUsageError(
                "completed review not found for document "
                + ", ".join(str(item) for item in missing)
            )
    if not found:
        raise ReviewUsageError("no completed review matched the request")
    return found


def _load_document(
    reviews: Path, doc_root: Path, batches_dir: Path | None
) -> _LoadedDocument:
    batch_id = doc_root.parent.parent.name.removeprefix("batch_")
    document_id = int(doc_root.name.removeprefix("document_"))
    state = _checkpoint(reviews, batch_id)
    document = _require_finalized(state, document_id)
    manifest, report, report_sha = _final_artifacts(doc_root, document)
    _require_same(
        manifest.batch_id,
        batch_id,
        "review manifest batch does not match the directory",
    )
    _require_same(
        manifest.document_id,
        document_id,
        "review manifest document does not match the directory",
    )
    _require_same(
        report.batch_id,
        batch_id,
        "review report batch does not match the directory",
    )
    _require_same(
        report.document_id,
        document_id,
        "review report document does not match the directory",
    )
    if report.review_completion != "completed":
        raise ReviewIntegrityError("only a completed review can be exported")
    source = _source_manifest(doc_root, document)
    _require_same(
        source.source_bundle_sha256,
        manifest.source_bundle_sha256,
        "source bundle hash does not match the review manifest",
    )
    _require_same(
        document.source_bundle_sha256,
        manifest.source_bundle_sha256,
        "source bundle hash does not match the review checkpoint",
    )
    _verify_pages(doc_root, source, document)
    comparison, comparison_sha = _comparison(doc_root, manifest, document, report)
    gpt_sha, grok_sha, first_pass = _provider_outputs(
        doc_root, manifest, document, source
    )
    if comparison_sha is not None and (
        gpt_sha != comparison.gpt_output_sha256
        or grok_sha != comparison.grok_output_sha256
    ):
        raise ReviewIntegrityError("comparison names a different provider output")
    crops = _crops(doc_root, source)
    if batches_dir is not None:
        _verify_sealed(batches_dir, state, source)
    closure: dict[str, str] = {
        item.dispute_id: item.status
        for item in (
            report.challenge_closure.unresolved if report.challenge_closure else []
        )
    }
    context = _Context(
        batch_id=batch_id,
        document_id=document_id,
        source_bundle=manifest.source_bundle_sha256,
        report_sha=report_sha,
        comparison_sha=comparison_sha,
        gpt_output_sha=gpt_sha,
        grok_output_sha=grok_sha,
        pages={page.page_id: page for page in source.pages},
        crops=crops,
        closure=closure,
        region_boxes=_region_boxes(first_pass, source),
    )
    candidates, omitted = _candidates(context, comparison, report)
    summary = DocumentSummary(
        batch_id=batch_id,
        document_id=document_id,
        source_bundle_sha256=manifest.source_bundle_sha256,
        review_report_sha256=report_sha,
        comparison_sha256=comparison_sha,
        quality=report.quality.value,
        review_completion="completed",
        sealed_bundle_checked=batches_dir is not None,
        candidate_count=len(candidates),
        model_supported_count=sum(item.model_supported for item in candidates),
        unresolved_count=sum(
            item.learning_category == "UNRESOLVED_DISAGREEMENT" for item in candidates
        ),
        rejected_or_superseded_count=sum(
            item.learning_category == "REJECTED_OR_SUPERSEDED_READING"
            for item in candidates
        ),
        pages=[
            PageRef(
                page_id=page.page_id,
                reading_order=page.reading_order,
                sha256=page.sha256,
                filename=page.filename,
                mime_type=page.mime_type,
                width=page.width,
                height=page.height,
                derived_png_sha256=page.derived_png_sha256,
            )
            for page in source.pages
        ],
    )
    return _LoadedDocument(summary, candidates, omitted)


def _assemble(loaded: Sequence[_LoadedDocument]) -> LearningCandidateExport:
    omitted = OmissionCounts()
    candidates: list[LearningCandidate] = []
    documents: list[DocumentSummary] = []
    for item in loaded:
        documents.append(item.summary)
        candidates.extend(item.candidates)
        omitted.nonsubstantive_punctuation += item.omitted.nonsubstantive_punctuation
        omitted.nonsubstantive_coverage_uncertainty += (
            item.omitted.nonsubstantive_coverage_uncertainty
        )
        omitted.other_nonsubstantive += item.omitted.other_nonsubstantive
        omitted.other_nonsubstantive_dispute_ids.extend(
            item.omitted.other_nonsubstantive_dispute_ids
        )
    documents.sort(key=lambda item: (item.batch_id, item.document_id))
    candidates.sort(
        key=lambda item: (
            item.batch_id,
            item.document_id,
            item.page_id,
            item.dispute_id,
            _part_rank(item),
            item.candidate_id,
        )
    )
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            raise ReviewIntegrityError("learning candidate identifiers are not unique")
        seen.add(candidate.candidate_id)
    limitations = list(_LIMITATIONS)
    if any(not item.sealed_bundle_checked for item in documents):
        limitations.append(
            "A reviews directory was exported without a sealed batches directory. "
            "Review-local manifest and page hashes were checked. The sealed bundle was not re-read."
        )
    return LearningCandidateExport(
        limitations=limitations,
        documents=documents,
        candidates=candidates,
        omitted=omitted,
    )


def _checkpoint(reviews: Path, batch_id: str) -> ReviewBatchState:
    """Replay the ledger in memory and require the checkpoint to match.

    ``ReviewStore.load`` may write a checkpoint. This path does not.
    """

    store = ReviewStore(reviews, batch_id)
    if not store.checkpoint_path.is_file():
        raise ReviewIntegrityError("completed review checkpoint is missing")
    if not store.ledger_path.is_file():
        raise ReviewIntegrityError("completed review ledger is missing")
    try:
        replayed = _replay(batch_id, store.read_ledger())
        checkpoint = _read_checkpoint(store.checkpoint_path)
    except ReviewIntegrityError:
        raise
    except ValueError as exc:
        raise ReviewIntegrityError("review ledger replay failed") from exc
    if checkpoint != replayed:
        raise ReviewIntegrityError("review checkpoint does not match ledger replay")
    return checkpoint


def _require_finalized(
    state: ReviewBatchState, document_id: int
) -> DocumentReviewState:
    matches = [item for item in state.documents if item.document_id == document_id]
    if not matches:
        raise ReviewIntegrityError(
            f"document {document_id} is not in the review checkpoint"
        )
    document = matches[0]
    if not document.finalized:
        raise ReviewIntegrityError("an unfinished review cannot be exported")
    return document


def _final_artifacts(
    doc_root: Path, document: DocumentReviewState
) -> tuple[ReviewManifest, ReviewReport, str]:
    final_dir = doc_root / "final"
    paths = {
        name: final_dir / name
        for name in (
            "proposed_transcription.txt",
            "review_report.json",
            "review_manifest.json",
            "review_summary.md",
        )
    }
    for name, path in paths.items():
        if not path.is_file():
            raise ReviewIntegrityError(f"final artifact {name} is missing")
        digest = sha256_bytes(path.read_bytes())
        recorded = document.final_sha256.get(name)
        if recorded is None or recorded != digest:
            raise ReviewIntegrityError(
                f"final artifact {name} hash does not match the checkpoint"
            )
    manifest_bytes = paths["review_manifest.json"].read_bytes()
    report_bytes = paths["review_report.json"].read_bytes()
    try:
        manifest = ReviewManifest.model_validate_json(manifest_bytes)
        report = ReviewReport.model_validate_json(report_bytes)
    except ValueError as exc:
        raise ReviewIntegrityError(
            "final review artifact does not match the schema"
        ) from exc
    recorded_report = next(
        (
            item.sha256
            for item in manifest.artifacts
            if item.name == "review_report.json"
        ),
        None,
    )
    report_sha = sha256_bytes(report_bytes)
    if recorded_report != report_sha:
        raise ReviewIntegrityError(
            "review report hash does not match the review manifest"
        )
    for artifact in manifest.artifacts:
        artifact_path = paths.get(artifact.name)
        if (
            artifact_path is None
            or sha256_bytes(artifact_path.read_bytes()) != artifact.sha256
        ):
            raise ReviewIntegrityError(
                f"manifest artifact {artifact.name} hash does not match the file"
            )
    return manifest, report, report_sha


def _source_manifest(doc_root: Path, document: DocumentReviewState) -> SourceManifest:
    path = doc_root / "input" / "source_manifest.json"
    if not path.is_file():
        raise ReviewIntegrityError("source manifest is missing")
    try:
        source = SourceManifest.model_validate_json(path.read_bytes())
    except ValueError as exc:
        raise ReviewIntegrityError("source manifest does not match the schema") from exc
    if source.document_id != document.document_id:
        raise ReviewIntegrityError("source manifest is for a different document")
    return source


def _verify_pages(
    doc_root: Path, source: SourceManifest, document: DocumentReviewState
) -> None:
    by_id = {page.page_id: page for page in source.pages}
    if len(by_id) != len(source.pages):
        raise ReviewIntegrityError("source manifest repeats a page id")
    state_pages = {page.page_id: page for page in document.pages}
    if set(by_id) != set(state_pages):
        raise ReviewIntegrityError("source manifest pages do not match the checkpoint")
    for page_id, page in by_id.items():
        state_page = state_pages[page_id]
        if (
            state_page.sha256 != page.sha256
            or state_page.reading_order != page.reading_order
        ):
            raise ReviewIntegrityError("source page hash does not match the checkpoint")
        if (
            state_page.original_filename
            and state_page.original_filename != page.filename
        ):
            raise ReviewIntegrityError(
                "source page filename does not match the checkpoint"
            )
        if state_page.derived_png_sha256 != page.derived_png_sha256:
            raise ReviewIntegrityError(
                "derived page hash does not match the checkpoint"
            )
        _require_file_hash(
            doc_root / "input" / page.filename, page.sha256, page.filename
        )
        if page.derived_png_sha256 is not None:
            if page.derived_png_filename is None:
                raise ReviewIntegrityError("derived page hash has no filename")
            _require_file_hash(
                doc_root / "input" / page.derived_png_filename,
                page.derived_png_sha256,
                page.derived_png_filename,
            )


def _comparison(
    doc_root: Path,
    manifest: ReviewManifest,
    document: DocumentReviewState,
    report: ReviewReport,
) -> tuple[ComparisonFile, str | None]:
    path = doc_root / "comparison" / "comparison.json"
    if manifest.comparison_sha256 is None:
        if report.disputes:
            raise ReviewIntegrityError("disputes exist without a comparison hash")
        return ComparisonFile.model_construct(
            batch_id=manifest.batch_id,
            document_id=manifest.document_id,
            source_bundle_sha256=manifest.source_bundle_sha256,
            gpt_output_sha256="0" * 64,
            grok_output_sha256="0" * 64,
            page_originals=[],
            disputes=[],
            unresolved_text_page_ids=[],
        ), None
    if not path.is_file():
        raise ReviewIntegrityError("comparison file is missing")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != manifest.comparison_sha256:
        raise ReviewIntegrityError("comparison hash does not match the review manifest")
    if document.comparison_sha256 is not None and document.comparison_sha256 != digest:
        raise ReviewIntegrityError("comparison hash does not match the checkpoint")
    try:
        comparison = ComparisonFile.model_validate_json(data)
    except ValueError as exc:
        raise ReviewIntegrityError("comparison does not match the schema") from exc
    if comparison.source_bundle_sha256 != manifest.source_bundle_sha256:
        raise ReviewIntegrityError("comparison names a different source bundle")
    if comparison.document_id != manifest.document_id:
        raise ReviewIntegrityError("comparison is for a different document")
    if [item.model_dump(mode="json") for item in report.disputes] != [
        item.model_dump(mode="json") for item in comparison.disputes
    ]:
        raise ReviewIntegrityError("review report disputes do not match the comparison")
    return comparison, digest


def _provider_outputs(
    doc_root: Path,
    manifest: ReviewManifest,
    document: DocumentReviewState,
    source: SourceManifest,
) -> tuple[str | None, str | None, dict[str, FirstPassOutput]]:
    recorded = {
        runtime.task_id: runtime
        for runtime in (
            *document.first_pass,
            *(
                provider
                for challenge in document.challenge_rounds
                for provider in challenge.providers
            ),
        )
        if runtime.task_id is not None
    }
    hashes: dict[str, str] = {}
    first_pass: dict[str, FirstPassOutput] = {}
    for row in manifest.providers:
        runtime = recorded.get(row.task_id)
        if runtime is not None and runtime.output_sha256 not in {
            None,
            row.output_sha256,
        }:
            raise ReviewIntegrityError(
                "provider output hash does not match the checkpoint"
            )
        if runtime is not None and runtime.output_relative_path:
            relative = runtime.output_relative_path
        else:
            match = _TASK_ID.fullmatch(row.task_id)
            if match is None:
                raise ReviewIntegrityError("provider task id is not a review task id")
            relative = output_relative_path(
                match.group(3),
                match.group(5),
                int(match.group(4)),
                int(match.group(6)),
            )
        path = _contained_file(doc_root, relative)
        data = _require_file_hash(path, row.output_sha256, relative)
        model = _check_provider_file(
            data, row.provider, row.task_id, source.source_bundle_sha256, row.stage
        )
        if row.stage == "first_pass":
            if not isinstance(model, FirstPassOutput):
                raise ReviewIntegrityError(
                    "first-pass manifest entry is not a first-pass output"
                )
            hashes[row.provider] = row.output_sha256
            first_pass[row.provider] = model
    return hashes.get("gpt"), hashes.get("grok"), first_pass


def _check_provider_file(
    data: bytes, provider: str, task_id: str, source_bundle: str, stage: str
) -> FirstPassOutput | ChallengeOutput | ResearchOutput:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ReviewIntegrityError("provider output is not JSON") from exc
    if not isinstance(payload, dict):
        raise ReviewIntegrityError("provider output is not an object")
    if payload.get("provider") != provider or payload.get("task_id") != task_id:
        raise ReviewIntegrityError(
            "provider output identity does not match the manifest"
        )
    if payload.get("source_bundle_sha256") != source_bundle:
        raise ReviewIntegrityError("provider output names a different source bundle")
    if payload.get("stage") != stage:
        raise ReviewIntegrityError("provider output stage does not match the manifest")
    try:
        if stage == "first_pass":
            return FirstPassOutput.model_validate(payload)
        if stage == "challenge":
            return ChallengeOutput.model_validate(payload)
        if stage == "research":
            return ResearchOutput.model_validate(payload)
        raise ReviewIntegrityError("provider output stage is not a review stage")
    except ReviewIntegrityError:
        raise
    except ValueError as exc:
        raise ReviewIntegrityError(
            "provider output does not match the review schema"
        ) from exc


def _crops(doc_root: Path, source: SourceManifest) -> dict[str, CropRecord]:
    pages = {page.page_id: page for page in source.pages}
    found: dict[str, CropRecord] = {}
    for path in sorted(doc_root.glob("challenge_round_*/input/crops/*.json")):
        try:
            crop = CropRecord.model_validate_json(path.read_bytes())
        except ValueError as exc:
            raise ReviewIntegrityError(
                "challenge crop does not match the schema"
            ) from exc
        page = pages.get(crop.page_id)
        if page is None or page.sha256 != crop.page_sha256:
            raise ReviewIntegrityError(
                "challenge crop names a page that is not the source image"
            )
        if crop.crop_sha256 is not None:
            image_name = crop.image_filename or f"{crop.crop_id}.png"
            _require_file_hash(path.parent / image_name, crop.crop_sha256, image_name)
        if crop.crop_id in found and found[crop.crop_id] != crop:
            raise ReviewIntegrityError("two challenge crops share an id and differ")
        found[crop.crop_id] = crop
    return found


def _verify_sealed(
    batches_dir: Path, state: ReviewBatchState, source: SourceManifest
) -> None:
    batch_dir = batches_dir / state.batch_id
    manifest = verify_batch_bundle(batch_dir)
    batch_hash = sha256_bytes((batch_dir / "batch_manifest.json").read_bytes())
    if state.source_batch_manifest_sha256 not in {None, batch_hash}:
        raise ReviewIntegrityError(
            "sealed batch manifest hash does not match the checkpoint"
        )
    if manifest.batch_id != state.batch_id:
        raise ReviewIntegrityError("sealed batch manifest is for a different batch")
    document_dir = batch_dir / "documents" / str(source.document_id)
    document_manifest = document_dir / "document_manifest.json"
    if not document_manifest.is_file():
        raise ReviewIntegrityError("sealed document manifest is missing")
    if sha256_bytes(document_manifest.read_bytes()) != source.source_bundle_sha256:
        raise ReviewIntegrityError("sealed document hash does not match the review")
    from transcription_review_agent.bundling.manifest import DocumentBundleManifest

    try:
        sealed = DocumentBundleManifest.model_validate_json(
            document_manifest.read_bytes()
        )
    except ValueError as exc:
        raise ReviewIntegrityError(
            "sealed document manifest does not match the schema"
        ) from exc
    images = [
        item
        for item in sorted(sealed.sources, key=lambda item: item.order_index)
        if item.mime_type in {"image/png", "image/jpeg"}
    ]
    if [item.sha256 for item in images] != [page.sha256 for page in source.pages]:
        raise ReviewIntegrityError(
            "sealed page hashes do not match the source manifest"
        )


def _region_boxes(
    first_pass: dict[str, FirstPassOutput], source: SourceManifest
) -> dict[tuple[str, str, str], BBox]:
    """Copy in-page region boxes from manifest-verified first-pass outputs.

    A box outside the sealed page is ignored. Nothing is clipped or invented.
    Outputs that the manifest did not hash are not read.
    """

    pages = {page.page_id: page for page in source.pages}
    found: dict[tuple[str, str, str], BBox] = {}
    for provider in ("gpt", "grok"):
        output = first_pass.get(provider)
        if output is None:
            continue
        for page in output.pages:
            source_page = pages.get(page.page_id)
            if source_page is None:
                continue
            frame = BBox(x=0, y=0, width=source_page.width, height=source_page.height)
            for region in page.regions:
                if region.bbox is None or not frame.contains(region.bbox):
                    continue
                found[(provider, page.page_id, region.region_id)] = region.bbox
    return found


def _candidates(
    context: _Context, comparison: ComparisonFile, report: ReviewReport
) -> tuple[list[LearningCandidate], OmissionCounts]:
    decisions = {item.dispute_id: item for item in report.decisions}
    if len(decisions) != len(report.decisions):
        raise ReviewIntegrityError("review report repeats a reconciled dispute")
    omitted = OmissionCounts()
    built: list[tuple[int, int, LearningCandidate]] = []
    order = {page.page_id: page.reading_order for page in context.pages.values()}
    for index, dispute in enumerate(comparison.disputes):
        if dispute.page_id not in context.pages:
            raise ReviewIntegrityError(
                "dispute names a page that is not in the source manifest"
            )
        decision = decisions.get(dispute.dispute_id)
        if not dispute.substantive and decision is None:
            _count_omission(omitted, dispute)
            continue
        span = _image_span(context, dispute, decision)
        group = _dispute_candidates(context, dispute, decision, span)
        page_order = order[dispute.page_id]
        built.extend((page_order, index, item) for item in group)
    unknown = [
        item.dispute_id
        for item in report.decisions
        if item.dispute_id not in {row.dispute_id for row in comparison.disputes}
    ]
    if unknown:
        raise ReviewIntegrityError(
            "a reconciled decision names a dispute that is not in the comparison"
        )
    built.sort(
        key=lambda item: (item[0], item[1], _part_rank(item[2]), item[2].candidate_id)
    )
    return [item[2] for item in built], omitted


def _count_omission(omitted: OmissionCounts, dispute: Dispute) -> None:
    category = dispute.category.value
    if category == "PUNCTUATION":
        omitted.nonsubstantive_punctuation += 1
    elif category == "COVERAGE_UNCERTAINTY":
        omitted.nonsubstantive_coverage_uncertainty += 1
    else:
        omitted.other_nonsubstantive += 1
        omitted.other_nonsubstantive_dispute_ids.append(dispute.dispute_id)


def _dispute_candidates(
    context: _Context,
    dispute: Dispute,
    decision: ReconciledDecision | None,
    span: _ImageSpan,
) -> list[LearningCandidate]:
    status = _reconciled_status(decision)
    if decision is not None and decision.result_class == "AUTOMATICALLY_SUPPORTED":
        selected = decision.selected_reading
        if selected is None:
            raise ReviewIntegrityError(
                "an automatically supported decision has no selected reading"
            )
        primary = _make_candidate(
            context,
            dispute,
            decision,
            span,
            part="primary",
            part_index=0,
            learning_category="MODEL_SUPPORTED_READING",
            model_supported=True,
            reading_role="FULL_DISPUTE",
            gpt_text=dispute.gpt_text,
            grok_text=dispute.grok_text,
            model_selected_reading=selected,
            superseded_text=None,
            text_span=_full_span(dispute),
            reconciled_status=status,
            partial_role=None,
            related_candidate_id=None,
        )
        rows = [primary]
        rows.extend(
            _rejected(context, dispute, decision, span, primary.candidate_id, selected)
        )
        rows.extend(
            _superseded(
                context, dispute, decision, span, primary.candidate_id, selected
            )
        )
        return rows
    primary = _make_candidate(
        context,
        dispute,
        decision,
        span,
        part="primary",
        part_index=0,
        learning_category="UNRESOLVED_DISAGREEMENT",
        model_supported=False,
        reading_role="FULL_DISPUTE",
        gpt_text=dispute.gpt_text,
        grok_text=dispute.grok_text,
        model_selected_reading=None,
        superseded_text=None,
        text_span=_full_span(dispute),
        reconciled_status=status,
        partial_role=None,
        related_candidate_id=None,
    )
    rows = [primary]
    if decision is not None and decision.partial_agreements:
        for index, partial in enumerate(decision.partial_agreements):
            rows.append(
                _partial_candidate(
                    context, dispute, decision, span, primary, index, partial
                )
            )
    if decision is not None:
        rows.extend(
            _superseded(
                context,
                dispute,
                decision,
                span,
                primary.candidate_id,
                decision.selected_reading,
            )
        )
    return rows


def _rejected(
    context: _Context,
    dispute: Dispute,
    decision: ReconciledDecision,
    span: _ImageSpan,
    related_id: str,
    selected: str,
) -> list[LearningCandidate]:
    readings = [dispute.gpt_text, dispute.grok_text]
    if decision.gpt_third_reading:
        readings.append(decision.gpt_third_reading)
    if decision.grok_third_reading:
        readings.append(decision.grok_third_reading)
    unique: list[str] = []
    for reading in readings:
        if reading != selected and reading not in unique:
            unique.append(reading)
    return [
        _make_candidate(
            context,
            dispute,
            decision,
            span,
            part="rejected",
            part_index=index,
            learning_category="REJECTED_OR_SUPERSEDED_READING",
            model_supported=False,
            reading_role="SUPERSEDED_READING",
            gpt_text=dispute.gpt_text,
            grok_text=dispute.grok_text,
            model_selected_reading=None,
            superseded_text=reading,
            text_span=None,
            reconciled_status="AUTOMATICALLY_SUPPORTED",
            partial_role=None,
            related_candidate_id=related_id,
        )
        for index, reading in enumerate(unique)
    ]


def _superseded(
    context: _Context,
    dispute: Dispute,
    decision: ReconciledDecision,
    span: _ImageSpan,
    related_id: str,
    selected: str | None,
) -> list[LearningCandidate]:
    rows: list[LearningCandidate] = []
    for index, earlier in enumerate(decision.other_round_decisions or []):
        earlier_selected = earlier.selected_reading
        if (
            earlier.result_class != "AUTOMATICALLY_SUPPORTED"
            or earlier_selected is None
        ):
            continue
        if earlier_selected == selected:
            continue
        rows.append(
            _make_candidate(
                context,
                dispute,
                earlier,
                span,
                part="superseded",
                part_index=index,
                learning_category="REJECTED_OR_SUPERSEDED_READING",
                model_supported=False,
                reading_role="SUPERSEDED_READING",
                gpt_text=dispute.gpt_text,
                grok_text=dispute.grok_text,
                model_selected_reading=None,
                superseded_text=earlier_selected,
                text_span=None,
                reconciled_status="AUTOMATICALLY_SUPPORTED",
                partial_role=None,
                related_candidate_id=related_id,
            )
        )
    return rows


def _partial_candidate(
    context: _Context,
    dispute: Dispute,
    decision: ReconciledDecision,
    span: _ImageSpan,
    primary: LearningCandidate,
    index: int,
    partial: PartialAgreementSpan,
) -> LearningCandidate:
    adopted = partial.role == "ADOPTED"
    return _make_candidate(
        context,
        dispute,
        decision,
        span,
        part="partial",
        part_index=index,
        learning_category=(
            "MODEL_SUPPORTED_READING" if adopted else "UNRESOLVED_DISAGREEMENT"
        ),
        model_supported=adopted,
        reading_role="ADOPTED_SPAN" if adopted else "UNRESOLVED_SPAN",
        gpt_text=partial.gpt_text,
        grok_text=partial.grok_text,
        model_selected_reading=partial.gpt_text if adopted else None,
        superseded_text=None,
        text_span=TextSpanRecord(
            scope="PARTIAL_AGREEMENT",
            role="ADOPTED" if adopted else "UNRESOLVED",
            gpt=TextBoundary(
                start=partial.gpt_start, end=partial.gpt_end, text=partial.gpt_text
            ),
            grok=TextBoundary(
                start=partial.grok_start, end=partial.grok_end, text=partial.grok_text
            ),
        ),
        reconciled_status="PARTIAL_AGREEMENT",
        partial_role=partial.role,
        related_candidate_id=primary.candidate_id,
        evidence=_span_evidence(context, dispute, partial),
    )


def _make_candidate(
    context: _Context,
    dispute: Dispute,
    decision: ReconciledDecision | None,
    span: _ImageSpan,
    *,
    part: CandidatePart,
    part_index: int,
    learning_category: LearningCategory,
    model_supported: bool,
    reading_role: Literal[
        "FULL_DISPUTE", "ADOPTED_SPAN", "UNRESOLVED_SPAN", "SUPERSEDED_READING"
    ],
    gpt_text: str,
    grok_text: str,
    model_selected_reading: str | None,
    superseded_text: str | None,
    text_span: TextSpanRecord | None,
    reconciled_status: ReconciledStatus,
    partial_role: Literal["ADOPTED", "UNRESOLVED"] | None,
    related_candidate_id: str | None,
    evidence: list[EvidenceRefRecord] | None = None,
) -> LearningCandidate:
    page = context.pages[dispute.page_id]
    identity = {
        "batch_id": context.batch_id,
        "dispute_gpt_text": dispute.gpt_text,
        "dispute_grok_text": dispute.grok_text,
        "dispute_id": dispute.dispute_id,
        "document_id": context.document_id,
        "gpt_text": gpt_text,
        "grok_text": grok_text,
        "learning_category": learning_category,
        "model_selected_reading": model_selected_reading,
        "part": part,
        "part_index": part_index,
        "source_bundle_sha256": context.source_bundle,
        "superseded_text": superseded_text,
        "v": 1,
    }
    return LearningCandidate(
        candidate_id="lc_"
        + sha256_text(json.dumps(identity, sort_keys=True, separators=(",", ":")))[:20],
        batch_id=context.batch_id,
        document_id=context.document_id,
        source_bundle_sha256=context.source_bundle,
        page_id=dispute.page_id,
        page_image_sha256=page.sha256,
        page_image_filename=page.filename,
        page_mime_type=page.mime_type,
        derived_png_sha256=page.derived_png_sha256,
        region_id=dispute.region_id,
        gpt_region_ids=list(dispute.gpt_region_ids),
        grok_region_ids=list(dispute.grok_region_ids),
        dispute_id=dispute.dispute_id,
        dispute_category=dispute.category.value,
        learning_category=learning_category,
        model_supported=model_supported,
        reading_role=reading_role,
        gpt_text=gpt_text,
        grok_text=grok_text,
        dispute_gpt_text=dispute.gpt_text,
        dispute_grok_text=dispute.grok_text,
        model_selected_reading=model_selected_reading,
        superseded_text=superseded_text,
        text_span=text_span,
        image_span_status=span.status,
        image_span_limitation=span.limitation,
        image_bbox=span.bbox,
        reconciled_status=reconciled_status,
        challenge_decision=None if decision is None else decision.decision,
        reconciliation_basis=None if decision is None else decision.basis,
        providers_agreed=None if decision is None else decision.agreed,
        gpt_decision=None if decision is None else decision.gpt_decision,
        grok_decision=None if decision is None else decision.grok_decision,
        gpt_third_reading=None if decision is None else decision.gpt_third_reading,
        grok_third_reading=None if decision is None else decision.grok_third_reading,
        evidence=(
            _evidence(context, decision, dispute) if evidence is None else evidence
        ),
        partial_agreement_role=partial_role,
        closure_status=context.closure.get(dispute.dispute_id),
        related_candidate_id=related_candidate_id,
        comparison_sha256=context.comparison_sha,
        review_report_sha256=context.report_sha,
        gpt_output_sha256=context.gpt_output_sha,
        grok_output_sha256=context.grok_output_sha,
    )


def _full_span(dispute: Dispute) -> TextSpanRecord:
    return TextSpanRecord(
        scope="DISPUTE_TEXT",
        role="FULL",
        gpt=TextBoundary(start=0, end=len(dispute.gpt_text), text=dispute.gpt_text),
        grok=TextBoundary(start=0, end=len(dispute.grok_text), text=dispute.grok_text),
    )


def _reconciled_status(decision: ReconciledDecision | None) -> ReconciledStatus:
    if decision is None:
        return "NOT_CHALLENGED"
    if decision.result_class is None:
        return "LEGACY_UNCLASSIFIED"
    if decision.result_class == "AUTOMATICALLY_SUPPORTED":
        return "AUTOMATICALLY_SUPPORTED"
    return "UNRESOLVED"


def _span_evidence(
    context: _Context, dispute: Dispute, partial: PartialAgreementSpan
) -> list[EvidenceRefRecord]:
    """Copy the span's own evidence, not the parent decision's."""

    if partial.dispute_id != dispute.dispute_id:
        raise ReviewIntegrityError("partial agreement is for a different dispute")
    return [
        _evidence_row(
            context,
            "gpt",
            partial.gpt_evidence_ref,
            partial.gpt_evidence_assessment,
            dispute,
        ),
        _evidence_row(
            context,
            "grok",
            partial.grok_evidence_ref,
            partial.grok_evidence_assessment,
            dispute,
        ),
    ]


def _evidence(
    context: _Context, decision: ReconciledDecision | None, dispute: Dispute
) -> list[EvidenceRefRecord]:
    if decision is None:
        return []
    rows: list[EvidenceRefRecord] = []
    pairs: tuple[tuple[ProviderName, EvidenceReference | None, str | None], ...] = (
        ("gpt", decision.gpt_evidence_ref, decision.gpt_evidence_assessment),
        ("grok", decision.grok_evidence_ref, decision.grok_evidence_assessment),
    )
    for provider, ref, assessment in pairs:
        if ref is None:
            continue
        rows.append(_evidence_row(context, provider, ref, assessment, dispute))
    return rows


def _evidence_row(
    context: _Context,
    provider: ProviderName,
    ref: EvidenceReference,
    assessment: str | None,
    dispute: Dispute,
) -> EvidenceRefRecord:
    if ref.page_id != dispute.page_id:
        raise ReviewIntegrityError(
            "evidence reference is on a different page from the dispute"
        )
    crop = _crop_for_reference(context, ref)
    if crop is not None:
        _require_crop_association(crop, ref, dispute)
    return EvidenceRefRecord(
        provider=provider,
        kind=ref.kind,
        evidence_id=ref.evidence_id,
        page_id=ref.page_id,
        assessment=assessment,
        window_kind=None if crop is None else crop.window_kind,
        crop_sha256=None if crop is None else crop.crop_sha256,
    )


def _crop_for_reference(context: _Context, ref: EvidenceReference) -> CropRecord | None:
    """Return the recorded crop this reference names, if the review stored one.

    A missing crop is not a contradiction. Legacy decisions also omit the
    reference entirely and never reach this lookup.
    """

    found: list[CropRecord] = []
    for crop in context.crops.values():
        if ref.kind == "GROUP":
            if crop.group_id == ref.evidence_id:
                found.append(crop)
        elif crop.crop_id == ref.evidence_id:
            found.append(crop)
    if not found:
        return None
    if any(item != found[0] for item in found[1:]):
        raise ReviewIntegrityError("evidence reference matches contradictory crops")
    return found[0]


def _require_crop_association(
    crop: CropRecord, ref: EvidenceReference, dispute: Dispute
) -> None:
    if crop.page_id != ref.page_id or crop.page_id != dispute.page_id:
        raise ReviewIntegrityError(
            "evidence reference page does not match the challenge crop"
        )
    if ref.kind == "CROP" and crop.window_kind != "TARGETED":
        raise ReviewIntegrityError("a crop evidence reference is not a targeted window")
    if ref.kind == "FULL_PAGE_FALLBACK" and crop.window_kind != "FULL_PAGE_FALLBACK":
        raise ReviewIntegrityError(
            "a full-page evidence reference is not a full-page fallback"
        )
    if ref.kind == "GROUP" and crop.group_id != ref.evidence_id:
        raise ReviewIntegrityError(
            "a group evidence reference does not name that group"
        )
    if crop.dispute_id is not None and crop.dispute_id != dispute.dispute_id:
        raise ReviewIntegrityError("challenge crop belongs to a different dispute")
    if crop.dispute_ids is not None and dispute.dispute_id not in crop.dispute_ids:
        raise ReviewIntegrityError("dispute is not a member of the evidence crop")


def _image_span(
    context: _Context,
    dispute: Dispute,
    decision: ReconciledDecision | None,
) -> _ImageSpan:
    boxes: list[BBox] = []
    for provider, region_ids in (
        (
            "gpt",
            list(dispute.gpt_region_ids)
            or ([dispute.region_id] if dispute.region_id else []),
        ),
        (
            "grok",
            list(dispute.grok_region_ids)
            or ([dispute.region_id] if dispute.region_id else []),
        ),
    ):
        for region_id in region_ids:
            box = context.region_boxes.get((provider, dispute.page_id, region_id))
            if box is not None and box not in boxes:
                boxes.append(box)
    if len(boxes) == 1:
        return _ImageSpan(
            "RECORDED_REGION_BBOX",
            "The box is the region rectangle recorded on the first-pass output. It is not a character span.",
            boxes[0],
        )
    if len(boxes) > 1:
        return _ImageSpan(
            "CONFLICTING_REGION_BBOX",
            "The providers recorded different region boxes. Neither box was chosen.",
            None,
        )
    targeted = _targeted_windows(context, dispute, decision)
    if len(targeted) == 1:
        return _ImageSpan(
            "RECORDED_TARGETED_WINDOW",
            "The box is the recorded targeted challenge window. It is not a character span.",
            targeted[0],
        )
    if len(targeted) > 1:
        return _ImageSpan(
            "CONFLICTING_REGION_BBOX",
            "The challenge windows disagree. Neither box was chosen.",
            None,
        )
    if _full_page_only(context, decision):
        return _ImageSpan(
            "UNAVAILABLE",
            "The challenge window is a full-page fallback. It does not locate the disputed words.",
            None,
        )
    return _ImageSpan(
        "UNAVAILABLE",
        "No region box or targeted window is recorded for this dispute.",
        None,
    )


def _targeted_windows(
    context: _Context,
    dispute: Dispute,
    decision: ReconciledDecision | None,
) -> list[BBox]:
    if decision is None:
        return []
    page = context.pages[dispute.page_id]
    found: list[BBox] = []
    for ref in (decision.gpt_evidence_ref, decision.grok_evidence_ref):
        if ref is None:
            continue
        crop = context.crops.get(ref.evidence_id)
        if crop is None or crop.window_kind != "TARGETED":
            continue
        if _fills_page(crop.target_bbox, page):
            continue
        frame = BBox(x=0, y=0, width=page.width, height=page.height)
        if not frame.contains(crop.target_bbox):
            continue
        if crop.target_bbox not in found:
            found.append(crop.target_bbox)
    return found


def _full_page_only(context: _Context, decision: ReconciledDecision | None) -> bool:
    if decision is None:
        return False
    saw = False
    for ref in (decision.gpt_evidence_ref, decision.grok_evidence_ref):
        if ref is None:
            continue
        crop = context.crops.get(ref.evidence_id)
        if crop is None:
            continue
        saw = True
        if crop.window_kind != "FULL_PAGE_FALLBACK":
            return False
    return saw


def _fills_page(box: BBox, page: SourcePageRecord) -> bool:
    return (
        box.x == 0
        and box.y == 0
        and box.width == page.width
        and box.height == page.height
    )


def _part_rank(candidate: LearningCandidate) -> int:
    if candidate.reading_role == "FULL_DISPUTE":
        return 0
    if candidate.reading_role in {"ADOPTED_SPAN", "UNRESOLVED_SPAN"}:
        return 1
    return 2


def _document_markdown(
    summary: DocumentSummary, candidates: Sequence[LearningCandidate]
) -> list[str]:
    lines = [
        f"## Document {summary.document_id}",
        "",
        f"- Batch: `{summary.batch_id}`",
        f"- Source bundle: `{summary.source_bundle_sha256}`",
        f"- Review report: `{summary.review_report_sha256}`",
        f"- Quality: {summary.quality}",
        f"- Sealed bundle re-read: {'yes' if summary.sealed_bundle_checked else 'no'}",
        f"- Candidates: {summary.candidate_count}",
        f"- Model-supported, not human-approved: {summary.model_supported_count}",
        f"- Unresolved: {summary.unresolved_count}",
        f"- Rejected or superseded: {summary.rejected_or_superseded_count}",
        "",
        "Pages:",
        "",
    ]
    for page in summary.pages:
        lines.append(
            f"- `{page.page_id}` `{page.filename}` `{page.sha256}` "
            f"{page.width}x{page.height}"
        )
    lines.append("")
    sections = (
        (
            "Model-supported readings",
            "These spans are dual-model agreements. They are not human-approved examples.",
            [
                item
                for item in candidates
                if item.learning_category == "MODEL_SUPPORTED_READING"
            ],
        ),
        (
            "Rejected or superseded readings",
            "A model-selected alternative set these readings aside. That is not a correction.",
            [
                item
                for item in candidates
                if item.learning_category == "REJECTED_OR_SUPERSEDED_READING"
            ],
        ),
        (
            "Unresolved challenged disputes",
            "Includes legacy decisions that predate evidence binding and partial agreements.",
            [
                item
                for item in candidates
                if item.learning_category == "UNRESOLVED_DISAGREEMENT"
                and item.reconciled_status != "NOT_CHALLENGED"
            ],
        ),
        (
            "Other substantive disagreements",
            "Wording, names, numbers, deletions, insertions, and missing content that were not challenged.",
            [
                item
                for item in candidates
                if item.learning_category == "UNRESOLVED_DISAGREEMENT"
                and item.reconciled_status == "NOT_CHALLENGED"
                and item.dispute_category in _MARKDOWN_CATEGORIES
            ],
        ),
    )
    shown: set[str] = set()
    for title, note, rows in sections:
        visible = [item for item in rows if item.candidate_id not in shown]
        if not visible:
            continue
        lines.extend([f"### {title}", "", note, ""])
        for item in visible:
            shown.add(item.candidate_id)
            lines.extend(_candidate_lines(item))
    hidden = [
        item
        for item in candidates
        if item.candidate_id not in shown and item.dispute_category == "PUNCTUATION"
    ]
    if hidden:
        lines.append(
            f"Substantive punctuation disputes kept in JSON only: {len(hidden)}."
        )
        lines.append("")
    return lines


def _candidate_lines(candidate: LearningCandidate) -> list[str]:
    lines = [
        f"- `{candidate.candidate_id}` {candidate.page_id} {candidate.dispute_category} "
        f"`{candidate.dispute_id}`",
        f"  - Learning category: {candidate.learning_category}",
        f"  - Human verification: {candidate.human_verification_status}",
        f"  - Reconciled status: {candidate.reconciled_status}",
        f"  - Image span: {candidate.image_span_status}. {candidate.image_span_limitation}",
    ]
    if candidate.image_bbox is not None:
        box = candidate.image_bbox
        lines.append(
            f"  - Recorded box: x={box.x} y={box.y} width={box.width} height={box.height}"
        )
    if (
        candidate.text_span is not None
        and candidate.text_span.scope == "PARTIAL_AGREEMENT"
    ):
        span = candidate.text_span
        lines.append(
            f"  - Text span: GPT {span.gpt.start}:{span.gpt.end} / "
            f"Grok {span.grok.start}:{span.grok.end} ({span.role})"
        )
    lines.append(f"  - GPT: {_clip(candidate.gpt_text)}")
    lines.append(f"  - Grok: {_clip(candidate.grok_text)}")
    if candidate.model_selected_reading is not None:
        lines.append(
            f"  - Model-selected reading: {_clip(candidate.model_selected_reading)}"
        )
    if candidate.superseded_text is not None:
        lines.append(f"  - Set aside: {_clip(candidate.superseded_text)}")
    if candidate.evidence:
        cited = ", ".join(
            f"{item.provider} {item.kind} `{item.evidence_id}`"
            for item in candidate.evidence
        )
        lines.append(f"  - Evidence: {cited}")
    if candidate.challenge_decision is not None:
        lines.append(
            f"  - Decisions: record {candidate.challenge_decision}; "
            f"GPT {candidate.gpt_decision}; Grok {candidate.grok_decision}; "
            f"basis {candidate.reconciliation_basis}"
        )
    lines.append("")
    return lines


def _clip(text: str) -> str:
    if text == "":
        return "∅"
    if len(text) <= 240:
        return text.replace("\n", "↵")
    return text[:240].replace("\n", "↵") + "… [exact text is in the JSON]"


def _write_export(
    output_dir: Path,
    export: LearningCandidateExport,
    review_dirs: Sequence[Path],
    batches_dir: Path | None,
) -> None:
    resolved = output_dir.resolve()
    protected = list(review_dirs)
    if batches_dir is not None:
        protected.append(batches_dir)
    for root in protected:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            raise ReviewPathError(
                "refusing to write the export inside a review or sealed bundle"
            )
        if resolved in root_resolved.parents:
            raise ReviewPathError(
                "refusing to write the export over a review or sealed bundle"
            )
    files = {
        JSON_NAME: canonical_json(export),
        MARKDOWN_NAME: render_review_markdown(export),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _publish_pair(output_dir, files)


def _publish_pair(directory: Path, files: dict[str, str]) -> None:
    """Install both preview files, or leave the previous pair untouched.

    The files are written in a staging directory and moved into place only
    after both exist. A failure while moving restores the previous pair.
    """

    names = (JSON_NAME, MARKDOWN_NAME)
    staging = Path(
        tempfile.mkdtemp(prefix=".learning-candidate-export-", dir=directory)
    )
    try:
        for name in names:
            _write_fsync(staging / name, files[name])
        previous = staging / "previous"
        previous.mkdir()
        published: list[str] = []
        try:
            for name in names:
                target = directory / name
                if target.exists():
                    os.replace(target, previous / name)
                os.replace(staging / name, target)
                published.append(name)
        except Exception:
            for name in reversed(published):
                target = directory / name
                if target.exists():
                    target.unlink()
                backup = previous / name
                if backup.exists():
                    os.replace(backup, target)
            for name in names:
                target = directory / name
                backup = previous / name
                if not target.exists() and backup.exists():
                    os.replace(backup, target)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_fsync(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_file_hash(path: Path, expected: str, label: str) -> bytes:
    if not path.is_file():
        raise ReviewIntegrityError(f"missing {label}")
    data = path.read_bytes()
    if sha256_bytes(data) != expected:
        raise ReviewIntegrityError(f"{label} hash does not match the manifest")
    return data


def _contained_file(doc_root: Path, relative: str) -> Path:
    safe = assert_safe_relative_path(relative)
    return contained(doc_root, doc_root / safe)


def _require_same(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ReviewIntegrityError(message)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewUsageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except ReviewError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
