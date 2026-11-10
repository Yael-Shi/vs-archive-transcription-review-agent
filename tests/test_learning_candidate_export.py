"""Read-only learning-candidate export."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
    ReviewUsageError,
)
from transcription_review_agent.review.learning_export import (
    JSON_NAME,
    MARKDOWN_NAME,
    _Context,
    _ImageSpan,
    _dispute_candidates,
    export_learning_candidates,
    main,
)
from transcription_review_agent.review.models import (
    BBox,
    ComparisonFile,
    CoverageReport,
    CropRecord,
    Dispute,
    DisputeCategory,
    EvidenceReference,
    FirstPassOutput,
    LineReading,
    ManifestArtifact,
    ManifestProvider,
    PageOriginals,
    PageTranscription,
    PartialAgreementSpan,
    QualityLabel,
    ReconciledDecision,
    ReviewManifest,
    ReviewReport,
    SourceManifest,
    SourcePageRecord,
    TextRegion,
)
from transcription_review_agent.review.pngutil import encode_png
from transcription_review_agent.review.runtime import (
    DocumentOpenedEntry,
    PageAsset,
    ReviewFinalizedEntry,
    _replay,
)

BATCH = "batch-000009"
DOCUMENT = 7
NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)
BATCH_HASH = hashlib.sha256(b"sealed-batch").hexdigest()
ARCHIVE_HASH = hashlib.sha256(b"archive-transcription").hexdigest()
PILOT_ROOTS = (
    Path("outputs/reviews_track_a_pilot_104"),
    Path("outputs/reviews_track_a_pilot_105"),
    Path("outputs/reviews_track_a_pilot_108"),
)
BATCHES = Path("outputs/batches")


def _page_image() -> bytes:
    return encode_png(2, 2, [bytearray(b"\x00\x00"), bytearray(b"\x00\x00")], 1)


def _transcription(
    provider: str, text: str, bbox: BBox | None, bundle: str
) -> FirstPassOutput:
    task = f"{BATCH}:{DOCUMENT}:first_pass:r0:{provider}:a1"
    return FirstPassOutput(
        provider=provider,  # type: ignore[arg-type]
        provider_interface="GPT_WORK" if provider == "gpt" else "GROK",
        task_id=task,
        job_id=f"job:{task}",
        batch_id=BATCH,
        document_id=DOCUMENT,
        attempt=1,
        source_bundle_sha256=bundle,
        pages=[
            PageTranscription(
                page_id="page_001",
                reading_order=1,
                diplomatic_transcription=text,
                regions=[
                    TextRegion(
                        region_id="r1",
                        order=1,
                        lines=[LineReading(line_id="l1", text=text)],
                        bbox=bbox,
                    )
                ],
                unreadable_spans=[],
                alternative_readings=[],
                attention_items=[],
                coverage=CoverageReport(
                    coverage_complete=True,
                    unresolved_text_present=False,
                    examined_region_ids=["r1"],
                ),
                cancellations=[],
                insertions=[],
                marginal_notes=[],
                inaccessible_or_ambiguous=[],
            )
        ],
    )


def _dispute(
    dispute_id: str,
    category: DisputeCategory,
    gpt: str,
    grok: str,
    *,
    substantive: bool = True,
) -> Dispute:
    return Dispute(
        dispute_id=dispute_id,
        category=category,
        page_id="page_001",
        region_id="r1",
        gpt_region_ids=["r1"],
        grok_region_ids=["r1"],
        gpt_text=gpt,
        grok_text=grok,
        substantive=substantive,
    )


def _write_review(
    root: Path,
    *,
    disputes: list[Dispute],
    decisions: list[ReconciledDecision] | None = None,
    gpt_bbox: BBox | None = None,
    grok_bbox: BBox | None = None,
    crops: list[CropRecord] | None = None,
    bundle_sha: str | None = None,
) -> tuple[Path, str]:
    """Write one completed review whose hashes agree. Return reviews dir and bundle hash."""

    image = _page_image()
    page_sha = sha256_bytes(image)
    bundle = bundle_sha or hashlib.sha256(b"bundle-" + image).hexdigest()
    reviews = root / "reviews"
    doc = reviews / f"batch_{BATCH}" / "documents" / f"document_{DOCUMENT}"
    (doc / "input").mkdir(parents=True)
    (doc / "input" / "page_001.png").write_bytes(image)
    source = SourceManifest(
        batch_id=BATCH,
        document_id=DOCUMENT,
        language="en",
        source_bundle_sha256=bundle,
        pages=[
            SourcePageRecord(
                page_id="page_001",
                reading_order=1,
                sha256=page_sha,
                width=2,
                height=2,
                filename="page_001.png",
                mime_type="image/png",
            )
        ],
    )
    (doc / "input" / "source_manifest.json").write_text(
        canonical_json(source), encoding="utf-8"
    )
    gpt = _transcription(
        "gpt", disputes[0].gpt_text if disputes else "same", gpt_bbox, bundle
    )
    grok = _transcription(
        "grok", disputes[0].grok_text if disputes else "same", grok_bbox, bundle
    )
    gpt_path = doc / "first_pass" / "output" / "transcription_gpt.json"
    grok_path = doc / "first_pass" / "output" / "transcription_grok.json"
    gpt_path.parent.mkdir(parents=True)
    gpt_path.write_text(canonical_json(gpt), encoding="utf-8")
    grok_path.write_text(canonical_json(grok), encoding="utf-8")
    gpt_sha = sha256_bytes(gpt_path.read_bytes())
    grok_sha = sha256_bytes(grok_path.read_bytes())
    comparison = ComparisonFile(
        batch_id=BATCH,
        document_id=DOCUMENT,
        source_bundle_sha256=bundle,
        gpt_output_sha256=gpt_sha,
        grok_output_sha256=grok_sha,
        page_originals=[
            PageOriginals(
                page_id="page_001",
                gpt_diplomatic_transcription=gpt.pages[0].diplomatic_transcription,
                grok_diplomatic_transcription=grok.pages[0].diplomatic_transcription,
            )
        ],
        disputes=disputes,
        unresolved_text_page_ids=[],
    )
    comparison_path = doc / "comparison" / "comparison.json"
    comparison_path.parent.mkdir(parents=True)
    comparison_path.write_text(canonical_json(comparison), encoding="utf-8")
    comparison_sha = sha256_bytes(comparison_path.read_bytes())
    report = ReviewReport(
        batch_id=BATCH,
        document_id=DOCUMENT,
        review_completion="completed",
        quality=QualityLabel.UNRESOLVED_CONTENT,
        single_provider=False,
        coverage_gaps=[],
        unresolved_text_page_ids=[],
        disputes=disputes,
        decisions=decisions or [],
        identifications=[],
        provider_failure_count=0,
        estimated_work_units=2,
        page_ids=["page_001"],
    )
    report_bytes = canonical_json(report).encode("utf-8")
    summary = b"summary\n"
    proposed = b"proposed\n"
    manifest = ReviewManifest(
        batch_id=BATCH,
        document_id=DOCUMENT,
        source_bundle_sha256=bundle,
        source_batch_manifest_sha256=BATCH_HASH,
        providers=[
            ManifestProvider(
                provider="gpt",
                provider_interface="GPT_WORK",
                stage="first_pass",
                task_id=gpt.task_id,
                job_id=gpt.job_id,
                output_sha256=gpt_sha,
            ),
            ManifestProvider(
                provider="grok",
                provider_interface="GROK",
                stage="first_pass",
                task_id=grok.task_id,
                job_id=grok.job_id,
                output_sha256=grok_sha,
            ),
        ],
        artifacts=[
            ManifestArtifact(
                name="proposed_transcription.txt", sha256=sha256_bytes(proposed)
            ),
            ManifestArtifact(
                name="review_report.json", sha256=sha256_bytes(report_bytes)
            ),
            ManifestArtifact(name="review_summary.md", sha256=sha256_bytes(summary)),
        ],
        comparison_sha256=comparison_sha,
    )
    final = doc / "final"
    final.mkdir()
    (final / "proposed_transcription.txt").write_bytes(proposed)
    (final / "review_report.json").write_bytes(report_bytes)
    (final / "review_summary.md").write_bytes(summary)
    (final / "review_manifest.json").write_text(
        canonical_json(manifest), encoding="utf-8"
    )
    for crop in crops or []:
        crop_dir = doc / "challenge_round_01" / "input" / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / f"{crop.crop_id}.json").write_text(
            canonical_json(crop), encoding="utf-8"
        )
        if crop.crop_sha256 == page_sha:
            (crop_dir / f"{crop.crop_id}.png").write_bytes(image)
    hashes = {
        name: sha256_bytes((final / name).read_bytes())
        for name in (
            "proposed_transcription.txt",
            "review_report.json",
            "review_manifest.json",
            "review_summary.md",
        )
    }
    opened = DocumentOpenedEntry(
        sequence=1,
        occurred_at=NOW,
        batch_id=BATCH,
        document_id=DOCUMENT,
        language="en",
        source_bundle_sha256=bundle,
        source_batch_manifest_sha256=BATCH_HASH,
        archive_transcription_sha256=ARCHIVE_HASH,
        pages=[
            PageAsset(
                page_id="page_001",
                reading_order=1,
                sha256=page_sha,
                width=2,
                height=2,
                mime_type="image/png",
                original_filename="page_001.png",
            )
        ],
        estimated_work_units=2,
    )
    finalized = ReviewFinalizedEntry(
        sequence=2,
        occurred_at=NOW,
        batch_id=BATCH,
        document_id=DOCUMENT,
        quality=QualityLabel.UNRESOLVED_CONTENT,
        final_sha256=hashes,
    )
    replayed = _replay(BATCH, [opened, finalized])
    controller = reviews / f"batch_{BATCH}" / "controller"
    controller.mkdir()
    (controller / "ledger.jsonl").write_text(
        opened.model_dump_json() + "\n" + finalized.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (controller / "state.json").write_text(
        replayed.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return reviews, bundle


def _evidence(evidence_id: str) -> EvidenceReference:
    return EvidenceReference(
        kind="FULL_PAGE_FALLBACK",
        evidence_id=evidence_id,
        page_id="page_001",
    )


def _supported(gpt: str, grok: str) -> ReconciledDecision:
    ref = _evidence("crop_dsp_supported")
    return ReconciledDecision(
        dispute_id="dsp_supported",
        decision="CANDIDATE_Y",
        agreed=True,
        gpt_decision="CANDIDATE_Y",
        grok_decision="CANDIDATE_Y",
        result_class="AUTOMATICALLY_SUPPORTED",
        basis="DUAL_MODEL_VISUAL_SUPPORT",
        selected_reading=grok,
        candidate_x_text=gpt,
        candidate_y_text=grok,
        gpt_evidence_ref=ref,
        grok_evidence_ref=ref,
        gpt_evidence_assessment="SUPPORTS_DECISION",
        grok_evidence_assessment="SUPPORTS_DECISION",
    )


def _full_page_crop(page_sha: str) -> CropRecord:
    full = BBox(x=0, y=0, width=2, height=2)
    return CropRecord(
        crop_id="crop_dsp_supported",
        dispute_id="dsp_supported",
        page_id="page_001",
        page_sha256=page_sha,
        page_width=2,
        page_height=2,
        target_bbox=full,
        crop_bbox=full,
        enlarged=False,
        window_kind="FULL_PAGE_FALLBACK",
        fallback_reason="NO_RECORDED_COORDINATES",
        source_mime_type="image/png",
        crop_sha256=page_sha,
        crop_png_width=2,
        crop_png_height=2,
    )


def test_export_is_deterministic_and_not_verified(tmp_path: Path) -> None:
    gpt = "the school was"
    grok = "the schools were"
    dispute = _dispute("dsp_supported", DisputeCategory.WORDING, gpt, grok)
    image_sha = sha256_bytes(_page_image())
    reviews, _bundle = _write_review(
        tmp_path,
        disputes=[dispute],
        decisions=[_supported(gpt, grok)],
        crops=[_full_page_crop(image_sha)],
    )
    first = tmp_path / "out-a"
    second = tmp_path / "out-b"
    left = export_learning_candidates([reviews], output_dir=first)
    right = export_learning_candidates([reviews], output_dir=second)
    assert (first / "learning_candidates.json").read_bytes() == (
        second / "learning_candidates.json"
    ).read_bytes()
    assert left.model_dump() == right.model_dump()
    assert left.human_verified_count == 0
    assert left.authoritative_learning_eligible_count == 0
    assert left.correction_candidate_count == 0
    assert left.persistent_learning_ledger_written is False
    supported = [
        item
        for item in left.candidates
        if item.learning_category == "MODEL_SUPPORTED_READING"
    ]
    assert len(supported) == 1
    assert supported[0].human_verification_status == "NOT_VERIFIED"
    assert supported[0].eligible_for_authoritative_learning is False
    assert supported[0].model_selected_reading == grok
    assert supported[0].gpt_text == gpt
    assert supported[0].grok_text == grok
    assert supported[0].image_span_status == "UNAVAILABLE"
    assert supported[0].image_bbox is None
    rejected = [
        item
        for item in left.candidates
        if item.learning_category == "REJECTED_OR_SUPERSEDED_READING"
    ]
    assert [item.superseded_text for item in rejected] == [gpt]
    markdown = (first / "learning_candidates_review.md").read_text(encoding="utf-8")
    assert "NOT_VERIFIED" in markdown
    assert "not human-approved" in markdown
    assert supported[0].candidate_id in markdown


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    reviews, _bundle = _write_review(
        tmp_path,
        disputes=[_dispute("dsp_word", DisputeCategory.WORDING, "alpha", "beta")],
    )
    output = tmp_path / "missing"
    export_learning_candidates([reviews], output_dir=output, dry_run=True)
    assert not output.exists()
    assert (
        main(
            [
                "--reviews-dir",
                str(reviews),
                "--output-dir",
                str(output),
                "--preview",
            ]
        )
        == 0
    )
    assert not output.exists()


def test_manifest_and_source_hash_failures_write_nothing(tmp_path: Path) -> None:
    reviews, _bundle = _write_review(
        tmp_path,
        disputes=[_dispute("dsp_word", DisputeCategory.WORDING, "alpha", "beta")],
    )
    report = (
        reviews
        / f"batch_{BATCH}"
        / "documents"
        / f"document_{DOCUMENT}"
        / "final"
        / "review_report.json"
    )
    report.write_bytes(report.read_bytes().replace(b"alpha", b"alphb"))
    output = tmp_path / "out"
    with pytest.raises(ReviewIntegrityError):
        export_learning_candidates([reviews], output_dir=output)
    assert not output.exists()

    report.write_bytes(report.read_bytes().replace(b"alphb", b"alpha"))
    page = (
        reviews
        / f"batch_{BATCH}"
        / "documents"
        / f"document_{DOCUMENT}"
        / "input"
        / "page_001.png"
    )
    page.write_bytes(page.read_bytes() + b"\x00")
    with pytest.raises(ReviewIntegrityError):
        export_learning_candidates([reviews], output_dir=output)
    assert not output.exists()


def test_missing_page_reference_fails(tmp_path: Path) -> None:
    dispute = _dispute("dsp_word", DisputeCategory.WORDING, "alpha", "beta")
    dispute = dispute.model_copy(update={"page_id": "page_009"})
    reviews, _bundle = _write_review(tmp_path, disputes=[dispute])
    with pytest.raises(ReviewIntegrityError, match="source manifest"):
        export_learning_candidates([reviews], output_dir=tmp_path / "out")


def test_recorded_region_box_is_copied_and_conflict_is_not_chosen(
    tmp_path: Path,
) -> None:
    dispute = _dispute("dsp_name", DisputeCategory.NAME_OR_ENTITY, "Aaron", "Aron")
    reviews, _bundle = _write_review(
        tmp_path / "one",
        disputes=[dispute],
        gpt_bbox=BBox(x=0, y=0, width=1, height=1),
        grok_bbox=BBox(x=0, y=0, width=1, height=1),
    )
    export = export_learning_candidates([reviews], output_dir=tmp_path / "one-out")
    assert export.candidates[0].image_span_status == "RECORDED_REGION_BBOX"
    assert export.candidates[0].image_bbox is not None
    assert export.candidates[0].image_bbox.model_dump() == {
        "x": 0,
        "y": 0,
        "width": 1,
        "height": 1,
    }

    conflict_root = tmp_path / "two"
    conflict_reviews, _bundle = _write_review(
        conflict_root,
        disputes=[dispute],
        gpt_bbox=BBox(x=0, y=0, width=1, height=1),
        grok_bbox=BBox(x=1, y=0, width=1, height=1),
    )
    conflict = export_learning_candidates(
        [conflict_reviews], output_dir=tmp_path / "two-out"
    )
    assert conflict.candidates[0].image_span_status == "CONFLICTING_REGION_BBOX"
    assert conflict.candidates[0].image_bbox is None


def test_targeted_window_is_kept_and_full_page_is_not_a_text_span(
    tmp_path: Path,
) -> None:
    dispute = _dispute("dsp_supported", DisputeCategory.WORDING, "old", "new")
    window = BBox(x=0, y=0, width=1, height=1)
    crop = CropRecord(
        crop_id="crop_dsp_supported",
        dispute_id="dsp_supported",
        page_id="page_001",
        page_sha256=sha256_bytes(_page_image()),
        page_width=2,
        page_height=2,
        target_bbox=window,
        crop_bbox=window,
        enlarged=False,
        window_kind="TARGETED",
        coordinate_sources=["region.bbox"],
        source_mime_type="image/png",
    )
    ref = EvidenceReference(kind="CROP", evidence_id=crop.crop_id, page_id="page_001")
    decision = _supported("old", "new").model_copy(
        update={"gpt_evidence_ref": ref, "grok_evidence_ref": ref}
    )
    reviews, _bundle = _write_review(
        tmp_path, disputes=[dispute], decisions=[decision], crops=[crop]
    )
    export = export_learning_candidates([reviews], output_dir=tmp_path / "out")
    selected = next(
        item
        for item in export.candidates
        if item.learning_category == "MODEL_SUPPORTED_READING"
    )
    assert selected.image_span_status == "RECORDED_TARGETED_WINDOW"
    assert selected.image_bbox == window
    assert selected.text_span is not None
    assert selected.text_span.gpt.text == "old"
    assert selected.text_span.grok.text == "new"


def test_legacy_agreement_is_not_model_support(tmp_path: Path) -> None:
    dispute = _dispute("dsp_legacy", DisputeCategory.WORDING, "left", "right")
    legacy = ReconciledDecision(
        dispute_id="dsp_legacy",
        decision="CANDIDATE_X",
        agreed=True,
        gpt_decision="CANDIDATE_X",
        grok_decision="CANDIDATE_X",
    )
    assert legacy.result_class is None
    reviews, _bundle = _write_review(tmp_path, disputes=[dispute], decisions=[legacy])
    export = export_learning_candidates([reviews], output_dir=tmp_path / "out")
    assert len(export.candidates) == 1
    candidate = export.candidates[0]
    assert candidate.learning_category == "UNRESOLVED_DISAGREEMENT"
    assert candidate.reconciled_status == "LEGACY_UNCLASSIFIED"
    assert candidate.model_supported is False
    assert candidate.model_selected_reading is None
    assert candidate.human_verification_status == "NOT_VERIFIED"


def test_partial_agreement_keeps_exact_bounds(tmp_path: Path) -> None:
    gpt = "cat sat"
    grok = "cat seat"
    dispute = _dispute("dsp_partial", DisputeCategory.WORDING, gpt, grok)
    ref = _evidence("crop_missing")
    decision = ReconciledDecision(
        dispute_id="dsp_partial",
        decision="UNRESOLVED",
        agreed=False,
        gpt_decision="CANDIDATE_X",
        grok_decision="CANDIDATE_Y",
        result_class="UNRESOLVED",
        basis="PROVIDER_DISAGREEMENT",
        candidate_x_text=gpt,
        candidate_y_text=grok,
        gpt_evidence_ref=ref,
        grok_evidence_ref=ref,
        gpt_evidence_assessment="SUPPORTS_DECISION",
        grok_evidence_assessment="SUPPORTS_DECISION",
        partial_agreements=[
            PartialAgreementSpan(
                dispute_id="dsp_partial",
                role="ADOPTED",
                gpt_start=0,
                gpt_end=4,
                grok_start=0,
                grok_end=4,
                gpt_text="cat ",
                grok_text="cat ",
                gpt_decision="CANDIDATE_X",
                grok_decision="CANDIDATE_Y",
                gpt_evidence_ref=ref,
                grok_evidence_ref=ref,
                gpt_evidence_assessment="SUPPORTS_DECISION",
                grok_evidence_assessment="SUPPORTS_DECISION",
            ),
            PartialAgreementSpan(
                dispute_id="dsp_partial",
                role="UNRESOLVED",
                gpt_start=4,
                gpt_end=7,
                grok_start=4,
                grok_end=8,
                gpt_text="sat",
                grok_text="seat",
                gpt_decision="CANDIDATE_X",
                grok_decision="CANDIDATE_Y",
                gpt_evidence_ref=ref,
                grok_evidence_ref=ref,
                gpt_evidence_assessment="SUPPORTS_DECISION",
                grok_evidence_assessment="SUPPORTS_DECISION",
            ),
        ],
    )
    reviews, _bundle = _write_review(tmp_path, disputes=[dispute], decisions=[decision])
    export = export_learning_candidates([reviews], output_dir=tmp_path / "out")
    by_role = {item.reading_role: item for item in export.candidates}
    assert by_role["FULL_DISPUTE"].learning_category == "UNRESOLVED_DISAGREEMENT"
    assert by_role["FULL_DISPUTE"].model_supported is False
    adopted = by_role["ADOPTED_SPAN"]
    assert adopted.learning_category == "MODEL_SUPPORTED_READING"
    assert adopted.human_verification_status == "NOT_VERIFIED"
    assert adopted.text_span is not None
    assert (adopted.text_span.gpt.start, adopted.text_span.gpt.end) == (0, 4)
    assert adopted.gpt_text == "cat "
    unresolved = by_role["UNRESOLVED_SPAN"]
    assert unresolved.gpt_text == "sat"
    assert unresolved.grok_text == "seat"
    assert unresolved.model_supported is False
    markdown = (tmp_path / "out" / "learning_candidates_review.md").read_text(
        encoding="utf-8"
    )
    assert "cat " in markdown
    assert "sat" in markdown


def test_cancellation_insertion_and_missing_content_keep_exact_text(
    tmp_path: Path,
) -> None:
    cancellation = _dispute(
        "dsp_cancel",
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "[deleted: was]",
        "",
    )
    missing = _dispute(
        "dsp_missing",
        DisputeCategory.MISSING_CONTENT,
        "",
        "(1)",
    )
    punctuation = _dispute(
        "dsp_punct",
        DisputeCategory.PUNCTUATION,
        "end.",
        "end",
        substantive=False,
    )
    reviews, _bundle = _write_review(
        tmp_path, disputes=[cancellation, missing, punctuation]
    )
    export = export_learning_candidates([reviews], output_dir=tmp_path / "out")
    by_id = {item.dispute_id: item for item in export.candidates}
    assert set(by_id) == {"dsp_cancel", "dsp_missing"}
    assert by_id["dsp_cancel"].gpt_text == "[deleted: was]"
    assert by_id["dsp_cancel"].grok_text == ""
    assert by_id["dsp_missing"].gpt_text == ""
    assert by_id["dsp_missing"].grok_text == "(1)"
    assert by_id["dsp_cancel"].reconciled_status == "NOT_CHALLENGED"
    assert export.omitted.nonsubstantive_punctuation == 1
    markdown = (tmp_path / "out" / "learning_candidates_review.md").read_text(
        encoding="utf-8"
    )
    assert "[deleted: was]" in markdown
    assert "(1)" in markdown
    assert "end." not in markdown
    assert "Punctuation disputes omitted: 1" in markdown
    assert "are not candidate records" in markdown


def test_refuses_to_write_inside_the_review(tmp_path: Path) -> None:
    reviews, _bundle = _write_review(
        tmp_path,
        disputes=[_dispute("dsp_word", DisputeCategory.WORDING, "alpha", "beta")],
    )
    with pytest.raises(ReviewPathError):
        export_learning_candidates([reviews], output_dir=reviews)


def test_usage_errors() -> None:
    with pytest.raises(ReviewUsageError):
        export_learning_candidates([])
    with pytest.raises(ReviewUsageError):
        main(["--dry-run"])


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_bytes(path.read_bytes())
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def test_pilot_reviews_are_not_mutated(tmp_path: Path) -> None:
    if not all(path.is_dir() for path in PILOT_ROOTS) or not BATCHES.is_dir():
        pytest.skip("completed pilot reviews are not in this checkout")
    before = {str(path): _tree_hashes(path) for path in (*PILOT_ROOTS, BATCHES)}
    output = tmp_path / "pilot-export"
    export = export_learning_candidates(
        PILOT_ROOTS,
        output_dir=output,
        batches_dir=BATCHES,
        document_ids=[104, 105, 108],
    )
    again = export_learning_candidates(
        PILOT_ROOTS,
        output_dir=tmp_path / "pilot-export-2",
        batches_dir=BATCHES,
        document_ids=[104, 105, 108],
    )
    after = {str(path): _tree_hashes(path) for path in (*PILOT_ROOTS, BATCHES)}
    assert before == after
    assert export.model_dump() == again.model_dump()
    assert {item.document_id for item in export.documents} == {104, 105, 108}
    assert export.human_verified_count == 0
    assert export.correction_candidate_count == 0
    assert all(
        item.human_verification_status == "NOT_VERIFIED" for item in export.candidates
    )
    assert all(
        not item.eligible_for_authoritative_learning for item in export.candidates
    )
    payload = json.loads(
        (output / "learning_candidates.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "learning-candidate-export-v1"
    assert payload["persistent_learning_ledger_written"] is False
    assert len({item.candidate_id for item in export.candidates}) == len(
        export.candidates
    )


def _owned_crop(crop_id: str, dispute_id: str, *, targeted: bool) -> CropRecord:
    box = (
        BBox(x=0, y=0, width=1, height=1)
        if targeted
        else BBox(x=0, y=0, width=2, height=2)
    )
    return CropRecord(
        crop_id=crop_id,
        dispute_id=dispute_id,
        page_id="page_001",
        page_sha256="ab" * 32,
        page_width=2,
        page_height=2,
        target_bbox=box,
        crop_bbox=box,
        enlarged=False,
        window_kind="TARGETED" if targeted else "FULL_PAGE_FALLBACK",
        fallback_reason=None if targeted else "NO_RECORDED_COORDINATES",
        coordinate_sources=["region.bbox"] if targeted else None,
        source_mime_type="image/png",
    )


def test_partial_span_keeps_its_own_evidence() -> None:
    gpt = "cat sat"
    grok = "cat seat"
    dispute = _dispute("dsp_partial", DisputeCategory.WORDING, gpt, grok)
    parent = EvidenceReference(
        kind="FULL_PAGE_FALLBACK", evidence_id="crop_parent", page_id="page_001"
    )
    span_gpt = EvidenceReference(
        kind="CROP", evidence_id="crop_span_gpt", page_id="page_001"
    )
    span_grok = EvidenceReference(
        kind="CROP", evidence_id="crop_span_grok", page_id="page_001"
    )
    spans = []
    for role, gpt_text, grok_text, gpt_end, grok_end, start in (
        ("ADOPTED", "cat ", "cat ", 4, 4, 0),
        ("UNRESOLVED", "sat", "seat", 7, 8, 4),
    ):
        spans.append(
            PartialAgreementSpan(
                dispute_id="dsp_partial",
                role=role,  # type: ignore[arg-type]
                gpt_start=start,
                gpt_end=gpt_end,
                grok_start=start,
                grok_end=grok_end,
                gpt_text=gpt_text,
                grok_text=grok_text,
                gpt_decision="CANDIDATE_X",
                grok_decision="CANDIDATE_Y",
                gpt_evidence_ref=span_gpt,
                grok_evidence_ref=span_grok,
                gpt_evidence_assessment="SUPPORTS_DECISION",
                grok_evidence_assessment="SUPPORTS_DECISION",
            )
        )
    decision = ReconciledDecision.model_construct(
        dispute_id="dsp_partial",
        decision="UNRESOLVED",
        third_reading=None,
        agreed=False,
        agreement_is_not_verification=True,
        gpt_decision="CANDIDATE_X",
        grok_decision="CANDIDATE_Y",
        gpt_visual_evidence=None,
        grok_visual_evidence=None,
        remaining_uncertainty=True,
        result_class="UNRESOLVED",
        basis="PROVIDER_DISAGREEMENT",
        selected_reading=None,
        candidate_x_text=gpt,
        candidate_y_text=grok,
        gpt_third_reading=None,
        grok_third_reading=None,
        gpt_evidence_ref=parent,
        grok_evidence_ref=parent,
        gpt_evidence_assessment="SUPPORTS_DECISION",
        grok_evidence_assessment="SUPPORTS_DECISION",
        independently_verified=False,
        eligible_for_writer_memory=False,
        eligible_for_publication=False,
        partial_agreements=spans,
        other_round_decisions=None,
    )
    page = SourcePageRecord(
        page_id="page_001",
        reading_order=1,
        sha256="cd" * 32,
        width=2,
        height=2,
        filename="page_001.png",
        mime_type="image/png",
    )
    context = _Context(
        batch_id=BATCH,
        document_id=DOCUMENT,
        source_bundle="ef" * 32,
        report_sha="11" * 32,
        comparison_sha="22" * 32,
        gpt_output_sha="33" * 32,
        grok_output_sha="44" * 32,
        pages={"page_001": page},
        crops={
            "crop_parent": _owned_crop("crop_parent", "dsp_partial", targeted=False),
            "crop_span_gpt": _owned_crop("crop_span_gpt", "dsp_partial", targeted=True),
            "crop_span_grok": _owned_crop(
                "crop_span_grok", "dsp_partial", targeted=True
            ),
        },
        closure={},
        region_boxes={},
    )
    rows = _dispute_candidates(
        context, dispute, decision, _ImageSpan("UNAVAILABLE", "none", None)
    )
    primary = next(item for item in rows if item.reading_role == "FULL_DISPUTE")
    partials = [item for item in rows if item.reading_role != "FULL_DISPUTE"]
    assert {item.evidence_id for item in primary.evidence} == {"crop_parent"}
    assert {item.window_kind for item in primary.evidence} == {"FULL_PAGE_FALLBACK"}
    assert partials
    for item in partials:
        assert [(ref.provider, ref.evidence_id) for ref in item.evidence] == [
            ("gpt", "crop_span_gpt"),
            ("grok", "crop_span_grok"),
        ]
        assert {ref.window_kind for ref in item.evidence} == {"TARGETED"}
        assert {ref.assessment for ref in item.evidence} == {"SUPPORTS_DECISION"}


def test_region_boxes_come_only_from_verified_outputs(tmp_path: Path) -> None:
    dispute = _dispute("dsp_name", DisputeCategory.NAME_OR_ENTITY, "Aaron", "Aron")
    reviews, _bundle = _write_review(tmp_path, disputes=[dispute])
    doc = reviews / f"batch_{BATCH}" / "documents" / f"document_{DOCUMENT}"
    decoy = doc / "first_pass" / "unverified" / "transcription_gpt.json"
    decoy.parent.mkdir()
    verified = json.loads(
        (doc / "first_pass" / "output" / "transcription_gpt.json").read_text(
            encoding="utf-8"
        )
    )
    verified["pages"][0]["regions"][0]["bbox"] = {
        "x": 0,
        "y": 0,
        "width": 1,
        "height": 1,
    }
    decoy.write_text(json.dumps(verified), encoding="utf-8")
    export = export_learning_candidates([reviews], output_dir=tmp_path / "out")
    assert export.candidates[0].image_span_status == "UNAVAILABLE"
    assert export.candidates[0].image_bbox is None

    outside, _bundle = _write_review(
        tmp_path / "outside",
        disputes=[dispute],
        gpt_bbox=BBox(x=1, y=0, width=2, height=1),
        grok_bbox=BBox(x=1, y=0, width=2, height=1),
    )
    ignored = export_learning_candidates([outside], output_dir=tmp_path / "outside-out")
    assert ignored.candidates[0].image_span_status == "UNAVAILABLE"
    assert ignored.candidates[0].image_bbox is None


def test_contradictory_evidence_fails_and_legacy_absence_does_not(
    tmp_path: Path,
) -> None:
    dispute = _dispute("dsp_supported", DisputeCategory.WORDING, "old", "new")
    wrong_page = _supported("old", "new").model_copy(
        update={
            "gpt_evidence_ref": EvidenceReference(
                kind="FULL_PAGE_FALLBACK",
                evidence_id="crop_dsp_supported",
                page_id="page_009",
            ),
            "grok_evidence_ref": EvidenceReference(
                kind="FULL_PAGE_FALLBACK",
                evidence_id="crop_dsp_supported",
                page_id="page_009",
            ),
        }
    )
    reviews, _bundle = _write_review(
        tmp_path / "page", disputes=[dispute], decisions=[wrong_page]
    )
    with pytest.raises(ReviewIntegrityError, match="different page"):
        export_learning_candidates([reviews], output_dir=tmp_path / "page-out")
    assert not (tmp_path / "page-out").exists()

    crop = _full_page_crop(sha256_bytes(_page_image()))
    crop = crop.model_copy(update={"dispute_id": "dsp_other"})
    owned, _bundle = _write_review(
        tmp_path / "owner",
        disputes=[dispute],
        decisions=[_supported("old", "new")],
        crops=[crop],
    )
    with pytest.raises(ReviewIntegrityError, match="different dispute"):
        export_learning_candidates([owned], output_dir=tmp_path / "owner-out")
    assert not (tmp_path / "owner-out").exists()


def test_candidate_identifiers_are_unique(tmp_path: Path) -> None:
    dispute = _dispute("dsp_word", DisputeCategory.WORDING, "alpha", "beta")
    reviews, _bundle = _write_review(tmp_path, disputes=[dispute, dispute])
    output = tmp_path / "out"
    with pytest.raises(ReviewIntegrityError, match="not unique"):
        export_learning_candidates([reviews], output_dir=output)
    assert not output.exists()


def test_publish_failure_restores_the_previous_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews, _bundle = _write_review(
        tmp_path,
        disputes=[_dispute("dsp_word", DisputeCategory.WORDING, "alpha", "beta")],
    )
    output = tmp_path / "out"
    export_learning_candidates([reviews], output_dir=output)
    before = {name: (output / name).read_bytes() for name in (JSON_NAME, MARKDOWN_NAME)}
    real = os.replace
    failed = {"done": False}

    def flaky(src: str | Path, dst: str | Path) -> None:
        destination = Path(dst)
        if (
            not failed["done"]
            and destination.name == MARKDOWN_NAME
            and destination.parent == output
        ):
            failed["done"] = True
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(
        "transcription_review_agent.review.learning_export.os.replace", flaky
    )
    with pytest.raises(OSError, match="disk full"):
        export_learning_candidates([reviews], output_dir=output)
    assert {name: (output / name).read_bytes() for name in before} == before
    assert list(output.glob(".learning-candidate-export-*")) == []


def test_publish_failure_leaves_no_partial_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reviews, _bundle = _write_review(
        tmp_path,
        disputes=[_dispute("dsp_word", DisputeCategory.WORDING, "alpha", "beta")],
    )
    output = tmp_path / "fresh"
    real = os.replace
    failed = {"done": False}

    def flaky(src: str | Path, dst: str | Path) -> None:
        destination = Path(dst)
        if (
            not failed["done"]
            and destination.name == MARKDOWN_NAME
            and destination.parent == output
        ):
            failed["done"] = True
            raise OSError("disk full")
        real(src, dst)

    monkeypatch.setattr(
        "transcription_review_agent.review.learning_export.os.replace", flaky
    )
    with pytest.raises(OSError, match="disk full"):
        export_learning_candidates([reviews], output_dir=output)
    assert not (output / JSON_NAME).exists()
    assert not (output / MARKDOWN_NAME).exists()
    assert list(output.glob(".learning-candidate-export-*")) == []
