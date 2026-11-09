"""Acceptance tests for the local review_next v1 workflow."""

from __future__ import annotations

import json
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.layout import source_relative_path
from transcription_review_agent.bundling.manifest import (
    BatchDocumentRef,
    BatchManifest,
    BundleFile,
    DocumentBundleManifest,
    LocalProvenance,
    SourceArtifact,
    TranscriptionArtifact,
    publish_batch_manifest,
    publish_document_manifest,
)
from transcription_review_agent.review.config import ReviewConfig
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.crop import build_crop
from transcription_review_agent.review.errors import (
    ReviewOutputRejected,
    ReviewPolicyError,
    ReviewResultMissing,
)
from transcription_review_agent.review.models import (
    BBox,
    ChallengeDecision,
    ChallengeOutput,
    CoverageReport,
    FirstPassOutput,
    Identification,
    LineReading,
    PageTranscription,
    ResearchOutput,
    ResearchSource,
    TextRegion,
    WorkRequest,
)
from transcription_review_agent.review.paths import document_root

NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)
BATCH_ID = "batch-000001"
DOCUMENT_ID = 101
OCR = "ARCHIVE OCR TRANSCRIPTION THAT MUST NOT LEAK INTO PROVIDER FILES"
API_PAYLOAD = b'{"outcome":"REVIEWABLE"}'


def _png(width: int, height: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _seal(batches: Path, pages: list[bytes], mime: str = "image/png") -> Path:
    batch = batches / BATCH_ID
    document_dir = batch / "documents" / str(DOCUMENT_ID)
    text = OCR.encode("utf-8")
    files: dict[str, bytes] = {
        "api/review_bundle.json": API_PAYLOAD,
        "transcription/transcription.txt": text,
    }
    sources: list[SourceArtifact] = []
    file_records: list[BundleFile] = [
        BundleFile(
            relative_path="api/review_bundle.json",
            sha256=sha256_bytes(API_PAYLOAD),
            size_bytes=len(API_PAYLOAD),
            role="api_review_bundle",
        ),
        BundleFile(
            relative_path="transcription/transcription.txt",
            sha256=sha256_text(OCR),
            size_bytes=len(text),
            role="transcription_text",
        ),
    ]
    for index, payload in enumerate(pages):
        relative = source_relative_path(index, "PRIMARY_FILE", mime)
        files[relative] = payload
        sources.append(
            SourceArtifact(
                order_index=index,
                source_kind="PRIMARY_FILE",
                mime_type=mime,
                relative_path=relative,
                sha256=sha256_bytes(payload),
                size_bytes=len(payload),
            )
        )
        file_records.append(
            BundleFile(
                relative_path=relative,
                sha256=sha256_bytes(payload),
                size_bytes=len(payload),
                role="source",
            )
        )
    for relative, payload in files.items():
        path = document_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = DocumentBundleManifest(
        document_id=DOCUMENT_ID,
        archive_item_id=501,
        language="en",
        batch_id=BATCH_ID,
        created_at=NOW,
        transcription=TranscriptionArtifact(
            relative_path="transcription/transcription.txt",
            sha256=sha256_text(OCR),
            size_bytes=len(text),
            text_result_id=9001,
            result_type="SOURCE_TEXT",
            status="NEEDS_REVIEW",
            verification_status="UNVERIFIED",
        ),
        sources=sources,
        provenance=LocalProvenance(
            displayed_transcription_resolver="resolve_displayed_transcription_result",
            source_set="primary_file",
        ),
        files=file_records,
    )
    document_manifest = document_dir / "document_manifest.json"
    publish_document_manifest(document_manifest, manifest)
    publish_batch_manifest(
        batch / "batch_manifest.json",
        BatchManifest(
            batch_id=BATCH_ID,
            target_reviewable_count=1,
            document_ids=[DOCUMENT_ID],
            short_batch=False,
            discovery_exhausted=False,
            created_at=NOW,
            documents=[
                BatchDocumentRef(
                    document_id=DOCUMENT_ID,
                    manifest_relative_path=f"documents/{DOCUMENT_ID}/document_manifest.json",
                    manifest_sha256=sha256_bytes(document_manifest.read_bytes()),
                )
            ],
        ),
    )
    return batch


def _controller(
    tmp_path: Path,
    pages: list[bytes] | None = None,
    mime: str = "image/png",
) -> ReviewController:
    batches = tmp_path / "batches"
    _seal(batches, pages or [_png(20, 20), _png(20, 20)], mime=mime)
    return ReviewController(
        batches_dir=batches,
        reviews_dir=tmp_path / "reviews",
        clock=lambda: NOW,
    )


def _region(
    text: str, *, bbox: BBox | None = None, region_id: str = "r1"
) -> TextRegion:
    return TextRegion(
        region_id=region_id,
        order=1 if region_id == "r1" else 2,
        lines=[LineReading(line_id=f"{region_id}-l1", text=text)],
        bbox=bbox,
    )


def _page(
    page_id: str,
    order: int,
    text: str,
    *,
    complete: bool = True,
    unresolved: bool = False,
    name: str | None = None,
    bbox: BBox | None = None,
) -> PageTranscription:
    attention = []
    if name is not None:
        attention.append(
            {
                "kind": "name",
                "text": name,
                "region_id": "r1",
            }
        )
    return PageTranscription(
        page_id=page_id,
        reading_order=order,
        diplomatic_transcription=text,
        regions=[_region(text, bbox=bbox)],
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=attention,  # type: ignore[arg-type]
        coverage=CoverageReport(
            coverage_complete=complete,
            unresolved_text_present=unresolved,
            examined_region_ids=["r1"],
        ),
        cancellations=[],
        insertions=[],
        marginal_notes=[],
        inaccessible_or_ambiguous=[],
    )


def _write_output(
    controller: ReviewController, request_path: Path, payload: dict[str, object]
) -> Path:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("stage") == "challenge":
        disputes_path = request_path.parent / "input" / "disputes.json"
        published = json.loads(disputes_path.read_text(encoding="utf-8"))
        if published.get("challenge_contract") == "EVIDENCE_BOUND_V1":
            refs = {
                dispute_id: group["evidence_refs"][0]
                for group in published["groups"]
                for dispute_id in group["dispute_ids"]
            }
            for decision in payload.get("decisions", []):  # type: ignore[union-attr]
                if (
                    isinstance(decision, dict)
                    and decision.get("evidence_ref") is None
                    and decision.get("dispute_id") in refs
                ):
                    decision["evidence_ref"] = refs[decision["dispute_id"]]
                    decision["evidence_assessment"] = "SUPPORTS_DECISION"
    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    target = doc_root / request["output_relative_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _first_pass_body(
    request_path: Path,
    pages: list[PageTranscription],
) -> dict[str, object]:
    request = WorkRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    model = FirstPassOutput(
        provider=request.provider,
        provider_interface=request.provider_interface,
        task_id=request.task_id,
        job_id=request.job_id,
        batch_id=request.batch_id,
        document_id=request.document_id,
        attempt=request.attempt,
        source_bundle_sha256=request.source_bundle_sha256,
        pages=pages,
    )
    return model.model_dump(mode="json")


def _deliver_first_pass(
    controller: ReviewController,
    provider: str,
    pages: list[PageTranscription],
) -> None:
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, provider)
    _write_output(controller, request_path, _first_pass_body(request_path, pages))
    controller.accept(BATCH_ID, DOCUMENT_ID, provider)


def _agreeing_pages(
    text: str = "The visible line is secure.",
) -> list[PageTranscription]:
    return [
        _page("page_001", 1, text, bbox=BBox(x=4, y=4, width=4, height=4)),
        _page("page_002", 2, "Second page remains."),
    ]


def _close_challenge(
    controller: ReviewController,
    mode: str = "SELECTED_ROUNDS_COMPLETE",
    reason: str = "Operator finished the challenge stage.",
):
    """A4: a completed proposal with substantive disputes needs explicit closure."""

    return controller.close_challenge(BATCH_ID, DOCUMENT_ID, mode=mode, reason=reason)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_sealed_multipage_document_enters_and_keeps_page_order(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    before = _snapshot(tmp_path / "batches")
    status = controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    assert status.page_ids == ("page_001", "page_002")
    assert status.execution_state == "READY_FOR_PROVIDER"
    assert status.estimated_work_units == 4
    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    source = json.loads(
        (doc_root / "input" / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert source["archive_transcription_included"] is False
    assert [page["reading_order"] for page in source["pages"]] == [1, 2]
    assert (
        OCR.encode("utf-8")
        not in (doc_root / "input" / "source_manifest.json").read_bytes()
    )
    request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    task_dir = request.parent
    names = {path.name for path in task_dir.rglob("*") if path.is_file()}
    assert "transcription.txt" not in names
    assert "transcription_grok.json" not in names
    work = json.loads(request.read_text(encoding="utf-8"))
    assert work["dispatch"] == "manual"
    assert work["unattended_api"] is False
    assert _snapshot(tmp_path / "batches") == before


def _reject_once(tmp_path: Path, mutate) -> ReviewOutputRejected:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    body = _first_pass_body(request_path, _agreeing_pages())
    mutate(body)
    _write_output(controller, request_path, body)
    with pytest.raises(ReviewOutputRejected) as caught:
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    return caught.value


def test_provider_files_are_validated_and_a_partial_page_cannot_finish(
    tmp_path: Path,
) -> None:
    assert (
        _reject_once(
            tmp_path / "label",
            lambda body: body.update(provider="grok", provider_interface="GROK"),
        ).reason
        == "invalid_label"
    )
    assert (
        _reject_once(
            tmp_path / "stale",
            lambda body: body.update(source_bundle_sha256="cd" * 32),
        ).reason
        == "stale"
    )
    assert (
        _reject_once(
            tmp_path / "mismatch",
            lambda body: body.update(batch_id="batch-000009"),
        ).reason
        == "mismatched"
    )
    assert (
        _reject_once(tmp_path / "schema", lambda body: body.pop("pages")).reason
        == "schema"
    )

    controller = _controller(tmp_path / "partial")
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    _write_output(
        controller, request_path, _first_pass_body(request_path, _agreeing_pages()[:1])
    )
    with pytest.raises(ReviewOutputRejected) as pages:
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    assert pages.value.reason == "mismatched"
    status = controller.status(BATCH_ID, DOCUMENT_ID)
    assert status.finalized is False
    assert status.execution_state != "COMPLETED"
    assert status.page_ids == ("page_001", "page_002")


def test_two_providers_disputes_challenge_and_research_do_not_rewrite_text(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    bbox = BBox(x=4, y=4, width=4, height=4)
    _deliver_first_pass(
        controller,
        "gpt",
        [
            _page("page_001", 1, "Ionnes", name="Ionnes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    _deliver_first_pass(
        controller,
        "grok",
        [
            _page("page_001", 1, "Iannes", name="Iannes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    assert comparison.selected_by_confidence is False
    assert any(item.category.value == "NAME_OR_ENTITY" for item in comparison.disputes)
    assert comparison.page_originals[0].gpt_diplomatic_transcription == "Ionnes"
    assert comparison.page_originals[0].grok_diplomatic_transcription == "Iannes"
    dispute_ids = [item.dispute_id for item in comparison.disputes if item.substantive]
    assert dispute_ids

    gpt_request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    grok_request = controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    disputes = json.loads(
        (doc_root / "challenge_round_01" / "input" / "disputes.json").read_text(
            encoding="utf-8"
        )
    )
    assert "provider" not in json.dumps(disputes)
    assert disputes["disputes"][0]["candidate_x"] == "Ionnes"
    assert disputes["disputes"][0]["candidate_y"] == "Iannes"
    crop_path = next(
        (doc_root / "challenge_round_01" / "input" / "crops").glob("*.json")
    )
    crop = json.loads(crop_path.read_text(encoding="utf-8"))
    assert crop["window_kind"] == "TARGETED"
    assert crop["contains_target"] is True
    assert crop["represents_new_visual_detail"] is False
    assert crop["scale"] == 1
    assert crop["full_page_retained"] is True
    assert (
        crop["crop_bbox"]["width"] * crop["crop_bbox"]["height"]
        > crop["target_bbox"]["width"] * crop["target_bbox"]["height"]
    )
    crop_png = crop_path.with_suffix(".png")
    assert sha256_bytes(crop_png.read_bytes()) == crop["crop_sha256"]
    assert (crop["crop_png_width"], crop["crop_png_height"]) == (
        crop["crop_bbox"]["width"],
        crop["crop_bbox"]["height"],
    )
    assert (
        doc_root / "challenge_round_01" / "input" / "page_001.png"
    ).read_bytes() == (doc_root / "input" / "page_001.png").read_bytes()

    for request_path, provider in ((gpt_request, "gpt"), (grok_request, "grok")):
        request = WorkRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        challenge = ChallengeOutput(
            provider=request.provider,
            provider_interface=request.provider_interface,
            task_id=request.task_id,
            job_id=request.job_id,
            batch_id=request.batch_id,
            document_id=request.document_id,
            attempt=request.attempt,
            source_bundle_sha256=request.source_bundle_sha256,
            round_index=request.round_index,
            decisions=[
                ChallengeDecision(
                    dispute_id=dispute_id,
                    decision="CANDIDATE_X",
                    visual_evidence="The crop shows the first candidate.",
                )
                for dispute_id in dispute_ids
            ],
        )
        _write_output(controller, request_path, challenge.model_dump(mode="json"))
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")

    research_request = controller.publish(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="research"
    )
    request = WorkRequest.model_validate_json(
        research_request.read_text(encoding="utf-8")
    )
    research = ResearchOutput(
        provider=request.provider,
        provider_interface=request.provider_interface,
        task_id=request.task_id,
        job_id=request.job_id,
        batch_id=request.batch_id,
        document_id=request.document_id,
        attempt=request.attempt,
        source_bundle_sha256=request.source_bundle_sha256,
        identifications=[
            Identification(
                original_visual_reading="Ionnes",
                historical_identity_hypothesis="Johannes",
                supporting_evidence=["A later register uses Johannes."],
                opposing_evidence=["The page spells Ionnes."],
                sources=[
                    ResearchSource(
                        title="Parish register",
                        url="https://example.test/register",
                        source_class="scholarly",
                    )
                ],
                original_image_inspected=True,
                provenance_limitations="The register is not this page.",
                confidence="low",
                unresolved_questions=["Is the register the same person?"],
                uses_target_document_transcription=True,
            )
        ],
    )
    _write_output(controller, research_request, research.model_dump(mode="json"))
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    _close_challenge(controller)
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert "Ionnes" in text
    assert "Johannes" not in text
    assert "[unresolved:" not in text
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert report["diplomatic_text_independent_of_research"] is True
    assert report["identifications"][0]["corroboration_rejected"] is True
    assert (
        report["identifications"][0]["identification"]["historical_identity_hypothesis"]
        == "Johannes"
    )
    assert report["production_writeback"] is False
    manifest = json.loads(paths["review_manifest.json"].read_text(encoding="utf-8"))
    assert manifest["source_bundle_sha256"]
    assert {item["provider_interface"] for item in manifest["providers"]} >= {
        "GPT_WORK",
        "GROK",
    }
    summary = paths["review_summary.md"].read_text(encoding="utf-8")
    assert "סיכום ביקורת תמלול" in summary
    assert (
        controller.status(BATCH_ID, DOCUMENT_ID).research_changes_diplomatic_text
        is False
    )


def test_coverage_gap_and_unresolved_challenge_remain_explicit(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    pages_gpt = [
        _page("page_001", 1, "shared line"),
        _page("page_002", 2, "faint line", complete=False),
    ]
    pages_grok = [
        _page("page_001", 1, "shared line"),
        _page("page_002", 2, "faint line", complete=False),
    ]
    _deliver_first_pass(controller, "gpt", pages_gpt)
    _deliver_first_pass(controller, "grok", pages_grok)
    controller.compare(BATCH_ID, DOCUMENT_ID)
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert report["quality"] == "INCOMPLETE_COVERAGE"
    assert report["coverage_gaps"] == ["page_002"]
    assert report["execution_state"] == "COMPLETED"
    assert "[coverage-incomplete]" in paths["proposed_transcription.txt"].read_text(
        encoding="utf-8"
    )
    assert "כיסוי חסר" in paths["review_summary.md"].read_text(encoding="utf-8")

    other = _controller(tmp_path / "unresolved")
    other.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(
        other, "gpt", [_page("page_001", 1, "Alpha"), _page("page_002", 2, "Beta")]
    )
    _deliver_first_pass(
        other, "grok", [_page("page_001", 1, "Alfa"), _page("page_002", 2, "Beta")]
    )
    comparison = other.compare(BATCH_ID, DOCUMENT_ID)
    dispute_ids = [item.dispute_id for item in comparison.disputes if item.substantive]
    for provider in ("gpt", "grok"):
        request_path = other.publish(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
        request = WorkRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        challenge = ChallengeOutput(
            provider=request.provider,
            provider_interface=request.provider_interface,
            task_id=request.task_id,
            job_id=request.job_id,
            batch_id=request.batch_id,
            document_id=request.document_id,
            attempt=request.attempt,
            source_bundle_sha256=request.source_bundle_sha256,
            round_index=request.round_index,
            decisions=[
                ChallengeDecision(
                    dispute_id=dispute_id,
                    decision="UNRESOLVED",
                    visual_evidence="Both shapes remain possible.",
                )
                for dispute_id in dispute_ids
            ],
        )
        _write_output(other, request_path, challenge.model_dump(mode="json"))
        other.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    _close_challenge(other)
    paths = other.finalize(BATCH_ID, DOCUMENT_ID)
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert "[Alpha / Alfa]" in text
    assert "Alpha" in text and "Alfa" in text
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert report["execution_state"] == "COMPLETED"


def test_single_provider_quota_and_total_failure(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(controller, "gpt", _agreeing_pages("Only one reading."))
    controller.publish(BATCH_ID, DOCUMENT_ID, "grok")
    blocked = controller.record_failure(BATCH_ID, DOCUMENT_ID, "grok", kind="quota")
    assert blocked.execution_state == "BLOCKED"
    assert blocked.providers["grok"] == "QUOTA_BLOCKED"
    again = controller.record_failure(BATCH_ID, DOCUMENT_ID, "grok", kind="quota")
    assert again.ledger_sequence == blocked.ledger_sequence
    with pytest.raises(ReviewPolicyError):
        controller.publish(BATCH_ID, DOCUMENT_ID, "grok")
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert report["quality"] == "SINGLE_PROVIDER_PROVISIONAL"
    assert report["review_completion"] == "interim_provisional"
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is False
    assert "Only one reading." in paths["proposed_transcription.txt"].read_text(
        encoding="utf-8"
    )
    assert controller.status(BATCH_ID, DOCUMENT_ID).providers["gpt"] == "SUCCEEDED"

    failed = _controller(tmp_path / "failed")
    failed.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    for provider in ("gpt", "grok"):
        request_path = failed.publish(BATCH_ID, DOCUMENT_ID, provider)
        _write_output(failed, request_path, {"provider": "nope"})
        with pytest.raises(ReviewOutputRejected):
            failed.accept(BATCH_ID, DOCUMENT_ID, provider)
    with pytest.raises(ReviewPolicyError):
        failed.finalize(BATCH_ID, DOCUMENT_ID)
    doc_root = document_root(failed.reviews_dir, BATCH_ID, DOCUMENT_ID)
    assert not (doc_root / "final" / "proposed_transcription.txt").exists()


def test_retry_is_single_and_resume_does_not_duplicate(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    once = controller.record_failure(BATCH_ID, DOCUMENT_ID, "gpt", kind="transient")
    assert once.providers["gpt"] == "RETRYABLE_FAILURE"
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    twice = controller.record_failure(BATCH_ID, DOCUMENT_ID, "gpt", kind="transient")
    assert twice.providers["gpt"] == "FAILED"
    with pytest.raises(ReviewPolicyError):
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")

    resumed = _controller(tmp_path / "resume")
    resumed.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(resumed, "gpt", _agreeing_pages())
    ledger = (
        resumed.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "ledger.jsonl"
    ).read_text(encoding="utf-8")
    assert ledger.count('"event_type":"result_accepted"') == 1
    again = ReviewController(
        batches_dir=tmp_path / "resume" / "batches",
        reviews_dir=tmp_path / "resume" / "reviews",
        clock=lambda: NOW,
    )
    again.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    checkpoint = again.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "state.json"
    checkpoint.unlink()
    status = again.status(BATCH_ID, DOCUMENT_ID)
    assert status.providers["gpt"] == "SUCCEEDED"
    _deliver_first_pass(again, "grok", _agreeing_pages())
    again.compare(BATCH_ID, DOCUMENT_ID)
    first_final = again.finalize(BATCH_ID, DOCUMENT_ID)
    before = {name: path.read_bytes() for name, path in first_final.items()}
    second = ReviewController(
        batches_dir=tmp_path / "resume" / "batches",
        reviews_dir=tmp_path / "resume" / "reviews",
        clock=lambda: NOW,
    )
    second_final = second.finalize(BATCH_ID, DOCUMENT_ID)
    assert {name: path.read_bytes() for name, path in second_final.items()} == before
    ledger = (
        second.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "ledger.jsonl"
    ).read_text(encoding="utf-8")
    assert ledger.count('"event_type":"result_accepted"') == 2
    assert ledger.count('"event_type":"review_finalized"') == 1
    assert set(before) == {
        "proposed_transcription.txt",
        "review_report.json",
        "review_manifest.json",
        "review_summary.md",
    }


def test_crop_window_contains_the_target_and_adds_no_detail() -> None:
    page = BBox(x=0, y=0, width=20, height=20)
    target = BBox(x=4, y=4, width=4, height=4)
    crop = build_crop(
        crop_id="crop_dsp_test",
        dispute_id="dsp_test",
        page_id="page_001",
        page_sha256="ab" * 32,
        page_width=page.width,
        page_height=page.height,
        target=target,
    )
    assert crop.crop_bbox.contains(target)
    assert crop.represents_new_visual_detail is False
    with pytest.raises(ValueError):
        build_crop(
            crop_id="crop_outside",
            dispute_id="dsp_out",
            page_id="page_001",
            page_sha256="ab" * 32,
            page_width=10,
            page_height=10,
            target=BBox(x=8, y=8, width=4, height=4),
        )


def test_second_challenge_round_requires_new_evidence(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.config = ReviewConfig(
        max_challenge_rounds=2,
        max_document_challenge_dispatches=4,
    )
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(
        controller, "gpt", [_page("page_001", 1, "Alpha"), _page("page_002", 2, "Beta")]
    )
    _deliver_first_pass(
        controller, "grok", [_page("page_001", 1, "Alfa"), _page("page_002", 2, "Beta")]
    )
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    dispute_ids = [item.dispute_id for item in comparison.disputes if item.substantive]
    for provider in ("gpt", "grok"):
        request_path = controller.publish(
            BATCH_ID, DOCUMENT_ID, provider, stage="challenge"
        )
        request = WorkRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        challenge = ChallengeOutput(
            provider=request.provider,
            provider_interface=request.provider_interface,
            task_id=request.task_id,
            job_id=request.job_id,
            batch_id=request.batch_id,
            document_id=request.document_id,
            attempt=request.attempt,
            source_bundle_sha256=request.source_bundle_sha256,
            round_index=request.round_index,
            decisions=[
                ChallengeDecision(
                    dispute_id=dispute_id,
                    decision="UNRESOLVED",
                    visual_evidence="The first shape is clearer.",
                )
                for dispute_id in dispute_ids
            ],
        )
        _write_output(controller, request_path, challenge.model_dump(mode="json"))
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    with pytest.raises(ReviewPolicyError):
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    with pytest.raises(ReviewPolicyError, match="explicit dispute selection"):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="challenge",
            evidence="new_crop",
            evidence_path=tmp_path / "unused.png",
        )
    with pytest.raises(ReviewPolicyError, match="not a new evidence file"):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="challenge",
            evidence="new_crop",
            selected_dispute_ids=dispute_ids,
        )
    page_png = (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "input"
        / "page_001.png"
    )
    with pytest.raises(ReviewPolicyError, match="already examined"):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="challenge",
            evidence="different_source_scan",
            evidence_path=page_png,
            selected_dispute_ids=dispute_ids,
        )
    evidence = tmp_path / "other-scan.png"
    evidence.write_bytes(_png(8, 8))
    second = controller.publish(
        BATCH_ID,
        DOCUMENT_ID,
        "gpt",
        stage="challenge",
        evidence="new_crop",
        evidence_path=evidence,
        selected_dispute_ids=dispute_ids,
    )
    assert "challenge_round_02" in second.as_posix()
    stored = next(
        (
            document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
            / "challenge_round_02"
            / "evidence"
        ).iterdir()
    )
    assert sha256_bytes(stored.read_bytes()) == sha256_bytes(evidence.read_bytes())
    assert sha256_bytes(stored.read_bytes()) != sha256_bytes(page_png.read_bytes())
    with pytest.raises(ReviewPolicyError):
        controller.publish(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", evidence="please_agree"
        )


def test_missing_region_box_is_labeled_full_page_fallback(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(
        controller,
        "gpt",
        [
            _page("page_001", 1, "Ionnes", name="Ionnes"),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    _deliver_first_pass(
        controller,
        "grok",
        [
            _page("page_001", 1, "Iannes", name="Iannes"),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    controller.compare(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    crop_path = next(
        (doc_root / "challenge_round_01" / "input" / "crops").glob("*.json")
    )
    crop = json.loads(crop_path.read_text(encoding="utf-8"))
    assert crop["window_kind"] == "FULL_PAGE_FALLBACK"
    assert (crop["crop_png_width"], crop["crop_png_height"]) == (20, 20)
    assert (crop["crop_bbox"]["width"], crop["crop_bbox"]["height"]) == (20, 20)
    page = (doc_root / "challenge_round_01" / "input" / "page_001.png").read_bytes()
    assert page == (doc_root / "input" / "page_001.png").read_bytes()


def test_accept_before_a_file_arrives_is_missing(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    with pytest.raises(ReviewResultMissing):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")


def test_first_pass_task_uses_a_positive_allowlist(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    (doc_root / "input" / "archive_ocr.txt").write_text(OCR, encoding="utf-8")
    (doc_root / "input" / "transcription_grok.json").write_text(
        '{"pages":[]}', encoding="utf-8"
    )
    (doc_root / "input" / "earlier_reading.json").write_text(
        '{"hypothesis":"Johannes"}',
        encoding="utf-8",
    )
    request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    task_dir = request.parent
    names = {
        path.relative_to(task_dir).as_posix()
        for path in task_dir.rglob("*")
        if path.is_file()
    }
    assert names == {
        "README.txt",
        "instructions_gpt.txt",
        "output_schema.json",
        "output_example.json",
        "work_request.json",
        "input/source_manifest.json",
        "input/page_001.png",
        "input/page_002.png",
    }
    readme = (task_dir / "README.txt").read_text(encoding="utf-8")
    assert "not an operating-system sandbox" in readme
    assert OCR not in readme
    assert (task_dir.parent.parent / "output").is_dir()
    grok_request = controller.publish(BATCH_ID, DOCUMENT_ID, "grok")
    assert grok_request.parent != request.parent
    assert grok_request.parent.parent.parent == request.parent.parent.parent


def test_pending_challenge_or_research_cannot_be_finalized(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    bbox = BBox(x=4, y=4, width=4, height=4)
    _deliver_first_pass(
        controller,
        "gpt",
        [
            _page("page_001", 1, "Ionnes", name="Ionnes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    _deliver_first_pass(
        controller,
        "grok",
        [
            _page("page_001", 1, "Iannes", name="Iannes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    controller.compare(BATCH_ID, DOCUMENT_ID)
    with pytest.raises(ReviewPolicyError, match="explicitly closed challenge stage"):
        controller.finalize(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    with pytest.raises(ReviewPolicyError, match="still pending"):
        controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)

    research = _controller(tmp_path / "research")
    research.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(
        research,
        "gpt",
        [
            _page("page_001", 1, "Ionnes", name="Ionnes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    _deliver_first_pass(
        research,
        "grok",
        [
            _page("page_001", 1, "Iannes", name="Iannes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    research.compare(BATCH_ID, DOCUMENT_ID)
    research.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    with pytest.raises(
        ReviewPolicyError, match="opened research stage is still pending"
    ):
        research.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)


def test_unresolved_challenge_can_complete_and_agreement_is_not_proof(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    bbox = BBox(x=4, y=4, width=4, height=4)
    _deliver_first_pass(
        controller,
        "gpt",
        [
            _page("page_001", 1, "Ionnes", name="Ionnes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    _deliver_first_pass(
        controller,
        "grok",
        [
            _page("page_001", 1, "Iannes", name="Iannes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    dispute_ids = [item.dispute_id for item in comparison.disputes if item.substantive]
    evidence = {
        "gpt": "The left stroke is a single vertical.",
        "grok": "The left stroke may be two marks.",
    }
    for provider in ("gpt", "grok"):
        request_path = controller.publish(
            BATCH_ID, DOCUMENT_ID, provider, stage="challenge"
        )
        request = WorkRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        challenge = ChallengeOutput(
            provider=request.provider,
            provider_interface=request.provider_interface,
            task_id=request.task_id,
            job_id=request.job_id,
            batch_id=request.batch_id,
            document_id=request.document_id,
            attempt=request.attempt,
            source_bundle_sha256=request.source_bundle_sha256,
            round_index=request.round_index,
            decisions=[
                ChallengeDecision(
                    dispute_id=dispute_id,
                    decision="CANDIDATE_X",
                    visual_evidence=evidence[provider],
                )
                for dispute_id in dispute_ids
            ],
        )
        _write_output(controller, request_path, challenge.model_dump(mode="json"))
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    _close_challenge(controller)
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    decision = report["decisions"][0]
    assert report["review_completion"] == "completed"
    assert decision["agreed"] is True
    assert decision["agreement_is_not_verification"] is True
    assert decision["remaining_uncertainty"] is True
    assert decision["gpt_decision"] == "CANDIDATE_X"
    assert decision["grok_decision"] == "CANDIDATE_X"
    assert decision["gpt_visual_evidence"] != decision["grok_visual_evidence"]
    assert "confidence" not in decision

    unresolved = _controller(tmp_path / "unresolved")
    unresolved.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(
        unresolved,
        "gpt",
        [
            _page("page_001", 1, "Ionnes", name="Ionnes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    _deliver_first_pass(
        unresolved,
        "grok",
        [
            _page("page_001", 1, "Iannes", name="Iannes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    )
    comparison = unresolved.compare(BATCH_ID, DOCUMENT_ID)
    dispute_ids = [item.dispute_id for item in comparison.disputes if item.substantive]
    for provider, choice in (("gpt", "CANDIDATE_X"), ("grok", "CANDIDATE_Y")):
        request_path = unresolved.publish(
            BATCH_ID, DOCUMENT_ID, provider, stage="challenge"
        )
        request = WorkRequest.model_validate_json(
            request_path.read_text(encoding="utf-8")
        )
        challenge = ChallengeOutput(
            provider=request.provider,
            provider_interface=request.provider_interface,
            task_id=request.task_id,
            job_id=request.job_id,
            batch_id=request.batch_id,
            document_id=request.document_id,
            attempt=request.attempt,
            source_bundle_sha256=request.source_bundle_sha256,
            round_index=request.round_index,
            decisions=[
                ChallengeDecision(
                    dispute_id=dispute_id,
                    decision=choice,
                    visual_evidence="The mark is ambiguous.",
                )
                for dispute_id in dispute_ids
            ],
        )
        _write_output(unresolved, request_path, challenge.model_dump(mode="json"))
        unresolved.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    _close_challenge(unresolved)
    paths = unresolved.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert report["review_completion"] == "completed"
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert report["decisions"][0]["decision"] == "UNRESOLVED"
    assert report["decisions"][0]["remaining_uncertainty"] is True
    assert "[Ionnes / Iannes]" in text
    assert "Ionnes" in text and "Iannes" in text
