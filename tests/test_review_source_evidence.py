"""Supplementary evidence from an original provider file survives replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.evidence import collect_source_evidence
from transcription_review_agent.review.models import (
    CoverageReport,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    ReviewReport,
    TextRegion,
)
from transcription_review_agent.review.runtime import ReviewStore
from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _agreeing_pages,
    _controller,
    _first_pass_body,
    _write_output,
)


def _repaired() -> FirstPassOutput:
    page = PageTranscription(
        page_id="page_001",
        reading_order=1,
        diplomatic_transcription="alpha",
        regions=[
            TextRegion(
                region_id="r1",
                order=1,
                lines=[LineReading(line_id="r1-l1", text="alpha")],
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
    return FirstPassOutput(
        provider="gpt",
        provider_interface="GPT_WORK",
        task_id="task-gpt",
        job_id="job-gpt",
        batch_id="batch-000001",
        document_id=101,
        attempt=1,
        source_bundle_sha256="ab" * 32,
        pages=[page],
    )


def test_original_details_absent_from_the_repaired_file_are_kept() -> None:
    original = {
        "provider": "gpt",
        "pages": [
            {
                "page_id": "page_001",
                "regions": [
                    {
                        "region_id": "r1",
                        "lines": [{"line_id": "r1-l1", "text": "alpha"}],
                    }
                ],
                "unreadable_spans": [
                    {
                        "line_id": "r1-l1",
                        "alternatives": ["vie", "rref"],
                        "reason": "letter forms unclear",
                    }
                ],
                "marginal_notes": [
                    {
                        "region_id": "r1",
                        "text": "",
                        "description": "A blue oval surrounds the date.",
                    }
                ],
            }
        ],
    }
    evidence = collect_source_evidence(original, _repaired(), original_sha256="ab" * 32)
    values = {item.value for item in evidence}
    assert '"vie"' in values
    assert '"rref"' in values
    assert '"letter forms unclear"' in values
    assert '"A blue oval surrounds the date."' in values
    assert '"alpha"' not in values
    span = next(item for item in evidence if item.value == '"vie"')
    assert span.provider == "gpt"
    assert span.json_path.endswith("alternatives[0]")
    assert span.line_id == "r1-l1"
    assert span.region_id == "r1"
    assert span.page_id == "page_001"
    assert span.original_sha256 == "ab" * 32
    linked = {
        "provider": "gpt",
        "pages": [
            {
                "page_id": "page_001",
                "regions": [
                    {
                        "region_id": "r1",
                        "lines": [{"line_id": "r1-l1", "text": "alpha"}],
                    }
                ],
                "attention_items": [
                    {
                        "line_ids": ["r1-l1"],
                        "description": "names need review",
                    },
                    {
                        "line_ids": [],
                        "description": "no line was named",
                    },
                ],
            }
        ],
    }
    linked_evidence = collect_source_evidence(
        linked, _repaired(), original_sha256="ab" * 32
    )
    named = next(
        item for item in linked_evidence if item.value == '"names need review"'
    )
    assert named.line_id == "r1-l1"
    assert named.region_id == "r1"
    unnamed = next(
        item for item in linked_evidence if item.value == '"no line was named"'
    )
    assert unnamed.line_id is None
    assert unnamed.region_id is None


def test_schema_valid_output_without_a_source_file_keeps_no_invented_evidence(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    output = _write_output(
        controller, request, _first_pass_body(request, _agreeing_pages())
    )
    before = output.read_bytes()
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    assert output.read_bytes() == before
    state = ReviewStore(tmp_path / "reviews", BATCH_ID).load()
    assert state is not None
    runtime = next(
        item for item in state.documents[0].first_pass if item.provider == "gpt"
    )
    assert runtime.source_evidence == []


def test_source_evidence_survives_checkpoint_replay_and_the_report(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    gpt_request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    grok_request = controller.publish(BATCH_ID, DOCUMENT_ID, "grok")
    gpt_output = _write_output(
        controller, gpt_request, _first_pass_body(gpt_request, _agreeing_pages())
    )
    _write_output(
        controller, grok_request, _first_pass_body(grok_request, _agreeing_pages())
    )
    source = {
        "provider": "gpt",
        "pages": [
            {
                "page_id": "page_001",
                "regions": [{"region_id": "r1", "lines": [{"line_id": "r1-l1"}]}],
                "unreadable_spans": [
                    {
                        "line_id": "r1-l1",
                        "alternatives": ["vie", "(cancelled stroke)", "y"],
                        "reason": "single stroke overwritten",
                    }
                ],
            }
        ],
        "attention_items": [
            {
                "page_id": "page_001",
                "region_id": "r1",
                "description": "reverse-side show-through is visible",
            }
        ],
    }
    source_path = tmp_path / "original_gpt.json"
    source_bytes = (json.dumps(source) + "\n").encode("utf-8")
    source_path.write_bytes(source_bytes)
    accepted = gpt_output.read_bytes()
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", source_json=source_path)
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok")
    assert source_path.read_bytes() == source_bytes
    assert gpt_output.read_bytes() == accepted
    ReviewStore(tmp_path / "reviews", BATCH_ID).checkpoint_path.unlink()
    restarted = ReviewController(
        batches_dir=tmp_path / "batches",
        reviews_dir=tmp_path / "reviews",
    )
    paths = restarted.finalize(BATCH_ID, DOCUMENT_ID)
    report = ReviewReport.model_validate_json(paths["review_report.json"].read_bytes())
    values = {item.value for item in report.source_evidence}
    assert '"vie"' in values
    assert '"(cancelled stroke)"' in values
    assert '"y"' in values
    assert '"reverse-side show-through is visible"' in values
    assert {item.provider for item in report.source_evidence} == {"gpt"}
    span = next(item for item in report.source_evidence if item.value == '"vie"')
    assert span.line_id == "r1-l1"
    assert span.region_id == "r1"
    assert span.original_sha256 == sha256_bytes(source_bytes)
    replayed = ReviewStore(tmp_path / "reviews", BATCH_ID).load()
    assert replayed is not None
    gpt = next(
        item for item in replayed.documents[0].first_pass if item.provider == "gpt"
    )
    assert any(item.value == '"vie"' for item in gpt.source_evidence)


def test_real_originals_keep_dropped_evidence_and_their_bytes() -> None:
    root = Path(
        "/mnt/c/Users/User/vs-archive-transcription-review-agent/"
        "desktop_pilot/local_image_worker_integration_01"
    )
    pairs = (
        (
            "gpt",
            root / "first_pass/output/transcription_gpt.json",
            root / "first_pass/schema_repair/gpt/transcription_gpt.schema_repair.json",
            "b71599aece333382d145b40c6c23dca7f33e4510161e0a2912efc61da97255fd",
        ),
        (
            "grok",
            root / "first_pass/output/transcription_grok.json",
            root
            / "first_pass/schema_repair/grok/transcription_grok.schema_repair.json",
            "a59ae837ecc87e332bc4b52a7664588a1278325fdc9738777f4bbc7b72f710be",
        ),
    )
    if any(
        not original.is_file() or not repaired.is_file()
        for _, original, repaired, _ in pairs
    ):
        pytest.skip("real provider files are not on this machine")
    for provider, original_path, repaired_path, digest in pairs:
        original_bytes = original_path.read_bytes()
        repaired_bytes = repaired_path.read_bytes()
        assert sha256_bytes(original_bytes) == digest
        repaired = FirstPassOutput.model_validate_json(repaired_bytes)
        evidence = collect_source_evidence(
            json.loads(original_bytes),
            repaired,
            original_sha256=digest,
        )
        assert evidence
        assert {item.provider for item in evidence} == {provider}
        assert {item.original_sha256 for item in evidence} == {digest}
        if provider == "grok":
            blob = "\n".join(item.value for item in evidence)
            assert '"vie"' in blob
            assert '"rref"' in blob
            assert '"(cancelled stroke)"' in blob
            assert '"y"' in blob
            assert "Hand-drawn oval" in blob
        if provider == "gpt":
            names = next(
                item
                for item in evidence
                if "membership list require human review" in item.value
            )
            assert names.line_id == "page_001-r02-l07"
            assert names.region_id == "page_001-r02"
            oval = next(
                item
                for item in evidence
                if "blue oval surrounds the date" in item.value
            )
            assert oval.region_id is None
            assert oval.line_id is None
        assert original_path.read_bytes() == original_bytes
        assert repaired_path.read_bytes() == repaired_bytes
