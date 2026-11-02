"""An interim report stays in history and does not close the document."""

from __future__ import annotations

import json

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    NOW,
    _agreeing_pages,
    _close_challenge,
    _controller,
    _deliver_first_pass,
    _page,
    _write_output,
)
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.models import (
    BBox,
    ChallengeDecision,
    ChallengeOutput,
    WorkRequest,
)
from transcription_review_agent.review.paths import document_root


def _ledger(controller: ReviewController) -> str:
    return (
        controller.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "ledger.jsonl"
    ).read_text(encoding="utf-8")


def test_single_provider_provisional_resumes_and_can_be_completed(tmp_path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(controller, "gpt", _agreeing_pages("Only one reading."))
    first = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert "final/versions/v001/" in first["review_report.json"].as_posix()
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is False
    saved = {name: path.read_bytes() for name, path in first.items()}
    report = json.loads(saved["review_report.json"])
    assert report["review_completion"] == "interim_provisional"
    assert _ledger(controller).count('"event_type":"review_provisional"') == 1
    assert _ledger(controller).count('"event_type":"review_finalized"') == 0
    repeated = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert {name: path.read_bytes() for name, path in repeated.items()} == saved
    assert _ledger(controller).count('"event_type":"review_provisional"') == 1

    checkpoint = (
        controller.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "state.json"
    )
    checkpoint.unlink()
    resumed = ReviewController(
        batches_dir=tmp_path / "batches",
        reviews_dir=tmp_path / "reviews",
        clock=lambda: NOW,
    )
    status = resumed.status(BATCH_ID, DOCUMENT_ID)
    assert status.finalized is False
    assert status.providers["gpt"] == "SUCCEEDED"
    assert {name: path.read_bytes() for name, path in first.items()} == saved

    _deliver_first_pass(resumed, "grok", _agreeing_pages("Only one reading."))
    completed = resumed.finalize(BATCH_ID, DOCUMENT_ID)
    completed_bytes = {name: path.read_bytes() for name, path in completed.items()}
    assert {name: path.read_bytes() for name, path in first.items()} == saved
    assert completed_bytes != saved
    final_report = json.loads(completed_bytes["review_report.json"])
    assert final_report["review_completion"] == "completed"
    assert resumed.status(BATCH_ID, DOCUMENT_ID).finalized is True
    version = (
        document_root(resumed.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "final"
        / "versions"
        / "v002"
    )
    assert {
        name: (version / name).read_bytes() for name in completed
    } == completed_bytes
    assert _ledger(resumed).count('"event_type":"review_provisional"') == 1
    assert _ledger(resumed).count('"event_type":"review_finalized"') == 1
    resumed.finalize(BATCH_ID, DOCUMENT_ID)
    assert _ledger(resumed).count('"event_type":"review_finalized"') == 1


def test_skipped_challenge_can_be_completed_without_replacing_the_interim(
    tmp_path,
) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    bbox = BBox(x=4, y=4, width=4, height=4)
    pages = {
        "gpt": [
            _page("page_001", 1, "Ionnes", name="Ionnes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
        "grok": [
            _page("page_001", 1, "Iannes", name="Iannes", bbox=bbox),
            _page("page_002", 2, "Second page remains."),
        ],
    }
    _deliver_first_pass(controller, "gpt", pages["gpt"])
    _deliver_first_pass(controller, "grok", pages["grok"])
    controller.compare(BATCH_ID, DOCUMENT_ID)
    interim = controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)
    interim_bytes = {name: path.read_bytes() for name, path in interim.items()}
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is False
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
                    decision="CANDIDATE_X",
                    visual_evidence="The stroke is a single vertical.",
                )
                for dispute_id in dispute_ids
            ],
        )
        _write_output(controller, request_path, challenge.model_dump(mode="json"))
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    _close_challenge(controller)
    completed = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert {name: path.read_bytes() for name, path in interim.items()} == interim_bytes
    report = json.loads(completed["review_report.json"].read_text(encoding="utf-8"))
    assert report["review_completion"] == "completed"
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is True
    assert _ledger(controller).count('"event_type":"review_provisional"') == 1
    assert _ledger(controller).count('"event_type":"review_finalized"') == 1
