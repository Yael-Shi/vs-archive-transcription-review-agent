"""Document resume from sealed image through packet, with fixture challenge files.

The first-pass bytes stand in for desktop output. Challenge bytes are fixtures.
Nothing here calls a model or treats the sealed archive text as a reading.
"""

from __future__ import annotations

import json
from pathlib import Path

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.config import ReviewConfig
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.document_run import advance_document
from transcription_review_agent.review.models import WorkRequest
from test_review_workflow import BATCH_ID, DOCUMENT_ID, NOW, OCR, _png, _seal
from test_desktop_e2e import QUEUE, _fixture, _mount

GPT_TEXT = "BLUE HARBOUR OPENS AT 12"
GROK_TEXT = "BLUE HARBOR OPENS AT 42"


def _controller(tmp_path: Path) -> ReviewController:
    image = _png(8, 8)
    assert OCR.encode("utf-8") not in image
    _seal(tmp_path / "batches", [image])
    return ReviewController(
        batches_dir=tmp_path / "batches",
        reviews_dir=tmp_path / "reviews",
        config=ReviewConfig(
            max_challenge_rounds=1,
            max_document_challenge_dispatches=2,
        ),
        clock=lambda: NOW,
    )


def _advance(tmp_path: Path, controller: ReviewController, **kwargs: object):
    return advance_document(
        controller,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        preservation_dir=tmp_path / "raw",
        mount_root=_mount(tmp_path),
        **kwargs,  # type: ignore[arg-type]
    )


def _queue_document(tmp_path: Path) -> Path:
    return (
        _mount(tmp_path)
        / "d"
        / "archive-queue"
        / "tasks"
        / BATCH_ID
        / f"document_{DOCUMENT_ID}"
    )


def _write_first_pass(tmp_path: Path, controller: ReviewController) -> dict[str, bytes]:
    from transcription_review_agent.review.desktop_e2e import (
        register_desktop_first_pass,
    )

    registered = register_desktop_first_pass(
        controller,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        mount_root=_mount(tmp_path),
    )
    gpt_request, grok_request = registered.work_requests
    raw = {
        "gpt": _fixture(gpt_request, GPT_TEXT),
        "grok": _fixture(grok_request, GROK_TEXT),
    }
    registered.queue.gpt.result_path.write_bytes(raw["gpt"])
    registered.queue.grok.result_path.write_bytes(raw["grok"])
    return raw


def _challenge_file(task: Path, decision: str) -> bytes:
    request = WorkRequest.model_validate_json(
        (task / "work_request.json").read_text(encoding="utf-8")
    )
    published = json.loads(
        (task / "input" / "disputes.json").read_text(encoding="utf-8")
    )
    decisions = []
    for item in published["disputes"]:
        group = next(
            group
            for group in published["groups"]
            if item["dispute_id"] in group["dispute_ids"]
        )
        decisions.append(
            {
                "decision": decision,
                "dispute_id": item["dispute_id"],
                "evidence_assessment": "SUPPORTS_DECISION",
                "evidence_ref": group["evidence_refs"][0],
                "visual_evidence": (
                    f"{request.provider} inspected pixels for {item['dispute_id']}"
                ),
            }
        )
    payload = {
        "attempt": request.attempt,
        "batch_id": request.batch_id,
        "decisions": decisions,
        "document_id": request.document_id,
        "job_id": request.job_id,
        "provider": request.provider,
        "provider_interface": request.provider_interface,
        "round_index": request.round_index,
        "schema_version": "review-provider-output-v1",
        "source_bundle_sha256": request.source_bundle_sha256,
        "stage": "challenge",
        "task_id": request.task_id,
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def test_resume_preserves_disagreement_and_builds_packet_from_fixtures(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    raw = _write_first_pass(tmp_path, controller)
    first = _advance(tmp_path, controller)
    assert first.stopped_at == "waiting_for_challenge"
    assert first.blocker == "desktop_challenge"
    assert [item["executable"] for item in first.execution] == [False, False]
    assert {item["provider"] for item in first.execution} == {"gpt", "grok"}
    assert first.finalized is False
    assert first.archive_transcription_used_as_reading is False
    assert first.ledger_consistent is True
    assert {item["gpt_text"] for item in first.disputes} == {GPT_TEXT}
    assert {item["grok_text"] for item in first.disputes} == {GROK_TEXT}
    assert any(item["substantive"] for item in first.disputes)
    document = _queue_document(tmp_path)
    for provider, payload in raw.items():
        assert (
            document / provider / "output" / f"transcription_{provider}.json"
        ).read_bytes() == payload
    challenge_root = document.parent / f"document_{DOCUMENT_ID}_challenge_round_01"
    assert (
        OCR.encode("utf-8") not in (challenge_root / "gpt" / "prompt.txt").read_bytes()
    )
    assert (
        OCR.encode("utf-8") not in (challenge_root / "grok" / "prompt.txt").read_bytes()
    )
    assert (
        b"archive transcription"
        in (challenge_root / "gpt" / "prompt.txt").read_bytes().lower()
        or b"Do not use an archive transcription"
        in (challenge_root / "gpt" / "prompt.txt").read_bytes()
    )
    assert "Ask every time" in (challenge_root / "grok" / "prompt.txt").read_text(
        encoding="utf-8"
    )
    assert "working directory" in (challenge_root / "grok" / "prompt.txt").read_text(
        encoding="utf-8"
    )
    repeated = _advance(tmp_path, controller)
    assert repeated.ledger_sequence == first.ledger_sequence
    assert repeated.blocker == "desktop_challenge"
    restarted = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        config=controller.config,
        clock=lambda: NOW,
    )
    resumed = _advance(tmp_path, restarted)
    assert resumed.ledger_sequence == first.ledger_sequence
    (challenge_root / "gpt" / "output" / "challenge_gpt.json").write_bytes(
        _challenge_file(challenge_root / "gpt", "CANDIDATE_X")
    )
    (challenge_root / "grok" / "output" / "challenge_grok.json").write_bytes(
        _challenge_file(challenge_root / "grok", "CANDIDATE_Y")
    )
    collected = _advance(tmp_path, restarted)
    assert collected.stopped_at == "waiting_for_human"
    assert collected.blocker == "human_challenge_closure"
    assert collected.finalized is False
    assert collected.challenge_reconciled is True
    assert collected.decisions
    assert all(item["decision"] == "UNRESOLVED" for item in collected.decisions)
    assert all(item["agreed"] is False for item in collected.decisions)
    assert all(
        item["result_class"] != "AUTOMATICALLY_SUPPORTED"
        for item in collected.decisions
    )
    held = _advance(tmp_path, restarted)
    assert held.ledger_sequence == collected.ledger_sequence
    pending_dir = tmp_path / "pending"
    pending = _advance(tmp_path, restarted, packet_dir=pending_dir)
    assert pending.stopped_at == "waiting_for_human"
    assert pending.finalized is False
    assert pending.packet_dir == str(pending_dir)
    assert pending.ledger_sequence == collected.ledger_sequence
    pending_page = (pending_dir / "index.html").read_text(encoding="utf-8")
    assert "Pending human review" in pending_page
    assert "No reading is approved" in pending_page
    assert "UNRESOLVED" in pending_page
    assert "Agreement is not human verification" in pending_page
    assert "Human verification is not recorded" in pending_page
    assert "not eligible for publication" in pending_page
    assert GPT_TEXT in pending_page
    assert GROK_TEXT in pending_page
    assert "approved a reading" not in pending_page.lower()
    for provider, payload in raw.items():
        assert (
            document / provider / "output" / f"transcription_{provider}.json"
        ).read_bytes() == payload
    packet = tmp_path / "packet"
    closed = _advance(
        tmp_path,
        restarted,
        closure_mode="SELECTED_ROUNDS_COMPLETE",
        closure_reason=(
            "Fixture challenge round is complete. This records no human approval."
        ),
        packet_dir=packet,
    )
    assert closed.stopped_at == "human_packet_written"
    assert closed.finalized is True
    assert closed.blocker is None
    page = (packet / "index.html").read_text(encoding="utf-8")
    assert GPT_TEXT in page
    assert GROK_TEXT in page
    assert OCR not in page
    assert "לא נרשמה החלטה" in page or "אין כאן אישור" in page
    again = _advance(tmp_path, restarted, packet_dir=packet)
    assert again.ledger_sequence == closed.ledger_sequence
    assert sha256_bytes((packet / "index.html").read_bytes()) == sha256_bytes(
        (packet / "index.html").read_bytes()
    )
