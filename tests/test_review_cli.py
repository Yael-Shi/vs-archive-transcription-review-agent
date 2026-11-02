"""Local review-next operations do not scan or call a model."""

from __future__ import annotations

import json

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _agreeing_pages,
    _controller,
    _deliver_first_pass,
)
from transcription_review_agent.cli import main
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.paths import document_root


def _args(tmp_path, operation: str, *extra: str) -> list[str]:
    return [
        operation,
        "--batches-dir",
        str(tmp_path / "batches"),
        "--reviews-dir",
        str(tmp_path / "reviews"),
        "--batch-id",
        BATCH_ID,
        "--document-id",
        str(DOCUMENT_ID),
        *extra,
    ]


def test_review_next_prepares_accepts_and_finalizes_without_discovery(
    tmp_path, capsys
) -> None:
    seeded = _controller(tmp_path)
    del seeded
    assert main([]) == 2
    assert "No VS Archive request is made" in capsys.readouterr().err

    assert main(_args(tmp_path, "prepare")) == 0
    assert main(_args(tmp_path, "status")) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["execution_state"] == "WAITING_FOR_PROVIDER"
    assert status["providers"]["gpt"] == "AWAITING_MANUAL_DISPATCH"

    controller = ReviewController(
        batches_dir=tmp_path / "batches",
        reviews_dir=tmp_path / "reviews",
    )
    _deliver_first_pass(controller, "gpt", _agreeing_pages())
    _deliver_first_pass(controller, "grok", _agreeing_pages())
    assert main(_args(tmp_path, "finalize")) == 0
    final_dir = document_root(tmp_path / "reviews", BATCH_ID, DOCUMENT_ID) / "final"
    report = json.loads((final_dir / "review_report.json").read_text(encoding="utf-8"))
    assert report["review_completion"] == "completed"
    assert report["production_writeback"] is False
    assert main(_args(tmp_path, "prepare")) == 0
