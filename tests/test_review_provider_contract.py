"""Contract fixtures for provider files.

These are not desktop end-to-end captures. They check that incompatible GPT
Work and Grok Bot replies are rejected, and that a schema-valid file can be
accepted without inventing missing coverage, lines, or visual evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _agreeing_pages,
    _controller,
    _first_pass_body,
    _write_output,
)
from transcription_review_agent.review.errors import ReviewOutputRejected

FIXTURES = Path(__file__).parent / "fixtures" / "provider_contracts"


def _reject(tmp_path: Path, name: str) -> ReviewOutputRejected:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    target = _write_output(
        controller,
        request_path,
        json.loads(FIXTURES.joinpath(name).read_text(encoding="utf-8")),
    )
    del target
    with pytest.raises(ReviewOutputRejected) as caught:
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    return caught.value


def test_prose_and_chat_wrappers_are_rejected(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    output = _write_output(controller, request_path, {"unused": True})
    output.write_text(
        FIXTURES.joinpath("gpt_work_prose.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ReviewOutputRejected) as caught:
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    assert caught.value.reason == "schema"


def test_partial_provider_json_reports_missing_mandatory_fields(tmp_path: Path) -> None:
    rejected = _reject(tmp_path, "gpt_work_partial.json")
    assert rejected.reason == "schema"
    message = str(rejected)
    assert "missing mandatory fields:" in message
    assert "task_id" in message
    assert "pages" in message


def test_grok_chat_envelope_is_not_adapted_into_a_transcription(tmp_path: Path) -> None:
    rejected = _reject(tmp_path / "grok", "grok_bot_message.json")
    assert rejected.reason == "schema"
    assert "missing mandatory fields:" in str(rejected)
    assert "pages" in str(rejected)


def test_model_reported_confidence_is_rejected(tmp_path: Path) -> None:
    rejected = _reject(tmp_path, "grok_with_confidence.json")
    assert rejected.reason == "schema"
    assert "missing mandatory fields" not in str(rejected)


def test_schema_valid_output_is_accepted_without_added_fields(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    body = _first_pass_body(request_path, _agreeing_pages())
    _write_output(controller, request_path, body)
    status = controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    assert status.providers["gpt"] == "SUCCEEDED"
    stored = (
        controller.reviews_dir
        / f"batch_{BATCH_ID}"
        / "documents"
        / f"document_{DOCUMENT_ID}"
        / "first_pass"
        / "output"
        / "transcription_gpt.json"
    )
    assert json.loads(stored.read_text(encoding="utf-8")) == body
