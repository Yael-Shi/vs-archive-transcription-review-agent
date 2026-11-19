"""Collector checks for raw desktop results. No model or desktop app is called."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.desktop_result_intake import (
    collect_desktop_first_pass,
)
from transcription_review_agent.review.errors import ReviewIntegrityError
from transcription_review_agent.review.models import FirstPassOutput
from transcription_review_agent.review.paths import document_root
from test_desktop_file_queue import QUEUE, _publish
from test_original_image_stage import (
    BATCH_ID,
    DOCUMENT_ID,
    TRANSCRIPTION,
    _approved,
    _png,
)


def _reading(task_dir: Path, text: str, **updates: object) -> bytes:
    example = json.loads((task_dir / "output_example.json").read_text(encoding="utf-8"))

    def walk(value: object) -> object:
        if isinstance(value, str):
            return text if "TEMPLATE" in value else value
        if isinstance(value, list):
            return [walk(item) for item in value]
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        return value

    payload = walk(example)
    assert isinstance(payload, dict)
    payload.update(updates)
    if not updates:
        FirstPassOutput.model_validate(payload)
    return json.dumps(payload).encode("utf-8") + b"\n"


def _collect(tmp_path: Path, published, sealed: Path):
    return collect_desktop_first_pass(
        queue_root=QUEUE,
        sealed_document_dir=sealed,
        work_requests=_approved(tmp_path / "batches"),
        reviews_dir=tmp_path / "reviews",
        preservation_dir=tmp_path / "raw",
        mount_root=tmp_path / "mnt",
    )


def test_ready_results_are_copied_once_and_isolated(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    gpt_bytes = _reading(published.gpt.directory, "BLUE HARBOR OPENS AT 42")
    grok_bytes = _reading(published.grok.directory, "BLUE HARBOR OPENS AT 42.")
    published.gpt.result_path.write_bytes(gpt_bytes)
    published.grok.result_path.write_bytes(grok_bytes)
    image = published.gpt.image_path.read_bytes()
    report = _collect(tmp_path, published, sealed)
    assert report.both_ready_for_accept is True
    assert report.gpt.review_output_path.read_bytes() == gpt_bytes
    assert report.grok.review_output_path.read_bytes() == grok_bytes
    assert report.gpt.preservation_path is not None
    assert report.gpt.preservation_path.read_bytes() == gpt_bytes
    assert b"BLUE HARBOR OPENS AT 42." not in report.gpt.review_output_path.read_bytes()
    assert TRANSCRIPTION.encode("utf-8") not in published.gpt.prompt_path.read_bytes()
    assert published.gpt.image_path.read_bytes() == image
    assert report.gpt.review_output_path == (
        document_root(tmp_path / "reviews", BATCH_ID, DOCUMENT_ID)
        / "first_pass"
        / "output"
        / "transcription_gpt.json"
    )
    again = _collect(tmp_path, published, sealed)
    assert again.gpt.state == "unchanged"
    assert again.grok.state == "unchanged"
    assert again.gpt.review_output_path.read_bytes() == gpt_bytes
    assert again.gpt.raw_sha256 == sha256_bytes(gpt_bytes)


def test_missing_and_partial_outputs_collect_nothing_for_the_absent_provider(
    tmp_path: Path,
) -> None:
    published, sealed = _publish(tmp_path, _png())
    published.gpt.result_path.write_bytes(
        _reading(published.gpt.directory, "BLUE HARBOR OPENS AT 42")
    )
    report = _collect(tmp_path, published, sealed)
    assert report.gpt.state == "ready"
    assert report.grok.state == "missing"
    assert report.both_ready_for_accept is False
    assert not report.grok.review_output_path.exists()
    assert report.gpt.review_output_path.is_file()


def test_malformed_json_is_preserved_and_not_handed_off(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    raw = b"{not json"
    published.gpt.result_path.write_bytes(raw)
    published.grok.result_path.write_bytes(
        _reading(published.grok.directory, "BLUE HARBOR OPENS AT 42")
    )
    report = _collect(tmp_path, published, sealed)
    assert report.gpt.state == "invalid"
    assert report.gpt.failure == "provider output is not JSON"
    assert report.gpt.preservation_path is not None
    assert report.gpt.preservation_path.read_bytes() == raw
    assert not report.gpt.review_output_path.exists()
    assert report.grok.state == "ready"


def test_identity_mismatch_is_preserved_and_not_handed_off(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    raw = _reading(published.gpt.directory, "BLUE HARBOR OPENS AT 42", task_id="other")
    published.gpt.result_path.write_bytes(raw)
    report = _collect(tmp_path, published, sealed)
    assert report.gpt.schema_valid is True
    assert report.gpt.identity_matches is False
    assert report.gpt.failure == "provider output does not match the published task"
    assert report.gpt.preservation_path is not None
    assert report.gpt.preservation_path.read_bytes() == raw
    assert not report.gpt.review_output_path.exists()


def test_changed_result_is_not_overwritten(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    first = _reading(published.gpt.directory, "BLUE HARBOR OPENS AT 42")
    second = _reading(published.gpt.directory, "A DIFFERENT READING")
    published.gpt.result_path.write_bytes(first)
    published.grok.result_path.write_bytes(
        _reading(published.grok.directory, "BLUE HARBOR OPENS AT 42")
    )
    _collect(tmp_path, published, sealed)
    published.gpt.result_path.write_bytes(second)
    with pytest.raises(ReviewIntegrityError, match="refusing to change"):
        _collect(tmp_path, published, sealed)
    review = (
        document_root(tmp_path / "reviews", BATCH_ID, DOCUMENT_ID)
        / "first_pass"
        / "output"
        / "transcription_gpt.json"
    )
    assert review.read_bytes() == first
    assert published.gpt.result_path.read_bytes() == second


def test_intake_module_does_not_launch_a_model_or_cli() -> None:
    source = Path(
        "src/transcription_review_agent/review/desktop_result_intake.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = ("subprocess", "codex_image_transcription", "openai", "grok_desktop")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(alias.name for alias in node.names)]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names = [node.func.attr]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names = [node.func.id]
        else:
            continue
        assert not any(name in {"system", "Popen", "exec", *banned} for name in names)
