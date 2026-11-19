"""Ledger, desktop queue, and accept using fixture files. No model is called."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.desktop_e2e import (
    accept_desktop_first_pass,
    main,
    register_desktop_first_pass,
)
from transcription_review_agent.review.models import (
    CoverageReport,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    TextRegion,
    WorkRequest,
)
from test_review_workflow import BATCH_ID, DOCUMENT_ID, NOW, OCR, _png, _seal

QUEUE = "D:/archive-queue/tasks"


def _controller(tmp_path: Path) -> tuple[ReviewController, bytes]:
    image = _png(8, 8)
    assert OCR.encode("utf-8") not in image
    _seal(tmp_path / "batches", [image])
    controller = ReviewController(
        batches_dir=tmp_path / "batches",
        reviews_dir=tmp_path / "reviews",
        clock=lambda: NOW,
    )
    return controller, image


def _mount(tmp_path: Path) -> Path:
    mount = tmp_path / "mnt"
    (mount / "d").mkdir(parents=True, exist_ok=True)
    return mount


def _fixture(request: WorkRequest, text: str, **updates: object) -> bytes:
    page = PageTranscription(
        page_id=request.page_ids[0],
        reading_order=1,
        diplomatic_transcription=text,
        regions=[
            TextRegion(
                region_id="r1",
                order=1,
                lines=[LineReading(line_id="r1-l1", text=text)],
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
    model = FirstPassOutput(
        provider=request.provider,
        provider_interface=request.provider_interface,
        task_id=request.task_id,
        job_id=request.job_id,
        batch_id=request.batch_id,
        document_id=request.document_id,
        attempt=request.attempt,
        source_bundle_sha256=request.source_bundle_sha256,
        pages=[page],
    )
    payload = json.loads(canonical_json(model))
    payload.update(updates)
    return json.dumps(payload).encode("utf-8") + b"\n"


def _register(tmp_path: Path, controller: ReviewController):
    return register_desktop_first_pass(
        controller,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        mount_root=_mount(tmp_path),
    )


def _accept(tmp_path: Path, controller: ReviewController):
    return accept_desktop_first_pass(
        controller,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        preservation_dir=tmp_path / "raw",
        mount_root=_mount(tmp_path),
    )


def test_registered_tasks_are_accepted_from_exact_fixture_bytes(tmp_path: Path) -> None:
    controller, image = _controller(tmp_path)
    registered = _register(tmp_path, controller)
    again = _register(tmp_path, controller)
    assert [item.task_id for item in again.work_requests] == [
        item.task_id for item in registered.work_requests
    ]
    gpt_request, grok_request = registered.work_requests
    gpt_task = registered.queue.gpt
    grok_task = registered.queue.grok
    assert (gpt_task.directory / "work_request.json").read_bytes() == canonical_json(
        gpt_request
    ).encode("utf-8")
    assert OCR.encode("utf-8") not in gpt_task.prompt_path.read_bytes()
    assert OCR.encode("utf-8") not in grok_task.prompt_path.read_bytes()
    assert gpt_task.image_path.read_bytes() == image
    gpt_raw = _fixture(gpt_request, "BLUE HARBOR OPENS AT 42")
    grok_raw = _fixture(grok_request, "BLUE HARBOR OPENS AT 42.")
    assert gpt_raw not in image and grok_raw not in image
    gpt_task.result_path.write_bytes(gpt_raw)
    grok_task.result_path.write_bytes(grok_raw)
    accepted = _accept(tmp_path, controller)
    assert accepted.comparison_ready is True
    assert accepted.finalized is False
    assert accepted.execution_state == "IN_PROGRESS"
    assert accepted.providers == {"gpt": "SUCCEEDED", "grok": "SUCCEEDED"}
    assert gpt_task.result_path.read_bytes() == gpt_raw
    assert accepted.intake.gpt.preservation_path is not None
    assert accepted.intake.gpt.preservation_path.read_bytes() == gpt_raw
    assert accepted.intake.gpt.review_output_path.read_bytes() == gpt_raw
    assert grok_raw not in accepted.intake.gpt.review_output_path.read_bytes()
    repeated = _accept(tmp_path, controller)
    assert repeated.ledger_sequence == accepted.ledger_sequence
    assert repeated.intake.gpt.state == "unchanged"
    assert gpt_task.image_path.read_bytes() == image
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    status = controller.status(BATCH_ID, DOCUMENT_ID)
    assert status.finalized is False
    assert status.challenge_rounds == 0
    assert comparison.disputes


def test_one_missing_provider_is_left_waiting(tmp_path: Path) -> None:
    controller, _image = _controller(tmp_path)
    registered = _register(tmp_path, controller)
    registered.queue.gpt.result_path.write_bytes(
        _fixture(registered.work_requests[0], "BLUE HARBOR OPENS AT 42")
    )
    accepted = _accept(tmp_path, controller)
    assert accepted.comparison_ready is False
    assert accepted.providers["gpt"] == "SUCCEEDED"
    assert accepted.providers["grok"] == "AWAITING_MANUAL_DISPATCH"
    assert accepted.notes["grok"] == "provider output is missing"
    assert accepted.finalized is False


def test_invalid_and_mismatched_outputs_are_not_accepted(tmp_path: Path) -> None:
    controller, _image = _controller(tmp_path)
    registered = _register(tmp_path, controller)
    gpt_request = registered.work_requests[0]
    malformed = b"{not json"
    registered.queue.gpt.result_path.write_bytes(malformed)
    registered.queue.grok.result_path.write_bytes(
        _fixture(
            registered.work_requests[1], "BLUE HARBOR OPENS AT 42", task_id="other"
        )
    )
    accepted = _accept(tmp_path, controller)
    assert accepted.providers == {
        "gpt": "AWAITING_MANUAL_DISPATCH",
        "grok": "AWAITING_MANUAL_DISPATCH",
    }
    assert accepted.intake.gpt.failure == "provider output is not JSON"
    assert accepted.intake.grok.identity_matches is False
    assert accepted.intake.gpt.preservation_path is not None
    assert accepted.intake.gpt.preservation_path.read_bytes() == malformed
    assert not accepted.intake.gpt.review_output_path.exists()
    assert not accepted.intake.grok.review_output_path.exists()
    assert (
        gpt_request.task_id
        in (registered.queue.gpt.directory / "work_request.json").read_text()
    )


def test_cli_register_and_accept_use_the_same_tasks(tmp_path: Path, capsys) -> None:
    image = _png(8, 8)
    _seal(tmp_path / "batches", [image])
    mount = _mount(tmp_path)
    common = [
        "--batches-dir",
        str(tmp_path / "batches"),
        "--reviews-dir",
        str(tmp_path / "reviews"),
        "--batch-id",
        BATCH_ID,
        "--document-id",
        str(DOCUMENT_ID),
        "--queue-root",
        QUEUE,
        "--mount-root",
        str(mount),
    ]
    assert main(["register", *common]) == 0
    payload = json.loads(capsys.readouterr().out)
    document = (
        mount / "d" / "archive-queue" / "tasks" / BATCH_ID / f"document_{DOCUMENT_ID}"
    )
    gpt_request = WorkRequest.model_validate_json(
        (document / "gpt" / "work_request.json").read_text(encoding="utf-8")
    )
    grok_request = WorkRequest.model_validate_json(
        (document / "grok" / "work_request.json").read_text(encoding="utf-8")
    )
    assert payload["task_ids"]["gpt"] == gpt_request.task_id
    (document / "gpt" / "output" / "transcription_gpt.json").write_bytes(
        _fixture(gpt_request, "BLUE HARBOR OPENS AT 42")
    )
    (document / "grok" / "output" / "transcription_grok.json").write_bytes(
        _fixture(grok_request, "BLUE HARBOR OPENS AT 42")
    )
    assert (
        main(
            [
                "accept",
                *common,
                "--preservation-dir",
                str(tmp_path / "raw"),
            ]
        )
        == 0
    )
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["comparison_ready"] is True
    assert accepted["finalized"] is False


def test_desktop_e2e_does_not_launch_a_model_or_cli() -> None:
    source = Path("src/transcription_review_agent/review/desktop_e2e.py").read_text(
        encoding="utf-8"
    )
    banned = {"subprocess", "codex_image_transcription", "openai", "grok_desktop"}
    for node in ast.walk(ast.parse(source)):
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
        assert not any(name in {"system", "Popen", *banned} for name in names)
