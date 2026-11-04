"""Provider output contract, read-only preflight, and invalid-output recovery."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from test_review_a3_controller_integration import _deliver, _requests
from test_review_challenge_groups import _document, _published, _root, _setup
from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _controller,
    _deliver_first_pass,
    _page,
    _snapshot,
)
from transcription_review_agent.cli import main
import transcription_review_agent.review.controller as review_controller
from transcription_review_agent.review.controller import (
    _grouped_challenge_note,
    _instructions,
    _research_input_matches_contract,
    _schema_text,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewOutputRejected,
    ReviewPolicyError,
)
from transcription_review_agent.review.models import (
    BBox,
    ChallengeOutput,
    FirstPassOutput,
    WorkRequest,
)
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.provider_output_contract import (
    challenge_instructions,
    challenge_schema_text,
    legacy_challenge_instructions,
    legacy_challenge_schema_text,
)
from transcription_review_agent.review.runtime import (
    MAX_INVALID_OUTPUT_RECOVERIES,
    ReviewStore,
)


def _ledger(controller) -> list[dict[str, object]]:
    path = controller.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _events(controller, event_type: str) -> list[dict[str, object]]:
    return [item for item in _ledger(controller) if item["event_type"] == event_type]


def _gpt(controller):
    return next(
        item
        for item in _document(controller).challenge_rounds[0].providers
        if item.provider == "gpt"
    )


def _grok(controller):
    return next(
        item
        for item in _document(controller).challenge_rounds[0].providers
        if item.provider == "grok"
    )


def _invalidate(controller, request: Path) -> bytes:
    target = (
        _root(controller)
        / json.loads(request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"provider": "gpt", "provider_interface": "GPT_WORK"}),
        encoding="utf-8",
    )
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    return target.read_bytes()


def test_generated_challenge_example_matches_the_validation_model(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request_path = _requests(controller, [names["word_r1"], names["note_r1"]])["gpt"]
    task = request_path.parent
    request = WorkRequest.model_validate_json(request_path.read_bytes())
    example = json.loads((task / "output_example.json").read_text(encoding="utf-8"))
    schema = json.loads((task / "output_schema.json").read_text(encoding="utf-8"))
    model = ChallengeOutput.model_validate(example)
    required = {
        name
        for name, field in ChallengeOutput.model_fields.items()
        if field.is_required()
    }
    assert required <= set(example)
    assert set(schema["required_top_level_fields"]) >= required
    assert schema["round_contract"] == "EVIDENCE_BOUND_V1"
    assert "evidence_ref" in schema["decisions"]["fields"]
    assert "evidence_assessment" in schema["decisions"]["fields"]
    instructions = (task / "instructions_gpt.txt").read_text(encoding="utf-8")
    assert (
        instructions
        == _instructions("challenge", challenge_contract="EVIDENCE_BOUND_V1")
        + _grouped_challenge_note()
    )
    assert "provider_interface" in instructions
    assert "job_id" in instructions
    assert "output_example.json" in instructions
    for field in (
        "provider",
        "provider_interface",
        "task_id",
        "job_id",
        "batch_id",
        "document_id",
        "attempt",
        "source_bundle_sha256",
        "stage",
        "round_index",
    ):
        assert example[field] == getattr(request, field)
    assert [item["dispute_id"] for item in example["decisions"]] == [
        item.dispute_id for item in _published(controller).disputes
    ]
    assert all(item.evidence_ref is not None for item in model.decisions)
    assert all(item.evidence_assessment == "INSUFFICIENT" for item in model.decisions)
    assert _schema_text("challenge", challenge_contract="EVIDENCE_BOUND_V1") == (
        task / "output_schema.json"
    ).read_text(encoding="utf-8")


def test_historical_challenge_schema_still_accepts(tmp_path: Path) -> None:
    controller, _comparison, names = _setup(tmp_path)
    selected = [names["word_r1"]]
    request = _requests(controller, selected)["gpt"]
    task = request.parent
    (task / "output_schema.json").write_text(
        legacy_challenge_schema_text(evidence_bound=True), encoding="utf-8"
    )
    (task / "instructions_gpt.txt").write_text(
        legacy_challenge_instructions(evidence_bound=True) + _grouped_challenge_note(),
        encoding="utf-8",
    )
    (task / "output_example.json").unlink()
    _deliver(controller, request)
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert _gpt(controller).status.value == "SUCCEEDED"


def test_challenge_contract_requires_a_matched_instruction_schema_pair(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request = _requests(controller, [names["word_r1"]])["gpt"]
    task = request.parent
    note = _grouped_challenge_note()
    current_instructions = challenge_instructions(evidence_bound=True) + note
    legacy_instructions = legacy_challenge_instructions(evidence_bound=True) + note
    current_schema = challenge_schema_text(evidence_bound=True)
    legacy_schema = legacy_challenge_schema_text(evidence_bound=True)
    assert (task / "instructions_gpt.txt").read_text(encoding="utf-8") == (
        current_instructions
    )
    assert (task / "output_schema.json").read_text(encoding="utf-8") == current_schema

    def republish() -> Path:
        return controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")

    assert republish() == request
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"

    (task / "instructions_gpt.txt").write_text(legacy_instructions, encoding="utf-8")
    (task / "output_schema.json").write_text(legacy_schema, encoding="utf-8")
    (task / "output_example.json").unlink()
    assert republish() == request
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"

    (task / "instructions_gpt.txt").write_text(current_instructions, encoding="utf-8")
    before = _snapshot(tmp_path)
    with pytest.raises(ReviewIntegrityError, match="mixes contract versions"):
        republish()
    assert _snapshot(tmp_path) == before

    (task / "instructions_gpt.txt").write_text(legacy_instructions, encoding="utf-8")
    (task / "output_schema.json").write_text(current_schema, encoding="utf-8")
    before = _snapshot(tmp_path)
    with pytest.raises(ReviewIntegrityError, match="mixes contract versions"):
        republish()
    assert _snapshot(tmp_path) == before
    assert not (task / "output_example.json").exists()


def test_preflight_reports_invalid_label_and_missing_metadata_without_writes(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request = _requests(controller, [names["word_r1"]])["gpt"]
    output = (
        _root(controller)
        / json.loads(request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        '{"provider":"grok","provider_interface":"GROK"}', encoding="utf-8"
    )
    before = _snapshot(tmp_path)
    report = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert report["output_valid"] is False
    assert report["eligible_for_acceptance"] is True
    assert report["failures"][0]["reason"] == "invalid_label"
    assert _snapshot(tmp_path) == before

    example = json.loads(
        (request.parent / "output_example.json").read_text(encoding="utf-8")
    )
    for field in (
        "job_id",
        "batch_id",
        "document_id",
        "attempt",
        "source_bundle_sha256",
    ):
        example.pop(field)
    output.write_text(json.dumps(example), encoding="utf-8")
    before = _snapshot(tmp_path)
    report = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert report["output_valid"] is False
    assert report["eligible_for_acceptance"] is True
    message = report["failures"][0]["message"]
    assert report["failures"][0]["reason"] == "schema"
    for field in (
        "job_id",
        "batch_id",
        "document_id",
        "attempt",
        "source_bundle_sha256",
    ):
        assert field in message
    assert _snapshot(tmp_path) == before
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"


def test_preflight_success_is_read_only_for_first_pass_and_challenge(
    tmp_path: Path,
) -> None:
    first = _controller(tmp_path / "first")
    first.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request = first.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    example = (request.parent / "output_example.json").read_bytes()
    FirstPassOutput.model_validate_json(example)
    from transcription_review_agent.review.paths import document_root

    target = (
        document_root(first.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / json.loads(request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_without_template(example))
    before = _snapshot(tmp_path / "first")
    report = first.validate_output(BATCH_ID, DOCUMENT_ID, "gpt")
    assert report["output_valid"] is True
    assert report["eligible_for_acceptance"] is True
    assert report["failures"] == []
    assert _snapshot(tmp_path / "first") == before
    target.write_text('{"provider":"gpt"}', encoding="utf-8")
    before = _snapshot(tmp_path / "first")
    report = first.validate_output(BATCH_ID, DOCUMENT_ID, "gpt")
    assert report["output_valid"] is False
    assert report["eligible_for_acceptance"] is True
    assert report["failures"][0]["reason"] == "invalid_label"
    assert _snapshot(tmp_path / "first") == before

    controller, _comparison, names = _setup(tmp_path / "challenge")
    challenge = _requests(controller, [names["word_r1"]])["gpt"]
    _deliver(controller, challenge)
    before = _snapshot(tmp_path / "challenge")
    report = controller.validate_output(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert report["output_valid"] is True
    assert report["eligible_for_acceptance"] is True
    assert _snapshot(tmp_path / "challenge") == before


def test_recovery_preserves_selection_rejection_and_the_other_provider(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    selected = [names["word_r1"], names["note_r1"]]
    requests = _requests(controller, selected)
    original = _invalidate(controller, requests["gpt"])
    rejected_report = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert rejected_report["output_valid"] is False
    assert rejected_report["eligible_for_acceptance"] is False
    assert rejected_report["eligibility"] == "INVALID_OUTPUT"
    grok_before = _grok(controller).model_dump(mode="json")
    grok_files = _snapshot(requests["grok"].parent)
    comparison = (_root(controller) / "comparison" / "comparison.json").read_bytes()
    disputes = (
        _root(controller) / "challenge_round_01" / "input" / "disputes.json"
    ).read_bytes()
    batches = _snapshot(tmp_path / "batches")
    frozen = _document(controller).challenge_rounds[0]
    selected_ids = list(frozen.selected_dispute_ids or [])
    selection_hash = frozen.selection_comparison_sha256
    contract = frozen.challenge_contract
    rejected = _events(controller, "result_rejected")

    recovered = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    request = WorkRequest.model_validate_json(recovered.read_bytes())
    assert request.attempt == 2
    assert request.task_id == f"{BATCH_ID}:{DOCUMENT_ID}:challenge:r1:gpt:a2"
    assert request.job_id == f"job:{request.task_id}"
    assert request.output_relative_path.endswith(
        "challenge_round_01/attempts/a02/output/challenge_gpt.json"
    )
    assert (
        _root(controller) / "challenge_round_01" / "output" / "challenge_gpt.json"
    ).read_bytes() == original
    preserved = WorkRequest.model_validate_json(
        (
            _root(controller)
            / "challenge_round_01"
            / "attempts"
            / "a01"
            / "task"
            / "work_request.json"
        ).read_bytes()
    )
    assert preserved.attempt == 1
    assert preserved.task_id != request.task_id
    current = _document(controller).challenge_rounds[0]
    assert current.selected_dispute_ids == selected_ids
    assert current.selection_comparison_sha256 == selection_hash
    assert current.challenge_contract == contract == "EVIDENCE_BOUND_V1"
    assert current.round_index == 1
    assert len(_document(controller).challenge_rounds) == 1
    assert (_root(controller) / "comparison" / "comparison.json").read_bytes() == (
        comparison
    )
    assert (
        _root(controller) / "challenge_round_01" / "input" / "disputes.json"
    ).read_bytes() == disputes
    assert _snapshot(tmp_path / "batches") == batches
    assert _grok(controller).model_dump(mode="json") == grok_before
    assert _snapshot(requests["grok"].parent) == grok_files
    assert _events(controller, "result_rejected") == rejected
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"
    assert _gpt(controller).invalid_output_recoveries == 1

    again = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert again == recovered
    published = [
        item
        for item in _events(controller, "work_published")
        if item["provider"] == "gpt" and item["stage"] == "challenge"
    ]
    assert [item["attempt"] for item in published] == [1, 2]

    store = controller._store(BATCH_ID)
    checkpoint = store.load()
    store.checkpoint_path.unlink()
    assert store.load() == checkpoint
    assert any(
        item["reason"] == "schema" for item in _events(controller, "result_rejected")
    )

    attempt1 = ChallengeOutput(
        provider=preserved.provider,
        provider_interface=preserved.provider_interface,
        task_id=preserved.task_id,
        job_id=preserved.job_id,
        batch_id=preserved.batch_id,
        document_id=preserved.document_id,
        attempt=preserved.attempt,
        source_bundle_sha256=preserved.source_bundle_sha256,
        round_index=preserved.round_index,
        decisions=ChallengeOutput.model_validate_json(
            (recovered.parent / "output_example.json").read_bytes()
        ).decisions,
    )
    stale_path = _root(controller) / request.output_relative_path
    stale_path.write_bytes(attempt1.model_dump_json().encode("utf-8"))
    before = _snapshot(controller.reviews_dir)
    report = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert report["output_valid"] is False
    assert report["eligible_for_acceptance"] is True
    assert report["failures"][0]["reason"] in {"stale", "mismatched"}
    assert _snapshot(controller.reviews_dir) == before
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert _gpt(controller).status.value != "SUCCEEDED"
    assert (
        _root(controller) / "challenge_round_01" / "output" / "challenge_gpt.json"
    ).read_bytes() == original


def test_recovered_attempt_accepts_only_its_own_evidence_bound_output(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    selected = [names["word_r1"]]
    requests = _requests(controller, selected)
    _invalidate(controller, requests["gpt"])
    recovered = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    _deliver(controller, recovered)
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert _gpt(controller).status.value == "SUCCEEDED"
    accepted = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    assert accepted["output_valid"] is True
    assert accepted["eligible_for_acceptance"] is False
    assert accepted["eligibility"] == "SUCCEEDED"
    assert _gpt(controller).attempt == 2
    assert _grok(controller).status.value == "AWAITING_MANUAL_DISPATCH"
    _deliver(controller, requests["grok"])
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    assert _document(controller).challenge_rounds[0].reconciled is True
    with pytest.raises(ReviewPolicyError, match="already succeeded"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
        )


def test_recovery_is_refused_after_closure_or_a_second_rejection(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    requests = _requests(controller, [names["word_r1"]])
    _invalidate(controller, requests["gpt"])
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "grok", kind="quota", stage="challenge"
    )
    controller.close_challenge(
        BATCH_ID,
        DOCUMENT_ID,
        mode="PROVIDER_TERMINAL_FAILURE",
        reason="The challenge round ended without a usable provider result.",
    )
    with pytest.raises(ReviewPolicyError, match="closed"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
        )
    controller.finalize(BATCH_ID, DOCUMENT_ID)
    with pytest.raises(ReviewPolicyError, match="finalized"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
        )

    other, _comparison, names = _setup(tmp_path / "limit")
    request = _requests(other, [names["word_r1"]])["gpt"]
    _invalidate(other, request)
    recovered = other.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    bad = (
        _root(other)
        / WorkRequest.model_validate_json(recovered.read_bytes()).output_relative_path
    )
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ReviewOutputRejected):
        other.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert _gpt(other).invalid_output_recoveries == MAX_INVALID_OUTPUT_RECOVERIES
    with pytest.raises(ReviewPolicyError, match="limit"):
        other.recover_invalid_output(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    with pytest.raises(ReviewPolicyError, match="no repeated challenge"):
        other.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")


def test_recovery_replay_is_idempotent_across_a_crash(
    tmp_path: Path, monkeypatch
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request = _requests(controller, [names["word_r1"]])["gpt"]
    original = _invalidate(controller, request)
    output = _root(controller) / "challenge_round_01" / "output" / "challenge_gpt.json"

    def crash(self, entry):  # type: ignore[no-untyped-def]
        del self, entry
        raise RuntimeError("crash before commit")

    monkeypatch.setattr(ReviewStore, "commit", crash)
    with pytest.raises(RuntimeError, match="crash before commit"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
        )
    assert _gpt(controller).status.value == "INVALID_OUTPUT"
    assert output.read_bytes() == original
    monkeypatch.undo()

    first = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    task_bytes = first.read_bytes()
    second = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    assert second.read_bytes() == task_bytes
    assert [
        item["attempt"]
        for item in _events(controller, "work_published")
        if item["provider"] == "gpt" and item["stage"] == "challenge"
    ] == [1, 2]
    assert output.read_bytes() == original
    checkpoint = controller._store(BATCH_ID).load()
    assert checkpoint is not None
    raw = json.loads(controller._store(BATCH_ID).checkpoint_path.read_text())

    def strip(value: object) -> None:
        if isinstance(value, dict):
            value.pop("invalid_output_recoveries", None)
            for item in value.values():
                strip(item)
        elif isinstance(value, list):
            for item in value:
                strip(item)

    strip(raw)
    controller._store(BATCH_ID).checkpoint_path.write_text(
        json.dumps(raw), encoding="utf-8"
    )
    # The stripped checkpoint no longer matches, because recovery stored a count.
    # A checkpoint from before any recovery omits the field and still replays.
    with pytest.raises(ReviewIntegrityError, match="does not match"):
        controller._store(BATCH_ID).load()


def test_checkpoint_without_recovery_count_replays(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    _requests(controller, [names["word_r1"]])
    path = controller._store(BATCH_ID).checkpoint_path
    raw = json.loads(path.read_text(encoding="utf-8"))

    def strip(value: object) -> None:
        if isinstance(value, dict):
            value.pop("invalid_output_recoveries", None)
            for item in value.values():
                strip(item)
        elif isinstance(value, list):
            for item in value:
                strip(item)

    strip(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = controller._store(BATCH_ID).load()
    assert loaded is not None
    assert _gpt(controller).invalid_output_recoveries == 0


def _without_template(payload: bytes) -> bytes:
    text = payload.decode("utf-8")
    text = text.replace(
        "TEMPLATE ONLY. Replace this page with your transcription.",
        "The visible line reads Alpha.",
    )
    text = text.replace(
        "TEMPLATE ONLY. Replace this sentence with the visual evidence you examined.",
        "The crop is too faint to choose a candidate.",
    )
    text = text.replace("TEMPLATE ALTERNATIVE", "Beta reading")
    text = text.replace("TEMPLATE ONLY", "a faint mark")
    text = text.replace("TEMPLATE", "Alpha")
    return text.encode("utf-8")


def _published_attempts(controller) -> list[object]:
    return [
        item["attempt"]
        for item in _events(controller, "work_published")
        if item["provider"] == "gpt" and item["stage"] == "challenge"
    ]


def _assert_recovered_bundle(controller, archived: dict[str, bytes]) -> None:
    task = _root(controller) / "challenge_round_01" / "tasks" / "gpt"
    archive = _root(controller) / "challenge_round_01" / "attempts" / "a01" / "task"
    assert _snapshot(archive) == archived
    request = WorkRequest.model_validate_json((task / "work_request.json").read_bytes())
    assert request.attempt == 2
    assert request.task_id.endswith(":a2")
    assert (task / "output_schema.json").read_text(encoding="utf-8") == (
        challenge_schema_text(evidence_bound=True)
    )
    assert (task / "output_example.json").is_file()
    assert not (task.parent / "gpt.next").exists()
    assert not (task.parent / "gpt.previous").exists()
    assert _published_attempts(controller) == [1, 2]
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"
    assert _gpt(controller).attempt == 2


def test_recovery_resumes_after_archive_and_mid_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request = _requests(controller, [names["word_r1"]])["gpt"]
    original_output = _invalidate(controller, request)
    task = request.parent
    archived_source = _snapshot(task)
    output = _root(controller) / "challenge_round_01" / "output" / "challenge_gpt.json"
    partial = (
        _root(controller) / "challenge_round_01" / "attempts" / "a01" / "task.partial"
    )
    partial.mkdir(parents=True)
    (partial / "work_request.json").write_text("{}", encoding="utf-8")

    real_write = review_controller._write_if_same

    def crash_before_staging(path: Path, payload: bytes) -> None:
        if any(part.endswith(".next") for part in path.parts):
            raise RuntimeError("crash after archive, before replacement")
        real_write(path, payload)

    monkeypatch.setattr(review_controller, "_write_if_same", crash_before_staging)
    with pytest.raises(RuntimeError, match="before replacement"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
        )
    assert _published_attempts(controller) == [1]
    assert _gpt(controller).status.value == "INVALID_OUTPUT"
    assert output.read_bytes() == original_output
    assert _snapshot(task) == archived_source
    archive = _root(controller) / "challenge_round_01" / "attempts" / "a01" / "task"
    assert _snapshot(archive) == archived_source
    assert not partial.exists()
    monkeypatch.undo()

    original_rename = review_controller.os.rename
    seen = {"install_renames": 0}

    def crash_on_second_install_rename(src: Path, dst: Path) -> None:
        if src.name == "gpt" and dst.name == "gpt.previous":
            seen["install_renames"] += 1
            if seen["install_renames"] == 1:
                raise RuntimeError("crash after staging, before swap")
        original_rename(src, dst)

    monkeypatch.setattr(review_controller.os, "rename", crash_on_second_install_rename)
    with pytest.raises(RuntimeError, match="before swap"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
        )
    assert _published_attempts(controller) == [1]
    assert _snapshot(archive) == archived_source
    assert _snapshot(task) == archived_source
    assert output.read_bytes() == original_output
    monkeypatch.undo()

    def crash_after_live_dir_moves(src: Path, dst: Path) -> None:
        original_rename(src, dst)
        if src.name == "gpt" and dst.name == "gpt.previous":
            raise RuntimeError("crash with the live task directory moved aside")

    monkeypatch.setattr(review_controller.os, "rename", crash_after_live_dir_moves)
    with pytest.raises(RuntimeError, match="moved aside"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
        )
    assert not task.exists()
    assert (task.parent / "gpt.previous").is_dir()
    assert (task.parent / "gpt.next").is_dir()
    assert _snapshot(archive) == archived_source
    assert _published_attempts(controller) == [1]
    assert output.read_bytes() == original_output
    monkeypatch.undo()

    controller.recover_invalid_output(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    _assert_recovered_bundle(controller, archived_source)
    assert output.read_bytes() == original_output
    again = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    assert WorkRequest.model_validate_json(again.read_bytes()).attempt == 2
    assert _published_attempts(controller) == [1, 2]


def test_corrupt_live_task_is_replaced_from_previous(tmp_path: Path) -> None:
    controller, _comparison, names = _setup(tmp_path)
    selected = [names["word_r1"], names["note_r1"]]
    requests = _requests(controller, selected)
    original_output = _invalidate(controller, requests["gpt"])
    task = requests["gpt"].parent
    valid = _snapshot(task)
    previous = task.with_name("gpt.previous")
    shutil.copytree(task, previous)
    (task / "output_schema.json").write_text(
        '{"schema_version": "corrupt"}\n', encoding="utf-8"
    )
    frozen = list(_document(controller).challenge_rounds[0].selected_dispute_ids or [])
    grok_files = _snapshot(requests["grok"].parent)
    output = _root(controller) / "challenge_round_01" / "output" / "challenge_gpt.json"
    recovered = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    archive = _root(controller) / "challenge_round_01" / "attempts" / "a01" / "task"
    assert _snapshot(archive) == valid
    assert b"corrupt" not in (archive / "output_schema.json").read_bytes()
    assert not previous.exists()
    assert not task.with_name("gpt.next").exists()
    assert not task.with_name("gpt.corrupt").exists()
    request = WorkRequest.model_validate_json(recovered.read_bytes())
    assert request.attempt == 2
    assert (task / "output_schema.json").read_text(encoding="utf-8") == (
        challenge_schema_text(evidence_bound=True)
    )
    assert output.read_bytes() == original_output
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen
    assert _snapshot(requests["grok"].parent) == grok_files
    again = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert again == recovered
    assert _published_attempts(controller) == [1, 2]


def test_invalid_utf8_live_task_restores_previous_and_publishes_attempt_2(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    selected = [names["word_r1"], names["note_r1"]]
    requests = _requests(controller, selected)
    original_output = _invalidate(controller, requests["gpt"])
    task = requests["gpt"].parent
    valid = _snapshot(task)
    previous = task.with_name("gpt.previous")
    shutil.copytree(task, previous)
    instructions = task / "instructions_gpt.txt"
    instructions.write_bytes(instructions.read_bytes() + b"\xff")
    frozen = list(_document(controller).challenge_rounds[0].selected_dispute_ids or [])
    grok_files = _snapshot(requests["grok"].parent)
    output = _root(controller) / "challenge_round_01" / "output" / "challenge_gpt.json"
    ledger_before = _ledger(controller)
    recovered = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    archive = _root(controller) / "challenge_round_01" / "attempts" / "a01" / "task"
    assert _snapshot(archive) == valid
    assert not previous.exists()
    assert not task.with_name("gpt.corrupt").exists()
    request = WorkRequest.model_validate_json(recovered.read_bytes())
    assert request.attempt == 2
    assert request.provider == "gpt"
    assert b"\xff" not in instructions.read_bytes()
    assert output.read_bytes() == original_output
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen
    assert _snapshot(requests["grok"].parent) == grok_files
    added = _ledger(controller)[len(ledger_before) :]
    assert _ledger(controller)[: len(ledger_before)] == ledger_before
    assert len(added) == 1
    assert added[0]["event_type"] == "work_published"
    assert added[0]["provider"] == "gpt"
    assert added[0]["stage"] == "challenge"
    assert added[0]["attempt"] == 2
    assert added[0]["task_id"] == request.task_id


@pytest.mark.parametrize(
    "filename",
    ["instructions_gpt.txt", "output_schema.json", "work_request.json"],
)
def test_malformed_task_utf8_stays_an_integrity_error(
    tmp_path: Path, filename: str
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request = _requests(controller, [names["word_r1"]])["gpt"]
    _deliver(controller, request)
    task = request.parent
    target = task / filename
    target.write_bytes(target.read_bytes() + b"\xff")
    corrupt = target.read_bytes()
    before = _snapshot(tmp_path)
    ledger_before = _ledger(controller)
    report = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert report["failures"] == [
        {
            "reason": "integrity",
            "message": "published provider task text is not valid UTF-8",
        }
    ]
    assert report["eligible_for_acceptance"] is True
    assert report["eligibility"] == "awaiting_manual_dispatch"
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"
    with pytest.raises(
        ReviewIntegrityError, match="published provider task text is not valid UTF-8"
    ):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"
    assert target.read_bytes() == corrupt
    assert _snapshot(tmp_path) == before
    assert _ledger(controller) == ledger_before
    assert _events(controller, "result_rejected") == []


def test_invalid_utf8_previous_is_left_in_place(tmp_path: Path) -> None:
    controller, _comparison, names = _setup(tmp_path)
    requests = _requests(controller, [names["word_r1"], names["note_r1"]])
    original_output = _invalidate(controller, requests["gpt"])
    task = requests["gpt"].parent
    live = _snapshot(task)
    previous = task.with_name("gpt.previous")
    shutil.copytree(task, previous)
    previous_instructions = previous / "instructions_gpt.txt"
    previous_instructions.write_bytes(previous_instructions.read_bytes() + b"\xff")
    previous_files = _snapshot(previous)
    frozen = list(_document(controller).challenge_rounds[0].selected_dispute_ids or [])
    grok_files = _snapshot(requests["grok"].parent)
    output = _root(controller) / "challenge_round_01" / "output" / "challenge_gpt.json"
    ledger_before = _ledger(controller)
    with pytest.raises(ReviewIntegrityError, match="invalid UTF-8"):
        controller.recover_invalid_output(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
        )
    assert _snapshot(task) == live
    assert _snapshot(previous) == previous_files
    assert previous_instructions.read_bytes().endswith(b"\xff")
    assert not task.with_name("gpt.next").exists()
    assert not task.with_name("gpt.corrupt").exists()
    assert output.read_bytes() == original_output
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen
    assert _snapshot(requests["grok"].parent) == grok_files
    assert _ledger(controller) == ledger_before
    assert _gpt(controller).status.value == "INVALID_OUTPUT"


def test_template_is_rejected_and_unresolved_text_is_not(
    tmp_path: Path,
) -> None:
    first = _controller(tmp_path / "first")
    first.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request = first.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    from transcription_review_agent.review.paths import document_root

    target = (
        document_root(first.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / json.loads(request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    example = (request.parent / "output_example.json").read_bytes()
    target.write_bytes(example)
    before = _snapshot(tmp_path / "first")
    report = first.validate_output(BATCH_ID, DOCUMENT_ID, "gpt")
    assert report["output_valid"] is False
    assert report["eligible_for_acceptance"] is True
    assert report["failures"][0]["reason"] == "schema"
    assert "task template" in report["failures"][0]["message"]
    assert _snapshot(tmp_path / "first") == before
    with pytest.raises(ReviewOutputRejected, match="task template"):
        first.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    assert first.status(BATCH_ID, DOCUMENT_ID).providers["gpt"] == "INVALID_OUTPUT"

    other = _controller(tmp_path / "readable")
    other.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    other_request = other.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    other_target = (
        document_root(other.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / json.loads(other_request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    other_target.parent.mkdir(parents=True, exist_ok=True)
    other_target.write_bytes(
        _without_template((other_request.parent / "output_example.json").read_bytes())
    )
    accepted = other.validate_output(BATCH_ID, DOCUMENT_ID, "gpt")
    assert accepted["output_valid"] is True
    assert b"[illegible]" in other_target.read_bytes()

    controller, _comparison, names = _setup(tmp_path / "challenge")
    challenge = _requests(controller, [names["word_r1"]])["gpt"]
    challenge_output = (
        _root(controller)
        / json.loads(challenge.read_text(encoding="utf-8"))["output_relative_path"]
    )
    challenge_output.parent.mkdir(parents=True, exist_ok=True)
    template = (challenge.parent / "output_example.json").read_bytes()
    challenge_output.write_bytes(template)
    before = _snapshot(tmp_path / "challenge")
    rejected = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    assert rejected["output_valid"] is False
    assert "task template" in rejected["failures"][0]["message"]
    assert _snapshot(tmp_path / "challenge") == before
    challenge_output.write_bytes(_without_template(template))
    unresolved = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    assert unresolved["output_valid"] is True
    parsed = ChallengeOutput.model_validate_json(challenge_output.read_bytes())
    assert parsed.decisions[0].decision == "UNRESOLVED"
    assert parsed.decisions[0].evidence_assessment == "INSUFFICIENT"


def test_historical_task_recovers_without_changing_the_selection(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    selected = [names["word_r1"], names["note_r1"]]
    requests = _requests(controller, selected)
    task = requests["gpt"].parent
    (task / "output_schema.json").write_text(
        legacy_challenge_schema_text(evidence_bound=True), encoding="utf-8"
    )
    (task / "instructions_gpt.txt").write_text(
        legacy_challenge_instructions(evidence_bound=True) + _grouped_challenge_note(),
        encoding="utf-8",
    )
    (task / "output_example.json").unlink()
    historical = _snapshot(task)
    assert "output_example.json" not in historical
    frozen = list(_document(controller).challenge_rounds[0].selected_dispute_ids or [])
    _invalidate(controller, requests["gpt"])
    recovered = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    archive = _root(controller) / "challenge_round_01" / "attempts" / "a01" / "task"
    assert _snapshot(archive) == historical
    request = WorkRequest.model_validate_json(recovered.read_bytes())
    assert request.attempt == 2
    assert (
        request.task_id
        != WorkRequest.model_validate_json(
            (archive / "work_request.json").read_bytes()
        ).task_id
    )
    assert (task / "output_schema.json").read_text(encoding="utf-8") == (
        challenge_schema_text(evidence_bound=True)
    )
    assert (task / "output_example.json").is_file()
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen
    assert (
        _document(controller).challenge_rounds[0].challenge_contract
        == "EVIDENCE_BOUND_V1"
    )
    _deliver(controller, recovered)
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert _gpt(controller).status.value == "SUCCEEDED"
    assert _gpt(controller).attempt == 2
    assert _gpt(controller).task_id == request.task_id
    assert _gpt(controller).job_id == request.job_id
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen


def test_validate_cli_does_not_record_a_rejection(tmp_path: Path, capsys) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request = _requests(controller, [names["word_r1"]])["gpt"]
    output = (
        _root(controller)
        / json.loads(request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}", encoding="utf-8")
    before = _snapshot(tmp_path)
    code = main(
        [
            "validate",
            "--batches-dir",
            str(tmp_path / "batches"),
            "--reviews-dir",
            str(tmp_path / "reviews"),
            "--batch-id",
            BATCH_ID,
            "--document-id",
            str(DOCUMENT_ID),
            "--provider",
            "gpt",
            "--stage",
            "challenge",
            "--round-index",
            "1",
        ]
    )
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["output_valid"] is False
    assert report["eligible_for_acceptance"] is True
    assert _snapshot(tmp_path) == before


def _assert_same_attempt(task: Path, attempt: int) -> str:
    request = json.loads((task / "work_request.json").read_text(encoding="utf-8"))
    example = json.loads((task / "output_example.json").read_text(encoding="utf-8"))
    assert request["attempt"] == attempt
    assert example["attempt"] == attempt
    assert example["task_id"] == request["task_id"]
    assert example["job_id"] == request["job_id"]
    return request["task_id"]


def test_transient_retry_refreshes_first_pass_and_challenge_identity(
    tmp_path: Path,
) -> None:
    from transcription_review_agent.review.paths import document_root

    first = _controller(tmp_path / "first")
    first.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    request_path = first.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    task = request_path.parent
    _assert_same_attempt(task, 1)
    output = (
        document_root(first.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / json.loads(request_path.read_text(encoding="utf-8"))["output_relative_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((task / "output_example.json").read_bytes())
    first.record_failure(BATCH_ID, DOCUMENT_ID, "gpt", kind="transient")
    retried = first.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    task_id = _assert_same_attempt(task, 2)
    assert json.loads(retried.read_text(encoding="utf-8"))["task_id"] == task_id
    stale = first.validate_output(BATCH_ID, DOCUMENT_ID, "gpt")
    assert stale["failures"][0]["reason"] == "stale"
    assert stale["eligible_for_acceptance"] is True
    assert first.status(BATCH_ID, DOCUMENT_ID).providers["gpt"] == (
        "AWAITING_MANUAL_DISPATCH"
    )
    before = _snapshot(tmp_path / "first")
    assert first.publish(BATCH_ID, DOCUMENT_ID, "gpt") == retried
    assert _snapshot(tmp_path / "first") == before
    with pytest.raises(ReviewOutputRejected, match="current source or attempt"):
        first.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    assert first.status(BATCH_ID, DOCUMENT_ID).providers["gpt"] == "INVALID_OUTPUT"

    controller, _comparison, names = _setup(tmp_path / "challenge")
    selected = [names["word_r1"], names["note_r1"]]
    requests = _requests(controller, selected)
    task = requests["gpt"].parent
    _assert_same_attempt(task, 1)
    _deliver(controller, requests["gpt"])
    evidence = _snapshot(_root(controller) / "challenge_round_01" / "input")
    grok_files = _snapshot(requests["grok"].parent)
    frozen = list(_document(controller).challenge_rounds[0].selected_dispute_ids or [])
    assert frozen == selected
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="challenge"
    )
    recovered = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    task_id = _assert_same_attempt(task, 2)
    assert json.loads(recovered.read_text(encoding="utf-8"))["task_id"] == task_id
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen
    assert _snapshot(_root(controller) / "challenge_round_01" / "input") == evidence
    assert _snapshot(requests["grok"].parent) == grok_files
    stale = controller.validate_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge", round_index=1
    )
    assert stale["failures"][0]["reason"] == "stale"
    assert stale["eligible_for_acceptance"] is True
    assert _gpt(controller).status.value == "AWAITING_MANUAL_DISPATCH"
    before = _snapshot(tmp_path / "challenge")
    assert (
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge") == recovered
    )
    assert _snapshot(tmp_path / "challenge") == before
    assert _published_attempts(controller) == [1, 2]
    with pytest.raises(ReviewOutputRejected, match="current source or attempt"):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert _gpt(controller).status.value == "INVALID_OUTPUT"
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen


def test_transient_retry_resumes_after_an_interrupted_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    selected = [names["word_r1"], names["note_r1"]]
    requests = _requests(controller, selected)
    task = requests["gpt"].parent
    evidence = _snapshot(_root(controller) / "challenge_round_01" / "input")
    frozen = list(_document(controller).challenge_rounds[0].selected_dispute_ids or [])
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="challenge"
    )
    original_rename = review_controller.os.rename

    def crash_after_live_dir_moves(src: Path, dst: Path) -> None:
        original_rename(src, dst)
        if src.name == "gpt" and dst.name == "gpt.previous":
            raise RuntimeError("interrupted identity swap")

    monkeypatch.setattr(review_controller.os, "rename", crash_after_live_dir_moves)
    with pytest.raises(RuntimeError, match="interrupted identity swap"):
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    previous = task.with_name("gpt.previous")
    staging = task.with_name("gpt.next")
    assert not task.exists()
    assert _assert_same_attempt(previous, 1)
    assert _assert_same_attempt(staging, 2)
    assert _published_attempts(controller) == [1]
    assert _snapshot(_root(controller) / "challenge_round_01" / "input") == evidence
    monkeypatch.undo()

    recovered = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    task_id = _assert_same_attempt(task, 2)
    assert json.loads(recovered.read_text(encoding="utf-8"))["task_id"] == task_id
    assert not previous.exists()
    assert not staging.exists()
    assert _published_attempts(controller) == [1, 2]
    assert (
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge") == recovered
    )
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == frozen
    assert frozen == selected
    assert _snapshot(_root(controller) / "challenge_round_01" / "input") == evidence


def _research_runtime(controller, provider: str = "gpt"):
    return next(
        item for item in _document(controller).research if item.provider == provider
    )


def _research_attempts(controller) -> list[object]:
    return [
        item["attempt"]
        for item in _events(controller, "work_published")
        if item["provider"] == "gpt" and item["stage"] == "research"
    ]


def _open_research(tmp_path: Path):
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
    return controller


def _research_fixed_files(task: Path) -> dict[str, bytes]:
    return {
        relative: (task / relative).read_bytes()
        for relative in (
            "instructions_gpt.txt",
            "output_schema.json",
            "README.txt",
            "input/research_request.json",
        )
    }


def test_research_transient_retry_refreshes_only_attempt_identity(
    tmp_path: Path,
) -> None:
    controller = _open_research(tmp_path)
    with pytest.raises(
        ReviewPolicyError,
        match="invalid-output recovery is only available for a challenge round",
    ):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="research",
            recover_invalid_output=True,
        )
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    task = request_path.parent
    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    _assert_same_attempt(task, 1)
    fixed = _research_fixed_files(task)
    published_input = _snapshot(doc_root / "research" / "input")
    output = (
        doc_root
        / json.loads(request_path.read_text(encoding="utf-8"))["output_relative_path"]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((task / "output_example.json").read_bytes())
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="research"
    )
    retried = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    task_id = _assert_same_attempt(task, 2)
    assert json.loads(retried.read_text(encoding="utf-8"))["task_id"] == task_id
    assert _research_fixed_files(task) == fixed
    assert _snapshot(doc_root / "research" / "input") == published_input
    assert (task / "instructions_gpt.txt").read_text(encoding="utf-8") == _instructions(
        "research"
    )
    assert (task / "output_schema.json").read_text(encoding="utf-8") == _schema_text(
        "research"
    )
    stale = controller.validate_output(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    assert stale["failures"][0]["reason"] == "stale"
    assert stale["eligible_for_acceptance"] is True
    assert _research_runtime(controller).status.value == "AWAITING_MANUAL_DISPATCH"
    assert _research_runtime(controller).attempt == 2
    before = _snapshot(tmp_path)
    assert controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research") == retried
    assert _snapshot(tmp_path) == before
    assert _research_attempts(controller) == [1, 2]
    with pytest.raises(ReviewOutputRejected, match="current source or attempt"):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    assert _research_runtime(controller).status.value == "INVALID_OUTPUT"
    assert _research_attempts(controller) == [1, 2]
    with pytest.raises(
        ReviewPolicyError,
        match="invalid-output recovery is only available for a challenge round",
    ):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="research",
            recover_invalid_output=True,
        )
    assert _research_runtime(controller).status.value == "INVALID_OUTPUT"
    assert _research_attempts(controller) == [1, 2]


def test_research_transient_retry_resumes_after_an_interrupted_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = _open_research(tmp_path)
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    task = request_path.parent
    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    fixed = _research_fixed_files(task)
    published_input = _snapshot(doc_root / "research" / "input")
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="research"
    )
    original_rename = review_controller.os.rename

    def crash_after_live_dir_moves(src: Path, dst: Path) -> None:
        original_rename(src, dst)
        if src.name == "gpt" and dst.name == "gpt.previous":
            raise RuntimeError("interrupted identity swap")

    monkeypatch.setattr(review_controller.os, "rename", crash_after_live_dir_moves)
    with pytest.raises(RuntimeError, match="interrupted identity swap"):
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    previous = task.with_name("gpt.previous")
    staging = task.with_name("gpt.next")
    assert not task.exists()
    assert _assert_same_attempt(previous, 1)
    assert _assert_same_attempt(staging, 2)
    assert _research_fixed_files(previous) == fixed
    assert _research_fixed_files(staging) == fixed
    assert _research_attempts(controller) == [1]
    assert _snapshot(doc_root / "research" / "input") == published_input
    monkeypatch.undo()

    recovered = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    task_id = _assert_same_attempt(task, 2)
    assert json.loads(recovered.read_text(encoding="utf-8"))["task_id"] == task_id
    assert not previous.exists()
    assert not staging.exists()
    assert _research_fixed_files(task) == fixed
    assert _research_attempts(controller) == [1, 2]
    assert _snapshot(doc_root / "research" / "input") == published_input
    before = _snapshot(tmp_path)
    assert (
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research") == recovered
    )
    assert _snapshot(tmp_path) == before
    assert _research_attempts(controller) == [1, 2]


def _malformed_research_input(kind: str) -> bytes:
    if kind == "invalid-utf8":
        return b"\xff"
    label: object = ["visual_a"] if kind == "unhashable-label" else 1
    return (
        json.dumps(
            {
                "archive_transcription_included": False,
                "diplomatic_spelling_is_fixed": True,
                "schema_version": "review-research-input-v1",
                "target_transcription_cannot_corroborate_itself": True,
                "visual_readings": [{"label": label, "pages": []}],
            }
        )
        + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize(
    "kind", ["invalid-utf8", "unhashable-label", "non-string-label"]
)
def test_malformed_research_input_recovery_leaves_previous_in_place(
    tmp_path: Path, kind: str
) -> None:
    controller = _open_research(tmp_path)
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    task = request_path.parent
    previous = task.with_name("gpt.previous")
    shutil.copytree(task, previous)
    previous_files = _snapshot(previous)
    payload = _malformed_research_input(kind)
    assert _research_input_matches_contract(payload) is False
    for path in (
        task / "input" / "research_request.json",
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "research"
        / "input"
        / "research_request.json",
    ):
        path.write_bytes(payload)
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="research"
    )
    before = _snapshot(tmp_path)
    ledger_before = _ledger(controller)
    with pytest.raises(
        ReviewIntegrityError, match="previous directory was left in place"
    ):
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="research")
    assert _snapshot(tmp_path) == before
    assert _ledger(controller) == ledger_before
    assert _snapshot(previous) == previous_files
    assert _assert_same_attempt(previous, 1)
    assert (task / "input" / "research_request.json").read_bytes() == payload
    assert not task.with_name("gpt.next").exists()
    assert not task.with_name("gpt.corrupt").exists()
    assert _research_attempts(controller) == [1]
    assert _research_runtime(controller).status.value == "RETRYABLE_FAILURE"
