"""A3 controller integration: immutable evidence-bound challenge rounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_review_challenge_groups import (
    _document,
    _input,
    _pages,
    _published,
    _root,
    _setup,
)
from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    NOW,
    _close_challenge,
    _snapshot,
    _write_output,
)
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewOutputRejected,
)
from transcription_review_agent.review.models import (
    ChallengeDecision,
    ChallengeOutput,
    EvidenceReference,
    WorkRequest,
)


def _reference_for(
    controller: ReviewController, dispute_id: str, *, index: int = 0
) -> EvidenceReference:
    published = _published(controller)
    group = next(
        item for item in published.groups or [] if dispute_id in item.dispute_ids
    )
    return group.evidence_refs[index]


def _deliver(
    controller: ReviewController,
    request_path: Path,
    *,
    choice: str = "CANDIDATE_X",
    assessment: str = "SUPPORTS_DECISION",
    third: str | None = None,
    reference_index: int = 0,
    reference_overrides: dict[str, EvidenceReference] | None = None,
    evidence: str = "Exact visual evidence statement.",
) -> Path:
    published = _published(controller)
    request = WorkRequest.model_validate_json(request_path.read_bytes())
    decisions = []
    for item in published.disputes:
        decisions.append(
            ChallengeDecision(
                dispute_id=item.dispute_id,
                decision=choice,  # type: ignore[arg-type]
                third_reading=third,
                visual_evidence=evidence,
                evidence_ref=(reference_overrides or {}).get(
                    item.dispute_id,
                    _reference_for(controller, item.dispute_id, index=reference_index),
                ),
                evidence_assessment=assessment,  # type: ignore[arg-type]
            )
        )
    output = ChallengeOutput(
        provider=request.provider,
        provider_interface=request.provider_interface,
        task_id=request.task_id,
        job_id=request.job_id,
        batch_id=request.batch_id,
        document_id=request.document_id,
        attempt=request.attempt,
        source_bundle_sha256=request.source_bundle_sha256,
        round_index=request.round_index,
        decisions=decisions,
    )
    return _write_output(controller, request_path, output.model_dump(mode="json"))


def _requests(controller: ReviewController, selected: list[str]) -> dict[str, Path]:
    return {
        "gpt": controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="challenge",
            selected_dispute_ids=selected,
        ),
        "grok": controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge"),
    }


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("CANDIDATE_X", "gpt"),
        ("CANDIDATE_Y", "grok"),
        ("THIRD_READING", "third"),
    ],
)
def test_new_round_supports_exact_x_y_or_third_reading(
    tmp_path: Path, choice: str, expected: str
) -> None:
    controller, comparison, names = _setup(tmp_path / choice.lower())
    dispute_id = names["word_r1"]
    dispute = next(
        item for item in comparison.disputes if item.dispute_id == dispute_id
    )
    requests = _requests(controller, [dispute_id])
    third = "  Exact third\nreading  " if choice == "THIRD_READING" else None
    for provider, request in requests.items():
        _deliver(controller, request, choice=choice, third=third)
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    decision = _document(controller).reconciled_decisions[0]
    selected = {
        "gpt": dispute.gpt_text,
        "grok": dispute.grok_text,
        "third": third,
    }[expected]
    assert decision.decision == choice
    assert decision.selected_reading == selected
    assert decision.result_class == "AUTOMATICALLY_SUPPORTED"
    assert decision.basis == "DUAL_MODEL_VISUAL_SUPPORT"
    assert decision.independently_verified is False
    assert decision.eligible_for_writer_memory is False
    assert decision.eligible_for_publication is False
    assert (
        decision.gpt_visual_evidence
        == decision.grok_visual_evidence
        == ("Exact visual evidence statement.")
    )


@pytest.mark.parametrize("assessment", ["INSUFFICIENT", "UNREADABLE"])
def test_same_label_with_weak_evidence_is_unresolved(
    tmp_path: Path, assessment: str
) -> None:
    controller, _comparison, names = _setup(tmp_path / assessment.lower())
    requests = _requests(controller, [names["word_r1"]])
    for provider, request in requests.items():
        _deliver(controller, request, assessment=assessment)
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    decision = _document(controller).reconciled_decisions[0]
    assert decision.agreed is True
    assert decision.decision == "UNRESOLVED"
    assert decision.result_class == "UNRESOLVED"
    assert decision.selected_reading is None
    assert decision.basis == (
        "INSUFFICIENT_EVIDENCE"
        if assessment == "INSUFFICIENT"
        else "UNREADABLE_EVIDENCE"
    )


def test_provider_disagreement_is_unresolved_and_keeps_both_records(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    requests = _requests(controller, [names["word_r1"]])
    for provider, choice in (("gpt", "CANDIDATE_X"), ("grok", "CANDIDATE_Y")):
        _deliver(
            controller,
            requests[provider],
            choice=choice,
            evidence=f"{provider} exact prose",
        )
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    decision = _document(controller).reconciled_decisions[0]
    assert decision.decision == "UNRESOLVED"
    assert decision.basis == "PROVIDER_DISAGREEMENT"
    assert decision.gpt_decision == "CANDIDATE_X"
    assert decision.grok_decision == "CANDIDATE_Y"
    assert decision.gpt_visual_evidence == "gpt exact prose"
    assert decision.grok_visual_evidence == "grok exact prose"


def test_group_member_cannot_cite_another_groups_crop(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    selected = [names["word_r1"], names["note_r1"], names["word_r2"]]
    request = _requests(controller, selected)["gpt"]
    published = _published(controller)
    shared = next(
        group for group in published.groups or [] if len(group.dispute_ids) == 2
    )
    other = next(
        group for group in published.groups or [] if len(group.dispute_ids) == 1
    )
    wrong = EvidenceReference(
        kind="CROP", evidence_id=other.crop_id, page_id=other.page_id
    )
    with pytest.raises(
        ReviewOutputRejected, match="REFERENCE_NOT_IN_DISPUTE"
    ) as raised:
        _deliver(
            controller,
            request,
            reference_overrides={shared.dispute_ids[0]: wrong},
        )
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert raised.value.reason == "mismatched"
    assert comparison.page_originals[0].gpt_diplomatic_transcription == "\n".join(
        _pages("gpt")[0].diplomatic_transcription.splitlines()
    )


def test_full_page_fallback_reference_is_bound_and_wrong_kind_is_rejected(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path, no_bbox=(0,))
    selected = [names["word_r1"], names["note_r1"]]
    requests = _requests(controller, selected)
    group = (_published(controller).groups or [])[0]
    assert group.window_kind == "FULL_PAGE_FALLBACK"
    for provider, request in requests.items():
        _deliver(controller, request, reference_index=1)
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    assert all(
        item.gpt_evidence_ref.kind == "FULL_PAGE_FALLBACK"
        and item.grok_evidence_ref.kind == "FULL_PAGE_FALLBACK"
        for item in _document(controller).reconciled_decisions
    )

    wrong, _comparison, wrong_names = _setup(tmp_path / "wrong", no_bbox=(0,))
    request = _requests(wrong, [wrong_names["word_r1"]])["gpt"]
    published = _published(wrong)
    group = (published.groups or [])[0]
    bad = EvidenceReference(
        kind="CROP", evidence_id=group.crop_id, page_id=group.page_id
    )
    with pytest.raises(ReviewOutputRejected, match="REFERENCE_NOT_IN_DISPUTE"):
        _deliver(
            wrong,
            request,
            reference_overrides={wrong_names["word_r1"]: bad},
        )
        wrong.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")


def test_missing_evidence_fields_are_rejected_not_downgraded(tmp_path: Path) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request_path = _requests(controller, [names["word_r1"]])["gpt"]
    request = WorkRequest.model_validate_json(request_path.read_bytes())
    payload = ChallengeOutput(
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
                dispute_id=names["word_r1"],
                decision="CANDIDATE_X",
                visual_evidence="Unbound prose is not enough.",
            )
        ],
    ).model_dump(mode="json")
    target = _root(controller) / request.output_relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ReviewOutputRejected, match="MISSING_EVIDENCE_BINDING"
    ) as raised:
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert raised.value.reason == "schema"
    round_state = _document(controller).challenge_rounds[0]
    assert round_state.challenge_contract == "EVIDENCE_BOUND_V1"


def test_new_round_contract_is_in_ledger_input_instructions_and_schema(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    request = _requests(controller, [names["word_r1"]])["gpt"]
    published = _published(controller)
    assert published.schema_version == "review-challenge-input-v3"
    assert published.challenge_contract == "EVIDENCE_BOUND_V1"
    ledger = (
        controller.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "ledger.jsonl"
    ).read_text(encoding="utf-8")
    events = [json.loads(line) for line in ledger.splitlines()]
    challenge_events = [
        item
        for item in events
        if item["event_type"] == "work_published" and item["stage"] == "challenge"
    ]
    assert {item["challenge_contract"] for item in challenge_events} == {
        "EVIDENCE_BOUND_V1"
    }
    instructions = request.parent / "instructions_gpt.txt"
    schema = request.parent / "output_schema.json"
    assert "evidence_ref" in instructions.read_text(encoding="utf-8")
    schema_payload = json.loads(schema.read_text(encoding="utf-8"))
    assert schema_payload["round_contract"] == "EVIDENCE_BOUND_V1"
    assert "evidence_assessment" in schema_payload["decisions"]["fields"]


@pytest.mark.parametrize("missing", ["disputes", "crop", "task_copy"])
def test_missing_or_inconsistent_published_task_fails_closed(
    tmp_path: Path, missing: str
) -> None:
    controller, _comparison, names = _setup(tmp_path / missing)
    _requests(controller, [names["word_r1"]])
    published = _published(controller)
    group = (published.groups or [])[0]
    if missing == "disputes":
        (_input(controller) / "disputes.json").unlink()
    elif missing == "crop":
        (_input(controller) / "crops" / f"{group.crop_id}.png").unlink()
    else:
        (
            _root(controller)
            / "challenge_round_01"
            / "tasks"
            / "gpt"
            / "input"
            / "disputes.json"
        ).write_text("{}", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError):
        controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")


def test_replay_and_versions_keep_supported_result_and_prior_bytes(
    tmp_path: Path,
) -> None:
    controller, _comparison, names = _setup(tmp_path)
    interim = controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)
    prior = {name: path.read_bytes() for name, path in interim.items()}
    requests = _requests(controller, [names["word_r1"]])
    for provider, request in requests.items():
        _deliver(controller, request)
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    _close_challenge(controller)
    completed = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert {name: path.read_bytes() for name, path in interim.items()} == prior
    version_two = _root(controller) / "final" / "versions" / "v002"
    assert version_two.is_dir()
    assert (version_two / "review_report.json").read_bytes() == completed[
        "review_report.json"
    ].read_bytes()
    final_snapshot = _snapshot(_root(controller) / "final")
    checkpoint = (
        controller.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "state.json"
    )
    checkpoint.unlink()
    replayed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    decision = _document(replayed).reconciled_decisions[0]
    assert decision.result_class == "AUTOMATICALLY_SUPPORTED"
    assert decision.independently_verified is False
    assert _snapshot(_root(replayed) / "final") == final_snapshot


def test_source_crop_and_exact_text_stay_immutable(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    before_source = _snapshot(tmp_path / "batches")
    before_review_input = _snapshot(_root(controller) / "input")
    dispute = next(
        item for item in comparison.disputes if item.dispute_id == names["word_r1"]
    )
    requests = _requests(controller, [dispute.dispute_id])
    crop_before = _snapshot(_input(controller))
    for provider, request in requests.items():
        _deliver(controller, request, choice="CANDIDATE_Y")
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    decision = _document(controller).reconciled_decisions[0]
    assert decision.selected_reading == dispute.grok_text
    assert decision.candidate_x_text == dispute.gpt_text
    assert decision.candidate_y_text == dispute.grok_text
    assert _snapshot(tmp_path / "batches") == before_source
    assert _snapshot(_root(controller) / "input") == before_review_input
    assert _snapshot(_input(controller)) == crop_before
