"""A4: explicit challenge-stage closure, stopping, and local finalization.

Synthetic fixtures only. No database, network, provider API, archive write,
or operational review run. Tests are not executed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_review_challenge_selection import (
    _answer_and_accept,
    _controller_dir,
    _document,
    _ledger_lines,
    _publish,
    _root,
    _setup,
)
from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    NOW,
    _agreeing_pages,
    _controller as _workflow_controller,
    _deliver_first_pass,
    _png,
    _snapshot,
)
from transcription_review_agent.cli import main
from transcription_review_agent.review.config import ReviewConfig
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewOutputRejected,
    ReviewPolicyError,
)
from transcription_review_agent.review.models import QualityLabel, ReconciledDecision
from transcription_review_agent.review.runtime import (
    ChallengeClosedEntry,
    ChallengeReconciledEntry,
    ReviewFinalizedEntry,
    ReviewStore,
    accumulate_decision,
    apply_review_entry,
    challenge_round_failed,
    challenge_round_open,
    describe_unresolved,
    find_document,
    parse_review_entry,
)

REASON = "Operator closed the challenge stage for this local proposal."


def _close(controller: ReviewController, mode: str, reason: str = REASON):
    return controller.close_challenge(BATCH_ID, DOCUMENT_ID, mode=mode, reason=reason)


def _ledger(controller: ReviewController) -> str:
    return (_controller_dir(controller) / "ledger.jsonl").read_text(encoding="utf-8")


def _events(controller: ReviewController, event_type: str) -> list[dict]:
    return [
        item for item in _ledger_lines(controller) if item["event_type"] == event_type
    ]


def _report(paths: dict[str, Path]) -> dict:
    return json.loads(paths["review_report.json"].read_text(encoding="utf-8"))


def _new_evidence(tmp_path: Path) -> Path:
    path = tmp_path / "new-evidence.png"
    path.write_bytes(_png(8, 8))
    return path


def _decision(
    *,
    dispute_id: str = "dsp_1",
    code: str = "CANDIDATE_X",
    third: str | None = None,
    selected: str | None = "Alpha",
) -> ReconciledDecision:
    unresolved = code == "UNRESOLVED"
    return ReconciledDecision(
        dispute_id=dispute_id,
        decision=code,  # type: ignore[arg-type]
        third_reading=third,
        agreed=not unresolved,
        remaining_uncertainty=unresolved,
        result_class="UNRESOLVED" if unresolved else "AUTOMATICALLY_SUPPORTED",
        basis="PROVIDER_DISAGREEMENT" if unresolved else "DUAL_MODEL_VISUAL_SUPPORT",
        selected_reading=None if unresolved else selected,
        independently_verified=False,
        eligible_for_writer_memory=False,
        eligible_for_publication=False,
    )


def _cli_args(tmp_path: Path, operation: str, *extra: str) -> list[str]:
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


# --- intentional zero-round completion ----------------------------------------


def test_intentional_zero_round_completion_keeps_every_alternative(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    document = _document(controller)
    assert document.challenge_rounds == []
    assert document.challenge_closure is None
    before = _snapshot(controller.batches_dir)

    status = _close(controller, "NO_CHALLENGE_REQUESTED")
    assert status.challenge_closure == "NO_CHALLENGE_REQUESTED"
    assert status.finalized is False
    document = _document(controller)
    assert document.challenge_rounds == []
    assert [item.dispute_id for item in document.reconciled_decisions] == []
    unresolved = document.challenge_closure.unresolved
    assert [item.dispute_id for item in unresolved] == ids
    assert {item.status for item in unresolved} == {"NOT_CHALLENGED"}

    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = _report(paths)
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert report["review_completion"] == "completed"
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert report["decisions"] == []
    assert report["challenge_closure"]["mode"] == "NO_CHALLENGE_REQUESTED"
    assert report["challenge_closure"]["initiated_by"] == "OPERATOR"
    assert report["challenge_closure"]["reason"] == REASON
    assert report["production_writeback"] is False
    assert "challenge_selections" not in report
    assert "[Alpha / Alfa] one" in text
    assert "Bravo [two / twa]" in text
    assert "[Charlie / Charly] three" in text
    assert "Echo [five / fyve]" in text
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is True
    assert _events(controller, "challenge_closed")
    assert _events(controller, "review_finalized")
    assert _snapshot(controller.batches_dir) == before


def test_zero_round_mode_is_refused_after_a_round_exists(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=ids[:1]),
            ids[:1],
        )
    with pytest.raises(ReviewPolicyError, match="no challenge round was opened"):
        _close(controller, "NO_CHALLENGE_REQUESTED")


def test_closure_requires_a_recorded_comparison(tmp_path: Path) -> None:
    controller = _workflow_controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(controller, "gpt", _agreeing_pages())
    _deliver_first_pass(controller, "grok", _agreeing_pages())
    with pytest.raises(ReviewPolicyError, match="comparison must be recorded"):
        _close(controller, "NO_CHALLENGE_REQUESTED")


# --- no closure means provisional only ----------------------------------------


def test_no_closure_cannot_complete_and_inactivity_is_not_closure(
    tmp_path: Path,
) -> None:
    controller, _ids, _other = _setup(tmp_path)
    assert controller.status(BATCH_ID, DOCUMENT_ID).challenge_closure is None
    with pytest.raises(ReviewPolicyError, match="explicitly closed challenge stage"):
        controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is False
    assert not _events(controller, "challenge_closed")

    interim = controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)
    report = _report(interim)
    assert report["review_completion"] == "interim_provisional"
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert "challenge_closure" not in report
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is False
    assert controller.status(BATCH_ID, DOCUMENT_ID).challenge_closure is None
    assert _ledger(controller).count('"event_type":"review_provisional"') == 1
    assert _ledger(controller).count('"event_type":"review_finalized"') == 0
    assert not _events(controller, "challenge_closed")


def test_selected_rounds_complete_is_refused_when_no_round_exists(
    tmp_path: Path,
) -> None:
    controller, _ids, _other = _setup(tmp_path)
    with pytest.raises(ReviewPolicyError, match="at least one challenge round"):
        _close(controller, "SELECTED_ROUNDS_COMPLETE")


# --- open dispatch cannot be closed as completed ------------------------------


def test_open_dispatch_cannot_be_closed_or_finalized(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    _publish(controller, "gpt", selected_dispute_ids=ids[:1])
    assert challenge_round_open(_document(controller).challenge_rounds[0]) is True
    assert challenge_round_failed(_document(controller).challenge_rounds[0]) is False

    with pytest.raises(ReviewPolicyError, match="open provider dispatch"):
        _close(controller, "SELECTED_ROUNDS_COMPLETE")
    with pytest.raises(ReviewPolicyError, match="open provider dispatch"):
        _close(controller, "PROVIDER_TERMINAL_FAILURE")
    with pytest.raises(ReviewPolicyError, match="still pending"):
        controller.finalize(BATCH_ID, DOCUMENT_ID)
    with pytest.raises(ReviewPolicyError, match="still pending"):
        controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)
    assert _document(controller).challenge_closure is None
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is False


def test_retryable_failure_is_still_an_open_dispatch(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    _publish(controller, "gpt", selected_dispute_ids=ids[:1])
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="challenge"
    )
    assert (
        _document(controller).challenge_rounds[0].providers[0].status.value
        == "RETRYABLE_FAILURE"
    )
    with pytest.raises(ReviewPolicyError, match="open provider dispatch"):
        _close(controller, "PROVIDER_TERMINAL_FAILURE")
    with pytest.raises(ReviewPolicyError, match="still pending"):
        controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)


def test_quota_on_one_side_does_not_close_while_the_other_is_still_dispatched(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = ids[:1]
    _publish(controller, "gpt", selected_dispute_ids=chosen)
    _publish(controller, "grok", selected_dispute_ids=chosen)
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "grok", kind="quota", stage="challenge"
    )
    with pytest.raises(ReviewPolicyError, match="open provider dispatch"):
        _close(controller, "PROVIDER_TERMINAL_FAILURE")
    assert challenge_round_failed(_document(controller).challenge_rounds[0]) is False


# --- partial-round closure with unresolved disputes ---------------------------


def test_partial_round_closure_completes_with_unselected_alternatives(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[0], ids[1]]
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=chosen),
            chosen,
        )
    _close(controller, "SELECTED_ROUNDS_COMPLETE")
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = _report(paths)
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    document = _document(controller)
    assert report["review_completion"] == "completed"
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert [item["dispute_id"] for item in report["decisions"]] == chosen
    unresolved = {
        item["dispute_id"]: item["status"]
        for item in report["challenge_closure"]["unresolved"]
    }
    assert unresolved == {ids[2]: "NOT_SELECTED", ids[3]: "NOT_SELECTED"}
    assert "Alpha one" in text
    assert "[Charlie / Charly] three" in text
    assert "Echo [five / fyve]" in text
    assert [item.dispute_id for item in document.reconciled_decisions] == chosen
    assert ids[2] not in {item.dispute_id for item in document.reconciled_decisions}


# --- terminal provider failure ------------------------------------------------


def test_terminal_quota_failure_closes_selected_items_unresolved(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[0]]
    gpt_request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    _publish(controller, "grok", selected_dispute_ids=chosen)
    _answer_and_accept(controller, "gpt", gpt_request, chosen)
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "grok", kind="quota", stage="challenge"
    )
    round_state = _document(controller).challenge_rounds[0]
    assert challenge_round_open(round_state) is False
    assert challenge_round_failed(round_state) is True
    with pytest.raises(ReviewPolicyError, match="PROVIDER_TERMINAL_FAILURE"):
        _close(controller, "SELECTED_ROUNDS_COMPLETE")

    _close(controller, "PROVIDER_TERMINAL_FAILURE", reason="Grok hit a quota block.")
    document = _document(controller)
    assert document.challenge_closure.mode == "PROVIDER_TERMINAL_FAILURE"
    assert document.challenge_closure.failed_round_indexes == [1]
    assert document.reconciled_decisions == []
    unresolved = {
        item.dispute_id: item.status for item in document.challenge_closure.unresolved
    }
    assert unresolved[ids[0]] == "PROVIDER_FAILURE"
    assert unresolved[ids[1]] == "NOT_SELECTED"

    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = _report(paths)
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert report["review_completion"] == "completed"
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert report["decisions"] == []
    assert "[Alpha / Alfa] one" in text
    assert "winning" not in text.lower()


def test_invalid_output_is_a_confirmed_terminal_failure(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    request = _publish(controller, "gpt", selected_dispute_ids=ids[:1])
    target = (
        _root(controller)
        / json.loads(request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"provider":"nope"}', encoding="utf-8")
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    round_state = _document(controller).challenge_rounds[0]
    assert round_state.providers[0].status.value == "INVALID_OUTPUT"
    assert challenge_round_failed(round_state) is True
    _close(controller, "PROVIDER_TERMINAL_FAILURE")
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert _report(paths)["decisions"] == []
    assert "[Alpha / Alfa] one" in paths["proposed_transcription.txt"].read_text(
        encoding="utf-8"
    )


# --- multiple rounds and accumulated decisions --------------------------------


def test_later_round_accumulates_decisions_and_leaves_unselected_visible(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    controller.config = ReviewConfig(max_challenge_rounds=2)
    first = [ids[0], ids[1]]
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=first),
            first,
            per_id={ids[1]: "UNRESOLVED"},
        )
    evidence = _new_evidence(tmp_path)
    second = [ids[1], ids[3]]
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(
                controller,
                provider,
                evidence="new_crop",
                evidence_path=evidence,
                selected_dispute_ids=second,
            ),
            second,
            decision="CANDIDATE_Y",
        )
    _close(controller, "SELECTED_ROUNDS_COMPLETE")
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = _report(paths)
    by_id = {item["dispute_id"]: item for item in report["decisions"]}
    assert by_id[ids[0]]["decision"] == "CANDIDATE_X"
    assert by_id[ids[1]]["decision"] == "CANDIDATE_Y"
    assert by_id[ids[1]]["other_round_decisions"][0]["decision"] == "UNRESOLVED"
    assert ids[2] not in by_id
    assert by_id[ids[3]]["decision"] == "CANDIDATE_Y"
    unresolved = {
        item["dispute_id"]: item["status"]
        for item in report["challenge_closure"]["unresolved"]
    }
    assert unresolved == {ids[2]: "NOT_SELECTED"}
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert "[Charlie / Charly] three" in text
    assert "Echo fyve" in text or "Echo five" in text


def test_later_round_rejects_resolved_ids_and_omitted_selection_targets_unresolved(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    controller.config = ReviewConfig(max_challenge_rounds=2)
    first = [ids[0], ids[1]]
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=first),
            first,
            per_id={ids[1]: "UNRESOLVED"},
        )
    evidence = _new_evidence(tmp_path)
    with pytest.raises(ReviewPolicyError, match="already resolved"):
        _publish(
            controller,
            "gpt",
            evidence="new_crop",
            evidence_path=evidence,
            selected_dispute_ids=[ids[0], ids[2]],
        )
    request = _publish(controller, "gpt", evidence="new_crop", evidence_path=evidence)
    selected = _document(controller).challenge_rounds[1].selected_dispute_ids
    assert ids[0] not in selected
    assert ids[1] in selected
    assert set(selected) <= set(ids)
    published = json.loads(
        (
            _root(controller) / "challenge_round_02" / "input" / "disputes.json"
        ).read_text(encoding="utf-8")
    )
    assert ids[0] not in [item["dispute_id"] for item in published["disputes"]]
    assert request.name == "work_request.json"


# --- contradictory evidence ---------------------------------------------------


def test_accumulate_decision_preserves_uncertainty_instead_of_overwriting() -> None:
    earlier_x = _decision(code="CANDIDATE_X", selected="Alpha")
    later_y = _decision(code="CANDIDATE_Y", selected="Alfa")
    later_unresolved = _decision(code="UNRESOLVED", selected=None)
    later_x = _decision(code="CANDIDATE_X", selected="Alpha")

    held = accumulate_decision(earlier_x, later_unresolved)
    assert held.decision == "CANDIDATE_X"
    assert held.selected_reading == "Alpha"
    assert held.other_round_decisions[0].decision == "UNRESOLVED"

    same = accumulate_decision(earlier_x, later_x)
    assert same.decision == "CANDIDATE_X"
    assert same.other_round_decisions[0].decision == "CANDIDATE_X"

    from_unresolved = accumulate_decision(later_unresolved, later_y)
    assert from_unresolved.decision == "CANDIDATE_Y"
    assert from_unresolved.other_round_decisions[0].decision == "UNRESOLVED"

    conflict = accumulate_decision(earlier_x, later_y)
    assert conflict.decision == "UNRESOLVED"
    assert conflict.third_reading is None
    assert conflict.selected_reading is None
    assert conflict.agreed is False
    assert conflict.remaining_uncertainty is True
    assert conflict.result_class == "UNRESOLVED"
    assert conflict.basis == "CONTRADICTORY_ROUND_EVIDENCE"
    assert conflict.independently_verified is False
    assert conflict.eligible_for_writer_memory is False
    assert conflict.eligible_for_publication is False
    assert [item.decision for item in conflict.other_round_decisions] == [
        "CANDIDATE_X",
        "CANDIDATE_Y",
    ]


def test_contradictory_later_round_does_not_silently_keep_a_winner(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[0]]
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=chosen),
            chosen,
        )
    state, document = controller._require_document(BATCH_ID, DOCUMENT_ID)
    round1 = document.challenge_rounds[0]
    round2 = round1.model_copy(update={"round_index": 2, "reconciled": False})
    document = document.model_copy(update={"challenge_rounds": [round1, round2]})
    state = state.model_copy(update={"documents": [document]})
    earlier = document.reconciled_decisions[0]
    later = earlier.model_copy(
        update={
            "decision": "CANDIDATE_Y",
            "gpt_decision": "CANDIDATE_Y",
            "grok_decision": "CANDIDATE_Y",
            "selected_reading": "Alfa one",
            "other_round_decisions": None,
        }
    )
    updated = apply_review_entry(
        state,
        ChallengeReconciledEntry(
            sequence=state.applied_ledger_sequence + 1,
            occurred_at=NOW,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            round_index=2,
            decisions=[later],
            accumulates_history=True,
        ),
    )
    standing = find_document(updated, DOCUMENT_ID).reconciled_decisions[0]
    assert standing.decision == "UNRESOLVED"
    assert standing.basis == "CONTRADICTORY_ROUND_EVIDENCE"
    assert standing.selected_reading is None
    assert standing.independently_verified is False
    assert {item.decision for item in standing.other_round_decisions} == {
        "CANDIDATE_X",
        "CANDIDATE_Y",
    }


# --- configured round cap and no-new-information stop -------------------------


def test_configured_round_cap_stops_without_a_hardcoded_second_round(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    assert controller.config.max_challenge_rounds == 1
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=ids[:1]),
            ids[:1],
            decision="UNRESOLVED",
        )
    evidence = _new_evidence(tmp_path)
    with pytest.raises(ReviewPolicyError, match="challenge round cap is reached"):
        _publish(
            controller,
            "gpt",
            evidence="new_crop",
            evidence_path=evidence,
            selected_dispute_ids=ids[:1],
        )
    assert len(_document(controller).challenge_rounds) == 1
    controller.config = ReviewConfig(max_challenge_rounds=2)
    with pytest.raises(ReviewPolicyError, match="cap has not been reached"):
        _close(controller, "ROUND_CAP")
    controller.config = ReviewConfig(max_challenge_rounds=1)
    _close(controller, "ROUND_CAP")
    assert _document(controller).challenge_closure.mode == "ROUND_CAP"
    assert _document(controller).challenge_closure.max_challenge_rounds == 1
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert _report(paths)["challenge_closure"]["mode"] == "ROUND_CAP"
    assert not (_root(controller) / "challenge_round_02").exists()


def test_no_new_information_is_recorded_instead_of_launching_another_round(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    controller.config = ReviewConfig(max_challenge_rounds=2)
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=ids[:2]),
            ids[:2],
            per_id={ids[1]: "UNRESOLVED"},
        )
    with pytest.raises(
        ReviewPolicyError, match="no repeated challenge without new evidence"
    ):
        _publish(controller, "gpt")
    page_png = _root(controller) / "input" / "page_001.png"
    with pytest.raises(ReviewPolicyError, match="already examined"):
        _publish(
            controller,
            "gpt",
            evidence="different_source_scan",
            evidence_path=page_png,
        )
    _close(controller, "NO_NEW_INFORMATION", reason="No new scan is available.")
    assert _document(controller).challenge_closure.mode == "NO_NEW_INFORMATION"
    assert len(_document(controller).challenge_rounds) == 1
    assert not (_root(controller) / "challenge_round_02").exists()
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = _report(paths)
    assert report["review_completion"] == "completed"
    assert report["challenge_closure"]["mode"] == "NO_NEW_INFORMATION"
    unresolved = {
        item["dispute_id"]: item["status"]
        for item in report["challenge_closure"]["unresolved"]
    }
    assert unresolved[ids[1]] == "SELECTED_UNRESOLVED"
    assert unresolved[ids[2]] == "NOT_SELECTED"


def test_two_configured_rounds_then_the_cap(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    controller.config = ReviewConfig(max_challenge_rounds=2)
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=[ids[1]]),
            [ids[1]],
            decision="UNRESOLVED",
        )
    evidence = _new_evidence(tmp_path)
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(
                controller,
                provider,
                evidence="new_crop",
                evidence_path=evidence,
                selected_dispute_ids=[ids[1]],
            ),
            [ids[1]],
            decision="UNRESOLVED",
        )
    more = tmp_path / "third.png"
    more.write_bytes(_png(9, 9))
    with pytest.raises(ReviewPolicyError, match="challenge round cap is reached"):
        _publish(
            controller,
            "gpt",
            evidence="another_occurrence",
            evidence_path=more,
            selected_dispute_ids=[ids[1]],
        )
    _close(controller, "ROUND_CAP")
    assert _document(controller).challenge_closure.round_count == 2
    assert _document(controller).challenge_closure.max_challenge_rounds == 2


# --- ledger replay and legacy compatibility -----------------------------------


def test_replay_restores_closure_accumulated_decisions_and_completion(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    controller.config = ReviewConfig(max_challenge_rounds=2)
    first = [ids[0], ids[1]]
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=first),
            first,
            per_id={ids[1]: "UNRESOLVED"},
        )
    evidence = _new_evidence(tmp_path)
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(
                controller,
                provider,
                evidence="new_crop",
                evidence_path=evidence,
                selected_dispute_ids=[ids[1]],
            ),
            [ids[1]],
            decision="CANDIDATE_Y",
        )
    _close(controller, "SELECTED_ROUNDS_COMPLETE")
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    saved = {name: path.read_bytes() for name, path in paths.items()}
    ledger_text = _ledger(controller)
    (_controller_dir(controller) / "state.json").unlink()
    replayed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    document = _document(replayed)
    assert document.challenge_closure.mode == "SELECTED_ROUNDS_COMPLETE"
    assert document.finalized is True
    by_id = {item.dispute_id: item.decision for item in document.reconciled_decisions}
    assert by_id == {ids[0]: "CANDIDATE_X", ids[1]: "CANDIDATE_Y"}
    assert document.challenge_rounds[0].selected_dispute_ids == first
    assert document.challenge_rounds[1].selected_dispute_ids == [ids[1]]
    assert _ledger(replayed) == ledger_text
    assert {name: path.read_bytes() for name, path in paths.items()} == saved


def test_legacy_finalized_event_without_closure_stays_finalized(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=ids[:1]),
            ids[:1],
        )
    store = ReviewStore(controller.reviews_dir, BATCH_ID)
    current = store.load()
    assert current is not None
    store.commit(
        ReviewFinalizedEntry(
            sequence=current.applied_ledger_sequence + 1,
            occurred_at=NOW,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            quality=QualityLabel.UNRESOLVED_CONTENT,
            final_sha256={"proposed_transcription.txt": "ab" * 32},
        )
    )
    document = _document(controller)
    assert document.finalized is True
    assert document.challenge_closure is None
    assert "challenge_closed" not in _ledger(controller)
    (_controller_dir(controller) / "state.json").unlink()
    replayed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    document = _document(replayed)
    assert document.finalized is True
    assert document.challenge_closure is None
    assert document.execution_state.value == "COMPLETED"


def test_legacy_checkpoint_without_a_closure_field_still_replays(
    tmp_path: Path,
) -> None:
    controller, _ids, _other = _setup(tmp_path)
    path = _controller_dir(controller) / "state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["documents"][0].pop("challenge_closure", None)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    resumed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    document = _document(resumed)
    assert document.challenge_closure is None
    assert document.finalized is False


def test_challenge_closed_event_is_append_only_and_auditable() -> None:
    line = {
        "sequence": 12,
        "occurred_at": "2026-01-01T00:00:00Z",
        "batch_id": BATCH_ID,
        "document_id": DOCUMENT_ID,
        "event_type": "challenge_closed",
        "closure": {
            "mode": "NO_CHALLENGE_REQUESTED",
            "reason": "Operator skipped the challenge stage.",
            "initiated_by": "OPERATOR",
            "comparison_sha256": "ab" * 32,
            "round_count": 0,
            "failed_round_indexes": [],
            "unresolved": [
                {
                    "dispute_id": "dsp_test",
                    "status": "NOT_CHALLENGED",
                    "selected_in_rounds": [],
                    "basis": None,
                }
            ],
        },
    }
    entry = parse_review_entry(json.loads(json.dumps(line)))
    assert isinstance(entry, ChallengeClosedEntry)
    assert entry.closure.mode == "NO_CHALLENGE_REQUESTED"
    assert entry.closure.initiated_by == "OPERATOR"
    dumped = json.loads(entry.model_dump_json())
    assert dumped["event_type"] == "challenge_closed"
    with pytest.raises(ReviewIntegrityError):
        parse_review_entry(
            {**line, "closure": {**line["closure"], "mode": "INFERRED_TIMEOUT"}}
        )


def test_blank_reason_unknown_mode_and_repeat_closure_are_rejected(
    tmp_path: Path,
) -> None:
    controller, _ids, _other = _setup(tmp_path)
    with pytest.raises(ReviewPolicyError, match="unknown challenge closure mode"):
        _close(controller, "INFERRED_TIMEOUT")
    with pytest.raises(ReviewPolicyError, match="explicit reason"):
        _close(controller, "NO_CHALLENGE_REQUESTED", reason="   ")
    _close(controller, "NO_CHALLENGE_REQUESTED")
    sequence = len(_ledger_lines(controller))
    again = _close(controller, "NO_CHALLENGE_REQUESTED")
    assert again.challenge_closure == "NO_CHALLENGE_REQUESTED"
    assert len(_ledger_lines(controller)) == sequence
    with pytest.raises(ReviewPolicyError, match="already closed"):
        _close(controller, "NO_CHALLENGE_REQUESTED", reason="A different reason.")
    with pytest.raises(ReviewPolicyError, match="challenge stage is closed"):
        _publish(controller, "gpt")


# --- immutable versioned outputs and no writeback -----------------------------


def test_completed_proposal_is_the_next_version_and_does_not_write_back(
    tmp_path: Path,
) -> None:
    controller, _ids, _other = _setup(tmp_path)
    before_source = _snapshot(controller.batches_dir)
    interim = controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)
    v001 = {name: path.read_bytes() for name, path in interim.items()}
    assert "final/versions/v001/" in interim["review_report.json"].as_posix()
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is False

    _close(controller, "NO_CHALLENGE_REQUESTED")
    completed = controller.finalize(BATCH_ID, DOCUMENT_ID)
    v001_after = {name: path.read_bytes() for name, path in interim.items()}
    assert v001_after == v001
    report = _report(completed)
    assert report["review_completion"] == "completed"
    assert report["production_writeback"] is False
    manifest = json.loads(completed["review_manifest.json"].read_text(encoding="utf-8"))
    assert manifest["production_writeback"] is False
    version = _root(controller) / "final" / "versions"
    assert (version / "v001" / "review_report.json").read_bytes() == v001[
        "review_report.json"
    ]
    assert (version / "v002" / "review_report.json").read_bytes() == completed[
        "review_report.json"
    ].read_bytes()
    assert controller.status(BATCH_ID, DOCUMENT_ID).finalized is True
    assert _snapshot(controller.batches_dir) == before_source
    document = _document(controller)
    assert describe_unresolved(document) == list(document.challenge_closure.unresolved)


def test_cli_requires_explicit_close_then_finalizes(tmp_path: Path) -> None:
    controller, _ids, _other = _setup(tmp_path)
    del controller
    assert main(_cli_args(tmp_path, "close-challenge")) == 2
    assert (
        main(
            _cli_args(
                tmp_path,
                "close-challenge",
                "--closure-mode",
                "NO_CHALLENGE_REQUESTED",
                "--reason",
                REASON,
            )
        )
        == 0
    )
    assert main(_cli_args(tmp_path, "finalize")) == 0
    report = json.loads(
        (
            _root(
                ReviewController(
                    batches_dir=tmp_path / "batches",
                    reviews_dir=tmp_path / "reviews",
                )
            )
            / "final"
            / "review_report.json"
        ).read_text(encoding="utf-8")
    )
    assert report["review_completion"] == "completed"
    assert report["challenge_closure"]["mode"] == "NO_CHALLENGE_REQUESTED"
    assert report["production_writeback"] is False
