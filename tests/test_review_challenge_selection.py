"""A1: a persisted, frozen, explicit subset of disputes for one challenge round.

Synthetic fixtures only. No database, network, provider, or archive API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    NOW,
    _controller,
    _deliver_first_pass,
    _png,
    _close_challenge,
    _snapshot,
    _write_output,
)
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.cli import main
from transcription_review_agent.review.config import ReviewConfig
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewOutputRejected,
    ReviewPolicyError,
)
from transcription_review_agent.review.models import (
    BBox,
    ChallengeDecision,
    ChallengeOutput,
    CoverageReport,
    LineReading,
    PageTranscription,
    TextRegion,
    WorkRequest,
)
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.runtime import (
    WorkPublishedEntry,
    apply_review_entry,
    find_document,
    round_selection,
)


def _multi_page(page_id: str, order: int, texts: list[str]) -> PageTranscription:
    regions = [
        TextRegion(
            region_id=f"r{index + 1}",
            order=index + 1,
            lines=[LineReading(line_id=f"r{index + 1}-l1", text=text)],
            bbox=BBox(x=1 + 3 * index, y=1, width=3, height=3),
        )
        for index, text in enumerate(texts)
    ]
    return PageTranscription(
        page_id=page_id,
        reading_order=order,
        diplomatic_transcription="\n".join(texts),
        regions=regions,
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=[],
        coverage=CoverageReport(
            coverage_complete=True,
            unresolved_text_present=False,
            examined_region_ids=[region.region_id for region in regions],
        ),
        cancellations=[],
        insertions=[],
        marginal_notes=[],
        inaccessible_or_ambiguous=[],
    )


GPT_TEXT = {
    "page_001": ["Alpha one", "Bravo two", "Charlie three", "Delta four"],
    "page_002": ["Echo five"],
}
GROK_TEXT = {
    "page_001": ["Alfa one", "Bravo twa", "Charly three", "Delta, four"],
    "page_002": ["Echo fyve"],
}


def _setup(tmp_path: Path):
    """Two providers, four substantive disputes and one punctuation dispute."""

    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    for provider, text in (("gpt", GPT_TEXT), ("grok", GROK_TEXT)):
        _deliver_first_pass(
            controller,
            provider,
            [
                _multi_page("page_001", 1, text["page_001"]),
                _multi_page("page_002", 2, text["page_002"]),
            ],
        )
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    substantive = [item.dispute_id for item in comparison.disputes if item.substantive]
    other = [item.dispute_id for item in comparison.disputes if not item.substantive]
    assert len(substantive) == 4
    assert other, "the fixture must contain a non-substantive dispute"
    return controller, substantive, other


def _root(controller: ReviewController) -> Path:
    return document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)


def _controller_dir(controller: ReviewController) -> Path:
    return controller.reviews_dir / f"batch_{BATCH_ID}" / "controller"


def _ledger_lines(controller: ReviewController) -> list[dict]:
    text = (_controller_dir(controller) / "ledger.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _document(controller: ReviewController):
    return controller._require_document(BATCH_ID, DOCUMENT_ID)[1]


def _publish(controller: ReviewController, provider: str, **kwargs) -> Path:
    return controller.publish(
        BATCH_ID, DOCUMENT_ID, provider, stage="challenge", **kwargs
    )


def _answer(
    controller: ReviewController,
    request_path: Path,
    ids: list[str],
    *,
    decision: str = "CANDIDATE_X",
    per_id: dict[str, str] | None = None,
) -> None:
    request = WorkRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
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
                decision=(per_id or {}).get(dispute_id, decision),  # type: ignore[arg-type]
                visual_evidence="The crop shows the first candidate.",
            )
            for dispute_id in ids
        ],
    )
    _write_output(controller, request_path, challenge.model_dump(mode="json"))


def _answer_and_accept(controller, provider, request_path, ids, **kwargs) -> None:
    _answer(controller, request_path, ids, **kwargs)
    controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")


def _work_events(controller: ReviewController) -> list[dict]:
    return [
        item
        for item in _ledger_lines(controller)
        if item["event_type"] == "work_published"
    ]


def _challenge_events(controller: ReviewController) -> list[dict]:
    return [item for item in _work_events(controller) if item["stage"] == "challenge"]


# --- two selected out of many -------------------------------------------------


def test_two_selected_of_many_publish_and_accept_only_those_ids(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[1], ids[3]]
    gpt_request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    grok_request = _publish(controller, "grok", selected_dispute_ids=chosen)

    # Persisted in round state and in both append-only publication events.
    round_state = _document(controller).challenge_rounds[0]
    assert round_state.selected_dispute_ids == chosen
    events = _challenge_events(controller)
    assert [item["selected_dispute_ids"] for item in events] == [chosen, chosen]
    comparison_sha = _document(controller).comparison_sha256
    assert {item["selection_comparison_sha256"] for item in events} == {comparison_sha}

    # The task bundle carries only the selected disputes and their crops.
    stage = _root(controller) / "challenge_round_01" / "input"
    disputes = json.loads((stage / "disputes.json").read_text(encoding="utf-8"))
    assert [item["dispute_id"] for item in disputes["disputes"]] == chosen
    assert sorted(path.name for path in (stage / "crops").glob("*.json")) == sorted(
        f"crop_{item}.json" for item in chosen
    )
    for provider in ("gpt", "grok"):
        task_dir = _root(controller) / "challenge_round_01" / "tasks" / provider
        listed = json.loads(
            (task_dir / "input" / "disputes.json").read_text(encoding="utf-8")
        )
        assert [item["dispute_id"] for item in listed["disputes"]] == chosen
        for unselected in (ids[0], ids[2]):
            assert not (
                task_dir / "input" / "crops" / f"crop_{unselected}.json"
            ).exists()

    _answer_and_accept(controller, "gpt", gpt_request, chosen)
    status = controller.status(BATCH_ID, DOCUMENT_ID)
    assert status.providers["gpt"] == "SUCCEEDED"
    _answer_and_accept(controller, "grok", grok_request, list(reversed(chosen)))

    document = _document(controller)
    assert [item.dispute_id for item in document.reconciled_decisions] == chosen
    assert document.challenge_rounds[0].reconciled is True
    for unselected in (ids[0], ids[2]):
        assert unselected not in {
            item.dispute_id for item in document.reconciled_decisions
        }
    assert document.substantive_dispute_ids == ids


def test_unselected_disputes_stay_unresolved_and_report_lists_the_selection(
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
    _close_challenge(controller)
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    text = paths["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert "Alpha one" in text and "Bravo two" in text
    # Unselected spans keep both readings; no winner is fabricated.
    assert "[Charlie / Charly] three" in text
    assert "Echo [five / fyve]" in text
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert [item["dispute_id"] for item in report["decisions"]] == chosen
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert report["challenge_selections"] == [
        {
            "round_index": 1,
            "selected_dispute_ids": chosen,
            "comparison_sha256": _document(controller).comparison_sha256,
        }
    ]
    assert report["production_writeback"] is False


# --- full set -----------------------------------------------------------------


def test_explicit_full_set_selection_is_valid(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    requests = {
        provider: _publish(
            controller, provider, selected_dispute_ids=list(reversed(ids))
        )
        for provider in ("gpt", "grok")
    }
    # Stored in comparison order, whatever order the operator typed.
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == ids
    for provider, request in requests.items():
        _answer_and_accept(controller, provider, request, ids)
    assert [
        item.dispute_id for item in _document(controller).reconciled_decisions
    ] == ids


def test_omitted_round_one_selection_is_recorded_for_both_providers(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    requests = {
        provider: _publish(controller, provider) for provider in ("gpt", "grok")
    }
    document = _document(controller)
    round_state = document.challenge_rounds[0]
    assert round_state.selected_dispute_ids == ids
    assert round_state.selection_comparison_sha256 == document.comparison_sha256
    assert round_state.evidence_identity == f"source:{document.source_bundle_sha256}"
    assert round_selection(document, 1) == ids
    assert [item["selected_dispute_ids"] for item in _challenge_events(controller)] == [
        ids,
        ids,
    ]
    disputes = json.loads(
        (
            _root(controller) / "challenge_round_01" / "input" / "disputes.json"
        ).read_text(encoding="utf-8")
    )
    assert [item["dispute_id"] for item in disputes["disputes"]] == ids
    for provider, request in requests.items():
        _answer_and_accept(controller, provider, request, ids)
    _close_challenge(controller)
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert report["challenge_selections"] == [
        {
            "round_index": 1,
            "selected_dispute_ids": ids,
            "comparison_sha256": _document(controller).comparison_sha256,
        }
    ]
    assert [item["dispute_id"] for item in report["decisions"]] == ids


def test_legacy_full_set_round_still_refuses_a_partial_output(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    request = _publish(controller, "gpt")
    _answer(controller, request, ids[:2])
    with pytest.raises(ReviewOutputRejected, match="do not match"):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")


# --- invalid selections -------------------------------------------------------


def _no_challenge_side_effects(
    controller: ReviewController, before: dict[str, bytes]
) -> None:
    assert _challenge_events(controller) == []
    assert _document(controller).challenge_rounds == []
    assert not (_root(controller) / "challenge_round_01").exists()
    assert _snapshot(controller.batches_dir) == before


@pytest.mark.parametrize(
    "case",
    ["empty", "duplicate", "unknown", "stale", "non_substantive", "empty_string"],
)
def test_invalid_selection_fails_before_anything_is_published(
    tmp_path: Path, case: str
) -> None:
    controller, ids, other = _setup(tmp_path)
    selection = {
        "empty": [],
        "duplicate": [ids[0], ids[0]],
        "unknown": [ids[0], "dsp_not_in_the_comparison"],
        "stale": ["dsp_58bbbcddcd7b7948"],  # an ID from a different comparison
        "non_substantive": [ids[0], other[0]],
        "empty_string": [""],
    }[case]
    before = _snapshot(controller.batches_dir)
    with pytest.raises(ReviewPolicyError):
        _publish(controller, "gpt", selected_dispute_ids=selection)
    _no_challenge_side_effects(controller, before)


def test_selection_is_refused_outside_the_challenge_stage(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    with pytest.raises(ReviewPolicyError, match="only to a challenge"):
        controller.publish(
            BATCH_ID, DOCUMENT_ID, "gpt", stage="research", selected_dispute_ids=ids[:1]
        )
    assert _challenge_events(controller) == []


def test_resolved_id_cannot_be_selected_for_a_later_round(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    controller.config = ReviewConfig(
        max_challenge_rounds=2,
        max_document_challenge_dispatches=4,
    )
    chosen = [ids[0], ids[1]]
    for provider in ("gpt", "grok"):
        request = _publish(controller, provider, selected_dispute_ids=chosen)
        _answer_and_accept(
            controller, provider, request, chosen, per_id={ids[1]: "UNRESOLVED"}
        )
    evidence = controller.batches_dir.parent / "other-scan.png"
    evidence.write_bytes(_png(8, 8))
    before_ledger = len(_ledger_lines(controller))
    with pytest.raises(ReviewPolicyError, match="already resolved"):
        _publish(
            controller,
            "gpt",
            evidence="new_crop",
            evidence_path=evidence,
            selected_dispute_ids=[ids[0], ids[2]],
        )
    assert len(_ledger_lines(controller)) == before_ledger
    assert not (_root(controller) / "challenge_round_02").exists()


# --- freezing and provider parity ---------------------------------------------


def test_selection_freezes_at_first_provider_publication(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[0], ids[2]]
    _publish(controller, "gpt", selected_dispute_ids=chosen)
    sequence = len(_ledger_lines(controller))
    with pytest.raises(ReviewPolicyError, match="frozen"):
        _publish(controller, "grok", selected_dispute_ids=[ids[0], ids[3]])
    with pytest.raises(ReviewPolicyError, match="frozen"):
        _publish(controller, "gpt", selected_dispute_ids=ids)
    assert len(_ledger_lines(controller)) == sequence
    # Omitted means "reuse the frozen subset"; the same set in any order is accepted.
    _publish(controller, "grok")
    assert [item["selected_dispute_ids"] for item in _challenge_events(controller)] == [
        chosen,
        chosen,
    ]
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == chosen
    _publish(controller, "gpt", selected_dispute_ids=list(reversed(chosen)))


def test_provider_retry_keeps_the_frozen_selection(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[2], ids[3]]
    _publish(controller, "gpt", selected_dispute_ids=chosen)
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="challenge"
    )
    with pytest.raises(ReviewPolicyError, match="frozen"):
        _publish(controller, "gpt", selected_dispute_ids=[ids[0]])
    _publish(controller, "gpt")
    events = _challenge_events(controller)
    assert [item["attempt"] for item in events] == [1, 2]
    assert [item["selected_dispute_ids"] for item in events] == [chosen, chosen]


def test_ledger_rejects_providers_with_different_selections(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[0], ids[1]]
    _publish(controller, "gpt", selected_dispute_ids=chosen)
    state = controller._store(BATCH_ID).load()
    assert state is not None
    first = _challenge_events(controller)[0]

    def event(provider: str, **changes: object) -> WorkPublishedEntry:
        payload = dict(first)
        payload.update(
            sequence=state.applied_ledger_sequence + 1,
            provider=provider,
            task_id=f"{first['task_id']}-{provider}",
            job_id=f"{first['job_id']}-{provider}",
            output_relative_path=first["output_relative_path"].replace("gpt", provider),
        )
        payload.update(changes)
        return WorkPublishedEntry.model_validate(payload)

    # Same subset: accepted.
    apply_review_entry(state, event("grok"))
    for bad in (
        [ids[0], ids[2]],
        [ids[0]],
        ids,
        None,
    ):
        with pytest.raises(ReviewIntegrityError, match="frozen"):
            apply_review_entry(state, event("grok", selected_dispute_ids=bad))
    with pytest.raises(ReviewIntegrityError, match="frozen"):
        apply_review_entry(state, event("grok", selection_comparison_sha256="0" * 64))


# --- provider output validation ------------------------------------------------


@pytest.mark.parametrize(
    "case", ["extra_unselected", "extra_unknown", "missing", "duplicate", "empty_other"]
)
def test_provider_output_must_equal_the_selected_ids(tmp_path: Path, case: str) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[0], ids[1]]
    request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    _publish(controller, "grok", selected_dispute_ids=chosen)
    answer = {
        "extra_unselected": chosen + [ids[2]],
        "extra_unknown": chosen + ["dsp_not_in_the_comparison"],
        "missing": chosen[:1],
        "duplicate": chosen + chosen[:1],
        "empty_other": [ids[2], ids[3]],
    }[case]
    _answer(controller, request, answer)
    before = len(_ledger_lines(controller))
    with pytest.raises(ReviewOutputRejected) as caught:
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert caught.value.reason == "mismatched"
    lines = _ledger_lines(controller)
    assert [item["event_type"] for item in lines[before:]] == ["result_rejected"]
    document = _document(controller)
    assert document.challenge_rounds[0].reconciled is False
    assert document.reconciled_decisions == []


def test_ledger_rejects_an_accepted_result_that_differs_from_the_selection(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[0], ids[1]]
    request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    _answer(controller, request, chosen)
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    # Hand-built ledger entry for grok with an extra ID.
    _publish(controller, "grok")
    state = controller._store(BATCH_ID).load()
    assert state is not None
    from transcription_review_agent.review.runtime import ResultAcceptedEntry

    accepted = [
        item
        for item in _ledger_lines(controller)
        if item["event_type"] == "result_accepted"
    ][-1]
    payload = dict(accepted)
    payload.update(
        sequence=state.applied_ledger_sequence + 1,
        provider="grok",
        output_relative_path=accepted["output_relative_path"].replace("gpt", "grok"),
        decisions=accepted["decisions"]
        + [{**accepted["decisions"][0], "dispute_id": ids[2]}],
    )
    with pytest.raises(ReviewIntegrityError, match="selected disputes"):
        apply_review_entry(state, ResultAcceptedEntry.model_validate(payload))


# --- replay -------------------------------------------------------------------


def test_checkpoint_deletion_and_replay_restore_the_exact_selection(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = [ids[1], ids[3]]
    gpt_request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    checkpoint = _controller_dir(controller) / "state.json"
    live = checkpoint.read_text(encoding="utf-8")

    # Frozen after only one provider published.
    checkpoint.unlink()
    resumed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    assert _document(resumed).challenge_rounds[0].selected_dispute_ids == chosen
    assert checkpoint.read_text(encoding="utf-8") == live
    grok_request = _publish(resumed, "grok")
    _answer_and_accept(resumed, "gpt", gpt_request, chosen)
    _answer_and_accept(resumed, "grok", grok_request, chosen)
    final_state = checkpoint.read_text(encoding="utf-8")

    checkpoint.unlink()
    replayed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    document = _document(replayed)
    assert document.challenge_rounds[0].selected_dispute_ids == chosen
    assert (
        document.challenge_rounds[0].selection_comparison_sha256
        == document.comparison_sha256
    )
    assert [item.dispute_id for item in document.reconciled_decisions] == chosen
    assert checkpoint.read_text(encoding="utf-8") == final_state


def test_selection_is_stored_in_comparison_order(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    _publish(controller, "gpt", selected_dispute_ids=[ids[3], ids[0], ids[2]])
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == [
        ids[0],
        ids[2],
        ids[3],
    ]
    assert _challenge_events(controller)[0]["selected_dispute_ids"] == [
        ids[0],
        ids[2],
        ids[3],
    ]


def test_unordered_event_for_a_new_round_is_rejected(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    state = controller._store(BATCH_ID).load()
    assert state is not None
    document = find_document(state, DOCUMENT_ID)
    base = {
        "event_type": "work_published",
        "sequence": state.applied_ledger_sequence + 1,
        "occurred_at": NOW.isoformat(),
        "batch_id": BATCH_ID,
        "document_id": DOCUMENT_ID,
        "provider": "gpt",
        "stage": "challenge",
        "round_index": 1,
        "attempt": 1,
        "task_id": "t1",
        "job_id": "j1",
        "output_relative_path": "challenge_round_01/output/challenge_gpt.json",
        "selection_comparison_sha256": document.comparison_sha256,
    }
    with pytest.raises(ReviewIntegrityError, match="comparison order"):
        apply_review_entry(
            state,
            WorkPublishedEntry.model_validate(
                {**base, "selected_dispute_ids": [ids[1], ids[0]]}
            ),
        )
    ok = apply_review_entry(
        state,
        WorkPublishedEntry.model_validate(
            {**base, "selected_dispute_ids": [ids[0], ids[1]]}
        ),
    )
    assert find_document(ok, DOCUMENT_ID).challenge_rounds[0].selected_dispute_ids == [
        ids[0],
        ids[1],
    ]
    # Selection integrity failures, each rejected at replay.
    for changes, message in (
        ({"selected_dispute_ids": []}, "empty"),
        ({"selected_dispute_ids": [ids[0], ids[0]]}, "duplicate"),
        ({"selected_dispute_ids": [ids[0], "dsp_missing"]}, "not current substantive"),
        (
            {"selected_dispute_ids": [ids[0]], "selection_comparison_sha256": "1" * 64},
            "bound",
        ),
        (
            {"selected_dispute_ids": [ids[0]], "selection_comparison_sha256": None},
            "bound",
        ),
        ({"selected_dispute_ids": None}, "comparison binding"),
    ):
        with pytest.raises(ReviewIntegrityError, match=message):
            apply_review_entry(
                state, WorkPublishedEntry.model_validate({**base, **changes})
            )


# --- legacy ledgers -----------------------------------------------------------


def _strip_selection_keys(controller: ReviewController) -> None:
    """Rewrite the ledger and checkpoint as an older version would have written them."""

    directory = _controller_dir(controller)
    lines = []
    for item in _ledger_lines(controller):
        item.pop("selected_dispute_ids", None)
        item.pop("selection_comparison_sha256", None)
        lines.append(json.dumps(item, separators=(",", ":"), sort_keys=True))
    (directory / "ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    for document in state["documents"]:
        for challenge in document["challenge_rounds"]:
            challenge.pop("selected_dispute_ids", None)
            challenge.pop("selection_comparison_sha256", None)
    (directory / "state.json").write_text(json.dumps(state), encoding="utf-8")


def test_legacy_ledger_without_selection_replays_as_the_full_substantive_set(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    requests = {
        provider: _publish(controller, provider) for provider in ("gpt", "grok")
    }
    for provider, request in requests.items():
        _answer_and_accept(controller, provider, request, ids)
    modern_decisions = [
        item.model_dump() for item in _document(controller).reconciled_decisions
    ]
    _strip_selection_keys(controller)
    ledger_text = (_controller_dir(controller) / "ledger.jsonl").read_text(
        encoding="utf-8"
    )
    assert "selected_dispute_ids" not in ledger_text

    # An old checkpoint (no new fields) still matches its ledger replay ...
    legacy = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    document = _document(legacy)
    assert document.challenge_rounds[0].selected_dispute_ids is None
    assert round_selection(document, 1) == ids
    assert [
        item.model_dump() for item in document.reconciled_decisions
    ] == modern_decisions
    assert document.challenge_rounds[0].reconciled is True

    # ... and so does a replay with no checkpoint at all.
    (_controller_dir(controller) / "state.json").unlink()
    replayed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    document = _document(replayed)
    assert document.challenge_rounds[0].selected_dispute_ids is None
    assert round_selection(document, 1) == ids
    assert [item.dispute_id for item in document.reconciled_decisions] == ids
    # The ledger on disk is untouched by replay.
    assert (_controller_dir(controller) / "ledger.jsonl").read_text(
        encoding="utf-8"
    ) == ledger_text
    # Legacy reports do not gain a selection key.
    _close_challenge(replayed)
    paths = replayed.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert "challenge_selections" not in report


def test_legacy_open_round_still_requires_the_full_set_and_rejects_a_new_subset(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    _publish(controller, "gpt")
    _strip_selection_keys(controller)
    resumed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    with pytest.raises(ReviewPolicyError, match="frozen"):
        _publish(resumed, "grok", selected_dispute_ids=ids[:2])
    request = _publish(resumed, "grok", selected_dispute_ids=list(reversed(ids)))
    assert _document(resumed).challenge_rounds[0].selected_dispute_ids is None
    _answer(resumed, request, ids[:2])
    with pytest.raises(ReviewOutputRejected):
        resumed.accept(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")


# --- later rounds accumulate ---------------------------------------------------


def test_second_round_keeps_earlier_decisions_and_selects_only_unresolved_ids(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    controller.config = ReviewConfig(
        max_challenge_rounds=2,
        max_document_challenge_dispatches=4,
    )
    chosen = [ids[0], ids[1]]
    for provider in ("gpt", "grok"):
        request = _publish(controller, provider, selected_dispute_ids=chosen)
        _answer_and_accept(
            controller, provider, request, chosen, per_id={ids[1]: "UNRESOLVED"}
        )
    first = _document(controller).reconciled_decisions
    assert {item.dispute_id: item.decision for item in first} == {
        ids[0]: "CANDIDATE_X",
        ids[1]: "UNRESOLVED",
    }
    evidence = tmp_path / "other-scan.png"
    evidence.write_bytes(_png(8, 8))
    second_chosen = [ids[1], ids[3]]
    requests = {
        provider: _publish(
            controller,
            provider,
            evidence="new_crop",
            evidence_path=evidence,
            selected_dispute_ids=second_chosen,
        )
        for provider in ("gpt", "grok")
    }
    document = _document(controller)
    assert document.challenge_rounds[1].selected_dispute_ids == second_chosen
    assert document.challenge_rounds[0].selected_dispute_ids == chosen
    for provider, request in requests.items():
        _answer_and_accept(
            controller, provider, request, second_chosen, decision="CANDIDATE_Y"
        )
    final = {
        item.dispute_id: item.decision
        for item in _document(controller).reconciled_decisions
    }
    # Round-one decision for ids[0] survives; round two decides its own IDs only.
    assert final == {
        ids[0]: "CANDIDATE_X",
        ids[1]: "CANDIDATE_Y",
        ids[3]: "CANDIDATE_Y",
    }
    assert ids[2] not in final


# --- hash binding ---------------------------------------------------------------


def test_comparison_tampering_after_recording_blocks_publication(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    path = _root(controller) / "comparison" / "comparison.json"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"Alpha one", b"Alpha ONE"))
    with pytest.raises(ReviewIntegrityError, match="comparison file changed"):
        _publish(controller, "gpt", selected_dispute_ids=ids[:2])
    assert _challenge_events(controller) == []
    path.write_bytes(original)
    _publish(controller, "gpt", selected_dispute_ids=ids[:2])


def test_source_page_tampering_blocks_publication_and_acceptance(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = ids[:2]
    request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    _answer(controller, request, chosen)
    page = _root(controller) / "input" / "page_001.png"
    saved = page.read_bytes()
    page.write_bytes(saved + b"\x00")
    with pytest.raises(ReviewIntegrityError):
        _publish(controller, "grok", selected_dispute_ids=chosen)
    with pytest.raises(ReviewIntegrityError):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    page.write_bytes(saved)
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")


def test_selection_round_is_bound_to_the_recorded_comparison_hash(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    _publish(controller, "gpt", selected_dispute_ids=ids[:2])
    path = _root(controller) / "comparison" / "comparison.json"
    document = _document(controller)
    assert document.challenge_rounds[0].selection_comparison_sha256 == sha256_bytes(
        path.read_bytes()
    )
    assert (
        document.challenge_rounds[0].selection_comparison_sha256
        == document.comparison_sha256
    )


def test_provider_output_for_a_selection_is_still_bound_to_the_source_bundle(
    tmp_path: Path,
) -> None:
    controller, ids, _other = _setup(tmp_path)
    chosen = ids[:2]
    request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    _answer(controller, request, chosen)
    output = (
        _root(controller)
        / WorkRequest.model_validate_json(
            request.read_text(encoding="utf-8")
        ).output_relative_path
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["source_bundle_sha256"] = "2" * 64
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReviewOutputRejected) as caught:
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    assert caught.value.reason == "stale"


# --- sealed data and the CLI ----------------------------------------------------


def test_selection_does_not_mutate_sealed_bundles(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    before = _snapshot(controller.batches_dir)
    chosen = ids[:2]
    for provider in ("gpt", "grok"):
        _answer_and_accept(
            controller,
            provider,
            _publish(controller, provider, selected_dispute_ids=chosen),
            chosen,
        )
    _close_challenge(controller)
    controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert _snapshot(controller.batches_dir) == before


def test_cli_dispute_id_selects_a_subset_for_both_providers(tmp_path: Path) -> None:
    controller, ids, _other = _setup(tmp_path)
    args = [
        "prepare",
        "--batches-dir",
        str(tmp_path / "batches"),
        "--reviews-dir",
        str(tmp_path / "reviews"),
        "--batch-id",
        BATCH_ID,
        "--document-id",
        str(DOCUMENT_ID),
        "--stage",
        "challenge",
        "--dispute-id",
        ids[2],
        "--dispute-id",
        ids[0],
    ]
    assert main(args) == 0
    events = _challenge_events(controller)
    assert [item["provider"] for item in events] == ["gpt", "grok"]
    assert [item["selected_dispute_ids"] for item in events] == [[ids[0], ids[2]]] * 2
