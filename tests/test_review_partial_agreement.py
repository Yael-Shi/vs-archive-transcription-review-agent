"""Partial agreement inside one challenged dispute.

A shared span is adopted only when both providers support it with a bound
visual assessment. The tests build those decisions in memory. The document
108 preview reads local files and does not write them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from test_review_a3_controller_integration import _deliver, _requests
from test_review_challenge_groups import _document, _setup
from test_review_passage_alignment import (
    CINEMA_GPT,
    CINEMA_GROK,
    VINEYARD_GPT,
    VINEYARD_GROK,
    _compare,
    _output,
    _page,
    _region,
)
from test_review_workflow import BATCH_ID, DOCUMENT_ID
from transcription_review_agent.review.adjudicate import (
    crop_scope,
    reconcile_challenge,
)
from transcription_review_agent.review.assemble import render_transcription
from transcription_review_agent.review.formatting import render_aligned_span
from transcription_review_agent.review.models import (
    ChallengeDecision,
    ChallengeOutput,
    Dispute,
    EvidenceReference,
    FirstPassOutput,
    ReconciledDecision,
)
from transcription_review_agent.review.runtime import ReviewStore

PILOT = Path("/mnt/c/Users/User/vs-archive-transcription-review-agent/desktop_pilot")
CINEMA_ID = "dsp_cinema"


def _dispute(
    gpt_text: str,
    grok_text: str,
    *,
    dispute_id: str = CINEMA_ID,
    page_id: str = "page_001",
    category: str = "WORDING",
) -> Dispute:
    return Dispute(
        dispute_id=dispute_id,
        category=category,  # type: ignore[arg-type]
        page_id=page_id,
        gpt_text=gpt_text,
        grok_text=grok_text,
        substantive=True,
    )


def _decision(
    dispute: Dispute,
    label: str,
    assessment: str = "SUPPORTS_DECISION",
    *,
    third: str | None = None,
) -> ChallengeDecision:
    return ChallengeDecision(
        dispute_id=dispute.dispute_id,
        decision=label,  # type: ignore[arg-type]
        third_reading=third,
        visual_evidence="The examined window supports this decision.",
        evidence_ref=EvidenceReference(
            kind="CROP",
            evidence_id=f"crop_{dispute.dispute_id}",
            page_id=dispute.page_id,
        ),
        evidence_assessment=assessment,  # type: ignore[arg-type]
    )


def _reconcile(
    dispute: Dispute,
    gpt: ChallengeDecision,
    grok: ChallengeDecision,
) -> ReconciledDecision:
    scope = crop_scope(
        dispute_id=dispute.dispute_id,
        page_id=dispute.page_id,
        crop_id=f"crop_{dispute.dispute_id}",
        window_kind="TARGETED",
    )
    return reconcile_challenge(dispute, gpt, grok, scope)


def _show(dispute: Dispute, decision: ReconciledDecision) -> str:
    return render_aligned_span(
        dispute.gpt_text,
        dispute.grok_text,
        [dispute],
        {dispute.dispute_id: decision},
    )


def test_shared_supported_span_keeps_the_trailing_omission() -> None:
    dispute = _dispute(CINEMA_GPT, CINEMA_GROK)
    third = (
        "gens avaient provoqué une petite explosion dans un cinéma sans causer "
        "des dégâts et sans atteindre de"
    )
    first = _reconcile(
        dispute,
        _decision(dispute, "THIRD_READING", third=third),
        _decision(dispute, "CANDIDATE_Y"),
    )
    again = _reconcile(
        dispute,
        _decision(dispute, "THIRD_READING", third=third),
        _decision(dispute, "CANDIDATE_Y"),
    )
    assert again == first
    assert first.decision == "UNRESOLVED"
    assert first.result_class == "UNRESOLVED"
    assert first.selected_reading is None
    assert first.partial_agreements is not None
    adopted = [span for span in first.partial_agreements if span.role == "ADOPTED"]
    unresolved = [
        span for span in first.partial_agreements if span.role == "UNRESOLVED"
    ]
    assert "".join(span.gpt_text for span in adopted) == (
        "gens avaient provoqué une petite explosion dans un cinéma sans causer "
        "des dégâts et sans atteindre"
    )
    assert [(span.gpt_text, span.grok_text) for span in unresolved] == [(" de", "")]
    span = adopted[0]
    assert span.dispute_id == dispute.dispute_id
    assert span.gpt_decision == "THIRD_READING"
    assert span.grok_decision == "CANDIDATE_Y"
    assert span.gpt_evidence_ref == first.gpt_evidence_ref
    assert span.grok_evidence_ref == first.grok_evidence_ref
    assert span.gpt_evidence_assessment == "SUPPORTS_DECISION"
    assert third[span.gpt_start : span.gpt_end] == span.gpt_text
    assert CINEMA_GROK[span.grok_start : span.grok_end] == span.grok_text
    assert first.gpt_third_reading == third
    assert first.candidate_y_text == CINEMA_GROK
    assert _show(dispute, first) == (
        "gens avaient provoqué une petite explosion dans un cinéma sans causer "
        "des dégâts et sans atteindre[ de / ∅]"
    )
    assert "cuisine" not in _show(dispute, first)


def test_same_wording_without_visual_support_adopts_nothing() -> None:
    dispute = _dispute(CINEMA_GPT, CINEMA_GROK)
    third = CINEMA_GROK
    result = _reconcile(
        dispute,
        _decision(dispute, "THIRD_READING", "INSUFFICIENT", third=third),
        _decision(dispute, "THIRD_READING", "INSUFFICIENT", third=third),
    )
    assert result.partial_agreements is None
    assert result.selected_reading is None
    assert result.basis == "INSUFFICIENT_EVIDENCE"
    assert "[cuisine / cinéma]" in _show(dispute, result)
    assert "un cinéma sans" not in _show(dispute, result)


def test_missing_evidence_binding_does_not_adopt_shared_words() -> None:
    dispute = _dispute("Jean Dupont", "Jean Durand", category="NAME_OR_ENTITY")
    legacy = ReconciledDecision(
        dispute_id=dispute.dispute_id,
        decision="UNRESOLVED",
        agreed=False,
        gpt_decision="CANDIDATE_X",
        grok_decision="CANDIDATE_Y",
    )
    assert legacy.partial_agreements is None
    assert "partial_agreements" not in legacy.model_dump(mode="json")
    shown = _show(dispute, legacy)
    assert "Dupont" in shown and "Durand" in shown
    assert shown != "Jean Dupont" and shown != "Jean Durand"


def test_conflicting_evidence_assessments_adopt_nothing() -> None:
    dispute = _dispute(CINEMA_GPT, CINEMA_GROK)
    third = (
        "gens avaient provoqué une petite explosion dans un cinéma sans causer "
        "des dégâts et sans atteindre de"
    )
    result = _reconcile(
        dispute,
        _decision(dispute, "THIRD_READING", third=third),
        _decision(dispute, "CANDIDATE_Y", "INSUFFICIENT"),
    )
    assert result.partial_agreements is None
    assert result.basis == "PROVIDER_DISAGREEMENT"
    assert "[cuisine / cinéma]" in _show(dispute, result)


def test_third_reading_and_candidate_share_one_supported_name() -> None:
    dispute = _dispute(
        "Jean Dupont resta", "Jean Durand resta", category="NAME_OR_ENTITY"
    )
    result = _reconcile(
        dispute,
        _decision(dispute, "THIRD_READING", third="Jean Durand resta"),
        _decision(dispute, "CANDIDATE_Y"),
    )
    assert result.decision == "UNRESOLVED"
    assert _show(dispute, result) == "Jean Durand resta"
    assert result.partial_agreements is not None
    assert [span.role for span in result.partial_agreements] == ["ADOPTED"]


def test_names_numbers_marks_and_punctuation_stay_in_their_spans() -> None:
    pairs = {
        "name": ("Jean Dupont", "Jean Durand", "Jean [Dupont / Durand]"),
        "number": ("le 14 mai", "le 15 mai", "le [14 / 15] mai"),
        "cancellation": (
            "fin [biffé : mot] suite",
            "fin [illisible] suite",
            "fin [A: [biffé : mot] / B: [illisible]] suite",
        ),
        "insertion": (
            "avant [inserted: ici] après",
            "avant après",
            "avant [A: [inserted: ici]  / B: ∅]après",
        ),
        "punctuation": ("mai ,", "mai .", "mai [, / .]"),
        "omission": ("sans de", "sans", "sans[ de / ∅]"),
    }
    for gpt_text, grok_text, expected in pairs.values():
        dispute = _dispute(gpt_text, grok_text)
        result = _reconcile(
            dispute,
            _decision(dispute, "CANDIDATE_X"),
            _decision(dispute, "CANDIDATE_Y"),
        )
        assert _show(dispute, result) == expected
        assert result.decision == "UNRESOLVED"
        adopted = "".join(
            span.gpt_text
            for span in result.partial_agreements or []
            if span.role == "ADOPTED"
        )
        assert "Dupont" not in adopted or "Dupont" in grok_text
        assert "14" not in adopted or "14" in grok_text


def test_reordered_text_and_unrelated_names_are_not_partially_adopted() -> None:
    reordered = _dispute("un cinéma ouvert", "cinéma un ouvert")
    moved = _reconcile(
        reordered,
        _decision(reordered, "CANDIDATE_X"),
        _decision(reordered, "CANDIDATE_Y"),
    )
    assert moved.partial_agreements is None
    assert _show(reordered, moved) == "[un cinéma ouvert / cinéma un ouvert]"

    names = _dispute(VINEYARD_GPT, VINEYARD_GROK, page_id="page_005")
    listed = _reconcile(
        names, _decision(names, "CANDIDATE_X"), _decision(names, "CANDIDATE_Y")
    )
    shown = _show(names, listed)
    assert "[Bey, / -]" not in shown
    assert "Hasson" not in "".join(
        span.gpt_text
        for span in listed.partial_agreements or []
        if span.role == "ADOPTED"
    )


def test_partial_agreement_does_not_cross_pages() -> None:
    cinema_left = _page("page_001", [_region("a", 1, CINEMA_GPT)])
    cinema_right = _page("page_001", [_region("b", 1, CINEMA_GROK)])
    vineyard_left = _page("page_005", [_region("c", 1, VINEYARD_GPT)])
    vineyard_right = _page("page_005", [_region("d", 1, VINEYARD_GROK)])
    comparison = _compare([cinema_left, vineyard_left], [cinema_right, vineyard_right])
    dispute = next(item for item in comparison.disputes if item.page_id == "page_001")
    third = (
        "gens avaient provoqué une petite explosion dans un cinéma sans causer "
        "des dégâts et sans atteindre de"
    )
    decision = _reconcile(
        dispute,
        _decision(dispute, "THIRD_READING", third=third),
        _decision(dispute, "CANDIDATE_Y"),
    )
    text = render_transcription(
        page_ids=["page_001", "page_005"],
        gpt=_output([cinema_left, vineyard_left], "gpt"),
        grok=_output([cinema_right, vineyard_right], "grok"),
        comparison=comparison,
        decisions=[decision],
    )
    page_one, page_five = text.split("@@ page_005\n", maxsplit=1)
    assert "un cinéma sans causer" in page_one
    assert "[cuisine / cinéma]" not in page_one
    assert "[ de / ∅]" in page_one
    assert "cinéma" not in page_five
    assert "Hasson" in page_five
    assert "[Bey, / -]" not in page_five


@pytest.mark.parametrize("weak", ["INSUFFICIENT", "UNREADABLE"])
@pytest.mark.parametrize(
    "which", ["gpt_evidence_assessment", "grok_evidence_assessment"]
)
def test_parent_assessment_must_support_every_adopted_span(
    weak: str, which: str
) -> None:
    dispute = _dispute(CINEMA_GPT, CINEMA_GROK)
    third = (
        "gens avaient provoqué une petite explosion dans un cinéma sans causer "
        "des dégâts et sans atteindre de"
    )
    record = _reconcile(
        dispute,
        _decision(dispute, "THIRD_READING", third=third),
        _decision(dispute, "CANDIDATE_Y"),
    )
    assert record.partial_agreements
    payload = record.model_dump(mode="json")
    assert payload["gpt_evidence_assessment"] == "SUPPORTS_DECISION"
    assert payload["grok_evidence_assessment"] == "SUPPORTS_DECISION"
    assert ReconciledDecision.model_validate(payload) == record
    payload[which] = weak
    with pytest.raises(ValidationError, match="partial agreement requires both"):
        ReconciledDecision.model_validate(payload)


def test_legacy_reconciled_record_round_trips_without_partial_agreement() -> None:
    legacy = {
        "dispute_id": "dsp_old",
        "decision": "CANDIDATE_X",
        "agreed": True,
    }
    record = ReconciledDecision.model_validate(legacy)
    assert record.partial_agreements is None
    assert "partial_agreements" not in record.model_dump(mode="json")
    again = ReconciledDecision.model_validate_json(record.model_dump_json())
    assert again == record


def test_repeated_accept_keeps_one_partial_agreement(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    dispute = next(
        item for item in comparison.disputes if item.dispute_id == names["word_r1"]
    )
    requests = _requests(controller, [dispute.dispute_id])
    _deliver(
        controller,
        requests["gpt"],
        choice="THIRD_READING",
        third=dispute.grok_text + " extra",
    )
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    _deliver(controller, requests["grok"], choice="CANDIDATE_Y")
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    decision = _document(controller).reconciled_decisions[0]
    assert decision.partial_agreements is not None
    assert _show(dispute, decision) == f"{dispute.grok_text}[ extra / ∅]"
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    reloaded = ReviewStore(controller.reviews_dir, BATCH_ID).load()
    assert reloaded is not None
    assert reloaded.documents[0].reconciled_decisions == [decision]
    ledger = ReviewStore(controller.reviews_dir, BATCH_ID).ledger_path.read_text(
        encoding="utf-8"
    )
    assert ledger.count('"event_type":"challenge_reconciled"') == 1
    stored = json.loads(
        next(line for line in ledger.splitlines() if '"challenge_reconciled"' in line)
    )
    stripped = json.loads(json.dumps(stored))
    for item in stripped["decisions"]:
        item.pop("partial_agreements", None)
    from transcription_review_agent.review.runtime import parse_review_entry

    legacy_entry = parse_review_entry(stripped)
    assert legacy_entry.decisions[0].partial_agreements is None  # type: ignore[attr-defined]


def test_document_108_preview_adopts_cinema_without_writing() -> None:
    paths = [
        PILOT / "first_pass_gpt.json",
        PILOT / "transcription_grok.json",
        PILOT / "challenge_round_01" / "output" / "challenge_gpt.json",
        PILOT / "challenge_grok.json",
        PILOT / "doc108_final" / "proposed_transcription.txt",
        PILOT / "doc108_final" / "review_report.json",
        PILOT / "doc108_final" / "review_manifest.json",
        PILOT / "doc108_final" / "review_summary.md",
    ]
    if any(not path.is_file() for path in paths):
        pytest.skip("document 108 review files are not on this machine")
    before = {path: path.read_bytes() for path in paths}
    gpt = FirstPassOutput.model_validate_json(before[paths[0]])
    grok = FirstPassOutput.model_validate_json(before[paths[1]])
    gpt_challenge = ChallengeOutput.model_validate_json(before[paths[2]])
    grok_challenge = ChallengeOutput.model_validate_json(before[paths[3]])
    if gpt.document_id != 108 or grok.document_id != 108:
        pytest.skip("provider files are not document 108")
    from transcription_review_agent.review.compare import compare_transcriptions

    comparison = compare_transcriptions(
        gpt, grok, gpt_sha256="ab" * 32, grok_sha256="ab" * 32
    )
    dispute = next(
        item
        for item in comparison.disputes
        if item.gpt_text == CINEMA_GPT and item.grok_text == CINEMA_GROK
    )
    left = next(
        item
        for item in gpt_challenge.decisions
        if item.dispute_id == dispute.dispute_id
    )
    right = next(
        item
        for item in grok_challenge.decisions
        if item.dispute_id == dispute.dispute_id
    )
    ref = left.evidence_ref
    assert ref is not None and right.evidence_ref == ref
    scope = crop_scope(
        dispute_id=dispute.dispute_id,
        page_id=dispute.page_id,
        crop_id=ref.evidence_id,
        window_kind="FULL_PAGE_FALLBACK",
    )
    decision = reconcile_challenge(dispute, left, right, scope)
    preview = render_transcription(
        page_ids=[page.page_id for page in gpt.pages],
        gpt=gpt,
        grok=grok,
        comparison=comparison,
        decisions=[decision],
    )
    finalized = before[paths[4]].decode("utf-8")
    cinema = next(line for line in preview.splitlines() if "atteindre" in line)
    assert cinema == (
        "gens avaient provoqué une petite explosion dans un cinéma sans causer "
        "des dégâts et sans atteindre[ de / ∅]"
    )
    assert "[cuisine / cinéma]" in finalized
    assert "[cuisine / cinéma]" not in cinema
    assert {path: path.read_bytes() for path in paths} == before
