"""A5: reversible merged-text display and structured quality.

Pure fixtures only. No database, network, provider API, archive write, or
operational review run. These tests are not executed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.review.assemble import (
    build_report,
    classify_quality,
    render_transcription,
)
from transcription_review_agent.review.compare import compare_transcriptions
from transcription_review_agent.review.formatting import (
    ILLEGIBLE_DISPLAY,
    MISSING_SIDE,
    content_is_unresolved,
    format_unresolved_alternatives,
    render_aligned_span,
)
from transcription_review_agent.review.models import (
    AlternativeReading,
    ChallengeClosureRecord,
    ComparisonFile,
    CoverageReport,
    Dispute,
    DisputeCategory,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    ReconciledDecision,
    RegionAlignment,
    ReviewReport,
    TextRegion,
    UnreadableSpan,
    UnresolvedDispute,
)
from transcription_review_agent.review.spans import same_span

SHA = "ab" * 32
BATCH = "batch-000001"


def _coverage(region_ids: list[str], *, unresolved: bool = False) -> CoverageReport:
    return CoverageReport(
        coverage_complete=True,
        unresolved_text_present=unresolved,
        examined_region_ids=region_ids,
    )


def _page(
    text: str,
    regions: list[TextRegion] | None = None,
    *,
    page_id: str = "page_001",
    unreadable: list[UnreadableSpan] | None = None,
    alternatives: list[AlternativeReading] | None = None,
    unresolved: bool = False,
) -> PageTranscription:
    if regions is None:
        regions = [
            TextRegion(
                region_id="r1",
                order=1,
                lines=[LineReading(line_id="r1-l1", text=text)],
            )
        ]
    return PageTranscription(
        page_id=page_id,
        reading_order=1,
        diplomatic_transcription=text,
        regions=regions,
        unreadable_spans=[] if unreadable is None else unreadable,
        alternative_readings=[] if alternatives is None else alternatives,
        attention_items=[],
        coverage=_coverage(
            [region.region_id for region in regions], unresolved=unresolved
        ),
        cancellations=[],
        insertions=[],
        marginal_notes=[],
        inaccessible_or_ambiguous=[],
    )


def _region(region_id: str, order: int, text: str) -> TextRegion:
    return TextRegion(
        region_id=region_id,
        order=order,
        lines=[LineReading(line_id=f"{region_id}-l1", text=text)],
    )


def _output(page: PageTranscription, provider: str = "gpt") -> FirstPassOutput:
    interface = "GPT_WORK" if provider == "gpt" else "GROK"
    return FirstPassOutput(
        provider=provider,  # type: ignore[arg-type]
        provider_interface=interface,  # type: ignore[arg-type]
        task_id=f"task-{provider}",
        job_id=f"job-{provider}",
        batch_id=BATCH,
        document_id=101,
        attempt=1,
        source_bundle_sha256=SHA,
        pages=[page],
    )


def _dispute(
    category: DisputeCategory,
    gpt_text: str,
    grok_text: str,
    *,
    dispute_id: str,
    substantive: bool = True,
    region_ids: list[str] | None = None,
) -> Dispute:
    ids = ["r1"] if region_ids is None else region_ids
    return Dispute(
        dispute_id=dispute_id,
        category=category,
        page_id="page_001",
        region_id=ids[0] if ids else None,
        gpt_region_ids=list(ids),
        grok_region_ids=list(ids),
        gpt_text=gpt_text,
        grok_text=grok_text,
        substantive=substantive,
    )


def _alignment(
    gpt_text: str,
    grok_text: str,
    *,
    alignment_id: str = "aln-1",
    sequence: int = 0,
    region_ids: list[str] | None = None,
) -> RegionAlignment:
    ids = ["r1"] if region_ids is None else region_ids
    return RegionAlignment(
        alignment_id=alignment_id,
        page_id="page_001",
        kind="matched",
        gpt_region_ids=list(ids),
        grok_region_ids=list(ids),
        gpt_text=gpt_text,
        grok_text=grok_text,
        sequence=sequence,
    )


def _comparison(
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    disputes: list[Dispute],
    alignments: list[RegionAlignment],
    *,
    unresolved_pages: list[str] | None = None,
) -> ComparisonFile:
    return ComparisonFile(
        batch_id=BATCH,
        document_id=101,
        source_bundle_sha256=SHA,
        gpt_output_sha256=SHA,
        grok_output_sha256=SHA,
        page_originals=[],
        alignments=alignments,
        disputes=disputes,
        unresolved_text_page_ids=[] if unresolved_pages is None else unresolved_pages,
    )


def _with_originals(
    comparison: ComparisonFile, gpt: FirstPassOutput, grok: FirstPassOutput
) -> ComparisonFile:
    from transcription_review_agent.review.models import PageOriginals

    return comparison.model_copy(
        update={
            "page_originals": [
                PageOriginals(
                    page_id="page_001",
                    gpt_diplomatic_transcription=gpt.pages[0].diplomatic_transcription,
                    grok_diplomatic_transcription=grok.pages[
                        0
                    ].diplomatic_transcription,
                )
            ]
        }
    )


def _supported(
    dispute: Dispute,
    code: str,
    selected: str,
    *,
    third: str | None = None,
) -> ReconciledDecision:
    return ReconciledDecision(
        dispute_id=dispute.dispute_id,
        decision=code,  # type: ignore[arg-type]
        third_reading=third,
        agreed=True,
        gpt_decision=code,  # type: ignore[arg-type]
        grok_decision=code,  # type: ignore[arg-type]
        gpt_visual_evidence="The crop shows this ink.",
        grok_visual_evidence="The crop shows this ink.",
        remaining_uncertainty=True,
        result_class="AUTOMATICALLY_SUPPORTED",
        basis="DUAL_MODEL_VISUAL_SUPPORT",
        selected_reading=selected,
        candidate_x_text=dispute.gpt_text,
        candidate_y_text=dispute.grok_text,
        gpt_third_reading=third,
        grok_third_reading=third,
        independently_verified=False,
        eligible_for_writer_memory=False,
        eligible_for_publication=False,
    )


def _unresolved(
    dispute: Dispute,
    basis: str = "INSUFFICIENT_EVIDENCE",
) -> ReconciledDecision:
    return ReconciledDecision(
        dispute_id=dispute.dispute_id,
        decision="UNRESOLVED",
        agreed=False,
        gpt_decision="CANDIDATE_X",
        grok_decision="CANDIDATE_Y",
        gpt_visual_evidence="The crop does not settle it.",
        grok_visual_evidence="The crop does not settle it.",
        remaining_uncertainty=True,
        result_class="UNRESOLVED",
        basis=basis,  # type: ignore[arg-type]
        selected_reading=None,
        candidate_x_text=dispute.gpt_text,
        candidate_y_text=dispute.grok_text,
        independently_verified=False,
        eligible_for_writer_memory=False,
        eligible_for_publication=False,
    )


def _closure(
    disputes: list[Dispute],
    *,
    mode: str = "NO_CHALLENGE_REQUESTED",
    status: str = "NOT_CHALLENGED",
    round_count: int = 0,
    failed: list[int] | None = None,
) -> ChallengeClosureRecord:
    return ChallengeClosureRecord(
        mode=mode,  # type: ignore[arg-type]
        reason="Operator closed the stage for this local proposal.",
        comparison_sha256=SHA,
        round_count=round_count,
        failed_round_indexes=[] if failed is None else failed,
        unresolved=[
            UnresolvedDispute(
                dispute_id=item.dispute_id,
                status=status,  # type: ignore[arg-type]
                selected_in_rounds=[1]
                if status in {"SELECTED_UNRESOLVED", "PROVIDER_FAILURE"}
                else [],
            )
            for item in disputes
        ],
    )


def _render(
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    comparison: ComparisonFile,
    decisions: list[ReconciledDecision],
) -> str:
    return render_transcription(
        page_ids=["page_001"],
        gpt=gpt,
        grok=grok,
        comparison=comparison,
        decisions=decisions,
    )


def _quality(
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    comparison: ComparisonFile,
    decisions: list[ReconciledDecision],
    closure: ChallengeClosureRecord | None = None,
) -> str:
    return classify_quality(
        single_provider=False,
        gpt=gpt,
        grok=grok,
        comparison=comparison,
        decisions=decisions,
        challenge_closure=closure,
    ).value


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --- exact selected readings --------------------------------------------------


def test_supported_x_y_and_third_readings_are_exact() -> None:
    gpt = _output(_page("Alpha, "))
    grok = _output(_page("alpha."), "grok")
    dispute = _dispute(
        DisputeCategory.WORDING, "Alpha, ", "alpha.", dispute_id="dsp-word"
    )
    comparison = _with_originals(
        _comparison(gpt, grok, [dispute], [_alignment("Alpha, ", "alpha.")]),
        gpt,
        grok,
    )
    originals = comparison.model_dump()
    for code, selected, absent in (
        ("CANDIDATE_X", "Alpha, ", "alpha."),
        ("CANDIDATE_Y", "alpha.", "Alpha, "),
    ):
        decision = _supported(dispute, code, selected)
        text = _render(gpt, grok, comparison, [decision])
        assert text == f"@@ page_001\n{selected}\n"
        assert absent not in text
        assert decision.selected_reading == selected
        assert decision.independently_verified is False
        assert decision.eligible_for_publication is False
        assert decision.eligible_for_writer_memory is False
    third = "Alpha — kept"
    decision = _supported(dispute, "THIRD_READING", third, third=third)
    text = _render(gpt, grok, comparison, [decision])
    assert text == f"@@ page_001\n{third}\n"
    assert "Alpha, " not in text
    assert comparison.model_dump() == originals
    assert comparison.disputes[0].gpt_text == "Alpha, "
    assert comparison.disputes[0].grok_text == "alpha."


def test_supported_reading_must_be_the_recorded_a3_string() -> None:
    gpt = _output(_page("Alpha"))
    grok = _output(_page("Alfa"), "grok")
    dispute = _dispute(DisputeCategory.WORDING, "Alpha", "Alfa", dispute_id="dsp-word")
    comparison = _comparison(gpt, grok, [dispute], [_alignment("Alpha", "Alfa")])
    decision = _supported(dispute, "CANDIDATE_X", "alpha")
    text = _render(gpt, grok, comparison, [decision])
    assert "[Alpha / Alfa]" in text
    assert _quality(gpt, grok, comparison, [decision]) == "UNRESOLVED_CONTENT"
    payload = decision.model_dump()
    payload["eligible_for_publication"] = True
    with pytest.raises(ValidationError):
        ReconciledDecision.model_validate(payload)


def test_legacy_label_still_selects_the_aligned_side() -> None:
    gpt = _output(_page("Ionnes"))
    grok = _output(_page("Iannes"), "grok")
    dispute = _dispute(
        DisputeCategory.NAME_OR_ENTITY, "Ionnes", "Iannes", dispute_id="dsp-name"
    )
    comparison = _comparison(gpt, grok, [dispute], [_alignment("Ionnes", "Iannes")])
    decision = ReconciledDecision(
        dispute_id=dispute.dispute_id,
        decision="CANDIDATE_X",
        agreed=True,
        gpt_visual_evidence="historical label",
        grok_visual_evidence="historical label differs",
        remaining_uncertainty=True,
    )
    text = _render(gpt, grok, comparison, [decision])
    assert text == "@@ page_001\nIonnes\n"
    assert decision.result_class is None
    assert _quality(gpt, grok, comparison, [decision]) == "RESOLVED_WITH_WARNINGS"


# --- ordinary, missing, multiline, cancellation, illegible --------------------


def test_ordinary_unresolved_pair_and_exact_punctuation() -> None:
    assert format_unresolved_alternatives("Alpha, ", "alpha.") == "[Alpha,  / alpha.]"
    assert format_unresolved_alternatives("a / b", "c") == "[a / b / c]"
    gpt = _output(_page("a / b"))
    grok = _output(_page("c"), "grok")
    dispute = _dispute(DisputeCategory.WORDING, "a / b", "c", dispute_id="dsp-slash")
    comparison = _with_originals(
        _comparison(gpt, grok, [dispute], [_alignment("a / b", "c")]),
        gpt,
        grok,
    )
    assert "[a / b / c]" in _render(gpt, grok, comparison, [])
    assert comparison.disputes[0].gpt_text == "a / b"
    assert comparison.page_originals[0].gpt_diplomatic_transcription == "a / b"


def test_missing_side_uses_display_empty_and_keeps_the_empty_string() -> None:
    assert MISSING_SIDE == "∅"
    assert format_unresolved_alternatives("secret", "") == "[secret / ∅]"
    assert format_unresolved_alternatives("", "secret") == "[∅ / secret]"
    gpt = _output(_page("secret"))
    grok = _output(_page(""), "grok")
    dispute = _dispute(
        DisputeCategory.MISSING_CONTENT, "secret", "", dispute_id="dsp-miss"
    )
    comparison = _with_originals(
        _comparison(gpt, grok, [dispute], [_alignment("secret", "")]),
        gpt,
        grok,
    )
    before = comparison.model_dump()
    assert "[secret / ∅]" in _render(gpt, grok, comparison, [])
    assert comparison.model_dump() == before
    assert comparison.disputes[0].grok_text == ""
    assert "∅" not in comparison.disputes[0].grok_text


def test_multiline_alternatives_keep_line_breaks_and_recover_from_records() -> None:
    left = "kept line\nstill kept"
    right = "kept line\naltered line"
    assert format_unresolved_alternatives(left, right) == (
        "[A:\nkept line\nstill kept\n/\nB:\nkept line\naltered line]"
    )
    assert "kept line still kept" not in format_unresolved_alternatives(left, right)
    gpt_page = _page(
        left,
        [_region("r1", 1, left)],
    )
    grok_page = _page(right, [_region("r1", 1, right)])
    gpt = _output(gpt_page)
    grok = _output(grok_page, "grok")
    dispute = _dispute(DisputeCategory.WORDING, left, right, dispute_id="dsp-lines")
    comparison = _with_originals(
        _comparison(gpt, grok, [dispute], [_alignment(left, right)]),
        gpt,
        grok,
    )
    text = _render(gpt, grok, comparison, [])
    assert "[A:" in text and "\n/\nB:\n" in text
    assert text.splitlines().count("kept line") == 2
    assert "still kept" in text.splitlines()
    assert "altered line]" in text.splitlines()
    recovered_gpt = comparison.page_originals[0].gpt_diplomatic_transcription
    recovered_grok = comparison.page_originals[0].grok_diplomatic_transcription
    assert recovered_gpt == left
    assert recovered_grok == right
    assert comparison.disputes[0].gpt_text.split("\n") == ["kept line", "still kept"]


def test_cancellations_and_insertions_keep_provider_text() -> None:
    left = "united the ⟦ashkenazi⟧ sephardic"
    right = "united the [deleted: [illegible]] sephardic"
    shown = format_unresolved_alternatives(left, right)
    assert shown == f"[{left} / {right}]"
    assert "⟦illegible⟧" not in shown
    gpt = _output(_page(left))
    grok = _output(_page(right), "grok")
    wording = _dispute(DisputeCategory.WORDING, left, right, dispute_id="dsp-word")
    comparison = _comparison(gpt, grok, [wording], [_alignment(left, right)])
    assert _render(gpt, grok, comparison, []) == f"@@ page_001\n[{left} / {right}]\n"
    shared = "youth and the middle"
    note = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "and",
        "",
        dispute_id="dsp-note",
    )
    noted = _comparison(gpt, grok, [note], [_alignment(shared, shared)])
    text = _render(
        _output(_page(shared)),
        _output(_page(shared), "grok"),
        noted,
        [],
    )
    assert text == "@@ page_001\nyouth and the middle\n[and / ∅]\n"
    assert "⟦and⟧" not in text
    assert "[deleted:" not in text
    left_note = "line a\nline b"
    right_note = "line a\nline c"
    multiline = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        left_note,
        right_note,
        dispute_id="dsp-note-lines",
    )
    block = _render(
        _output(_page(left_note)),
        _output(_page(right_note), "grok"),
        _comparison(
            _output(_page(left_note)),
            _output(_page(right_note), "grok"),
            [multiline],
            [_alignment(left_note, right_note)],
        ),
        [],
    )
    assert block == ("@@ page_001\n[A:\nline a\nline b\n/\nB:\nline a\nline c]\n")
    assert multiline.gpt_text == left_note
    assert multiline.grok_text == right_note


def test_explicit_illegible_is_not_inferred_from_uncertainty() -> None:
    assert ILLEGIBLE_DISPLAY == "[illegible]"
    gpt = _output(_page("Johannes"))
    marked = _page(
        "illegible",
        unreadable=[UnreadableSpan(region_id="r1", text_marker="illegible")],
    )
    grok = _output(marked, "grok")
    dispute = _dispute(
        DisputeCategory.WORDING, "Johannes", "illegible", dispute_id="dsp-ill"
    )
    comparison = _with_originals(
        _comparison(gpt, grok, [dispute], [_alignment("Johannes", "illegible")]),
        gpt,
        grok,
    )
    assert (
        _render(gpt, grok, comparison, []) == "@@ page_001\n[Johannes / [illegible]]\n"
    )
    assert comparison.disputes[0].grok_text == "illegible"
    uncertain = _page("Johanes?", unresolved=True)
    grok_uncertain = _output(uncertain, "grok")
    soft = _dispute(
        DisputeCategory.WORDING, "Johannes", "Johanes?", dispute_id="dsp-soft"
    )
    soft_comparison = _comparison(
        gpt,
        grok_uncertain,
        [soft],
        [_alignment("Johannes", "Johanes?")],
        unresolved_pages=["page_001"],
    )
    soft_text = _render(gpt, grok_uncertain, soft_comparison, [])
    assert "[Johannes / Johanes?]" in soft_text
    assert "[illegible]" not in soft_text
    diplomatic = _dispute(
        DisputeCategory.WORDING,
        "Johannes",
        "united the [deleted: [illegible]] sephardic",
        dispute_id="dsp-keep",
    )
    kept = _render(
        gpt,
        _output(_page("united the [deleted: [illegible]] sephardic"), "grok"),
        _comparison(
            gpt,
            _output(_page("united the [deleted: [illegible]] sephardic"), "grok"),
            [diplomatic],
            [_alignment("Johannes", "united the [deleted: [illegible]] sephardic")],
        ),
        [],
    )
    assert "united the [deleted: [illegible]] sephardic" in kept
    assert "[Johannes / [illegible]]" not in kept
    failed = _unresolved(soft, "UNREADABLE_EVIDENCE")
    assert failed.basis == "UNREADABLE_EVIDENCE"
    assert "[illegible]" not in _render(gpt, grok_uncertain, soft_comparison, [failed])


# --- closure, failure, and unsupported evidence -------------------------------


def test_zero_round_and_partial_challenge_leave_alternatives() -> None:
    gpt = _output(_page("Alpha"))
    grok = _output(_page("Alfa"), "grok")
    first = _dispute(DisputeCategory.WORDING, "Alpha", "Alfa", dispute_id="dsp-1")
    second = _dispute(
        DisputeCategory.WORDING,
        "Bravo",
        "Brava",
        dispute_id="dsp-2",
        region_ids=["r2"],
    )
    comparison = _comparison(
        gpt,
        grok,
        [first, second],
        [
            _alignment("Alpha", "Alfa", alignment_id="aln-1", sequence=0),
            _alignment(
                "Bravo", "Brava", alignment_id="aln-2", sequence=1, region_ids=["r2"]
            ),
        ],
    )
    # The second alignment names r2, so the page regions must include it or the
    # span is still rendered from the alignment record itself.
    closed = _closure([first, second])
    text = _render(gpt, grok, comparison, [])
    assert "[Alpha / Alfa]" in text
    assert "[Bravo / Brava]" in text
    assert _quality(gpt, grok, comparison, [], closed) == "UNRESOLVED_CONTENT"
    assert content_is_unresolved(comparison, [], closed) is True
    chosen = _supported(first, "CANDIDATE_X", "Alpha")
    partial = _closure(
        [second], mode="SELECTED_ROUNDS_COMPLETE", status="NOT_SELECTED", round_count=1
    )
    partial_text = _render(gpt, grok, comparison, [chosen])
    assert "@@ page_001\nAlpha\n[Bravo / Brava]\n" == partial_text
    assert _quality(gpt, grok, comparison, [chosen], partial) == "UNRESOLVED_CONTENT"


def test_provider_failure_and_insufficient_evidence_invent_no_winner() -> None:
    gpt = _output(_page("Alpha"))
    grok = _output(_page("Alfa"), "grok")
    dispute = _dispute(DisputeCategory.WORDING, "Alpha", "Alfa", dispute_id="dsp-1")
    comparison = _comparison(gpt, grok, [dispute], [_alignment("Alpha", "Alfa")])
    failure = _closure(
        [dispute],
        mode="PROVIDER_TERMINAL_FAILURE",
        status="PROVIDER_FAILURE",
        round_count=1,
        failed=[1],
    )
    text = _render(gpt, grok, comparison, [])
    assert text == "@@ page_001\n[Alpha / Alfa]\n"
    assert _quality(gpt, grok, comparison, [], failure) == "UNRESOLVED_CONTENT"
    weak = _unresolved(dispute, "INSUFFICIENT_EVIDENCE")
    split = _unresolved(dispute, "PROVIDER_DISAGREEMENT")
    for decision in (weak, split):
        assert decision.selected_reading is None
        assert _render(gpt, grok, comparison, [decision]) == text
        assert _quality(gpt, grok, comparison, [decision]) == "UNRESOLVED_CONTENT"
        assert decision.gpt_visual_evidence == "The crop does not settle it."


def test_quality_comes_from_structured_state_not_marker_text() -> None:
    decoy = "[unresolved: Alpha || Alfa]\n[unreadable]\n[illegible]"
    gpt = _output(_page(decoy))
    grok = _output(_page(decoy), "grok")
    comparison = _comparison(gpt, grok, [], [_alignment(decoy, decoy)])
    assert _render(gpt, grok, comparison, []) == f"@@ page_001\n{decoy}\n"
    assert _quality(gpt, grok, comparison, []) == "NO_MATERIAL_DISPUTES"
    dispute = _dispute(
        DisputeCategory.PUNCTUATION,
        "Alpha",
        "alpha",
        dispute_id="dsp-punct",
        substantive=False,
    )
    punctuated = _comparison(gpt, grok, [dispute], [_alignment("Alpha", "alpha")])
    punctuated_text = _render(
        _output(_page("Alpha")),
        _output(_page("alpha"), "grok"),
        punctuated,
        [],
    )
    assert "[Alpha / alpha]" in punctuated_text
    assert _quality(
        _output(_page("Alpha")), _output(_page("alpha"), "grok"), punctuated, []
    ) == ("NO_MATERIAL_DISPUTES")
    substantive = _dispute(
        DisputeCategory.WORDING, "Alpha", "Alfa", dispute_id="dsp-real"
    )
    real = _comparison(
        _output(_page("Alpha")),
        _output(_page("Alfa"), "grok"),
        [substantive],
        [_alignment("Alpha", "Alfa")],
    )
    assert "[unresolved:" not in _render(
        _output(_page("Alpha")), _output(_page("Alfa"), "grok"), real, []
    )
    assert (
        _quality(_output(_page("Alpha")), _output(_page("Alfa"), "grok"), real, [])
        == "UNRESOLVED_CONTENT"
    )


# --- several disputes, reading order, alternatives ----------------------------


def test_several_disputes_on_one_span_are_not_duplicated() -> None:
    left = "Alpha one"
    right = "Alfa one"
    alignment = _alignment(left, right)
    wording = _dispute(DisputeCategory.WORDING, left, right, dispute_id="dsp-word")
    name = _dispute(DisputeCategory.NAME_OR_ENTITY, left, right, dispute_id="dsp-name")
    assert same_span(wording, alignment)
    assert same_span(name, alignment)
    gpt = _output(_page(left))
    grok = _output(_page(right), "grok")
    comparison = _comparison(gpt, grok, [wording, name], [alignment])
    text = _render(gpt, grok, comparison, [])
    assert text.count(f"[{left} / {right}]") == 1
    assert text.count(left) == 1
    assert text.count(right) == 1
    shared = "youth and the middle"
    cancellation = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "and",
        "",
        dispute_id="dsp-cancel",
    )
    insertion = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "middle",
        "centre",
        dispute_id="dsp-insert",
    )
    both = _render(
        _output(_page(shared)),
        _output(_page(shared), "grok"),
        _comparison(
            _output(_page(shared)),
            _output(_page(shared), "grok"),
            [cancellation, insertion],
            [_alignment(shared, shared)],
        ),
        [],
    )
    assert both.count(shared) == 1
    assert both.count("[and / ∅]") == 1
    assert both.count("[middle / centre]") == 1


def test_reading_order_shows_each_diplomatic_text_once() -> None:
    gpt_page = _page(
        "alpha beta",
        [_region("gpt-a", 1, "alpha"), _region("gpt-b", 2, "beta")],
    )
    grok_page = _page(
        "beta alpha",
        [_region("grok-b", 1, "beta"), _region("grok-a", 2, "alpha")],
    )
    gpt = _output(gpt_page)
    grok = _output(grok_page, "grok")
    comparison = compare_transcriptions(gpt, grok, gpt_sha256=SHA, grok_sha256=SHA)
    order = next(
        item
        for item in comparison.disputes
        if item.category is DisputeCategory.READING_ORDER
    )
    text = _render(gpt, grok, comparison, [])
    assert text == "@@ page_001\n[alpha beta / beta alpha]\n"
    assert text.count("alpha") == 2
    assert text.count("beta") == 2
    supported = _supported(order, "CANDIDATE_X", order.gpt_text)
    chosen = _render(gpt, grok, comparison, [supported])
    assert order.gpt_text not in chosen
    assert chosen.count("alpha") == 1
    assert chosen.count("beta") == 1
    third = _supported(order, "THIRD_READING", "beta,alpha", third="beta,alpha")
    assert _render(gpt, grok, comparison, [third]) == text
    multiline_gpt = gpt_page.model_copy(
        update={"diplomatic_transcription": "alpha\nbeta"}
    )
    multiline_grok = grok_page.model_copy(
        update={"diplomatic_transcription": "beta\nalpha"}
    )
    multiline = compare_transcriptions(
        _output(multiline_gpt),
        _output(multiline_grok, "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    block = _render(
        _output(multiline_gpt), _output(multiline_grok, "grok"), multiline, []
    )
    assert "[A:\nalpha\nbeta\n/\nB:\nbeta\nalpha]" in block
    assert "alpha beta" not in block


def test_alternative_readings_stay_structured_and_are_not_promoted() -> None:
    alternatives = AlternativeReading(region_id="r1", readings=[" Colour ", "Color"])
    gpt_page = _page("colour", alternatives=[alternatives])
    grok_page = _page("colour")
    gpt = _output(gpt_page)
    grok = _output(grok_page, "grok")
    before_gpt = gpt.model_dump()
    comparison = compare_transcriptions(gpt, grok, gpt_sha256=SHA, grok_sha256=SHA)
    text = _render(gpt, grok, comparison, [])
    assert "Color" not in text
    assert " Colour " not in text
    assert "colour" in text
    report = build_report(
        batch_id=BATCH,
        document_id=101,
        quality=classify_quality(
            single_provider=False,
            gpt=gpt,
            grok=grok,
            comparison=comparison,
            decisions=[],
        ),
        page_ids=["page_001"],
        single_provider=False,
        succeeded_provider=None,
        comparison=comparison,
        decisions=[],
        identifications=[],
        review_completion="completed",
        provider_failure_count=0,
        estimated_work_units=0,
        gpt=gpt,
        grok=grok,
    )
    assert report.alternative_readings is not None
    assert [(item.provider, item.readings) for item in report.alternative_readings] == [
        ("gpt", [" Colour ", "Color"])
    ]
    assert gpt.model_dump() == before_gpt
    assert report.quality.value == "NO_MATERIAL_DISPUTES"
    assert report.production_writeback is False


# --- legacy artifacts, provenance, no writeback --------------------------------


def test_legacy_report_and_old_display_bytes_stay_in_place(tmp_path: Path) -> None:
    gpt = _output(_page("Alpha"))
    grok = _output(_page("Alfa"), "grok")
    dispute = _dispute(DisputeCategory.WORDING, "Alpha", "Alfa", dispute_id="dsp-1")
    comparison = _comparison(gpt, grok, [dispute], [_alignment("Alpha", "Alfa")])
    report = build_report(
        batch_id=BATCH,
        document_id=101,
        quality="UNRESOLVED_CONTENT",  # type: ignore[arg-type]
        page_ids=["page_001"],
        single_provider=False,
        succeeded_provider=None,
        comparison=comparison,
        decisions=[],
        identifications=[],
        review_completion="interim_provisional",
        provider_failure_count=0,
        estimated_work_units=1,
        gpt=gpt,
        grok=grok,
    )
    payload = json.loads(canonical_json(report))
    assert "alternative_readings" not in payload
    assert "challenge_closure" not in payload
    legacy_dir = tmp_path / "final" / "versions" / "v001"
    legacy_dir.mkdir(parents=True)
    old_text = b"@@ page_001\n[unresolved: Alpha || Alfa]\n"
    old_report = canonical_json(report).encode()
    (legacy_dir / "proposed_transcription.txt").write_bytes(old_text)
    (legacy_dir / "review_report.json").write_bytes(old_report)
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"event_type":"review_finalized"}\n', encoding="utf-8")
    text_hash = _digest(old_text)
    report_hash = _digest(old_report)
    ledger_hash = _digest(ledger.read_bytes())
    comparison_hash = _digest(canonical_json(comparison).encode())
    new_text = _render(gpt, grok, comparison, [])
    restored = ReviewReport.model_validate_json(
        (legacy_dir / "review_report.json").read_bytes()
    )
    assert restored.quality.value == "UNRESOLVED_CONTENT"
    assert restored.production_writeback is False
    assert "[Alpha / Alfa]" in new_text
    assert "[unresolved:" not in new_text
    assert (
        _digest((legacy_dir / "proposed_transcription.txt").read_bytes()) == text_hash
    )
    assert _digest((legacy_dir / "review_report.json").read_bytes()) == report_hash
    assert _digest(ledger.read_bytes()) == ledger_hash
    assert _digest(canonical_json(comparison).encode()) == comparison_hash
    assert not (tmp_path / "archive").exists()


def test_render_does_not_mutate_sources_evidence_or_decisions(tmp_path: Path) -> None:
    gpt = _output(_page("Alpha, "))
    grok = _output(_page("alpha."), "grok")
    dispute = _dispute(DisputeCategory.WORDING, "Alpha, ", "alpha.", dispute_id="dsp-1")
    comparison = _with_originals(
        _comparison(gpt, grok, [dispute], [_alignment("Alpha, ", "alpha.")]),
        gpt,
        grok,
    )
    decision = _unresolved(dispute)
    before = (
        gpt.model_dump(),
        grok.model_dump(),
        comparison.model_dump(),
        decision.model_dump(),
    )
    evidence = decision.gpt_visual_evidence
    _render(gpt, grok, comparison, [decision])
    classify_quality(
        single_provider=False,
        gpt=gpt,
        grok=grok,
        comparison=comparison,
        decisions=[decision],
    )
    render_aligned_span("Alpha, ", "alpha.", [dispute], {dispute.dispute_id: decision})
    assert (
        gpt.model_dump(),
        grok.model_dump(),
        comparison.model_dump(),
        decision.model_dump(),
    ) == before
    assert decision.gpt_visual_evidence == evidence
    assert comparison.gpt_output_sha256 == SHA
    assert comparison.page_originals[0].gpt_diplomatic_transcription == "Alpha, "
    archive = tmp_path / "production"
    assert not archive.exists()
