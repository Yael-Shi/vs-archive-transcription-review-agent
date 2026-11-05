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
    assert text == "@@ page_001\nyouth and the middle\n"
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
    assert text == "@@ page_001\n[Alpha / Alfa] one\n"
    assert text.count("Alpha") == 1
    assert text.count("Alfa") == 1
    assert text.count("one") == 1
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
    assert both.count("[and / ∅]") == 0
    assert both.count("[middle / centre]") == 1


def test_fragment_decision_keeps_both_line_readings() -> None:
    left = (
        "Egyptian Hymn were played by the philharmonic of the Alexandria Jewish school."
    )
    right = "Egyptian Hymn were played by the philharmonic of the Alexandria Jewish School ~"
    fragment = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "school",
        "",
        dispute_id="dsp-school",
    )
    punctuation = _dispute(
        DisputeCategory.PUNCTUATION,
        left,
        right,
        dispute_id="dsp-school-punct",
        substantive=False,
    )
    gpt = _output(_page(left))
    grok = _output(_page(right), "grok")
    comparison = _comparison(
        gpt, grok, [fragment, punctuation], [_alignment(left, right)]
    )
    decision = _supported(fragment, "CANDIDATE_X", "school")
    text = _render(gpt, grok, comparison, [decision])
    assert text == (
        "@@ page_001\n"
        "Egyptian Hymn were played by the philharmonic of the Alexandria Jewish "
        "[school. / School ~]\n"
    )
    assert "[school / ∅]" not in text
    assert decision.selected_reading == "school"


def test_third_reading_replaces_only_a_full_line() -> None:
    left = (
        "Egyptian Hymn were played by the philharmonic of the Alexandria Jewish school."
    )
    right = (
        "Egyptian Hymn were played by the philharmonic of the "
        "Alexandria Jewish School ~"
    )
    fragment = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "school",
        "",
        dispute_id="dsp-school-third",
    )
    gpt = _output(_page(left))
    grok = _output(_page(right), "grok")
    fragment_decision = _supported(
        fragment, "THIRD_READING", "schoolhouse", third="schoolhouse"
    )
    fragment_text = _render(
        gpt,
        grok,
        _comparison(gpt, grok, [fragment], [_alignment(left, right)]),
        [fragment_decision],
    )
    assert fragment_text == (
        "@@ page_001\n"
        "Egyptian Hymn were played by the philharmonic of the Alexandria Jewish "
        "[school. / School ~]\n"
    )
    assert "schoolhouse" not in fragment_text

    whole = _dispute(DisputeCategory.WORDING, left, right, dispute_id="dsp-school-line")
    third = (
        "Egyptian Hymn were played by the philharmonic of the Alexandria Jewish school"
    )
    whole_decision = _supported(whole, "THIRD_READING", third, third=third)
    whole_text = _render(
        gpt,
        grok,
        _comparison(gpt, grok, [whole], [_alignment(left, right)]),
        [whole_decision],
    )
    assert whole_text == f"@@ page_001\n{third}\n"
    assert "School ~" not in whole_text
    assert whole_decision.selected_reading == third


def test_full_line_decisions_still_replace_aghion_and_hazan() -> None:
    aghion_left = (
        "The biggest relevant school was that of the Jewish community, "
        "Aslan school, built in the 20's of the last cent."
    )
    aghion_right = (
        "The biggest & oldest schools [deleted: was] that of the Jewish community "
        "Aghion Schools, built in the 70's of the last century"
    )
    aghion = _dispute(
        DisputeCategory.NAME_OR_ENTITY,
        aghion_left,
        aghion_right,
        dispute_id="dsp-aghion",
    )
    gpt = _output(_page(aghion_left))
    grok = _output(_page(aghion_right), "grok")
    aghion_text = _render(
        gpt,
        grok,
        _comparison(gpt, grok, [aghion], [_alignment(aghion_left, aghion_right)]),
        [_supported(aghion, "CANDIDATE_Y", aghion_right)],
    )
    assert aghion_text == f"@@ page_001\n{aghion_right}\n"
    assert "Aslan school" not in aghion_text

    hazan_left = (
        "Rabbi Bekhor Eliahou Hazan (1878-1908), the Chief Rabbi of Alexandria "
        "organized a committee for the"
    )
    hazan_right = (
        "(Rabbi Bekhor Eliahou Hazan) (1888-1908), the Chief Rabbi of Alexandria "
        "organized a committee for the"
    )
    hazan = _dispute(
        DisputeCategory.NAME_OR_ENTITY,
        hazan_left,
        hazan_right,
        dispute_id="dsp-hazan",
    )
    hazan_gpt = _output(_page(hazan_left))
    hazan_grok = _output(_page(hazan_right), "grok")
    hazan_text = _render(
        hazan_gpt,
        hazan_grok,
        _comparison(
            hazan_gpt,
            hazan_grok,
            [hazan],
            [_alignment(hazan_left, hazan_right)],
        ),
        [_supported(hazan, "CANDIDATE_Y", hazan_right)],
    )
    assert hazan_text == f"@@ page_001\n{hazan_right}\n"
    assert "1878-1908" not in hazan_text


def test_one_sided_notes_already_in_the_line_are_not_repeated() -> None:
    footnote_left = '1) For more information see next-year "Egyptian Jews" (hebrew)'
    footnote_right = '1) For more information see Bat-Yeor "Egyptian Jews) (hebrew)'
    footnote = _dispute(
        DisputeCategory.NAME_OR_ENTITY,
        footnote_left,
        footnote_right,
        dispute_id="dsp-footnote",
    )
    echo = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        footnote_left,
        "",
        dispute_id="dsp-footnote-echo",
    )
    gpt = _output(_page(footnote_left))
    grok = _output(_page(footnote_right), "grok")
    footnote_text = _render(
        gpt,
        grok,
        _comparison(
            gpt,
            grok,
            [footnote, echo],
            [_alignment(footnote_left, footnote_right)],
        ),
        [],
    )
    assert footnote_text == (
        "@@ page_001\n"
        "1) For more information see [next-year / Bat-Yeor] "
        '"Egyptian [Jews" / Jews)] (hebrew)\n'
    )
    assert footnote_text.count("next-year") == 1
    assert footnote_text.count("Bat-Yeor") == 1
    assert footnote_text.count(footnote_left) == 0
    assert "[∅]" not in footnote_text

    wars_left = (
        "Important events took place in the Synagogues. "
        "Enlisting of soldiers in the two world wars"
    )
    wars_right = (
        "Important events took place in the (synagogues). "
        "Enlisting of soldiers in the two world wars"
    )
    wars = _dispute(
        DisputeCategory.WORDING, wars_left, wars_right, dispute_id="dsp-wars"
    )
    wars_note = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "in the two world wars",
        "",
        dispute_id="dsp-wars-note",
    )
    wars_text = _render(
        _output(_page(wars_left)),
        _output(_page(wars_right), "grok"),
        _comparison(
            _output(_page(wars_left)),
            _output(_page(wars_right), "grok"),
            [wars, wars_note],
            [_alignment(wars_left, wars_right)],
        ),
        [],
    )
    assert wars_text == (
        "@@ page_001\n"
        "Important events took place in the [Synagogues. / (synagogues).] "
        "Enlisting of soldiers in the two world wars\n"
    )
    assert "[in the two world wars / ∅]" not in wars_text

    egypt_lines = (
        (
            "1947. In 1948 he was expelled from Egypt for Zionist activity.",
            "1947. In 1948 he was expelled from Egypt for Zionist activity.",
        ),
        (
            "Nahum Effendi, Chief Rabbi of Egypt, born in Turkey.",
            "Nahum Effendi, chief Rabi of [inserted: all] Egypt, born in Turkey.",
        ),
        (
            'England. Up with Egypt. Evacuation, Evacuation."',
            'England. Up with Egypt. Evacuation, Evacuation".',
        ),
    )
    egypt_note = _dispute(
        DisputeCategory.CANCELLATION_OR_INSERTION,
        "Egypt",
        "",
        dispute_id="dsp-egypt",
    )
    egypt_disputes = [egypt_note]
    egypt_alignments = []
    for index, (left, right) in enumerate(egypt_lines):
        egypt_disputes.append(
            _dispute(
                DisputeCategory.WORDING,
                left,
                right,
                dispute_id=f"dsp-egypt-line-{index}",
            )
        )
        egypt_alignments.append(
            _alignment(left, right, alignment_id=f"aln-egypt-{index}", sequence=index)
        )
    egypt_text = _render(
        _output(_page(egypt_lines[0][0])),
        _output(_page(egypt_lines[0][1]), "grok"),
        _comparison(
            _output(_page(egypt_lines[0][0])),
            _output(_page(egypt_lines[0][1]), "grok"),
            egypt_disputes,
            egypt_alignments,
        ),
        [],
    )
    assert egypt_text.count("[Egypt / ∅]") == 0
    assert egypt_text.count(egypt_lines[0][0]) == 1
    assert f"[{egypt_lines[1][0]} / {egypt_lines[1][1]}]" in egypt_text
    assert (
        'England. Up with Egypt. Evacuation, [Evacuation." / Evacuation".]'
        in egypt_text
    )


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


# --- minimized alternatives and synthetic unreadable markers ------------------


def _full_line(left: str, right: str) -> str:
    return f"[{left} / {right}]"


def test_document_104_lines_share_unchanged_words_once() -> None:
    assert (
        format_unresolved_alternatives(
            "and cannot accept presents.",
            "and canot accept presents.",
        )
        == "and [cannot / canot] accept presents."
    )
    assert (
        format_unresolved_alternatives(
            "on the 14th of May 1948, because of Zionist activities.",
            "on the 17th of may 1948, because of Zionist activities.",
        )
        == "on the [14th / 17th] of [May / may] 1948, because of Zionist activities."
    )
    propaganda_left = (
        "3- german propaganda and its influence on the palace, the Army, "
        "and the government"
    )
    propaganda_right = (
        "3- german propaganda, and its influence on the public, the Army, "
        "and the goverment"
    )
    assert format_unresolved_alternatives(propaganda_left, propaganda_right) == (
        "3- german [propaganda / propaganda,] and its influence on the "
        "[palace, / public,] the Army, and the [government / goverment]"
    )
    assert format_unresolved_alternatives(
        "and  cannot accept", "and  canot accept"
    ) == ("and  [cannot / canot] accept")
    shown = format_unresolved_alternatives(
        "and cannot accept presents.",
        "and canot accept presents.",
    )
    assert "cannot" in shown and "canot" in shown
    assert shown != "and cannot accept presents."
    assert shown != "and canot accept presents."


def test_repeated_annotation_stays_outside_the_alternative() -> None:
    left = "pray [inserted: at the synagogue] for the victory"
    right = "pray [inserted: at the synagogue] for the Victory"
    assert format_unresolved_alternatives(left, right) == (
        "pray [inserted: at the synagogue] for the [victory / Victory]"
    )


def test_word_insertion_and_deletion_keep_both_sides() -> None:
    assert (
        format_unresolved_alternatives(
            "the youth and the poor",
            "the youth extra and the poor",
        )
        == "the youth[∅ /  extra] and the poor"
    )
    assert (
        format_unresolved_alternatives("foo X bar baz", "foo bar baz Y")
        == "foo[ X / ∅] bar baz[∅ /  Y]"
    )


def test_diplomatic_annotations_are_not_split() -> None:
    cancelled = (
        "movement. He was supported by the youth [cancelled: and] the "
        "middle-class, and the poor."
    )
    deleted = (
        "movement. He was supported by the youth [deleted: and], the "
        "middle class, and the poors"
    )
    assert format_unresolved_alternatives(cancelled, deleted) == _full_line(
        cancelled, deleted
    )
    assert "[middle-class, / middle class,]" not in format_unresolved_alternatives(
        cancelled, deleted
    )
    nested_left = "easy nationalisation for [inserted: Palestine] Jews."
    nested_right = "easy naturalisation for [inserted: [unclear: a fortune?]] Jews."
    nested = format_unresolved_alternatives(nested_left, nested_right)
    assert nested == _full_line(nested_left, nested_right)
    assert "a fortune? / Palestine" not in nested
    unclear_left = "On World War II, nightly bombing occured in Alexandria."
    unclear_right = "On World War II [unclear: nightly] bombing occured in Alexandria."
    unclear = format_unresolved_alternatives(unclear_left, unclear_right)
    assert unclear == _full_line(unclear_left, unclear_right)
    assert "[nightly / unclear:" not in unclear
    illegible_left = "the late Elie Cohen from [illegible]"
    illegible_right = "the late Eli Cohen from [unclear: Damascus?]"
    illegible = format_unresolved_alternatives(illegible_left, illegible_right)
    assert illegible == _full_line(illegible_left, illegible_right)
    assert "[[illegible] /" not in illegible


def test_unsafe_alignment_keeps_the_full_line() -> None:
    assert format_unresolved_alternatives("see a / b now", "see c now") == _full_line(
        "see a / b now", "see c now"
    )
    assert format_unresolved_alternatives("alpha beta", "beta alpha") == _full_line(
        "alpha beta", "beta alpha"
    )
    assert format_unresolved_alternatives("[deleted: foo", "bar") == _full_line(
        "[deleted: foo", "bar"
    )


def test_unresolved_text_flag_does_not_invent_an_unreadable_marker() -> None:
    gpt = _output(_page("shared line", unresolved=True))
    grok = _output(_page("shared line", unresolved=True), "grok")
    comparison = _comparison(
        gpt,
        grok,
        [],
        [_alignment("shared line", "shared line")],
        unresolved_pages=["page_001"],
    )
    text = _render(gpt, grok, comparison, [])
    assert text == "@@ page_001\nshared line\n"
    assert "[unreadable]" not in text
    assert _quality(gpt, grok, comparison, []) == "UNRESOLVED_CONTENT"
    single = render_transcription(
        page_ids=["page_001"],
        gpt=gpt,
        grok=None,
        comparison=None,
        decisions=[],
    )
    assert single == "@@ page_001\nshared line\n"
    assert "[unreadable]" not in single


def test_provider_unreadable_marker_stays_in_the_diplomatic_text() -> None:
    body = "faint ink\n[unreadable]"
    gpt = _output(_page(body, unresolved=True))
    grok = _output(_page(body, unresolved=True), "grok")
    text = _render(
        gpt,
        grok,
        _comparison(gpt, grok, [], [_alignment(body, body)]),
        [],
    )
    assert text == f"@@ page_001\n{body}\n"
    assert text.count("[unreadable]") == 1
    kept = "line with [unreadable] in the margin"
    other = "line with [unreadable] in the note"
    assert (
        format_unresolved_alternatives(kept, other)
        == "line with [unreadable] in the [margin / note]"
    )
    differing = format_unresolved_alternatives("faint [unreadable]", "faint ink")
    assert differing == _full_line("faint [unreadable]", "faint ink")
    assert "[[unreadable] /" not in differing
