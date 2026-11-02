"""Adversarial region alignment shared by comparison and assembly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from transcription_review_agent.review.align import (
    MAX_ALIGN_REGIONS,
    align_page_regions,
)
from transcription_review_agent.review.assemble import render_transcription
from transcription_review_agent.review.compare import compare_transcriptions
from transcription_review_agent.review.errors import AlignmentLimitError
from transcription_review_agent.review.models import (
    ComparisonFile,
    CoverageReport,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    TextRegion,
)

SHA = "ab" * 32


def _coverage(region_ids: list[str]) -> CoverageReport:
    return CoverageReport(
        coverage_complete=True,
        unresolved_text_present=False,
        examined_region_ids=region_ids,
    )


def _region(region_id: str, order: int, text: str) -> TextRegion:
    return TextRegion(
        region_id=region_id,
        order=order,
        lines=[LineReading(line_id=f"{region_id}-line", text=text)],
    )


def _page(text: str, regions: list[TextRegion]) -> PageTranscription:
    return PageTranscription(
        page_id="page_001",
        reading_order=1,
        diplomatic_transcription=text,
        regions=regions,
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=[],
        coverage=_coverage([region.region_id for region in regions]),
        cancellations=[],
        insertions=[],
        marginal_notes=[],
        inaccessible_or_ambiguous=[],
    )


def _output(page: PageTranscription, provider: str) -> FirstPassOutput:
    interface = "GPT_WORK" if provider == "gpt" else "GROK"
    return FirstPassOutput(
        provider=provider,  # type: ignore[arg-type]
        provider_interface=interface,  # type: ignore[arg-type]
        task_id=f"task-{provider}",
        job_id=f"job-{provider}",
        batch_id="batch-000001",
        document_id=101,
        attempt=1,
        source_bundle_sha256=SHA,
        pages=[page],
    )


def _compare(left: PageTranscription, right: PageTranscription) -> ComparisonFile:
    return compare_transcriptions(
        _output(left, "gpt"),
        _output(right, "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )


def _text(
    left: PageTranscription, right: PageTranscription, comparison: ComparisonFile
) -> str:
    return render_transcription(
        page_ids=["page_001"],
        gpt=_output(left, "gpt"),
        grok=_output(right, "grok"),
        comparison=comparison,
        decisions=[],
    )


def _counts(text: str, word: str) -> int:
    return text.split().count(word)


def test_different_region_ids_for_the_same_content_are_not_duplicated() -> None:
    left = _page(
        "alpha beta",
        [_region("gpt-a", 1, "alpha"), _region("gpt-b", 2, "beta")],
    )
    right = _page(
        "alpha beta",
        [_region("grok-a", 1, "alpha"), _region("grok-b", 2, "beta")],
    )
    comparison = _compare(left, right)
    text = _text(left, right, comparison)
    assert [item.kind for item in comparison.alignments] == ["matched", "matched"]
    assert comparison.alignments[0].gpt_region_ids == ["gpt-a"]
    assert comparison.alignments[0].grok_region_ids == ["grok-a"]
    assert _counts(text, "alpha") == 1
    assert _counts(text, "beta") == 1
    assert not any(
        item.category.value == "MISSING_CONTENT" for item in comparison.disputes
    )


def test_one_region_split_across_many_is_emitted_once() -> None:
    left = _page("alpha beta gamma", [_region("whole", 1, "alpha beta gamma")])
    right = _page(
        "alpha beta gamma",
        [
            _region("part-1", 1, "alpha"),
            _region("part-2", 2, "beta"),
            _region("part-3", 3, "gamma"),
        ],
    )
    comparison = _compare(left, right)
    text = _text(left, right, comparison)
    assert comparison.alignments[0].kind == "split"
    assert comparison.alignments[0].gpt_region_ids == ["whole"]
    assert comparison.alignments[0].grok_region_ids == ["part-1", "part-2", "part-3"]
    assert _counts(text, "alpha") == 1
    assert _counts(text, "beta") == 1
    assert _counts(text, "gamma") == 1
    assert not any(
        item.category.value == "MISSING_CONTENT" for item in comparison.disputes
    )


def test_merged_regions_are_emitted_once() -> None:
    left = _page(
        "alpha beta",
        [_region("left-1", 1, "alpha"), _region("left-2", 2, "beta")],
    )
    right = _page("alpha beta", [_region("joined", 1, "alpha beta")])
    comparison = _compare(left, right)
    text = _text(left, right, comparison)
    assert comparison.alignments[0].kind == "merged"
    assert _counts(text, "alpha") == 1
    assert _counts(text, "beta") == 1


def test_reordered_regions_keep_both_orders_without_a_third_copy() -> None:
    left = _page(
        "alpha beta",
        [_region("a", 1, "alpha"), _region("b", 2, "beta")],
    )
    right = _page(
        "beta alpha",
        [_region("b", 1, "beta"), _region("a", 2, "alpha")],
    )
    comparison = _compare(left, right)
    text = _text(left, right, comparison)
    assert any(item.kind == "reordered" for item in comparison.alignments)
    assert "[alpha beta / beta alpha]" in text
    assert text.count("alpha") == 2
    assert text.count("beta") == 2
    assert text.count("[alpha beta / beta alpha]") == 1


def test_missing_content_is_marked_instead_of_dropped() -> None:
    left = _page(
        "alpha secret",
        [_region("keep", 1, "alpha"), _region("extra", 2, "secret")],
    )
    right = _page("alpha", [_region("other", 1, "alpha")])
    comparison = _compare(left, right)
    text = _text(left, right, comparison)
    missing = [item for item in comparison.alignments if item.kind == "missing"]
    assert len(missing) == 1
    assert missing[0].gpt_text == "secret"
    assert missing[0].grok_text == ""
    assert text.count("[secret / ∅]") == 1
    assert _counts(text, "alpha") == 1


def test_long_region_sequence_does_not_recurse() -> None:
    count = 40
    left = _page(
        " ".join(f"word{index}" for index in range(count)),
        [_region(f"g{index}", index + 1, f"word{index}") for index in range(count)],
    )
    right = _page(
        " ".join(f"word{index}" for index in range(count)),
        [_region(f"k{index}", index + 1, f"word{index}") for index in range(count)],
    )
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(64)
    try:
        comparison = _compare(left, right)
        text = _text(left, right, comparison)
    finally:
        sys.setrecursionlimit(previous)
    assert len(comparison.alignments) == count
    assert all(item.kind == "matched" for item in comparison.alignments)
    assert text.count("word0") == 1
    assert text.count("word39") == 1


def test_region_sequence_past_the_bound_fails_before_allocating_a_search() -> None:
    count = MAX_ALIGN_REGIONS + 1
    left = _page(
        "bound",
        [_region(f"g{index}", index + 1, f"word{index}") for index in range(count)],
    )
    right = _page(
        "bound",
        [_region(f"k{index}", index + 1, f"word{index}") for index in range(count)],
    )
    with pytest.raises(AlignmentLimitError, match="refuses more than"):
        align_page_regions(left, right)


def _lines(page: PageTranscription) -> list[str]:
    ordered = sorted(page.regions, key=lambda region: (region.order, region.region_id))
    return [line.text for region in ordered for line in region.lines]


def _recovered(comparison: ComparisonFile, side: str) -> list[str]:
    rows: list[str] = []
    for alignment in comparison.alignments:
        text = alignment.gpt_text if side == "gpt" else alignment.grok_text
        present = (
            alignment.gpt_region_ids if side == "gpt" else alignment.grok_region_ids
        )
        if not present:
            continue
        rows.extend(text.split("\n") if text else [""])
    return rows


def test_three_regions_are_not_paired_with_unrelated_six_region_spans() -> None:
    """The real pilot's shape: one provider splits the letter differently and adds empty regions."""

    gpt_regions = [
        _region("dateline", 1, "Leguevin. le 28 Décembre 1952."),
        TextRegion(
            region_id="body",
            order=2,
            lines=[
                LineReading(line_id="b1", text="Cher Victor,"),
                LineReading(line_id="b2", text="Élie P., Claucent, Roeliebas."),
                LineReading(line_id="b3", text="le Bakad reste."),
            ],
        ),
        TextRegion(
            region_id="list",
            order=3,
            lines=[
                LineReading(line_id="n1", text="1) la cuisine"),
                LineReading(line_id="n2", text="des deux Khoutot"),
            ],
        ),
    ]
    grok_regions = [
        _region("heading", 1, "Seguein, le 28 Décembre 1952."),
        _region("salutation", 2, "Chèr Victor,"),
        TextRegion(
            region_id="main",
            order=3,
            lines=[
                LineReading(line_id="m1", text="Elie P., Clement, Rochel S."),
                LineReading(line_id="m2", text="le Bahad reste."),
                LineReading(line_id="m3", text="1) la cuisine"),
                LineReading(line_id="m4", text="des deux Khoulou"),
            ],
        ),
        TextRegion(region_id="oval", order=4, lines=[]),
        TextRegion(region_id="verso", order=5, lines=[]),
        TextRegion(region_id="margins", order=6, lines=[]),
    ]
    left = _page("gpt", gpt_regions)
    right = _page("grok", grok_regions)
    comparison = _compare(left, right)
    assert _recovered(comparison, "gpt") == _lines(left)
    assert _recovered(comparison, "grok") == _lines(right)
    assert not any(item.kind == "missing" for item in comparison.alignments)
    paired = {(item.gpt_text, item.grok_text) for item in comparison.alignments}
    assert ("Leguevin. le 28 Décembre 1952.", "Seguein, le 28 Décembre 1952.") in paired
    assert ("Cher Victor,", "Chèr Victor,") in paired
    assert ("Élie P., Claucent, Roeliebas.", "Elie P., Clement, Rochel S.") in paired
    assert ("le Bakad reste.", "le Bahad reste.") in paired
    assert ("1) la cuisine", "1) la cuisine") in paired
    assert ("des deux Khoutot", "des deux Khoulou") in paired
    assert all(
        "Cher Victor," not in item.gpt_text or "Khoutot" not in item.gpt_text
        for item in comparison.alignments
    )


def test_shifted_lines_are_not_zipped_by_index() -> None:
    left = _page(
        "left",
        [
            _region("a", 1, "alpha"),
            _region("b", 2, "UNIQUEINSERT"),
            _region("c", 3, "gamma"),
        ],
    )
    right = _page(
        "right",
        [
            _region("a", 1, "alpha"),
            _region("c", 2, "gamma"),
            _region("d", 3, "TRAILINGONLY"),
        ],
    )
    comparison = _compare(left, right)
    assert _recovered(comparison, "gpt") == ["alpha", "UNIQUEINSERT", "gamma"]
    assert _recovered(comparison, "grok") == ["alpha", "gamma", "TRAILINGONLY"]
    paired = {(item.gpt_text, item.grok_text) for item in comparison.alignments}
    assert ("alpha", "alpha") in paired
    assert ("gamma", "gamma") in paired
    assert ("UNIQUEINSERT", "gamma") not in paired
    assert ("gamma", "TRAILINGONLY") not in paired
    missing = [item for item in comparison.alignments if item.kind == "missing"]
    assert {item.gpt_text or item.grok_text for item in missing} == {
        "UNIQUEINSERT",
        "TRAILINGONLY",
    }


def test_real_provider_lines_stay_with_their_counterparts() -> None:
    root = Path(
        "/mnt/c/Users/User/vs-archive-transcription-review-agent/"
        "desktop_pilot/local_image_worker_integration_01"
    )
    gpt_path = (
        root / "first_pass/schema_repair/gpt/transcription_gpt.schema_repair.json"
    )
    grok_path = (
        root / "first_pass/schema_repair/grok/transcription_grok.schema_repair.json"
    )
    if not gpt_path.is_file() or not grok_path.is_file():
        pytest.skip("real provider repair files are not on this machine")
    from transcription_review_agent.review.models import FirstPassOutput

    gpt = FirstPassOutput.model_validate_json(gpt_path.read_bytes())
    grok = FirstPassOutput.model_validate_json(grok_path.read_bytes())
    comparison = compare_transcriptions(gpt, grok, gpt_sha256=SHA, grok_sha256=SHA)
    assert _recovered(comparison, "gpt") == _lines(gpt.pages[0])
    assert _recovered(comparison, "grok") == _lines(grok.pages[0])
    paired = {(item.gpt_text, item.grok_text) for item in comparison.alignments}
    assert any(
        left.startswith("Leguevin") and right.startswith("Seguein")
        for left, right in paired
    )
    assert ("Cher Victor,", "Chèr Victor,") in paired
    assert any("Bakad" in left and "Bahad" in right for left, right in paired)
    assert any("Khoutot" in left and "Khoulou" in right for left, right in paired)
    assert not any(
        item.kind == "missing" and item.grok_text == "Chèr Victor,"
        for item in comparison.alignments
    )
    assert not any(
        item.gpt_text.startswith("Leguevin") and "Cher Victor," in item.gpt_text
        for item in comparison.alignments
    )
