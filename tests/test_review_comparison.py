"""Deterministic comparison fixtures for review_next v1."""

from __future__ import annotations

from pathlib import Path

import pytest

from transcription_review_agent.review.compare import (
    challengeable_disputes,
    compare_transcriptions,
)
from transcription_review_agent.review.models import (
    ApparatusNote,
    AttentionItem,
    CoverageReport,
    DisputeCategory,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    TextRegion,
)

SHA = "ab" * 32


def _coverage(*, complete: bool = True, unresolved: bool = False) -> CoverageReport:
    return CoverageReport(
        coverage_complete=complete,
        unresolved_text_present=unresolved,
        examined_region_ids=["r1"],
    )


def _page(
    text: str,
    *,
    lines: list[str] | None = None,
    regions: list[TextRegion] | None = None,
    attention: list[AttentionItem] | None = None,
    complete: bool = True,
    unresolved: bool = False,
    cancellations: list[ApparatusNote] | None = None,
    insertions: list[ApparatusNote] | None = None,
) -> PageTranscription:
    if regions is None:
        line_text = [text] if lines is None else lines
        regions = [
            TextRegion(
                region_id="r1",
                order=1,
                lines=[
                    LineReading(line_id=f"l{index}", text=line)
                    for index, line in enumerate(line_text, start=1)
                ],
            )
        ]
    return PageTranscription(
        page_id="page_001",
        reading_order=1,
        diplomatic_transcription=text,
        regions=regions,
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=[] if attention is None else attention,
        coverage=_coverage(complete=complete, unresolved=unresolved),
        cancellations=[] if cancellations is None else cancellations,
        insertions=[] if insertions is None else insertions,
        marginal_notes=[],
        inaccessible_or_ambiguous=[],
    )


def _output(page: PageTranscription, provider: str = "gpt") -> FirstPassOutput:
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


def _categories(
    left: PageTranscription, right: PageTranscription
) -> list[DisputeCategory]:
    comparison = compare_transcriptions(
        _output(left, "gpt"),
        _output(right, "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    return [item.category for item in comparison.disputes]


def test_line_wrap_is_not_a_substantive_disagreement() -> None:
    left = _page("hello world again", lines=["hello world", "again"])
    right = _page("hello world again", lines=["hello", "world again"])
    assert _categories(left, right) == []


def test_missing_line_reading_order_names_dates_and_wording() -> None:
    missing = _categories(
        _page("hello world secret line", lines=["hello world", "secret line"]),
        _page("hello world", lines=["hello world"]),
    )
    assert missing == [DisputeCategory.MISSING_CONTENT]

    left_regions = [
        TextRegion(
            region_id="r1", order=1, lines=[LineReading(line_id="l1", text="alpha")]
        ),
        TextRegion(
            region_id="r2", order=2, lines=[LineReading(line_id="l2", text="beta")]
        ),
    ]
    right_regions = [
        TextRegion(
            region_id="r2", order=1, lines=[LineReading(line_id="l2", text="beta")]
        ),
        TextRegion(
            region_id="r1", order=2, lines=[LineReading(line_id="l1", text="alpha")]
        ),
    ]
    order = _categories(
        _page("alpha beta", regions=left_regions),
        _page("beta alpha", regions=right_regions),
    )
    assert DisputeCategory.READING_ORDER in order

    name = _categories(
        _page(
            "Ionnes",
            attention=[AttentionItem(kind="name", text="Ionnes", region_id="r1")],
        ),
        _page(
            "Iannes",
            attention=[AttentionItem(kind="name", text="Iannes", region_id="r1")],
        ),
    )
    assert name == [DisputeCategory.NAME_OR_ENTITY]

    dated = _categories(_page("12 March 1812"), _page("13 March 1812"))
    assert dated == [DisputeCategory.DATE_OR_NUMBER]
    assert _categories(_page("the house"), _page("the home")) == [
        DisputeCategory.WORDING
    ]


def test_punctuation_is_kept_and_not_challenged() -> None:
    comparison = compare_transcriptions(
        _output(_page("Hello, world.")),
        _output(_page("Hello world."), "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    assert [item.category for item in comparison.disputes] == [
        DisputeCategory.PUNCTUATION
    ]
    assert comparison.disputes[0].gpt_text == "Hello, world."
    assert comparison.disputes[0].grok_text == "Hello world."
    assert challengeable_disputes(comparison) == []
    assert comparison.selected_by_confidence is False
    assert "winner" not in comparison.model_dump()


def test_case_spacing_and_quotes_stay_visible_but_are_not_substantive() -> None:
    named = compare_transcriptions(
        _output(
            _page(
                "lord Moyne",
                attention=[
                    AttentionItem(kind="name", text="lord Moyne", region_id="r1")
                ],
            )
        ),
        _output(
            _page(
                "Lord Moyne",
                attention=[
                    AttentionItem(kind="name", text="Lord Moyne", region_id="r1")
                ],
            ),
            "grok",
        ),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    assert [item.category for item in named.disputes] == [DisputeCategory.PUNCTUATION]
    assert named.disputes[0].gpt_text == "lord Moyne"
    assert named.disputes[0].grok_text == "Lord Moyne"
    assert named.disputes[0].substantive is False

    corrupt = _categories(_page("Corrupt"), _page("corrupt."))
    assert corrupt == [DisputeCategory.PUNCTUATION]
    synagogues = _categories(_page("synagogues"), _page("Synagogues,"))
    assert synagogues == [DisputeCategory.PUNCTUATION]
    spaced = _categories(_page("anybody"), _page("any body"))
    assert spaced == [DisputeCategory.PUNCTUATION]
    quoted = _categories(_page("“Hello”"), _page('"Hello"'))
    assert quoted == [DisputeCategory.PUNCTUATION]


def test_accents_letters_dates_and_names_remain_substantive() -> None:
    accent = _categories(
        _page(
            "Moshe",
            attention=[AttentionItem(kind="name", text="Moshe", region_id="r1")],
        ),
        _page(
            "Moshé",
            attention=[AttentionItem(kind="name", text="Moshé", region_id="r1")],
        ),
    )
    assert accent == [DisputeCategory.NAME_OR_ENTITY]
    assert _categories(_page("community"), _page("comunity")) == [
        DisputeCategory.WORDING
    ]
    assert _categories(_page("In 1946"), _page("In 1944")) == [
        DisputeCategory.DATE_OR_NUMBER
    ]
    named = _categories(
        _page(
            "Nokrachi",
            attention=[AttentionItem(kind="name", text="Nokrachi", region_id="r1")],
        ),
        _page(
            "Nokrashi",
            attention=[AttentionItem(kind="name", text="Nokrashi", region_id="r1")],
        ),
    )
    assert named == [DisputeCategory.NAME_OR_ENTITY]


def test_apparatus_note_is_not_repeated_when_the_span_already_records_it() -> None:
    repeated = compare_transcriptions(
        _output(
            _page(
                "united the ⟦ashkenazi⟧ sephardic",
                cancellations=[ApparatusNote(region_id="r1", text="ashkenazi")],
            )
        ),
        _output(
            _page(
                "united the [deleted: [illegible]] sephardic",
                cancellations=[ApparatusNote(region_id="r1", text="illegible")],
            ),
            "grok",
        ),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    assert [item.category for item in repeated.disputes] == [DisputeCategory.WORDING]
    assert "⟦ashkenazi⟧" in repeated.disputes[0].gpt_text
    assert "[deleted: [illegible]]" in repeated.disputes[0].grok_text

    only_note = compare_transcriptions(
        _output(
            _page(
                "youth and the middle",
                cancellations=[ApparatusNote(region_id="r1", text="and")],
            )
        ),
        _output(_page("youth and the middle"), "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    assert [item.category for item in only_note.disputes] == [
        DisputeCategory.CANCELLATION_OR_INSERTION
    ]
    assert only_note.disputes[0].gpt_text == "and"
    assert only_note.disputes[0].grok_text == ""
    assert only_note.disputes[0].substantive is True


def test_complementary_page_numerals_are_one_dispute_and_keep_both_texts() -> None:
    left = _page(
        "hello world\n3",
        regions=[
            TextRegion(
                region_id="main",
                order=1,
                lines=[LineReading(line_id="l1", text="hello world")],
            ),
            TextRegion(
                region_id="margin", order=2, lines=[LineReading(line_id="l2", text="3")]
            ),
        ],
    )
    right = _page(
        "3 /\nhello world\nsecret line",
        regions=[
            TextRegion(
                region_id="head", order=1, lines=[LineReading(line_id="g1", text="3 /")]
            ),
            TextRegion(
                region_id="body",
                order=2,
                lines=[LineReading(line_id="g2", text="hello world")],
            ),
            TextRegion(
                region_id="tail",
                order=3,
                lines=[LineReading(line_id="g3", text="secret line")],
            ),
        ],
    )
    comparison = compare_transcriptions(
        _output(left),
        _output(right, "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    by_category = {item.category: item for item in comparison.disputes}
    assert set(by_category) == {
        DisputeCategory.PUNCTUATION,
        DisputeCategory.MISSING_CONTENT,
    }
    numeral = by_category[DisputeCategory.PUNCTUATION]
    assert numeral.gpt_text == "3"
    assert numeral.grok_text == "3 /"
    assert numeral.substantive is False
    assert numeral.gpt_region_ids == ["margin"]
    assert numeral.grok_region_ids == ["head"]
    missing = by_category[DisputeCategory.MISSING_CONTENT]
    assert missing.gpt_text == ""
    assert missing.grok_text == "secret line"
    assert missing.substantive is True


def test_document_104_comparison_keeps_material_disputes_and_exact_text() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "outputs/reviews/batch_batch-000001/documents/document_104/first_pass/output"
    )
    gpt_path = root / "transcription_gpt.json"
    grok_path = root / "transcription_grok.json"
    if not gpt_path.is_file() or not grok_path.is_file():
        pytest.skip("document 104 provider outputs are not in this checkout")
    gpt_bytes = gpt_path.read_bytes()
    grok_bytes = grok_path.read_bytes()
    gpt = FirstPassOutput.model_validate_json(gpt_bytes)
    grok = FirstPassOutput.model_validate_json(grok_bytes)
    comparison = compare_transcriptions(gpt, grok, gpt_sha256=SHA, grok_sha256=SHA)
    assert gpt_path.read_bytes() == gpt_bytes
    assert grok_path.read_bytes() == grok_bytes
    assert (
        comparison.page_originals[0].gpt_diplomatic_transcription
        == gpt.pages[0].diplomatic_transcription
    )
    assert (
        comparison.page_originals[0].grok_diplomatic_transcription
        == grok.pages[0].diplomatic_transcription
    )

    def contains(
        category: DisputeCategory,
        gpt_fragment: str,
        grok_fragment: str,
        *,
        substantive: bool,
    ) -> None:
        matches = [
            item
            for item in comparison.disputes
            if item.category is category
            and gpt_fragment in item.gpt_text
            and grok_fragment in item.grok_text
            and item.substantive is substantive
        ]
        assert matches, (category, gpt_fragment, grok_fragment)

    contains(DisputeCategory.PUNCTUATION, "lord Moyne", "Lord Moyne", substantive=False)
    contains(DisputeCategory.PUNCTUATION, "Corrupt", "corrupt.", substantive=False)
    contains(DisputeCategory.DATE_OR_NUMBER, "1946", "1944", substantive=True)
    contains(DisputeCategory.DATE_OR_NUMBER, "1946", "1942", substantive=True)
    contains(DisputeCategory.DATE_OR_NUMBER, "82", "88", substantive=True)
    contains(DisputeCategory.NAME_OR_ENTITY, "Moshe", "Moshé", substantive=True)
    contains(DisputeCategory.NAME_OR_ENTITY, "Nokrachi", "Nokrashi", substantive=True)
    contains(DisputeCategory.WORDING, "community", "comunity", substantive=True)
    contains(
        DisputeCategory.MISSING_CONTENT,
        "⟦and⟧ prevented him from ⟦resig⟧ resigning.",
        "",
        substantive=True,
    )
    contains(
        DisputeCategory.MISSING_CONTENT,
        "",
        "[deleted: [illegible]] prevented him from [deleted: [illegible]]",
        substantive=True,
    )
    numeral_missing = [
        item
        for item in comparison.disputes
        if item.category is DisputeCategory.MISSING_CONTENT
        and (
            item.gpt_text.strip() in {"3", "6", "7", "8", "9"}
            or item.grok_text.strip() in {"3 /", "6", "7", "8", "[?9]"}
        )
    ]
    assert numeral_missing == []
    cancellations = [
        item
        for item in comparison.disputes
        if item.category is DisputeCategory.CANCELLATION_OR_INSERTION
    ]
    assert [(item.gpt_text, item.grok_text) for item in cancellations] == [("", "the")]
    assert sum(item.substantive for item in comparison.disputes) == 86
    assert len(comparison.disputes) == 138


def test_coverage_gap_stays_visible_without_choosing_a_reading() -> None:
    comparison = compare_transcriptions(
        _output(_page("same words", complete=False)),
        _output(_page("same words", complete=False), "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )
    assert comparison.disputes[0].category is DisputeCategory.COVERAGE_UNCERTAINTY
    assert comparison.disputes[0].substantive is False
    assert challengeable_disputes(comparison) == []
