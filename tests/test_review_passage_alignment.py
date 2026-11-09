"""Document 108 regressions for passage alignment and assembly.

The excerpts are the two spans that failed in that review. The read-only
preview loads the local provider files when they are present and does not
write a bundle, a review, or a transcription.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from transcription_review_agent.review.align import _similarity
from transcription_review_agent.review.assemble import render_transcription
from transcription_review_agent.review.compare import compare_transcriptions
from transcription_review_agent.review.formatting import format_unresolved_alternatives
from transcription_review_agent.review.models import (
    ComparisonFile,
    CoverageReport,
    DisputeCategory,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    TextRegion,
)

SHA = "ab" * 32

# Page 5 of document 108. The names differ enough to fall under the ordinary
# similarity cutoff, and the providers put the line in different regions.
# The shared tail is the same passage, not a second copy of missing text.
VINEYARD_GPT = (
    "Bey, Cattelaoui-Heller, Hadsor, Roffe, Elco-Israélis, Chubary, Elbanaki, "
    "ont fait [illisible] pousser dans"
)
VINEYARD_GROK = (
    "- grec, Catt... [illisible : fin de nom]-Hél... [illisible : fin de nom], "
    "Hasson, Roffé, Elia - israélites, Chic... [illisible : nom], libanais, "
    "ont fait [biffé : [illisible]] pousser dans"
)
VINEYARD_BEFORE = (
    "gréco-romaine, le grenier à blé de l'Egypte. A une époque plus récente, "
    "des particuliers comme Gianaclis"
)
VINEYARD_AFTER = (
    "ce désert des vignobles réputés pour la qualité et l'abondance de leurs "
    "récoltes, mais ces domaines"
)

# Page 1. The article, the noun, and the trailing word differ. The rest of
# the line is the same passage and has to stay outside those spans.
CINEMA_GPT = (
    "gens avaient provoqué une petite explosion dans une cuisine sans causer "
    "des dégâts et sans atteindre de"
)
CINEMA_GROK = (
    "gens avaient provoqué une petite explosion dans un cinéma sans causer "
    "des dégâts et sans atteindre"
)

_PILOT = Path("/mnt/c/Users/User/vs-archive-transcription-review-agent/desktop_pilot")


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


def _page(page_id: str, regions: list[TextRegion]) -> PageTranscription:
    return PageTranscription(
        page_id=page_id,
        reading_order=1,
        diplomatic_transcription="\n".join(
            line.text for region in regions for line in region.lines
        ),
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


def _output(pages: list[PageTranscription], provider: str) -> FirstPassOutput:
    interface = "GPT_WORK" if provider == "gpt" else "GROK"
    return FirstPassOutput(
        provider=provider,  # type: ignore[arg-type]
        provider_interface=interface,  # type: ignore[arg-type]
        task_id=f"task-{provider}",
        job_id=f"job-{provider}",
        batch_id="batch-000001",
        document_id=108,
        attempt=1,
        source_bundle_sha256=SHA,
        pages=pages,
    )


def _compare(
    left: list[PageTranscription], right: list[PageTranscription]
) -> ComparisonFile:
    return compare_transcriptions(
        _output(left, "gpt"),
        _output(right, "grok"),
        gpt_sha256=SHA,
        grok_sha256=SHA,
    )


def _render(
    left: list[PageTranscription],
    right: list[PageTranscription],
    comparison: ComparisonFile,
) -> str:
    return render_transcription(
        page_ids=[page.page_id for page in left],
        gpt=_output(left, "gpt"),
        grok=_output(right, "grok"),
        comparison=comparison,
        decisions=[],
    )


def test_vineyard_overlap_is_one_passage_and_keeps_both_name_lists() -> None:
    from transcription_review_agent.review.align import _Piece

    left_piece = _Piece("gpt", VINEYARD_GPT, 0)
    right_piece = _Piece("grok", VINEYARD_GROK, 0)
    similarity = _similarity(left_piece, right_piece)
    assert similarity < 0.62

    left = _page(
        "page_005",
        [
            TextRegion(
                region_id="page_005-r02",
                order=1,
                lines=[
                    LineReading(line_id="g1", text=VINEYARD_BEFORE),
                    LineReading(line_id="g2", text=VINEYARD_GPT),
                    LineReading(line_id="g3", text=VINEYARD_AFTER),
                ],
            )
        ],
    )
    right = _page(
        "page_005",
        [
            TextRegion(
                region_id="page_005-r06",
                order=1,
                lines=[
                    LineReading(line_id="k1", text=VINEYARD_BEFORE),
                    LineReading(line_id="k2", text=VINEYARD_GROK),
                    LineReading(line_id="k3", text=VINEYARD_AFTER.replace(",", ";")),
                ],
            )
        ],
    )
    comparison = _compare([left], [right])
    text = _render([left], [right], comparison)

    vineyard = [
        item
        for item in comparison.alignments
        if "pousser dans" in item.gpt_text or "pousser dans" in item.grok_text
    ]
    assert len(vineyard) == 1
    assert vineyard[0].kind == "matched"
    assert vineyard[0].gpt_region_ids == ["page_005-r02"]
    assert vineyard[0].grok_region_ids == ["page_005-r06"]
    assert vineyard[0].gpt_text == VINEYARD_GPT
    assert vineyard[0].grok_text == VINEYARD_GROK

    disputes = [
        item
        for item in comparison.disputes
        if "pousser dans" in item.gpt_text or "pousser dans" in item.grok_text
    ]
    assert len(disputes) == 1
    assert disputes[0].category is not DisputeCategory.MISSING_CONTENT
    assert disputes[0].gpt_text == VINEYARD_GPT
    assert disputes[0].grok_text == VINEYARD_GROK
    assert disputes[0].substantive is True

    assert text.count("pousser dans") == 1
    assert text.count("ont fait") == 1
    assert text.count("Cattelaoui-Heller") == 1
    assert text.count("Hasson") == 1
    assert text.count("[biffé : [illisible]]") == 1
    assert "[illisible] pousser dans / ∅]" not in text
    assert "[∅ / - grec" not in text
    assert VINEYARD_BEFORE in text
    assert "ce désert des vignobles" in text
    # The name lists are one disagreement. They are not zipped name by name.
    assert "[Bey, / -]" not in text
    assert "[Cattelaoui-Heller, / grec,]" not in text


def test_cinema_partial_agreement_stays_outside_the_disputed_words() -> None:
    left = _page("page_001", [_region("page_001-r02", 1, CINEMA_GPT)])
    right = _page("page_001", [_region("page_001-r04", 1, CINEMA_GROK)])
    comparison = _compare([left], [right])
    text = _render([left], [right], comparison)
    line = text.splitlines()[1]
    assert line == (
        "gens avaient provoqué une petite explosion dans [une / un] "
        "[cuisine / cinéma] sans causer des dégâts et sans atteindre[ de / ∅]"
    )
    assert line.count("gens avaient provoqué une petite explosion dans ") == 1
    assert "cuisine" in line and "cinéma" in line
    assert "[ de / ∅]" in line
    assert "[gens avaient" not in line

    dispute = comparison.disputes[0]
    assert dispute.gpt_text == CINEMA_GPT
    assert dispute.grok_text == CINEMA_GROK
    assert dispute.gpt_region_ids == ["page_001-r02"]
    assert dispute.grok_region_ids == ["page_001-r04"]
    assert dispute.category is not DisputeCategory.MISSING_CONTENT

    # A disputed annotation beside the same words must not swallow the line.
    marked = format_unresolved_alternatives(
        "dans une cuisine [illisible] fin",
        "dans un cinéma [biffé : mot] fin",
    )
    assert marked == (
        "dans [une / un] [cuisine / cinéma] [A: [illisible] / B: [biffé : mot]] fin"
    )


def test_genuine_missing_content_stays_missing() -> None:
    left = _page(
        "page_001",
        [
            _region("keep", 1, "alpha"),
            _region("extra", 2, "secret inventory that nobody else has"),
        ],
    )
    right = _page("page_001", [_region("other", 1, "alpha")])
    comparison = _compare([left], [right])
    text = _render([left], [right], comparison)
    missing = [item for item in comparison.alignments if item.kind == "missing"]
    assert len(missing) == 1
    assert missing[0].gpt_text == "secret inventory that nobody else has"
    assert missing[0].grok_text == ""
    assert text.count("[secret inventory that nobody else has / ∅]") == 1
    assert text.split().count("alpha") == 1


def test_different_sentences_with_a_short_shared_opening_stay_unaligned() -> None:
    gpt = (
        "In 1933, the Alexandrian Jewry donated 15000 Eg. Pounds for the "
        "purchase of a land in Israel for"
    )
    grok = (
        "In 1933, the Bné Brit organization held a big meeting at the "
        "orphanage of Cairo to protest against the Nazi"
    )
    left = _page("page_003", [_region("gpt-a", 1, gpt)])
    right = _page("page_003", [_region("grok-b", 1, grok)])
    comparison = _compare([left], [right])
    assert [item.kind for item in comparison.alignments] == ["missing", "missing"]
    text = _render([left], [right], comparison)
    assert gpt in text
    assert grok in text
    assert "Alexandrian" in text and "Bné Brit" in text


def test_page_boundaries_keep_each_passage_on_its_page() -> None:
    cinema_left = _page("page_001", [_region("a", 1, CINEMA_GPT)])
    cinema_right = _page("page_001", [_region("b", 1, CINEMA_GROK)])
    vineyard_left = _page("page_005", [_region("c", 1, VINEYARD_GPT)])
    vineyard_right = _page("page_005", [_region("d", 1, VINEYARD_GROK)])
    comparison = _compare([cinema_left, vineyard_left], [cinema_right, vineyard_right])
    text = _render(
        [cinema_left, vineyard_left],
        [cinema_right, vineyard_right],
        comparison,
    )
    page_one, page_five = text.split("@@ page_005\n", maxsplit=1)
    assert page_one.startswith("@@ page_001\n")
    assert "cinéma" in page_one
    assert "pousser dans" not in page_one
    assert "pousser dans" in page_five
    assert "cinéma" not in page_five
    assert text.count("pousser dans") == 1
    assert text.count("[cuisine / cinéma]") == 1


def test_document_108_reassembly_is_read_only() -> None:
    gpt_path = _PILOT / "first_pass_gpt.json"
    grok_path = _PILOT / "transcription_grok.json"
    if not gpt_path.is_file() or not grok_path.is_file():
        pytest.skip("document 108 provider files are not on this machine")
    gpt_bytes = gpt_path.read_bytes()
    grok_bytes = grok_path.read_bytes()
    gpt = FirstPassOutput.model_validate_json(gpt_bytes)
    grok = FirstPassOutput.model_validate_json(grok_bytes)
    if gpt.document_id != 108 or grok.document_id != 108:
        pytest.skip("provider files are not document 108")
    comparison = compare_transcriptions(gpt, grok, gpt_sha256=SHA, grok_sha256=SHA)
    text = render_transcription(
        page_ids=[page.page_id for page in gpt.pages],
        gpt=gpt,
        grok=grok,
        comparison=comparison,
        decisions=[],
    )
    assert gpt_path.read_bytes() == gpt_bytes
    assert grok_path.read_bytes() == grok_bytes

    vineyard = [
        item
        for item in comparison.disputes
        if item.page_id == "page_005"
        and "pousser dans" in item.gpt_text + item.grok_text
    ]
    assert len(vineyard) == 1
    assert vineyard[0].category is not DisputeCategory.MISSING_CONTENT
    assert vineyard[0].gpt_text == VINEYARD_GPT
    assert vineyard[0].grok_text == VINEYARD_GROK

    page_one, page_five = text.split("@@ page_005\n", maxsplit=1)
    assert "@@ page_001\n" in text
    assert page_one.index("@@ page_001") < text.index("@@ page_005")
    cinema = next(line for line in page_one.splitlines() if "cinéma" in line)
    assert cinema == (
        "gens avaient provoqué une petite explosion dans [une / un] "
        "[cuisine / cinéma] sans causer des dégâts et sans atteindre[ de / ∅]"
    )
    assert page_five.count("pousser dans") == 1
    assert "Cattelaoui-Heller" in page_five
    assert "Hasson" in page_five
    assert "[Bey, / -]" not in page_five
    assert "[∅ / - grec" not in page_five
