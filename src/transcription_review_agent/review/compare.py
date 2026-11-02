"""Deterministic region alignment and dispute detection.

Line wrapping is not a disagreement. Confidence is not read and cannot
select a reading.
"""

from __future__ import annotations

import re
from collections import Counter

from transcription_review_agent.bundling.hashing import sha256_text
from transcription_review_agent.review.align import align_page_regions
from transcription_review_agent.review.models import (
    CHALLENGEABLE_CATEGORIES,
    DISPUTE_PRIORITY,
    ApparatusNote,
    ComparisonFile,
    Dispute,
    DisputeCategory,
    FirstPassOutput,
    LineReading,
    PageOriginals,
    PageTranscription,
    RegionAlignment,
)

_WORD = re.compile(r"\w+", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")
_MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "janvier",
    "février",
    "fevrier",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "aout",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
    "decembre",
}


def compare_transcriptions(
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    *,
    gpt_sha256: str,
    grok_sha256: str,
) -> ComparisonFile:
    """Align pages and regions, then classify disagreements."""

    if gpt.batch_id != grok.batch_id or gpt.document_id != grok.document_id:
        raise ValueError("provider files are not from the same document")
    if gpt.source_bundle_sha256 != grok.source_bundle_sha256:
        raise ValueError("provider files do not share a source bundle hash")

    gpt_pages = {page.page_id: page for page in gpt.pages}
    grok_pages = {page.page_id: page for page in grok.pages}
    page_ids = list(
        dict.fromkeys(
            [page.page_id for page in gpt.pages] + [page.page_id for page in grok.pages]
        )
    )
    disputes: list[Dispute] = []
    alignments: list[RegionAlignment] = []
    originals: list[PageOriginals] = []
    unresolved_pages: list[str] = []

    for page_id in page_ids:
        left = gpt_pages.get(page_id)
        right = grok_pages.get(page_id)
        originals.append(
            PageOriginals(
                page_id=page_id,
                gpt_diplomatic_transcription=""
                if left is None
                else left.diplomatic_transcription,
                grok_diplomatic_transcription=""
                if right is None
                else right.diplomatic_transcription,
            )
        )
        if left is None or right is None:
            disputes.append(
                _dispute(
                    DisputeCategory.MISSING_CONTENT,
                    page_id,
                    None,
                    "" if left is None else left.diplomatic_transcription,
                    "" if right is None else right.diplomatic_transcription,
                )
            )
            continue
        if (
            left.coverage.unresolved_text_present
            or right.coverage.unresolved_text_present
        ):
            unresolved_pages.append(page_id)
        page_alignments = align_page_regions(left, right)
        alignments.extend(page_alignments)
        disputes.extend(_compare_page(left, right, page_alignments))

    disputes.sort(
        key=lambda item: (
            DISPUTE_PRIORITY[item.category],
            item.page_id,
            item.region_id or "",
            item.dispute_id,
        )
    )
    return ComparisonFile(
        batch_id=gpt.batch_id,
        document_id=gpt.document_id,
        source_bundle_sha256=gpt.source_bundle_sha256,
        gpt_output_sha256=gpt_sha256,
        grok_output_sha256=grok_sha256,
        page_originals=originals,
        alignments=alignments,
        disputes=disputes,
        unresolved_text_page_ids=unresolved_pages,
    )


def challengeable_disputes(comparison: ComparisonFile) -> list[Dispute]:
    """Punctuation and coverage stay in the report and are not challenged."""

    return [
        item
        for item in comparison.disputes
        if item.category in CHALLENGEABLE_CATEGORIES
    ]


def _compare_page(
    left: PageTranscription,
    right: PageTranscription,
    alignments: list[RegionAlignment],
) -> list[Dispute]:
    found: list[Dispute] = []
    if not left.coverage.coverage_complete or not right.coverage.coverage_complete:
        found.append(
            _dispute(
                DisputeCategory.COVERAGE_UNCERTAINTY,
                left.page_id,
                None,
                _coverage_text(left),
                _coverage_text(right),
            )
        )
    elif (
        left.coverage.inaccessible_or_ambiguous
        or right.coverage.inaccessible_or_ambiguous
    ):
        if (
            left.coverage.inaccessible_or_ambiguous
            != right.coverage.inaccessible_or_ambiguous
        ):
            found.append(
                _dispute(
                    DisputeCategory.COVERAGE_UNCERTAINTY,
                    left.page_id,
                    None,
                    _coverage_text(left),
                    _coverage_text(right),
                )
            )
    if any(item.kind == "reordered" for item in alignments):
        found.append(
            _dispute(
                DisputeCategory.READING_ORDER,
                left.page_id,
                None,
                ",".join(_region_sequence(left)),
                ",".join(_region_sequence(right)),
                gpt_region_ids=_region_sequence(left),
                grok_region_ids=_region_sequence(right),
            )
        )
    seen_notes: set[tuple[str, str, str]] = set()
    for alignment in alignments:
        if alignment.kind == "missing":
            found.append(
                _dispute(
                    DisputeCategory.MISSING_CONTENT,
                    left.page_id,
                    alignment.gpt_region_ids[0]
                    if alignment.gpt_region_ids
                    else (
                        alignment.grok_region_ids[0]
                        if alignment.grok_region_ids
                        else None
                    ),
                    alignment.gpt_text,
                    alignment.grok_text,
                    gpt_region_ids=list(alignment.gpt_region_ids),
                    grok_region_ids=list(alignment.grok_region_ids),
                )
            )
            continue
        found.extend(_compare_span(left, right, alignment, seen_notes))
    return _collapse_complementary_page_numerals(found)


def _compare_span(
    left_page: PageTranscription,
    right_page: PageTranscription,
    alignment: RegionAlignment,
    seen_notes: set[tuple[str, str, str]],
) -> list[Dispute]:
    """Compare the lines in one alignment span, not the rest of their regions."""

    found: list[Dispute] = []
    left_text = alignment.gpt_text
    right_text = alignment.grok_text
    left_words = _words(left_text)
    right_words = _words(right_text)
    region_ids = set(alignment.gpt_region_ids) | set(alignment.grok_region_ids)
    region_id = (
        alignment.gpt_region_ids[0]
        if alignment.gpt_region_ids
        else (alignment.grok_region_ids[0] if alignment.grok_region_ids else None)
    )
    surface_only = _same_diplomatic_surface(left_text, right_text)
    substantive_span = False
    if surface_only and (
        left_words != right_words or _skeleton(left_text) != _skeleton(right_text)
    ):
        # Case, spacing, punctuation, and quote glyphs. Letters, digits, and
        # accents still differ in _diplomatic_surface, so they are not folded here.
        found.append(
            _span_dispute(
                DisputeCategory.PUNCTUATION,
                left_page,
                region_id,
                alignment,
                left_text,
                right_text,
            )
        )
    elif not surface_only and left_words != right_words:
        category = _classify_word_diff(
            left_page,
            right_page,
            _span_lines(left_text),
            _span_lines(right_text),
            left_words,
            right_words,
            region_ids,
        )
        found.append(
            _span_dispute(
                category, left_page, region_id, alignment, left_text, right_text
            )
        )
        substantive_span = category in CHALLENGEABLE_CATEGORIES
    elif not surface_only and _skeleton(left_text) != _skeleton(right_text):
        found.append(
            _span_dispute(
                DisputeCategory.PUNCTUATION,
                left_page,
                region_id,
                alignment,
                left_text,
                right_text,
            )
        )
    found.extend(
        _apparatus_disputes(
            left_page,
            right_page,
            region_id,
            region_ids,
            left_text,
            right_text,
            seen_notes,
            list(alignment.gpt_region_ids),
            list(alignment.grok_region_ids),
            record_cancellation=not substantive_span,
        )
    )
    return found


def _span_dispute(
    category: DisputeCategory,
    page: PageTranscription,
    region_id: str | None,
    alignment: RegionAlignment,
    left_text: str,
    right_text: str,
) -> Dispute:
    return _dispute(
        category,
        page.page_id,
        region_id,
        left_text,
        right_text,
        gpt_region_ids=list(alignment.gpt_region_ids),
        grok_region_ids=list(alignment.grok_region_ids),
    )


def _span_lines(text: str) -> list[LineReading]:
    if text == "":
        return []
    return [
        LineReading(line_id=f"span-{index}", text=line)
        for index, line in enumerate(text.split("\n"))
    ]


def _classify_word_diff(
    left_page: PageTranscription,
    right_page: PageTranscription,
    left_lines: list,
    right_lines: list,
    left_words: list[str],
    right_words: list[str],
    region_ids: set[str],
) -> DisputeCategory:
    if _whole_lines_missing(left_lines, right_lines):
        return DisputeCategory.MISSING_CONTENT
    differing = set(left_words).symmetric_difference(right_words)
    names = _attention_words(
        left_page,
        right_page,
        region_ids,
        {"name", "place", "organization", "event", "term"},
    )
    if differing & names:
        return DisputeCategory.NAME_OR_ENTITY
    if (
        _attention_words(left_page, right_page, region_ids, {"date", "number"})
        & differing
    ):
        return DisputeCategory.DATE_OR_NUMBER
    if any(_is_date_or_number_token(token) for token in differing):
        return DisputeCategory.DATE_OR_NUMBER
    return DisputeCategory.WORDING


def _apparatus_disputes(
    left_page: PageTranscription,
    right_page: PageTranscription,
    region_id: str | None,
    region_ids: set[str],
    left_span: str,
    right_span: str,
    seen_notes: set[tuple[str, str, str]],
    gpt_region_ids: list[str],
    grok_region_ids: list[str],
    *,
    record_cancellation: bool,
) -> list[Dispute]:
    pairs = (
        (left_page.cancellations, right_page.cancellations, True),
        (left_page.insertions, right_page.insertions, True),
        (left_page.marginal_notes, right_page.marginal_notes, False),
    )
    found: list[Dispute] = []
    for left_notes, right_notes, cancellation in pairs:
        left_text = _notes_for_span(
            left_notes, region_ids, left_span, seen_notes, "gpt"
        )
        right_text = _notes_for_span(
            right_notes, region_ids, right_span, seen_notes, "grok"
        )
        if left_text == right_text:
            continue
        if (
            cancellation
            and not record_cancellation
            and _notes_already_in_span(left_text, right_text, left_span, right_span)
        ):
            continue
        found.append(
            _dispute(
                DisputeCategory.CANCELLATION_OR_INSERTION,
                left_page.page_id,
                region_id,
                left_text,
                right_text,
                gpt_region_ids=gpt_region_ids,
                grok_region_ids=grok_region_ids,
            )
        )
    return found


def _notes_already_in_span(
    left_note: str, right_note: str, left_span: str, right_span: str
) -> bool:
    """True when each recorded note is already present in that side's span text."""

    return _note_text_in_span(left_note, left_span) and _note_text_in_span(
        right_note, right_span
    )


def _note_text_in_span(note: str, span: str) -> bool:
    if not note.strip():
        return True
    return all(_note_words_in(span, line) for line in note.split("\n") if line.strip())


def _whole_lines_missing(left_lines: list, right_lines: list) -> bool:
    left = Counter(tuple(_words(line.text)) for line in left_lines)
    right = Counter(tuple(_words(line.text)) for line in right_lines)
    if left == right:
        return False
    return _counter_subset(left, right) or _counter_subset(right, left)


def _counter_subset(
    small: Counter[tuple[str, ...]], large: Counter[tuple[str, ...]]
) -> bool:
    if not small or not large:
        return bool(small) != bool(large) and (bool(small) or bool(large))
    return all(large[key] >= count for key, count in small.items()) and small != large


def _attention_words(
    left_page: PageTranscription,
    right_page: PageTranscription,
    region_ids: set[str],
    kinds: set[str],
) -> set[str]:
    words: set[str] = set()
    for page in (left_page, right_page):
        for item in page.attention_items:
            if item.region_id in region_ids and item.kind in kinds:
                words.update(_words(item.text))
    return words


def _is_date_or_number_token(token: str) -> bool:
    lowered = token.casefold()
    return token.isdigit() or lowered in _MONTHS


def _region_sequence(page: PageTranscription) -> list[str]:
    return [
        region.region_id for region in sorted(page.regions, key=lambda item: item.order)
    ]


def _diplomatic_surface(text: str) -> str:
    """Letters and digits only, casefolded. Accents and different letters remain."""

    return "".join(character for character in text.casefold() if character.isalnum())


def _same_diplomatic_surface(left: str, right: str) -> bool:
    return _diplomatic_surface(left) == _diplomatic_surface(right)


def _collapse_complementary_page_numerals(found: list[Dispute]) -> list[Dispute]:
    """One dispute when two one-sided spans are the same page numeral.

    The original strings are kept. A missing sentence, a year, or two different
    numerals are left unchanged.
    """

    groups: dict[str, list[Dispute]] = {}
    kept: list[Dispute] = []
    for item in found:
        key = _page_numeral_key(item)
        if key is None:
            kept.append(item)
            continue
        groups.setdefault(key, []).append(item)
    for items in groups.values():
        if len(items) == 2 and _complementary_sides(items[0], items[1]):
            kept.append(_merge_page_numeral(items[0], items[1]))
        else:
            kept.extend(items)
    return kept


def _page_numeral_key(item: Dispute) -> str | None:
    if item.category is not DisputeCategory.MISSING_CONTENT:
        return None
    left = item.gpt_text.strip()
    right = item.grok_text.strip()
    if bool(left) == bool(right):
        return None
    text = left or right
    digits: list[str] = []
    for character in text:
        if character.isalpha():
            return None
        if character.isdigit():
            digits.append(character)
    if not digits or len(digits) > 3:
        return None
    return "".join(digits)


def _complementary_sides(first: Dispute, second: Dispute) -> bool:
    def gpt_only(item: Dispute) -> bool:
        return bool(item.gpt_text.strip()) and not item.grok_text.strip()

    def grok_only(item: Dispute) -> bool:
        return bool(item.grok_text.strip()) and not item.gpt_text.strip()

    return (gpt_only(first) and grok_only(second)) or (
        grok_only(first) and gpt_only(second)
    )


def _merge_page_numeral(first: Dispute, second: Dispute) -> Dispute:
    gpt = first if first.gpt_text.strip() else second
    grok = second if gpt is first else first
    if not grok.grok_text.strip():
        gpt, grok = grok, gpt
    return _dispute(
        DisputeCategory.PUNCTUATION,
        gpt.page_id,
        gpt.region_id or grok.region_id,
        gpt.gpt_text,
        grok.grok_text,
        gpt_region_ids=list(dict.fromkeys([*gpt.gpt_region_ids, *grok.gpt_region_ids])),
        grok_region_ids=list(
            dict.fromkeys([*gpt.grok_region_ids, *grok.grok_region_ids])
        ),
    )


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def _skeleton(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _notes_for_span(
    notes: list[ApparatusNote],
    region_ids: set[str],
    span: str,
    seen_notes: set[tuple[str, str, str]],
    side: str,
) -> str:
    """Keep a note on the one span that contains it, so a long region is not repeated."""

    chosen: list[str] = []
    for note in notes:
        if note.region_id not in region_ids:
            continue
        key = (side, note.region_id, note.text)
        if key in seen_notes:
            continue
        if note.text.strip() and not _note_words_in(span, note.text):
            continue
        seen_notes.add(key)
        chosen.append(note.text)
    return "\n".join(chosen)


def _note_words_in(span: str, note: str) -> bool:
    words = _words(span)
    note_words = _words(note)
    if not note_words:
        return False
    width = len(note_words)
    return any(
        words[index : index + width] == note_words
        for index in range(0, len(words) - width + 1)
    )


def _coverage_text(page: PageTranscription) -> str:
    inaccessible = ",".join(page.coverage.inaccessible_or_ambiguous)
    return f"coverage_complete={page.coverage.coverage_complete}; inaccessible={inaccessible}"


def _dispute(
    category: DisputeCategory,
    page_id: str,
    region_id: str | None,
    gpt_text: str,
    grok_text: str,
    *,
    gpt_region_ids: list[str] | None = None,
    grok_region_ids: list[str] | None = None,
) -> Dispute:
    raw = f"{category.value}|{page_id}|{region_id or ''}|{gpt_text}|{grok_text}"
    return Dispute(
        dispute_id="dsp_" + sha256_text(raw)[:16],
        category=category,
        page_id=page_id,
        region_id=region_id,
        gpt_region_ids=[] if gpt_region_ids is None else gpt_region_ids,
        grok_region_ids=[] if grok_region_ids is None else grok_region_ids,
        gpt_text=gpt_text,
        grok_text=grok_text,
        substantive=category in CHALLENGEABLE_CATEGORIES,
    )
