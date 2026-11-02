"""Structural links between disputes and aligned spans (shared helper).

One place decides whether a dispute belongs to an aligned span. Final
assembly and challenge grouping (slice A2) both use it, so the two can
never disagree about which span a dispute came from.

Linking is structural: it reads only the page and the GPT/Grok region IDs
that the comparison already recorded, plus the exact text that the
alignment recorded for the same span. Nothing here measures similarity
between different spans, and nothing crosses a page.

The functions are pure. They read no files, state, ledger, network,
database, provider, or archive API.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from transcription_review_agent.review.models import (
    Dispute,
    DisputeCategory,
    RegionAlignment,
)

_WORD = re.compile(r"\w+", flags=re.UNICODE)

SpanBasis = Literal["ALIGNED_SPAN", "PAGE_READING_ORDER", "UNLINKED"]


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def note_words_in(span: str, note: str) -> bool:
    """True when the note's words appear in order inside the span's words."""

    span_words = words(span)
    note_tokens = words(note)
    if not note_tokens:
        return False
    width = len(note_tokens)
    return any(
        span_words[index : index + width] == note_tokens
        for index in range(0, len(span_words) - width + 1)
    )


def regions_equal(dispute: Dispute, alignment: RegionAlignment) -> bool:
    """The dispute names exactly the GPT and Grok region IDs of the alignment."""

    return list(dispute.gpt_region_ids) == list(alignment.gpt_region_ids) and list(
        dispute.grok_region_ids
    ) == list(alignment.grok_region_ids)


def same_span(dispute: Dispute, alignment: RegionAlignment) -> bool:
    """Whether ``dispute`` was raised for ``alignment``.

    This is the rule final assembly has always used. It is the only
    definition of the link; callers must not restate it.
    """

    equal = regions_equal(dispute, alignment)
    if (
        equal
        and dispute.gpt_text == alignment.gpt_text
        and dispute.grok_text == alignment.grok_text
    ):
        return True
    if (
        equal
        and dispute.category.value == "CANCELLATION_OR_INSERTION"
        and (
            note_words_in(alignment.gpt_text, dispute.gpt_text)
            or note_words_in(alignment.grok_text, dispute.grok_text)
        )
    ):
        return True
    if dispute.gpt_region_ids or dispute.grok_region_ids:
        return False
    return (
        dispute.region_id is not None and dispute.region_id in alignment.gpt_region_ids
    )


def link_alignment(
    dispute: Dispute,
    alignments: Sequence[RegionAlignment],
) -> RegionAlignment | None:
    """The one aligned span of the dispute's page that the dispute belongs to.

    Region IDs are matched exactly. A dispute that carries no region-ID lists
    (a record written before they existed) is matched through its single
    ``region_id`` instead. When several spans of the page still qualify, the
    exact span text decides; if that does not leave exactly one span, the
    dispute is not linked. A dispute is never linked by similarity.
    """

    if dispute.category is DisputeCategory.READING_ORDER:
        return None
    on_page = [item for item in alignments if item.page_id == dispute.page_id]
    if dispute.gpt_region_ids or dispute.grok_region_ids:
        candidates = [item for item in on_page if regions_equal(dispute, item)]
    elif dispute.region_id is not None:
        candidates = [
            item for item in on_page if dispute.region_id in item.gpt_region_ids
        ]
    else:
        candidates = []
    if len(candidates) > 1:
        candidates = [item for item in candidates if same_span(dispute, item)]
    return candidates[0] if len(candidates) == 1 else None


@dataclass(frozen=True, slots=True)
class SpanKey:
    """The structural identity a dispute is grouped by.

    ``token`` is the aligned span's ID, the fixed marker for a page's
    reading-order dispute, or the dispute's own ID when it cannot be linked.
    """

    page_id: str
    basis: SpanBasis
    token: str


def span_key(dispute: Dispute, alignments: Sequence[RegionAlignment]) -> SpanKey:
    if dispute.category is DisputeCategory.READING_ORDER:
        return SpanKey(dispute.page_id, "PAGE_READING_ORDER", "page-reading-order")
    linked = link_alignment(dispute, alignments)
    if linked is None:
        return SpanKey(dispute.page_id, "UNLINKED", dispute.dispute_id)
    return SpanKey(dispute.page_id, "ALIGNED_SPAN", linked.alignment_id)
