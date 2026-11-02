"""Display-only formatting for a local merged transcription (slice A5).

The functions here are pure. They do not read files, state, a ledger, a
network, a database, a provider, or an archive, and they do not mutate
their inputs.

Structured comparison rows, decisions, and closure records stay
authoritative. The strings returned here are a view. They are not parsed
back into readings.

Conventions, applied only while rendering a new proposal:

- Two nonempty single-line readings: ``[reading A / reading B]``.
  Side A is the GPT reading and side B the Grok reading.
- A missing side (the empty string) is the display token ``∅``:
  ``[reading / ∅]`` or ``[∅ / reading]``. ``∅`` is not an archival character.
- A reading that contains a line break uses one labeled block. Line breaks
  inside each side are kept. The block is not flattened and is not a parser::

      [A:
      <exact side A, or ∅>
      /
      B:
      <exact side B, or ∅>]

- Cancellation and insertion text is copied as the provider wrote it.
- ``[illegible]`` is substituted only when ``unreadable_spans`` explicitly
  marks that whole side. Generic uncertainty is not illegible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from transcription_review_agent.review.models import (
    ChallengeClosureRecord,
    ComparisonFile,
    Dispute,
    FirstPassOutput,
    PageTranscription,
    ReconciledDecision,
    RecordedAlternativeReading,
)
from transcription_review_agent.review.spans import words

MISSING_SIDE = "∅"
ILLEGIBLE_DISPLAY = "[illegible]"
_ILLEGIBLE_MARKERS = frozenset({"illegible", "[illegible]"})


def format_unresolved_alternatives(
    left: str,
    right: str,
    *,
    left_illegible: bool = False,
    right_illegible: bool = False,
) -> str:
    """Format two exact readings. ``left`` is GPT and ``right`` is Grok.

    ``left_illegible`` / ``right_illegible`` must already have been decided
    from structured unreadable spans. This function does not inspect the
    words for uncertainty.
    """

    left_view = _display_side(left, left_illegible)
    right_view = _display_side(right, right_illegible)
    if "\n" in left or "\n" in right:
        return f"[A:\n{left_view}\n/\nB:\n{right_view}]"
    return f"[{left_view} / {right_view}]"


def explicit_illegible_side(
    page: PageTranscription | None,
    region_ids: Sequence[str],
    text: str,
) -> bool:
    """Whether this whole side is explicitly marked illegible.

    The mark is an ``unreadable_spans`` ``text_marker`` of ``illegible`` or
    ``[illegible]`` on one of ``region_ids``. The side text must be empty or
    exactly that marker. A longer diplomatic string is never rewritten, and
    ``coverage.unresolved_text_present`` is not illegibility.
    """

    if page is None or not region_ids or text not in {"", *_ILLEGIBLE_MARKERS}:
        return False
    ids = set(region_ids)
    return any(
        span.region_id in ids and span.text_marker in _ILLEGIBLE_MARKERS
        for span in page.unreadable_spans
    )


def decision_selects_dispute(
    decision: ReconciledDecision | None,
    dispute: Dispute,
) -> bool:
    """Whether ``decision`` may replace this dispute with one exact reading.

    ``AUTOMATICALLY_SUPPORTED`` may select only the reading A3 recorded:
    the GPT side, the Grok side, or the exact third-reading string.
    ``UNRESOLVED``, a missing decision, and a supported record whose
    ``selected_reading`` is not that string do not select. A legacy record
    (``result_class is None``) still selects from its historical label.
    """

    if decision is None or decision.decision == "UNRESOLVED":
        return False
    if decision.result_class == "UNRESOLVED":
        return False
    if decision.result_class == "AUTOMATICALLY_SUPPORTED":
        selected = decision.selected_reading
        if decision.decision == "CANDIDATE_X":
            return selected == dispute.gpt_text
        if decision.decision == "CANDIDATE_Y":
            return selected == dispute.grok_text
        if decision.decision == "THIRD_READING":
            return selected is not None and selected == decision.third_reading
        return False
    if decision.result_class is not None:
        return False
    if decision.decision == "THIRD_READING":
        return decision.third_reading is not None
    return decision.decision in {"CANDIDATE_X", "CANDIDATE_Y"}


def selected_span_text(
    decision: ReconciledDecision | None,
    dispute: Dispute,
    left: str,
    right: str,
) -> str | None:
    """The span text a selecting decision may emit, or ``None``.

    A supported X or Y emits that provider's exact aligned span (``left`` or
    ``right``), and only when ``selected_reading`` is exactly the dispute
    side A3 stored. A third reading emits that exact string. No other string
    is invented. ``None`` means the span stays an alternative.
    """

    if not decision_selects_dispute(decision, dispute):
        return None
    assert decision is not None
    if decision.decision == "THIRD_READING":
        if decision.result_class == "AUTOMATICALLY_SUPPORTED":
            return decision.selected_reading
        return decision.third_reading
    if decision.decision == "CANDIDATE_X":
        return left
    if decision.decision == "CANDIDATE_Y":
        return right
    return None


def render_aligned_span(
    left: str,
    right: str,
    disputes: Sequence[Dispute],
    decisions: Mapping[str, ReconciledDecision],
    *,
    left_illegible: bool = False,
    right_illegible: bool = False,
) -> str:
    """One aligned span: shared text once, or both sides, never a guessed winner.

    Several disputes on the same span contribute one view. Unresolved,
    unselected, failed, and conflicting decisions leave both source sides
    visible. Agreed selections collapse to the one exact selected span.
    """

    if not disputes:
        if words(left) == words(right):
            return left
        return format_unresolved_alternatives(
            left,
            right,
            left_illegible=left_illegible,
            right_illegible=right_illegible,
        )
    chosen: list[str] = []
    for dispute in disputes:
        selected = selected_span_text(
            decisions.get(dispute.dispute_id), dispute, left, right
        )
        if selected is None:
            return _unresolved_span(
                left,
                right,
                disputes,
                left_illegible=left_illegible,
                right_illegible=right_illegible,
            )
        chosen.append(selected)
    if len(set(chosen)) != 1:
        return _unresolved_span(
            left,
            right,
            disputes,
            left_illegible=left_illegible,
            right_illegible=right_illegible,
        )
    return chosen[0]


def content_is_unresolved(
    comparison: ComparisonFile | None,
    decisions: Sequence[ReconciledDecision] | None,
    challenge_closure: ChallengeClosureRecord | None = None,
) -> bool:
    """Structured unresolved content. Display markers are not consulted."""

    if challenge_closure is not None and challenge_closure.unresolved:
        return True
    if comparison is None:
        return False
    if comparison.unresolved_text_page_ids:
        return True
    by_id = {item.dispute_id: item for item in (decisions or [])}
    return any(
        dispute.substantive
        and not decision_selects_dispute(by_id.get(dispute.dispute_id), dispute)
        for dispute in comparison.disputes
    )


def recorded_alternative_readings(
    gpt: FirstPassOutput | None,
    grok: FirstPassOutput | None,
) -> list[RecordedAlternativeReading]:
    """Copy first-pass alternatives in provider and page order.

    The copy does not sort, drop, or choose a reading. Diplomatic text is
    left to the formatter, which does not read this list.
    """

    found: list[RecordedAlternativeReading] = []
    for provider, output in (("gpt", gpt), ("grok", grok)):
        if output is None:
            continue
        for page in output.pages:
            for item in page.alternative_readings:
                found.append(
                    RecordedAlternativeReading(
                        provider=provider,  # type: ignore[arg-type]
                        page_id=page.page_id,
                        region_id=item.region_id,
                        readings=list(item.readings),
                    )
                )
    return found


def _display_side(text: str, illegible: bool) -> str:
    if illegible:
        return ILLEGIBLE_DISPLAY
    if text == "":
        return MISSING_SIDE
    return text


def _unresolved_span(
    left: str,
    right: str,
    disputes: Sequence[Dispute],
    *,
    left_illegible: bool,
    right_illegible: bool,
) -> str:
    """Show the aligned source once, then any dispute text it does not already show."""

    alignment = (left, right)
    extras: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    show_alignment = left != right
    for dispute in disputes:
        pair = (dispute.gpt_text, dispute.grok_text)
        if pair == alignment:
            show_alignment = True
            continue
        if pair in seen or _contained_in_differing_alignment(pair, alignment):
            continue
        seen.add(pair)
        extras.append(pair)
    parts: list[str] = []
    if left == right and left != "":
        parts.append(left)
        show_alignment = False
    show_empty_alignment = not (left == "" and right == "" and extras)
    if (show_alignment or not parts) and show_empty_alignment:
        parts.append(
            format_unresolved_alternatives(
                left,
                right,
                left_illegible=left_illegible,
                right_illegible=right_illegible,
            )
        )
    for extra_left, extra_right in extras:
        parts.append(format_unresolved_alternatives(extra_left, extra_right))
    return "\n".join(parts)


def _contained_in_differing_alignment(
    pair: tuple[str, str],
    alignment: tuple[str, str],
) -> bool:
    """A note already visible inside two different aligned sides is not repeated."""

    left, right = pair
    aligned_left, aligned_right = alignment
    if aligned_left == aligned_right or left == "" or right == "":
        return False
    if pair == alignment:
        return True
    return left in aligned_left and right in aligned_right
