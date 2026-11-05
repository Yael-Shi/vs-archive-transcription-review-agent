"""Display-only formatting for a local merged transcription (slice A5).

The functions here are pure. They do not read files, state, a ledger, a
network, a database, a provider, or an archive, and they do not mutate
their inputs.

Structured comparison rows, decisions, and closure records stay
authoritative. The strings returned here are a view. They are not parsed
back into readings.

Conventions, applied only while rendering a new proposal:

- A single-line disagreement keeps text that is the same on both sides
  once, and wraps only the spans that differ: ``and [cannot / canot]
  accept presents.`` Side A is the GPT reading and side B the Grok reading.
  Nothing in that view selects a winner.
- Shared tokens are whitespace runs and other runs that do not split a
  diplomatic annotation. ``[deleted: ...]``, ``[cancelled: ...]``,
  ``[inserted: ...]``, and ``[unclear: ...]`` stay whole, including a
  nested annotation inside them.
- The minimized view is refused when it would nest brackets, put ``/``
  inside an alternative, split an annotation, or show the same words once
  after they moved. The line then stays ``[reading A / reading B]``.
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
from difflib import SequenceMatcher

from transcription_review_agent.review.models import (
    ChallengeClosureRecord,
    ComparisonFile,
    Dispute,
    FirstPassOutput,
    PageTranscription,
    ReconciledDecision,
    RecordedAlternativeReading,
)
from transcription_review_agent.review.spans import note_words_in, words

MISSING_SIDE = "∅"
ILLEGIBLE_DISPLAY = "[illegible]"
_ILLEGIBLE_MARKERS = frozenset({"illegible", "[illegible]"})
_ANNOTATION_MARKERS = ("[deleted:", "[cancelled:", "[inserted:", "[unclear:")
_UNSAFE_ALTERNATIVE_CHARS = frozenset("[]/")


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
    minimized = _minimize_line_alternatives(left_view, right_view)
    if minimized is None:
        return f"[{left_view} / {right_view}]"
    return minimized


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

    A supported X, Y, or third reading replaces the aligned line only when
    the dispute text is that entire line on both sides. A fragment such as
    ``school`` does not stand in for the rest of the line. A full-line third
    reading emits that exact string. No other string is invented. ``None``
    means both line readings stay visible.
    """

    if not decision_selects_dispute(decision, dispute):
        return None
    assert decision is not None
    covers_line = dispute.gpt_text == left and dispute.grok_text == right
    if not covers_line:
        return None
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


def _minimize_line_alternatives(left: str, right: str) -> str | None:
    """Shared tokens once, or ``None`` when the full-line form is safer.

    ``None`` keeps the caller on ``[left / right]``. A minimized string
    still contains both readings and does not prefer either one.
    """

    if left == right or "\n" in left or "\n" in right:
        return None
    left_tokens = _tokenize_line(left)
    right_tokens = _tokenize_line(right)
    if left_tokens is None or right_tokens is None:
        return None
    spans = _changed_spans(left_tokens, right_tokens)
    if not any(kind == "equal" for kind, _, _ in spans):
        return None
    if not any(kind == "change" for kind, _, _ in spans):
        return None
    if _change_is_a_move(spans):
        return None
    parts: list[str] = []
    rebuilt_left: list[str] = []
    rebuilt_right: list[str] = []
    for kind, left_hunk, right_hunk in spans:
        if kind == "equal":
            parts.append(left_hunk)
            rebuilt_left.append(left_hunk)
            rebuilt_right.append(left_hunk)
            continue
        if _alternative_hunk_unsafe(left_hunk, right_hunk):
            return None
        parts.append(
            f"[{_display_side(left_hunk, False)} / {_display_side(right_hunk, False)}]"
        )
        rebuilt_left.append(left_hunk)
        rebuilt_right.append(right_hunk)
    if "".join(rebuilt_left) != left or "".join(rebuilt_right) != right:
        return None
    return "".join(parts)


def _tokenize_line(text: str) -> list[str] | None:
    """Whitespace runs, annotation atoms, and the remaining exact runs.

    ``None`` means an annotation marker was opened and never closed.
    """

    tokens: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if _starts_annotation(text, index):
            end = _annotation_end(text, index)
            if end is None:
                return None
            tokens.append(text[index:end])
            index = end
            continue
        if text[index].isspace():
            next_index = index + 1
            while next_index < length and text[next_index].isspace():
                next_index += 1
            tokens.append(text[index:next_index])
            index = next_index
            continue
        next_index = index + 1
        while next_index < length and not text[next_index].isspace():
            if _starts_annotation(text, next_index):
                break
            next_index += 1
        tokens.append(text[index:next_index])
        index = next_index
    return tokens


def _starts_annotation(text: str, index: int) -> bool:
    return any(text.startswith(marker, index) for marker in _ANNOTATION_MARKERS)


def _annotation_end(text: str, start: int) -> int | None:
    """Index after a balanced annotation, or ``None`` when it does not close."""

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _changed_spans(
    left_tokens: list[str], right_tokens: list[str]
) -> list[tuple[str, str, str]]:
    """Equal runs and contiguous changes. A change stores both exact hunks."""

    matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)
    spans: list[tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            spans.append(("equal", "".join(left_tokens[i1:i2]), ""))
            continue
        left_hunk = "".join(left_tokens[i1:i2])
        right_hunk = "".join(right_tokens[j1:j2])
        if spans and spans[-1][0] == "change":
            _, previous_left, previous_right = spans[-1]
            spans[-1] = (
                "change",
                previous_left + left_hunk,
                previous_right + right_hunk,
            )
        else:
            spans.append(("change", left_hunk, right_hunk))
    return spans


def _alternative_hunk_unsafe(left: str, right: str) -> bool:
    """Brackets or a slash inside a span would make ``[A / B]`` ambiguous."""

    return any(char in _UNSAFE_ALTERNATIVE_CHARS for char in left + right)


def _change_is_a_move(spans: Sequence[tuple[str, str, str]]) -> bool:
    """An insertion and a deletion of the same words is not a local edit.

    Showing the shared word once would hide that the providers ordered it
    differently. The full line stays visible instead.
    """

    inserted = [
        right
        for kind, left, right in spans
        if kind == "change" and left == "" and right != ""
    ]
    deleted = [
        left
        for kind, left, right in spans
        if kind == "change" and right == "" and left != ""
    ]
    if not inserted or not deleted:
        return False
    if set(words(" ".join(inserted))) & set(words(" ".join(deleted))):
        return True
    inserted_text = "".join(inserted).strip()
    deleted_text = "".join(deleted).strip()
    return inserted_text != "" and inserted_text == deleted_text


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
    """A note already visible in the aligned text is not printed again.

    A two-sided note is skipped when each side already sits inside a
    differing aligned line. A one-sided note is skipped when its text is
    already on the corresponding aligned side, including when both aligned
    sides are the same line.
    """

    left, right = pair
    aligned_left, aligned_right = alignment
    if pair == alignment:
        return True
    if left == "" and right == "":
        return False
    if left == "" or right == "":
        note, side = (left, aligned_left) if right == "" else (right, aligned_right)
        return _note_already_in_side(note, side)
    if aligned_left == aligned_right:
        return False
    return left in aligned_left and right in aligned_right


def _note_already_in_side(note: str, side: str) -> bool:
    """Whether ``note`` is the side or its words already occur there in order."""

    if note == side:
        return True
    return note_words_in(side, note)
