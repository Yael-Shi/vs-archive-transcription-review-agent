"""Line alignment shared by comparison and final assembly.

Lines are taken in each provider's region order. Region identifiers and
region boundaries are labels, not the unit of alignment. A line is used
at most once. Consecutive lines are joined only when their word sequences
are the same, which covers a wrap or a split. A one-to-one substitution
is kept only when the lines are similar enough. Anything weaker stays an
unaligned span, with both texts retained.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Literal

from transcription_review_agent.bundling.hashing import sha256_text
from transcription_review_agent.review.errors import AlignmentLimitError
from transcription_review_agent.review.models import PageTranscription, RegionAlignment

# One page of lines in reading order. Above this the alignment table is
# refused before the path table can grow with the sequence length.
MAX_ALIGN_REGIONS = 48
_MAX_GROUP = 4
# Word or character similarity required before two unequal lines are paired.
# Below this the lines stay unaligned instead of being forced together.
_MATCH_SIMILARITY = 0.62

_WORD = re.compile(r"\w+", flags=re.UNICODE)

_MissingGpt = tuple[Literal["missing_gpt"], int]
_MissingGrok = tuple[Literal["missing_grok"], int]
_MatchStep = tuple[Literal["match"], int, int]
_SplitStep = tuple[Literal["split"], int, int, int]
_MergeStep = tuple[Literal["merge"], int, int, int]
_BlockStep = tuple[Literal["block"], int, int, int, int]
_Step = _MissingGpt | _MissingGrok | _MatchStep | _SplitStep | _MergeStep | _BlockStep


class _Piece:
    def __init__(self, region_id: str, text: str, index: int) -> None:
        self.region_id = region_id
        self.index = index
        self.text = text
        self.tokens = tuple(_WORD.findall(text))


def align_page_regions(
    left: PageTranscription, right: PageTranscription
) -> list[RegionAlignment]:
    """Align ``left`` (GPT) with ``right`` (Grok) without dropping or repeating a line."""

    gpt = _pieces(left)
    grok = _pieces(right)
    if len(gpt) > MAX_ALIGN_REGIONS or len(grok) > MAX_ALIGN_REGIONS:
        raise AlignmentLimitError(
            f"alignment refuses more than {MAX_ALIGN_REGIONS} lines on one page"
        )
    gpt_tokens = [piece.tokens for piece in gpt]
    grok_tokens = [piece.tokens for piece in grok]
    if (
        gpt
        and grok
        and Counter(gpt_tokens) == Counter(grok_tokens)
        and gpt_tokens != grok_tokens
    ):
        return _align_reordered(left.page_id, gpt, grok)
    return _align_monotonic(left.page_id, gpt, grok)


def _pieces(page: PageTranscription) -> list[_Piece]:
    ordered = sorted(page.regions, key=lambda region: (region.order, region.region_id))
    pieces: list[_Piece] = []
    for region in ordered:
        for line in region.lines:
            pieces.append(_Piece(region.region_id, line.text, len(pieces)))
    return pieces


def _align_reordered(
    page_id: str, gpt: list[_Piece], grok: list[_Piece]
) -> list[RegionAlignment]:
    """Match equal word sequences even when the providers ordered them differently."""

    used = [False] * len(grok)
    units: list[RegionAlignment] = []
    for piece in gpt:
        match = next(
            index
            for index, other in enumerate(grok)
            if not used[index] and other.tokens == piece.tokens
        )
        used[match] = True
        other = grok[match]
        units.append(
            _unit(
                page_id,
                "reordered" if match != piece.index else "matched",
                [piece],
                [other],
                piece.index,
            )
        )
    if len(units) != len(gpt) or not all(used):
        raise RuntimeError("reordered alignment used a line more than once")
    return units


def _align_monotonic(
    page_id: str, gpt: list[_Piece], grok: list[_Piece]
) -> list[RegionAlignment]:
    steps = _best_steps(gpt, grok)
    units: list[RegionAlignment] = []
    seen_gpt: set[int] = set()
    seen_grok: set[int] = set()
    for step in steps:
        if step[0] == "match":
            left, right = gpt[step[1]], grok[step[2]]
            _claim(seen_gpt, left.index)
            _claim(seen_grok, right.index)
            units.append(_unit(page_id, "matched", [left], [right], left.index))
        elif step[0] == "missing_gpt":
            left = gpt[step[1]]
            _claim(seen_gpt, left.index)
            units.append(_unit(page_id, "missing", [left], [], left.index))
        elif step[0] == "missing_grok":
            right = grok[step[1]]
            _claim(seen_grok, right.index)
            units.append(_unit(page_id, "missing", [], [right], right.index))
        elif step[0] == "split":
            left = gpt[step[1]]
            rights = grok[step[2] : step[2] + step[3]]
            _claim(seen_gpt, left.index)
            for right in rights:
                _claim(seen_grok, right.index)
            units.append(_unit(page_id, "split", [left], rights, left.index))
        elif step[0] == "merge":
            lefts = gpt[step[1] : step[1] + step[3]]
            right = grok[step[2]]
            for left in lefts:
                _claim(seen_gpt, left.index)
            _claim(seen_grok, right.index)
            units.append(_unit(page_id, "merged", lefts, [right], lefts[0].index))
        elif step[0] == "block":
            lefts = gpt[step[1] : step[1] + step[3]]
            rights = grok[step[2] : step[2] + step[4]]
            for left in lefts:
                _claim(seen_gpt, left.index)
            for right in rights:
                _claim(seen_grok, right.index)
            units.append(_unit(page_id, "matched", lefts, rights, lefts[0].index))
        else:
            raise RuntimeError(f"unknown alignment step {step[0]}")
    if len(seen_gpt) != len(gpt) or len(seen_grok) != len(grok):
        raise RuntimeError("alignment omitted a line")
    return units


def _best_steps(gpt: list[_Piece], grok: list[_Piece]) -> tuple[_Step, ...]:
    row_count = len(gpt) + 1
    column_count = len(grok) + 1
    scores: list[list[tuple[int, int]]] = [
        [(0, 0) for _ in range(column_count)] for _ in range(row_count)
    ]
    paths: list[list[tuple[_Step, ...]]] = [
        [() for _ in range(column_count)] for _ in range(row_count)
    ]
    for i in range(len(gpt), -1, -1):
        for j in range(len(grok), -1, -1):
            if i == len(gpt) and j == len(grok):
                continue
            options: list[tuple[tuple[int, int], tuple[_Step, ...]]] = []
            if i < len(gpt):
                missing_gpt: _MissingGpt = ("missing_gpt", i)
                options.append(
                    (
                        _add(scores[i + 1][j], _skip_cost(gpt[i]), 3),
                        (missing_gpt,) + paths[i + 1][j],
                    )
                )
            if j < len(grok):
                missing_grok: _MissingGrok = ("missing_grok", j)
                options.append(
                    (
                        _add(scores[i][j + 1], _skip_cost(grok[j]), 3),
                        (missing_grok,) + paths[i][j + 1],
                    )
                )
            if i < len(gpt) and j < len(grok):
                _add_groups(options, scores, paths, gpt, grok, i, j)
            scores[i][j], paths[i][j] = min(options)
    return paths[0][0]


def _add_groups(
    options: list[tuple[tuple[int, int], tuple[_Step, ...]]],
    scores: list[list[tuple[int, int]]],
    paths: list[list[tuple[_Step, ...]]],
    gpt: list[_Piece],
    grok: list[_Piece],
    i: int,
    j: int,
) -> None:
    max_left = min(_MAX_GROUP, len(gpt) - i)
    max_right = min(_MAX_GROUP, len(grok) - j)
    for width_left in range(1, max_left + 1):
        for width_right in range(1, max_right + 1):
            lefts = gpt[i : i + width_left]
            rights = grok[j : j + width_right]
            cost = _group_cost(lefts, rights)
            if cost is None:
                continue
            penalty = 0 if width_left == 1 and width_right == 1 else 1
            if width_left == 1 and width_right == 1:
                matched: _MatchStep = ("match", i, j)
                step: _Step = matched
            elif width_left == 1:
                split: _SplitStep = ("split", i, j, width_right)
                step = split
            elif width_right == 1:
                merged: _MergeStep = ("merge", i, j, width_left)
                step = merged
            else:
                block: _BlockStep = ("block", i, j, width_left, width_right)
                step = block
            options.append(
                (
                    _add(scores[i + width_left][j + width_right], cost, penalty),
                    (step,) + paths[i + width_left][j + width_right],
                )
            )


def _group_cost(lefts: list[_Piece], rights: list[_Piece]) -> int | None:
    left_tokens = _join([piece.tokens for piece in lefts])
    right_tokens = _join([piece.tokens for piece in rights])
    if left_tokens == right_tokens:
        return 0
    if len(lefts) != 1 or len(rights) != 1:
        return None
    similarity = _similarity(lefts[0], rights[0])
    if similarity < _MATCH_SIMILARITY:
        return None
    return int(round((1.0 - similarity) * 100)) + 1


def _similarity(left: _Piece, right: _Piece) -> float:
    word = _word_ratio(left.tokens, right.tokens)
    character = SequenceMatcher(
        a=left.text.casefold(),
        b=right.text.casefold(),
        autojunk=False,
    ).ratio()
    return max(word, character)


def _word_ratio(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if left == right:
        return 1.0
    folded_left = tuple(token.casefold() for token in left)
    folded_right = tuple(token.casefold() for token in right)
    if folded_left == folded_right:
        return 1.0
    if not folded_left or not folded_right:
        return 0.0
    return SequenceMatcher(a=folded_left, b=folded_right, autojunk=False).ratio()


def _skip_cost(piece: _Piece) -> int:
    # Higher than the cost of the weakest allowed substitution, so a similar
    # line is paired and a dissimilar line is left unaligned.
    return 50 + len(piece.tokens)


def _add(score: tuple[int, int], cost: int, penalty: int) -> tuple[int, int]:
    return score[0] + cost, score[1] + penalty


def _join(parts: Iterable[Iterable[str]]) -> tuple[str, ...]:
    joined: list[str] = []
    for part in parts:
        joined.extend(part)
    return tuple(joined)


def _claim(seen: set[int], index: int) -> None:
    if index in seen:
        raise RuntimeError("alignment repeated a line")
    seen.add(index)


def _region_ids(pieces: list[_Piece]) -> list[str]:
    ids: list[str] = []
    for piece in pieces:
        if piece.region_id not in ids:
            ids.append(piece.region_id)
    return ids


def _unit(
    page_id: str,
    kind: str,
    gpt: list[_Piece],
    grok: list[_Piece],
    sequence: int,
) -> RegionAlignment:
    gpt_ids = _region_ids(gpt)
    grok_ids = _region_ids(grok)
    gpt_text = "\n".join(piece.text for piece in gpt)
    grok_text = "\n".join(piece.text for piece in grok)
    identity = f"{page_id}|{kind}|{','.join(gpt_ids)}|{','.join(grok_ids)}|{gpt_text}|{grok_text}"
    return RegionAlignment(
        alignment_id="aln_" + sha256_text(identity)[:16],
        page_id=page_id,
        kind=kind,  # type: ignore[arg-type]
        gpt_region_ids=gpt_ids,
        grok_region_ids=grok_ids,
        gpt_text=gpt_text,
        grok_text=grok_text,
        sequence=sequence,
    )
