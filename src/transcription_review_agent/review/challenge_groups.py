"""Structural grouping of selected challenge disputes (slice A2).

Disputes that were raised for the same aligned span can share one visual
evidence window. Grouping changes only how the evidence is shown. Every
dispute keeps its own ID, its exact candidate texts, and its own decision.

A group is formed only from disputes that are already selected. It never
adds a neighbour, never crosses a page, and never compares different spans
by meaning. Two disputes share a group when all of the following hold:

* the same page;
* the same structural span (``spans.span_key``: the aligned span their GPT
  and Grok region IDs name; a page's reading-order dispute is a page-level
  span of its own);
* the same evidence window, so one crop is safe for every member. When the
  windows differ the disputes are split into separate groups.

Group IDs depend on that structure and window only. They do not depend on
which neighbours were selected, so a group keeps its ID and a dispute's
evidence reference is stable between selections. Order follows the
comparison's own dispute order, never the operator's order.

The functions are pure. They read no files, state, ledger, network,
database, provider, or archive API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from transcription_review_agent.bundling.hashing import sha256_text
from transcription_review_agent.review.adjudicate import (
    DisputeEvidenceScope,
    EvidenceBindingError,
)
from transcription_review_agent.review.models import (
    BBox,
    Dispute,
    EvidenceKind,
    RegionAlignment,
)
from transcription_review_agent.review.spans import SpanBasis, span_key

GROUP_ID_VERSION = "review-challenge-group-v1"

WindowKind = Literal["TARGETED", "FULL_PAGE_FALLBACK"]
WindowOf = Callable[[Dispute], tuple[BBox, WindowKind]]


@dataclass(frozen=True, slots=True)
class ChallengeGroup:
    """One visual evidence group and the selected disputes that share it."""

    group_id: str
    page_id: str
    basis: SpanBasis
    dispute_ids: tuple[str, ...]
    crop_id: str
    window_kind: WindowKind
    target: BBox
    alignment_id: str | None = None

    @property
    def shares_crop(self) -> bool:
        return len(self.dispute_ids) > 1

    @property
    def evidence_kind(self) -> EvidenceKind:
        """How the group's crop must be cited: a targeted crop or the labeled fallback."""

        return "CROP" if self.window_kind == "TARGETED" else "FULL_PAGE_FALLBACK"


def group_id_for(
    *,
    page_id: str,
    basis: str,
    token: str,
    window_kind: str,
    target: BBox,
) -> str:
    box = f"{target.x},{target.y},{target.width},{target.height}"
    identity = "|".join([GROUP_ID_VERSION, page_id, basis, token, window_kind, box])
    return "grp_" + sha256_text(identity)[:16]


def crop_id_for(group_id: str, dispute_ids: Sequence[str]) -> str:
    """A group of one keeps the crop name of an ungrouped dispute."""

    if len(dispute_ids) == 1:
        return f"crop_{dispute_ids[0]}"
    return f"crop_{group_id}"


def group_challenge_disputes(
    selected: Sequence[Dispute],
    alignments: Sequence[RegionAlignment],
    window_of: WindowOf,
) -> list[ChallengeGroup]:
    """Group the selected disputes. ``selected`` is in comparison order.

    ``window_of`` returns the evidence window of one dispute: its target box
    and whether it is a targeted window or the full-page fallback. Disputes
    share a group only when their windows are identical, which is what makes
    one crop safe for all of them.
    """

    ids = [item.dispute_id for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("a dispute is selected more than once")
    buckets: dict[tuple, list[Dispute]] = {}
    order: list[tuple] = []
    meta: dict[tuple, tuple[str, str, str, WindowKind, BBox]] = {}
    for dispute in selected:
        key = span_key(dispute, alignments)
        target, kind = window_of(dispute)
        bucket = (
            key.page_id,
            key.basis,
            key.token,
            kind,
            (target.x, target.y, target.width, target.height),
        )
        if bucket not in buckets:
            buckets[bucket] = []
            order.append(bucket)
            meta[bucket] = (key.page_id, key.basis, key.token, kind, target)
        buckets[bucket].append(dispute)
    groups: list[ChallengeGroup] = []
    for bucket in order:
        page_id, basis, token, kind, target = meta[bucket]
        members = tuple(item.dispute_id for item in buckets[bucket])
        group_id = group_id_for(
            page_id=page_id, basis=basis, token=token, window_kind=kind, target=target
        )
        groups.append(
            ChallengeGroup(
                group_id=group_id,
                page_id=page_id,
                basis=basis,  # type: ignore[arg-type]
                dispute_ids=members,
                crop_id=crop_id_for(group_id, members),
                window_kind=kind,
                target=target,
                alignment_id=token if basis == "ALIGNED_SPAN" else None,
            )
        )
    return groups


def required_decision_ids(groups: Sequence[ChallengeGroup]) -> list[str]:
    """Every member of every group, in group order. Grouping adds or drops nothing."""

    return [dispute_id for group in groups for dispute_id in group.dispute_ids]


def member_evidence_scope(
    group: ChallengeGroup, dispute_id: str
) -> DisputeEvidenceScope:
    """The evidence a member's decision may cite: its group, or the group's crop.

    A targeted crop is cited as ``CROP``; the labeled full-page fallback must
    be cited as ``FULL_PAGE_FALLBACK``. Citing a group does not decide the
    member's reading: each dispute is still decided on its own.
    """

    if dispute_id not in group.dispute_ids:
        raise EvidenceBindingError(
            "SCOPE_MISMATCH", "scope", "dispute is not a member of the evidence group"
        )
    return DisputeEvidenceScope(
        dispute_id,
        group.page_id,
        frozenset({("GROUP", group.group_id), (group.evidence_kind, group.crop_id)}),
    )
