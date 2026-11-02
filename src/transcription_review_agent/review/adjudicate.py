"""Evidence-bound automatic adjudication of one challenged dispute (slice A3).

This module is pure and deterministic. It reads no files, state, ledger,
network, database, provider, or archive API, and it never mutates its inputs.

The rule turns two provider decisions for one dispute into one
``ReconciledDecision``. The only possible outcomes are:

``AUTOMATICALLY_SUPPORTED``
    Both providers gave the same label (and, for ``THIRD_READING``, the exact
    same third-reading string), both cite a visual evidence reference that
    belongs to this dispute, and both state ``SUPPORTS_DECISION``.

``UNRESOLVED``
    Anything else that is still well formed: different labels, a provider
    that is itself unresolved, ``INSUFFICIENT`` or ``UNREADABLE`` evidence, or
    differing third readings.

Automatically supported is not independently verified. It means that two
model outputs met a reproducible rule. It is never ground truth, never
verified, and never eligible for writer memory or publication; the result
type pins those flags to ``False``.

Malformed bindings are different from weak evidence. A missing or invalid
evidence reference raises ``EvidenceBindingError`` instead of producing a
quiet ``UNRESOLVED``, so a caller can reject the provider file.

Nothing here compares free-text evidence descriptions, confidence, or
majorities. ``visual_evidence`` prose is preserved verbatim and is not an input
to the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transcription_review_agent.review.models import (
    ChallengeDecision,
    Dispute,
    EvidenceKind,
    EvidenceReference,
    ReconciledDecision,
    ReconciliationBasis,
)

BindingErrorCode = Literal[
    "MISSING_EVIDENCE_BINDING",
    "DISPUTE_MISMATCH",
    "PAGE_MISMATCH",
    "REFERENCE_NOT_IN_DISPUTE",
    "SCOPE_MISMATCH",
]


class EvidenceBindingError(ValueError):
    """A decision's evidence binding is absent or does not belong to the dispute.

    ``MISSING_EVIDENCE_BINDING`` also covers legacy decisions that predate
    evidence binding. Their safe interpretation is ambiguous, so they fail
    closed: they cannot be classified as automatically supported.
    """

    def __init__(self, code: BindingErrorCode, provider: str, detail: str) -> None:
        super().__init__(f"{provider}: {code}: {detail}")
        self.code: BindingErrorCode = code
        self.provider = provider
        self.detail = detail


@dataclass(frozen=True)
class DisputeEvidenceScope:
    """The visual evidence that was actually supplied for one dispute.

    ``allowed`` holds ``(kind, evidence_id)`` pairs. A reference is valid only
    when it names one of them on the dispute's own page.
    """

    dispute_id: str
    page_id: str
    allowed: frozenset[tuple[str, str]]


def crop_scope(
    *,
    dispute_id: str,
    page_id: str,
    crop_id: str,
    window_kind: Literal["TARGETED", "FULL_PAGE_FALLBACK"],
) -> DisputeEvidenceScope:
    """Scope for the single crop supplied for ``dispute_id``.

    A targeted window is cited as ``CROP``. The labeled full-page fallback
    window is cited as ``FULL_PAGE_FALLBACK``; it is not a targeted crop and
    cannot be cited as one.
    """

    kind: EvidenceKind = "CROP" if window_kind == "TARGETED" else "FULL_PAGE_FALLBACK"
    return DisputeEvidenceScope(dispute_id, page_id, frozenset({(kind, crop_id)}))


def group_scope(
    *,
    dispute_id: str,
    page_id: str,
    group_id: str,
    member_dispute_ids: frozenset[str] | set[str] | tuple[str, ...] | list[str],
) -> DisputeEvidenceScope:
    """Scope for a visual-review group. The dispute must be a member."""

    if dispute_id not in set(member_dispute_ids):
        raise EvidenceBindingError(
            "SCOPE_MISMATCH", "scope", "dispute is not a member of the evidence group"
        )
    return DisputeEvidenceScope(dispute_id, page_id, frozenset({("GROUP", group_id)}))


def check_evidence_binding(
    decision: ChallengeDecision,
    scope: DisputeEvidenceScope,
    *,
    provider: str,
) -> EvidenceReference:
    """Return the validated reference or raise ``EvidenceBindingError``.

    Applies to every decision, including ``UNRESOLVED`` ones: a reference that
    names evidence of another dispute is invalid whatever it is attached to.
    """

    if decision.dispute_id != scope.dispute_id:
        raise EvidenceBindingError(
            "DISPUTE_MISMATCH", provider, "decision is for a different dispute"
        )
    ref = decision.evidence_ref
    if ref is None or decision.evidence_assessment is None:
        raise EvidenceBindingError(
            "MISSING_EVIDENCE_BINDING",
            provider,
            "decision names no examined visual evidence and no assessment",
        )
    if ref.page_id != scope.page_id:
        raise EvidenceBindingError(
            "PAGE_MISMATCH", provider, "evidence reference is on a different page"
        )
    if (ref.kind, ref.evidence_id) not in scope.allowed:
        raise EvidenceBindingError(
            "REFERENCE_NOT_IN_DISPUTE",
            provider,
            "evidence reference was not supplied for this dispute",
        )
    return ref


def is_automatically_supported(decision: ReconciledDecision) -> bool:
    """True only for records that the A3 rule itself classified as supported.

    Legacy records (``result_class is None``) are never supported, even when
    they carry a matching label.
    """

    return decision.result_class == "AUTOMATICALLY_SUPPORTED"


def reconcile_challenge(
    dispute: Dispute,
    gpt: ChallengeDecision,
    grok: ChallengeDecision,
    scope: DisputeEvidenceScope,
) -> ReconciledDecision:
    """Reconcile two provider decisions for ``dispute``.

    Candidate X is the GPT side and candidate Y the Grok side of the dispute,
    exactly as the anonymous challenge input presented them. Strings are
    compared and returned unchanged; there is no trimming or normalization.
    """

    if scope.dispute_id != dispute.dispute_id or scope.page_id != dispute.page_id:
        raise EvidenceBindingError(
            "SCOPE_MISMATCH", "scope", "evidence scope is not for this dispute"
        )
    gpt_ref = check_evidence_binding(gpt, scope, provider="gpt")
    grok_ref = check_evidence_binding(grok, scope, provider="grok")
    assert gpt.evidence_assessment is not None and grok.evidence_assessment is not None

    labels_match = gpt.decision == grok.decision
    thirds_match = gpt.third_reading == grok.third_reading
    agreed = labels_match and thirds_match

    basis: ReconciliationBasis
    if not labels_match:
        basis = "PROVIDER_DISAGREEMENT"
    elif gpt.decision == "UNRESOLVED":
        basis = "PROVIDER_UNRESOLVED"
    elif "UNREADABLE" in (gpt.evidence_assessment, grok.evidence_assessment):
        basis = "UNREADABLE_EVIDENCE"
    elif "INSUFFICIENT" in (gpt.evidence_assessment, grok.evidence_assessment):
        basis = "INSUFFICIENT_EVIDENCE"
    elif not thirds_match:
        basis = "THIRD_READING_MISMATCH"
    else:
        basis = "DUAL_MODEL_VISUAL_SUPPORT"

    supported = basis == "DUAL_MODEL_VISUAL_SUPPORT"
    selected: str | None = None
    if supported:
        if gpt.decision == "CANDIDATE_X":
            selected = dispute.gpt_text
        elif gpt.decision == "CANDIDATE_Y":
            selected = dispute.grok_text
        else:
            selected = gpt.third_reading

    return ReconciledDecision(
        dispute_id=dispute.dispute_id,
        decision=gpt.decision if supported else "UNRESOLVED",
        third_reading=gpt.third_reading
        if supported and gpt.decision == "THIRD_READING"
        else None,
        agreed=agreed,
        gpt_decision=gpt.decision,
        grok_decision=grok.decision,
        gpt_visual_evidence=gpt.visual_evidence,
        grok_visual_evidence=grok.visual_evidence,
        # Automatic support never removes uncertainty.
        remaining_uncertainty=True,
        result_class="AUTOMATICALLY_SUPPORTED" if supported else "UNRESOLVED",
        basis=basis,
        selected_reading=selected,
        candidate_x_text=dispute.gpt_text,
        candidate_y_text=dispute.grok_text,
        gpt_third_reading=gpt.third_reading,
        grok_third_reading=grok.third_reading,
        gpt_evidence_ref=gpt_ref,
        grok_evidence_ref=grok_ref,
        gpt_evidence_assessment=gpt.evidence_assessment,
        grok_evidence_assessment=grok.evidence_assessment,
    )
