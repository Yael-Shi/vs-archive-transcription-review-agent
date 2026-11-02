"""Slice A3: the pure, evidence-bound reconciliation rule.

Fictional readings only. No database, network, provider, archive API, or
filesystem access: every input is built in memory.
"""

from __future__ import annotations

import itertools
import json

import pytest
from pydantic import ValidationError

from transcription_review_agent.review.adjudicate import (
    DisputeEvidenceScope,
    EvidenceBindingError,
    check_evidence_binding,
    crop_scope,
    group_scope,
    is_automatically_supported,
    reconcile_challenge,
)
from transcription_review_agent.review.models import (
    ChallengeDecision,
    ChallengeOutput,
    Dispute,
    EvidenceReference,
    ReconciledDecision,
)
from transcription_review_agent.review.runtime import ChallengeReconciledEntry

DISPUTE_ID = "d_" + "a" * 16
OTHER_DISPUTE_ID = "d_" + "b" * 16
PAGE = "page_001"
CROP = f"crop_{DISPUTE_ID}"
X_TEXT = "Ionnes  Smith"  # double space is deliberate: no normalization
Y_TEXT = "Iannes Smith"


def _dispute(dispute_id: str = DISPUTE_ID, page_id: str = PAGE) -> Dispute:
    return Dispute(
        dispute_id=dispute_id,
        category="NAME_OR_ENTITY",
        page_id=page_id,
        gpt_text=X_TEXT,
        grok_text=Y_TEXT,
        substantive=True,
    )


def _scope() -> DisputeEvidenceScope:
    return crop_scope(
        dispute_id=DISPUTE_ID, page_id=PAGE, crop_id=CROP, window_kind="TARGETED"
    )


def _ref(
    kind: str = "CROP", evidence_id: str = CROP, page_id: str = PAGE
) -> EvidenceReference:
    return EvidenceReference(kind=kind, evidence_id=evidence_id, page_id=page_id)  # type: ignore[arg-type]


def _decision(
    label: str = "CANDIDATE_X",
    assessment: str | None = "SUPPORTS_DECISION",
    *,
    third: str | None = None,
    evidence: str = "The first stroke is a single vertical.",
    ref: EvidenceReference | None | str = "default",
    dispute_id: str = DISPUTE_ID,
) -> ChallengeDecision:
    if ref == "default":
        ref = _ref() if assessment is not None else None
    return ChallengeDecision(
        dispute_id=dispute_id,
        decision=label,  # type: ignore[arg-type]
        third_reading=third,
        visual_evidence=evidence,
        evidence_ref=ref,  # type: ignore[arg-type]
        evidence_assessment=assessment,  # type: ignore[arg-type]
    )


# --- Agreement with explicit visual support -------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [("CANDIDATE_X", X_TEXT), ("CANDIDATE_Y", Y_TEXT)],
)
def test_matching_labels_with_valid_dual_support_are_automatically_supported(
    label: str, expected: str
) -> None:
    result = reconcile_challenge(
        _dispute(), _decision(label), _decision(label), _scope()
    )
    assert result.result_class == "AUTOMATICALLY_SUPPORTED"
    assert result.basis == "DUAL_MODEL_VISUAL_SUPPORT"
    assert result.decision == label
    assert result.selected_reading == expected
    assert result.agreed is True
    assert is_automatically_supported(result)


def test_candidate_x_is_the_exact_gpt_side_and_y_the_exact_grok_side() -> None:
    x = reconcile_challenge(
        _dispute(), _decision("CANDIDATE_X"), _decision("CANDIDATE_X"), _scope()
    )
    y = reconcile_challenge(
        _dispute(), _decision("CANDIDATE_Y"), _decision("CANDIDATE_Y"), _scope()
    )
    assert x.selected_reading == "Ionnes  Smith"
    assert y.selected_reading == "Iannes Smith"
    assert x.candidate_x_text == "Ionnes  Smith"
    assert x.candidate_y_text == "Iannes Smith"


def test_supported_result_is_never_verified_or_memory_or_publication_eligible() -> None:
    result = reconcile_challenge(_dispute(), _decision(), _decision(), _scope())
    dumped = result.model_dump(mode="json")
    assert dumped["independently_verified"] is False
    assert dumped["eligible_for_writer_memory"] is False
    assert dumped["eligible_for_publication"] is False
    assert dumped["agreement_is_not_verification"] is True
    assert dumped["remaining_uncertainty"] is True
    for forbidden in ("verified", "ground_truth", "confidence"):
        assert forbidden not in dumped
    # The flags cannot be set to true, by construction or by round trip.
    for flag in (
        "independently_verified",
        "eligible_for_writer_memory",
        "eligible_for_publication",
    ):
        with pytest.raises(ValidationError):
            ReconciledDecision.model_validate({**dumped, flag: True})


def test_agreement_without_support_is_not_enough() -> None:
    for first, second in itertools.product(("INSUFFICIENT", "UNREADABLE"), repeat=2):
        result = reconcile_challenge(
            _dispute(),
            _decision("CANDIDATE_X", first),
            _decision("CANDIDATE_X", second),
            _scope(),
        )
        assert result.agreed is True  # labels match ...
        assert result.result_class == "UNRESOLVED"  # ... but that is not support
        assert result.decision == "UNRESOLVED"
        assert result.selected_reading is None
        assert not is_automatically_supported(result)


# --- Insufficient / unreadable evidence -----------------------------------


@pytest.mark.parametrize("weak", ["INSUFFICIENT", "UNREADABLE"])
@pytest.mark.parametrize("which", ["gpt", "grok"])
def test_either_provider_weak_evidence_keeps_unresolved(weak: str, which: str) -> None:
    gpt = _decision("CANDIDATE_X", weak if which == "gpt" else "SUPPORTS_DECISION")
    grok = _decision("CANDIDATE_X", weak if which == "grok" else "SUPPORTS_DECISION")
    result = reconcile_challenge(_dispute(), gpt, grok, _scope())
    assert result.result_class == "UNRESOLVED"
    assert result.decision == "UNRESOLVED"
    assert result.basis == (
        "UNREADABLE_EVIDENCE" if weak == "UNREADABLE" else "INSUFFICIENT_EVIDENCE"
    )
    assert result.selected_reading is None


def test_unreadable_takes_precedence_over_insufficient_deterministically() -> None:
    result = reconcile_challenge(
        _dispute(),
        _decision("CANDIDATE_X", "INSUFFICIENT"),
        _decision("CANDIDATE_X", "UNREADABLE"),
        _scope(),
    )
    assert result.basis == "UNREADABLE_EVIDENCE"
    assert result.result_class == "UNRESOLVED"


def test_weak_evidence_retains_both_assessments_and_statements() -> None:
    result = reconcile_challenge(
        _dispute(),
        _decision("CANDIDATE_X", "INSUFFICIENT", evidence="Too faint."),
        _decision("CANDIDATE_X", "SUPPORTS_DECISION", evidence="Clear."),
        _scope(),
    )
    assert result.gpt_evidence_assessment == "INSUFFICIENT"
    assert result.grok_evidence_assessment == "SUPPORTS_DECISION"
    assert result.gpt_visual_evidence == "Too faint."
    assert result.grok_visual_evidence == "Clear."


# --- Disagreement ---------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("CANDIDATE_X", "CANDIDATE_Y"),
        ("CANDIDATE_X", "UNRESOLVED"),
        ("UNRESOLVED", "CANDIDATE_Y"),
    ],
)
def test_label_disagreement_is_unresolved_even_with_full_support(
    left: str, right: str
) -> None:
    result = reconcile_challenge(
        _dispute(), _decision(left), _decision(right), _scope()
    )
    assert result.result_class == "UNRESOLVED"
    assert result.basis == "PROVIDER_DISAGREEMENT"
    assert result.decision == "UNRESOLVED"
    assert result.agreed is False
    assert result.selected_reading is None
    assert result.gpt_decision == left
    assert result.grok_decision == right


def test_x_versus_third_reading_is_disagreement() -> None:
    result = reconcile_challenge(
        _dispute(),
        _decision("CANDIDATE_X"),
        _decision("THIRD_READING", third="Johannes"),
        _scope(),
    )
    assert result.result_class == "UNRESOLVED"
    assert result.basis == "PROVIDER_DISAGREEMENT"
    assert result.grok_third_reading == "Johannes"


def test_both_providers_unresolved_stays_unresolved_with_a_reason() -> None:
    result = reconcile_challenge(
        _dispute(),
        _decision("UNRESOLVED", "SUPPORTS_DECISION"),
        _decision("UNRESOLVED", "SUPPORTS_DECISION"),
        _scope(),
    )
    assert result.result_class == "UNRESOLVED"
    assert result.basis == "PROVIDER_UNRESOLVED"
    assert result.decision == "UNRESOLVED"


def test_full_matrix_only_one_cell_is_supported() -> None:
    labels = ("CANDIDATE_X", "CANDIDATE_Y", "UNRESOLVED")
    assessments = ("SUPPORTS_DECISION", "INSUFFICIENT", "UNREADABLE")
    supported = []
    for gl, ga, rl, ra in itertools.product(labels, assessments, labels, assessments):
        result = reconcile_challenge(
            _dispute(), _decision(gl, ga), _decision(rl, ra), _scope()
        )
        if result.result_class == "AUTOMATICALLY_SUPPORTED":
            supported.append((gl, ga, rl, ra))
            assert gl == rl != "UNRESOLVED"
            assert ga == ra == "SUPPORTS_DECISION"
        else:
            assert result.decision == "UNRESOLVED"
            assert result.selected_reading is None
    assert sorted(supported) == [
        ("CANDIDATE_X", "SUPPORTS_DECISION", "CANDIDATE_X", "SUPPORTS_DECISION"),
        ("CANDIDATE_Y", "SUPPORTS_DECISION", "CANDIDATE_Y", "SUPPORTS_DECISION"),
    ]


# --- Exact third readings -------------------------------------------------


def test_matching_exact_third_reading_with_dual_support_is_supported_but_unverified() -> (
    None
):
    result = reconcile_challenge(
        _dispute(),
        _decision("THIRD_READING", third=" Joannes\tSmith "),
        _decision("THIRD_READING", third=" Joannes\tSmith "),
        _scope(),
    )
    assert result.result_class == "AUTOMATICALLY_SUPPORTED"
    assert result.decision == "THIRD_READING"
    assert result.third_reading == " Joannes\tSmith "
    assert result.selected_reading == " Joannes\tSmith "
    assert result.independently_verified is False


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Johannes", "Joannes"),
        ("Johannes", "johannes"),  # case is not folded
        ("Johannes", "Johannes "),  # whitespace is not trimmed
        ("Johannes", "Johannes."),  # punctuation is not stripped
    ],
)
def test_differing_third_readings_are_unresolved_and_both_preserved(
    left: str, right: str
) -> None:
    result = reconcile_challenge(
        _dispute(),
        _decision("THIRD_READING", third=left),
        _decision("THIRD_READING", third=right),
        _scope(),
    )
    assert result.result_class == "UNRESOLVED"
    assert result.basis == "THIRD_READING_MISMATCH"
    assert result.decision == "UNRESOLVED"
    assert result.agreed is False
    assert result.third_reading is None
    assert result.selected_reading is None
    assert result.gpt_third_reading == left
    assert result.grok_third_reading == right


@pytest.mark.parametrize("weak", ["INSUFFICIENT", "UNREADABLE"])
def test_matching_third_reading_still_needs_support_from_both(weak: str) -> None:
    result = reconcile_challenge(
        _dispute(),
        _decision("THIRD_READING", third="Johannes"),
        _decision("THIRD_READING", weak, third="Johannes"),
        _scope(),
    )
    assert result.result_class == "UNRESOLVED"
    assert result.selected_reading is None


def test_third_reading_with_invalid_reference_is_rejected_not_unresolved() -> None:
    bad = _decision(
        "THIRD_READING", third="Johannes", ref=_ref(evidence_id="crop_other")
    )
    with pytest.raises(EvidenceBindingError) as info:
        reconcile_challenge(
            _dispute(),
            _decision("THIRD_READING", third="Johannes"),
            bad,
            _scope(),
        )
    assert info.value.code == "REFERENCE_NOT_IN_DISPUTE"
    assert info.value.provider == "grok"


# --- Evidence reference validation ----------------------------------------


def test_reference_to_another_disputes_crop_is_rejected() -> None:
    other = _ref(evidence_id=f"crop_{OTHER_DISPUTE_ID}")
    with pytest.raises(EvidenceBindingError) as info:
        reconcile_challenge(_dispute(), _decision(ref=other), _decision(), _scope())
    assert info.value.code == "REFERENCE_NOT_IN_DISPUTE"
    assert info.value.provider == "gpt"


def test_reference_on_another_page_is_rejected() -> None:
    with pytest.raises(EvidenceBindingError) as info:
        reconcile_challenge(
            _dispute(), _decision(), _decision(ref=_ref(page_id="page_002")), _scope()
        )
    assert info.value.code == "PAGE_MISMATCH"
    assert info.value.provider == "grok"


def test_wrong_kind_for_the_supplied_window_is_rejected() -> None:
    # The scope supplied a targeted crop. A group or full-page citation names
    # evidence that was not supplied for this dispute.
    for kind in ("GROUP", "FULL_PAGE_FALLBACK"):
        with pytest.raises(EvidenceBindingError) as info:
            reconcile_challenge(
                _dispute(), _decision(ref=_ref(kind=kind)), _decision(), _scope()
            )
        assert info.value.code == "REFERENCE_NOT_IN_DISPUTE"


def test_full_page_fallback_must_be_cited_as_such() -> None:
    scope = crop_scope(
        dispute_id=DISPUTE_ID,
        page_id=PAGE,
        crop_id=CROP,
        window_kind="FULL_PAGE_FALLBACK",
    )
    fallback = _ref(kind="FULL_PAGE_FALLBACK")
    result = reconcile_challenge(
        _dispute(), _decision(ref=fallback), _decision(ref=fallback), scope
    )
    assert result.result_class == "AUTOMATICALLY_SUPPORTED"
    assert result.gpt_evidence_ref == fallback
    with pytest.raises(EvidenceBindingError):
        reconcile_challenge(_dispute(), _decision(), _decision(), scope)


def test_group_reference_valid_only_for_members() -> None:
    members = {DISPUTE_ID, OTHER_DISPUTE_ID}
    scope = group_scope(
        dispute_id=DISPUTE_ID,
        page_id=PAGE,
        group_id="group_001",
        member_dispute_ids=members,
    )
    ref = _ref(kind="GROUP", evidence_id="group_001")
    result = reconcile_challenge(
        _dispute(), _decision(ref=ref), _decision(ref=ref), scope
    )
    assert result.result_class == "AUTOMATICALLY_SUPPORTED"
    with pytest.raises(EvidenceBindingError) as info:
        group_scope(
            dispute_id="d_" + "c" * 16,
            page_id=PAGE,
            group_id="group_001",
            member_dispute_ids=members,
        )
    assert info.value.code == "SCOPE_MISMATCH"
    with pytest.raises(EvidenceBindingError) as other:
        reconcile_challenge(_dispute(), _decision(), _decision(), scope)
    assert other.value.code == "REFERENCE_NOT_IN_DISPUTE"


def test_missing_reference_is_rejected_not_unresolved() -> None:
    legacy_style = _decision(assessment=None)
    assert legacy_style.evidence_ref is None
    with pytest.raises(EvidenceBindingError) as info:
        reconcile_challenge(_dispute(), _decision(), legacy_style, _scope())
    assert info.value.code == "MISSING_EVIDENCE_BINDING"
    assert info.value.provider == "grok"


def test_decision_for_a_different_dispute_is_rejected() -> None:
    with pytest.raises(EvidenceBindingError) as info:
        reconcile_challenge(
            _dispute(), _decision(dispute_id=OTHER_DISPUTE_ID), _decision(), _scope()
        )
    assert info.value.code == "DISPUTE_MISMATCH"


def test_scope_for_another_dispute_is_rejected() -> None:
    wrong = crop_scope(
        dispute_id=OTHER_DISPUTE_ID,
        page_id=PAGE,
        crop_id=f"crop_{OTHER_DISPUTE_ID}",
        window_kind="TARGETED",
    )
    with pytest.raises(EvidenceBindingError) as info:
        reconcile_challenge(_dispute(), _decision(), _decision(), wrong)
    assert info.value.code == "SCOPE_MISMATCH"
    off_page = crop_scope(
        dispute_id=DISPUTE_ID, page_id="page_002", crop_id=CROP, window_kind="TARGETED"
    )
    with pytest.raises(EvidenceBindingError):
        reconcile_challenge(_dispute(), _decision(), _decision(), off_page)


def test_unresolved_decisions_are_validated_too() -> None:
    bad = _decision("UNRESOLVED", ref=_ref(evidence_id="crop_other"))
    with pytest.raises(EvidenceBindingError):
        reconcile_challenge(_dispute(), _decision("UNRESOLVED"), bad, _scope())


def test_check_evidence_binding_returns_the_validated_reference() -> None:
    assert check_evidence_binding(_decision(), _scope(), provider="gpt") == _ref()


def test_half_bound_decision_is_a_schema_error() -> None:
    with pytest.raises(ValidationError, match="given together"):
        ChallengeDecision(
            dispute_id=DISPUTE_ID,
            decision="CANDIDATE_X",
            visual_evidence="x",
            evidence_ref=_ref(),
        )
    with pytest.raises(ValidationError, match="given together"):
        ChallengeDecision(
            dispute_id=DISPUTE_ID,
            decision="CANDIDATE_X",
            visual_evidence="x",
            evidence_assessment="SUPPORTS_DECISION",
        )


@pytest.mark.parametrize("bad", ["SUPPORTED", "CONFIDENT", "supports_decision", ""])
def test_unknown_assessment_is_a_schema_error(bad: str) -> None:
    with pytest.raises(ValidationError):
        ChallengeDecision(
            dispute_id=DISPUTE_ID,
            decision="CANDIDATE_X",
            visual_evidence="x",
            evidence_ref=_ref(),
            evidence_assessment=bad,  # type: ignore[arg-type]
        )


def test_reference_requires_known_kind_and_non_empty_ids() -> None:
    with pytest.raises(ValidationError):
        EvidenceReference(kind="PAGE", evidence_id="x", page_id=PAGE)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        EvidenceReference(kind="CROP", evidence_id="", page_id=PAGE)
    with pytest.raises(ValidationError):
        EvidenceReference(kind="CROP", evidence_id="x", page_id="")


# --- Free text is not a proxy for correctness -----------------------------


def test_free_text_evidence_does_not_change_the_outcome() -> None:
    same_prose = reconcile_challenge(
        _dispute(),
        _decision("CANDIDATE_X", evidence="Identical words."),
        _decision("CANDIDATE_Y", evidence="Identical words."),
        _scope(),
    )
    assert same_prose.result_class == "UNRESOLVED"  # equal prose, different labels

    different_prose = reconcile_challenge(
        _dispute(),
        _decision("CANDIDATE_X", evidence="Loop closes on the left."),
        _decision("CANDIDATE_X", evidence="Entirely different explanation."),
        _scope(),
    )
    assert different_prose.result_class == "AUTOMATICALLY_SUPPORTED"
    assert different_prose.remaining_uncertainty is True

    # Strong-sounding prose cannot rescue an INSUFFICIENT assessment.
    sure = reconcile_challenge(
        _dispute(),
        _decision("CANDIDATE_X", evidence="Absolutely certain, 100% confident."),
        _decision("CANDIDATE_X", "INSUFFICIENT", evidence="Absolutely certain."),
        _scope(),
    )
    assert sure.result_class == "UNRESOLVED"


# --- Preservation, purity, determinism ------------------------------------


def test_original_decisions_evidence_and_candidate_strings_are_preserved() -> None:
    gpt = _decision("CANDIDATE_X", evidence="  GPT says: loop.\nSecond line.  ")
    grok = _decision("CANDIDATE_X", evidence="Grok says: vertical.")
    result = reconcile_challenge(_dispute(), gpt, grok, _scope())
    assert result.gpt_visual_evidence == "  GPT says: loop.\nSecond line.  "
    assert result.grok_visual_evidence == "Grok says: vertical."
    assert result.gpt_decision == "CANDIDATE_X"
    assert result.grok_decision == "CANDIDATE_X"
    assert result.gpt_evidence_ref == _ref()
    assert result.grok_evidence_ref == _ref()
    assert result.candidate_x_text == X_TEXT
    assert result.candidate_y_text == Y_TEXT


def test_rule_is_pure_and_deterministic() -> None:
    dispute, gpt, grok, scope = _dispute(), _decision(), _decision(), _scope()
    before = (
        dispute.model_dump_json(),
        gpt.model_dump_json(),
        grok.model_dump_json(),
        scope,
    )
    first = reconcile_challenge(dispute, gpt, grok, scope)
    second = reconcile_challenge(dispute, gpt, grok, scope)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert (
        dispute.model_dump_json(),
        gpt.model_dump_json(),
        grok.model_dump_json(),
        scope,
    ) == before


def test_result_does_not_depend_on_provider_order_for_non_selection_fields() -> None:
    a = reconcile_challenge(
        _dispute(), _decision("CANDIDATE_X"), _decision("CANDIDATE_Y"), _scope()
    )
    b = reconcile_challenge(
        _dispute(), _decision("CANDIDATE_Y"), _decision("CANDIDATE_X"), _scope()
    )
    assert a.result_class == b.result_class == "UNRESOLVED"
    assert a.basis == b.basis


# --- Legacy compatibility -------------------------------------------------

LEGACY_CHALLENGE_DECISION = {
    "dispute_id": DISPUTE_ID,
    "decision": "CANDIDATE_X",
    "third_reading": None,
    "visual_evidence": "The crop shows the first candidate.",
}

LEGACY_RECONCILED = {
    "dispute_id": DISPUTE_ID,
    "decision": "CANDIDATE_X",
    "third_reading": None,
    "agreed": True,
    "agreement_is_not_verification": True,
    "gpt_decision": "CANDIDATE_X",
    "grok_decision": "CANDIDATE_X",
    "gpt_visual_evidence": "Left stroke is single.",
    "grok_visual_evidence": "Left stroke may be two marks.",
    "remaining_uncertainty": True,
}


def test_legacy_challenge_decision_parses_without_acquiring_binding_or_support() -> (
    None
):
    legacy = ChallengeDecision.model_validate(LEGACY_CHALLENGE_DECISION)
    assert legacy.evidence_ref is None
    assert legacy.evidence_assessment is None
    # The rule fails closed: it neither supports nor silently downgrades.
    with pytest.raises(EvidenceBindingError) as info:
        reconcile_challenge(_dispute(), legacy, legacy, _scope())
    assert info.value.code == "MISSING_EVIDENCE_BINDING"


def test_legacy_challenge_file_still_validates_and_stays_unbound() -> None:
    payload = {
        "schema_version": "review-provider-output-v1",
        "provider": "gpt",
        "provider_interface": "GPT_WORK",
        "task_id": "t",
        "job_id": "j",
        "batch_id": "batch-000001",
        "document_id": 7,
        "attempt": 1,
        "source_bundle_sha256": "0" * 64,
        "stage": "challenge",
        "round_index": 1,
        "decisions": [LEGACY_CHALLENGE_DECISION],
    }
    output = ChallengeOutput.model_validate(payload)
    assert output.decisions[0].evidence_ref is None
    assert output.decisions[0].evidence_assessment is None


def test_legacy_reconciled_decision_is_not_supported_and_not_verified() -> None:
    legacy = ReconciledDecision.model_validate(LEGACY_RECONCILED)
    assert legacy.agreed is True
    assert legacy.decision == "CANDIDATE_X"  # historical meaning is untouched
    assert legacy.result_class is None
    assert legacy.basis is None
    assert legacy.selected_reading is None
    assert not is_automatically_supported(legacy)
    assert legacy.independently_verified is False
    assert legacy.eligible_for_writer_memory is False
    assert legacy.eligible_for_publication is False
    assert legacy.agreement_is_not_verification is True
    assert legacy.gpt_evidence_ref is None
    assert legacy.grok_evidence_assessment is None


def test_legacy_ledger_event_replays_without_support_status() -> None:
    line = {
        "sequence": 9,
        "occurred_at": "2026-01-01T00:00:00Z",
        "batch_id": "batch-000001",
        "document_id": 7,
        "event_type": "challenge_reconciled",
        "round_index": 1,
        "decisions": [LEGACY_RECONCILED],
    }
    entry = ChallengeReconciledEntry.model_validate(json.loads(json.dumps(line)))
    assert len(entry.decisions) == 1
    assert entry.decisions[0].result_class is None
    assert not is_automatically_supported(entry.decisions[0])


def test_legacy_reconciled_round_trip_does_not_invent_support() -> None:
    legacy = ReconciledDecision.model_validate(LEGACY_RECONCILED)
    again = ReconciledDecision.model_validate_json(legacy.model_dump_json())
    assert again == legacy
    assert again.result_class is None
    assert not is_automatically_supported(again)
