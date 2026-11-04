"""Provider output contract shared by task generation and validation.

The example in a task bundle is a shape template. Identity values are copied
from the published work request. Decision text in a challenge example is not a
reading and is not a provider result.
"""

from __future__ import annotations

import json
from typing import Final, Literal

from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.review.adjudicate import (
    DisputeEvidenceScope,
    check_evidence_binding,
)
from transcription_review_agent.review.models import (
    AlternativeReading,
    ApparatusNote,
    AttentionItem,
    ChallengeDecision,
    ChallengeOutput,
    CoverageReport,
    EvidenceReference,
    FirstPassOutput,
    Identification,
    LineReading,
    PageTranscription,
    ResearchOutput,
    ResearchSource,
    TextRegion,
    UnreadableSpan,
    WorkRequest,
)

TEMPLATE_VISUAL_EVIDENCE: Final = (
    "TEMPLATE ONLY. Replace this sentence with the visual evidence you examined."
)
TEMPLATE_PAGE_TEXT: Final = "TEMPLATE ONLY. Replace this page with your transcription."
TEMPLATE_MARK: Final = "TEMPLATE ONLY"
EXACT_TEMPLATE_PLACEHOLDERS: Final = frozenset({"TEMPLATE", "TEMPLATE ALTERNATIVE"})

# Byte-stable copies of the challenge contract published before identity fields
# and output examples were required. Accepting an in-flight task still
# recognizes these texts. New tasks are not written with them.
LEGACY_CHALLENGE_INSTRUCTIONS: Final = (
    "Inspect the page image and the crop window. The crop is existing pixels,\n"
    "not new detail. FULL_PAGE_FALLBACK means no region box was usable,\n"
    "so the window is the whole page and is not a targeted crop.\n"
    "Candidate readings are anonymous.\n"
    "Choose CANDIDATE_X, CANDIDATE_Y, THIRD_READING, or UNRESOLVED.\n"
    "Cite visual evidence. Do not change a reading only to force agreement.\n"
    "Write one JSON file. Dispatch is manual. No API is assumed.\n"
)
LEGACY_EVIDENCE_INSTRUCTIONS: Final = (
    LEGACY_CHALLENGE_INSTRUCTIONS
    + "For every decision, provide evidence_ref and evidence_assessment.\n"
    + "evidence_ref must exactly copy one supplied reference for that dispute's\n"
    + "group/crop/full-page fallback, including kind, evidence_id, and page_id.\n"
    + "evidence_assessment is SUPPORTS_DECISION, INSUFFICIENT, or UNREADABLE.\n"
    + "Use INSUFFICIENT or UNREADABLE when the cited evidence does not support\n"
    + "your selected reading; matching another decision is not visual support.\n"
)

IDENTITY_INSTRUCTION_NOTE: Final = (
    "Copy schema_version, provider, provider_interface, task_id, job_id,\n"
    "batch_id, document_id, attempt, source_bundle_sha256, and stage from\n"
    "work_request.json. output_example.json already contains those values.\n"
    "provider_interface is copied from work_request.json. It is GPT_WORK or GROK.\n"
    "Do not omit provider_interface, job_id, batch_id, document_id, attempt,\n"
    "or source_bundle_sha256. A missing field is invalid output.\n"
    "output_example.json is a shape template. Replace its template text with\n"
    "your own reading. Do not submit the template unchanged.\n"
)
CHALLENGE_IDENTITY_NOTE: Final = (
    "Also copy round_index from work_request.json.\n"
    "Return one decision for every required dispute id, and no other dispute.\n"
    "For EVIDENCE_BOUND_V1, every decision includes evidence_ref and\n"
    "evidence_assessment. Copy evidence_ref from that dispute's supplied\n"
    "reference. Do not omit it.\n"
)


def template_placeholder_problem(payload: object) -> str | None:
    """Return an error when a provider file still contains the task template.

    ``UNRESOLVED``, ``INSUFFICIENT``, and an illegible marker are real
    decisions. Only the placeholder strings published in ``output_example.json``
    are rejected.
    """

    if _contains_template_placeholder(payload):
        return "provider output still contains the task template"
    return None


def _contains_template_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return value in EXACT_TEMPLATE_PLACEHOLDERS or TEMPLATE_MARK in value
    if isinstance(value, dict):
        return any(_contains_template_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_template_placeholder(item) for item in value)
    return False


def identity_from_request(request: WorkRequest) -> dict[str, object]:
    """Identity fields a provider file must copy from its work request."""

    return {
        "schema_version": "review-provider-output-v1",
        "provider": request.provider,
        "provider_interface": request.provider_interface,
        "task_id": request.task_id,
        "job_id": request.job_id,
        "batch_id": request.batch_id,
        "document_id": request.document_id,
        "attempt": request.attempt,
        "source_bundle_sha256": request.source_bundle_sha256,
        "stage": request.stage,
    }


def challenge_instructions(*, evidence_bound: bool) -> str:
    """Current challenge instructions. The legacy prefix stays first."""

    text = LEGACY_CHALLENGE_INSTRUCTIONS + IDENTITY_INSTRUCTION_NOTE
    text += CHALLENGE_IDENTITY_NOTE
    if not evidence_bound:
        return text
    return (
        text
        + "For every decision, provide evidence_ref and evidence_assessment.\n"
        + "evidence_ref must exactly copy one supplied reference for that dispute's\n"
        + "group/crop/full-page fallback, including kind, evidence_id, and page_id.\n"
        + "evidence_assessment is SUPPORTS_DECISION, INSUFFICIENT, or UNREADABLE.\n"
        + "Use INSUFFICIENT or UNREADABLE when the cited evidence does not support\n"
        + "your selected reading; matching another decision is not visual support.\n"
    )


def legacy_challenge_instructions(*, evidence_bound: bool) -> str:
    if evidence_bound:
        return LEGACY_EVIDENCE_INSTRUCTIONS
    return LEGACY_CHALLENGE_INSTRUCTIONS


def challenge_schema_text(*, evidence_bound: bool) -> str:
    decision_fields: dict[str, object] = {
        "dispute_id": "exact required decision ID",
        "decision": [
            "CANDIDATE_X",
            "CANDIDATE_Y",
            "THIRD_READING",
            "UNRESOLVED",
        ],
        "third_reading": "required only for THIRD_READING; otherwise null",
        "visual_evidence": "non-empty evidence statement",
    }
    document: dict[str, object] = {
        "schema_version": "review-provider-output-v1",
        "stage": "challenge",
        "note": (
            "Return one JSON object and no other fields. "
            "Copy every identity field from work_request.json. "
            "output_example.json is a shape template with those values already copied. "
            "Replace its decisions with your own visual decisions. "
            "Do not include commands."
        ),
        "required_top_level_fields": {
            "schema_version": "review-provider-output-v1",
            "provider": "gpt or grok, copied from work_request.json",
            "provider_interface": (
                "GPT_WORK when provider is gpt; GROK when provider is grok; "
                "copy from work_request.json"
            ),
            "task_id": "copy from work_request.json",
            "job_id": "copy from work_request.json",
            "batch_id": "copy from work_request.json",
            "document_id": "copy from work_request.json",
            "attempt": "copy from work_request.json",
            "source_bundle_sha256": "copy from work_request.json",
            "stage": "challenge",
            "round_index": "copy from work_request.json",
            "decisions": "one object for every required dispute id and no others",
        },
        "decisions": {
            "required_for_every_selected_dispute": True,
            "fields": decision_fields,
        },
    }
    if evidence_bound:
        document["round_contract"] = "EVIDENCE_BOUND_V1"
        decision_fields["evidence_ref"] = {
            "kind": ["GROUP", "CROP", "FULL_PAGE_FALLBACK"],
            "evidence_id": "exactly as published for this dispute",
            "page_id": "exactly as published for this dispute",
        }
        decision_fields["evidence_assessment"] = [
            "SUPPORTS_DECISION",
            "INSUFFICIENT",
            "UNREADABLE",
        ]
        document["evidence_binding"] = (
            "EVIDENCE_BOUND_V1 requires evidence_ref and evidence_assessment "
            "on every decision. evidence_ref must copy kind, evidence_id, and "
            "page_id from one reference supplied for that dispute."
        )
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def legacy_challenge_schema_text(*, evidence_bound: bool) -> str:
    if evidence_bound:
        return (
            json.dumps(
                {
                    "schema_version": "review-provider-output-v1",
                    "stage": "challenge",
                    "round_contract": "EVIDENCE_BOUND_V1",
                    "note": "Return data only. Do not include commands.",
                    "decisions": {
                        "required_for_every_selected_dispute": True,
                        "fields": {
                            "dispute_id": "exact required decision ID",
                            "decision": [
                                "CANDIDATE_X",
                                "CANDIDATE_Y",
                                "THIRD_READING",
                                "UNRESOLVED",
                            ],
                            "third_reading": (
                                "required only for THIRD_READING; preserve exact text"
                            ),
                            "visual_evidence": "non-empty evidence statement",
                            "evidence_ref": {
                                "kind": ["GROUP", "CROP", "FULL_PAGE_FALLBACK"],
                                "evidence_id": "exactly as published for this dispute",
                                "page_id": "exactly as published for this dispute",
                            },
                            "evidence_assessment": [
                                "SUPPORTS_DECISION",
                                "INSUFFICIENT",
                                "UNREADABLE",
                            ],
                        },
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return (
        json.dumps(
            {
                "schema_version": "review-provider-output-v1",
                "stage": "challenge",
                "note": "Return data only. Do not include commands.",
            },
            indent=2,
        )
        + "\n"
    )


def recognized_challenge_schema_texts(*, evidence_bound: bool) -> frozenset[str]:
    return frozenset(
        {
            challenge_schema_text(evidence_bound=evidence_bound),
            legacy_challenge_schema_text(evidence_bound=evidence_bound),
        }
    )


def recognized_challenge_instruction_texts(*, evidence_bound: bool) -> frozenset[str]:
    return frozenset(
        {
            challenge_instructions(evidence_bound=evidence_bound),
            legacy_challenge_instructions(evidence_bound=evidence_bound),
        }
    )


def matching_challenge_contract_pair(
    instruction_text: str,
    schema_text: str,
    *,
    evidence_bound: bool,
    grouped_note: str = "",
) -> Literal["current", "historical"] | None:
    """Return which published generation these two texts form, if either.

    A current instruction with a historical schema, or the reverse, is not a
    contract. Callers append ``grouped_note`` exactly as the task file does.
    """

    if instruction_text == challenge_instructions(
        evidence_bound=evidence_bound
    ) + grouped_note and schema_text == challenge_schema_text(
        evidence_bound=evidence_bound
    ):
        return "current"
    if instruction_text == legacy_challenge_instructions(
        evidence_bound=evidence_bound
    ) + grouped_note and schema_text == legacy_challenge_schema_text(
        evidence_bound=evidence_bound
    ):
        return "historical"
    return None


def render_output_example(
    request: WorkRequest,
    *,
    challenge_input: bytes | None = None,
    evidence_bound: bool = False,
) -> bytes:
    """A model-valid template whose identity fields match ``request``."""

    if request.stage == "first_pass":
        model: FirstPassOutput | ChallengeOutput | ResearchOutput = _first_pass_example(
            request
        )
    elif request.stage == "challenge":
        if challenge_input is None:
            raise ValueError("a challenge example requires the published disputes")
        model = _challenge_example(
            request, challenge_input, evidence_bound=evidence_bound
        )
    elif request.stage == "research":
        model = _research_example(request)
    else:
        raise ValueError(f"unknown stage {request.stage}")
    _require_identity(model, request)
    return canonical_json(model).encode("utf-8")


def _require_identity(
    model: FirstPassOutput | ChallengeOutput | ResearchOutput, request: WorkRequest
) -> None:
    if (
        model.provider != request.provider
        or model.provider_interface != request.provider_interface
        or model.task_id != request.task_id
        or model.job_id != request.job_id
        or model.batch_id != request.batch_id
        or model.document_id != request.document_id
        or model.attempt != request.attempt
        or model.source_bundle_sha256 != request.source_bundle_sha256
        or model.stage != request.stage
    ):
        raise ValueError("output example identity does not match the work request")


def _first_pass_example(request: WorkRequest) -> FirstPassOutput:
    pages = []
    for index, page_id in enumerate(request.page_ids, start=1):
        region_id = f"{page_id}-r01"
        pages.append(
            PageTranscription(
                page_id=page_id,
                reading_order=index,
                diplomatic_transcription=TEMPLATE_PAGE_TEXT,
                regions=[
                    TextRegion(
                        region_id=region_id,
                        order=1,
                        lines=[
                            LineReading(
                                line_id=f"{region_id}-l01",
                                text=TEMPLATE_PAGE_TEXT,
                            )
                        ],
                    )
                ],
                unreadable_spans=[
                    UnreadableSpan(region_id=region_id, text_marker="[illegible]")
                ],
                alternative_readings=[
                    AlternativeReading(
                        region_id=region_id,
                        readings=["TEMPLATE", "TEMPLATE ALTERNATIVE"],
                    )
                ],
                attention_items=[
                    AttentionItem(kind="term", text="TEMPLATE", region_id=region_id)
                ],
                coverage=CoverageReport(
                    coverage_complete=False,
                    unresolved_text_present=True,
                    examined_region_ids=[region_id],
                    inaccessible_or_ambiguous=["TEMPLATE ONLY"],
                ),
                cancellations=[ApparatusNote(region_id=region_id, text="TEMPLATE")],
                insertions=[ApparatusNote(region_id=region_id, text="TEMPLATE")],
                marginal_notes=[ApparatusNote(region_id=region_id, text="TEMPLATE")],
                inaccessible_or_ambiguous=["TEMPLATE ONLY"],
            )
        )
    return FirstPassOutput.model_validate(
        {**identity_from_request(request), "pages": pages}
    )


def _challenge_example(
    request: WorkRequest,
    challenge_input: bytes,
    *,
    evidence_bound: bool,
) -> ChallengeOutput:
    from transcription_review_agent.review.controller import AnonymousChallengeFile

    published = AnonymousChallengeFile.model_validate_json(challenge_input)
    if evidence_bound and not published.groups:
        raise ValueError("an evidence-bound example requires grouped references")
    decisions: list[ChallengeDecision] = []
    for item in published.disputes:
        evidence_ref = None
        assessment = None
        if evidence_bound:
            group = next(
                group
                for group in published.groups or []
                if item.dispute_id in group.dispute_ids
            )
            reference = group.evidence_refs[0]
            evidence_ref = EvidenceReference(
                kind=reference.kind,
                evidence_id=reference.evidence_id,
                page_id=reference.page_id,
            )
            assessment = "INSUFFICIENT"
        decision = ChallengeDecision(
            dispute_id=item.dispute_id,
            decision="UNRESOLVED",
            third_reading=None,
            visual_evidence=TEMPLATE_VISUAL_EVIDENCE,
            evidence_ref=evidence_ref,
            evidence_assessment=assessment,  # type: ignore[arg-type]
        )
        if evidence_bound and evidence_ref is not None:
            check_evidence_binding(
                decision,
                DisputeEvidenceScope(
                    dispute_id=item.dispute_id,
                    page_id=item.page_id,
                    allowed=frozenset(
                        (ref.kind, ref.evidence_id) for ref in group.evidence_refs
                    ),
                ),
                provider=request.provider,
            )
        decisions.append(decision)
    if not decisions:
        raise ValueError("a challenge example needs at least one dispute")
    return ChallengeOutput.model_validate(
        {
            **identity_from_request(request),
            "round_index": request.round_index,
            "decisions": decisions,
        }
    )


def _research_example(request: WorkRequest) -> ResearchOutput:
    return ResearchOutput.model_validate(
        {
            **identity_from_request(request),
            "identifications": [
                Identification(
                    original_visual_reading="TEMPLATE",
                    historical_identity_hypothesis=(
                        "TEMPLATE ONLY. Replace this with your hypothesis."
                    ),
                    supporting_evidence=["TEMPLATE ONLY"],
                    opposing_evidence=["TEMPLATE ONLY"],
                    sources=[
                        ResearchSource(
                            title="TEMPLATE",
                            url="https://example.invalid/template",
                            source_class="primary_image",
                            limitations="TEMPLATE ONLY. This example is not a finding.",
                        )
                    ],
                    original_image_inspected=False,
                    provenance_limitations=(
                        "TEMPLATE ONLY. This example is not a finding."
                    ),
                    confidence="low",
                    unresolved_questions=["TEMPLATE ONLY"],
                    uses_target_document_transcription=False,
                )
            ],
        }
    )
