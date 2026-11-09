"""Schemas for review work requests and provider files.

Provider files are data. Nothing in these models is executed.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from transcription_review_agent.constants import BATCH_ID_PATTERN, SHA256_PATTERN

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
BatchId = Annotated[str, Field(pattern=BATCH_ID_PATTERN)]

PROVIDER_OUTPUT_SCHEMA_VERSION: Final = "review-provider-output-v1"
WORK_REQUEST_SCHEMA_VERSION: Final = "review-work-request-v1"
COMPARISON_SCHEMA_VERSION: Final = "review-comparison-v1"
REVIEW_WORKFLOW_VERSION: Final = "review-next-v1"

ProviderName = Literal["gpt", "grok"]
ProviderInterface = Literal["GPT_WORK", "GROK"]
PROVIDER_INTERFACE: dict[str, str] = {"gpt": "GPT_WORK", "grok": "GROK"}

FORBIDDEN_INPUTS: tuple[str, ...] = (
    "original_ocr_transcription",
    "archive_transcription_of_this_document",
    "other_provider_output",
    "previous_ai_interpretation",
    "web_search_results",
    "historical_identity_suggestions",
)


class ExecutionState(str, Enum):
    QUEUED = "QUEUED"
    READY_FOR_PROVIDER = "READY_FOR_PROVIDER"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_FOR_PROVIDER = "WAITING_FOR_PROVIDER"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class ProviderStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    AWAITING_MANUAL_DISPATCH = "AWAITING_MANUAL_DISPATCH"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    QUOTA_BLOCKED = "QUOTA_BLOCKED"
    INVALID_OUTPUT = "INVALID_OUTPUT"
    FAILED = "FAILED"


class QualityLabel(str, Enum):
    NO_MATERIAL_DISPUTES = "NO_MATERIAL_DISPUTES"
    RESOLVED_WITH_WARNINGS = "RESOLVED_WITH_WARNINGS"
    UNRESOLVED_CONTENT = "UNRESOLVED_CONTENT"
    INCOMPLETE_COVERAGE = "INCOMPLETE_COVERAGE"
    SINGLE_PROVIDER_PROVISIONAL = "SINGLE_PROVIDER_PROVISIONAL"


class DisputeCategory(str, Enum):
    MISSING_CONTENT = "MISSING_CONTENT"
    READING_ORDER = "READING_ORDER"
    NAME_OR_ENTITY = "NAME_OR_ENTITY"
    DATE_OR_NUMBER = "DATE_OR_NUMBER"
    WORDING = "WORDING"
    CANCELLATION_OR_INSERTION = "CANCELLATION_OR_INSERTION"
    PUNCTUATION = "PUNCTUATION"
    COVERAGE_UNCERTAINTY = "COVERAGE_UNCERTAINTY"


CHALLENGEABLE_CATEGORIES: frozenset[DisputeCategory] = frozenset(
    {
        DisputeCategory.MISSING_CONTENT,
        DisputeCategory.READING_ORDER,
        DisputeCategory.NAME_OR_ENTITY,
        DisputeCategory.DATE_OR_NUMBER,
        DisputeCategory.WORDING,
        DisputeCategory.CANCELLATION_OR_INSERTION,
    }
)

DISPUTE_PRIORITY: dict[DisputeCategory, int] = {
    DisputeCategory.MISSING_CONTENT: 0,
    DisputeCategory.READING_ORDER: 1,
    DisputeCategory.NAME_OR_ENTITY: 2,
    DisputeCategory.DATE_OR_NUMBER: 3,
    DisputeCategory.WORDING: 4,
    DisputeCategory.CANCELLATION_OR_INSERTION: 5,
    DisputeCategory.COVERAGE_UNCERTAINTY: 6,
    DisputeCategory.PUNCTUATION: 7,
}

NEW_EVIDENCE_KINDS: frozenset[str] = frozenset(
    {
        "different_source_scan",
        "new_crop",
        "another_occurrence",
        "adjacent_page",
    }
)

ChallengeDecisionCode = Literal[
    "CANDIDATE_X",
    "CANDIDATE_Y",
    "THIRD_READING",
    "UNRESOLVED",
]


class BBox(BaseModel):
    """Pixel window on a page image. Origin is the top-left."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)

    def contains(self, inner: BBox) -> bool:
        return (
            inner.x >= self.x
            and inner.y >= self.y
            and inner.x + inner.width <= self.x + self.width
            and inner.y + inner.height <= self.y + self.height
        )

    @property
    def area(self) -> int:
        return self.width * self.height


class LineReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_id: str = Field(min_length=1)
    text: str


class TextRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(min_length=1)
    order: int = Field(ge=1)
    lines: list[LineReading]
    bbox: BBox | None = None


class UnreadableSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(min_length=1)
    text_marker: str = Field(min_length=1)


class AlternativeReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(min_length=1)
    readings: list[str] = Field(min_length=2)


class RecordedAlternativeReading(BaseModel):
    """A first-pass alternative kept as structured uncertainty (A5).

    The list is the provider's list, in its order. Nothing here chooses a
    reading or rewrites one.
    """

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    page_id: str = Field(min_length=1)
    region_id: str = Field(min_length=1)
    readings: list[str] = Field(min_length=2)


class AttentionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["name", "date", "number", "place", "organization", "event", "term"]
    text: str = Field(min_length=1)
    region_id: str = Field(min_length=1)


class ApparatusNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(min_length=1)
    text: str


class CoverageReport(BaseModel):
    """Visual completeness is independent of whether every word is secure."""

    model_config = ConfigDict(extra="forbid")

    coverage_complete: bool
    unresolved_text_present: bool
    examined_region_ids: list[str]
    inaccessible_or_ambiguous: list[str] = Field(default_factory=list)


class PageTranscription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str = Field(pattern=r"^page_[0-9]{3}$")
    reading_order: int = Field(ge=1)
    diplomatic_transcription: str
    regions: list[TextRegion]
    unreadable_spans: list[UnreadableSpan]
    alternative_readings: list[AlternativeReading]
    attention_items: list[AttentionItem]
    coverage: CoverageReport
    cancellations: list[ApparatusNote]
    insertions: list[ApparatusNote]
    marginal_notes: list[ApparatusNote]
    inaccessible_or_ambiguous: list[str]


class ProviderFileBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["review-provider-output-v1"] = (
        PROVIDER_OUTPUT_SCHEMA_VERSION
    )
    provider: ProviderName
    provider_interface: ProviderInterface
    task_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    batch_id: BatchId
    document_id: int = Field(ge=1)
    attempt: int = Field(ge=1)
    source_bundle_sha256: Sha256

    @model_validator(mode="after")
    def _interface_matches_provider(self) -> ProviderFileBase:
        expected = PROVIDER_INTERFACE[self.provider]
        if self.provider_interface != expected:
            raise ValueError("provider_interface does not match provider")
        return self


class FirstPassOutput(ProviderFileBase):
    stage: Literal["first_pass"] = "first_pass"
    pages: list[PageTranscription] = Field(min_length=1)


EvidenceKind = Literal["CROP", "GROUP", "FULL_PAGE_FALLBACK"]
FallbackReason = Literal[
    "NO_RECORDED_COORDINATES",
    "INVALID_COORDINATES",
    "AMBIGUOUS_COORDINATES",
    "RECORDED_BOX_IS_FULL_PAGE",
]
EvidenceAssessment = Literal["SUPPORTS_DECISION", "INSUFFICIENT", "UNREADABLE"]
ReconciliationClass = Literal["AUTOMATICALLY_SUPPORTED", "UNRESOLVED"]
ReconciliationBasis = Literal[
    "DUAL_MODEL_VISUAL_SUPPORT",
    "PROVIDER_DISAGREEMENT",
    "THIRD_READING_MISMATCH",
    "PROVIDER_UNRESOLVED",
    "UNREADABLE_EVIDENCE",
    "INSUFFICIENT_EVIDENCE",
    # A4: a later round contradicted an earlier resolved decision.
    "CONTRADICTORY_ROUND_EVIDENCE",
]


class EvidenceReference(BaseModel):
    """The visual evidence a provider says it examined for one dispute.

    ``kind`` and ``evidence_id`` name a crop, a visual-review group, or the
    labeled full-page fallback window. The pure reconciliation rule checks
    the reference against the challenged dispute; the provider's claim is
    never trusted on its own.
    """

    model_config = ConfigDict(extra="forbid")

    kind: EvidenceKind
    evidence_id: str = Field(min_length=1)
    page_id: str = Field(min_length=1)


def _decision_reading(
    label: ChallengeDecisionCode | None,
    *,
    candidate_x: str | None,
    candidate_y: str | None,
    third_reading: str | None,
) -> str | None:
    """The exact reading a challenge label selects, or ``None``."""

    if label == "CANDIDATE_X":
        return candidate_x
    if label == "CANDIDATE_Y":
        return candidate_y
    if label == "THIRD_READING":
        return third_reading
    return None


class PartialAgreementSpan(BaseModel):
    """One exact slice of a dispute both providers supported, or still do not.

    Offsets index the reading that provider's decision supports: candidate X,
    candidate Y, or that provider's third reading. They are not offsets into
    a normalized or combined string. An adopted span is the same text on both
    sides. An unresolved span keeps both texts, including a genuine omission.
    """

    model_config = ConfigDict(extra="forbid")

    dispute_id: str = Field(min_length=1)
    role: Literal["ADOPTED", "UNRESOLVED"]
    gpt_start: int = Field(ge=0)
    gpt_end: int = Field(ge=0)
    grok_start: int = Field(ge=0)
    grok_end: int = Field(ge=0)
    gpt_text: str
    grok_text: str
    gpt_decision: ChallengeDecisionCode
    grok_decision: ChallengeDecisionCode
    gpt_evidence_ref: EvidenceReference
    grok_evidence_ref: EvidenceReference
    gpt_evidence_assessment: Literal["SUPPORTS_DECISION"]
    grok_evidence_assessment: Literal["SUPPORTS_DECISION"]

    @model_validator(mode="after")
    def _text_matches_bounds(self) -> PartialAgreementSpan:
        if self.gpt_end < self.gpt_start or self.grok_end < self.grok_start:
            raise ValueError("partial agreement boundary is reversed")
        if self.gpt_end - self.gpt_start != len(self.gpt_text):
            raise ValueError("partial agreement boundary does not match the GPT text")
        if self.grok_end - self.grok_start != len(self.grok_text):
            raise ValueError("partial agreement boundary does not match the Grok text")
        if self.role == "ADOPTED":
            if self.gpt_text == "" or self.gpt_text != self.grok_text:
                raise ValueError("an adopted span is the same non-empty text")
        elif self.gpt_text == self.grok_text:
            raise ValueError("an unresolved span must differ")
        return self


class ReconciledDecision(BaseModel):
    """Controller record of both decisions. Agreement is not verification.

    The A3 fields default to the legacy meaning: ``result_class is None``
    marks a record that was reconciled before evidence binding existed. Such
    a record never counts as automatically supported, verified, or
    eligible for writer memory or publication.

    ``partial_agreements`` is absent on legacy records and on disputes that
    did not safely share a supported span. It never replaces the provider
    readings stored beside it.
    """

    model_config = ConfigDict(extra="forbid")

    dispute_id: str = Field(min_length=1)
    decision: ChallengeDecisionCode
    third_reading: str | None = None
    agreed: bool
    agreement_is_not_verification: Literal[True] = True
    gpt_decision: ChallengeDecisionCode | None = None
    grok_decision: ChallengeDecisionCode | None = None
    gpt_visual_evidence: str | None = None
    grok_visual_evidence: str | None = None
    remaining_uncertainty: bool = False
    # A3. All additive; ``None`` means legacy / not classified.
    result_class: ReconciliationClass | None = None
    basis: ReconciliationBasis | None = None
    selected_reading: str | None = None
    candidate_x_text: str | None = None
    candidate_y_text: str | None = None
    gpt_third_reading: str | None = None
    grok_third_reading: str | None = None
    gpt_evidence_ref: EvidenceReference | None = None
    grok_evidence_ref: EvidenceReference | None = None
    gpt_evidence_assessment: EvidenceAssessment | None = None
    grok_evidence_assessment: EvidenceAssessment | None = None
    independently_verified: Literal[False] = False
    eligible_for_writer_memory: Literal[False] = False
    eligible_for_publication: Literal[False] = False
    # Exact shared spans inside an otherwise unresolved dispute. ``None`` is
    # omitted, so records from before partial agreement keep their bytes.
    partial_agreements: list[PartialAgreementSpan] | None = None
    # A4. Decisions for this same dispute from other challenge rounds, in
    # round order. This record is the standing decision; these are the
    # earlier (or set-aside) ones, kept verbatim. ``None`` is omitted from the
    # JSON, so records without a second round keep their historical bytes.
    other_round_decisions: list[ReconciledDecision] | None = None

    @model_validator(mode="after")
    def _partial_agreements_partition_the_supported_readings(
        self,
    ) -> ReconciledDecision:
        spans = self.partial_agreements
        if not spans:
            return self
        if (
            self.gpt_evidence_assessment != "SUPPORTS_DECISION"
            or self.grok_evidence_assessment != "SUPPORTS_DECISION"
        ):
            raise ValueError(
                "partial agreement requires both providers to support the decision"
            )
        if self.decision != "UNRESOLVED":
            raise ValueError("partial agreement stays on an unresolved dispute")
        gpt_reading = _decision_reading(
            self.gpt_decision,
            candidate_x=self.candidate_x_text,
            candidate_y=self.candidate_y_text,
            third_reading=self.gpt_third_reading,
        )
        grok_reading = _decision_reading(
            self.grok_decision,
            candidate_x=self.candidate_x_text,
            candidate_y=self.candidate_y_text,
            third_reading=self.grok_third_reading,
        )
        if gpt_reading is None or grok_reading is None:
            raise ValueError("partial agreement has no supported reading")
        gpt_at = 0
        grok_at = 0
        for span in spans:
            if span.dispute_id != self.dispute_id:
                raise ValueError("partial agreement is for a different dispute")
            if (
                span.gpt_decision != self.gpt_decision
                or span.grok_decision != self.grok_decision
                or span.gpt_evidence_ref != self.gpt_evidence_ref
                or span.grok_evidence_ref != self.grok_evidence_ref
            ):
                raise ValueError("partial agreement does not match the decision")
            if span.gpt_start != gpt_at or span.grok_start != grok_at:
                raise ValueError("partial agreement boundaries are not contiguous")
            if (
                gpt_reading[span.gpt_start : span.gpt_end] != span.gpt_text
                or grok_reading[span.grok_start : span.grok_end] != span.grok_text
            ):
                raise ValueError("partial agreement text is not the supported slice")
            gpt_at = span.gpt_end
            grok_at = span.grok_end
        if gpt_at != len(gpt_reading) or grok_at != len(grok_reading):
            raise ValueError("partial agreement does not cover the supported readings")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_history(self, handler):  # type: ignore[no-untyped-def]
        data = handler(self)
        if self.partial_agreements is None:
            data.pop("partial_agreements", None)
        if self.other_round_decisions is None:
            data.pop("other_round_decisions", None)
        return data


class ChallengeDecision(BaseModel):
    """One provider's decision for one dispute.

    ``evidence_ref`` and ``evidence_assessment`` are absent only in legacy
    files. They must be given together, so a half-bound decision is a schema
    error rather than a silent downgrade. Their absence never implies support.
    """

    model_config = ConfigDict(extra="forbid")

    dispute_id: str = Field(min_length=1)
    decision: ChallengeDecisionCode
    third_reading: str | None = None
    visual_evidence: str = Field(min_length=1)
    evidence_ref: EvidenceReference | None = None
    evidence_assessment: EvidenceAssessment | None = None

    @model_validator(mode="after")
    def _third_reading_only_when_chosen(self) -> ChallengeDecision:
        if self.decision == "THIRD_READING":
            if self.third_reading is None or not self.third_reading.strip():
                raise ValueError("THIRD_READING requires third_reading text")
        elif self.third_reading is not None:
            raise ValueError("third_reading is only valid for THIRD_READING")
        if (self.evidence_ref is None) != (self.evidence_assessment is None):
            raise ValueError(
                "evidence_ref and evidence_assessment must be given together"
            )
        return self


class ChallengeOutput(ProviderFileBase):
    stage: Literal["challenge"] = "challenge"
    round_index: int = Field(ge=1)
    decisions: list[ChallengeDecision] = Field(min_length=1)


class ResearchSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_class: Literal[
        "primary_image",
        "archival_document",
        "scholarly",
        "derived",
    ]
    limitations: str = ""

    @model_validator(mode="after")
    def _derived_limitations(self) -> ResearchSource:
        if self.source_class == "derived" and not self.limitations.strip():
            raise ValueError("derived sources require explicit limitations")
        return self


class Identification(BaseModel):
    """Historical identity is stored apart from the diplomatic reading."""

    model_config = ConfigDict(extra="forbid")

    original_visual_reading: str = Field(min_length=1)
    historical_identity_hypothesis: str = Field(min_length=1)
    supporting_evidence: list[str]
    opposing_evidence: list[str]
    sources: list[ResearchSource] = Field(min_length=1)
    original_image_inspected: bool
    provenance_limitations: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]
    unresolved_questions: list[str]
    uses_target_document_transcription: bool = False


class ResearchOutput(ProviderFileBase):
    stage: Literal["research"] = "research"
    identifications: list[Identification]


class WorkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["review-work-request-v1"] = WORK_REQUEST_SCHEMA_VERSION
    provider: ProviderName
    provider_interface: ProviderInterface
    task_id: str
    job_id: str
    batch_id: BatchId
    document_id: int = Field(ge=1)
    stage: Literal["first_pass", "challenge", "research"]
    round_index: int = Field(default=0, ge=0)
    attempt: int = Field(ge=1)
    source_bundle_sha256: Sha256
    dispatch: Literal["manual"] = "manual"
    unattended_api: Literal[False] = False
    output_relative_path: str
    page_ids: list[str] = Field(min_length=1)
    forbidden_inputs: list[str]

    @field_validator("output_relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        from transcription_review_agent.bundling.layout import assert_safe_relative_path

        return assert_safe_relative_path(value)


class SourceEvidence(BaseModel):
    """One original value that the canonical first-pass fields do not store.

    ``value`` is the JSON encoding of the original value. Nothing here is
    inferred from the page image.
    """

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    original_sha256: Sha256
    json_path: str = Field(min_length=1)
    value: str
    page_id: str | None = None
    region_id: str | None = None
    line_id: str | None = None


class PageOriginals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str
    gpt_diplomatic_transcription: str
    grok_diplomatic_transcription: str


class Dispute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispute_id: str
    category: DisputeCategory
    page_id: str
    region_id: str | None = None
    gpt_region_ids: list[str] = Field(default_factory=list)
    grok_region_ids: list[str] = Field(default_factory=list)
    gpt_text: str
    grok_text: str
    substantive: bool


class RegionAlignment(BaseModel):
    """One aligned span of line text.

    Region ids name the regions that supplied those lines. A region that
    holds several lines can appear in more than one span. Each line is in
    one span.
    """

    model_config = ConfigDict(extra="forbid")

    alignment_id: str
    page_id: str
    kind: Literal["matched", "split", "merged", "reordered", "missing"]
    gpt_region_ids: list[str]
    grok_region_ids: list[str]
    gpt_text: str
    grok_text: str
    sequence: int = Field(ge=0)


class ComparisonFile(BaseModel):
    """Deterministic comparison. Confidence is not a selector."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["review-comparison-v1"] = COMPARISON_SCHEMA_VERSION
    batch_id: BatchId
    document_id: int = Field(ge=1)
    source_bundle_sha256: Sha256
    gpt_output_sha256: Sha256
    grok_output_sha256: Sha256
    page_originals: list[PageOriginals]
    alignments: list[RegionAlignment] = Field(default_factory=list)
    disputes: list[Dispute]
    unresolved_text_page_ids: list[str]
    selected_by_confidence: Literal[False] = False
    gpt_provider_interface: Literal["GPT_WORK"] = "GPT_WORK"
    grok_provider_interface: Literal["GROK"] = "GROK"


class MemberFallback(BaseModel):
    """Why one dispute in a shared full-page crop had no targeted window."""

    model_config = ConfigDict(extra="forbid")

    dispute_id: str = Field(min_length=1)
    fallback_reason: FallbackReason


class CropRecord(BaseModel):
    """A window onto existing pixels. Enlargement adds context, not detail."""

    model_config = ConfigDict(extra="forbid")

    crop_id: str
    # A single-dispute crop names ``dispute_id``. A crop shared by a visual
    # evidence group (A2) names ``group_id`` and ``dispute_ids`` instead.
    dispute_id: str | None = None
    page_id: str
    page_sha256: Sha256
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    target_bbox: BBox
    crop_bbox: BBox
    contains_target: Literal[True] = True
    enlarged: bool
    scale: Literal[1] = 1
    represents_new_visual_detail: Literal[False] = False
    pixel_dimensions_unchanged: Literal[True] = True
    window_kind: Literal["TARGETED", "FULL_PAGE_FALLBACK"] = "TARGETED"
    # Set only for a new full-page fallback. Historical crops omit it.
    fallback_reason: FallbackReason | None = None
    # Present when members of one full-page crop do not share one reason.
    member_fallbacks: list[MemberFallback] | None = Field(default=None, min_length=1)
    # Recorded geometry that located a targeted crop. Not inferred from text.
    coordinate_sources: list[str] | None = Field(default=None, min_length=1)
    # The crop image is the page image already stored beside it.
    reuses_page_pixels: Literal[True] | None = None
    image_filename: str | None = None
    source_mime_type: Literal["image/png", "image/jpeg"] = "image/png"
    derived_png_sha256: Sha256 | None = None
    crop_sha256: Sha256 | None = None
    crop_png_width: int | None = Field(default=None, ge=1)
    crop_png_height: int | None = Field(default=None, ge=1)
    full_page_retained: Literal[True] = True
    group_id: str | None = None
    dispute_ids: list[str] | None = Field(default=None, min_length=2)

    @model_validator(mode="after")
    def _one_owner(self) -> CropRecord:
        single = self.dispute_id is not None
        grouped = self.group_id is not None or self.dispute_ids is not None
        if single == grouped or (
            grouped and (self.group_id is None or self.dispute_ids is None)
        ):
            raise ValueError(
                "a crop belongs to one dispute or to one group with its disputes"
            )
        if self.dispute_ids is not None and len(set(self.dispute_ids)) != len(
            self.dispute_ids
        ):
            raise ValueError("a crop group lists each dispute once")
        if self.window_kind == "TARGETED" and (
            self.fallback_reason is not None
            or self.member_fallbacks is not None
            or self.reuses_page_pixels is not None
        ):
            raise ValueError("a targeted crop is not a full-page fallback")
        if self.member_fallbacks is not None:
            if self.window_kind != "FULL_PAGE_FALLBACK":
                raise ValueError("member fallbacks belong to a full-page crop")
            member_ids = [item.dispute_id for item in self.member_fallbacks]
            if len(member_ids) != len(set(member_ids)):
                raise ValueError("a dispute fallback is listed twice")
            owners = self.dispute_ids if self.dispute_ids is not None else []
            if self.dispute_id is not None:
                owners = [self.dispute_id]
            if set(member_ids) != set(owners):
                raise ValueError("member fallbacks must match the crop's disputes")
        if self.reuses_page_pixels is True:
            if self.window_kind != "FULL_PAGE_FALLBACK" or self.image_filename is None:
                raise ValueError("a reused page image is a labeled full-page fallback")
        if self.image_filename is not None and (
            "/" in self.image_filename
            or "\\" in self.image_filename
            or self.image_filename in {".", ".."}
        ):
            raise ValueError("crop image filename must stay inside the task input")
        return self

    @model_serializer(mode="wrap")
    def _omit_absent_owner(self, handler):  # type: ignore[no-untyped-def]
        # Absent optional fields stay off the JSON so an older crop keeps its bytes.
        data = handler(self)
        for name in (
            "dispute_id",
            "group_id",
            "dispute_ids",
            "fallback_reason",
            "member_fallbacks",
            "coordinate_sources",
            "reuses_page_pixels",
            "image_filename",
        ):
            if data.get(name) is None:
                data.pop(name, None)
        return data


class IndependenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identification: Identification
    independent: bool
    corroboration_rejected: bool


class ChallengeSelectionRecord(BaseModel):
    """The frozen explicit selection of one challenge round (A1).

    Disputes of the comparison that are not listed here were not part of this
    round. They stay unresolved unless another decision covers them.
    """

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(ge=1)
    selected_dispute_ids: list[str] = Field(min_length=1)
    comparison_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]


ChallengeClosureMode = Literal[
    "NO_CHALLENGE_REQUESTED",
    "SELECTED_ROUNDS_COMPLETE",
    "NO_NEW_INFORMATION",
    "ROUND_CAP",
    "PROVIDER_TERMINAL_FAILURE",
]
UnresolvedStatus = Literal[
    "NOT_CHALLENGED",
    "NOT_SELECTED",
    "SELECTED_UNRESOLVED",
    "PROVIDER_FAILURE",
]


class UnresolvedDispute(BaseModel):
    """A substantive dispute that has no resolved decision when the stage closes.

    ``NOT_CHALLENGED``: no challenge round existed. ``NOT_SELECTED``: rounds
    existed but never selected it. ``SELECTED_UNRESOLVED``: a round decided it
    and the decision is ``UNRESOLVED``. ``PROVIDER_FAILURE``: its latest round
    ended in a terminal provider failure and carries no decision. None of
    these has a winning reading.
    """

    model_config = ConfigDict(extra="forbid")

    dispute_id: str = Field(min_length=1)
    status: UnresolvedStatus
    selected_in_rounds: list[int] = Field(default_factory=list)
    basis: ReconciliationBasis | None = None


class ChallengeClosureRecord(BaseModel):
    """An explicit operator closure of the challenge stage (A4).

    Closure is workflow closure, not certainty: unresolved disputes stay
    listed and keep their alternatives. It is never inferred from inactivity.
    """

    model_config = ConfigDict(extra="forbid")

    mode: ChallengeClosureMode
    reason: str = Field(min_length=1, max_length=2000)
    initiated_by: Literal["OPERATOR"] = "OPERATOR"
    comparison_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    round_count: int = Field(ge=0)
    failed_round_indexes: list[int] = Field(default_factory=list)
    max_challenge_rounds: int | None = Field(default=None, ge=1)
    unresolved: list[UnresolvedDispute] = Field(default_factory=list)

    @model_validator(mode="after")
    def _explicit_reason(self) -> ChallengeClosureRecord:
        if not self.reason.strip():
            raise ValueError("a challenge closure needs an explicit reason")
        return self


class ReviewReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["review-report-v1"] = "review-report-v1"
    workflow_version: Literal["review-next-v1"] = REVIEW_WORKFLOW_VERSION
    batch_id: BatchId
    document_id: int = Field(ge=1)
    execution_state: Literal["COMPLETED"] = "COMPLETED"
    review_completion: Literal["completed", "interim_provisional"]
    quality: QualityLabel
    production_writeback: Literal[False] = False
    diplomatic_text_independent_of_research: Literal[True] = True
    single_provider: bool
    succeeded_provider: ProviderName | None = None
    coverage_gaps: list[str]
    unresolved_text_page_ids: list[str]
    disputes: list[Dispute]
    decisions: list[ReconciledDecision]
    identifications: list[IndependenceRecord]
    provider_failure_count: int = Field(ge=0)
    estimated_work_units: int = Field(ge=0)
    page_ids: list[str]
    source_evidence: list[SourceEvidence] = Field(default_factory=list)
    # A1. ``None`` is omitted from the JSON so reports of documents without an
    # explicit selection keep their historical bytes.
    challenge_selections: list[ChallengeSelectionRecord] | None = None
    # A4. Present only when the operator explicitly closed the challenge stage.
    challenge_closure: ChallengeClosureRecord | None = None
    # A5. First-pass alternative readings, copied and not chosen among.
    # ``None`` is omitted so a report with no alternatives keeps its bytes.
    alternative_readings: list[RecordedAlternativeReading] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_selections(self, handler):  # type: ignore[no-untyped-def]
        data = handler(self)
        if self.challenge_selections is None:
            data.pop("challenge_selections", None)
        if self.challenge_closure is None:
            data.pop("challenge_closure", None)
        if not self.alternative_readings:
            data.pop("alternative_readings", None)
        return data


class ManifestProvider(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    provider_interface: ProviderInterface
    stage: str
    task_id: str
    job_id: str
    output_sha256: Sha256


class ManifestArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sha256: Sha256


class ReviewManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["review-manifest-v1"] = "review-manifest-v1"
    workflow_version: Literal["review-next-v1"] = REVIEW_WORKFLOW_VERSION
    batch_id: BatchId
    document_id: int = Field(ge=1)
    source_bundle_sha256: Sha256
    source_batch_manifest_sha256: Sha256
    providers: list[ManifestProvider]
    artifacts: list[ManifestArtifact]
    comparison_sha256: Sha256 | None = None
    production_writeback: Literal[False] = False


class SourcePageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_id: str
    reading_order: int = Field(ge=1)
    sha256: Sha256
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    filename: str
    mime_type: Literal["image/png", "image/jpeg"] = "image/png"
    derived_png_filename: str | None = None
    derived_png_sha256: Sha256 | None = None


class SourceManifest(BaseModel):
    """Provider-visible integrity metadata. The archive transcription is absent."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["review-source-manifest-v1"] = "review-source-manifest-v1"
    batch_id: BatchId
    document_id: int = Field(ge=1)
    language: Literal["en", "fr"]
    source_bundle_sha256: Sha256
    archive_transcription_included: Literal[False] = False
    pages: list[SourcePageRecord]


class ReviewBatchHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["review-batch-v1"] = "review-batch-v1"
    workflow_version: Literal["review-next-v1"] = REVIEW_WORKFLOW_VERSION
    batch_id: BatchId
    source_batch_manifest_sha256: Sha256
    document_ids: list[int]
    production_writeback: Literal[False] = False


def interface_for(provider: str) -> ProviderInterface:
    try:
        return PROVIDER_INTERFACE[provider]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError("provider must be gpt or grok") from exc
