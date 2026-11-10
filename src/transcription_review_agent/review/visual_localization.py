"""Model-proposed locations for the three pilot disputes.

A proposal is a new observation about the original page image. It is not a
human mark, not a verification of the reading, and not a pixel conversion of
a text offset or a reading-order index. Coordinates enter only as measured
rectangles that already lie on the page.
"""

from __future__ import annotations

import hashlib
from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.errors import ReviewIntegrityError
from transcription_review_agent.review.localization import (
    NORMALIZED_SCALE,
    NormalizedBox,
)

VISUAL_PROPOSAL_SCHEMA: Final = "dispute-visual-proposal-v1"
METHOD_VERSION: Final = "visual-proposal-pilot-1"
ProposalStatus = Literal["PROPOSED", "AMBIGUOUS", "NOT_FOUND", "ERROR"]
ProposalMethod = Literal["MODEL_PROPOSAL", "UNEXECUTED"]


class ProposalProvenance(BaseModel):
    """How a proposal was produced. This is not a writer identity."""

    model_config = ConfigDict(extra="forbid")

    image_sha256: str = Field(pattern=SHA256_PATTERN)
    method_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    request_sha256: str = Field(pattern=SHA256_PATTERN)
    executed: bool
    human_marked: Literal[False] = False


class VisualProposal(BaseModel):
    """One candidate. A model proposal cannot be stored as a human mark."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^lc_[0-9a-f]{20}$")
    dispute_id: str = Field(pattern=r"^dsp_[0-9a-f]{16}$")
    document_id: int = Field(ge=1)
    page_id: str = Field(min_length=1)
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    dispute_category: str = Field(min_length=1)
    status: ProposalStatus
    method: ProposalMethod
    provenance: ProposalProvenance
    boxes: list[NormalizedBox]
    explanation: str | None = Field(default=None, max_length=400)
    nearby_text_anchor: str | None = Field(default=None, max_length=240)
    explanation_verified: Literal[False] = False
    anchor_verified: Literal[False] = False
    location_authority: Literal["MODEL_PROPOSED"] = "MODEL_PROPOSED"
    human_verification_status: Literal["NOT_VERIFIED"] = "NOT_VERIFIED"
    eligible_for_authoritative_learning: Literal[False] = False

    @model_validator(mode="after")
    def _status_matches_boxes(self) -> VisualProposal:
        if self.provenance.image_sha256 != self.page_image_sha256:
            raise ValueError("provenance image hash does not match the page")
        if self.method == "UNEXECUTED":
            if self.status != "ERROR" or self.provenance.executed:
                raise ValueError("an unexecuted proposal is an error with no run")
        elif not self.provenance.executed:
            raise ValueError("a model proposal records that it was executed")
        if self.status == "PROPOSED" and not self.boxes:
            raise ValueError("a proposed location needs at least one box")
        if self.status == "AMBIGUOUS" and len(self.boxes) < 2:
            raise ValueError("an ambiguous location lists more than one box")
        if self.status in {"NOT_FOUND", "ERROR"} and self.boxes:
            raise ValueError("not-found and error results do not carry a box")
        if self.status == "ERROR" and not self.explanation:
            raise ValueError("an error result explains why no box was returned")
        seen: set[tuple[int, int, int, int]] = set()
        for box in self.boxes:
            key = (box.x, box.y, box.width, box.height)
            if key in seen:
                raise ValueError("a proposal repeats the same box")
            seen.add(key)
        return self


class VisualProposalSidecar(BaseModel):
    """Proposals live outside reviews, bundles, and the human-mark sidecar."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dispute-visual-proposal-v1"] = VISUAL_PROPOSAL_SCHEMA
    packet_role: Literal["VISUAL_PROPOSAL_SIDECAR"] = "VISUAL_PROPOSAL_SIDECAR"
    export_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    human_verified_count: Literal[0] = 0
    decision_ledger_written: Literal[False] = False
    results: list[VisualProposal]


class ProposalBinding(BaseModel):
    """Identity a proposal is allowed to cite. It carries no pixels."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    dispute_id: str
    document_id: int
    page_id: str
    page_image_sha256: str
    page_width: int
    page_height: int
    dispute_category: str
    pilot: bool


def prepare_request_sha256(
    *,
    candidate_id: str,
    dispute_id: str,
    document_id: int,
    page_id: str,
    page_image_sha256: str,
    gpt_text: str,
    grok_text: str,
) -> str:
    """Hash the request a later vision run would see.

    The hash covers identity, the image digest, and the two readings. It does
    not cover a coordinate, a line sequence, or a page count.
    """

    payload = "\n".join(
        [
            VISUAL_PROPOSAL_SCHEMA,
            candidate_id,
            dispute_id,
            str(document_id),
            page_id,
            page_image_sha256,
            gpt_text,
            grok_text,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_measured_pixels(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    page_width: int,
    page_height: int,
) -> NormalizedBox:
    """Convert a rectangle already measured on the page image.

    ``x`` and ``y`` are pixels from the top-left of that image. A full-page
    rectangle is rejected. This function does not accept text.
    """

    if page_width < 1 or page_height < 1:
        raise ReviewIntegrityError("page dimensions must be positive")
    if width < 1 or height < 1 or x < 0 or y < 0:
        raise ReviewIntegrityError("measured box is degenerate or outside the page")
    if x + width > page_width or y + height > page_height:
        raise ReviewIntegrityError("measured box is outside the page")
    if x == 0 and y == 0 and width == page_width and height == page_height:
        raise ReviewIntegrityError("a full-page box is not a dispute location")
    box = NormalizedBox(
        x=(x * NORMALIZED_SCALE) // page_width,
        y=(y * NORMALIZED_SCALE) // page_height,
        width=max(1, (width * NORMALIZED_SCALE) // page_width),
        height=max(1, (height * NORMALIZED_SCALE) // page_height),
    )
    return box


def load_visual_proposals(
    payload: str | bytes,
    bindings: Sequence[ProposalBinding],
    *,
    export_sha256: str | None = None,
) -> VisualProposalSidecar:
    """Parse proposals and refuse any result that does not match the pilot."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        sidecar = VisualProposalSidecar.model_validate_json(text)
    except ValidationError as exc:
        raise ReviewIntegrityError(
            "visual proposal sidecar does not match the schema"
        ) from exc
    return bind_visual_proposals(sidecar, bindings, export_sha256=export_sha256)


def bind_visual_proposals(
    sidecar: VisualProposalSidecar,
    bindings: Sequence[ProposalBinding],
    *,
    export_sha256: str | None = None,
) -> VisualProposalSidecar:
    """Refuse a wrong page, a non-pilot case, or a repeated candidate."""

    if export_sha256 is not None and sidecar.export_sha256 not in {None, export_sha256}:
        raise ReviewIntegrityError(
            "visual proposals were prepared for a different learning export"
        )
    index = {item.candidate_id: item for item in bindings}
    if len(index) != len(bindings):
        raise ReviewIntegrityError("visual proposal bindings repeat a candidate")
    seen: set[str] = set()
    for result in sidecar.results:
        if result.candidate_id in seen:
            raise ReviewIntegrityError("visual proposals repeat a candidate")
        seen.add(result.candidate_id)
        binding = index.get(result.candidate_id)
        if binding is None or not binding.pilot:
            raise ReviewIntegrityError(
                "visual proposal names a case that is not one of the three pilot disputes"
            )
        if (
            result.dispute_id != binding.dispute_id
            or result.document_id != binding.document_id
            or result.page_id != binding.page_id
            or result.page_image_sha256 != binding.page_image_sha256
            or result.page_width != binding.page_width
            or result.page_height != binding.page_height
            or result.dispute_category != binding.dispute_category
        ):
            raise ReviewIntegrityError(
                "visual proposal does not match the original page image"
            )
    return sidecar


def export_visual_proposals(sidecar: VisualProposalSidecar) -> str:
    """Serialize a validated proposal sidecar. The caller writes it outside reviews."""

    return canonical_json(sidecar)


def unexecuted_placeholder(
    binding: ProposalBinding,
    *,
    gpt_text: str,
    grok_text: str,
    reason: str,
) -> VisualProposal:
    """A pilot row for a vision run that was not executed. It has no box."""

    return VisualProposal(
        candidate_id=binding.candidate_id,
        dispute_id=binding.dispute_id,
        document_id=binding.document_id,
        page_id=binding.page_id,
        page_image_sha256=binding.page_image_sha256,
        page_width=binding.page_width,
        page_height=binding.page_height,
        dispute_category=binding.dispute_category,
        status="ERROR",
        method="UNEXECUTED",
        provenance=ProposalProvenance(
            image_sha256=binding.page_image_sha256,
            method_version=METHOD_VERSION,
            model_id="none",
            tool="unexecuted",
            request_sha256=prepare_request_sha256(
                candidate_id=binding.candidate_id,
                dispute_id=binding.dispute_id,
                document_id=binding.document_id,
                page_id=binding.page_id,
                page_image_sha256=binding.page_image_sha256,
                gpt_text=gpt_text,
                grok_text=grok_text,
            ),
            executed=False,
        ),
        boxes=[],
        explanation=reason,
    )
