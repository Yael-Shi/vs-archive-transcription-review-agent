"""Manual localization sidecar for three pilot disputes.

Coordinates are normalized to the original page image. A mark is bound to
one candidate id and that image's SHA-256. Nothing here verifies a reading
or writes a review.
"""

from __future__ import annotations

from typing import Final, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.errors import ReviewIntegrityError

LOCALIZATION_SCHEMA: Final = "dispute-localization-v1"
NORMALIZED_SCALE: Final = 1_000_000
PILOT_DISPUTES: Final[tuple[tuple[int, str, str], ...]] = (
    (105, "page_001", "MISSING_CONTENT"),
    (104, "page_002", "NAME_OR_ENTITY"),
    (108, "page_005", "DATE_OR_NUMBER"),
)
LocationStatus = Literal["HUMAN_MARKED", "UNCLEAR", "NO_LOCATION"]


class NormalizedBox(BaseModel):
    """A rectangle as millionths of the original image, origin at the top-left."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0, le=NORMALIZED_SCALE)
    y: int = Field(ge=0, le=NORMALIZED_SCALE)
    width: int = Field(ge=1, le=NORMALIZED_SCALE)
    height: int = Field(ge=1, le=NORMALIZED_SCALE)
    scale: Literal[1000000] = NORMALIZED_SCALE

    @model_validator(mode="after")
    def _inside_image(self) -> NormalizedBox:
        if (
            self.x + self.width > NORMALIZED_SCALE
            or self.y + self.height > NORMALIZED_SCALE
        ):
            raise ValueError("box extends outside the image")
        if (
            self.x == 0
            and self.y == 0
            and self.width == NORMALIZED_SCALE
            and self.height == NORMALIZED_SCALE
        ):
            raise ValueError("a full-page box is not a dispute location")
        return self


class LocalizationMark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^lc_[0-9a-f]{20}$")
    document_id: int = Field(ge=1)
    page_id: str = Field(min_length=1)
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    status: LocationStatus
    method: Literal["HUMAN_MARK"] = "HUMAN_MARK"
    bbox: NormalizedBox | None = None
    coordinate_space: Literal["NORMALIZED_IMAGE"] = "NORMALIZED_IMAGE"
    human_verification_status: Literal["NOT_VERIFIED"] = "NOT_VERIFIED"
    eligible_for_authoritative_learning: Literal[False] = False

    @model_validator(mode="after")
    def _box_matches_status(self) -> LocalizationMark:
        if self.status == "HUMAN_MARKED" and self.bbox is None:
            raise ValueError("a human-marked location needs a box")
        if self.status != "HUMAN_MARKED" and self.bbox is not None:
            raise ValueError("unclear and no-location marks do not carry a box")
        return self


class LocalizationSidecar(BaseModel):
    """Exported manual marks. This file is not a review and not a ledger."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dispute-localization-v1"] = LOCALIZATION_SCHEMA
    packet_role: Literal["MANUAL_LOCALIZATION_SIDECAR"] = "MANUAL_LOCALIZATION_SIDECAR"
    human_verified_count: Literal[0] = 0
    decision_ledger_written: Literal[False] = False
    marks: list[LocalizationMark]


class CaseBinding(BaseModel):
    """The queue identity a mark is allowed to cite."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    document_id: int
    page_id: str
    page_image_sha256: str
    page_width: int
    page_height: int
    dispute_category: str
    pilot: bool


def is_pilot_dispute(document_id: int, page_id: str, dispute_category: str) -> bool:
    return (document_id, page_id, dispute_category) in PILOT_DISPUTES


def load_sidecar(
    payload: str | bytes, bindings: Sequence[CaseBinding]
) -> LocalizationSidecar:
    """Parse a sidecar and refuse marks that do not match the pilot cases."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        sidecar = LocalizationSidecar.model_validate_json(text)
    except ValidationError as exc:
        raise ReviewIntegrityError(
            "localization sidecar does not match the schema"
        ) from exc
    return bind_sidecar(sidecar, bindings)


def bind_sidecar(
    sidecar: LocalizationSidecar, bindings: Sequence[CaseBinding]
) -> LocalizationSidecar:
    """Refuse a mark for the wrong case or a different page image."""

    index = {item.candidate_id: item for item in bindings}
    if len(index) != len(bindings):
        raise ReviewIntegrityError("localization bindings repeat a candidate")
    seen: set[str] = set()
    for mark in sidecar.marks:
        if mark.candidate_id in seen:
            raise ReviewIntegrityError("localization repeats a candidate")
        seen.add(mark.candidate_id)
        binding = index.get(mark.candidate_id)
        if binding is None or not binding.pilot:
            raise ReviewIntegrityError(
                "localization names a case that is not one of the three pilot disputes"
            )
        if (
            mark.document_id != binding.document_id
            or mark.page_id != binding.page_id
            or mark.page_image_sha256 != binding.page_image_sha256
            or mark.page_width != binding.page_width
            or mark.page_height != binding.page_height
        ):
            raise ReviewIntegrityError(
                "localization does not match the original page image"
            )
    return sidecar


def export_sidecar(sidecar: LocalizationSidecar) -> str:
    """Serialize a validated sidecar. The caller writes it outside the reviews."""

    return canonical_json(sidecar)
