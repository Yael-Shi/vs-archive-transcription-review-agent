"""One-shot localization runner for the three pilot disputes.

The human-confirmed reference is a separate file. A model request never
receives it. A run writes a dry plan unless ``--execute`` is set, and then
it makes one process call per case with no retry. Overlap with a confirmed
box is geometry, not a check that the disputed words were read.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import BinaryIO, Callable, Final, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.config import FORBIDDEN_WRITE_ROOTS
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.errors import (
    ReviewError,
    ReviewIntegrityError,
    ReviewPathError,
    ReviewUsageError,
)
from transcription_review_agent.review.human_verification_packet import (
    PHASE1_DOCUMENT_IDS,
    locate_original_page,
    select_phase1_queue,
)
from transcription_review_agent.review.jpegutil import decode_jpeg, jpeg_dimensions
from transcription_review_agent.review.learning_export import (
    LearningCandidate,
    export_learning_candidates,
)
from transcription_review_agent.review.localization import (
    NORMALIZED_SCALE,
    PILOT_DISPUTES,
    NormalizedBox,
    is_pilot_dispute,
)
from transcription_review_agent.review.pngutil import (
    PNG_SIGNATURE,
    crop_rows,
    decode_png,
    encode_png,
)
from transcription_review_agent.review.visual_localization import (
    ProposalProvenance,
    VisualProposal,
    VisualProposalSidecar,
    normalize_measured_pixels,
    prepare_request_sha256,
)

CONFIRMED_SCHEMA: Final = "human-confirmed-location-v1"
CONFIRMED_VERSION: Final = "confirmed-location-reference-1"
DRY_RUN_SCHEMA: Final = "reproducible-localization-dry-run-v1"
RUN_SCHEMA: Final = "reproducible-localization-run-v1"
EVALUATION_SCHEMA: Final = "reproducible-localization-evaluation-v1"
RUNNER_METHOD: Final = "reproducible-localization-pilot-1"
PROMPT_VERSION: Final = "reproducible-localization-prompt-1"
DEFAULT_TIMEOUT_SECONDS: Final = 120
MAX_PROMPT_CHARS: Final = 100_000
MAX_PROCESS_OUTPUT_BYTES: Final = 1_048_576
MAX_MODEL_RESPONSE_BYTES: Final = 262_144
# normalize_measured_pixels truncates toward zero. One source pixel is
# therefore at most this many pixels away from the stored normalized box.
PIXEL_ROUNDING_TOLERANCE: Final = 1
_READ_CHUNK: Final = 65_536
_QUARTER: Final = 4
_MEASURED_PIXELS = re.compile(
    r"Measured pixels x=(\d+) y=(\d+) width=(\d+) height=(\d+) on (\d+)x(\d+)"
)
_READING_NOTE: Final = (
    "Geometric overlap does not show that the disputed text was read."
)
# Codex structured output rejects draft keywords and numeric bounds.
# Pixel bounds are checked after the response, in this process.
_OUTPUT_SCHEMA: Final = """{
  "type": "object",
  "additionalProperties": false,
  "required": ["candidate_id", "page_image_sha256", "status", "boxes"],
  "properties": {
    "candidate_id": {"type": "string"},
    "page_image_sha256": {"type": "string"},
    "status": {"enum": ["PROPOSED", "AMBIGUOUS", "NOT_FOUND"]},
    "boxes": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["x", "y", "width", "height"],
        "properties": {
          "x": {"type": "integer"},
          "y": {"type": "integer"},
          "width": {"type": "integer"},
          "height": {"type": "integer"}
        }
      }
    }
  }
}
"""
ResultOrigin = Literal["mock", "live_model", "prior_proposal"]
Outcome = Literal["SCORED", "AMBIGUOUS", "NOT_FOUND", "ERROR", "NO_RESULT"]


class LocalizationRunError(ReviewError):
    """One vision invocation failed. The runner does not try again."""


class BoundedCaptureError(LocalizationRunError):
    """Stdout or stderr crossed the capture limit. The bytes kept are a prefix."""

    def __init__(self, message: str, stdout: bytes, stderr: bytes) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class MeasuredPixels(BaseModel):
    """A rectangle already measured on the page, in pixels from the top-left."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)


class ConfirmedLocation(BaseModel):
    """A region a person confirmed on the page. The reading stays unverified."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^lc_[0-9a-f]{20}$")
    dispute_id: str = Field(pattern=r"^dsp_[0-9a-f]{16}$")
    document_id: int = Field(ge=1)
    page_id: str = Field(min_length=1)
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    dispute_category: str = Field(min_length=1)
    boxes: list[NormalizedBox] = Field(min_length=1)
    measured_pixels: list[MeasuredPixels]
    location_authority: Literal["HUMAN_CONFIRMED_LOCATION"] = "HUMAN_CONFIRMED_LOCATION"
    human_verification_status: Literal["NOT_VERIFIED"] = "NOT_VERIFIED"
    transcription_verified: Literal[False] = False
    eligible_for_authoritative_learning: Literal[False] = False
    confirmation: Literal["USER_VISUAL_CONFIRMATION"] = "USER_VISUAL_CONFIRMATION"
    source_method_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _pixels_match_boxes(self) -> ConfirmedLocation:
        if not self.measured_pixels:
            return self
        if len(self.measured_pixels) != len(self.boxes):
            raise ValueError("measured pixels do not match the confirmed boxes")
        for box, measured in zip(self.boxes, self.measured_pixels, strict=True):
            if (
                measured.page_width != self.page_width
                or measured.page_height != self.page_height
            ):
                raise ValueError("measured pixels name a different page size")
            try:
                expected = normalize_measured_pixels(
                    x=measured.x,
                    y=measured.y,
                    width=measured.width,
                    height=measured.height,
                    page_width=measured.page_width,
                    page_height=measured.page_height,
                )
            except ReviewIntegrityError as exc:
                raise ValueError("measured pixels are outside the page") from exc
            if not _within_rounding_tolerance(
                box, expected, self.page_width, self.page_height
            ):
                raise ValueError(
                    "measured pixels are not consistent with the confirmed boxes"
                )
        return self


class ConfirmedLocationReference(BaseModel):
    """Versioned copy of confirmed regions. It is not a proposal or a review."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["human-confirmed-location-v1"] = CONFIRMED_SCHEMA
    reference_role: Literal["HUMAN_CONFIRMED_LOCATION_REFERENCE"] = (
        "HUMAN_CONFIRMED_LOCATION_REFERENCE"
    )
    reference_version: Literal["confirmed-location-reference-1"] = CONFIRMED_VERSION
    human_verified_transcription_count: Literal[0] = 0
    authoritative_learning_eligible_count: Literal[0] = 0
    decision_ledger_written: Literal[False] = False
    source_proposal_schema: Literal["dispute-visual-proposal-v1"] = (
        "dispute-visual-proposal-v1"
    )
    source_proposal_sha256: str = Field(pattern=SHA256_PATTERN)
    locations: list[ConfirmedLocation]


class LocationConfirmation(BaseModel):
    """A recorded human action. It confirms boxes and does not verify the reading."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^lc_[0-9a-f]{20}$")
    dispute_id: str = Field(pattern=r"^dsp_[0-9a-f]{16}$")
    document_id: int = Field(ge=1)
    page_id: str = Field(min_length=1)
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    action: Literal["USER_VISUAL_CONFIRMATION"]
    boxes: list[NormalizedBox] = Field(min_length=1)
    human_verification_status: Literal["NOT_VERIFIED"] = "NOT_VERIFIED"
    transcription_verified: Literal[False] = False


class LocationConfirmationRecord(BaseModel):
    """Explicit confirmations supplied separately from a proposal sidecar."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["location-confirmation-v1"] = "location-confirmation-v1"
    confirmations: list[LocationConfirmation]


class PilotCase(BaseModel):
    """Dispute context sent to a model. It has no box."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    dispute_id: str
    document_id: int = Field(ge=1)
    page_id: str
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    page_width: int = Field(ge=1)
    page_height: int = Field(ge=1)
    dispute_category: str
    gpt_text: str
    grok_text: str
    image_path: str


class PixelBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)


class ModelLocalization(BaseModel):
    """The only JSON a localization model is asked to return."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    page_image_sha256: str = Field(pattern=SHA256_PATTERN)
    status: Literal["PROPOSED", "AMBIGUOUS", "NOT_FOUND"]
    boxes: list[PixelBox]


class DryRunCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    document_id: int
    page_id: str
    dispute_category: str
    page_image_sha256: str
    page_width: int
    page_height: int
    request_sha256: str
    prompt_sha256: str
    reference_boxes_included: Literal[False] = False
    model_called: Literal[False] = False


class DryRunPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reproducible-localization-dry-run-v1"] = DRY_RUN_SCHEMA
    execute: Literal[False] = False
    retries: Literal[0] = 0
    max_requests_per_case: Literal[1] = 1
    timeout_seconds: int = Field(ge=1, le=600)
    codex_executable: str | None
    codex_available: bool
    reference_boxes_included: Literal[False] = False
    model_called: Literal[False] = False
    cases: list[DryRunCase]


class RunCaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    request_sha256: str
    prompt_sha256: str
    calls: Literal[1] = 1
    status: Literal["PROPOSED", "AMBIGUOUS", "NOT_FOUND", "ERROR"]


class LocalizationRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reproducible-localization-run-v1"] = RUN_SCHEMA
    execute: Literal[True] = True
    retries: Literal[0] = 0
    max_requests_per_case: Literal[1] = 1
    timeout_seconds: int = Field(ge=1, le=600)
    tool: str
    model_id: str
    cases: list[RunCaseRecord]
    results: VisualProposalSidecar


class PairScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_index: int = Field(ge=0)
    proposal_index: int = Field(ge=0)
    iou: str


class CaseEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    document_id: int
    page_id: str
    result_status: str
    outcome: Outcome
    best_iou: str | None
    pairs: list[PairScore]
    inspection: list[str]
    geometric_overlap_is_not_a_reading: Literal[True] = True


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reproducible-localization-evaluation-v1"] = (
        EVALUATION_SCHEMA
    )
    origin: ResultOrigin
    live_model_executed: bool
    reading_note: Literal[
        "Geometric overlap does not show that the disputed text was read."
    ] = _READING_NOTE
    cases: list[CaseEvaluation]


class ModelCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    image_path: str
    prompt: str
    request_sha256: str
    prompt_sha256: str


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    model_id: str = Field(min_length=1)
    tool: str = Field(min_length=1)


class VisionRunner(Protocol):
    """One localization call. The runner does not invoke it twice for a case."""

    def run(self, call: ModelCall) -> ModelResponse:
        """Return the raw JSON text from a single invocation."""


def model_input_document(case: PilotCase) -> dict[str, object]:
    """Identity, image digest, and the two readings. No coordinates."""

    return {
        "prompt_version": PROMPT_VERSION,
        "candidate_id": case.candidate_id,
        "dispute_id": case.dispute_id,
        "document_id": case.document_id,
        "page_id": case.page_id,
        "page_image_sha256": case.page_image_sha256,
        "page_width": case.page_width,
        "page_height": case.page_height,
        "dispute_category": case.dispute_category,
        "gpt_text": case.gpt_text,
        "grok_text": case.grok_text,
    }


def render_prompt(case: PilotCase) -> str:
    """Instructions plus the dispute readings. Reference boxes are not added."""

    lines = [
        PROMPT_VERSION,
        "Locate the disputed reading on the attached page image.",
        "Reply with JSON only, matching the output schema.",
        "status is PROPOSED, AMBIGUOUS, or NOT_FOUND.",
        "PROPOSED means one or more pixel boxes.",
        "AMBIGUOUS means at least two pixel boxes and that none was chosen.",
        "NOT_FOUND means the boxes array is empty.",
        "Each box is pixels from the top-left of this image: x, y, width, height.",
        "x and y are whole pixels, zero or greater. width and height are at least 1.",
        "The box must lie inside this page. Do not use fractions or millionths.",
        "Do not decide which reading is correct.",
        "Echo candidate_id and page_image_sha256 from this request.",
        f"candidate_id: {case.candidate_id}",
        f"dispute_id: {case.dispute_id}",
        f"document_id: {case.document_id}",
        f"page_id: {case.page_id}",
        f"page_image_sha256: {case.page_image_sha256}",
        f"page_width: {case.page_width}",
        f"page_height: {case.page_height}",
        f"dispute_category: {case.dispute_category}",
        "gpt_reading:",
        case.gpt_text,
        "grok_reading:",
        case.grok_text,
    ]
    prompt = "\n".join(lines)
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ReviewIntegrityError("localization prompt exceeds the size bound")
    return prompt


def assert_reference_isolated(
    payload: str, reference: ConfirmedLocationReference
) -> None:
    """Refuse a model payload that contains a confirmed coordinate."""

    for location in reference.locations:
        values = [
            number
            for box in location.boxes
            for number in (box.x, box.y, box.width, box.height)
        ]
        values.extend(
            number
            for measured in location.measured_pixels
            for number in (measured.x, measured.y, measured.width, measured.height)
        )
        for value in values:
            if re.search(rf"(?<![0-9A-Fa-f]){value}(?![0-9A-Fa-f])", payload):
                raise ReviewIntegrityError(
                    "model input contains a confirmed reference coordinate"
                )


def build_confirmed_reference(
    proposal_bytes: bytes,
    sidecar: VisualProposalSidecar,
    confirmations: Sequence[LocationConfirmation],
) -> ConfirmedLocationReference:
    """Copy proposal boxes and confirm them only from a recorded human action.

    A proposal file alone is not confirmation. The reading stays unverified.
    """

    copied = _copied_pilot_proposals(sidecar)
    by_candidate = _confirmation_index(confirmations)
    if set(by_candidate) != {result.candidate_id for result, _measured in copied}:
        raise ReviewIntegrityError(
            "a confirmed location requires a recorded human confirmation"
        )
    locations: list[ConfirmedLocation] = []
    for result, measured in copied:
        confirmation = by_candidate[result.candidate_id]
        if (
            confirmation.dispute_id != result.dispute_id
            or confirmation.document_id != result.document_id
            or confirmation.page_id != result.page_id
            or confirmation.page_image_sha256 != result.page_image_sha256
            or list(confirmation.boxes) != list(result.boxes)
        ):
            raise ReviewIntegrityError(
                "human confirmation does not match the copied proposal"
            )
        locations.append(
            ConfirmedLocation(
                candidate_id=result.candidate_id,
                dispute_id=result.dispute_id,
                document_id=result.document_id,
                page_id=result.page_id,
                page_image_sha256=result.page_image_sha256,
                page_width=result.page_width,
                page_height=result.page_height,
                dispute_category=result.dispute_category,
                boxes=list(result.boxes),
                measured_pixels=measured,
                source_method_version=result.provenance.method_version,
            )
        )
    order = {key: index for index, key in enumerate(PILOT_DISPUTES)}
    locations.sort(
        key=lambda item: order[(item.document_id, item.page_id, item.dispute_category)]
    )
    return ConfirmedLocationReference(
        source_proposal_sha256=sha256_bytes(proposal_bytes),
        locations=locations,
    )


def _copied_pilot_proposals(
    sidecar: VisualProposalSidecar,
) -> list[tuple[VisualProposal, list[MeasuredPixels]]]:
    if len(sidecar.results) != len(PILOT_DISPUTES):
        raise ReviewIntegrityError(
            "confirmed reference requires the three pilot proposals"
        )
    seen: set[tuple[int, str, str]] = set()
    copied: list[tuple[VisualProposal, list[MeasuredPixels]]] = []
    for result in sidecar.results:
        key = (result.document_id, result.page_id, result.dispute_category)
        if not is_pilot_dispute(*key):
            raise ReviewIntegrityError(
                "confirmed reference contains a case outside the three pilot disputes"
            )
        if key in seen:
            raise ReviewIntegrityError("confirmed reference repeats a pilot dispute")
        seen.add(key)
        if result.status != "PROPOSED" or result.method != "MODEL_PROPOSAL":
            raise ReviewIntegrityError(
                "a confirmed location is copied only from a proposed box"
            )
        copied.append((result, _measured_pixels(result)))
    if seen != set(PILOT_DISPUTES):
        raise ReviewIntegrityError(
            "confirmed reference is missing one of the three pilot disputes"
        )
    return copied


def _confirmation_index(
    confirmations: Sequence[LocationConfirmation],
) -> dict[str, LocationConfirmation]:
    found: dict[str, LocationConfirmation] = {}
    for item in confirmations:
        if item.candidate_id in found:
            raise ReviewIntegrityError("human confirmation repeats a candidate")
        found[item.candidate_id] = item
    return found


def _within_rounding_tolerance(
    stored: NormalizedBox,
    expected: NormalizedBox,
    page_width: int,
    page_height: int,
) -> bool:
    """True when each side of the rectangle is at most one pixel away.

    ``expected`` is the normalized form of the measured pixels. Truncation
    can move one side by less than a source pixel. The left edge and the
    right edge are compared on their own, and the same for top and bottom,
    so a one-pixel shift of the origin plus a one-pixel shift of the size
    cannot hide a two-pixel move of the far edge.
    """

    tolerance_x = (
        PIXEL_ROUNDING_TOLERANCE * NORMALIZED_SCALE + page_width - 1
    ) // page_width
    tolerance_y = (
        PIXEL_ROUNDING_TOLERANCE * NORMALIZED_SCALE + page_height - 1
    ) // page_height
    return (
        abs(stored.x - expected.x) <= tolerance_x
        and abs((stored.x + stored.width) - (expected.x + expected.width))
        <= tolerance_x
        and abs(stored.y - expected.y) <= tolerance_y
        and abs((stored.y + stored.height) - (expected.y + expected.height))
        <= tolerance_y
    )


def load_pilot_cases(
    review_dirs: Sequence[Path], batches_dir: Path | None
) -> list[PilotCase]:
    """Load the three pilot disputes and their original page files."""

    if not review_dirs:
        raise ReviewUsageError("at least one reviews directory is required")
    roots = [_reviews_root(path) for path in review_dirs]
    sealed = _reviews_root(batches_dir) if batches_dir is not None else None
    export = export_learning_candidates(
        roots,
        batches_dir=sealed,
        document_ids=PHASE1_DOCUMENT_IDS,
        dry_run=True,
    )
    grouped: dict[tuple[int, str, str], PilotCase] = {}
    for item in select_phase1_queue(export, document_ids=PHASE1_DOCUMENT_IDS):
        candidate = item.candidate
        key = (candidate.document_id, candidate.page_id, candidate.dispute_category)
        if not is_pilot_dispute(*key):
            continue
        if key in grouped:
            raise ReviewIntegrityError("pilot dispute is not unique")
        image = locate_original_page(roots, candidate)
        grouped[key] = case_from_candidate(
            candidate,
            page_width=item.page.width,
            page_height=item.page.height,
            image_path=image,
        )
    if set(grouped) != set(PILOT_DISPUTES):
        raise ReviewIntegrityError("the three pilot disputes were not all found")
    return [grouped[key] for key in PILOT_DISPUTES]


def case_from_candidate(
    candidate: LearningCandidate,
    *,
    page_width: int,
    page_height: int,
    image_path: Path,
) -> PilotCase:
    """Copy dispute readings and page identity. A recorded image box is not copied."""

    return PilotCase(
        candidate_id=candidate.candidate_id,
        dispute_id=candidate.dispute_id,
        document_id=candidate.document_id,
        page_id=candidate.page_id,
        page_image_sha256=candidate.page_image_sha256,
        page_width=page_width,
        page_height=page_height,
        dispute_category=candidate.dispute_category,
        gpt_text=candidate.dispute_gpt_text,
        grok_text=candidate.dispute_grok_text,
        image_path=str(image_path),
    )


def run_localization(
    cases: Sequence[PilotCase],
    output_dir: Path,
    *,
    execute: bool,
    executor: VisionRunner | None = None,
    discover_executable: Callable[[], Path | None] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    protected_roots: Sequence[Path] = (),
) -> DryRunPlan | LocalizationRunRecord:
    """Write a dry plan, or one result per case. Existing output is left in place."""

    if not cases:
        raise ReviewUsageError("localization run has no cases")
    if timeout_seconds < 1 or timeout_seconds > 600:
        raise ReviewUsageError("timeout must be from 1 to 600 seconds")
    checked = [_checked_case(case) for case in cases]
    discover = discover_executable or discover_codex_executable
    executable = None if executor is not None else discover()
    if execute and executor is None and executable is None:
        raise ReviewUsageError(
            "no Codex executable was found. Nothing was sent and nothing was written."
        )
    target = _claim_output_dir(output_dir, protected_roots)
    if not execute:
        plan = DryRunPlan(
            timeout_seconds=timeout_seconds,
            codex_executable=None if executable is None else str(executable),
            codex_available=executable is not None,
            cases=[_dry_case(case) for case in checked],
        )
        _create_new(target / "dry_run_plan.json", canonical_json(plan).encode("utf-8"))
        return plan
    runner = executor
    if runner is None:
        if executable is None:
            raise ReviewUsageError(
                "no Codex executable was found. Nothing was sent and nothing was written."
            )
        runner = CodexCliRunner(
            executable=executable,
            output_dir=target,
            timeout_seconds=timeout_seconds,
        )
    _create_new(
        target / "model_output_schema.json",
        _OUTPUT_SCHEMA.encode("utf-8"),
    )
    results: list[VisualProposal] = []
    records: list[RunCaseRecord] = []
    tool = "codex-exec"
    model_id = "codex-cli-default"
    for case in checked:
        prompt = render_prompt(case)
        request_sha = _request_sha(case)
        call = ModelCall(
            candidate_id=case.candidate_id,
            image_path=case.image_path,
            prompt=prompt,
            request_sha256=request_sha,
            prompt_sha256=sha256_text(prompt),
        )
        try:
            response = runner.run(call)
        except (LocalizationRunError, subprocess.TimeoutExpired, OSError) as exc:
            reason = _clip(str(exc))
            results.append(_error_proposal(case, request_sha, reason, tool, model_id))
            records.append(
                RunCaseRecord(
                    candidate_id=case.candidate_id,
                    request_sha256=request_sha,
                    prompt_sha256=call.prompt_sha256,
                    status="ERROR",
                )
            )
            continue
        tool = response.tool
        model_id = response.model_id
        proposal = interpret_model_output(case, response.text, response)
        results.append(proposal)
        records.append(
            RunCaseRecord(
                candidate_id=case.candidate_id,
                request_sha256=request_sha,
                prompt_sha256=call.prompt_sha256,
                status=proposal.status,
            )
        )
    record = LocalizationRunRecord(
        timeout_seconds=timeout_seconds,
        tool=tool,
        model_id=model_id,
        cases=records,
        results=VisualProposalSidecar(results=results),
    )
    _create_new(
        target / "localization_run.json",
        canonical_json(record).encode("utf-8"),
    )
    return record


def extract_model_json(payload: str) -> str:
    """Return the JSON object from a final message, including a fenced one."""

    if len(payload) > MAX_MODEL_RESPONSE_BYTES:
        raise LocalizationRunError("model response exceeds the capture limit")
    text = payload.strip().lstrip("\ufeff")
    if text.startswith("{") and text.endswith("}"):
        return text
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced is not None:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def decode_model_payload(data: bytes) -> str:
    """Read a last-message file saved as UTF-8 or UTF-16."""

    if len(data) > MAX_MODEL_RESPONSE_BYTES:
        raise LocalizationRunError("model response exceeds the capture limit")
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    return data.decode("utf-8")


def interpret_model_output(
    case: PilotCase, payload: str, response: ModelResponse
) -> VisualProposal:
    """Validate one model payload. A bad payload becomes one ERROR result."""

    request_sha = _request_sha(case)
    try:
        parsed = ModelLocalization.model_validate_json(extract_model_json(payload))
    except LocalizationRunError as exc:
        return _error_proposal(
            case,
            request_sha,
            _clip(str(exc)),
            response.tool,
            response.model_id,
        )
    except ValidationError:
        return _error_proposal(
            case,
            request_sha,
            "model output did not match the localization schema",
            response.tool,
            response.model_id,
        )
    if parsed.candidate_id != case.candidate_id:
        return _error_proposal(
            case,
            request_sha,
            "model output names a different candidate",
            response.tool,
            response.model_id,
        )
    if parsed.page_image_sha256 != case.page_image_sha256:
        return _error_proposal(
            case,
            request_sha,
            "model output names a different page image",
            response.tool,
            response.model_id,
        )
    if parsed.status == "PROPOSED" and not parsed.boxes:
        return _error_proposal(
            case,
            request_sha,
            "a proposed location needs at least one box",
            response.tool,
            response.model_id,
        )
    if parsed.status == "AMBIGUOUS" and len(parsed.boxes) < 2:
        return _error_proposal(
            case,
            request_sha,
            "an ambiguous location needs at least two boxes",
            response.tool,
            response.model_id,
        )
    if parsed.status == "NOT_FOUND" and parsed.boxes:
        return _error_proposal(
            case,
            request_sha,
            "a not-found result included a box",
            response.tool,
            response.model_id,
        )
    try:
        boxes = [
            normalize_measured_pixels(
                x=box.x,
                y=box.y,
                width=box.width,
                height=box.height,
                page_width=case.page_width,
                page_height=case.page_height,
            )
            for box in parsed.boxes
        ]
        return VisualProposal(
            candidate_id=case.candidate_id,
            dispute_id=case.dispute_id,
            document_id=case.document_id,
            page_id=case.page_id,
            page_image_sha256=case.page_image_sha256,
            page_width=case.page_width,
            page_height=case.page_height,
            dispute_category=case.dispute_category,
            status=parsed.status,
            method="MODEL_PROPOSAL",
            provenance=_provenance(
                case,
                request_sha,
                response.tool,
                response.model_id,
                executed=True,
            ),
            boxes=boxes,
            explanation=_status_explanation(parsed.status),
        )
    except (ReviewIntegrityError, ValidationError, ValueError) as exc:
        return _error_proposal(
            case,
            request_sha,
            _clip(str(exc)),
            response.tool,
            response.model_id,
        )


def evaluate_locations(
    reference: ConfirmedLocationReference,
    results: Sequence[VisualProposal],
    *,
    origin: ResultOrigin,
    images: Mapping[str, Path] | None = None,
    inspection_dir: Path | None = None,
) -> EvaluationReport:
    """Score boxes against the confirmed reference. Ambiguous and absent stay so."""

    by_id = {item.candidate_id: item for item in results}
    if len(by_id) != len(results):
        raise ReviewIntegrityError("evaluation results repeat a candidate")
    rows: list[CaseEvaluation] = []
    for location in reference.locations:
        proposal = by_id.get(location.candidate_id)
        rows.append(
            _evaluate_case(
                location,
                proposal,
                images=images,
                inspection_dir=inspection_dir,
            )
        )
    return EvaluationReport(
        origin=origin,
        live_model_executed=origin == "live_model",
        cases=rows,
    )


def discover_codex_executable() -> Path | None:
    """Find a Codex CLI on PATH or in the per-user Windows install. Do not read auth."""

    found = shutil.which("codex") or shutil.which("codex.exe")
    if found:
        return Path(found)
    roots: list[Path] = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots.append(Path(local) / "OpenAI" / "Codex" / "bin")
    users = Path("/mnt/c/Users")
    if users.is_dir():
        for user_dir in users.iterdir():
            if user_dir.name.startswith("."):
                continue
            roots.append(user_dir / "AppData" / "Local" / "OpenAI" / "Codex" / "bin")
    matches: list[Path] = []
    for root in roots:
        if root.is_dir():
            matches.extend(path for path in root.glob("*/codex.exe") if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def codex_exec_argv(
    *,
    executable: Path,
    image: Path,
    schema_path: Path,
    last_message: Path,
    prompt: str,
    path_for_tool: Callable[[Path], str],
) -> list[str]:
    """One ephemeral exec. No resume, no model override, no reference file."""

    return [
        str(executable),
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--json",
        "--color",
        "never",
        "-s",
        "read-only",
        "-i",
        path_for_tool(image),
        "--output-schema",
        path_for_tool(schema_path),
        "-o",
        path_for_tool(last_message),
        prompt,
    ]


class CodexCliRunner:
    """Dispatch one ``codex exec`` and read its final message file."""

    def __init__(
        self,
        *,
        executable: Path,
        output_dir: Path,
        timeout_seconds: int,
        path_for_tool: Callable[[Path], str] | None = None,
        stage_root: Path | None = None,
    ) -> None:
        self.executable = executable
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.path_for_tool = path_for_tool or _path_for_executable(executable)
        self.stage_root = stage_root

    def run(self, call: ModelCall) -> ModelResponse:
        schema_path = self.output_dir / "model_output_schema.json"
        if not schema_path.is_file():
            raise LocalizationRunError("model output schema is missing")
        calls = self.output_dir / "calls"
        calls.mkdir(parents=True, exist_ok=True)
        ident = _safe_id(call.candidate_id)
        last_message = calls / f"{ident}.last.txt"
        if last_message.exists():
            raise LocalizationRunError("refusing to overwrite a model response")
        image_path, schema_for_tool, message_for_tool, workdir = prepare_tool_paths(
            executable=self.executable,
            image=Path(call.image_path),
            schema_path=schema_path,
            last_message=last_message,
            stage_root=self.stage_root,
        )
        owned_stage = workdir if _windows_executable(self.executable) else None
        try:
            argv = codex_exec_argv(
                executable=self.executable,
                image=image_path,
                schema_path=schema_for_tool,
                last_message=message_for_tool,
                prompt=call.prompt,
                path_for_tool=self.path_for_tool,
            )
            _create_new(
                calls / f"{ident}.argv.txt", ("\n".join(argv) + "\n").encode("utf-8")
            )
            try:
                completed = _run_bounded_process(
                    argv, timeout=self.timeout_seconds, cwd=workdir
                )
            except subprocess.TimeoutExpired as exc:
                _store_streams(
                    calls,
                    ident,
                    _bounded_bytes(getattr(exc, "stdout", None)),
                    _bounded_bytes(getattr(exc, "stderr", None)),
                )
                _capture_staged_message(message_for_tool, last_message, calls, ident)
                raise LocalizationRunError(
                    f"codex exec timed out after {self.timeout_seconds} seconds"
                ) from exc
            except BoundedCaptureError as exc:
                _store_streams(calls, ident, exc.stdout, exc.stderr)
                _capture_staged_message(message_for_tool, last_message, calls, ident)
                raise LocalizationRunError(
                    "codex output exceeded the capture limit"
                ) from exc
            _store_streams(calls, ident, completed.stdout, completed.stderr)
            _capture_staged_message(message_for_tool, last_message, calls, ident)
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")
                raise LocalizationRunError(f"codex exec failed: {_clip(detail)}")
            if not last_message.is_file():
                raise LocalizationRunError("codex exec wrote no final message")
            return ModelResponse(
                text=decode_model_payload(_read_bounded_file(last_message)),
                model_id=_model_id_from_events(completed.stdout),
                tool="codex-exec",
            )
        finally:
            if owned_stage is not None:
                _remove_owned_stage(owned_stage, owned_stage.parent)


def write_reference_files(
    proposal_path: Path,
    output_dir: Path,
    *,
    confirmations: Sequence[LocationConfirmation],
    images: Mapping[str, Path] | None = None,
    protected_roots: Sequence[Path] = (),
) -> ConfirmedLocationReference:
    """Read proposals and write a new reference. The proposal file is not opened for write.

    An existing output directory is left unchanged. Confirmation is taken from
    ``confirmations`` and is not invented from the proposal.
    """

    source = proposal_path.expanduser().resolve()
    before = source.read_bytes()
    try:
        sidecar = VisualProposalSidecar.model_validate_json(before)
    except ValidationError as exc:
        raise ReviewIntegrityError(
            "visual proposal sidecar does not match the schema"
        ) from exc
    reference = build_confirmed_reference(before, sidecar, confirmations)
    if images is not None:
        _check_reference_images(reference, images)
    target = _claim_output_dir(output_dir, protected_roots)
    _create_new(
        target / "reference.json",
        canonical_json(reference).encode("utf-8"),
    )
    if images is not None:
        _write_location_inspection(
            reference.locations,
            images,
            target / "inspection",
            proposal_boxes={},
        )
    if source.read_bytes() != before:
        raise ReviewIntegrityError(
            "proposal sidecar changed while copying the reference"
        )
    return reference


def write_evaluation_files(
    reference_path: Path,
    results_path: Path,
    output_dir: Path,
    *,
    origin: ResultOrigin,
    images: Mapping[str, Path] | None = None,
    protected_roots: Sequence[Path] = (),
) -> EvaluationReport:
    """Write the overlap report and inspection crops beside it."""

    reference = ConfirmedLocationReference.model_validate_json(
        reference_path.read_text(encoding="utf-8")
    )
    results = _load_results(results_path)
    target = _claim_output_dir(output_dir, protected_roots)
    inspection = target / "inspection" if images is not None else None
    report = evaluate_locations(
        reference,
        results,
        origin=origin,
        images=images,
        inspection_dir=inspection,
    )
    _create_new(target / "evaluation.json", canonical_json(report).encode("utf-8"))
    return report


def select_cases(
    cases: Sequence[PilotCase], requested: Sequence[str] | None
) -> list[PilotCase]:
    """Keep the pilot order. An unknown or repeated candidate is refused."""

    if not requested:
        return list(cases)
    if len(set(requested)) != len(requested):
        raise ReviewUsageError("a localization candidate was repeated")
    chosen = [case for case in cases if case.candidate_id in set(requested)]
    if len(chosen) != len(requested):
        raise ReviewUsageError(
            "a requested candidate is not one of the three pilot disputes"
        )
    return chosen


def _fixed_executable(path: Path) -> Callable[[], Path | None]:
    def discover() -> Path | None:
        return path

    return discover


def main(argv: list[str] | None = None) -> int:
    """Copy a reference, dry-run a localization, or score boxes. No default model call."""

    parser = argparse.ArgumentParser(prog="reproducible-localization", add_help=False)
    parser.add_argument("operation", choices=("reference", "run", "evaluate"))
    parser.add_argument("--proposals")
    parser.add_argument("--confirmations")
    parser.add_argument("--reviews-dir", action="append")
    parser.add_argument("--batches-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--reference")
    parser.add_argument("--results")
    parser.add_argument(
        "--origin",
        choices=("mock", "live_model", "prior_proposal"),
    )
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--codex")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise ReviewUsageError(
            "reproducible-localization usage was not recognized. Nothing was written."
        ) from exc
    if not args.output_dir:
        raise ReviewUsageError("reproducible-localization requires --output-dir")
    reviews = [Path(path) for path in args.reviews_dir] if args.reviews_dir else []
    batches = Path(args.batches_dir) if args.batches_dir else None
    protected = [*reviews, *([batches] if batches is not None else [])]
    images = _image_index(reviews, batches) if reviews else None
    if args.operation == "reference":
        if not args.proposals or not args.confirmations:
            raise ReviewUsageError("reference requires --proposals and --confirmations")
        _refuse_sidecar_path(Path(args.proposals), protected)
        confirmation_path = Path(args.confirmations)
        _refuse_sidecar_path(confirmation_path, protected)
        try:
            confirmation_record = LocationConfirmationRecord.model_validate_json(
                confirmation_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ReviewIntegrityError(
                "location confirmation does not match the schema"
            ) from exc
        write_reference_files(
            Path(args.proposals),
            Path(args.output_dir),
            confirmations=confirmation_record.confirmations,
            images=images,
            protected_roots=protected,
        )
        print(str(Path(args.output_dir)))
        return 0
    if args.operation == "run":
        if not reviews:
            raise ReviewUsageError("run requires --reviews-dir")
        cases = select_cases(load_pilot_cases(reviews, batches), args.candidate)
        discover = _fixed_executable(Path(args.codex)) if args.codex else None
        run_localization(
            cases,
            Path(args.output_dir),
            execute=args.execute,
            discover_executable=discover,
            timeout_seconds=args.timeout_seconds,
            protected_roots=protected,
        )
        print(str(Path(args.output_dir)))
        return 0
    if not args.reference or not args.results or not args.origin:
        raise ReviewUsageError("evaluate requires --reference, --results, and --origin")
    write_evaluation_files(
        Path(args.reference),
        Path(args.results),
        Path(args.output_dir),
        origin=args.origin,
        images=images,
        protected_roots=protected,
    )
    print(str(Path(args.output_dir)))
    return 0


def _measured_pixels(result: VisualProposal) -> list[MeasuredPixels]:
    if len(result.boxes) != 1 or not result.explanation:
        return []
    match = _MEASURED_PIXELS.search(result.explanation)
    if match is None:
        return []
    x, y, width, height, page_width, page_height = (
        int(item) for item in match.groups()
    )
    if page_width != result.page_width or page_height != result.page_height:
        raise ReviewIntegrityError("measured pixels name a different page size")
    normalized = normalize_measured_pixels(
        x=x,
        y=y,
        width=width,
        height=height,
        page_width=page_width,
        page_height=page_height,
    )
    if normalized != result.boxes[0]:
        raise ReviewIntegrityError("measured pixels do not match the stored box")
    return [
        MeasuredPixels(
            x=x,
            y=y,
            width=width,
            height=height,
            page_width=page_width,
            page_height=page_height,
        )
    ]


def _checked_case(case: PilotCase) -> PilotCase:
    path = Path(case.image_path)
    if not path.is_file():
        raise ReviewIntegrityError(f"page image is missing for {case.candidate_id}")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != case.page_image_sha256:
        raise ReviewIntegrityError(
            f"page image hash does not match candidate {case.candidate_id}"
        )
    width, height = _image_dimensions(data)
    if width != case.page_width or height != case.page_height:
        raise ReviewIntegrityError(
            f"page image dimensions do not match candidate {case.candidate_id}"
        )
    return case


def _image_dimensions(data: bytes) -> tuple[int, int]:
    if data.startswith(b"\xff\xd8"):
        return jpeg_dimensions(data)
    if data.startswith(PNG_SIGNATURE):
        width, height, _rows, _bpp = decode_png(data)
        return width, height
    raise ReviewIntegrityError("page image is not a JPEG or PNG")


def _dry_case(case: PilotCase) -> DryRunCase:
    prompt = render_prompt(case)
    return DryRunCase(
        candidate_id=case.candidate_id,
        document_id=case.document_id,
        page_id=case.page_id,
        dispute_category=case.dispute_category,
        page_image_sha256=case.page_image_sha256,
        page_width=case.page_width,
        page_height=case.page_height,
        request_sha256=_request_sha(case),
        prompt_sha256=sha256_text(prompt),
    )


def _request_sha(case: PilotCase) -> str:
    return prepare_request_sha256(
        candidate_id=case.candidate_id,
        dispute_id=case.dispute_id,
        document_id=case.document_id,
        page_id=case.page_id,
        page_image_sha256=case.page_image_sha256,
        gpt_text=case.gpt_text,
        grok_text=case.grok_text,
    )


def _provenance(
    case: PilotCase,
    request_sha: str,
    tool: str,
    model_id: str,
    *,
    executed: bool,
) -> ProposalProvenance:
    return ProposalProvenance(
        image_sha256=case.page_image_sha256,
        method_version=RUNNER_METHOD,
        model_id=model_id,
        tool=tool,
        request_sha256=request_sha,
        executed=executed,
    )


def _error_proposal(
    case: PilotCase,
    request_sha: str,
    reason: str,
    tool: str,
    model_id: str,
) -> VisualProposal:
    return VisualProposal(
        candidate_id=case.candidate_id,
        dispute_id=case.dispute_id,
        document_id=case.document_id,
        page_id=case.page_id,
        page_image_sha256=case.page_image_sha256,
        page_width=case.page_width,
        page_height=case.page_height,
        dispute_category=case.dispute_category,
        status="ERROR",
        method="MODEL_PROPOSAL",
        provenance=_provenance(case, request_sha, tool, model_id, executed=True),
        boxes=[],
        explanation=_clip(reason) or "model output failed",
    )


def _status_explanation(status: str) -> str:
    if status == "AMBIGUOUS":
        return (
            "The model returned more than one box and did not choose one. Not verified."
        )
    if status == "NOT_FOUND":
        return "The model returned no box. Not verified."
    return "Unverified pixel boxes from one localization request. This does not confirm the reading."


def _evaluate_case(
    location: ConfirmedLocation,
    proposal: VisualProposal | None,
    *,
    images: Mapping[str, Path] | None,
    inspection_dir: Path | None,
) -> CaseEvaluation:
    if proposal is None:
        return CaseEvaluation(
            candidate_id=location.candidate_id,
            document_id=location.document_id,
            page_id=location.page_id,
            result_status="NO_RESULT",
            outcome="NO_RESULT",
            best_iou=None,
            pairs=[],
            inspection=_inspection_paths(
                location, None, images=images, inspection_dir=inspection_dir
            ),
        )
    if not _same_page(location, proposal):
        return CaseEvaluation(
            candidate_id=location.candidate_id,
            document_id=location.document_id,
            page_id=location.page_id,
            result_status="ERROR",
            outcome="ERROR",
            best_iou=None,
            pairs=[],
            inspection=[],
        )
    outcome = _outcome(proposal.status)
    pairs: list[PairScore] = []
    best: str | None = None
    if proposal.status in {"PROPOSED", "AMBIGUOUS"} and proposal.boxes:
        pairs = _pair_scores(location.boxes, proposal.boxes)
        best_value: Decimal | None = None
        for item in pairs:
            value = Decimal(item.iou)
            if best_value is None or value > best_value:
                best_value = value
                best = item.iou
    return CaseEvaluation(
        candidate_id=location.candidate_id,
        document_id=location.document_id,
        page_id=location.page_id,
        result_status=proposal.status,
        outcome=outcome,
        best_iou=best,
        pairs=pairs,
        inspection=_inspection_paths(
            location,
            proposal,
            images=images,
            inspection_dir=inspection_dir,
        ),
    )


def _same_page(location: ConfirmedLocation, proposal: VisualProposal) -> bool:
    return (
        proposal.dispute_id == location.dispute_id
        and proposal.document_id == location.document_id
        and proposal.page_id == location.page_id
        and proposal.page_image_sha256 == location.page_image_sha256
        and proposal.page_width == location.page_width
        and proposal.page_height == location.page_height
        and proposal.dispute_category == location.dispute_category
    )


def _outcome(status: str) -> Outcome:
    if status == "PROPOSED":
        return "SCORED"
    if status == "AMBIGUOUS":
        return "AMBIGUOUS"
    if status == "NOT_FOUND":
        return "NOT_FOUND"
    return "ERROR"


def _pair_scores(
    reference: Sequence[NormalizedBox], proposed: Sequence[NormalizedBox]
) -> list[PairScore]:
    scores: list[PairScore] = []
    for ref_index, ref_box in enumerate(reference):
        for prop_index, prop_box in enumerate(proposed):
            scores.append(
                PairScore(
                    reference_index=ref_index,
                    proposal_index=prop_index,
                    iou=_iou_text(ref_box, prop_box),
                )
            )
    return scores


def _iou_text(left: NormalizedBox, right: NormalizedBox) -> str:
    intersection = _intersection_area(left, right)
    union = left.width * left.height + right.width * right.height - intersection
    if union <= 0:
        return "0.000000"
    quantized = (Decimal(intersection) / Decimal(union)).quantize(
        Decimal("0.000001"),
        rounding=ROUND_HALF_UP,
    )
    return f"{quantized:.6f}"


def _intersection_area(left: NormalizedBox, right: NormalizedBox) -> int:
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    if x1 <= x0 or y1 <= y0:
        return 0
    return (x1 - x0) * (y1 - y0)


def _inspection_paths(
    location: ConfirmedLocation,
    proposal: VisualProposal | None,
    *,
    images: Mapping[str, Path] | None,
    inspection_dir: Path | None,
) -> list[str]:
    if images is None or inspection_dir is None:
        return []
    image = images.get(location.page_image_sha256)
    if image is None:
        return []
    written = _write_case_inspection(location, proposal, image, inspection_dir)
    return [path.name for path in written]


def _write_location_inspection(
    locations: Sequence[ConfirmedLocation],
    images: Mapping[str, Path],
    inspection_dir: Path,
    *,
    proposal_boxes: Mapping[str, VisualProposal],
) -> None:
    inspection_dir.mkdir(parents=True, exist_ok=True)
    for location in locations:
        image = images.get(location.page_image_sha256)
        if image is None:
            raise ReviewIntegrityError(
                f"no page image for confirmed location {location.candidate_id}"
            )
        _write_case_inspection(
            location,
            proposal_boxes.get(location.candidate_id),
            image,
            inspection_dir,
        )


def _write_case_inspection(
    location: ConfirmedLocation,
    proposal: VisualProposal | None,
    image_path: Path,
    inspection_dir: Path,
) -> list[Path]:
    data = image_path.read_bytes()
    if sha256_bytes(data) != location.page_image_sha256:
        raise ReviewIntegrityError("inspection image hash does not match the reference")
    width, height, rows, bpp = _decode_page(data)
    if width != location.page_width or height != location.page_height:
        raise ReviewIntegrityError(
            "inspection image dimensions do not match the reference"
        )
    inspection_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    reference_pixels = [
        _reference_pixels(location, index) for index in range(len(location.boxes))
    ]
    for index, pixels in enumerate(reference_pixels):
        written.append(
            _write_crop(
                rows,
                pixels,
                bpp,
                width,
                height,
                inspection_dir / f"{location.candidate_id}_reference_crop_{index}.png",
            )
        )
    proposal_pixels: list[tuple[int, int, int, int]] = []
    if proposal is not None and proposal.boxes:
        proposal_pixels = [_denormalize(box, width, height) for box in proposal.boxes]
        for index, pixels in enumerate(proposal_pixels):
            written.append(
                _write_crop(
                    rows,
                    pixels,
                    bpp,
                    width,
                    height,
                    inspection_dir
                    / f"{location.candidate_id}_proposal_crop_{index}.png",
                )
            )
    page = _scaled_page(
        rows, width, height, bpp, reference_pixels, proposal_pixels, _QUARTER
    )
    page_path = inspection_dir / f"{location.candidate_id}_page_quarter.png"
    _create_new(page_path, page)
    written.append(page_path)
    overlay = _scaled_page(
        rows, width, height, bpp, reference_pixels, proposal_pixels, 2
    )
    overlay_path = inspection_dir / f"{location.candidate_id}_page_overlay.png"
    _create_new(overlay_path, overlay)
    written.append(overlay_path)
    return written


def _reference_pixels(
    location: ConfirmedLocation, index: int
) -> tuple[int, int, int, int]:
    if index < len(location.measured_pixels):
        measured = location.measured_pixels[index]
        return measured.x, measured.y, measured.width, measured.height
    return _denormalize(
        location.boxes[index], location.page_width, location.page_height
    )


def _denormalize(
    box: NormalizedBox, page_width: int, page_height: int
) -> tuple[int, int, int, int]:
    return (
        (box.x * page_width) // NORMALIZED_SCALE,
        (box.y * page_height) // NORMALIZED_SCALE,
        max(1, (box.width * page_width) // NORMALIZED_SCALE),
        max(1, (box.height * page_height) // NORMALIZED_SCALE),
    )


def _decode_page(data: bytes) -> tuple[int, int, list[bytearray], int]:
    if data.startswith(b"\xff\xd8"):
        return decode_jpeg(data)
    if data.startswith(PNG_SIGNATURE):
        return decode_png(data)
    raise ReviewIntegrityError("inspection image is not a JPEG or PNG")


def _write_crop(
    rows: list[bytearray],
    pixels: tuple[int, int, int, int],
    bpp: int,
    page_width: int,
    page_height: int,
    path: Path,
) -> Path:
    x, y, width, height = pixels
    cropped = crop_rows(
        rows,
        x=x,
        y=y,
        width=width,
        height=height,
        bpp=bpp,
        page_width=page_width,
        page_height=page_height,
    )
    _create_new(path, encode_png(width, height, cropped, bpp))
    return path


def _scaled_page(
    rows: list[bytearray],
    width: int,
    height: int,
    bpp: int,
    reference_pixels: Sequence[tuple[int, int, int, int]],
    proposal_pixels: Sequence[tuple[int, int, int, int]],
    divisor: int,
) -> bytes:
    stroked = [bytearray(row) for row in rows]
    thickness = max(4, min(width, height) // 100)
    for pixels in reference_pixels:
        _stroke(stroked, pixels, bpp, (32, 140, 70), thickness)
    for pixels in proposal_pixels:
        _stroke(stroked, pixels, bpp, (190, 40, 40), thickness)
    out_w = max(1, width // divisor)
    out_h = max(1, height // divisor)
    reduced: list[bytearray] = []
    for y in range(out_h):
        source = stroked[min(height - 1, y * divisor)]
        row = bytearray()
        for x in range(out_w):
            start = min(width - 1, x * divisor) * bpp
            row.extend(source[start : start + bpp])
        reduced.append(row)
    return encode_png(out_w, out_h, reduced, bpp)


def _stroke(
    rows: list[bytearray],
    pixels: tuple[int, int, int, int],
    bpp: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x, y, width, height = pixels
    page_height = len(rows)
    page_width = len(rows[0]) // bpp if rows else 0
    if width < 1 or height < 1 or thickness < 1:
        return
    paint = bytes((*color, 255)[:bpp])
    x1 = min(page_width, x + width)
    y1 = min(page_height, y + height)
    x0 = min(max(0, x), page_width)
    y0 = min(max(0, y), page_height)
    band = min(thickness, max(1, (y1 - y0) // 2), max(1, (x1 - x0) // 2))
    for row_index in range(y0, y1):
        row = rows[row_index]
        for column in range(x0, x1):
            edge = (
                row_index < y0 + band
                or row_index >= y1 - band
                or column < x0 + band
                or column >= x1 - band
            )
            if edge:
                start = column * bpp
                row[start : start + bpp] = paint


def _check_reference_images(
    reference: ConfirmedLocationReference, images: Mapping[str, Path]
) -> None:
    for location in reference.locations:
        path = images.get(location.page_image_sha256)
        if path is None:
            raise ReviewIntegrityError(
                f"page image for {location.candidate_id} was not found"
            )
        data = path.read_bytes()
        if sha256_bytes(data) != location.page_image_sha256:
            raise ReviewIntegrityError(
                f"page image hash does not match confirmed location {location.candidate_id}"
            )


def _image_index(
    review_dirs: Sequence[Path], batches_dir: Path | None
) -> dict[str, Path]:
    return {
        case.page_image_sha256: Path(case.image_path)
        for case in load_pilot_cases(review_dirs, batches_dir)
    }


def _load_results(path: Path) -> list[VisualProposal]:
    payload = path.read_text(encoding="utf-8")
    try:
        run = LocalizationRunRecord.model_validate_json(payload)
    except ValidationError:
        run = None
    if run is not None:
        return list(run.results.results)
    try:
        sidecar = VisualProposalSidecar.model_validate_json(payload)
    except ValidationError as exc:
        raise ReviewIntegrityError(
            "localization results do not match the schema"
        ) from exc
    return list(sidecar.results)


def _claim_output_dir(path: Path, protected_roots: Sequence[Path]) -> Path:
    resolved = path.expanduser().resolve()
    _refuse_output(resolved, protected_roots)
    if resolved.exists():
        if not resolved.is_dir():
            raise ReviewPathError("localization output path is not a directory")
        if any(resolved.iterdir()):
            raise ReviewIntegrityError("refusing to overwrite localization output")
    else:
        resolved.mkdir(parents=True)
    return resolved


def _refuse_output(path: Path, protected_roots: Sequence[Path]) -> None:
    for root in FORBIDDEN_WRITE_ROOTS:
        forbidden = root.expanduser().resolve()
        if path == forbidden or forbidden in path.parents:
            raise ReviewPathError("refusing to write inside the VS Archive checkout")
    for root in protected_roots:
        forbidden = root.expanduser().resolve()
        if path == forbidden or forbidden in path.parents or path in forbidden.parents:
            raise ReviewPathError(
                "refusing to write localization output inside a review or sealed bundle"
            )


def _refuse_sidecar_path(path: Path, protected_roots: Sequence[Path]) -> None:
    resolved = path.expanduser().resolve()
    for root in protected_roots:
        forbidden = root.expanduser().resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ReviewPathError(
                "refusing to read proposals from a review or sealed bundle"
            )


def _reviews_root(path: Path) -> Path:
    from transcription_review_agent.review.paths import reviews_root

    return reviews_root(path)


def _create_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ReviewIntegrityError(f"refusing to overwrite {path.name}") from exc
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def prepare_tool_paths(
    *,
    executable: Path,
    image: Path,
    schema_path: Path,
    last_message: Path,
    stage_root: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Return the image, schema, last-message, and working directory for one exec.

    A Windows executable cannot be handed a ``\\\\wsl$`` page. The bytes are
    copied onto a drive-letter directory and the original file is not opened
    for writing.
    """

    if not _windows_executable(executable):
        return image, schema_path, last_message, last_message.parent
    root = stage_root if stage_root is not None else _windows_stage_root(executable)
    stage = root / last_message.stem
    if stage.exists():
        raise LocalizationRunError("refusing to reuse a Windows staging directory")
    try:
        stage.mkdir(parents=True)
    except FileExistsError as exc:
        raise LocalizationRunError(
            "refusing to reuse a Windows staging directory"
        ) from exc
    try:
        suffix = (
            image.suffix.lower()
            if image.suffix.lower() in {".jpg", ".jpeg", ".png"}
            else ".img"
        )
        staged_image = stage / f"page{suffix}"
        shutil.copyfile(image, staged_image)
        if staged_image.read_bytes() != image.read_bytes():
            raise LocalizationRunError("staged page image does not match the original")
        staged_schema = stage / "model_output_schema.json"
        shutil.copyfile(schema_path, staged_schema)
    except Exception:
        _remove_owned_stage(stage, root)
        raise
    return staged_image, staged_schema, stage / "last.txt", stage


def _windows_stage_root(executable: Path) -> Path:
    parts = executable.resolve().parts
    if "Users" not in parts or "AppData" not in parts:
        raise LocalizationRunError(
            "could not place a Windows staging directory beside the Codex install"
        )
    users = parts.index("Users")
    root = Path(*parts[: users + 2]) / "AppData" / "Local" / "Temp"
    stage = root / "transcription-localization-staging"
    stage.mkdir(parents=True, exist_ok=True)
    return stage


def _store_streams(
    calls: Path, ident: str, stdout: bytes | None, stderr: bytes | None
) -> None:
    _create_new(calls / f"{ident}.stdout.txt", stdout or b"")
    _create_new(calls / f"{ident}.stderr.txt", stderr or b"")


def _bounded_bytes(value: object) -> bytes:
    if not isinstance(value, bytes):
        return b""
    return value[:MAX_PROCESS_OUTPUT_BYTES]


def _read_bounded_file(path: Path) -> bytes:
    size = path.stat().st_size
    if size > MAX_MODEL_RESPONSE_BYTES:
        raise LocalizationRunError("model response exceeds the capture limit")
    data = path.read_bytes()
    if len(data) > MAX_MODEL_RESPONSE_BYTES:
        raise LocalizationRunError("model response exceeds the capture limit")
    return data


def _capture_staged_message(
    message_for_tool: Path, last_message: Path, calls: Path, ident: str
) -> None:
    """Copy the final message out of a staging directory before that directory goes away."""

    if message_for_tool == last_message or not message_for_tool.is_file():
        return
    size = message_for_tool.stat().st_size
    if size > MAX_MODEL_RESPONSE_BYTES:
        _create_new(
            calls / f"{ident}.response-rejected.txt",
            f"model response bytes: {size}\n".encode("utf-8"),
        )
        raise LocalizationRunError("model response exceeds the capture limit")
    _create_new(last_message, message_for_tool.read_bytes())


def _run_bounded_process(
    argv: Sequence[str], *, timeout: int, cwd: Path
) -> subprocess.CompletedProcess[bytes]:
    """Run one process and keep only a bounded prefix of each stream.

    There is no second attempt. Output past the limit kills the process.
    """

    process = subprocess.Popen(
        list(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    if stdout_stream is None or stderr_stream is None:
        _stop_process(process)
        raise LocalizationRunError("codex output pipes were not opened")
    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()

    def read_stream(stream: BinaryIO, sink: bytearray) -> None:
        while True:
            chunk = stream.read(_READ_CHUNK)
            if not chunk:
                return
            room = MAX_PROCESS_OUTPUT_BYTES - len(sink)
            if room <= 0:
                overflow.set()
                return
            if len(chunk) > room:
                sink.extend(chunk[:room])
                overflow.set()
                return
            sink.extend(chunk)

    threads = (
        threading.Thread(target=read_stream, args=(stdout_stream, stdout), daemon=True),
        threading.Thread(target=read_stream, args=(stderr_stream, stderr), daemon=True),
    )
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while process.poll() is None:
            if overflow.is_set():
                _stop_process(process)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _stop_process(process)
                break
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                continue
        process.wait()
        for thread in threads:
            thread.join(timeout=2)
    finally:
        _stop_process(process)
        stdout_stream.close()
        stderr_stream.close()
    captured_out = bytes(stdout)
    captured_err = bytes(stderr)
    if timed_out:
        raise subprocess.TimeoutExpired(
            list(argv), timeout, output=captured_out, stderr=captured_err
        )
    if overflow.is_set():
        raise BoundedCaptureError(
            "codex output exceeded the capture limit", captured_out, captured_err
        )
    returncode = process.returncode
    if returncode is None:
        raise LocalizationRunError("codex exec ended without a return code")
    return subprocess.CompletedProcess(
        list(argv), returncode, captured_out, captured_err
    )


def _windows_executable(executable: Path) -> bool:
    return executable.name.lower().endswith(".exe")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return
    process.wait()


def _remove_owned_stage(stage: Path, root: Path) -> None:
    """Delete a staging directory this run created. Nothing outside it is removed."""

    if not stage.exists() and not stage.is_symlink():
        return
    if stage.is_symlink() or not stage.is_dir():
        raise LocalizationRunError(
            "refusing to delete a staging path that is not the directory this run created"
        )
    stage_resolved = stage.resolve()
    root_resolved = root.resolve()
    if stage_resolved == root_resolved or stage_resolved.parent != root_resolved:
        raise LocalizationRunError(
            "refusing to delete a staging directory outside its root"
        )
    shutil.rmtree(stage_resolved)


def _model_id_from_events(stdout: bytes) -> str:
    text = stdout.decode("utf-8", errors="replace")
    found = re.findall(r'"model"\s*:\s*"([A-Za-z0-9_.:-]{1,80})"', text)
    if not found:
        return "codex-cli-default"
    return found[-1]


def _path_for_executable(executable: Path) -> Callable[[Path], str]:
    if executable.name.lower().endswith(".exe"):
        return _windows_path
    return _as_text


def _as_text(path: Path) -> str:
    return str(path)


def _windows_path(path: Path) -> str:
    completed = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise LocalizationRunError("could not convert a path for the Windows Codex CLI")
    converted = completed.stdout.strip()
    if not converted or any(mark in converted for mark in "\n\r"):
        raise LocalizationRunError("Windows path conversion returned an unusable path")
    return converted


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"lc_[0-9a-f]{20}", value):
        raise ReviewIntegrityError("candidate id is not a localization id")
    return value


def _clip(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= 240:
        return cleaned
    return cleaned[:237] + "..."


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewUsageError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    except ReviewError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
