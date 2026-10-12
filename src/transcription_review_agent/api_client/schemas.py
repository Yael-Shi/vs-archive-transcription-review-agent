"""Pydantic models for the Pilot 01 scan and review-bundle responses.

These models describe the merged VS Archive contract. They do not fetch it.
HTTP 404 is not a JSON outcome: missing and not-viewable documents are
signaled by status 404, and the body is unspecified. Scan query failures are
HTTP 400 with ``invalid_after_id`` or ``invalid_limit``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from transcription_review_agent.constants import (
    BUNDLE_PATH_TEMPLATE,
    BUNDLE_SCHEMA_VERSION,
    SCAN_SCHEMA_VERSION,
    SHA256_PATTERN,
)

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class BundleOutcome(str, Enum):
    REVIEWABLE = "REVIEWABLE"
    NON_ELIGIBLE = "NON_ELIGIBLE"
    BLOCKING_INTEGRITY_ERROR = "BLOCKING_INTEGRITY_ERROR"


class NonEligibleCode(str, Enum):
    LANGUAGE_NOT_IN_SCOPE = "LANGUAGE_NOT_IN_SCOPE"
    NOT_OCR_DOCUMENT = "NOT_OCR_DOCUMENT"
    NOT_READY_UPLOAD = "NOT_READY_UPLOAD"
    NOT_READY_PROCESSING = "NOT_READY_PROCESSING"
    VERIFIED_TRANSCRIPTION_OUT_OF_SCOPE = "VERIFIED_TRANSCRIPTION_OUT_OF_SCOPE"


class BlockingIntegrityCode(str, Enum):
    MISSING_DISPLAYED_TRANSCRIPTION = "MISSING_DISPLAYED_TRANSCRIPTION"
    MISSING_SOURCE_FILE = "MISSING_SOURCE_FILE"
    INCOMPLETE_MULTI_IMAGE_SOURCES = "INCOMPLETE_MULTI_IMAGE_SOURCES"
    MISSING_PRIMARY_SOURCE = "MISSING_PRIMARY_SOURCE"
    UPLOADED_DISPLAY_ONLY_SOURCE_MISSING_KEY = (
        "UPLOADED_DISPLAY_ONLY_SOURCE_MISSING_KEY"
    )
    UNSUPPORTED_SOURCE_MIME = "UNSUPPORTED_SOURCE_MIME"
    SOURCE_URL_UNAVAILABLE = "SOURCE_URL_UNAVAILABLE"


def bundle_path(document_id: int) -> str:
    """Return the review-bundle URL path. This does not perform a request."""

    if document_id < 1:
        raise ValueError("document_id must be a positive integer")
    return BUNDLE_PATH_TEMPLATE.format(document_id=document_id)


class _ApiModel(BaseModel):
    """Ignore additive server fields. Required fields stay strict."""

    model_config = ConfigDict(extra="ignore")


class DiscoveryCandidate(_ApiModel):
    document_id: int = Field(ge=1)
    archive_item_id: int = Field(ge=1)
    language: Literal["en", "fr"]
    doc_type: str
    item_type: str
    upload_status: str
    processing_state_user: str
    visibility: str
    expected_source_file_count: int | None = Field(default=None, ge=0)


class ScanResponse(_ApiModel):
    schema_version: Literal["transcription-review-scan-v1"] = SCAN_SCHEMA_VERSION
    after_id: int = Field(ge=0)
    limit: int = Field(ge=1, le=200)
    next_after_id: int | None = Field(default=None, ge=1)
    exhausted: bool
    items: list[DiscoveryCandidate]

    @model_validator(mode="after")
    def _page_consistency(self) -> ScanResponse:
        ids = [item.document_id for item in self.items]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("scan items must be strictly ascending document ids")
        if any(document_id <= self.after_id for document_id in ids):
            raise ValueError("scan items must be after after_id")
        if not ids:
            if not self.exhausted or self.next_after_id is not None:
                raise ValueError("an empty scan page is exhausted and has no next_after_id")
            return self
        if self.exhausted or self.next_after_id != ids[-1]:
            raise ValueError("next_after_id must be the last discovery id when items exist")
        return self


class ScanQueryErrorBody(_ApiModel):
    error: Literal["invalid_after_id", "invalid_limit"]
    detail: str = Field(min_length=1)


class DocumentPayload(_ApiModel):
    document_id: int = Field(ge=1)
    archive_item_id: int | None = Field(default=None, ge=1)
    language: str
    doc_type: str
    item_type: str | None = None
    upload_status: str
    processing_state_user: str
    visibility: str | None = None
    expected_source_file_count: int | None = Field(default=None, ge=0)
    updated_at: datetime | None = None


class TranscriptionPayload(_ApiModel):
    text_result_id: int = Field(ge=1)
    result_type: str
    status: str
    verification_status: str
    engine_key: str | None = None
    engine: str | None = None
    source_revision: int | None = None
    based_on_source_revision: int | None = None
    updated_at: datetime | None = None
    text: str
    text_sha256: Sha256 | None = None


class SourcePagePayload(_ApiModel):
    source_file_id: int | None = Field(default=None, ge=1)
    order_index: int = Field(ge=0)
    display_number: int = Field(ge=1)
    source_kind: Literal["SOURCE_FILE", "PRIMARY_FILE"]
    file_s3_key: str = Field(min_length=1)
    mime_type: str = Field(min_length=1)
    size_bytes: int | None = Field(default=None, ge=0)
    include_in_ocr: bool
    upload_status: str
    download_url: str = Field(min_length=1)
    download_url_expires_in: int = Field(ge=1)


class ApiError(_ApiModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, object] | None = None


class ProvenancePayload(_ApiModel):
    displayed_transcription_resolver: str = Field(min_length=1)
    source_set: Literal["display_physical", "primary_file"]
    presign_expires_in: int = Field(ge=1)


class ReviewBundleResponse(_ApiModel):
    schema_version: Literal["transcription-review-bundle-v1"] = BUNDLE_SCHEMA_VERSION
    outcome: BundleOutcome
    document: DocumentPayload
    transcription: TranscriptionPayload | None = None
    source_pages: list[SourcePagePayload] = Field(default_factory=list)
    error: ApiError | None = None
    provenance: ProvenancePayload | None = None

    @model_validator(mode="after")
    def _outcome_contract(self) -> ReviewBundleResponse:
        if self.document.document_id < 1:
            raise ValueError("document_id must be positive")
        orders = [page.order_index for page in self.source_pages]
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise ValueError("source_pages order_index values must be strictly increasing")

        if self.outcome is BundleOutcome.REVIEWABLE:
            self._require_reviewable()
        elif self.outcome is BundleOutcome.NON_ELIGIBLE:
            self._require_classified_error(NonEligibleCode)
        elif self.outcome is BundleOutcome.BLOCKING_INTEGRITY_ERROR:
            self._require_classified_error(BlockingIntegrityCode)
        return self

    def _require_reviewable(self) -> None:
        if self.error is not None or self.provenance is None:
            raise ValueError("REVIEWABLE bundles have provenance and no error")
        if self.document.language not in {"en", "fr"}:
            raise ValueError("REVIEWABLE documents must be en or fr")
        if self.document.archive_item_id is None:
            raise ValueError("REVIEWABLE documents require archive_item_id")
        transcription = self.transcription
        if transcription is None or not transcription.text.strip():
            raise ValueError("REVIEWABLE documents require exact transcription text")
        if transcription.verification_status == "VERIFIED":
            raise ValueError("VERIFIED transcriptions are not REVIEWABLE")
        if not self.source_pages:
            raise ValueError("REVIEWABLE documents require source files")

    def _require_classified_error(self, codes: type[Enum]) -> None:
        if self.provenance is not None:
            raise ValueError("non-reviewable bundles do not include provenance")
        if self.source_pages:
            raise ValueError("non-reviewable bundles do not include source pages")
        if self.error is None:
            raise ValueError("non-reviewable bundles require an error")
        allowed = {item.value for item in codes}
        if self.error.code not in allowed:
            raise ValueError(f"unexpected error code {self.error.code}")


def parse_scan_response(data: dict[str, object]) -> ScanResponse:
    return ScanResponse.model_validate(data)


def parse_scan_query_error(data: dict[str, object]) -> ScanQueryErrorBody:
    return ScanQueryErrorBody.model_validate(data)


def parse_review_bundle(data: dict[str, object]) -> ReviewBundleResponse:
    return ReviewBundleResponse.model_validate(data)
