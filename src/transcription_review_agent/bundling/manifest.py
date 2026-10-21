"""Local immutable bundle manifests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from transcription_review_agent.bundling.layout import (
    API_PAYLOAD_RELATIVE_PATH,
    TRANSCRIPTION_RELATIVE_PATH,
    assert_safe_relative_path,
    document_manifest_relative_path,
)
from transcription_review_agent.bundling.publish import canonical_json, publish_exclusive
from transcription_review_agent.constants import (
    BATCH_ID_PATTERN,
    BUNDLE_SCHEMA_VERSION,
    LOCAL_BATCH_SCHEMA_VERSION,
    LOCAL_DOCUMENT_BUNDLE_SCHEMA_VERSION,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    PILOT_TARGET_REVIEWABLE_COUNT,
    SHA256_PATTERN,
)

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
BatchId = Annotated[str, Field(pattern=BATCH_ID_PATTERN)]


class BundleFile(BaseModel):
    """One file inside a document bundle, excluding the manifest itself."""

    model_config = ConfigDict(extra="forbid")

    relative_path: str
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    role: Literal["api_review_bundle", "transcription_text", "source"]

    @field_validator("relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class SourceArtifact(BaseModel):
    """Original downloaded source bytes. PDFs are not rasterized."""

    model_config = ConfigDict(extra="forbid")

    order_index: int = Field(ge=0)
    source_file_id: int | None = Field(default=None, ge=1)
    source_kind: Literal["SOURCE_FILE", "PRIMARY_FILE"]
    mime_type: str = Field(min_length=1)
    relative_path: str
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    file_s3_key: str | None = None
    include_in_ocr: bool | None = None
    bytes_role: Literal["original"] = "original"

    @field_validator("relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class TranscriptionArtifact(BaseModel):
    """Exact transcription text and the local SHA-256 of its UTF-8 bytes."""

    model_config = ConfigDict(extra="forbid")

    relative_path: str
    sha256: Sha256
    size_bytes: int = Field(ge=1)
    encoding: Literal["utf-8"] = "utf-8"
    server_text_sha256: Sha256 | None = None
    text_result_id: int = Field(ge=1)
    result_type: str = Field(min_length=1)
    status: str = Field(min_length=1)
    verification_status: str = Field(min_length=1)
    engine_key: str | None = None
    engine: str | None = None
    source_revision: int | None = None
    based_on_source_revision: int | None = None
    api_updated_at: datetime | None = None

    @field_validator("relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)

    @field_validator("verification_status")
    @classmethod
    def _not_verified(cls, value: str) -> str:
        if value == "VERIFIED":
            raise ValueError("VERIFIED transcriptions are out of Pilot 01 scope")
        return value


class LocalProvenance(BaseModel):
    """API revision fields copied into the local manifest.

    ``local_hashes_authoritative`` is always true: ``server_text_sha256`` is
    recorded when the server sent one, and the local digest wins.
    """

    model_config = ConfigDict(extra="forbid")

    api_schema_version: Literal["transcription-review-bundle-v1"] = (
        BUNDLE_SCHEMA_VERSION
    )
    displayed_transcription_resolver: str = Field(min_length=1)
    source_set: Literal["display_physical", "primary_file"]
    presign_expires_in: int | None = Field(default=None, ge=1)
    server_text_sha256: Sha256 | None = None
    local_hashes_authoritative: Literal[True] = True
    pdf_rasterization: Literal["not_in_pilot_01"] = "not_in_pilot_01"


class DocumentBundleManifest(BaseModel):
    """Sealed description of one atomic document bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["local-document-bundle-v1"] = (
        LOCAL_DOCUMENT_BUNDLE_SCHEMA_VERSION
    )
    sealed: Literal[True] = True
    document_id: int = Field(ge=1)
    archive_item_id: int = Field(ge=1)
    language: Literal["en", "fr"]
    batch_id: BatchId
    transcription: TranscriptionArtifact
    sources: list[SourceArtifact] = Field(min_length=1)
    provenance: LocalProvenance
    files: list[BundleFile] = Field(min_length=3)
    created_at: datetime

    @model_validator(mode="after")
    def _files_match_artifacts(self) -> DocumentBundleManifest:
        orders = [source.order_index for source in self.sources]
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            raise ValueError("source order_index values must be strictly increasing")
        if self.transcription.relative_path != TRANSCRIPTION_RELATIVE_PATH:
            raise ValueError("transcription text must live at transcription/transcription.txt")

        by_role: dict[str, list[BundleFile]] = {}
        paths: list[str] = []
        for entry in self.files:
            by_role.setdefault(entry.role, []).append(entry)
            paths.append(entry.relative_path)
        if len(paths) != len(set(paths)):
            raise ValueError("bundle file paths must be unique")

        api_files = by_role.get("api_review_bundle", [])
        text_files = by_role.get("transcription_text", [])
        source_files = by_role.get("source", [])
        if len(api_files) != 1 or api_files[0].relative_path != API_PAYLOAD_RELATIVE_PATH:
            raise ValueError("manifest must describe api/review_bundle.json exactly once")
        if len(text_files) != 1:
            raise ValueError("manifest must describe the transcription text exactly once")
        text_file = text_files[0]
        if text_file.relative_path != self.transcription.relative_path:
            raise ValueError("transcription file path does not match the artifact")
        if text_file.sha256 != self.transcription.sha256:
            raise ValueError("transcription file hash does not match the artifact")
        if text_file.size_bytes != self.transcription.size_bytes:
            raise ValueError("transcription file size does not match the artifact")
        if (
            self.provenance.server_text_sha256 != self.transcription.server_text_sha256
        ):
            raise ValueError("server transcription hash must match provenance")

        if len(source_files) != len(self.sources):
            raise ValueError("each source artifact needs one file record")
        source_by_path = {source.relative_path: source for source in self.sources}
        for entry in source_files:
            source = source_by_path.get(entry.relative_path)
            if source is None:
                raise ValueError("source file record does not match a source artifact")
            if entry.sha256 != source.sha256 or entry.size_bytes != source.size_bytes:
                raise ValueError("source file hash does not match the artifact")
        return self


class BatchDocumentRef(BaseModel):
    """Pointer from a batch manifest to one sealed document manifest."""

    model_config = ConfigDict(extra="forbid")

    document_id: int = Field(ge=1)
    manifest_relative_path: str
    manifest_sha256: Sha256

    @field_validator("manifest_relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class BatchManifest(BaseModel):
    """Sealed batch of document bundles. The target is the size captured at allocation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["local-batch-v1"] = LOCAL_BATCH_SCHEMA_VERSION
    batch_id: BatchId
    sealed: Literal[True] = True
    target_reviewable_count: int = Field(
        default=PILOT_TARGET_REVIEWABLE_COUNT,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
    )
    document_ids: list[int]
    short_batch: bool
    discovery_exhausted: bool
    documents: list[BatchDocumentRef]
    created_at: datetime

    @model_validator(mode="after")
    def _batch_shape(self) -> BatchManifest:
        ids = self.document_ids
        target = self.target_reviewable_count
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("batch document ids must be strictly increasing")
        if any(document_id < 1 for document_id in ids):
            raise ValueError("document ids must be positive")
        if len(ids) > target:
            raise ValueError("a batch cannot contain more documents than its target")
        ref_ids = [ref.document_id for ref in self.documents]
        if ref_ids != ids:
            raise ValueError("batch document refs must follow document_ids")
        for ref in self.documents:
            expected = document_manifest_relative_path(ref.document_id)
            if ref.manifest_relative_path != expected:
                raise ValueError("document manifest path does not match the layout")
        if self.short_batch:
            if not ids or len(ids) >= target or not self.discovery_exhausted:
                raise ValueError(
                    "a short batch is smaller than its target and discovery is exhausted"
                )
        elif len(ids) != target:
            raise ValueError("a full batch contains its target number of documents")
        return self


def publish_document_manifest(path: Path, manifest: DocumentBundleManifest) -> str:
    """Write a sealed document manifest. Return the canonical JSON."""

    payload = canonical_json(manifest)
    publish_exclusive(path, payload)
    return payload


def publish_batch_manifest(path: Path, manifest: BatchManifest) -> str:
    """Write a sealed batch manifest. Return the canonical JSON."""

    payload = canonical_json(manifest)
    publish_exclusive(path, payload)
    return payload
