"""Hot checkpoint for the controller.

The checkpoint is the restart cursor and the open batch. It is not the
history. History lives in the append-only ledger. ``last_scanned_document_id``
is independent of whether selected documents have been bundled.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from transcription_review_agent.api_client.schemas import BlockingIntegrityCode
from transcription_review_agent.bundling.layout import assert_safe_relative_path
from transcription_review_agent.constants import (
    BATCH_ID_PATTERN,
    PILOT_TARGET_REVIEWABLE_COUNT,
    SHA256_PATTERN,
    STATE_SCHEMA_VERSION,
)
from transcription_review_agent.errors import StateInvariantError

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
BatchId = Annotated[str, Field(pattern=BATCH_ID_PATTERN)]

INITIAL_UPDATED_AT = datetime(1970, 1, 1, tzinfo=timezone.utc)


class BatchStatus(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    ACCUMULATING = "accumulating"
    BUNDLING = "bundling"
    BLOCKED = "blocked"
    SEALED = "sealed"
    NO_WORK = "no_work"


class DocumentPhase(str, Enum):
    """Lifecycle of a document inside the current batch.

    ``selected`` means the bundle outcome was ``REVIEWABLE`` and the scan
    cursor advanced. Bytes are not durable yet. ``bundling`` is the in-progress
    download. ``bundled`` means the sealed document manifest exists.
    """

    SELECTED = "selected"
    BUNDLING = "bundling"
    BUNDLED = "bundled"


class SourceHashRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_index: int = Field(ge=0)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    relative_path: str

    @field_validator("relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class DocumentLifecycle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: int = Field(ge=1)
    archive_item_id: int = Field(ge=1)
    language: str
    phase: DocumentPhase
    transcription_sha256: Sha256 | None = None
    source_hashes: list[SourceHashRecord] = Field(default_factory=list)
    bundle_manifest_path: str | None = None

    @field_validator("language")
    @classmethod
    def _pilot_language(cls, value: str) -> str:
        if value not in {"en", "fr"}:
            raise ValueError("batch documents must be en or fr")
        return value

    @field_validator("bundle_manifest_path")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return assert_safe_relative_path(value)


class BlockedDocumentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: int = Field(ge=1)
    code: BlockingIntegrityCode
    message: str = Field(min_length=1, max_length=500)
    details: dict[str, object] | None = None
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _require_aware(value)


class ControllerState(BaseModel):
    """Restart checkpoint. Validate with ``validate_controller_state``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = STATE_SCHEMA_VERSION
    applied_ledger_sequence: int = Field(default=0, ge=0)
    last_scanned_document_id: int | None = Field(default=None, ge=1)
    last_sealed_batch_id: BatchId | None = None
    current_batch_id: BatchId | None = None
    current_batch_document_ids: list[int] = Field(default_factory=list)
    batch_status: BatchStatus = BatchStatus.IDLE
    discovery_exhausted: bool = False
    blocked: BlockedDocumentState | None = None
    documents: list[DocumentLifecycle] = Field(default_factory=list)
    updated_at: datetime = INITIAL_UPDATED_AT

    @field_validator("updated_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _require_aware(value)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


def initial_state() -> ControllerState:
    """Checkpoint before any ledger entry. Not written until the first commit."""

    state = ControllerState()
    validate_controller_state(state)
    return state


def validate_controller_state(state: ControllerState) -> None:
    """Raise ``StateInvariantError`` when the checkpoint contradicts itself."""

    if state.schema_version != STATE_SCHEMA_VERSION:
        raise StateInvariantError("unsupported state schema_version")
    _validate_document_list(state)
    status = state.batch_status
    if status is BatchStatus.IDLE:
        _require_idle(state)
    elif status is BatchStatus.SCANNING:
        _require_scanning(state)
    elif status is BatchStatus.ACCUMULATING:
        _require_open_batch(state, phases={DocumentPhase.SELECTED})
        if not state.documents:
            raise StateInvariantError(
                "an accumulating batch contains its first selected document"
            )
    elif status is BatchStatus.BUNDLING:
        _require_open_batch(
            state,
            phases={DocumentPhase.SELECTED, DocumentPhase.BUNDLING, DocumentPhase.BUNDLED},
        )
        if len(state.documents) < 1:
            raise StateInvariantError("bundling requires at least one document")
        if (
            len(state.documents) != PILOT_TARGET_REVIEWABLE_COUNT
            and not state.discovery_exhausted
        ):
            raise StateInvariantError(
                "bundling requires a full batch or exhausted discovery"
            )
        _validate_bundle_pipeline(state.documents)
    elif status is BatchStatus.BLOCKED:
        _require_blocked(state)
    elif status is BatchStatus.SEALED:
        _require_sealed(state)
    elif status is BatchStatus.NO_WORK:
        _require_no_work(state)
    else:
        raise StateInvariantError(f"unknown batch_status {status}")


def _validate_document_list(state: ControllerState) -> None:
    ids = [document.document_id for document in state.documents]
    if ids != state.current_batch_document_ids:
        raise StateInvariantError(
            "current_batch_document_ids must match documents in order"
        )
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise StateInvariantError("batch document ids must be strictly increasing")
    if len(ids) > PILOT_TARGET_REVIEWABLE_COUNT:
        raise StateInvariantError("a batch cannot contain more than 5 documents")
    for document in state.documents:
        _validate_document_phase(document)
    if state.documents and state.last_scanned_document_id is None:
        raise StateInvariantError("selected documents require a scan cursor")
    if state.documents and state.last_scanned_document_id is not None:
        if state.last_scanned_document_id < state.documents[-1].document_id:
            raise StateInvariantError(
                "scan cursor must be at or after the last selected document"
            )


def _validate_document_phase(document: DocumentLifecycle) -> None:
    orders = [item.order_index for item in document.source_hashes]
    if orders != sorted(orders) or len(orders) != len(set(orders)):
        raise StateInvariantError("source hash order_index values must be unique and sorted")
    if document.phase is DocumentPhase.BUNDLED:
        if document.transcription_sha256 is None or not document.bundle_manifest_path:
            raise StateInvariantError("bundled documents require a hash and manifest path")
        if not document.source_hashes:
            raise StateInvariantError("bundled documents require source hashes")
        return
    if (
        document.transcription_sha256 is not None
        or document.bundle_manifest_path is not None
        or document.source_hashes
    ):
        raise StateInvariantError(
            "only bundled documents carry local hashes and a manifest path"
        )


def _require_idle(state: ControllerState) -> None:
    if (
        state.current_batch_id is not None
        or state.documents
        or state.blocked is not None
        or state.discovery_exhausted
        or state.last_scanned_document_id is not None
        or state.last_sealed_batch_id is not None
        or state.applied_ledger_sequence != 0
    ):
        raise StateInvariantError("idle state is only the pre-ledger checkpoint")


def _require_scanning(state: ControllerState) -> None:
    if (
        state.current_batch_id is not None
        or state.documents
        or state.blocked is not None
        or state.discovery_exhausted
        or state.last_scanned_document_id is None
        or state.applied_ledger_sequence < 1
    ):
        raise StateInvariantError(
            "scanning has a cursor, no open batch, and discovery is not finished"
        )
    _require_unsealed_open_id(state)


def _require_no_work(state: ControllerState) -> None:
    if (
        state.current_batch_id is not None
        or state.documents
        or state.blocked is not None
        or not state.discovery_exhausted
        or state.applied_ledger_sequence < 1
    ):
        raise StateInvariantError(
            "no-work has no open batch, no documents, and exhausted discovery"
        )


def _require_open_batch(
    state: ControllerState,
    *,
    phases: set[DocumentPhase],
) -> None:
    if state.current_batch_id is None:
        raise StateInvariantError("an open batch requires current_batch_id")
    if state.blocked is not None:
        raise StateInvariantError("an open batch is not blocked")
    _require_unsealed_open_id(state)
    for document in state.documents:
        if document.phase not in phases:
            raise StateInvariantError(
                f"document {document.document_id} phase {document.phase} "
                f"is illegal while {state.batch_status.value}"
            )


def _require_blocked(state: ControllerState) -> None:
    blocked = state.blocked
    if blocked is None:
        raise StateInvariantError("blocked state requires a blocked document")
    if state.discovery_exhausted:
        raise StateInvariantError("a blocked run is not discovery exhaustion")
    if blocked.document_id in state.current_batch_document_ids:
        raise StateInvariantError("a blocked document is not part of the batch")
    cursor = state.last_scanned_document_id
    if cursor is not None and cursor >= blocked.document_id:
        raise StateInvariantError(
            "scan cursor must not advance to or past the blocked document"
        )
    if state.documents:
        if state.current_batch_id is None:
            raise StateInvariantError("selected documents require their batch id")
        _require_unsealed_open_id(state)
        for document in state.documents:
            if document.phase is not DocumentPhase.SELECTED:
                raise StateInvariantError("a blocked batch has not started bundling")
        return
    if state.current_batch_id is not None:
        raise StateInvariantError("a block with no selected documents has no batch id")


def _require_unsealed_open_id(state: ControllerState) -> None:
    if (
        state.current_batch_id is not None
        and state.last_sealed_batch_id == state.current_batch_id
    ):
        raise StateInvariantError("an open batch is not yet the last sealed batch")


def _require_sealed(state: ControllerState) -> None:
    if state.current_batch_id is None or state.blocked is not None:
        raise StateInvariantError("sealed state keeps the batch and clears the block")
    if state.last_sealed_batch_id != state.current_batch_id:
        raise StateInvariantError("sealed state records last_sealed_batch_id")
    if any(document.phase is not DocumentPhase.BUNDLED for document in state.documents):
        raise StateInvariantError("every document in a sealed batch is bundled")
    count = len(state.documents)
    if count < 1:
        raise StateInvariantError("a sealed batch contains at least one document")
    if count < PILOT_TARGET_REVIEWABLE_COUNT:
        if not state.discovery_exhausted:
            raise StateInvariantError("a short sealed batch requires exhausted discovery")
    elif count != PILOT_TARGET_REVIEWABLE_COUNT:
        raise StateInvariantError("a sealed batch has at most 5 documents")


def _validate_bundle_pipeline(documents: list[DocumentLifecycle]) -> None:
    """Bundled documents come first, then at most one bundling, then selected."""

    seen_incomplete = False
    seen_bundling = False
    for document in documents:
        if document.phase is DocumentPhase.BUNDLED:
            if seen_incomplete:
                raise StateInvariantError("bundled documents must precede incomplete ones")
        elif document.phase is DocumentPhase.BUNDLING:
            if seen_bundling or seen_incomplete:
                raise StateInvariantError("only one document may be bundling, in order")
            seen_bundling = True
            seen_incomplete = True
        elif document.phase is DocumentPhase.SELECTED:
            seen_incomplete = True
        else:
            raise StateInvariantError(f"unexpected phase {document.phase}")


# Re-exported so ledger entries and state share one code list.
__all__ = [
    "INITIAL_UPDATED_AT",
    "BatchStatus",
    "BlockedDocumentState",
    "BlockingIntegrityCode",
    "ControllerState",
    "DocumentLifecycle",
    "DocumentPhase",
    "SourceHashRecord",
    "initial_state",
    "validate_controller_state",
]
