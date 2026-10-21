"""Append-only ledger entries.

Each entry is one processing decision. Sequence numbers start at 1 and have
no gaps. The checkpoint stores ``applied_ledger_sequence``; entries after that
sequence are replayed on restart.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from transcription_review_agent.api_client.schemas import (
    BlockingIntegrityCode,
    NonEligibleCode,
)
from transcription_review_agent.bundling.layout import assert_safe_relative_path
from transcription_review_agent.constants import (
    BATCH_ID_PATTERN,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    SHA256_PATTERN,
)
from transcription_review_agent.state.schema import SourceHashRecord

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
BatchId = Annotated[str, Field(pattern=BATCH_ID_PATTERN)]


class LedgerEntryBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    occurred_at: datetime
    batch_id: BatchId | None = None
    cursor_before: int | None = Field(default=None, ge=1)
    cursor_after: int | None = Field(default=None, ge=1)

    @field_validator("occurred_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _batch_id_policy(self) -> LedgerEntryBase:
        event_type = getattr(self, "event_type", None)
        if event_type == "no_work":
            if self.batch_id is not None:
                raise ValueError("no_work must not carry a batch_id")
            return self
        if event_type in {
            "document_not_found",
            "document_non_eligible",
            "run_blocked",
        }:
            return self
        if self.batch_id is None:
            raise ValueError("batch_id is required")
        return self


class DocumentNotFoundEntry(LedgerEntryBase):
    event_type: Literal["document_not_found"] = "document_not_found"
    document_id: int = Field(ge=1)
    http_status: Literal[404] = 404


class DocumentNonEligibleEntry(LedgerEntryBase):
    event_type: Literal["document_non_eligible"] = "document_non_eligible"
    document_id: int = Field(ge=1)
    code: NonEligibleCode


class DocumentSelectedEntry(LedgerEntryBase):
    """One reviewable document joined a batch.

    The first selection for a batch also allocates that batch id. There is no
    separate open event that can be checkpointed with an empty document list.
    Selecting the document named by a blocked checkpoint also clears that block.
    """

    event_type: Literal["document_selected"] = "document_selected"
    document_id: int = Field(ge=1)
    archive_item_id: int = Field(ge=1)
    language: Literal["en", "fr"]
    # Set only on the selection that allocates the batch. Absent on older
    # ledger lines and on selections that join an open batch; replay then
    # keeps the target already recorded for that batch, or 5 when none was.
    target_reviewable_count: int | None = Field(
        default=None,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
    )


class RunBlockedEntry(LedgerEntryBase):
    """A blocking document stopped the run, or replaced a changed block.

    ``batch_id`` is null when the block happens before any reviewable document
    has opened a batch. Re-fetching the same document, code, message, and
    details is not an event. A changed code, message, or details replaces the
    blocked record and holds the cursor.
    """

    event_type: Literal["run_blocked"] = "run_blocked"
    document_id: int = Field(ge=1)
    code: BlockingIntegrityCode
    message: str = Field(min_length=1, max_length=500)
    details: dict[str, object] | None = None


class DiscoveryExhaustedEntry(LedgerEntryBase):
    event_type: Literal["discovery_exhausted"] = "discovery_exhausted"


class NoWorkEntry(LedgerEntryBase):
    """Discovery exhausted with zero reviewable documents.

    This is not a batch. It has no batch id and publishes nothing.
    """

    event_type: Literal["no_work"] = "no_work"


class BundlingStartedEntry(LedgerEntryBase):
    event_type: Literal["bundling_started"] = "bundling_started"


class DocumentBundlingEntry(LedgerEntryBase):
    event_type: Literal["document_bundling"] = "document_bundling"
    document_id: int = Field(ge=1)


class DocumentBundledEntry(LedgerEntryBase):
    event_type: Literal["document_bundled"] = "document_bundled"
    document_id: int = Field(ge=1)
    transcription_sha256: Sha256
    sources: list[SourceHashRecord] = Field(min_length=1)
    bundle_manifest_path: str

    @field_validator("bundle_manifest_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class SealedDocumentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: int = Field(ge=1)
    bundle_manifest_path: str
    transcription_sha256: Sha256

    @field_validator("bundle_manifest_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)


class BatchSealedEntry(LedgerEntryBase):
    event_type: Literal["batch_sealed"] = "batch_sealed"
    document_ids: list[int]
    short_batch: bool
    discovery_exhausted: bool
    documents: list[SealedDocumentRef]


LedgerEntry = Annotated[
    Union[
        DocumentNotFoundEntry,
        DocumentNonEligibleEntry,
        DocumentSelectedEntry,
        RunBlockedEntry,
        DiscoveryExhaustedEntry,
        NoWorkEntry,
        BundlingStartedEntry,
        DocumentBundlingEntry,
        DocumentBundledEntry,
        BatchSealedEntry,
    ],
    Field(discriminator="event_type"),
]

LEDGER_ENTRY_ADAPTER = TypeAdapter(LedgerEntry)


def parse_ledger_entry(data: dict[str, object]) -> LedgerEntry:
    return LEDGER_ENTRY_ADAPTER.validate_python(data)


def dump_ledger_entry(entry: LedgerEntry) -> dict[str, object]:
    if isinstance(entry, BaseModel):
        return entry.model_dump(mode="json")
    raise TypeError("ledger entry must be a pydantic model")
