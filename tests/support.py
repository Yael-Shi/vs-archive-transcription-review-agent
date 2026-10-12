"""Shared fixtures for state and bundle tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.state.schema import ControllerState, SourceHashRecord
from transcription_review_agent.state.transitions import (
    apply_ledger_entry,
    begin_bundling,
    mark_bundled,
    mark_bundling,
    note_discovery_exhausted,
    seal_batch,
    select_reviewable,
)

NOW = datetime(2026, 10, 11, 11, 0, tzinfo=timezone.utc)


def at(seconds: int) -> datetime:
    return NOW + timedelta(seconds=seconds)


def text_digest(text: str = "exact text") -> str:
    return sha256_text(text)


def source_record(payload: bytes = b"page-bytes", *, order_index: int = 0) -> SourceHashRecord:
    return SourceHashRecord(
        order_index=order_index,
        sha256=sha256_bytes(payload),
        size_bytes=len(payload),
        relative_path=f"sources/{order_index:04d}_primary_file.jpg",
    )


def apply_events(state: ControllerState, *entries) -> ControllerState:
    for entry in entries:
        state = apply_ledger_entry(state, entry)
    return state


def select_one(state: ControllerState, document_id: int, *, at_seconds: int) -> ControllerState:
    entry = select_reviewable(
        state,
        document_id=document_id,
        archive_item_id=5000 + document_id,
        language="en",
        occurred_at=at(at_seconds),
    )
    return apply_ledger_entry(state, entry)


def bundle_one(
    state: ControllerState,
    document_id: int,
    *,
    at_seconds: int,
) -> ControllerState:
    bundling = mark_bundling(
        state,
        document_id=document_id,
        occurred_at=at(at_seconds),
    )
    state = apply_ledger_entry(state, bundling)
    bundled = mark_bundled(
        state,
        document_id=document_id,
        transcription_sha256=text_digest(f"text-{document_id}"),
        sources=[source_record(f"page-{document_id}".encode())],
        bundle_manifest_path=f"documents/{document_id}/document_manifest.json",
        occurred_at=at(at_seconds + 1),
    )
    return apply_ledger_entry(state, bundled)


def seal_short_batch(state: ControllerState) -> ControllerState:
    """Exhaust discovery, bundle the already selected documents, and seal."""

    state = apply_ledger_entry(
        state,
        note_discovery_exhausted(state, occurred_at=at(50)),
    )
    state = apply_ledger_entry(state, begin_bundling(state, occurred_at=at(51)))
    for offset, document_id in enumerate(state.current_batch_document_ids):
        state = bundle_one(state, document_id, at_seconds=60 + offset * 2)
    return apply_ledger_entry(state, seal_batch(state, occurred_at=at(90)))
