"""Seal a batch whose document bundles are already published.

Sealing writes ``batch_manifest.json`` only after every selected document
bundle verifies, then appends one ``batch_sealed`` ledger entry. The manifest
is published by a complete temp-file write, fsync, and exclusive link, so a
crash cannot replace a manifest that already exists. A later call reuses that
file when it still matches the checkpoint and does not append a second seal.

``config.batch_size`` is not consulted. The target is the one recorded when
the batch was allocated.

The state store has no lock, and this module does not add one. One process
may seal a batch. A unique temporary name avoids truncating another attempt's
temp file, which is crash hygiene, not multi-writer safety. Two processes
can still interleave the ledger and the checkpoint.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.layout import (
    DOCUMENT_MANIFEST_NAME,
    batch_dir,
    batch_manifest_path,
    document_manifest_relative_path,
)
from transcription_review_agent.bundling.manifest import BatchDocumentRef, BatchManifest
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.bundling.verify import (
    verify_batch_bundle,
    verify_document_bundle,
)
from transcription_review_agent.config import AgentConfig
from transcription_review_agent.errors import (
    BundleConflictError,
    BundleVerificationError,
    ImmutableBundleError,
)
from transcription_review_agent.state.schema import (
    BatchStatus,
    ControllerState,
    DocumentLifecycle,
    DocumentPhase,
)
from transcription_review_agent.state.store import StateStore
from transcription_review_agent.state.transitions import seal_batch

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class SealResult:
    """A batch whose manifest and ledger seal both exist. No document bytes."""

    batch_id: str
    batch_status: BatchStatus
    document_ids: tuple[int, ...]
    short_batch: bool
    discovery_exhausted: bool
    target_reviewable_count: int
    manifest_sha256: str


def seal_selected_batch(
    config: AgentConfig,
    state_store: StateStore,
    *,
    clock: Clock | None = None,
) -> SealResult:
    """Publish the batch manifest, if needed, and record one seal.

    Raises when the batch is not fully bundled, a document bundle does not
    verify, or an existing manifest disagrees with the checkpoint. Those
    failures leave the ledger unsealed and do not overwrite the manifest.
    """

    now = _utc_now if clock is None else clock
    state = state_store.load()
    path = _manifest_path(config, state)
    if state.batch_status is BatchStatus.SEALED:
        manifest = _require_matching_manifest(config, state, path)
        _remove_abandoned_temps(path)
        return _result(state, path, manifest)

    entry = seal_batch(state, occurred_at=now())
    if path.is_file():
        manifest = _require_matching_manifest(config, state, path)
    else:
        manifest = _expected_manifest(config, state, created_at=now())
        _publish_new_manifest(path, canonical_json(manifest))
        manifest = _require_matching_manifest(config, state, path)
    _remove_abandoned_temps(path)
    sealed = state_store.commit(entry)
    return _result(sealed, path, manifest)


def _manifest_path(config: AgentConfig, state: ControllerState) -> Path:
    if state.current_batch_id is None:
        raise BundleConflictError("sealing requires an open batch id")
    return batch_manifest_path(config.batches_dir, state.current_batch_id)


def _require_matching_manifest(
    config: AgentConfig,
    state: ControllerState,
    path: Path,
) -> BatchManifest:
    """Verify the on-disk batch and require it to describe this checkpoint."""

    if not path.is_file():
        raise BundleConflictError(
            f"batch {state.current_batch_id} is missing its batch manifest"
        )
    existing = verify_batch_bundle(path.parent)
    expected = _expected_manifest(config, state, created_at=existing.created_at)
    if _seal_body(existing) != _seal_body(expected):
        raise BundleConflictError(
            f"batch manifest for {state.current_batch_id} does not match the checkpoint"
        )
    return existing


def _expected_manifest(
    config: AgentConfig,
    state: ControllerState,
    *,
    created_at: datetime,
) -> BatchManifest:
    batch_id = state.current_batch_id
    if batch_id is None:
        raise BundleConflictError("sealing requires an open batch id")
    root = batch_dir(config.batches_dir, batch_id)
    _require_document_dirs(root, state.current_batch_document_ids)
    references = [
        _document_ref(root, document, batch_id=batch_id) for document in state.documents
    ]
    target = state.batch_target_count
    return BatchManifest(
        batch_id=batch_id,
        target_reviewable_count=target,
        document_ids=[document.document_id for document in state.documents],
        short_batch=len(state.documents) < target,
        discovery_exhausted=state.discovery_exhausted,
        documents=references,
        created_at=created_at,
    )


def _document_ref(
    batch_root: Path,
    document: DocumentLifecycle,
    *,
    batch_id: str,
) -> BatchDocumentRef:
    if document.phase is not DocumentPhase.BUNDLED:
        raise BundleConflictError(f"document {document.document_id} is not bundled")
    relative = document_manifest_relative_path(document.document_id)
    if document.bundle_manifest_path != relative:
        raise BundleConflictError(
            f"bundled document {document.document_id} has an unexpected manifest path"
        )
    bundle_root = batch_root / "documents" / str(document.document_id)
    manifest = verify_document_bundle(bundle_root)
    if (
        manifest.document_id != document.document_id
        or manifest.archive_item_id != document.archive_item_id
        or manifest.language != document.language
        or manifest.batch_id != batch_id
        or manifest.transcription.sha256 != document.transcription_sha256
    ):
        raise BundleConflictError(
            f"document bundle {document.document_id} does not match the checkpoint"
        )
    checkpoint_sources = [
        (item.order_index, item.sha256, item.size_bytes, item.relative_path)
        for item in document.source_hashes
    ]
    manifest_sources = [
        (item.order_index, item.sha256, item.size_bytes, item.relative_path)
        for item in manifest.sources
    ]
    if checkpoint_sources != manifest_sources:
        raise BundleConflictError(
            f"document bundle {document.document_id} source hashes do not match"
        )
    digest = sha256_bytes((bundle_root / DOCUMENT_MANIFEST_NAME).read_bytes())
    return BatchDocumentRef(
        document_id=document.document_id,
        manifest_relative_path=relative,
        manifest_sha256=digest,
    )


def _require_document_dirs(batch_root: Path, document_ids: list[int]) -> None:
    documents = batch_root / "documents"
    if not documents.is_dir():
        raise BundleVerificationError(f"missing documents directory {documents}")
    names: set[str] = set()
    for child in documents.iterdir():
        if not child.is_dir():
            raise BundleConflictError(
                f"unexpected file in the batch documents directory: {child.name}"
            )
        names.add(child.name)
    expected = {str(document_id) for document_id in document_ids}
    if names != expected:
        raise BundleConflictError("document bundles do not match the selected batch")


def _publish_new_manifest(path: Path, payload: str) -> None:
    """Create ``path`` by linking a synced temp file. Never replaces ``path``.

    The temp name is unique to this attempt and is created with ``O_EXCL``.
    A crash leaves that temp unpublished. Recovery writes a new temp rather
    than trusting a partial one. The final name is linked only after the
    temp bytes and the directory entry are synced.
    """

    if path.exists():
        raise ImmutableBundleError(
            f"refusing to overwrite completed bundle file {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_temp(path)
    data = payload.encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(temporary, flags, 0o644)
    except FileExistsError as exc:
        raise ImmutableBundleError(
            f"refusing to overwrite completed bundle file {temporary}"
        ) from exc
    try:
        try:
            _write_all(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_dir(path.parent)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ImmutableBundleError(
                f"refusing to overwrite completed bundle file {path}"
            ) from exc
        _fsync_dir(path.parent)
    finally:
        _unlink_file(temporary)
        if path.parent.is_dir():
            _fsync_dir(path.parent)


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write every byte. ``os.write`` may return after a short write."""

    view = memoryview(payload)
    while len(view):
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("incomplete manifest write")
        view = view[written:]


def _unique_temp(path: Path) -> Path:
    token = f"{os.getpid()}.{secrets.token_hex(8)}"
    return path.with_name(f".{path.name}.{token}.tmp")


def _remove_abandoned_temps(path: Path) -> None:
    """Remove unpublished temp names left by a crashed single writer.

    Called only after the final manifest has been verified. This is not safe
    if another process is sealing the same batch.
    """

    parent = path.parent
    if not parent.is_dir():
        return
    prefix = f".{path.name}."
    fixed = f".{path.name}.tmp"
    removed = False
    for candidate in parent.iterdir():
        if not candidate.is_file():
            continue
        name = candidate.name
        if name == fixed or (name.startswith(prefix) and name.endswith(".tmp")):
            candidate.unlink()
            removed = True
    if removed:
        _fsync_dir(parent)


def _unlink_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _seal_body(manifest: BatchManifest) -> dict[str, object]:
    body = manifest.model_dump(mode="json")
    body.pop("created_at", None)
    return body


def _result(state: ControllerState, path: Path, manifest: BatchManifest) -> SealResult:
    if state.current_batch_id is None:
        raise BundleConflictError("a sealed batch requires a batch id")
    return SealResult(
        batch_id=state.current_batch_id,
        batch_status=state.batch_status,
        document_ids=tuple(state.current_batch_document_ids),
        short_batch=manifest.short_batch,
        discovery_exhausted=manifest.discovery_exhausted,
        target_reviewable_count=manifest.target_reviewable_count,
        manifest_sha256=sha256_bytes(path.read_bytes()),
    )


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
