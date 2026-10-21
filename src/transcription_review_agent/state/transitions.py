"""Pure checkpoint transitions.

Applying a ledger entry is the only way to change controller state. These
functions do not scan, download, or write files.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from transcription_review_agent.api_client.schemas import (
    BlockingIntegrityCode,
    NonEligibleCode,
)
from transcription_review_agent.constants import (
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    PILOT_TARGET_REVIEWABLE_COUNT,
)
from transcription_review_agent.discovery.cursor import (
    CursorEffect,
    assert_document_after_cursor,
    cursor_effect,
)
from transcription_review_agent.errors import StateInvariantError
from transcription_review_agent.state.ledger import (
    BatchSealedEntry,
    BundlingStartedEntry,
    DiscoveryExhaustedEntry,
    DocumentBundledEntry,
    DocumentBundlingEntry,
    DocumentNonEligibleEntry,
    DocumentNotFoundEntry,
    DocumentSelectedEntry,
    LedgerEntry,
    NoWorkEntry,
    RunBlockedEntry,
    SealedDocumentRef,
)
from transcription_review_agent.state.schema import (
    BatchStatus,
    BlockedDocumentState,
    ControllerState,
    DocumentLifecycle,
    DocumentPhase,
    SourceHashRecord,
    initial_state,
    validate_controller_state,
)


def replay_ledger(
    entries: list[LedgerEntry],
    *,
    initial: ControllerState | None = None,
) -> ControllerState:
    """Fold ledger entries from the initial checkpoint."""

    state = initial_state() if initial is None else initial
    for entry in entries:
        state = apply_ledger_entry(state, entry)
    return state


def apply_ledger_entry(state: ControllerState, entry: LedgerEntry) -> ControllerState:
    """Return the checkpoint after ``entry``. ``state`` is left unchanged."""

    if entry.sequence != state.applied_ledger_sequence + 1:
        raise StateInvariantError(
            f"ledger sequence {entry.sequence} does not follow "
            f"{state.applied_ledger_sequence}"
        )
    if entry.cursor_before != state.last_scanned_document_id:
        raise StateInvariantError("cursor_before does not match the checkpoint")

    if isinstance(entry, DocumentNotFoundEntry):
        updated = _apply_not_found(state, entry)
    elif isinstance(entry, DocumentNonEligibleEntry):
        updated = _apply_non_eligible(state, entry)
    elif isinstance(entry, DocumentSelectedEntry):
        updated = _apply_selected(state, entry)
    elif isinstance(entry, RunBlockedEntry):
        updated = _apply_blocked(state, entry)
    elif isinstance(entry, DiscoveryExhaustedEntry):
        updated = _apply_exhausted(state, entry)
    elif isinstance(entry, NoWorkEntry):
        updated = _apply_no_work(state, entry)
    elif isinstance(entry, BundlingStartedEntry):
        updated = _apply_bundling_started(state, entry)
    elif isinstance(entry, DocumentBundlingEntry):
        updated = _apply_document_bundling(state, entry)
    elif isinstance(entry, DocumentBundledEntry):
        updated = _apply_document_bundled(state, entry)
    elif isinstance(entry, BatchSealedEntry):
        updated = _apply_batch_sealed(state, entry)
    else:
        raise StateInvariantError(f"unsupported ledger entry {entry!r}")

    candidate = updated.model_copy(
        update={
            "applied_ledger_sequence": entry.sequence,
            "last_scanned_document_id": entry.cursor_after,
            "updated_at": entry.occurred_at,
        }
    )
    validate_controller_state(candidate)
    return candidate


def note_not_found(
    state: ControllerState,
    *,
    document_id: int,
    occurred_at: datetime,
) -> DocumentNotFoundEntry:
    _require_effect(CursorEffect.ADVANCE, http_status=404, outcome=None)
    return DocumentNotFoundEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=_visit_batch_id(state, document_id),
        cursor_before=state.last_scanned_document_id,
        cursor_after=document_id,
        document_id=document_id,
    )


def note_non_eligible(
    state: ControllerState,
    *,
    document_id: int,
    code: NonEligibleCode,
    occurred_at: datetime,
) -> DocumentNonEligibleEntry:
    _require_effect(CursorEffect.ADVANCE, http_status=200, outcome="NON_ELIGIBLE")
    return DocumentNonEligibleEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=_visit_batch_id(state, document_id),
        cursor_before=state.last_scanned_document_id,
        cursor_after=document_id,
        document_id=document_id,
        code=code,
    )


def select_reviewable(
    state: ControllerState,
    *,
    document_id: int,
    archive_item_id: int,
    language: Literal["en", "fr"],
    occurred_at: datetime,
    target_reviewable_count: int | None = None,
) -> DocumentSelectedEntry:
    """Plan the selection of one reviewable document.

    If no batch is open, this same entry allocates the next batch id and
    records ``target_reviewable_count``. ``None`` keeps the historical default
    of 5 for ledger lines that predate a configured size. A selection that
    joins an open batch does not record a new target, so a configuration
    change cannot resize that batch.
    """

    if state.batch_status is BatchStatus.BUNDLING:
        raise StateInvariantError("cannot select a document while bundling")
    _require_effect(CursorEffect.ADVANCE, http_status=200, outcome="REVIEWABLE")
    joins_open_batch = state.batch_status is BatchStatus.ACCUMULATING or (
        state.batch_status is BatchStatus.BLOCKED and bool(state.documents)
    )
    if joins_open_batch:
        recorded_target = None
    else:
        if target_reviewable_count is not None:
            _require_batch_target(target_reviewable_count)
        recorded_target = target_reviewable_count
    if state.batch_status is BatchStatus.BLOCKED:
        _require_blocked_document(state, document_id)
        batch_id = (
            _require_batch_id(state) if state.documents else _next_batch_id(state)
        )
    elif state.batch_status is BatchStatus.ACCUMULATING:
        batch_id = _require_batch_id(state)
    elif state.batch_status in {
        BatchStatus.IDLE,
        BatchStatus.SCANNING,
        BatchStatus.NO_WORK,
        BatchStatus.SEALED,
    }:
        batch_id = _next_batch_id(state)
    else:
        raise StateInvariantError("cannot select a document in this state")
    return DocumentSelectedEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=batch_id,
        cursor_before=state.last_scanned_document_id,
        cursor_after=document_id,
        document_id=document_id,
        archive_item_id=archive_item_id,
        language=language,
        target_reviewable_count=recorded_target,
    )


def note_blocking(
    state: ControllerState,
    *,
    document_id: int,
    code: BlockingIntegrityCode,
    message: str,
    occurred_at: datetime,
    details: dict[str, object] | None = None,
) -> RunBlockedEntry | None:
    """Plan a blocking stop.

    Re-fetching an unchanged block returns ``None``. That is not a transition
    and must not be appended. A changed code, message, or details on the same
    document returns one replacement entry and does not move the cursor.
    """

    _require_effect(
        CursorEffect.HOLD_AND_STOP,
        http_status=200,
        outcome="BLOCKING_INTEGRITY_ERROR",
    )
    if state.batch_status is BatchStatus.BLOCKED:
        _require_blocked_document(state, document_id)
        if _blocking_result_unchanged(
            state,
            code=code,
            message=message,
            details=details,
        ):
            return None
        batch_id = state.current_batch_id
    elif state.batch_status is BatchStatus.ACCUMULATING:
        batch_id = _require_batch_id(state)
    elif state.batch_status in {
        BatchStatus.IDLE,
        BatchStatus.SCANNING,
        BatchStatus.NO_WORK,
        BatchStatus.SEALED,
    }:
        batch_id = None
    else:
        raise StateInvariantError("cannot block while bundling")
    return RunBlockedEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=batch_id,
        cursor_before=state.last_scanned_document_id,
        cursor_after=state.last_scanned_document_id,
        document_id=document_id,
        code=code,
        message=message,
        details=details,
    )


def note_no_work(state: ControllerState, *, occurred_at: datetime) -> NoWorkEntry:
    """Plan a no-work terminal result. Does not allocate a batch id."""

    if state.batch_status is BatchStatus.NO_WORK:
        raise StateInvariantError("no-work is already recorded at this cursor")
    if state.batch_status not in {
        BatchStatus.IDLE,
        BatchStatus.SCANNING,
        BatchStatus.SEALED,
    }:
        raise StateInvariantError("no-work requires no open batch")
    return NoWorkEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        cursor_before=state.last_scanned_document_id,
        cursor_after=state.last_scanned_document_id,
    )


def note_discovery_exhausted(
    state: ControllerState,
    *,
    occurred_at: datetime,
) -> DiscoveryExhaustedEntry:
    return DiscoveryExhaustedEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=_require_batch_id(state),
        cursor_before=state.last_scanned_document_id,
        cursor_after=state.last_scanned_document_id,
    )


def begin_bundling(
    state: ControllerState,
    *,
    occurred_at: datetime,
) -> BundlingStartedEntry:
    return BundlingStartedEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=_require_batch_id(state),
        cursor_before=state.last_scanned_document_id,
        cursor_after=state.last_scanned_document_id,
    )


def mark_bundling(
    state: ControllerState,
    *,
    document_id: int,
    occurred_at: datetime,
) -> DocumentBundlingEntry:
    return DocumentBundlingEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=_require_batch_id(state),
        cursor_before=state.last_scanned_document_id,
        cursor_after=state.last_scanned_document_id,
        document_id=document_id,
    )


def mark_bundled(
    state: ControllerState,
    *,
    document_id: int,
    transcription_sha256: str,
    sources: list[SourceHashRecord],
    bundle_manifest_path: str,
    occurred_at: datetime,
) -> DocumentBundledEntry:
    return DocumentBundledEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=_require_batch_id(state),
        cursor_before=state.last_scanned_document_id,
        cursor_after=state.last_scanned_document_id,
        document_id=document_id,
        transcription_sha256=transcription_sha256,
        sources=sources,
        bundle_manifest_path=bundle_manifest_path,
    )


def seal_batch(
    state: ControllerState,
    *,
    occurred_at: datetime,
) -> BatchSealedEntry:
    if state.batch_status is not BatchStatus.BUNDLING:
        raise StateInvariantError("only a bundling batch can be sealed")
    if any(document.phase is not DocumentPhase.BUNDLED for document in state.documents):
        raise StateInvariantError("every selected document must be bundled before seal")
    short_batch = len(state.documents) < state.batch_target_count
    return BatchSealedEntry(
        sequence=state.applied_ledger_sequence + 1,
        occurred_at=occurred_at,
        batch_id=_require_batch_id(state),
        cursor_before=state.last_scanned_document_id,
        cursor_after=state.last_scanned_document_id,
        document_ids=[document.document_id for document in state.documents],
        short_batch=short_batch,
        discovery_exhausted=state.discovery_exhausted,
        documents=[
            SealedDocumentRef(
                document_id=document.document_id,
                bundle_manifest_path=document.bundle_manifest_path or "",
                transcription_sha256=document.transcription_sha256 or "",
            )
            for document in state.documents
        ],
    )


def _apply_not_found(
    state: ControllerState,
    entry: DocumentNotFoundEntry,
) -> ControllerState:
    return _apply_scan_visit(state, entry, entry.document_id)


def _apply_non_eligible(
    state: ControllerState,
    entry: DocumentNonEligibleEntry,
) -> ControllerState:
    return _apply_scan_visit(state, entry, entry.document_id)


def _apply_selected(
    state: ControllerState,
    entry: DocumentSelectedEntry,
) -> ControllerState:
    _require_advance_cursor(state, entry, entry.document_id)
    document = DocumentLifecycle(
        document_id=entry.document_id,
        archive_item_id=entry.archive_item_id,
        language=entry.language,
        phase=DocumentPhase.SELECTED,
    )
    if state.batch_status is BatchStatus.BLOCKED:
        return _apply_blocked_selection(state, entry, document)
    if state.batch_status is BatchStatus.ACCUMULATING:
        _require_accumulating(state, entry)
        if state.discovery_exhausted:
            raise StateInvariantError(
                "cannot select documents after discovery is exhausted"
            )
        if len(state.documents) >= state.batch_target_count:
            raise StateInvariantError(
                "a batch already has its target number of reviewable documents"
            )
        if entry.document_id in state.current_batch_document_ids:
            raise StateInvariantError("document is already in the batch")
        documents = [*state.documents, document]
        return state.model_copy(
            update={
                "documents": documents,
                "current_batch_document_ids": [item.document_id for item in documents],
                "batch_target_count": _target_for_selection(
                    state, entry, allocates=False
                ),
            }
        )
    if state.batch_status not in {
        BatchStatus.IDLE,
        BatchStatus.SCANNING,
        BatchStatus.NO_WORK,
        BatchStatus.SEALED,
    }:
        raise StateInvariantError("cannot select a document in this state")
    expected = _next_batch_id(state)
    if entry.batch_id != expected:
        raise StateInvariantError(f"batch id must be {expected}")
    return state.model_copy(
        update={
            "current_batch_id": entry.batch_id,
            "documents": [document],
            "current_batch_document_ids": [entry.document_id],
            "batch_status": BatchStatus.ACCUMULATING,
            "discovery_exhausted": False,
            "blocked": None,
            "batch_target_count": _target_for_selection(state, entry, allocates=True),
        }
    )


def _apply_blocked_selection(
    state: ControllerState,
    entry: DocumentSelectedEntry,
    document: DocumentLifecycle,
) -> ControllerState:
    _require_blocked_document(state, entry.document_id)
    if state.documents:
        _require_batch(state, entry)
        if len(state.documents) >= state.batch_target_count:
            raise StateInvariantError(
                "a batch already has its target number of reviewable documents"
            )
        if entry.document_id in state.current_batch_document_ids:
            raise StateInvariantError("document is already in the batch")
        documents = [*state.documents, document]
        return state.model_copy(
            update={
                "documents": documents,
                "current_batch_document_ids": [item.document_id for item in documents],
                "batch_status": BatchStatus.ACCUMULATING,
                "blocked": None,
                "batch_target_count": _target_for_selection(
                    state, entry, allocates=False
                ),
            }
        )
    expected = _next_batch_id(state)
    if entry.batch_id != expected:
        raise StateInvariantError(f"batch id must be {expected}")
    return state.model_copy(
        update={
            "current_batch_id": entry.batch_id,
            "documents": [document],
            "current_batch_document_ids": [entry.document_id],
            "batch_status": BatchStatus.ACCUMULATING,
            "discovery_exhausted": False,
            "blocked": None,
            "batch_target_count": _target_for_selection(state, entry, allocates=True),
        }
    )


def _apply_blocked(state: ControllerState, entry: RunBlockedEntry) -> ControllerState:
    if state.batch_status is BatchStatus.BLOCKED:
        return _apply_block_replacement(state, entry)
    _require_hold_cursor(entry)
    try:
        assert_document_after_cursor(state.last_scanned_document_id, entry.document_id)
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    if entry.document_id in state.current_batch_document_ids:
        raise StateInvariantError("a selected document cannot also be blocking")
    blocked = BlockedDocumentState(
        document_id=entry.document_id,
        code=entry.code,
        message=entry.message,
        details=entry.details,
        occurred_at=entry.occurred_at,
    )
    if state.batch_status is BatchStatus.ACCUMULATING:
        _require_accumulating(state, entry)
        if not state.documents:
            raise StateInvariantError(
                "an accumulating batch contains its first selected document"
            )
        return state.model_copy(
            update={
                "batch_status": BatchStatus.BLOCKED,
                "blocked": blocked,
            }
        )
    if state.batch_status not in {
        BatchStatus.IDLE,
        BatchStatus.SCANNING,
        BatchStatus.NO_WORK,
        BatchStatus.SEALED,
    }:
        raise StateInvariantError("cannot block while bundling")
    if entry.batch_id is not None:
        raise StateInvariantError("a block before a batch has no batch id")
    last_sealed = state.last_sealed_batch_id
    if state.batch_status is BatchStatus.SEALED:
        last_sealed = state.current_batch_id
    return state.model_copy(
        update={
            "batch_status": BatchStatus.BLOCKED,
            "blocked": blocked,
            "current_batch_id": None,
            "current_batch_document_ids": [],
            "documents": [],
            "discovery_exhausted": False,
            "last_sealed_batch_id": last_sealed,
        }
    )


def _apply_block_replacement(
    state: ControllerState,
    entry: RunBlockedEntry,
) -> ControllerState:
    _require_blocked_document(state, entry.document_id)
    _require_hold_cursor(entry)
    if entry.batch_id != state.current_batch_id:
        raise StateInvariantError("a block update keeps the open batch id")
    if _blocking_result_unchanged(
        state,
        code=entry.code,
        message=entry.message,
        details=entry.details,
    ):
        raise StateInvariantError("an unchanged blocking result is not a ledger event")
    return state.model_copy(
        update={
            "blocked": BlockedDocumentState(
                document_id=entry.document_id,
                code=entry.code,
                message=entry.message,
                details=entry.details,
                occurred_at=entry.occurred_at,
            ),
        }
    )


def _apply_exhausted(
    state: ControllerState,
    entry: DiscoveryExhaustedEntry,
) -> ControllerState:
    _require_accumulating(state, entry)
    _require_hold_cursor(entry)
    if state.discovery_exhausted:
        raise StateInvariantError("discovery is already exhausted")
    if not state.documents:
        raise StateInvariantError(
            "exhaustion with zero reviewable documents is no-work, not a batch"
        )
    return state.model_copy(update={"discovery_exhausted": True})


def _apply_bundling_started(
    state: ControllerState,
    entry: BundlingStartedEntry,
) -> ControllerState:
    _require_accumulating(state, entry)
    _require_hold_cursor(entry)
    if not state.documents:
        raise StateInvariantError("cannot bundle zero documents")
    full = len(state.documents) == state.batch_target_count
    if not full and not state.discovery_exhausted:
        raise StateInvariantError(
            "bundling starts for a full batch or after discovery exhaustion"
        )
    return state.model_copy(update={"batch_status": BatchStatus.BUNDLING})


def _apply_document_bundling(
    state: ControllerState,
    entry: DocumentBundlingEntry,
) -> ControllerState:
    _require_batch(state, entry)
    _require_hold_cursor(entry)
    if state.batch_status is not BatchStatus.BUNDLING:
        raise StateInvariantError("document bundling requires batch status bundling")
    if any(document.phase is DocumentPhase.BUNDLING for document in state.documents):
        raise StateInvariantError("another document is already bundling")
    target = _next_incomplete(state)
    if target is None or target.document_id != entry.document_id:
        raise StateInvariantError("documents bundle in ascending id order")
    if target.phase is not DocumentPhase.SELECTED:
        raise StateInvariantError("only a selected document can start bundling")
    return _replace_document(
        state,
        target.model_copy(update={"phase": DocumentPhase.BUNDLING}),
    )


def _apply_document_bundled(
    state: ControllerState,
    entry: DocumentBundledEntry,
) -> ControllerState:
    _require_batch(state, entry)
    _require_hold_cursor(entry)
    if state.batch_status is not BatchStatus.BUNDLING:
        raise StateInvariantError("document completion requires batch status bundling")
    current = _find_document(state, entry.document_id)
    if current.phase is not DocumentPhase.BUNDLING:
        raise StateInvariantError("document must be bundling before it is bundled")
    bundled = current.model_copy(
        update={
            "phase": DocumentPhase.BUNDLED,
            "transcription_sha256": entry.transcription_sha256,
            "source_hashes": list(entry.sources),
            "bundle_manifest_path": entry.bundle_manifest_path,
        }
    )
    return _replace_document(state, bundled)


def _apply_batch_sealed(
    state: ControllerState,
    entry: BatchSealedEntry,
) -> ControllerState:
    _require_batch(state, entry)
    _require_hold_cursor(entry)
    if state.batch_status is not BatchStatus.BUNDLING:
        raise StateInvariantError("only a bundling batch can be sealed")
    if any(document.phase is not DocumentPhase.BUNDLED for document in state.documents):
        raise StateInvariantError("every selected document must be bundled before seal")
    ids = [document.document_id for document in state.documents]
    short_batch = len(ids) < state.batch_target_count
    if entry.document_ids != ids or entry.short_batch != short_batch:
        raise StateInvariantError("seal entry does not match the checkpoint documents")
    if entry.discovery_exhausted != state.discovery_exhausted:
        raise StateInvariantError("seal entry discovery flag does not match the checkpoint")
    expected_refs = [
        SealedDocumentRef(
            document_id=document.document_id,
            bundle_manifest_path=document.bundle_manifest_path or "",
            transcription_sha256=document.transcription_sha256 or "",
        )
        for document in state.documents
    ]
    if entry.documents != expected_refs:
        raise StateInvariantError("seal entry manifest refs do not match the checkpoint")
    return state.model_copy(
        update={
            "batch_status": BatchStatus.SEALED,
            "last_sealed_batch_id": state.current_batch_id,
        }
    )


def _target_for_selection(
    state: ControllerState,
    entry: DocumentSelectedEntry,
    *,
    allocates: bool,
) -> int:
    """Return the batch target after this selection.

    ``None`` on an allocating entry is the historical default of 5. The same
    ``None`` on a selection that joins an open batch, including a blocked
    batch that already has documents, keeps the target already recorded.
    """

    if not allocates:
        _reject_retarget(state, entry)
        return state.batch_target_count
    if entry.target_reviewable_count is None:
        return PILOT_TARGET_REVIEWABLE_COUNT
    return _require_batch_target(entry.target_reviewable_count)


def _reject_retarget(state: ControllerState, entry: DocumentSelectedEntry) -> None:
    recorded = entry.target_reviewable_count
    if recorded is not None and recorded != state.batch_target_count:
        raise StateInvariantError("an open batch keeps the target recorded at allocation")


def _require_batch_target(value: int) -> int:
    if value < MIN_BATCH_SIZE or value > MAX_BATCH_SIZE:
        raise StateInvariantError(
            f"batch target must be from {MIN_BATCH_SIZE} to {MAX_BATCH_SIZE}"
        )
    return value


def _next_batch_id(state: ControllerState) -> str:
    if state.batch_status is BatchStatus.SEALED:
        anchor = state.current_batch_id
    else:
        anchor = state.last_sealed_batch_id
    if anchor is None:
        return "batch-000001"
    number = int(anchor.split("-")[1]) + 1
    return f"batch-{number:06d}"


def _scan_batch_id(state: ControllerState) -> str | None:
    """Batch id for a 404 or non-eligible visit, or None when no batch is open."""

    if state.batch_status is BatchStatus.ACCUMULATING:
        return _require_batch_id(state)
    if state.batch_status in {
        BatchStatus.IDLE,
        BatchStatus.SCANNING,
        BatchStatus.SEALED,
        BatchStatus.NO_WORK,
    }:
        return None
    raise StateInvariantError("cannot record a scan visit in this status")


def _require_blocked_document(state: ControllerState, document_id: int) -> None:
    blocked = state.blocked
    if state.batch_status is not BatchStatus.BLOCKED or blocked is None:
        raise StateInvariantError("only a blocked checkpoint retries its blocked document")
    if document_id != blocked.document_id:
        raise StateInvariantError("retry must name the blocked document")


def _blocking_result_unchanged(
    state: ControllerState,
    *,
    code: BlockingIntegrityCode,
    message: str,
    details: dict[str, object] | None,
) -> bool:
    blocked = state.blocked
    if blocked is None:
        return False
    return (
        blocked.code == code
        and blocked.message == message
        and blocked.details == details
    )


def _visit_batch_id(state: ControllerState, document_id: int) -> str | None:
    """Batch id for a 404 or non-eligible visit.

    A blocked checkpoint resolves that same document and keeps its batch id.
    """

    if state.batch_status is BatchStatus.BLOCKED:
        _require_blocked_document(state, document_id)
        return state.current_batch_id
    return _scan_batch_id(state)


def _apply_scan_visit(
    state: ControllerState,
    entry: LedgerEntry,
    document_id: int,
) -> ControllerState:
    _require_advance_cursor(state, entry, document_id)
    if state.batch_status is BatchStatus.BLOCKED:
        _require_blocked_document(state, document_id)
        if state.documents:
            _require_batch(state, entry)
            return state.model_copy(
                update={
                    "batch_status": BatchStatus.ACCUMULATING,
                    "blocked": None,
                }
            )
        if entry.batch_id is not None:
            raise StateInvariantError("a scan visit outside an open batch has no batch id")
        return state.model_copy(
            update={
                "batch_status": BatchStatus.SCANNING,
                "blocked": None,
                "current_batch_id": None,
                "discovery_exhausted": False,
            }
        )
    if state.batch_status is BatchStatus.ACCUMULATING:
        _require_batch(state, entry)
        return state
    if entry.batch_id is not None:
        raise StateInvariantError("a scan visit outside an open batch has no batch id")
    if state.batch_status is BatchStatus.IDLE:
        return state.model_copy(update={"batch_status": BatchStatus.SCANNING})
    if state.batch_status is BatchStatus.NO_WORK:
        return state.model_copy(
            update={
                "batch_status": BatchStatus.SCANNING,
                "discovery_exhausted": False,
            }
        )
    if state.batch_status in {BatchStatus.SCANNING, BatchStatus.SEALED}:
        return state
    raise StateInvariantError("cannot record a scan visit in this status")


def _apply_no_work(state: ControllerState, entry: NoWorkEntry) -> ControllerState:
    _require_hold_cursor(entry)
    if state.batch_status is BatchStatus.NO_WORK:
        raise StateInvariantError("no-work is already recorded at this cursor")
    if state.batch_status not in {
        BatchStatus.IDLE,
        BatchStatus.SCANNING,
        BatchStatus.SEALED,
    }:
        raise StateInvariantError("no-work requires no open batch")
    last_sealed = state.last_sealed_batch_id
    if state.batch_status is BatchStatus.SEALED:
        last_sealed = state.current_batch_id
    return state.model_copy(
        update={
            "batch_status": BatchStatus.NO_WORK,
            "current_batch_id": None,
            "current_batch_document_ids": [],
            "documents": [],
            "blocked": None,
            "discovery_exhausted": True,
            "last_sealed_batch_id": last_sealed,
        }
    )


def _require_batch_id(state: ControllerState) -> str:
    if state.current_batch_id is None:
        raise StateInvariantError("no batch is open")
    return state.current_batch_id


def _require_batch(state: ControllerState, entry: LedgerEntry) -> None:
    if state.current_batch_id is None or entry.batch_id != state.current_batch_id:
        raise StateInvariantError("ledger entry batch_id does not match the checkpoint")


def _require_accumulating(state: ControllerState, entry: LedgerEntry) -> None:
    _require_batch(state, entry)
    if state.batch_status is not BatchStatus.ACCUMULATING:
        raise StateInvariantError("this decision requires an accumulating batch")


def _require_advance_cursor(
    state: ControllerState,
    entry: LedgerEntry,
    document_id: int,
) -> None:
    try:
        assert_document_after_cursor(state.last_scanned_document_id, document_id)
    except ValueError as exc:
        raise StateInvariantError(str(exc)) from exc
    if entry.cursor_after != document_id:
        raise StateInvariantError("advancing decisions set cursor_after to the document id")


def _require_hold_cursor(entry: LedgerEntry) -> None:
    if entry.cursor_after != entry.cursor_before:
        raise StateInvariantError("this event must not advance the scan cursor")


def _require_effect(
    expected: CursorEffect,
    *,
    http_status: int,
    outcome: str | None,
) -> None:
    actual = cursor_effect(http_status=http_status, outcome=outcome)
    if actual is not expected:
        raise StateInvariantError(f"cursor policy returned {actual}")


def _find_document(state: ControllerState, document_id: int) -> DocumentLifecycle:
    for document in state.documents:
        if document.document_id == document_id:
            return document
    raise StateInvariantError(f"document {document_id} is not in the current batch")


def _next_incomplete(state: ControllerState) -> DocumentLifecycle | None:
    for document in state.documents:
        if document.phase is not DocumentPhase.BUNDLED:
            return document
    return None


def _replace_document(
    state: ControllerState,
    replacement: DocumentLifecycle,
) -> ControllerState:
    documents = [
        replacement if document.document_id == replacement.document_id else document
        for document in state.documents
    ]
    return state.model_copy(update={"documents": documents})
