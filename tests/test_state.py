from __future__ import annotations

import pytest

from transcription_review_agent.api_client.schemas import BlockingIntegrityCode, NonEligibleCode
from transcription_review_agent.errors import StateInvariantError
from transcription_review_agent.state.schema import (
    BatchStatus,
    DocumentPhase,
    initial_state,
)
from transcription_review_agent.state.transitions import (
    apply_ledger_entry,
    begin_bundling,
    note_blocking,
    note_no_work,
    note_non_eligible,
    note_not_found,
    seal_batch,
    select_reviewable,
)
from support import at, seal_short_batch, select_one


def test_initial_state_round_trip() -> None:
    state = initial_state()
    assert state.batch_status is BatchStatus.IDLE
    assert state.last_scanned_document_id is None
    assert state.current_batch_id is None
    restored = type(state).model_validate_json(state.model_dump_json())
    assert restored == state


def test_not_found_and_non_eligible_advance_cursor_without_joining_the_batch() -> None:
    state = apply_ledger_entry(
        initial_state(),
        note_not_found(initial_state(), document_id=4, occurred_at=at(1)),
    )
    state = apply_ledger_entry(
        state,
        note_non_eligible(
            state,
            document_id=5,
            code=NonEligibleCode.VERIFIED_TRANSCRIPTION_OUT_OF_SCOPE,
            occurred_at=at(2),
        ),
    )
    assert state.last_scanned_document_id == 5
    assert state.current_batch_document_ids == []
    assert state.current_batch_id is None
    assert state.batch_status is BatchStatus.SCANNING


def test_reviewable_selection_advances_cursor_and_stays_unbundled() -> None:
    state = select_one(initial_state(), 10, at_seconds=1)
    assert state.last_scanned_document_id == 10
    assert state.current_batch_document_ids == [10]
    assert state.documents[0].phase is DocumentPhase.SELECTED
    assert state.documents[0].transcription_sha256 is None
    assert state.batch_status is BatchStatus.ACCUMULATING


def test_blocking_stops_without_advancing_past_the_document() -> None:
    state = select_one(initial_state(), 10, at_seconds=1)
    state = apply_ledger_entry(
        state,
        note_non_eligible(
            state,
            document_id=12,
            code=NonEligibleCode.NOT_READY_PROCESSING,
            occurred_at=at(2),
        ),
    )
    blocked = apply_ledger_entry(
        state,
        note_blocking(
            state,
            document_id=18,
            code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
            message="Document.file_s3_key is missing",
            occurred_at=at(3),
        ),
    )
    assert blocked.batch_status is BatchStatus.BLOCKED
    assert blocked.last_scanned_document_id == 12
    assert blocked.blocked is not None
    assert blocked.blocked.document_id == 18
    assert blocked.current_batch_document_ids == [10]
    assert 18 not in blocked.current_batch_document_ids
    with pytest.raises(StateInvariantError):
        select_reviewable(
            blocked,
            document_id=19,
            archive_item_id=5019,
            language="en",
            occurred_at=at(4),
        )
    assert (
        note_blocking(
            blocked,
            document_id=18,
            code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
            message="Document.file_s3_key is missing",
            occurred_at=at(4),
        )
        is None
    )
    assert blocked.last_scanned_document_id == 12
    assert blocked.updated_at == at(3)
    with pytest.raises(StateInvariantError):
        apply_ledger_entry(blocked, begin_bundling(blocked, occurred_at=at(4)))


def test_sixth_reviewable_document_is_rejected() -> None:
    state = initial_state()
    for index, document_id in enumerate((1, 2, 3, 4, 5)):
        state = select_one(state, document_id, at_seconds=index + 1)
    with pytest.raises(StateInvariantError):
        apply_ledger_entry(
            state,
            select_reviewable(
                state,
                document_id=6,
                archive_item_id=9,
                language="fr",
                occurred_at=at(7),
            ),
        )


def test_short_batch_seals_only_after_every_document_is_bundled() -> None:
    state = select_one(initial_state(), 7, at_seconds=1)
    state = select_one(state, 9, at_seconds=2)
    with pytest.raises(StateInvariantError):
        seal_batch(state, occurred_at=at(3))
    sealed = seal_short_batch(state)
    assert sealed.batch_status is BatchStatus.SEALED
    assert sealed.current_batch_document_ids == [7, 9]
    assert sealed.last_scanned_document_id == 9
    assert all(document.phase is DocumentPhase.BUNDLED for document in sealed.documents)
    assert sealed.documents[0].transcription_sha256 is not None
    assert sealed.discovery_exhausted is True


def test_next_batch_keeps_history_only_by_replacing_hot_batch_fields() -> None:
    first = seal_short_batch(select_one(initial_state(), 3, at_seconds=1))
    second = select_one(first, 9, at_seconds=100)
    assert second.current_batch_id == "batch-000002"
    assert second.current_batch_document_ids == [9]
    assert second.last_scanned_document_id == 9
    assert second.last_sealed_batch_id == "batch-000001"
    assert second.batch_status is BatchStatus.ACCUMULATING


def test_cursor_cannot_move_backwards_or_past_a_block_by_a_hand_built_entry() -> None:
    state = select_one(initial_state(), 4, at_seconds=1)
    entry = note_blocking(
        state,
        document_id=4,
        code=BlockingIntegrityCode.MISSING_DISPLAYED_TRANSCRIPTION,
        message="already selected",
        occurred_at=at(2),
    )
    with pytest.raises(StateInvariantError):
        apply_ledger_entry(state, entry)
    skipped = note_not_found(state, document_id=4, occurred_at=at(2))
    with pytest.raises(StateInvariantError):
        apply_ledger_entry(state, skipped)


def test_zero_reviewable_exhaustion_is_no_work_and_does_not_consume_batch_ids() -> None:
    state = apply_ledger_entry(
        initial_state(),
        note_no_work(initial_state(), occurred_at=at(0)),
    )
    assert state.batch_status is BatchStatus.NO_WORK
    assert state.current_batch_id is None
    assert state.last_sealed_batch_id is None
    assert state.documents == []
    assert state.discovery_exhausted is True
    assert state.last_scanned_document_id is None

    with pytest.raises(StateInvariantError):
        note_no_work(state, occurred_at=at(1))
    with pytest.raises(StateInvariantError):
        apply_ledger_entry(state, begin_bundling(state, occurred_at=at(1)))
    with pytest.raises(StateInvariantError):
        seal_batch(state, occurred_at=at(1))

    opened = select_one(state, 11, at_seconds=2)
    assert opened.current_batch_id == "batch-000001"
    assert opened.current_batch_document_ids == [11]
    assert opened.last_sealed_batch_id is None


def test_skips_then_exhaustion_still_allocate_no_batch_id() -> None:
    state = apply_ledger_entry(
        initial_state(),
        note_not_found(initial_state(), document_id=4, occurred_at=at(0)),
    )
    assert state.batch_status is BatchStatus.SCANNING
    assert state.current_batch_id is None
    assert state.last_scanned_document_id == 4

    state = apply_ledger_entry(state, note_no_work(state, occurred_at=at(1)))
    assert state.batch_status is BatchStatus.NO_WORK
    assert state.last_scanned_document_id == 4
    assert state.last_sealed_batch_id is None

    state = apply_ledger_entry(
        state,
        note_not_found(state, document_id=8, occurred_at=at(2)),
    )
    assert state.batch_status is BatchStatus.SCANNING
    state = apply_ledger_entry(state, note_no_work(state, occurred_at=at(3)))
    opened = select_one(state, 9, at_seconds=4)
    assert opened.current_batch_id == "batch-000001"
    assert opened.current_batch_document_ids == [9]
    assert opened.last_scanned_document_id == 9


def test_no_work_after_a_sealed_batch_keeps_the_next_real_id() -> None:
    sealed = seal_short_batch(select_one(initial_state(), 3, at_seconds=1))
    assert sealed.current_batch_id == "batch-000001"
    assert sealed.last_sealed_batch_id == "batch-000001"
    cursor = sealed.last_scanned_document_id

    quiet = apply_ledger_entry(sealed, note_no_work(sealed, occurred_at=at(100)))
    assert quiet.batch_status is BatchStatus.NO_WORK
    assert quiet.current_batch_id is None
    assert quiet.documents == []
    assert quiet.last_sealed_batch_id == "batch-000001"
    assert quiet.last_scanned_document_id == cursor
    with pytest.raises(StateInvariantError):
        note_no_work(quiet, occurred_at=at(101))

    opened = select_one(quiet, 4, at_seconds=102)
    assert opened.current_batch_id == "batch-000002"
    assert opened.current_batch_document_ids == [4]
    assert opened.last_sealed_batch_id == "batch-000001"


def test_one_selected_document_can_still_seal_a_short_batch() -> None:
    sealed = seal_short_batch(select_one(initial_state(), 3, at_seconds=1))
    assert sealed.batch_status is BatchStatus.SEALED
    assert sealed.current_batch_document_ids == [3]
    assert sealed.discovery_exhausted is True
    assert sealed.last_sealed_batch_id == "batch-000001"


def _block(
    state,
    document_id: int,
    *,
    at_seconds: int,
):
    entry = note_blocking(
        state,
        document_id=document_id,
        code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
        message="Document.file_s3_key is missing",
        details={"order_index": 0},
        occurred_at=at(at_seconds),
    )
    assert entry is not None
    return apply_ledger_entry(state, entry)


def test_blocking_before_any_batch_allocates_nothing() -> None:
    blocked = _block(initial_state(), 3, at_seconds=1)
    assert blocked.batch_status is BatchStatus.BLOCKED
    assert blocked.current_batch_id is None
    assert blocked.documents == []
    assert blocked.current_batch_document_ids == []
    assert blocked.last_scanned_document_id is None
    assert blocked.last_sealed_batch_id is None
    assert blocked.blocked is not None
    assert blocked.blocked.document_id == 3
    assert blocked.blocked.code is BlockingIntegrityCode.MISSING_PRIMARY_SOURCE
    assert blocked.blocked.message == "Document.file_s3_key is missing"
    assert blocked.blocked.details == {"order_index": 0}
    assert (
        note_blocking(
            blocked,
            document_id=3,
            code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
            message="Document.file_s3_key is missing",
            details={"order_index": 0},
            occurred_at=at(2),
        )
        is None
    )
    assert blocked.last_scanned_document_id is None
    assert blocked.updated_at == at(1)
    assert blocked.applied_ledger_sequence == 1


def test_blocking_after_skips_still_allocates_no_batch() -> None:
    scanning = apply_ledger_entry(
        initial_state(),
        note_not_found(initial_state(), document_id=4, occurred_at=at(1)),
    )
    blocked = _block(scanning, 8, at_seconds=2)
    assert blocked.batch_status is BatchStatus.BLOCKED
    assert blocked.last_scanned_document_id == 4
    assert blocked.current_batch_id is None
    assert blocked.documents == []
    assert blocked.last_sealed_batch_id is None
    assert (
        note_blocking(
            blocked,
            document_id=8,
            code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
            message="Document.file_s3_key is missing",
            details={"order_index": 0},
            occurred_at=at(3),
        )
        is None
    )
    assert blocked.batch_status is BatchStatus.BLOCKED
    assert blocked.last_scanned_document_id == 4
    assert blocked.updated_at == at(2)


def test_pre_batch_block_resolves_to_reviewable_in_one_event() -> None:
    blocked = _block(initial_state(), 3, at_seconds=1)
    selected = select_one(blocked, 3, at_seconds=2)
    assert selected.applied_ledger_sequence == blocked.applied_ledger_sequence + 1
    assert selected.batch_status is BatchStatus.ACCUMULATING
    assert selected.current_batch_id == "batch-000001"
    assert selected.current_batch_document_ids == [3]
    assert selected.last_scanned_document_id == 3
    assert selected.blocked is None
    assert selected.last_sealed_batch_id is None


def test_pre_batch_block_resolves_to_non_eligible_or_not_found_in_one_event() -> None:
    blocked = _block(initial_state(), 3, at_seconds=1)
    skipped = apply_ledger_entry(
        blocked,
        note_non_eligible(
            blocked,
            document_id=3,
            code=NonEligibleCode.NOT_OCR_DOCUMENT,
            occurred_at=at(2),
        ),
    )
    missing = apply_ledger_entry(
        blocked,
        note_not_found(blocked, document_id=3, occurred_at=at(2)),
    )
    for state in (skipped, missing):
        assert state.applied_ledger_sequence == blocked.applied_ledger_sequence + 1
        assert state.last_scanned_document_id == 3
        assert state.current_batch_id is None
        assert state.documents == []
        assert state.blocked is None
        assert state.batch_status is BatchStatus.SCANNING
        assert state.last_sealed_batch_id is None


def test_changed_blocking_result_replaces_the_record_without_moving_the_cursor() -> None:
    blocked = _block(initial_state(), 3, at_seconds=1)
    entry = note_blocking(
        blocked,
        document_id=3,
        code=BlockingIntegrityCode.MISSING_SOURCE_FILE,
        message="source file missing",
        details={"order_index": 1},
        occurred_at=at(2),
    )
    assert entry is not None
    updated = apply_ledger_entry(blocked, entry)
    assert updated.batch_status is BatchStatus.BLOCKED
    assert updated.last_scanned_document_id is None
    assert updated.current_batch_id is None
    assert updated.documents == []
    assert updated.last_sealed_batch_id is None
    assert updated.updated_at == at(2)
    assert updated.blocked is not None
    assert updated.blocked.document_id == 3
    assert updated.blocked.code is BlockingIntegrityCode.MISSING_SOURCE_FILE
    assert updated.blocked.details == {"order_index": 1}
    assert updated.applied_ledger_sequence == 2
    assert (
        note_blocking(
            updated,
            document_id=3,
            code=BlockingIntegrityCode.MISSING_SOURCE_FILE,
            message="source file missing",
            details={"order_index": 1},
            occurred_at=at(3),
        )
        is None
    )
    assert updated.updated_at == at(2)
    assert updated.last_scanned_document_id is None


def test_blocking_after_selection_keeps_the_open_batch() -> None:
    selected = select_one(initial_state(), 10, at_seconds=1)
    selected = select_one(selected, 11, at_seconds=2)
    blocked = _block(selected, 18, at_seconds=3)
    assert blocked.batch_status is BatchStatus.BLOCKED
    assert blocked.current_batch_id == "batch-000001"
    assert blocked.current_batch_document_ids == [10, 11]
    assert blocked.last_scanned_document_id == 11
    assert blocked.last_sealed_batch_id is None
    assert blocked.blocked is not None
    assert blocked.blocked.document_id == 18


def test_blocked_batch_resolves_to_reviewable_on_the_same_batch() -> None:
    selected = select_one(initial_state(), 10, at_seconds=1)
    blocked = _block(selected, 18, at_seconds=2)
    joined = select_one(blocked, 18, at_seconds=3)
    assert joined.applied_ledger_sequence == blocked.applied_ledger_sequence + 1
    assert joined.batch_status is BatchStatus.ACCUMULATING
    assert joined.current_batch_id == "batch-000001"
    assert joined.current_batch_document_ids == [10, 18]
    assert joined.last_scanned_document_id == 18
    assert joined.blocked is None
    assert joined.last_sealed_batch_id is None


def test_blocked_batch_resolves_to_non_eligible_or_not_found() -> None:
    selected = select_one(initial_state(), 10, at_seconds=1)
    blocked = _block(selected, 18, at_seconds=2)
    skipped = apply_ledger_entry(
        blocked,
        note_non_eligible(
            blocked,
            document_id=18,
            code=NonEligibleCode.LANGUAGE_NOT_IN_SCOPE,
            occurred_at=at(3),
        ),
    )
    missing = apply_ledger_entry(
        blocked,
        note_not_found(blocked, document_id=18, occurred_at=at(3)),
    )
    for state in (skipped, missing):
        assert state.applied_ledger_sequence == blocked.applied_ledger_sequence + 1
        assert state.batch_status is BatchStatus.ACCUMULATING
        assert state.current_batch_id == "batch-000001"
        assert state.current_batch_document_ids == [10]
        assert state.last_scanned_document_id == 18
        assert state.blocked is None
        assert state.last_sealed_batch_id is None


def test_unchanged_block_on_a_selected_batch_appends_nothing() -> None:
    selected = select_one(initial_state(), 10, at_seconds=1)
    blocked = _block(selected, 18, at_seconds=2)
    assert (
        note_blocking(
            blocked,
            document_id=18,
            code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
            message="Document.file_s3_key is missing",
            details={"order_index": 0},
            occurred_at=at(5),
        )
        is None
    )
    assert blocked.batch_status is BatchStatus.BLOCKED
    assert blocked.last_scanned_document_id == 10
    assert blocked.current_batch_id == "batch-000001"
    assert blocked.current_batch_document_ids == [10]
    assert blocked.last_sealed_batch_id is None
    assert blocked.updated_at == at(2)
    with pytest.raises(StateInvariantError):
        note_not_found(blocked, document_id=19, occurred_at=at(5))
    with pytest.raises(StateInvariantError):
        note_blocking(
            blocked,
            document_id=19,
            code=BlockingIntegrityCode.MISSING_SOURCE_FILE,
            message="different document",
            occurred_at=at(5),
        )
    assert blocked.last_scanned_document_id == 10


def test_resolving_a_block_after_a_sealed_batch_keeps_last_sealed_batch_id() -> None:
    sealed = seal_short_batch(select_one(initial_state(), 3, at_seconds=1))
    blocked = _block(sealed, 9, at_seconds=100)
    assert blocked.last_sealed_batch_id == "batch-000001"
    assert blocked.current_batch_id is None
    assert blocked.documents == []
    assert blocked.last_scanned_document_id == 3
    selected = select_one(blocked, 9, at_seconds=101)
    assert selected.applied_ledger_sequence == blocked.applied_ledger_sequence + 1
    assert selected.batch_status is BatchStatus.ACCUMULATING
    assert selected.current_batch_id == "batch-000002"
    assert selected.current_batch_document_ids == [9]
    assert selected.last_scanned_document_id == 9
    assert selected.blocked is None
    assert selected.last_sealed_batch_id == "batch-000001"
