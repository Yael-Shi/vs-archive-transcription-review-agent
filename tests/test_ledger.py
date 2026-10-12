from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcription_review_agent.api_client.schemas import BlockingIntegrityCode
from transcription_review_agent.errors import StateIntegrityError, StateInvariantError
from transcription_review_agent.state.ledger import RunBlockedEntry
from transcription_review_agent.state.schema import BatchStatus, initial_state, validate_controller_state
from transcription_review_agent.state.store import StateStore
from transcription_review_agent.state.transitions import note_blocking, select_reviewable
from support import at, select_one


def test_commit_round_trip_and_refuses_rewrite(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    assert store.load() == initial_state()
    assert not store.checkpoint_path.exists()

    selected = store.commit(
        select_reviewable(
            initial_state(),
            document_id=8,
            archive_item_id=80,
            language="fr",
            occurred_at=at(1),
        )
    )
    assert store.load() == selected
    assert selected.current_batch_id == "batch-000001"
    assert selected.current_batch_document_ids == [8]
    assert selected.last_scanned_document_id == 8
    lines = store.ledger_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["sequence"] for line in lines] == [1]
    with pytest.raises(StateIntegrityError):
        store.append_ledger_line(
            select_reviewable(
                initial_state(),
                document_id=8,
                archive_item_id=80,
                language="fr",
                occurred_at=at(1),
            )
        )


def test_ledger_ahead_of_checkpoint_is_replayed(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    first = store.commit(
        select_reviewable(
            initial_state(),
            document_id=2,
            archive_item_id=20,
            language="en",
            occurred_at=at(0),
        )
    )
    entry = select_reviewable(
        first,
        document_id=4,
        archive_item_id=40,
        language="en",
        occurred_at=at(1),
    )
    store.append_ledger_line(entry)
    stale = store.checkpoint_path.read_text(encoding="utf-8")
    assert '"applied_ledger_sequence": 1' in stale

    loaded = store.load()
    assert loaded.applied_ledger_sequence == 2
    assert loaded.last_scanned_document_id == 4
    assert loaded.current_batch_document_ids == [2, 4]
    assert loaded.batch_status is BatchStatus.ACCUMULATING
    assert '"applied_ledger_sequence": 2' in store.checkpoint_path.read_text(encoding="utf-8")


def test_checkpoint_that_disagrees_with_ledger_is_rejected(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    store.commit(
        select_reviewable(
            initial_state(),
            document_id=6,
            archive_item_id=60,
            language="en",
            occurred_at=at(0),
        )
    )
    checkpoint = json.loads(store.checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["last_scanned_document_id"] = 99
    store.checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(StateIntegrityError):
        store.load()


def test_checkpoint_ahead_of_ledger_is_rejected(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    committed = store.commit(
        select_reviewable(
            initial_state(),
            document_id=6,
            archive_item_id=60,
            language="en",
            occurred_at=at(0),
        )
    )
    ahead = select_one(committed, 7, at_seconds=1)
    store.write_checkpoint(ahead)
    with pytest.raises(StateIntegrityError):
        store.load()


def test_gapped_ledger_is_rejected(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    store.commit(
        select_reviewable(
            initial_state(),
            document_id=6,
            archive_item_id=60,
            language="en",
            occurred_at=at(0),
        )
    )
    with store.ledger_path.open("a", encoding="utf-8") as handle:
        handle.write('{"sequence": 3}\n')
    with pytest.raises(StateIntegrityError):
        store.read_ledger()


def test_interrupted_first_selection_replays_a_nonempty_batch(tmp_path: Path) -> None:
    """The earliest durable point already contains the first selected document."""

    store = StateStore(tmp_path / "state")
    entry = select_reviewable(
        initial_state(),
        document_id=5,
        archive_item_id=50,
        language="en",
        occurred_at=at(1),
    )
    store.append_ledger_line(entry)
    assert not store.checkpoint_path.exists()

    loaded = store.load()
    assert loaded.batch_status is BatchStatus.ACCUMULATING
    assert loaded.current_batch_id == "batch-000001"
    assert loaded.current_batch_document_ids == [5]
    assert loaded.last_scanned_document_id == 5
    assert loaded.documents[0].phase.value == "selected"
    assert loaded.applied_ledger_sequence == 1

    empty = loaded.model_copy(
        update={"documents": [], "current_batch_document_ids": []}
    )
    with pytest.raises(StateInvariantError):
        validate_controller_state(empty)


def _commit_block(store: StateStore):
    entry = note_blocking(
        initial_state(),
        document_id=3,
        code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
        message="Document.file_s3_key is missing",
        details={"order_index": 0},
        occurred_at=at(1),
    )
    assert entry is not None
    return store.commit(entry)


def test_unchanged_blocking_result_leaves_the_ledger_untouched(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    blocked = _commit_block(store)
    ledger_before = store.ledger_path.read_bytes()
    checkpoint_before = store.checkpoint_path.read_bytes()

    for _ in range(3):
        assert (
            note_blocking(
                blocked,
                document_id=3,
                code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
                message="Document.file_s3_key is missing",
                details={"order_index": 0},
                occurred_at=at(9),
            )
            is None
        )

    assert store.ledger_path.read_bytes() == ledger_before
    assert store.checkpoint_path.read_bytes() == checkpoint_before
    loaded = store.load()
    assert loaded == blocked
    assert loaded.updated_at == blocked.updated_at
    assert loaded.last_scanned_document_id is None

    duplicate = RunBlockedEntry(
        sequence=blocked.applied_ledger_sequence + 1,
        occurred_at=at(9),
        batch_id=None,
        cursor_before=None,
        cursor_after=None,
        document_id=3,
        code=BlockingIntegrityCode.MISSING_PRIMARY_SOURCE,
        message="Document.file_s3_key is missing",
        details={"order_index": 0},
    )
    with pytest.raises(StateInvariantError):
        store.commit(duplicate)
    assert store.ledger_path.read_bytes() == ledger_before
    assert store.load() == blocked


def test_crash_before_retry_outcome_keeps_the_blocked_checkpoint(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state")
    blocked = _commit_block(store)
    ledger_before = store.ledger_path.read_bytes()
    planned = select_reviewable(
        blocked,
        document_id=3,
        archive_item_id=50,
        language="en",
        occurred_at=at(2),
    )
    assert planned.event_type == "document_selected"
    assert planned.batch_id == "batch-000001"
    assert store.ledger_path.read_bytes() == ledger_before

    loaded = store.load()
    assert loaded == blocked
    assert loaded.batch_status is BatchStatus.BLOCKED
    assert loaded.blocked is not None
    assert loaded.blocked.document_id == 3
    assert loaded.current_batch_id is None
    assert loaded.last_scanned_document_id is None
