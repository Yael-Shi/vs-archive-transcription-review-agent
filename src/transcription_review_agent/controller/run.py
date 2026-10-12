"""Reserved entry point for a future ``review-next`` run.

The loop is not implemented. Nothing in this module reads configuration,
opens a socket, or writes state.
"""

from __future__ import annotations

from transcription_review_agent.config import AgentConfig


def review_next(config: AgentConfig | None = None) -> None:
    """Future run: scan, bundle, and checkpoint. Not implemented.

    When implemented, one invocation will:

    1. Read the checkpoint and replay the ledger if the checkpoint is behind.
    2. Scan forward from ``last_scanned_document_id``.
    3. Accumulate up to 5 ``REVIEWABLE`` documents. The first selection allocates
       its batch id in that same ledger entry.
    4. Stop immediately on ``BLOCKING_INTEGRITY_ERROR`` without moving the
       cursor past that document and without allocating a batch when none is
       open. A later run re-fetches that same document while the checkpoint
       stays blocked. An unchanged blocking result appends nothing. A
       reviewable, non-eligible, or 404 result is one ledger entry that
       resolves the block.
    5. If discovery exhausts with zero ``REVIEWABLE`` documents, record
       no-work, keep the cursor, and do not allocate a batch id or publish a
       manifest. The same cursor again is not a new no-work event.
    6. Otherwise download and fingerprint sources for the selected documents.
    7. Publish an immutable batch bundle.
    8. Checkpoint state.
    9. Stop.

    ``config`` is accepted so the signature can stay stable. This milestone
    does not read it.
    """

    del config
    raise NotImplementedError(
        "Pilot 01 controller loop is not implemented. "
        "No VS Archive request is made."
    )
