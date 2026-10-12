"""Pure scan-cursor decisions.

The scan cursor records which document ids have been visited. It is not batch
completion. ``404`` and ``NON_ELIGIBLE`` may advance it. ``BLOCKING_INTEGRITY_ERROR``
holds it. Any other HTTP status or outcome stops without advancing.
"""

from __future__ import annotations

from enum import Enum

class CursorEffect(str, Enum):
    """What a single bundle fetch may do to ``last_scanned_document_id``."""

    ADVANCE = "advance"
    HOLD_AND_STOP = "hold_and_stop"
    STOP_FAIL_CLOSED = "stop_fail_closed"


def assert_document_after_cursor(
    last_scanned_document_id: int | None,
    document_id: int,
) -> None:
    """Require ``document_id`` to be strictly after the exclusive scan cursor."""

    if document_id < 1:
        raise ValueError("document_id must be a positive integer")
    if last_scanned_document_id is None:
        return
    if last_scanned_document_id < 1:
        raise ValueError("last_scanned_document_id must be a positive integer")
    if document_id <= last_scanned_document_id:
        raise ValueError(
            "document_id must be greater than last_scanned_document_id"
        )


def cursor_effect(*, http_status: int, outcome: str | None) -> CursorEffect:
    """Map one bundle HTTP result to a cursor decision.

    ``outcome`` is the bundle JSON discriminant. It is ignored for non-200
    responses. HTTP 404 advances even when the body is not JSON, because the
    server uses 404 for missing and not-viewable documents.
    """

    if http_status == 404:
        return CursorEffect.ADVANCE
    if http_status != 200:
        return CursorEffect.STOP_FAIL_CLOSED
    if outcome == "NON_ELIGIBLE":
        return CursorEffect.ADVANCE
    if outcome == "REVIEWABLE":
        return CursorEffect.ADVANCE
    if outcome == "BLOCKING_INTEGRITY_ERROR":
        return CursorEffect.HOLD_AND_STOP
    return CursorEffect.STOP_FAIL_CLOSED
