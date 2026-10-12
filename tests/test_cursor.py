from __future__ import annotations

import pytest

from transcription_review_agent.discovery.cursor import (
    CursorEffect,
    assert_document_after_cursor,
    cursor_effect,
)


def test_404_and_non_eligible_advance_blocking_holds() -> None:
    assert cursor_effect(http_status=404, outcome=None) is CursorEffect.ADVANCE
    assert cursor_effect(http_status=200, outcome="NON_ELIGIBLE") is CursorEffect.ADVANCE
    assert cursor_effect(http_status=200, outcome="REVIEWABLE") is CursorEffect.ADVANCE
    assert (
        cursor_effect(http_status=200, outcome="BLOCKING_INTEGRITY_ERROR")
        is CursorEffect.HOLD_AND_STOP
    )


def test_unexpected_http_and_outcomes_do_not_advance() -> None:
    assert cursor_effect(http_status=500, outcome=None) is CursorEffect.STOP_FAIL_CLOSED
    assert cursor_effect(http_status=400, outcome=None) is CursorEffect.STOP_FAIL_CLOSED
    assert cursor_effect(http_status=200, outcome="SOMETHING_ELSE") is CursorEffect.STOP_FAIL_CLOSED
    assert cursor_effect(http_status=404, outcome="BLOCKING_INTEGRITY_ERROR") is CursorEffect.ADVANCE


def test_cursor_is_exclusive() -> None:
    assert_document_after_cursor(None, 1)
    assert_document_after_cursor(10, 11)
    with pytest.raises(ValueError):
        assert_document_after_cursor(10, 10)
