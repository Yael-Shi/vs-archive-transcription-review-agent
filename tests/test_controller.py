from __future__ import annotations

import pytest

from transcription_review_agent.controller.run import review_next


def test_review_next_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        review_next(None)
