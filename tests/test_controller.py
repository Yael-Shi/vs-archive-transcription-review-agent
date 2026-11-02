from __future__ import annotations

import pytest

from transcription_review_agent.controller.run import review_next
from transcription_review_agent.review.errors import ReviewUsageError


def test_review_next_without_an_operation_makes_no_archive_request() -> None:
    with pytest.raises(ReviewUsageError, match="No VS Archive request is made"):
        review_next(None)
