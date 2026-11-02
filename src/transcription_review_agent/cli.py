"""Command line entry points.

``review-next`` prepares, inspects, accepts, and finalizes a local review.
It does not scan VS Archive or write a sealed bundle.
"""

from __future__ import annotations

import sys

from transcription_review_agent.controller.run import review_next
from transcription_review_agent.review.errors import ReviewError, ReviewUsageError


def main(argv: list[str] | None = None) -> int:
    """Run one local review operation, or refuse an empty command."""

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        review_next(None, args)
    except ReviewUsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ReviewError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
