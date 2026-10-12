"""Command line entry points.

``review-next`` is reserved. It exits before configuration or network access.
"""

from __future__ import annotations

import sys

from transcription_review_agent.controller.run import review_next


def main(argv: list[str] | None = None) -> int:
    """Exit without scanning, downloading, or writing state."""

    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        print("usage: review-next", file=sys.stderr)
        return 2
    try:
        review_next(None)
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
