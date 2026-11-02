"""Errors for the local review workflow.

These are distinct from discovery and bundle sealing. Review code does not
rewrite a sealed archive bundle.
"""

from __future__ import annotations


class ReviewError(Exception):
    """Base error for review workflow failures."""


class ReviewPathError(ReviewError):
    """A path is outside the review workspace or is not allowlisted."""


class ReviewLimitError(ReviewError):
    """A configured page, document, or challenge limit was exceeded."""


class ReviewIntegrityError(ReviewError):
    """Review checkpoint and ledger cannot be reconciled, or a hash moved."""


class ReviewPolicyError(ReviewError):
    """The requested transition violates the v1 review policy."""


class ReviewUsageError(ReviewError):
    """The review command line was incomplete. No archive request is made."""


class AlignmentLimitError(ReviewError):
    """Region alignment stopped before recursion or memory could grow without a bound."""


class ReviewResultMissing(ReviewError):
    """An expected provider file is not in the shared output directory yet."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.reason = "missing"


class ReviewOutputRejected(ReviewError):
    """A provider file failed identity, schema, freshness, or duplication checks."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
