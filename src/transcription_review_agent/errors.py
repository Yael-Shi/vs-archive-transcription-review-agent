"""Errors for configuration, state integrity, and immutable bundles."""


class TranscriptionReviewAgentError(Exception):
    """Base error for this agent."""


class ConfigError(TranscriptionReviewAgentError):
    """Configuration is missing or violates the read-only boundary."""


class StateInvariantError(TranscriptionReviewAgentError):
    """A ledger transition would break the controller state model."""


class StateIntegrityError(TranscriptionReviewAgentError):
    """On-disk checkpoint and ledger cannot be reconciled safely."""


class ImmutableBundleError(TranscriptionReviewAgentError):
    """A completed bundle already exists and must not be overwritten."""


class BundleVerificationError(TranscriptionReviewAgentError):
    """A sealed bundle's bytes do not match its manifest. Nothing was rewritten."""
