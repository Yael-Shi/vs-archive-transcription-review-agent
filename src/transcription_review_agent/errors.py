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


class BundleAcquisitionError(TranscriptionReviewAgentError):
    """Bundling stopped before the document was marked bundled."""


class FreshBundleRejectedError(BundleAcquisitionError):
    """The fresh review bundle is missing, reclassified, or no longer the selected document."""


class SourceDownloadError(BundleAcquisitionError):
    """A source GET failed. The message never includes the presigned URL."""

    def __init__(
        self,
        message: str,
        *,
        document_id: int,
        order_index: int,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.document_id = document_id
        self.order_index = order_index
        self.status_code = status_code


class BundleConflictError(BundleAcquisitionError):
    """An existing final document bundle does not match the selected document."""
