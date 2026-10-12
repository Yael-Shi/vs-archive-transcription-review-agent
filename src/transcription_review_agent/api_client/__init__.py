"""Typed Pilot 01 HTTP contracts and the read-only client that calls them."""

from transcription_review_agent.api_client.client import (
    ReviewBundleNotFound,
    TranscriptionReviewApiClient,
)
from transcription_review_agent.api_client.errors import (
    ApiClientError,
    ApiInvalidJsonError,
    ApiProtocolError,
    ApiSchemaError,
    ApiTimeoutError,
    ApiTransportError,
    ScanQueryError,
    UnexpectedApiResponseError,
)
from transcription_review_agent.api_client.schemas import (
    BlockingIntegrityCode,
    BundleOutcome,
    NonEligibleCode,
    ReviewBundleResponse,
    ScanQueryErrorBody,
    ScanResponse,
    bundle_path,
    parse_review_bundle,
    parse_scan_query_error,
    parse_scan_response,
)

__all__ = [
    "ApiClientError",
    "ApiInvalidJsonError",
    "ApiProtocolError",
    "ApiSchemaError",
    "ApiTimeoutError",
    "ApiTransportError",
    "BlockingIntegrityCode",
    "BundleOutcome",
    "NonEligibleCode",
    "ReviewBundleNotFound",
    "ReviewBundleResponse",
    "ScanQueryError",
    "ScanQueryErrorBody",
    "ScanResponse",
    "TranscriptionReviewApiClient",
    "UnexpectedApiResponseError",
    "bundle_path",
    "parse_review_bundle",
    "parse_scan_query_error",
    "parse_scan_response",
]
