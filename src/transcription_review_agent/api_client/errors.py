"""Failures from the read-only Pilot 01 HTTP client.

Messages name the method, path, and HTTP status when there is one. They do
not include response bodies, secrets, or presigned URLs.
"""

from __future__ import annotations

from transcription_review_agent.api_client.schemas import ScanQueryErrorBody
from transcription_review_agent.errors import TranscriptionReviewAgentError


class ApiClientError(TranscriptionReviewAgentError):
    """A scan or review-bundle call failed explicitly."""

    def __init__(
        self,
        message: str,
        *,
        method: str,
        path: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.method = method
        self.path = path
        self.status_code = status_code


class ApiTimeoutError(ApiClientError):
    """The request exceeded the configured timeout. It was not retried."""


class ApiTransportError(ApiClientError):
    """The request failed before an HTTP response (connection or network)."""


class ApiInvalidJsonError(ApiClientError):
    """The response body was not valid JSON."""


class ApiSchemaError(ApiClientError):
    """JSON was readable but does not match the Pilot 01 schema."""


class ApiProtocolError(ApiClientError):
    """A schema-valid HTTP 200 does not match the request that produced it."""


class ScanQueryError(ApiClientError):
    """HTTP 400 whose body is a valid ``invalid_after_id`` or ``invalid_limit`` error."""

    def __init__(
        self,
        body: ScanQueryErrorBody,
        *,
        method: str,
        path: str,
        status_code: int,
    ) -> None:
        self.body = body
        super().__init__(
            f"{method} {path} returned HTTP {status_code} {body.error}",
            method=method,
            path=path,
            status_code=status_code,
        )


class UnexpectedApiResponseError(ApiClientError):
    """The HTTP status is not one this client interprets.

    This is not an empty scan page, an exhausted scan, or a missing bundle.
    """
