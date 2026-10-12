"""Synchronous read-only client for the Pilot 01 transcription-review API.

One method call performs one HTTP GET. The client does not retry, does not
download source files, and does not write controller state or bundles.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NoReturn, TypeVar

import httpx
from pydantic import ValidationError

from transcription_review_agent.api_client.errors import (
    ApiInvalidJsonError,
    ApiProtocolError,
    ApiSchemaError,
    ApiTimeoutError,
    ApiTransportError,
    ScanQueryError,
    UnexpectedApiResponseError,
)
from transcription_review_agent.api_client.schemas import (
    ReviewBundleResponse,
    ScanResponse,
    bundle_path,
    parse_review_bundle,
    parse_scan_query_error,
    parse_scan_response,
)
from transcription_review_agent.config import normalize_api_base_url
from transcription_review_agent.constants import SCAN_PATH

_Parsed = TypeVar("_Parsed", ScanResponse, ReviewBundleResponse)
_Parser = Callable[[dict[str, object]], _Parsed]

# Fixed paths only. Callers cannot supply a request URL.
_BUNDLE_PATH_PREFIX = "/api/agent/v1/transcription-review/documents/"
_BUNDLE_PATH_SUFFIX = "/review-bundle"


@dataclass(frozen=True, slots=True)
class ReviewBundleNotFound:
    """HTTP 404 for a review bundle. The response body is not parsed."""

    document_id: int
    result: Literal["NOT_FOUND"] = "NOT_FOUND"


class TranscriptionReviewApiClient:
    """GET scan pages and review bundles from one configured origin.

    ``transport`` is an optional ``httpx`` transport. Tests inject one.
    Production calls leave it unset and reuse a single client session.
    """

    def __init__(
        self,
        api_base_url: str,
        request_timeout: float,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if isinstance(request_timeout, bool) or not isinstance(
            request_timeout, (int, float)
        ):
            raise ValueError("request_timeout must be a positive number")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be a positive number")
        self.api_base_url = normalize_api_base_url(api_base_url)
        self.request_timeout = float(request_timeout)
        self._client = httpx.Client(
            base_url=self.api_base_url,
            timeout=httpx.Timeout(self.request_timeout),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def __enter__(self) -> TranscriptionReviewApiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def scan(self, after_id: int | None, limit: int) -> ScanResponse:
        """Fetch one scan page.

        A local cursor of ``None`` is sent as ``after_id=0``. Any other cursor
        and ``limit`` are sent unchanged. The echoed ``after_id`` and ``limit``
        must match that request.
        """

        cursor = 0 if after_id is None else after_id
        response = self._get(SCAN_PATH, params={"after_id": cursor, "limit": limit})
        if response.status_code == 200:
            page = self._parse_json(response, SCAN_PATH, parse_scan_response)
            if page.after_id != cursor or page.limit != limit:
                self._raise_protocol_mismatch(response, SCAN_PATH)
            return page
        if response.status_code == 400:
            self._raise_scan_query_error(response)
        self._raise_unexpected(response, SCAN_PATH)

    def review_bundle(
        self, document_id: int
    ) -> ReviewBundleResponse | ReviewBundleNotFound:
        """Fetch one review bundle, or ``NOT_FOUND`` on HTTP 404.

        ``REVIEWABLE``, ``NON_ELIGIBLE``, and ``BLOCKING_INTEGRITY_ERROR`` are
        returned as validated API payloads when ``document.document_id`` is
        ``document_id``. This method does not classify those outcomes.
        """

        path = bundle_path(document_id)
        response = self._get(path)
        if response.status_code == 200:
            bundle = self._parse_json(response, path, parse_review_bundle)
            if bundle.document.document_id != document_id:
                self._raise_protocol_mismatch(response, path)
            return bundle
        if response.status_code == 404:
            return ReviewBundleNotFound(document_id)
        self._raise_unexpected(response, path)

    def _get(
        self,
        path: str,
        params: dict[str, int] | None = None,
    ) -> httpx.Response:
        _require_fixed_path(path)
        try:
            return self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise ApiTimeoutError(
                f"GET {path} timed out",
                method="GET",
                path=path,
            ) from exc
        except httpx.RequestError as exc:
            raise ApiTransportError(
                f"GET {path} failed before an HTTP response",
                method="GET",
                path=path,
            ) from exc

    def _parse_json(
        self,
        response: httpx.Response,
        path: str,
        parser: _Parser[_Parsed],
    ) -> _Parsed:
        payload = self._json_object(response, path)
        try:
            return parser(payload)
        except ValidationError:
            raise self._schema_error(response, path) from None

    def _raise_scan_query_error(self, response: httpx.Response) -> NoReturn:
        payload = self._json_object(response, SCAN_PATH)
        try:
            body = parse_scan_query_error(payload)
        except ValidationError:
            raise self._schema_error(response, SCAN_PATH) from None
        raise ScanQueryError(
            body,
            method="GET",
            path=SCAN_PATH,
            status_code=response.status_code,
        )

    def _json_object(self, response: httpx.Response, path: str) -> dict[str, object]:
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiInvalidJsonError(
                f"GET {path} returned HTTP {response.status_code} with invalid JSON",
                method="GET",
                path=path,
                status_code=response.status_code,
            ) from None
        if not isinstance(payload, dict):
            raise self._schema_error(response, path)
        return payload

    def _schema_error(self, response: httpx.Response, path: str) -> ApiSchemaError:
        return ApiSchemaError(
            f"GET {path} returned HTTP {response.status_code} with a schema-invalid body",
            method="GET",
            path=path,
            status_code=response.status_code,
        )

    def _raise_protocol_mismatch(self, response: httpx.Response, path: str) -> NoReturn:
        raise ApiProtocolError(
            f"GET {path} returned HTTP {response.status_code} that does not match the request",
            method="GET",
            path=path,
            status_code=response.status_code,
        )

    def _raise_unexpected(self, response: httpx.Response, path: str) -> NoReturn:
        raise UnexpectedApiResponseError(
            f"GET {path} returned unexpected HTTP {response.status_code}",
            method="GET",
            path=path,
            status_code=response.status_code,
        )


def _require_fixed_path(path: str) -> None:
    """Allow only the scan path and a numeric review-bundle path."""

    if path == SCAN_PATH:
        return
    document_part = path.removeprefix(_BUNDLE_PATH_PREFIX)
    if (
        document_part != path
        and document_part.endswith(_BUNDLE_PATH_SUFFIX)
        and document_part.removesuffix(_BUNDLE_PATH_SUFFIX).isdigit()
        and not document_part.removesuffix(_BUNDLE_PATH_SUFFIX).startswith("0")
    ):
        return
    raise ValueError("refusing a request path that is not a fixed Pilot 01 API path")
