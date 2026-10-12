from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from transcription_review_agent.api_client import (
    ApiInvalidJsonError,
    ApiProtocolError,
    ApiSchemaError,
    ApiTimeoutError,
    ApiTransportError,
    ReviewBundleNotFound,
    ReviewBundleResponse,
    ScanQueryError,
    ScanResponse,
    TranscriptionReviewApiClient,
    UnexpectedApiResponseError,
    bundle_path,
)
from transcription_review_agent.bundling.hashing import sha256_text
from transcription_review_agent.constants import SCAN_PATH

BASE_URL = "http://127.0.0.1:8000"
PRESIGNED = "https://files.example.test/object?X-Amz-Signature=supersecretvalue"
Handler = Callable[[httpx.Request], httpx.Response]


def _client(
    handler: Handler,
) -> tuple[TranscriptionReviewApiClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = TranscriptionReviewApiClient(
        BASE_URL,
        5,
        transport=httpx.MockTransport(wrapped),
    )
    return client, seen


def _scan_page(
    *,
    after_id: int = 0,
    limit: int = 50,
    document_id: int = 15,
) -> dict[str, object]:
    return {
        "schema_version": "transcription-review-scan-v1",
        "after_id": after_id,
        "limit": limit,
        "next_after_id": document_id,
        "exhausted": False,
        "items": [
            {
                "document_id": document_id,
                "archive_item_id": 501,
                "language": "en",
                "doc_type": "IMAGE",
                "item_type": "OCR_DOCUMENT",
                "upload_status": "UPLOADED",
                "processing_state_user": "READY",
                "visibility": "public",
                "expected_source_file_count": 1,
            }
        ],
    }


def _document(document_id: int = 101) -> dict[str, object]:
    return {
        "document_id": document_id,
        "archive_item_id": 501,
        "language": "en",
        "doc_type": "IMAGE",
        "item_type": "OCR_DOCUMENT",
        "upload_status": "UPLOADED",
        "processing_state_user": "READY",
        "visibility": "public",
        "expected_source_file_count": 1,
    }


def _transcription(**overrides: object) -> dict[str, object]:
    text = "exact stored text"
    payload: dict[str, object] = {
        "text_result_id": 9001,
        "result_type": "SOURCE_TEXT",
        "status": "NEEDS_REVIEW",
        "verification_status": "UNVERIFIED",
        "text": text,
        "text_sha256": sha256_text(text),
    }
    payload.update(overrides)
    return payload


def _reviewable(
    download_url: str = "https://example.test/presigned",
    document_id: int = 101,
) -> dict[str, object]:
    return {
        "schema_version": "transcription-review-bundle-v1",
        "outcome": "REVIEWABLE",
        "document": _document(document_id),
        "transcription": _transcription(),
        "source_pages": [
            {
                "source_file_id": 11,
                "order_index": 0,
                "display_number": 1,
                "source_kind": "PRIMARY_FILE",
                "file_s3_key": "documents/101/source.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 120,
                "include_in_ocr": True,
                "upload_status": "UPLOADED",
                "download_url": download_url,
                "download_url_expires_in": 3600,
            }
        ],
        "provenance": {
            "displayed_transcription_resolver": "resolve_displayed_transcription_result",
            "source_set": "primary_file",
            "presign_expires_in": 3600,
        },
    }


def _classified(
    outcome: str,
    code: str,
    document_id: int = 101,
    message: str = "classified by the server",
) -> dict[str, object]:
    return {
        "schema_version": "transcription-review-bundle-v1",
        "outcome": outcome,
        "document": _document(document_id),
        "transcription": _transcription(),
        "source_pages": [],
        "error": {"code": code, "message": message},
    }


def _assert_one_get(seen: list[httpx.Request], path: str) -> httpx.Request:
    assert len(seen) == 1
    request = seen[0]
    assert request.method == "GET"
    assert request.url.path == path
    return request


def _assert_context(exc: object, *, path: str, status: int | None) -> None:
    assert isinstance(
        exc,
        (
            ApiInvalidJsonError,
            ApiProtocolError,
            ApiSchemaError,
            ApiTimeoutError,
            ApiTransportError,
            ScanQueryError,
            UnexpectedApiResponseError,
        ),
    )
    assert exc.method == "GET"
    assert exc.path == path
    assert exc.status_code == status
    assert PRESIGNED not in str(exc)
    assert "X-Amz-Signature" not in str(exc)


def test_client_accepts_only_an_origin_and_a_positive_timeout() -> None:
    with pytest.raises(ValueError):
        TranscriptionReviewApiClient("http://127.0.0.1:8000/api", 5)
    with pytest.raises(ValueError):
        TranscriptionReviewApiClient("http://user:secret@127.0.0.1:8000", 5)
    with pytest.raises(ValueError):
        TranscriptionReviewApiClient(BASE_URL, 0)

    client = TranscriptionReviewApiClient("http://127.0.0.1:8000/", 5)
    assert client.api_base_url == BASE_URL
    client.close()


def test_scan_none_cursor_sends_after_id_zero_and_exact_limit() -> None:
    client, seen = _client(
        lambda request: httpx.Response(200, json=_scan_page(limit=17))
    )

    with client:
        page = client.scan(None, 17)

    request = _assert_one_get(seen, SCAN_PATH)
    assert dict(request.url.params) == {"after_id": "0", "limit": "17"}
    assert isinstance(page, ScanResponse)
    assert page.items[0].document_id == 15


def test_scan_sends_explicit_cursor_and_limit_unchanged() -> None:
    client, seen = _client(
        lambda request: httpx.Response(
            200,
            json=_scan_page(after_id=41, limit=17, document_id=42),
        )
    )

    with client:
        page = client.scan(41, 17)

    request = _assert_one_get(seen, SCAN_PATH)
    assert dict(request.url.params) == {"after_id": "41", "limit": "17"}
    assert page.after_id == 41
    assert page.limit == 17
    assert page.items[0].document_id == 42


def test_scan_rejects_schema_valid_response_for_another_request() -> None:
    mismatched_cursor = _scan_page(after_id=40, limit=17, document_id=42)
    mismatched_cursor["download_url"] = PRESIGNED
    mismatched_limit = _scan_page(after_id=41, limit=50, document_id=42)
    mismatched_limit["download_url"] = PRESIGNED
    cases = (
        (mismatched_cursor, "40"),
        (mismatched_limit, "50"),
    )
    for body, leaked_value in cases:
        client, seen = _client(
            lambda request, payload=body: httpx.Response(200, json=payload)
        )
        with client, pytest.raises(ApiProtocolError) as exc_info:
            client.scan(41, 17)
        _assert_one_get(seen, SCAN_PATH)
        _assert_context(exc_info.value, path=SCAN_PATH, status=200)
        assert exc_info.value.__cause__ is None
        assert leaked_value not in str(exc_info.value)
        assert "download_url" not in str(exc_info.value)


def test_review_bundle_rejects_schema_valid_response_for_another_document() -> None:
    reviewable = _reviewable(download_url=PRESIGNED, document_id=102)
    non_eligible = _classified(
        "NON_ELIGIBLE",
        "LANGUAGE_NOT_IN_SCOPE",
        document_id=102,
        message=PRESIGNED,
    )
    for body in (reviewable, non_eligible):
        client, seen = _client(
            lambda request, payload=body: httpx.Response(200, json=payload)
        )
        with client, pytest.raises(ApiProtocolError) as exc_info:
            client.review_bundle(101)
        _assert_one_get(seen, bundle_path(101))
        _assert_context(exc_info.value, path=bundle_path(101), status=200)
        assert exc_info.value.__cause__ is None
        assert "102" not in str(exc_info.value)


def test_scan_200_invalid_json_fails() -> None:
    client, seen = _client(lambda request: httpx.Response(200, content=b"{"))

    with client, pytest.raises(ApiInvalidJsonError) as exc_info:
        client.scan(None, 50)

    _assert_one_get(seen, SCAN_PATH)
    _assert_context(exc_info.value, path=SCAN_PATH, status=200)
    assert exc_info.value.__cause__ is None


def test_scan_200_schema_invalid_body_fails_closed() -> None:
    body = {
        "schema_version": "transcription-review-scan-v1",
        "after_id": 0,
        "limit": 50,
        "next_after_id": None,
        "exhausted": False,
        "items": [],
    }
    client, seen = _client(lambda request: httpx.Response(200, json=body))

    with client, pytest.raises(ApiSchemaError) as exc_info:
        client.scan(None, 50)

    _assert_one_get(seen, SCAN_PATH)
    _assert_context(exc_info.value, path=SCAN_PATH, status=200)
    assert exc_info.value.__cause__ is None
    assert "exhaust" not in str(exc_info.value).lower()


def test_scan_400_structured_query_error() -> None:
    detail = f"after_id rejected; ignored reference {PRESIGNED}"
    client, seen = _client(
        lambda request: httpx.Response(
            400,
            json={"error": "invalid_after_id", "detail": detail},
        )
    )

    with client, pytest.raises(ScanQueryError) as exc_info:
        client.scan(-1, 50)

    _assert_one_get(seen, SCAN_PATH)
    _assert_context(exc_info.value, path=SCAN_PATH, status=400)
    assert exc_info.value.body.error == "invalid_after_id"
    assert exc_info.value.body.detail == detail


def test_scan_malformed_400_fails_closed() -> None:
    invalid_json, invalid_seen = _client(
        lambda request: httpx.Response(400, content=b"not-json")
    )
    with invalid_json, pytest.raises(ApiInvalidJsonError) as json_exc:
        invalid_json.scan(None, 50)
    _assert_one_get(invalid_seen, SCAN_PATH)
    _assert_context(json_exc.value, path=SCAN_PATH, status=400)
    assert json_exc.value.__cause__ is None

    invalid_schema, schema_seen = _client(
        lambda request: httpx.Response(400, json={"error": "nope", "detail": "x"})
    )
    with invalid_schema, pytest.raises(ApiSchemaError) as schema_exc:
        invalid_schema.scan(None, 50)
    _assert_one_get(schema_seen, SCAN_PATH)
    _assert_context(schema_exc.value, path=SCAN_PATH, status=400)
    assert not isinstance(schema_exc.value, ScanQueryError)


def test_scan_unexpected_status_is_not_an_empty_page() -> None:
    client, seen = _client(
        lambda request: httpx.Response(503, content=PRESIGNED.encode())
    )

    with client, pytest.raises(UnexpectedApiResponseError) as exc_info:
        client.scan(None, 50)

    _assert_one_get(seen, SCAN_PATH)
    _assert_context(exc_info.value, path=SCAN_PATH, status=503)
    assert "exhaust" not in str(exc_info.value).lower()


def test_scan_404_is_unexpected() -> None:
    client, seen = _client(lambda request: httpx.Response(404, json={"items": []}))

    with client, pytest.raises(UnexpectedApiResponseError) as exc_info:
        client.scan(None, 50)

    _assert_one_get(seen, SCAN_PATH)
    _assert_context(exc_info.value, path=SCAN_PATH, status=404)


def test_scan_redirect_is_not_followed() -> None:
    client, seen = _client(
        lambda request: httpx.Response(
            302,
            headers={"Location": PRESIGNED},
        )
    )

    with client, pytest.raises(UnexpectedApiResponseError) as exc_info:
        client.scan(None, 50)

    _assert_one_get(seen, SCAN_PATH)
    _assert_context(exc_info.value, path=SCAN_PATH, status=302)


def test_scan_timeout_and_network_fail_explicitly() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    timed, timed_seen = _client(timeout)
    with timed, pytest.raises(ApiTimeoutError) as timeout_exc:
        timed.scan(None, 50)
    _assert_one_get(timed_seen, SCAN_PATH)
    _assert_context(timeout_exc.value, path=SCAN_PATH, status=None)

    disconnected, disconnected_seen = _client(offline)
    with disconnected, pytest.raises(ApiTransportError) as transport_exc:
        disconnected.scan(4, 50)
    _assert_one_get(disconnected_seen, SCAN_PATH)
    _assert_context(transport_exc.value, path=SCAN_PATH, status=None)


def test_review_bundle_200_outcomes_use_the_document_path() -> None:
    cases = (
        (_reviewable(), "REVIEWABLE"),
        (_classified("NON_ELIGIBLE", "LANGUAGE_NOT_IN_SCOPE"), "NON_ELIGIBLE"),
        (
            _classified("BLOCKING_INTEGRITY_ERROR", "MISSING_SOURCE_FILE"),
            "BLOCKING_INTEGRITY_ERROR",
        ),
    )
    for body, outcome in cases:
        client, seen = _client(
            lambda request, payload=body: httpx.Response(200, json=payload)
        )
        with client:
            bundle = client.review_bundle(101)
        _assert_one_get(seen, bundle_path(101))
        assert isinstance(bundle, ReviewBundleResponse)
        assert not isinstance(bundle, ReviewBundleNotFound)
        assert bundle.outcome.value == outcome
        assert bundle.document.document_id == 101


def test_review_bundle_404_is_not_found_without_parsing_the_body() -> None:
    client, seen = _client(
        lambda request: httpx.Response(404, json=_reviewable(download_url=PRESIGNED))
    )

    with client:
        result = client.review_bundle(101)

    _assert_one_get(seen, bundle_path(101))
    assert isinstance(result, ReviewBundleNotFound)
    assert result.result == "NOT_FOUND"
    assert result.document_id == 101
    assert PRESIGNED not in str(result)
    assert PRESIGNED not in repr(result)

    garbage, garbage_seen = _client(
        lambda request: httpx.Response(404, content=PRESIGNED.encode())
    )
    with garbage:
        garbage_result = garbage.review_bundle(101)
    _assert_one_get(garbage_seen, bundle_path(101))
    assert isinstance(garbage_result, ReviewBundleNotFound)
    assert PRESIGNED not in str(garbage_result)


def test_review_bundle_200_invalid_json_fails() -> None:
    client, seen = _client(lambda request: httpx.Response(200, content=b"{"))

    with client, pytest.raises(ApiInvalidJsonError) as exc_info:
        client.review_bundle(101)

    _assert_one_get(seen, bundle_path(101))
    _assert_context(exc_info.value, path=bundle_path(101), status=200)
    assert exc_info.value.__cause__ is None


def test_review_bundle_200_schema_invalid_does_not_leak_presigned_url() -> None:
    body = _reviewable(download_url=PRESIGNED)
    del body["provenance"]
    client, seen = _client(lambda request: httpx.Response(200, json=body))

    with client, pytest.raises(ApiSchemaError) as exc_info:
        client.review_bundle(101)

    _assert_one_get(seen, bundle_path(101))
    _assert_context(exc_info.value, path=bundle_path(101), status=200)
    assert exc_info.value.__cause__ is None


def test_review_bundle_unexpected_status_fails() -> None:
    client, seen = _client(
        lambda request: httpx.Response(500, content=PRESIGNED.encode())
    )

    with client, pytest.raises(UnexpectedApiResponseError) as exc_info:
        client.review_bundle(101)

    _assert_one_get(seen, bundle_path(101))
    _assert_context(exc_info.value, path=bundle_path(101), status=500)


def test_review_bundle_timeout_and_network_fail_explicitly() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    timed, timed_seen = _client(timeout)
    with timed, pytest.raises(ApiTimeoutError) as timeout_exc:
        timed.review_bundle(101)
    _assert_one_get(timed_seen, bundle_path(101))
    _assert_context(timeout_exc.value, path=bundle_path(101), status=None)

    disconnected, disconnected_seen = _client(offline)
    with disconnected, pytest.raises(ApiTransportError) as transport_exc:
        disconnected.review_bundle(101)
    _assert_one_get(disconnected_seen, bundle_path(101))
    _assert_context(transport_exc.value, path=bundle_path(101), status=None)


def test_client_does_not_write_state_or_bundle_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client, seen = _client(
        lambda request: httpx.Response(
            200,
            json=_scan_page() if request.url.path == SCAN_PATH else _reviewable(),
        )
    )

    with client:
        client.scan(None, 50)
        client.review_bundle(101)

    assert len(seen) == 2
    assert list(tmp_path.iterdir()) == []
