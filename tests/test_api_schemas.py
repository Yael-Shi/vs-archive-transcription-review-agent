from __future__ import annotations

import pytest

from transcription_review_agent.api_client.schemas import (
    bundle_path,
    parse_review_bundle,
    parse_scan_query_error,
    parse_scan_response,
)
from transcription_review_agent.bundling.hashing import sha256_text
from transcription_review_agent.constants import BUNDLE_PATH_TEMPLATE, SCAN_PATH


def _document(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "document_id": 101,
        "archive_item_id": 501,
        "language": "en",
        "doc_type": "IMAGE",
        "item_type": "OCR_DOCUMENT",
        "upload_status": "UPLOADED",
        "processing_state_user": "READY",
        "visibility": "public",
        "expected_source_file_count": 1,
        "updated_at": "2026-10-11T09:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _transcription(**overrides: object) -> dict[str, object]:
    text = "exact stored text"
    payload: dict[str, object] = {
        "text_result_id": 9001,
        "result_type": "SOURCE_TEXT",
        "status": "NEEDS_REVIEW",
        "verification_status": "UNVERIFIED",
        "engine_key": "GEMINI",
        "engine": "engine_v1",
        "source_revision": 2,
        "based_on_source_revision": None,
        "updated_at": "2026-10-11T08:55:00+00:00",
        "text": text,
        "text_sha256": sha256_text(text),
    }
    payload.update(overrides)
    return payload


def _page(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_file_id": 11,
        "order_index": 0,
        "display_number": 1,
        "source_kind": "PRIMARY_FILE",
        "file_s3_key": "documents/101/source.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 120,
        "include_in_ocr": True,
        "upload_status": "UPLOADED",
        "download_url": "https://example.test/presigned",
        "download_url_expires_in": 3600,
    }
    payload.update(overrides)
    return payload


def _reviewable(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "transcription-review-bundle-v1",
        "outcome": "REVIEWABLE",
        "document": _document(doc_type="PDF", expected_source_file_count=None),
        "transcription": _transcription(),
        "source_pages": [_page()],
        "provenance": {
            "displayed_transcription_resolver": "resolve_displayed_transcription_result",
            "source_set": "primary_file",
            "presign_expires_in": 3600,
        },
        "unexpected_server_field": "ignored",
    }
    payload.update(overrides)
    return payload


def test_scan_path_and_bundle_path_match_the_pilot_contract() -> None:
    assert SCAN_PATH == "/api/agent/v1/transcription-review/documents/scan"
    assert bundle_path(101) == BUNDLE_PATH_TEMPLATE.format(document_id=101)


def test_scan_page_requires_ascending_ids_and_exhaustion_consistency() -> None:
    parsed = parse_scan_response(
        {
            "schema_version": "transcription-review-scan-v1",
            "after_id": 0,
            "limit": 50,
            "next_after_id": 120,
            "exhausted": False,
            "items": [
                {
                    "document_id": 101,
                    "archive_item_id": 501,
                    "language": "en",
                    "doc_type": "IMAGE",
                    "item_type": "OCR_DOCUMENT",
                    "upload_status": "UPLOADED",
                    "processing_state_user": "READY",
                    "visibility": "public",
                    "expected_source_file_count": 3,
                },
                {
                    "document_id": 120,
                    "archive_item_id": 520,
                    "language": "fr",
                    "doc_type": "PDF",
                    "item_type": "OCR_DOCUMENT",
                    "upload_status": "UPLOADED",
                    "processing_state_user": "PROCESSING",
                    "visibility": "public",
                    "expected_source_file_count": None,
                },
            ],
        }
    )
    assert parsed.next_after_id == 120

    with pytest.raises(ValueError):
        parse_scan_response(
            {
                "schema_version": "transcription-review-scan-v1",
                "after_id": 0,
                "limit": 50,
                "next_after_id": None,
                "exhausted": False,
                "items": [],
            }
        )


def test_scan_query_error_codes() -> None:
    parsed = parse_scan_query_error(
        {"error": "invalid_after_id", "detail": "after_id must be a non-negative integer"}
    )
    assert parsed.error == "invalid_after_id"
    with pytest.raises(ValueError):
        parse_scan_query_error({"error": "nope", "detail": "x"})


def test_reviewable_requires_text_and_sources() -> None:
    parsed = parse_review_bundle(_reviewable())
    assert parsed.outcome.value == "REVIEWABLE"
    assert parsed.source_pages[0].mime_type == "application/pdf"
    assert parsed.transcription is not None
    assert parsed.transcription.text == "exact stored text"

    missing_sources = _reviewable()
    missing_sources["source_pages"] = []
    with pytest.raises(ValueError):
        parse_review_bundle(missing_sources)


def test_non_eligible_and_blocking_do_not_carry_sources() -> None:
    non_eligible = parse_review_bundle(
        {
            "schema_version": "transcription-review-bundle-v1",
            "outcome": "NON_ELIGIBLE",
            "document": _document(),
            "error": {
                "code": "VERIFIED_TRANSCRIPTION_OUT_OF_SCOPE",
                "message": "Displayed transcription is VERIFIED and out of Pilot 01 scope",
            },
            "transcription": _transcription(verification_status="VERIFIED"),
            "source_pages": [],
        }
    )
    assert non_eligible.provenance is None

    blocking = parse_review_bundle(
        {
            "schema_version": "transcription-review-bundle-v1",
            "outcome": "BLOCKING_INTEGRITY_ERROR",
            "document": _document(),
            "error": {
                "code": "UPLOADED_DISPLAY_ONLY_SOURCE_MISSING_KEY",
                "message": "Uploaded display-only source is missing its key",
                "details": {"order_index": 2},
            },
            "transcription": _transcription(),
            "source_pages": [],
        }
    )
    assert blocking.error is not None
    assert blocking.source_pages == []

    with pytest.raises(ValueError):
        parse_review_bundle(
            {
                "schema_version": "transcription-review-bundle-v1",
                "outcome": "BLOCKING_INTEGRITY_ERROR",
                "document": _document(),
                "error": {"code": "NOT_A_REAL_CODE", "message": "no"},
                "transcription": None,
                "source_pages": [],
            }
        )
