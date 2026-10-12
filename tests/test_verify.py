from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.layout import source_relative_path
from transcription_review_agent.bundling.manifest import (
    BatchDocumentRef,
    BatchManifest,
    BundleFile,
    DocumentBundleManifest,
    LocalProvenance,
    SourceArtifact,
    TranscriptionArtifact,
    publish_batch_manifest,
    publish_document_manifest,
)
from transcription_review_agent.bundling.verify import verify_batch_bundle, verify_document_bundle
from transcription_review_agent.errors import BundleVerificationError

NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
TEXT = "exact stored text"
SOURCE = b"%PDF-1.7 sample"
API_PAYLOAD = b'{"outcome":"REVIEWABLE"}'


def _write_bundle(root: Path) -> tuple[Path, Path, dict[str, bytes]]:
    text_hash = sha256_text(TEXT)
    source_hash = sha256_bytes(SOURCE)
    api_hash = sha256_bytes(API_PAYLOAD)
    source_path = source_relative_path(0, "PRIMARY_FILE", "application/pdf")
    document_dir = root / "documents" / "101"
    files = {
        "api/review_bundle.json": API_PAYLOAD,
        "transcription/transcription.txt": TEXT.encode("utf-8"),
        source_path: SOURCE,
    }
    for relative, payload in files.items():
        path = document_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    manifest = DocumentBundleManifest(
        document_id=101,
        archive_item_id=501,
        language="en",
        batch_id="batch-000001",
        created_at=NOW,
        transcription=TranscriptionArtifact(
            relative_path="transcription/transcription.txt",
            sha256=text_hash,
            size_bytes=len(TEXT.encode("utf-8")),
            server_text_sha256="ab" * 32,
            text_result_id=9001,
            result_type="SOURCE_TEXT",
            status="NEEDS_REVIEW",
            verification_status="UNVERIFIED",
            engine_key="GEMINI",
            engine="engine_v1",
            source_revision=2,
        ),
        sources=[
            SourceArtifact(
                order_index=0,
                source_kind="PRIMARY_FILE",
                mime_type="application/pdf",
                relative_path=source_path,
                sha256=source_hash,
                size_bytes=len(SOURCE),
                file_s3_key="documents/101/source.pdf",
                include_in_ocr=True,
            )
        ],
        provenance=LocalProvenance(
            displayed_transcription_resolver="resolve_displayed_transcription_result",
            source_set="primary_file",
            presign_expires_in=3600,
            server_text_sha256="ab" * 32,
        ),
        files=[
            BundleFile(
                relative_path="api/review_bundle.json",
                sha256=api_hash,
                size_bytes=len(API_PAYLOAD),
                role="api_review_bundle",
            ),
            BundleFile(
                relative_path="transcription/transcription.txt",
                sha256=text_hash,
                size_bytes=len(TEXT.encode("utf-8")),
                role="transcription_text",
            ),
            BundleFile(
                relative_path=source_path,
                sha256=source_hash,
                size_bytes=len(SOURCE),
                role="source",
            ),
        ],
    )
    document_manifest = document_dir / "document_manifest.json"
    publish_document_manifest(document_manifest, manifest)
    batch_manifest = BatchManifest(
        batch_id="batch-000001",
        document_ids=[101],
        short_batch=True,
        discovery_exhausted=True,
        created_at=NOW,
        documents=[
            BatchDocumentRef(
                document_id=101,
                manifest_relative_path="documents/101/document_manifest.json",
                manifest_sha256=sha256_bytes(document_manifest.read_bytes()),
            )
        ],
    )
    batch_path = root / "batch_manifest.json"
    publish_batch_manifest(batch_path, batch_manifest)
    snapshot = _snapshot(root)
    return document_dir, root, snapshot


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_valid_sealed_bundle_verifies(tmp_path: Path) -> None:
    document_dir, batch_dir, before = _write_bundle(tmp_path)
    verified = verify_document_bundle(document_dir)
    assert verified.document_id == 101
    assert verified.transcription.sha256 != verified.transcription.server_text_sha256
    verify_batch_bundle(batch_dir)
    assert _snapshot(batch_dir) == before


def test_changed_bytes_and_missing_files_are_detected_without_repair(tmp_path: Path) -> None:
    document_dir, batch_dir, before = _write_bundle(tmp_path)
    text_path = document_dir / "transcription" / "transcription.txt"
    text_path.write_bytes(b"tampered")
    with pytest.raises(BundleVerificationError, match="transcription bytes do not match"):
        verify_document_bundle(document_dir)
    assert text_path.read_bytes() == b"tampered"

    text_path.write_bytes(before["documents/101/transcription/transcription.txt"])
    source_rel = source_relative_path(0, "PRIMARY_FILE", "application/pdf")
    source_path = document_dir / source_rel
    source_path.write_bytes(b"not-the-pdf")
    with pytest.raises(BundleVerificationError, match="source bytes do not match"):
        verify_document_bundle(document_dir)
    with pytest.raises(BundleVerificationError, match="source bytes do not match"):
        verify_batch_bundle(batch_dir)
    assert source_path.read_bytes() == b"not-the-pdf"

    source_path.write_bytes(before[f"documents/101/{source_rel}"])
    source_path.unlink()
    with pytest.raises(BundleVerificationError, match="missing source"):
        verify_document_bundle(document_dir)
    assert not source_path.exists()

    source_path.write_bytes(before[f"documents/101/{source_rel}"])
    api_path = document_dir / "api" / "review_bundle.json"
    api_path.write_bytes(b'{"outcome":"CHANGED"}')
    with pytest.raises(BundleVerificationError, match="api payload bytes do not match"):
        verify_document_bundle(document_dir)
    assert api_path.read_bytes() == b'{"outcome":"CHANGED"}'


def test_batch_verification_detects_manifest_mismatch_and_absence(tmp_path: Path) -> None:
    _document_dir, batch_dir, _before = _write_bundle(tmp_path)
    document_manifest = batch_dir / "documents" / "101" / "document_manifest.json"
    original = document_manifest.read_bytes()
    document_manifest.write_bytes(original.replace(b"101", b"102", 1))
    with pytest.raises(BundleVerificationError, match="document manifest bytes do not match"):
        verify_batch_bundle(batch_dir)
    assert document_manifest.read_bytes() != original

    document_manifest.write_bytes(original)
    document_manifest.unlink()
    with pytest.raises(BundleVerificationError, match="missing document manifest"):
        verify_batch_bundle(batch_dir)
    assert not document_manifest.exists()
