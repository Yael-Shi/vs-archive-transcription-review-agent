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
from transcription_review_agent.errors import ImmutableBundleError


NOW = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)


def _manifest() -> DocumentBundleManifest:
    text = "exact stored text"
    text_hash = sha256_text(text)
    source = b"%PDF-1.7 sample"
    source_hash = sha256_bytes(source)
    source_path = source_relative_path(0, "PRIMARY_FILE", "application/pdf")
    return DocumentBundleManifest(
        document_id=101,
        archive_item_id=501,
        language="fr",
        batch_id="batch-000001",
        created_at=NOW,
        transcription=TranscriptionArtifact(
            relative_path="transcription/transcription.txt",
            sha256=text_hash,
            size_bytes=len(text.encode("utf-8")),
            server_text_sha256="ab" * 32,
            text_result_id=9001,
            result_type="SOURCE_TEXT",
            status="NEEDS_REVIEW",
            verification_status="UNVERIFIED",
            engine_key="GEMINI",
            engine="engine_v1",
            source_revision=2,
            based_on_source_revision=None,
            api_updated_at=NOW,
        ),
        sources=[
            SourceArtifact(
                order_index=0,
                source_file_id=None,
                source_kind="PRIMARY_FILE",
                mime_type="application/pdf",
                relative_path=source_path,
                sha256=source_hash,
                size_bytes=len(source),
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
                sha256=sha256_bytes(b"{}"),
                size_bytes=2,
                role="api_review_bundle",
            ),
            BundleFile(
                relative_path="transcription/transcription.txt",
                sha256=text_hash,
                size_bytes=len(text.encode("utf-8")),
                role="transcription_text",
            ),
            BundleFile(
                relative_path=source_path,
                sha256=source_hash,
                size_bytes=len(source),
                role="source",
            ),
        ],
    )


def test_local_hash_is_exact_utf8_without_an_added_newline() -> None:
    assert sha256_text("café") == sha256_bytes("café".encode("utf-8"))
    assert sha256_text("a\n") != sha256_text("a")


def test_manifest_rejects_verified_text_and_hash_mismatch() -> None:
    with pytest.raises(ValueError):
        broken = _manifest()
        broken.transcription.verification_status = "VERIFIED"
        DocumentBundleManifest.model_validate(broken.model_dump())

    payload = _manifest().model_dump()
    payload["files"][1]["sha256"] = "cd" * 32
    with pytest.raises(ValueError):
        DocumentBundleManifest.model_validate(payload)
    payload = _manifest().model_dump()
    payload["sources"] = []
    with pytest.raises(ValueError):
        DocumentBundleManifest.model_validate(payload)


def test_publish_refuses_to_overwrite_a_sealed_manifest(tmp_path: Path) -> None:
    document_path = tmp_path / "documents" / "101" / "document_manifest.json"
    publish_document_manifest(document_path, _manifest())
    with pytest.raises(ImmutableBundleError):
        publish_document_manifest(document_path, _manifest())

    batch = BatchManifest(
        batch_id="batch-000001",
        document_ids=[101],
        short_batch=True,
        discovery_exhausted=True,
        created_at=NOW,
        documents=[
            BatchDocumentRef(
                document_id=101,
                manifest_relative_path="documents/101/document_manifest.json",
                manifest_sha256=sha256_bytes(document_path.read_bytes()),
            )
        ],
    )
    batch_path = tmp_path / "batch_manifest.json"
    publish_batch_manifest(batch_path, batch)
    with pytest.raises(ImmutableBundleError):
        publish_batch_manifest(batch_path, batch)


def test_full_batch_manifest_requires_five_documents() -> None:
    with pytest.raises(ValueError):
        BatchManifest(
            batch_id="batch-000001",
            document_ids=[1],
            short_batch=False,
            discovery_exhausted=False,
            created_at=NOW,
            documents=[
                BatchDocumentRef(
                    document_id=1,
                    manifest_relative_path="documents/1/document_manifest.json",
                    manifest_sha256="ab" * 32,
                )
            ],
        )


def test_empty_batch_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "batch_manifest.json"
    with pytest.raises(ValueError):
        publish_batch_manifest(
            path,
            BatchManifest(
                batch_id="batch-000001",
                document_ids=[],
                short_batch=True,
                discovery_exhausted=True,
                created_at=NOW,
                documents=[],
            ),
        )
    assert not path.exists()
