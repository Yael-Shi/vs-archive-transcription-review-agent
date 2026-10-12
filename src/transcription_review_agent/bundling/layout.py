"""Filesystem layout for immutable local batches.

``outputs/batches/<batch_id>/``
  ``batch_manifest.json``
  ``documents/<document_id>/``
    ``document_manifest.json``
    ``api/review_bundle.json``
    ``transcription/transcription.txt``
    ``sources/<order_index>_<source_kind>.<ext>``

A manifest file is the publication marker. It is not integrity proof.
Call ``verify_document_bundle`` or ``verify_batch_bundle`` to recompute hashes
from the bytes on disk. Callers must create a manifest with ``O_EXCL`` and
must not replace it.
"""

from __future__ import annotations

from pathlib import Path

DOCUMENT_MANIFEST_NAME = "document_manifest.json"
BATCH_MANIFEST_NAME = "batch_manifest.json"
API_PAYLOAD_RELATIVE_PATH = "api/review_bundle.json"
TRANSCRIPTION_RELATIVE_PATH = "transcription/transcription.txt"

_MIME_SUFFIX = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/tiff": ".tif",
    "image/tif": ".tif",
}


def assert_safe_relative_path(value: str) -> str:
    """Reject absolute paths and parent-directory segments."""

    if not value or value.startswith(("/", "\\")):
        raise ValueError("bundle paths must be relative")
    parts = Path(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("bundle paths must not contain '.' or '..'")
    return value.replace("\\", "/")


def batch_dir(batches_dir: Path, batch_id: str) -> Path:
    return batches_dir / batch_id


def document_dir(batches_dir: Path, batch_id: str, document_id: int) -> Path:
    if document_id < 1:
        raise ValueError("document_id must be a positive integer")
    return batch_dir(batches_dir, batch_id) / "documents" / str(document_id)


def document_manifest_path(batches_dir: Path, batch_id: str, document_id: int) -> Path:
    return document_dir(batches_dir, batch_id, document_id) / DOCUMENT_MANIFEST_NAME


def batch_manifest_path(batches_dir: Path, batch_id: str) -> Path:
    return batch_dir(batches_dir, batch_id) / BATCH_MANIFEST_NAME


def document_manifest_relative_path(document_id: int) -> str:
    """Path of a document manifest relative to the batch directory."""

    if document_id < 1:
        raise ValueError("document_id must be a positive integer")
    return f"documents/{document_id}/{DOCUMENT_MANIFEST_NAME}"


def source_filename(order_index: int, source_kind: str, mime_type: str) -> str:
    """Stable original-bytes filename. PDFs keep a ``.pdf`` suffix."""

    if order_index < 0:
        raise ValueError("order_index must be >= 0")
    if source_kind not in {"SOURCE_FILE", "PRIMARY_FILE"}:
        raise ValueError("source_kind must be SOURCE_FILE or PRIMARY_FILE")
    suffix = _MIME_SUFFIX.get(mime_type.strip().lower(), ".bin")
    return f"{order_index:04d}_{source_kind.lower()}{suffix}"


def source_relative_path(order_index: int, source_kind: str, mime_type: str) -> str:
    return "sources/" + source_filename(order_index, source_kind, mime_type)
