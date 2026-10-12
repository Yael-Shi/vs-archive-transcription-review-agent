"""Read-only verification of sealed local bundles.

Verification reads bytes and compares them with the manifest. It does not
create, rewrite, or repair files. A manifest file is not treated as proof
until these hashes match.
"""

from __future__ import annotations

from pathlib import Path

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.layout import (
    BATCH_MANIFEST_NAME,
    DOCUMENT_MANIFEST_NAME,
    assert_safe_relative_path,
)
from transcription_review_agent.bundling.manifest import (
    BatchManifest,
    BundleFile,
    DocumentBundleManifest,
)
from transcription_review_agent.errors import BundleVerificationError

_ARTIFACT_DIRS = ("api", "transcription", "sources")
_MISMATCH = {
    "transcription_text": "transcription bytes do not match",
    "api_review_bundle": "api payload bytes do not match",
    "source": "source bytes do not match",
}


def verify_document_bundle(bundle_dir: Path) -> DocumentBundleManifest:
    """Validate one sealed document directory. Return the parsed manifest."""

    manifest_path = bundle_dir / DOCUMENT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleVerificationError(f"missing document manifest {manifest_path}")
    manifest = _parse_document_manifest(manifest_path)
    expected = {entry.relative_path: entry for entry in manifest.files}
    _reject_unexpected_artifact_files(bundle_dir, set(expected))
    for relative_path, entry in expected.items():
        _verify_listed_file(bundle_dir, entry)
    return manifest


def verify_batch_bundle(batch_dir: Path) -> BatchManifest:
    """Validate a sealed batch directory and every document bundle inside it."""

    manifest_path = batch_dir / BATCH_MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleVerificationError(f"missing batch manifest {manifest_path}")
    try:
        manifest = BatchManifest.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise BundleVerificationError(
            f"malformed batch manifest {manifest_path}"
        ) from exc

    for ref in manifest.documents:
        document_manifest = _contained_path(batch_dir, ref.manifest_relative_path)
        if not document_manifest.is_file():
            raise BundleVerificationError(
                f"missing document manifest {ref.manifest_relative_path}"
            )
        data = document_manifest.read_bytes()
        if sha256_bytes(data) != ref.manifest_sha256:
            raise BundleVerificationError(
                f"document manifest bytes do not match {ref.manifest_relative_path}"
            )
        verify_document_bundle(document_manifest.parent)
    return manifest


def _parse_document_manifest(path: Path) -> DocumentBundleManifest:
    try:
        return DocumentBundleManifest.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise BundleVerificationError(f"malformed document manifest {path}") from exc


def _verify_listed_file(bundle_dir: Path, entry: BundleFile) -> None:
    path = _contained_path(bundle_dir, entry.relative_path)
    if not path.is_file():
        if entry.role == "source":
            raise BundleVerificationError(f"missing source {entry.relative_path}")
        raise BundleVerificationError(f"missing artifact {entry.relative_path}")
    data = path.read_bytes()
    if sha256_bytes(data) != entry.sha256 or len(data) != entry.size_bytes:
        raise BundleVerificationError(
            f"{_MISMATCH[entry.role]}: {entry.relative_path}"
        )


def _reject_unexpected_artifact_files(bundle_dir: Path, expected: set[str]) -> None:
    for folder in _ARTIFACT_DIRS:
        root = bundle_dir / folder
        if not root.exists():
            continue
        if not root.is_dir():
            raise BundleVerificationError(f"{folder} is not a directory")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(bundle_dir).as_posix()
            if relative not in expected:
                raise BundleVerificationError(f"unexpected artifact file {relative}")


def _contained_path(root: Path, relative_path: str) -> Path:
    try:
        safe = assert_safe_relative_path(relative_path)
    except ValueError as exc:
        raise BundleVerificationError(str(exc)) from exc
    path = root.joinpath(*Path(safe).parts)
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise BundleVerificationError(f"artifact path escapes the bundle: {relative_path}")
    return path
