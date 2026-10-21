"""Acquire sources for an already selected batch and publish document bundles.

Discovery does not store review-bundle payloads or presigned URLs. This step
re-fetches each selected document, downloads the source bytes named by that
fresh response, and publishes one immutable document directory. It does not
scan, seal the batch, or call an AI provider.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx

from transcription_review_agent.api_client.client import (
    ReviewBundleNotFound,
    TranscriptionReviewApiClient,
)
from transcription_review_agent.api_client.schemas import (
    BundleOutcome,
    ReviewBundleResponse,
    SourcePagePayload,
)
from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.layout import (
    API_PAYLOAD_RELATIVE_PATH,
    DOCUMENT_MANIFEST_NAME,
    TRANSCRIPTION_RELATIVE_PATH,
    batch_dir,
    document_dir,
    document_manifest_relative_path,
    source_relative_path,
)
from transcription_review_agent.bundling.manifest import (
    BundleFile,
    DocumentBundleManifest,
    LocalProvenance,
    SourceArtifact,
    TranscriptionArtifact,
    publish_document_manifest,
)
from transcription_review_agent.bundling.verify import verify_document_bundle
from transcription_review_agent.config import AgentConfig
from transcription_review_agent.errors import (
    BundleConflictError,
    FreshBundleRejectedError,
    SourceDownloadError,
)
from transcription_review_agent.state.schema import (
    BatchStatus,
    ControllerState,
    DocumentLifecycle,
    DocumentPhase,
    SourceHashRecord,
)
from transcription_review_agent.state.store import StateStore
from transcription_review_agent.state.transitions import (
    begin_bundling,
    mark_bundled,
    mark_bundling,
)

Clock = Callable[[], datetime]
_STAGING_DIRNAME = ".staging"
_URL_KEYS = frozenset({"download_url", "download_url_expires_in"})
_DOWNLOAD_CREDENTIAL_MARKERS = (
    b"download_url",
    b"download_url_expires_in",
    b"X-Amz-",
)


@dataclass(frozen=True, slots=True)
class BundleResult:
    """Outcome of one bundling attempt. No presigned URL and no API payload."""

    completed: bool
    batch_status: BatchStatus
    batch_id: str | None
    document_ids: tuple[int, ...]


def bundle_selected_batch(
    config: AgentConfig,
    api_client: TranscriptionReviewApiClient,
    state_store: StateStore,
    *,
    source_client: httpx.Client | None = None,
    clock: Clock | None = None,
) -> BundleResult:
    """Publish verified document bundles for the open selected batch.

    ``source_client`` is an optional HTTP client for presigned source GETs.
    Production builds one with redirects disabled, ``trust_env=False``, and the
    configured timeout. Download URLs are taken only from the fresh review
    bundle; this function does not accept a URL argument.
    """

    now = _utc_now if clock is None else clock
    state = state_store.load()
    if not _can_bundle(state):
        return _result(state, completed=False)

    owned_client = source_client is None
    client = (
        source_client
        if source_client is not None
        else _open_source_client(config.request_timeout_seconds)
    )
    try:
        if state.batch_status is BatchStatus.ACCUMULATING:
            state = state_store.commit(begin_bundling(state, occurred_at=now()))
        for document_id in tuple(state.current_batch_document_ids):
            state = _bundle_document(
                state,
                document_id=document_id,
                config=config,
                api_client=api_client,
                state_store=state_store,
                source_client=client,
                now=now,
            )
        return _result(state, completed=True)
    finally:
        if owned_client:
            client.close()


def _open_source_client(timeout: float) -> httpx.Client:
    """HTTP client for presigned GETs. No retries, redirects, or env credentials."""

    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,
        trust_env=False,
    )


def _bundle_document(
    state: ControllerState,
    *,
    document_id: int,
    config: AgentConfig,
    api_client: TranscriptionReviewApiClient,
    state_store: StateStore,
    source_client: httpx.Client,
    now: Clock,
) -> ControllerState:
    document = _require_document(state, document_id)
    batch_id = _require_batch_id(state)
    if document.phase is DocumentPhase.BUNDLED:
        _require_checkpoint_bundle(config, state, document)
        return state

    staging = _staging_document_dir(config.batches_dir, batch_id, document_id)
    existing = _matching_final_bundle(config, document, batch_id=batch_id)
    if existing is not None:
        if staging.exists():
            _discard_staging(staging)
    else:
        existing = _promote_complete_staging(
            config,
            document,
            batch_id=batch_id,
        )
    if document.phase is DocumentPhase.SELECTED:
        state = state_store.commit(
            mark_bundling(state, document_id=document_id, occurred_at=now())
        )
        document = _require_document(state, document_id)
    if existing is not None:
        return _commit_bundled(
            state,
            document_id=document_id,
            manifest=existing,
            state_store=state_store,
            now=now,
        )

    bundle = _fetch_reviewable(api_client, document)
    manifest = _publish_new_bundle(
        config,
        bundle,
        batch_id=batch_id,
        source_client=source_client,
        now=now,
    )
    return _commit_bundled(
        state,
        document_id=document_id,
        manifest=manifest,
        state_store=state_store,
        now=now,
    )


def _fetch_reviewable(
    api_client: TranscriptionReviewApiClient,
    document: DocumentLifecycle,
) -> ReviewBundleResponse:
    """Fetch one fresh bundle and require the same selected identity."""

    fetched = api_client.review_bundle(document.document_id)
    if isinstance(fetched, ReviewBundleNotFound):
        raise FreshBundleRejectedError(
            f"fresh review bundle for document {document.document_id} returned HTTP 404"
        )
    if fetched.outcome is not BundleOutcome.REVIEWABLE:
        raise FreshBundleRejectedError(
            f"fresh review bundle for document {document.document_id} "
            f"is {fetched.outcome.value}"
        )
    archive_item_id = fetched.document.archive_item_id
    if (
        archive_item_id != document.archive_item_id
        or fetched.document.language != document.language
        or fetched.transcription is None
        or fetched.provenance is None
        or not fetched.source_pages
    ):
        raise FreshBundleRejectedError(
            f"fresh review bundle for document {document.document_id} "
            "does not match the selected document"
        )
    return fetched


def _publish_new_bundle(
    config: AgentConfig,
    bundle: ReviewBundleResponse,
    *,
    batch_id: str,
    source_client: httpx.Client,
    now: Clock,
) -> DocumentBundleManifest:
    document_id = bundle.document.document_id
    final = document_dir(config.batches_dir, batch_id, document_id)
    staging = _staging_document_dir(config.batches_dir, batch_id, document_id)
    published = False
    try:
        if staging.exists():
            _discard_staging(staging)
        staging.mkdir(parents=True)
        _write_staged_bundle(
            staging,
            bundle,
            batch_id=batch_id,
            source_client=source_client,
            created_at=now(),
        )
        verify_document_bundle(staging)
        _reject_persisted_download_credentials(staging)
        if final.exists():
            raise BundleConflictError(
                f"refusing to overwrite document bundle {document_id}"
            )
        final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, final)
        published = True
        _fsync_dir(final.parent)
        verified = verify_document_bundle(final)
        _reject_persisted_download_credentials(final)
        return verified
    except Exception:
        if not published:
            _discard_staging(staging)
        raise
    finally:
        _remove_empty_dir(_staging_root(config.batches_dir, batch_id))


def _write_staged_bundle(
    staging: Path,
    bundle: ReviewBundleResponse,
    *,
    batch_id: str,
    source_client: httpx.Client,
    created_at: datetime,
) -> DocumentBundleManifest:
    transcription = bundle.transcription
    provenance = bundle.provenance
    archive_item_id = bundle.document.archive_item_id
    if transcription is None or provenance is None or archive_item_id is None:
        raise FreshBundleRejectedError(
            f"fresh review bundle for document {bundle.document.document_id} "
            "does not match the selected document"
        )
    text_bytes = transcription.text.encode("utf-8")
    local_hash = sha256_text(transcription.text)
    server_hash = transcription.text_sha256
    if server_hash is not None and server_hash != local_hash:
        raise FreshBundleRejectedError(
            "server transcription hash does not match the local hash for "
            f"document {bundle.document.document_id}"
        )

    api_bytes = _sanitized_api_payload(bundle)
    _write_exclusive(staging / API_PAYLOAD_RELATIVE_PATH, api_bytes)
    _write_exclusive(staging / TRANSCRIPTION_RELATIVE_PATH, text_bytes)

    source_artifacts: list[SourceArtifact] = []
    source_files: list[BundleFile] = []
    for page in bundle.source_pages:
        data = _download_source_page(
            source_client, page, document_id=bundle.document.document_id
        )
        relative = source_relative_path(
            page.order_index, page.source_kind, page.mime_type
        )
        _write_exclusive(staging / relative, data)
        digest = sha256_bytes(data)
        source_artifacts.append(
            SourceArtifact(
                order_index=page.order_index,
                source_file_id=page.source_file_id,
                source_kind=page.source_kind,
                mime_type=page.mime_type,
                relative_path=relative,
                sha256=digest,
                size_bytes=len(data),
                file_s3_key=page.file_s3_key,
                include_in_ocr=page.include_in_ocr,
            )
        )
        source_files.append(
            BundleFile(
                relative_path=relative,
                sha256=digest,
                size_bytes=len(data),
                role="source",
            )
        )

    language = _language(bundle.document.language, bundle.document.document_id)
    manifest = DocumentBundleManifest(
        document_id=bundle.document.document_id,
        archive_item_id=archive_item_id,
        language=language,
        batch_id=batch_id,
        created_at=created_at,
        transcription=TranscriptionArtifact(
            relative_path=TRANSCRIPTION_RELATIVE_PATH,
            sha256=local_hash,
            size_bytes=len(text_bytes),
            server_text_sha256=server_hash,
            text_result_id=transcription.text_result_id,
            result_type=transcription.result_type,
            status=transcription.status,
            verification_status=transcription.verification_status,
            engine_key=transcription.engine_key,
            engine=transcription.engine,
            source_revision=transcription.source_revision,
            based_on_source_revision=transcription.based_on_source_revision,
            api_updated_at=transcription.updated_at,
        ),
        sources=source_artifacts,
        provenance=LocalProvenance(
            displayed_transcription_resolver=provenance.displayed_transcription_resolver,
            source_set=provenance.source_set,
            presign_expires_in=provenance.presign_expires_in,
            server_text_sha256=server_hash,
        ),
        files=[
            BundleFile(
                relative_path=API_PAYLOAD_RELATIVE_PATH,
                sha256=sha256_bytes(api_bytes),
                size_bytes=len(api_bytes),
                role="api_review_bundle",
            ),
            BundleFile(
                relative_path=TRANSCRIPTION_RELATIVE_PATH,
                sha256=local_hash,
                size_bytes=len(text_bytes),
                role="transcription_text",
            ),
            *source_files,
        ],
    )
    publish_document_manifest(staging / DOCUMENT_MANIFEST_NAME, manifest)
    return manifest


def _download_source_page(
    client: httpx.Client,
    page: SourcePagePayload,
    *,
    document_id: int,
) -> bytes:
    """GET one presigned URL from the validated page. The URL is not stored."""

    try:
        response = client.get(
            page.download_url,
            follow_redirects=False,
            auth=None,
        )
    except httpx.TimeoutException:
        raise SourceDownloadError(
            f"source download timed out for document {document_id} order {page.order_index}",
            document_id=document_id,
            order_index=page.order_index,
        ) from None
    except httpx.RequestError:
        raise SourceDownloadError(
            "source download failed before an HTTP response for "
            f"document {document_id} order {page.order_index}",
            document_id=document_id,
            order_index=page.order_index,
        ) from None
    if response.status_code != 200:
        raise SourceDownloadError(
            f"source download for document {document_id} order {page.order_index} "
            f"returned HTTP {response.status_code}",
            document_id=document_id,
            order_index=page.order_index,
            status_code=response.status_code,
        )
    return bytes(response.content)


def _sanitized_api_payload(bundle: ReviewBundleResponse) -> bytes:
    """Durable API metadata with presigned URL fields removed."""

    payload = _scrub_urls(bundle.model_dump(mode="json"))
    if not isinstance(payload, dict):
        raise FreshBundleRejectedError(
            f"sanitized review bundle for document {bundle.document.document_id} "
            "still contains a download URL"
        )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    leaked_url = any(
        page.download_url.encode("utf-8") in encoded for page in bundle.source_pages
    )
    if _contains_download_credentials(encoded) or leaked_url:
        raise FreshBundleRejectedError(
            f"sanitized review bundle for document {bundle.document.document_id} "
            "still contains a download URL"
        )
    return encoded


def _scrub_urls(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _scrub_urls(item)
            for key, item in value.items()
            if key not in _URL_KEYS
        }
    if isinstance(value, list):
        return [_scrub_urls(item) for item in value]
    return value


def _matching_final_bundle(
    config: AgentConfig,
    document: DocumentLifecycle,
    *,
    batch_id: str,
) -> DocumentBundleManifest | None:
    final = document_dir(config.batches_dir, batch_id, document.document_id)
    if not final.exists():
        return None
    manifest_path = final / DOCUMENT_MANIFEST_NAME
    if not manifest_path.is_file():
        raise BundleConflictError(
            f"incomplete document bundle {document.document_id} is not publishable"
        )
    manifest = verify_document_bundle(final)
    _require_manifest_identity(manifest, document, batch_id=batch_id)
    _reject_persisted_download_credentials(final)
    return manifest


def _promote_complete_staging(
    config: AgentConfig,
    document: DocumentLifecycle,
    *,
    batch_id: str,
) -> DocumentBundleManifest | None:
    staging = _staging_document_dir(config.batches_dir, batch_id, document.document_id)
    if not staging.exists():
        return None
    if not (staging / DOCUMENT_MANIFEST_NAME).is_file():
        _discard_staging(staging)
        return None
    try:
        manifest = verify_document_bundle(staging)
        _require_manifest_identity(manifest, document, batch_id=batch_id)
    except Exception:
        _discard_staging(staging)
        raise
    _reject_persisted_download_credentials(staging)
    final = document_dir(config.batches_dir, batch_id, document.document_id)
    if final.exists():
        raise BundleConflictError(
            f"refusing to overwrite document bundle {document.document_id}"
        )
    final.parent.mkdir(parents=True, exist_ok=True)
    os.rename(staging, final)
    _fsync_dir(final.parent)
    verified = verify_document_bundle(final)
    _reject_persisted_download_credentials(final)
    return verified


def _require_checkpoint_bundle(
    config: AgentConfig,
    state: ControllerState,
    document: DocumentLifecycle,
) -> None:
    batch_id = _require_batch_id(state)
    expected_path = document_manifest_relative_path(document.document_id)
    if document.bundle_manifest_path != expected_path:
        raise BundleConflictError(
            f"bundled document {document.document_id} has an unexpected manifest path"
        )
    manifest = _matching_final_bundle(config, document, batch_id=batch_id)
    if manifest is None:
        raise BundleConflictError(
            f"bundled document {document.document_id} is missing its bundle"
        )
    checkpoint_sources = [
        (item.order_index, item.sha256, item.size_bytes, item.relative_path)
        for item in document.source_hashes
    ]
    manifest_sources = [
        (item.order_index, item.sha256, item.size_bytes, item.relative_path)
        for item in manifest.sources
    ]
    if (
        manifest.transcription.sha256 != document.transcription_sha256
        or checkpoint_sources != manifest_sources
    ):
        raise BundleConflictError(
            f"bundled document {document.document_id} does not match the checkpoint"
        )


def _commit_bundled(
    state: ControllerState,
    *,
    document_id: int,
    manifest: DocumentBundleManifest,
    state_store: StateStore,
    now: Clock,
) -> ControllerState:
    return state_store.commit(
        mark_bundled(
            state,
            document_id=document_id,
            transcription_sha256=manifest.transcription.sha256,
            sources=[
                SourceHashRecord(
                    order_index=source.order_index,
                    sha256=source.sha256,
                    size_bytes=source.size_bytes,
                    relative_path=source.relative_path,
                )
                for source in manifest.sources
            ],
            bundle_manifest_path=document_manifest_relative_path(document_id),
            occurred_at=now(),
        )
    )


def _require_manifest_identity(
    manifest: DocumentBundleManifest,
    document: DocumentLifecycle,
    *,
    batch_id: str,
) -> None:
    if (
        manifest.document_id != document.document_id
        or manifest.archive_item_id != document.archive_item_id
        or manifest.language != document.language
        or manifest.batch_id != batch_id
    ):
        raise BundleConflictError(
            f"existing bundle for document {document.document_id} "
            "does not match the selected document"
        )


def _reject_persisted_download_credentials(bundle_dir: Path) -> None:
    """Reject presigned URL material in durable JSON metadata.

    ``api/review_bundle.json`` is checked even when its hash matches the
    manifest. Source binaries and transcription text are not scanned.
    """

    payload = bundle_dir / API_PAYLOAD_RELATIVE_PATH
    if not payload.is_file():
        raise BundleConflictError(
            f"document bundle {bundle_dir.name} is missing its API metadata"
        )
    for relative in (API_PAYLOAD_RELATIVE_PATH, DOCUMENT_MANIFEST_NAME):
        path = bundle_dir / relative
        if path.is_file() and _contains_download_credentials(path.read_bytes()):
            raise BundleConflictError(
                f"document bundle {bundle_dir.name} contains a presigned URL"
            )


def _contains_download_credentials(payload: bytes) -> bool:
    return any(marker in payload for marker in _DOWNLOAD_CREDENTIAL_MARKERS)


def _can_bundle(state: ControllerState) -> bool:
    if state.batch_status is BatchStatus.BUNDLING:
        return bool(state.documents)
    if state.batch_status is not BatchStatus.ACCUMULATING:
        return False
    count = len(state.documents)
    target = state.batch_target_count
    if count == target:
        return True
    return 1 <= count < target and state.discovery_exhausted


def _result(state: ControllerState, *, completed: bool) -> BundleResult:
    return BundleResult(
        completed=completed,
        batch_status=state.batch_status,
        batch_id=state.current_batch_id,
        document_ids=tuple(state.current_batch_document_ids),
    )


def _require_document(state: ControllerState, document_id: int) -> DocumentLifecycle:
    for document in state.documents:
        if document.document_id == document_id:
            return document
    raise BundleConflictError(f"document {document_id} is not in the selected batch")


def _require_batch_id(state: ControllerState) -> str:
    if state.current_batch_id is None:
        raise BundleConflictError("bundling requires an open batch id")
    return state.current_batch_id


def _language(value: str, document_id: int) -> Literal["en", "fr"]:
    if value == "en":
        return "en"
    if value == "fr":
        return "fr"
    raise FreshBundleRejectedError(
        f"fresh review bundle for document {document_id} does not match the selected document"
    )


def _staging_root(batches_dir: Path, batch_id: str) -> Path:
    return batch_dir(batches_dir, batch_id) / _STAGING_DIRNAME


def _staging_document_dir(batches_dir: Path, batch_id: str, document_id: int) -> Path:
    return _staging_root(batches_dir, batch_id) / str(document_id)


def _discard_staging(path: Path) -> None:
    if path.parent.name != _STAGING_DIRNAME:
        raise BundleConflictError("refusing to delete a path outside staging")
    if path.exists():
        shutil.rmtree(path)


def _remove_empty_dir(path: Path) -> None:
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
