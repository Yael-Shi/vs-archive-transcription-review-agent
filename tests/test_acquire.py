"""Source acquisition against in-process HTTP. No live network and no AI calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from transcription_review_agent.api_client.client import TranscriptionReviewApiClient
from transcription_review_agent.api_client.schemas import BlockingIntegrityCode
from transcription_review_agent.bundling.acquire import (
    _staging_document_dir,
    bundle_selected_batch,
)
from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.layout import (
    document_dir,
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
from transcription_review_agent.constants import BUNDLE_SCHEMA_VERSION
from transcription_review_agent.errors import (
    BundleConflictError,
    BundleVerificationError,
    FreshBundleRejectedError,
    SourceDownloadError,
)
from transcription_review_agent.state.schema import (
    BatchStatus,
    ControllerState,
    DocumentPhase,
    initial_state,
)
from transcription_review_agent.state.store import StateStore
from transcription_review_agent.state.transitions import (
    note_blocking,
    note_discovery_exhausted,
    note_no_work,
    select_reviewable,
)
from support import at

BASE_URL = "http://127.0.0.1:8000"
SECRET = "presign-secret-TOKEN-do-not-persist"
AI_HOST = "api.openai.com"


@dataclass
class SourcePlan:
    body: bytes
    mime: str = "application/pdf"
    kind: str = "PRIMARY_FILE"
    status: int = 200
    fault: str | None = None


@dataclass
class DocPlan:
    archive_item_id: int
    language: str
    text: str
    sources: list[SourcePlan]
    outcome: str = "reviewable"
    code: str = "LANGUAGE_NOT_IN_SCOPE"
    server_hash: str | None = None
    archive_item_override: int | None = None
    language_override: str | None = None
    bundle_calls: int = 0


@dataclass
class ApiWorld:
    docs: dict[int, DocPlan]
    bundle_ids: list[int] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    source_headers: list[dict[str, str]] = field(default_factory=list)

    def api_client(self) -> TranscriptionReviewApiClient:
        return TranscriptionReviewApiClient(
            BASE_URL,
            5,
            transport=httpx.MockTransport(self._api),
        )

    def source_client(self, *, follow_redirects: bool = False) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self._source),
            follow_redirects=follow_redirects,
            trust_env=False,
        )

    def _api(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET" or not request.url.path.endswith("/review-bundle"):
            raise AssertionError(f"unexpected API call {request.method} {request.url}")
        document_id = int(request.url.path.split("/documents/", 1)[1].split("/", 1)[0])
        self.bundle_ids.append(document_id)
        plan = self.docs[document_id]
        plan.bundle_calls += 1
        if plan.outcome == "not_found":
            return httpx.Response(404, json={"detail": "missing"})
        if plan.outcome == "http":
            return httpx.Response(503, json={"detail": "down"})
        if plan.outcome == "schema":
            return httpx.Response(200, json={"schema_version": BUNDLE_SCHEMA_VERSION})
        if plan.outcome == "non_eligible":
            return httpx.Response(
                200, json=_classified(document_id, "NON_ELIGIBLE", plan)
            )
        if plan.outcome == "blocking":
            return httpx.Response(
                200,
                json=_classified(document_id, "BLOCKING_INTEGRITY_ERROR", plan),
            )
        return httpx.Response(200, json=_reviewable(document_id, plan))

    def _source(self, request: httpx.Request) -> httpx.Response:
        self.source_urls.append(str(request.url))
        self.source_headers.append(
            {key.lower(): value for key, value in request.headers.items()}
        )
        document_id, order_index = _source_identity(request.url.path)
        plan = self.docs[document_id].sources[order_index]
        if plan.fault == "timeout":
            raise httpx.ReadTimeout("timed out")
        if plan.fault == "transport":
            raise httpx.ConnectError("connection refused")
        if plan.fault == "redirect":
            return httpx.Response(
                302,
                headers={"Location": "http://sources.test/redirect-target"},
            )
        if plan.status != 200:
            return httpx.Response(plan.status, content=b"nope")
        return httpx.Response(200, content=plan.body)


def _source_url(document_id: int, order_index: int) -> str:
    return (
        f"http://sources.test/objects/{document_id}/{order_index}"
        f"?X-Amz-Signature={SECRET}"
    )


def _source_identity(path: str) -> tuple[int, int]:
    parts = path.strip("/").split("/")
    return int(parts[-2]), int(parts[-1])


def _reviewable(document_id: int, plan: DocPlan) -> dict[str, object]:
    transcription: dict[str, object] = {
        "text_result_id": 40 + document_id,
        "result_type": "SOURCE_TEXT",
        "status": "NEEDS_REVIEW",
        "verification_status": "UNVERIFIED",
        "text": plan.text,
    }
    if plan.server_hash is not None:
        transcription["text_sha256"] = plan.server_hash
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "outcome": "REVIEWABLE",
        "document": {
            "document_id": document_id,
            "archive_item_id": plan.archive_item_override or plan.archive_item_id,
            "language": plan.language_override or plan.language,
            "doc_type": "PDF",
            "item_type": "OCR_DOCUMENT",
            "upload_status": "UPLOADED",
            "processing_state_user": "READY",
            "visibility": "public",
        },
        "transcription": transcription,
        "source_pages": [
            {
                "source_file_id": 700 + document_id * 10 + index,
                "order_index": index,
                "display_number": index + 1,
                "source_kind": source.kind,
                "file_s3_key": f"documents/{document_id}/page-{index}",
                "mime_type": source.mime,
                "size_bytes": len(source.body),
                "include_in_ocr": True,
                "upload_status": "UPLOADED",
                "download_url": _source_url(document_id, index),
                "download_url_expires_in": 900,
            }
            for index, source in enumerate(plan.sources)
        ],
        "provenance": {
            "displayed_transcription_resolver": "resolve_displayed_transcription_result",
            "source_set": "primary_file",
            "presign_expires_in": 900,
        },
    }


def _classified(document_id: int, outcome: str, plan: DocPlan) -> dict[str, object]:
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "outcome": outcome,
        "document": {
            "document_id": document_id,
            "archive_item_id": plan.archive_item_id,
            "language": plan.language,
            "doc_type": "PDF",
            "item_type": "OCR_DOCUMENT",
            "upload_status": "UPLOADED",
            "processing_state_user": "READY",
            "visibility": "public",
        },
        "error": {"code": plan.code, "message": "classified by the server"},
    }


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        api_base_url=BASE_URL,
        scan_page_limit=50,
        state_dir=tmp_path / "state",
        batches_dir=tmp_path / "batches",
        request_timeout_seconds=5,
    )


def _clock():
    instant = {"n": 100}

    def now():
        instant["n"] += 1
        return at(instant["n"])

    return now


def _pdf(document_id: int, *, extra: bytes = b"") -> bytes:
    return b"%PDF-1.7\n" + f"original-{document_id}\n".encode() + extra


def _plan(
    document_id: int,
    *,
    language: str = "en",
    text: str | None = None,
    sources: list[SourcePlan] | None = None,
    **overrides: object,
) -> DocPlan:
    values = {
        "archive_item_id": 1000 + document_id,
        "language": language,
        "text": text if text is not None else f"text-{document_id}\n",
        "sources": sources if sources is not None else [SourcePlan(_pdf(document_id))],
    }
    values.update(overrides)
    return DocPlan(**values)  # type: ignore[arg-type]


def _seed(
    store: StateStore,
    plans: dict[int, DocPlan],
    *,
    exhausted: bool,
) -> ControllerState:
    state = initial_state()
    for offset, document_id in enumerate(sorted(plans), start=1):
        plan = plans[document_id]
        state = store.commit(
            select_reviewable(
                state,
                document_id=document_id,
                archive_item_id=plan.archive_item_id,
                language=plan.language,  # type: ignore[arg-type]
                occurred_at=at(offset),
            )
        )
    if exhausted:
        state = store.commit(note_discovery_exhausted(state, occurred_at=at(50)))
    return state


def _run(
    tmp_path: Path,
    plans: dict[int, DocPlan],
    *,
    exhausted: bool,
    world: ApiWorld | None = None,
    store: StateStore | None = None,
    follow_redirects: bool = False,
):
    config = _config(tmp_path)
    state_store = store if store is not None else StateStore(config.state_dir)
    if store is None:
        _seed(state_store, plans, exhausted=exhausted)
    scripted = world if world is not None else ApiWorld(plans)
    result = bundle_selected_batch(
        config,
        scripted.api_client(),
        state_store,
        source_client=scripted.source_client(follow_redirects=follow_redirects),
        clock=_clock(),
    )
    return result, state_store, scripted, config


def _files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_api_metadata_has_no_download_credentials(bundle_dir: Path) -> None:
    payload = (bundle_dir / "api" / "review_bundle.json").read_bytes()
    assert b"download_url" not in payload
    assert b"download_url_expires_in" not in payload
    assert SECRET.encode() not in payload
    assert b"X-Amz-" not in payload


def _retarget_api_payload(bundle_dir: Path, payload: bytes) -> None:
    """Replace API metadata and the manifest hash that describes it."""

    api_path = bundle_dir / "api" / "review_bundle.json"
    api_path.write_bytes(payload)
    manifest_path = bundle_dir / "document_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    updated = False
    for entry in manifest["files"]:
        if entry["relative_path"] == "api/review_bundle.json":
            entry["sha256"] = sha256_bytes(payload)
            entry["size_bytes"] = len(payload)
            updated = True
    assert updated
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _assert_no_secret(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        assert SECRET.encode() not in data
        assert b"download_url" not in data
        assert b"X-Amz-" not in data
        assert AI_HOST.encode() not in data


def _poisoned_api(document_id: int) -> bytes:
    payload = {
        "outcome": "REVIEWABLE",
        "source_pages": [
            {
                "download_url": _source_url(document_id, 0),
                "download_url_expires_in": 900,
            }
        ],
    }
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _materialize(
    batches_dir: Path,
    *,
    document_id: int,
    archive_item_id: int,
    language: str,
    text: str,
    source: bytes,
    root: Path | None = None,
    api: bytes | None = None,
) -> Path:
    root = (
        document_dir(batches_dir, "batch-000001", document_id) if root is None else root
    )
    text_bytes = text.encode("utf-8")
    source_rel = source_relative_path(0, "PRIMARY_FILE", "application/pdf")
    api = b'{"outcome":"REVIEWABLE"}\n' if api is None else api
    payloads = {
        "api/review_bundle.json": api,
        "transcription/transcription.txt": text_bytes,
        source_rel: source,
    }
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    text_hash = sha256_text(text)
    source_hash = sha256_bytes(source)
    manifest = DocumentBundleManifest(
        document_id=document_id,
        archive_item_id=archive_item_id,
        language=language,  # type: ignore[arg-type]
        batch_id="batch-000001",
        created_at=at(1),
        transcription=TranscriptionArtifact(
            relative_path="transcription/transcription.txt",
            sha256=text_hash,
            size_bytes=len(text_bytes),
            text_result_id=1,
            result_type="SOURCE_TEXT",
            status="NEEDS_REVIEW",
            verification_status="UNVERIFIED",
        ),
        sources=[
            SourceArtifact(
                order_index=0,
                source_kind="PRIMARY_FILE",
                mime_type="application/pdf",
                relative_path=source_rel,
                sha256=source_hash,
                size_bytes=len(source),
                file_s3_key=f"documents/{document_id}/page-0",
                include_in_ocr=True,
            )
        ],
        provenance=LocalProvenance(
            displayed_transcription_resolver="resolve_displayed_transcription_result",
            source_set="primary_file",
            presign_expires_in=900,
        ),
        files=[
            BundleFile(
                relative_path="api/review_bundle.json",
                sha256=sha256_bytes(api),
                size_bytes=len(api),
                role="api_review_bundle",
            ),
            BundleFile(
                relative_path="transcription/transcription.txt",
                sha256=text_hash,
                size_bytes=len(text_bytes),
                role="transcription_text",
            ),
            BundleFile(
                relative_path=source_rel,
                sha256=source_hash,
                size_bytes=len(source),
                role="source",
            ),
        ],
    )
    publish_document_manifest(root / "document_manifest.json", manifest)
    return root


def test_full_batch_enters_bundling_and_publishes_five_documents(
    tmp_path: Path,
) -> None:
    plans = {
        document_id: _plan(document_id, language="fr" if document_id == 3 else "en")
        for document_id in (1, 2, 3, 4, 5)
    }
    result, store, world, config = _run(tmp_path, plans, exhausted=False)
    state = store.load()

    assert result.completed is True
    assert result.document_ids == (1, 2, 3, 4, 5)
    assert result.batch_id == "batch-000001"
    assert state.batch_status is BatchStatus.BUNDLING
    assert [document.phase for document in state.documents] == [
        DocumentPhase.BUNDLED
    ] * 5
    assert world.bundle_ids == [1, 2, 3, 4, 5]
    assert [entry.event_type for entry in store.read_ledger()[-11:]] == [
        "bundling_started",
        "document_bundling",
        "document_bundled",
        "document_bundling",
        "document_bundled",
        "document_bundling",
        "document_bundled",
        "document_bundling",
        "document_bundled",
        "document_bundling",
        "document_bundled",
    ]
    assert "batch_sealed" not in {entry.event_type for entry in store.read_ledger()}
    assert not (config.batches_dir / "batch-000001" / "batch_manifest.json").exists()
    for document_id in plans:
        verified = verify_document_bundle(
            document_dir(config.batches_dir, "batch-000001", document_id)
        )
        assert verified.document_id == document_id
        assert verified.transcription.sha256 == sha256_text(plans[document_id].text)


def test_short_exhausted_batch_enters_bundling(tmp_path: Path) -> None:
    plans = {4: _plan(4), 9: _plan(9, language="fr")}
    result, store, world, _config_obj = _run(tmp_path, plans, exhausted=True)
    state = store.load()

    assert result.completed is True
    assert result.document_ids == (4, 9)
    assert state.batch_status is BatchStatus.BUNDLING
    assert state.discovery_exhausted is True
    assert [document.phase for document in state.documents] == [
        DocumentPhase.BUNDLED,
        DocumentPhase.BUNDLED,
    ]
    assert world.bundle_ids == [4, 9]
    assert len(world.source_urls) == 2


def test_no_work_and_unready_batches_do_not_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    store.commit(note_no_work(initial_state(), occurred_at=at(1)))
    before = (config.state_dir / "ledger.jsonl").read_bytes()
    world = ApiWorld({})
    result = bundle_selected_batch(
        config,
        world.api_client(),
        store,
        source_client=world.source_client(),
        clock=_clock(),
    )

    assert result.completed is False
    assert result.batch_status is BatchStatus.NO_WORK
    assert result.document_ids == ()
    assert world.bundle_ids == []
    assert world.source_urls == []
    assert (config.state_dir / "ledger.jsonl").read_bytes() == before
    assert _files(config.batches_dir) == set()

    short = {1: _plan(1), 2: _plan(2)}
    short_store = StateStore(tmp_path / "short-state")
    _seed(short_store, short, exhausted=False)
    short_before = len(short_store.read_ledger())
    short_config = AgentConfig(
        api_base_url=BASE_URL,
        scan_page_limit=50,
        state_dir=tmp_path / "short-state",
        batches_dir=tmp_path / "short-batches",
        request_timeout_seconds=5,
    )
    short_world = ApiWorld(short)
    short_result = bundle_selected_batch(
        short_config,
        short_world.api_client(),
        short_store,
        source_client=short_world.source_client(),
        clock=_clock(),
    )
    assert short_result.completed is False
    assert short_result.batch_status is BatchStatus.ACCUMULATING
    assert len(short_store.read_ledger()) == short_before
    assert short_world.bundle_ids == []


def test_blocked_state_does_not_bundle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    state = store.commit(
        select_reviewable(
            initial_state(),
            document_id=3,
            archive_item_id=1003,
            language="en",
            occurred_at=at(1),
        )
    )
    blocked = note_blocking(
        state,
        document_id=8,
        code=BlockingIntegrityCode.MISSING_SOURCE_FILE,
        message="source file is missing",
        occurred_at=at(2),
    )
    assert blocked is not None
    store.commit(blocked)
    before = (config.state_dir / "ledger.jsonl").read_bytes()
    world = ApiWorld({3: _plan(3)})
    result = bundle_selected_batch(
        config,
        world.api_client(),
        store,
        source_client=world.source_client(),
        clock=_clock(),
    )

    assert result.completed is False
    assert result.batch_status is BatchStatus.BLOCKED
    assert world.bundle_ids == []
    assert world.source_urls == []
    assert (config.state_dir / "ledger.jsonl").read_bytes() == before
    assert store.load().documents[0].phase is DocumentPhase.SELECTED


def test_documents_are_fetched_once_in_ascending_id_order(tmp_path: Path) -> None:
    plans = {9: _plan(9), 2: _plan(2), 5: _plan(5, language="fr")}
    _result, _store, world, _config_obj = _run(tmp_path, plans, exhausted=True)

    assert world.bundle_ids == [2, 5, 9]
    assert [plan.bundle_calls for plan in plans.values()] == [1, 1, 1]
    assert len(world.source_urls) == 3
    assert world.source_urls == [
        _source_url(2, 0),
        _source_url(5, 0),
        _source_url(9, 0),
    ]


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        ("not_found", "LANGUAGE_NOT_IN_SCOPE"),
        ("non_eligible", "NOT_OCR_DOCUMENT"),
        ("blocking", "MISSING_PRIMARY_SOURCE"),
    ],
)
def test_reclassified_refresh_stops_without_replacing_the_document(
    tmp_path: Path,
    outcome: str,
    code: str,
) -> None:
    plans = {
        2: _plan(2, outcome=outcome, code=code),
        6: _plan(6),
    }
    with pytest.raises(FreshBundleRejectedError) as exc_info:
        _run(tmp_path, plans, exhausted=True)
    store = StateStore(tmp_path / "state")
    types = [entry.event_type for entry in store.read_ledger()]

    message = str(exc_info.value)
    if outcome == "not_found":
        assert "404" in message
    elif outcome == "non_eligible":
        assert "NON_ELIGIBLE" in message
    else:
        assert "BLOCKING_INTEGRITY_ERROR" in message
    assert SECRET not in str(exc_info.value)
    assert "document_bundled" not in types
    assert types[-2:] == ["bundling_started", "document_bundling"]
    assert store.load().documents[0].phase is DocumentPhase.BUNDLING
    assert store.load().documents[1].phase is DocumentPhase.SELECTED
    assert _files(tmp_path / "batches") == set()


def test_archive_item_or_language_mismatch_fails_closed(tmp_path: Path) -> None:
    archive_plans = {4: _plan(4, archive_item_override=4242)}
    with pytest.raises(FreshBundleRejectedError, match="does not match"):
        _run(tmp_path, archive_plans, exhausted=True)
    assert "document_bundled" not in {
        entry.event_type for entry in StateStore(tmp_path / "state").read_ledger()
    }

    language_root = tmp_path / "language"
    language_plans = {4: _plan(4, language="en", language_override="fr")}
    with pytest.raises(FreshBundleRejectedError, match="does not match"):
        _run(language_root, language_plans, exhausted=True)
    assert _files(language_root / "batches") == set()


def test_exact_transcription_bytes_and_hash_are_preserved(tmp_path: Path) -> None:
    text = "keep\nthis\n"
    plans = {4: _plan(4, text=text, server_hash=sha256_text(text))}
    _result, store, _world, config = _run(tmp_path, plans, exhausted=True)
    document = document_dir(config.batches_dir, "batch-000001", 4)
    payload = (document / "transcription" / "transcription.txt").read_bytes()
    verified = verify_document_bundle(document)

    assert payload == b"keep\nthis\n"
    assert payload != b"keep\nthis"
    assert verified.transcription.sha256 == sha256_text(text)
    assert verified.transcription.sha256 != sha256_text(text.replace("\n", ""))
    assert verified.transcription.server_text_sha256 == sha256_text(text)
    bundled = store.read_ledger()[-1]
    assert bundled.transcription_sha256 == sha256_text(text)


def test_unicode_text_is_not_normalized(tmp_path: Path) -> None:
    nfd = "cafe\u0301"
    nfc = "caf\u00e9"
    plans = {4: _plan(4, text=nfd)}
    _result, _store, _world, config = _run(tmp_path, plans, exhausted=True)
    payload = (
        document_dir(config.batches_dir, "batch-000001", 4)
        / "transcription"
        / "transcription.txt"
    ).read_bytes()

    assert payload == nfd.encode("utf-8")
    assert payload != nfc.encode("utf-8")
    assert sha256_bytes(payload) == sha256_text(nfd)
    assert sha256_bytes(payload) != sha256_text(nfc)


def test_server_text_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    plans = {4: _plan(4, text="exact", server_hash="cd" * 32)}
    with pytest.raises(FreshBundleRejectedError, match="local hash"):
        _run(tmp_path, plans, exhausted=True)
    store = StateStore(tmp_path / "state")

    assert store.load().documents[0].phase is DocumentPhase.BUNDLING
    assert store.load().documents[0].transcription_sha256 is None
    assert _files(tmp_path / "batches") == set()


def test_source_bytes_and_pdf_are_preserved_and_downloaded_once(tmp_path: Path) -> None:
    pdf = b"%PDF-1.7\n%original-bytes\n%%EOF"
    image = b"\xff\xd8\xffjpeg-bytes"
    plans = {
        4: _plan(
            4,
            sources=[
                SourcePlan(pdf, mime="application/pdf", kind="PRIMARY_FILE"),
                SourcePlan(image, mime="image/jpeg", kind="SOURCE_FILE"),
            ],
        )
    }
    _result, _store, world, config = _run(tmp_path, plans, exhausted=True)
    root = document_dir(config.batches_dir, "batch-000001", 4)

    assert world.source_urls == [_source_url(4, 0), _source_url(4, 1)]
    assert (root / "sources" / "0000_primary_file.pdf").read_bytes() == pdf
    assert (root / "sources" / "0001_source_file.jpg").read_bytes() == image
    verified = verify_document_bundle(root)
    assert verified.sources[0].sha256 == sha256_bytes(pdf)
    assert verified.sources[1].sha256 == sha256_bytes(image)
    assert verified.provenance.pdf_rasterization == "not_in_pilot_01"
    assert all("authorization" not in headers for headers in world.source_headers)
    assert all("cookie" not in headers for headers in world.source_headers)


def test_redirect_and_http_failure_stop_without_mark_bundled(tmp_path: Path) -> None:
    redirect_plans = {4: _plan(4, sources=[SourcePlan(_pdf(4), fault="redirect")])}
    redirect_world = ApiWorld(redirect_plans)
    with pytest.raises(SourceDownloadError) as redirect_exc:
        _run(
            tmp_path,
            redirect_plans,
            exhausted=True,
            world=redirect_world,
            follow_redirects=True,
        )
    assert redirect_world.source_urls == [_source_url(4, 0)]
    assert redirect_exc.value.status_code == 302
    assert SECRET not in str(redirect_exc.value)
    assert "sources.test" not in str(redirect_exc.value)
    assert "X-Amz-" not in str(redirect_exc.value)
    redirect_store = StateStore(tmp_path / "state")
    assert redirect_store.load().documents[0].phase is DocumentPhase.BUNDLING
    assert "document_bundled" not in {
        entry.event_type for entry in redirect_store.read_ledger()
    }

    http_root = tmp_path / "http"
    http_plans = {4: _plan(4, sources=[SourcePlan(_pdf(4), status=500)])}
    world = ApiWorld(http_plans)
    with pytest.raises(SourceDownloadError) as http_exc:
        _run(http_root, http_plans, exhausted=True, world=world)
    assert http_exc.value.status_code == 500
    assert world.source_urls == [_source_url(4, 0)]
    assert "redirect-target" not in "".join(world.source_urls)
    assert _files(http_root / "batches") == set()


def test_timeout_and_transport_failures_stop_without_mark_bundled(
    tmp_path: Path,
) -> None:
    timeout_plans = {4: _plan(4, sources=[SourcePlan(_pdf(4), fault="timeout")])}
    with pytest.raises(SourceDownloadError, match="timed out") as timeout_exc:
        _run(tmp_path, timeout_plans, exhausted=True)
    assert SECRET not in str(timeout_exc.value)
    assert "sources.test" not in repr(timeout_exc.value)
    assert (
        StateStore(tmp_path / "state").load().documents[0].transcription_sha256 is None
    )

    transport_root = tmp_path / "transport"
    transport_plans = {4: _plan(4, sources=[SourcePlan(_pdf(4), fault="transport")])}
    with pytest.raises(SourceDownloadError, match="before an HTTP response"):
        _run(transport_root, transport_plans, exhausted=True)
    assert "document_bundled" not in {
        entry.event_type for entry in StateStore(transport_root / "state").read_ledger()
    }
    assert _files(transport_root / "batches") == set()


def test_presigned_url_is_absent_from_durable_files(tmp_path: Path) -> None:
    plans = {4: _plan(4, text="keep\nthis\n"), 8: _plan(8, language="fr")}
    _result, _store, _world, config = _run(tmp_path, plans, exhausted=True)

    _assert_no_secret(tmp_path)
    manifest = (
        document_dir(config.batches_dir, "batch-000001", 4) / "document_manifest.json"
    ).read_text(encoding="utf-8")
    assert "download_url" not in manifest
    checkpoint = (config.state_dir / "controller_state.json").read_text(
        encoding="utf-8"
    )
    ledger = (config.state_dir / "ledger.jsonl").read_text(encoding="utf-8")
    assert SECRET not in checkpoint
    assert SECRET not in ledger


def test_verification_failure_does_not_publish_or_mark_bundled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcription_review_agent.bundling.acquire as acquire

    def reject(path: Path) -> DocumentBundleManifest:
        raise BundleVerificationError(f"forced failure for {path.name}")

    monkeypatch.setattr(acquire, "verify_document_bundle", reject)
    plans = {4: _plan(4), 8: _plan(8)}
    with pytest.raises(BundleVerificationError):
        _run(tmp_path, plans, exhausted=True)
    store = StateStore(tmp_path / "state")

    assert store.load().documents[0].phase is DocumentPhase.BUNDLING
    assert store.load().documents[1].phase is DocumentPhase.SELECTED
    assert _files(tmp_path / "batches") == set()
    assert not any(
        path.name == ".staging" for path in (tmp_path / "batches").rglob("*")
    )


def test_later_document_does_not_start_after_source_failure(tmp_path: Path) -> None:
    plans = {
        2: _plan(2),
        5: _plan(5, sources=[SourcePlan(_pdf(5), status=500)]),
        9: _plan(9),
    }
    world = ApiWorld(plans)
    with pytest.raises(SourceDownloadError):
        _run(tmp_path, plans, exhausted=True, world=world)
    state = StateStore(tmp_path / "state").load()

    assert world.bundle_ids == [2, 5]
    assert state.documents[0].phase is DocumentPhase.BUNDLED
    assert state.documents[1].phase is DocumentPhase.BUNDLING
    assert state.documents[2].phase is DocumentPhase.SELECTED
    assert (tmp_path / "batches" / "batch-000001" / "documents" / "2").is_dir()
    assert not (tmp_path / "batches" / "batch-000001" / "documents" / "5").exists()
    assert not (tmp_path / "batches" / "batch-000001" / "documents" / "9").exists()


def test_restart_from_bundling_resumes_without_redownloading(
    tmp_path: Path,
) -> None:
    plans = {2: _plan(2), 6: _plan(6)}
    config = _config(tmp_path)
    seed_store = StateStore(config.state_dir)
    seeded = _seed(seed_store, plans, exhausted=True)
    crash_at = seeded.applied_ledger_sequence + 4

    class CrashStore(StateStore):
        def write_checkpoint(self, state: ControllerState) -> None:
            if state.applied_ledger_sequence == crash_at:
                raise RuntimeError("crash before checkpoint")
            super().write_checkpoint(state)

    world = ApiWorld(plans)
    with pytest.raises(RuntimeError, match="crash before checkpoint"):
        bundle_selected_batch(
            config,
            world.api_client(),
            CrashStore(config.state_dir),
            source_client=world.source_client(),
            clock=_clock(),
        )
    recovered = StateStore(config.state_dir).load()
    first_bytes = _snapshot(document_dir(config.batches_dir, "batch-000001", 2))
    assert recovered.batch_status is BatchStatus.BUNDLING
    assert recovered.documents[0].phase is DocumentPhase.BUNDLED
    assert recovered.documents[1].phase is DocumentPhase.BUNDLING
    assert world.bundle_ids == [2]

    resume = ApiWorld(plans)
    result = bundle_selected_batch(
        config,
        resume.api_client(),
        StateStore(config.state_dir),
        source_client=resume.source_client(),
        clock=_clock(),
    )
    state = StateStore(config.state_dir).load()

    assert result.completed is True
    assert resume.bundle_ids == [6]
    assert resume.source_urls == [_source_url(6, 0)]
    assert _snapshot(document_dir(config.batches_dir, "batch-000001", 2)) == first_bytes
    assert [document.phase for document in state.documents] == [
        DocumentPhase.BUNDLED,
        DocumentPhase.BUNDLED,
    ]


def test_restart_from_document_bundling_rebuilds_that_document(tmp_path: Path) -> None:
    plans = {4: _plan(4)}
    config = _config(tmp_path)
    seed_store = StateStore(config.state_dir)
    seeded = _seed(seed_store, plans, exhausted=True)
    crash_at = seeded.applied_ledger_sequence + 2

    class CrashStore(StateStore):
        def write_checkpoint(self, state: ControllerState) -> None:
            if state.applied_ledger_sequence == crash_at:
                raise RuntimeError("crash before checkpoint")
            super().write_checkpoint(state)

    world = ApiWorld(plans)
    with pytest.raises(RuntimeError, match="crash before checkpoint"):
        bundle_selected_batch(
            config,
            world.api_client(),
            CrashStore(config.state_dir),
            source_client=world.source_client(),
            clock=_clock(),
        )
    recovered = StateStore(config.state_dir).load()
    assert recovered.batch_status is BatchStatus.BUNDLING
    assert recovered.documents[0].phase is DocumentPhase.BUNDLING
    assert world.bundle_ids == []
    assert _files(config.batches_dir) == set()

    resume = ApiWorld(plans)
    result = bundle_selected_batch(
        config,
        resume.api_client(),
        StateStore(config.state_dir),
        source_client=resume.source_client(),
        clock=_clock(),
    )
    state = StateStore(config.state_dir).load()
    assert result.completed is True
    assert resume.bundle_ids == [4]
    assert resume.source_urls == [_source_url(4, 0)]
    assert state.documents[0].phase is DocumentPhase.BUNDLED
    verify_document_bundle(document_dir(config.batches_dir, "batch-000001", 4))


def test_crash_after_bundled_append_replays_without_another_download(
    tmp_path: Path,
) -> None:
    plans = {4: _plan(4)}
    config = _config(tmp_path)
    seed_store = StateStore(config.state_dir)
    seeded = _seed(seed_store, plans, exhausted=True)
    crash_at = seeded.applied_ledger_sequence + 3

    class CrashStore(StateStore):
        def write_checkpoint(self, state: ControllerState) -> None:
            if state.applied_ledger_sequence == crash_at:
                raise RuntimeError("crash before checkpoint")
            super().write_checkpoint(state)

    world = ApiWorld(plans)
    with pytest.raises(RuntimeError, match="crash before checkpoint"):
        bundle_selected_batch(
            config,
            world.api_client(),
            CrashStore(config.state_dir),
            source_client=world.source_client(),
            clock=_clock(),
        )
    recovered = StateStore(config.state_dir).load()
    assert recovered.documents[0].phase is DocumentPhase.BUNDLED
    before = _snapshot(config.batches_dir)

    resume = ApiWorld(plans)
    bundle_selected_batch(
        config,
        resume.api_client(),
        StateStore(config.state_dir),
        source_client=resume.source_client(),
        clock=_clock(),
    )
    assert resume.bundle_ids == []
    assert resume.source_urls == []
    assert _snapshot(config.batches_dir) == before


def test_second_run_verifies_bundled_documents_without_downloading(
    tmp_path: Path,
) -> None:
    plans = {4: _plan(4), 8: _plan(8)}
    _result, store, world, config = _run(tmp_path, plans, exhausted=True)
    before = _snapshot(config.batches_dir)
    again = ApiWorld(plans)
    result = bundle_selected_batch(
        config,
        again.api_client(),
        store,
        source_client=again.source_client(),
        clock=_clock(),
    )

    assert result.completed is True
    assert again.bundle_ids == []
    assert again.source_urls == []
    assert _snapshot(config.batches_dir) == before
    assert len(world.source_urls) == 2

    text_path = (
        document_dir(config.batches_dir, "batch-000001", 4)
        / "transcription"
        / "transcription.txt"
    )
    text_path.write_bytes(b"tampered")
    blocked = ApiWorld(plans)
    with pytest.raises(BundleVerificationError):
        bundle_selected_batch(
            config,
            blocked.api_client(),
            store,
            source_client=blocked.source_client(),
            clock=_clock(),
        )
    assert blocked.source_urls == []
    assert text_path.read_bytes() == b"tampered"


def test_conflicting_final_artifact_fails_closed(tmp_path: Path) -> None:
    plans = {4: _plan(4)}
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, plans, exhausted=True)
    root = _materialize(
        config.batches_dir,
        document_id=4,
        archive_item_id=9999,
        language="en",
        text="other",
        source=b"%PDF-1.7\nother\n",
    )
    before = _snapshot(root)
    world = ApiWorld(plans)
    with pytest.raises(BundleConflictError, match="does not match"):
        bundle_selected_batch(
            config,
            world.api_client(),
            store,
            source_client=world.source_client(),
            clock=_clock(),
        )

    assert world.bundle_ids == []
    assert world.source_urls == []
    assert _snapshot(root) == before
    assert "document_bundled" not in {entry.event_type for entry in store.read_ledger()}


def test_exact_existing_artifact_is_accepted_without_rewrite_or_download(
    tmp_path: Path,
) -> None:
    plans = {4: _plan(4, text="already-local\n")}
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, plans, exhausted=True)
    root = _materialize(
        config.batches_dir,
        document_id=4,
        archive_item_id=1004,
        language="en",
        text="already-local\n",
        source=b"%PDF-1.7\nkept\n",
    )
    before = _snapshot(root)
    world = ApiWorld(plans)
    result = bundle_selected_batch(
        config,
        world.api_client(),
        store,
        source_client=world.source_client(),
        clock=_clock(),
    )
    state = store.load()

    assert result.completed is True
    assert world.bundle_ids == []
    assert world.source_urls == []
    assert _snapshot(root) == before
    assert state.documents[0].phase is DocumentPhase.BUNDLED
    assert state.documents[0].transcription_sha256 == sha256_text("already-local\n")
    verify_document_bundle(root)
    _assert_api_metadata_has_no_download_credentials(root)


def test_resume_rejects_final_bundle_that_persists_a_presigned_url(
    tmp_path: Path,
) -> None:
    plans = {4: _plan(4, text="already-local\n")}
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, plans, exhausted=True)
    root = _materialize(
        config.batches_dir,
        document_id=4,
        archive_item_id=1004,
        language="en",
        text="already-local\n",
        source=b"%PDF-1.7\nkept\n",
        api=_poisoned_api(4),
    )
    before = _snapshot(root)
    world = ApiWorld(plans)

    with pytest.raises(BundleConflictError, match="presigned URL"):
        bundle_selected_batch(
            config,
            world.api_client(),
            store,
            source_client=world.source_client(),
            clock=_clock(),
        )

    assert world.bundle_ids == []
    assert world.source_urls == []
    assert _snapshot(root) == before
    assert "document_bundled" not in {entry.event_type for entry in store.read_ledger()}
    assert store.load().documents[0].phase is DocumentPhase.SELECTED


def test_resume_rejects_staging_bundle_that_persists_a_presigned_url(
    tmp_path: Path,
) -> None:
    plans = {4: _plan(4, text="already-local\n")}
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, plans, exhausted=True)
    staging = _staging_document_dir(config.batches_dir, "batch-000001", 4)
    root = _materialize(
        config.batches_dir,
        document_id=4,
        archive_item_id=1004,
        language="en",
        text="already-local\n",
        source=b"%PDF-1.7\nkept\n",
        root=staging,
        api=_poisoned_api(4),
    )
    before = _snapshot(root)
    world = ApiWorld(plans)

    with pytest.raises(BundleConflictError, match="presigned URL"):
        bundle_selected_batch(
            config,
            world.api_client(),
            store,
            source_client=world.source_client(),
            clock=_clock(),
        )

    assert world.bundle_ids == []
    assert world.source_urls == []
    assert _snapshot(root) == before
    assert root.is_dir()
    assert "document_bundled" not in {entry.event_type for entry in store.read_ledger()}
    assert not document_dir(config.batches_dir, "batch-000001", 4).exists()


def test_new_bundle_api_metadata_has_no_download_credentials(tmp_path: Path) -> None:
    plans = {4: _plan(4, text="keep\nthis\n")}
    _result, _store, _world, config = _run(tmp_path, plans, exhausted=True)
    root = document_dir(config.batches_dir, "batch-000001", 4)
    _assert_api_metadata_has_no_download_credentials(root)
    verify_document_bundle(root)


def test_bundled_checkpoint_rejects_poisoned_api_metadata(tmp_path: Path) -> None:
    plans = {4: _plan(4)}
    result, store, _world, config = _run(tmp_path, plans, exhausted=True)
    assert result.completed is True
    root = document_dir(config.batches_dir, "batch-000001", 4)
    poisoned = _poisoned_api(4)
    _retarget_api_payload(root, poisoned)
    before = _snapshot(root)
    before_events = [entry.event_type for entry in store.read_ledger()]
    verify_document_bundle(root)
    world = ApiWorld(plans)

    with pytest.raises(BundleConflictError, match="presigned URL"):
        bundle_selected_batch(
            config,
            world.api_client(),
            store,
            source_client=world.source_client(),
            clock=_clock(),
        )

    assert world.bundle_ids == []
    assert world.source_urls == []
    assert _snapshot(root) == before
    assert store.load().documents[0].phase is DocumentPhase.BUNDLED
    assert [entry.event_type for entry in store.read_ledger()] == before_events


def test_only_expected_state_and_document_files_are_written(tmp_path: Path) -> None:
    plans = {4: _plan(4)}
    _result, _store, _world, _config_obj = _run(tmp_path, plans, exhausted=True)
    files = _files(tmp_path)
    assert files == {
        "state/controller_state.json",
        "state/ledger.jsonl",
        "batches/batch-000001/documents/4/document_manifest.json",
        "batches/batch-000001/documents/4/api/review_bundle.json",
        "batches/batch-000001/documents/4/transcription/transcription.txt",
        "batches/batch-000001/documents/4/sources/0000_primary_file.pdf",
    }
    _assert_no_secret(tmp_path)


def test_default_source_client_disables_redirects_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcription_review_agent.bundling.acquire as acquire

    captured: dict[str, object] = {}
    real_client = httpx.Client

    def factory(**kwargs: object) -> httpx.Client:
        if "transport" in kwargs:
            return real_client(**kwargs)  # type: ignore[arg-type]
        captured.update(kwargs)
        return real_client(
            transport=httpx.MockTransport(ApiWorld({4: _plan(4)})._source),
            follow_redirects=bool(kwargs["follow_redirects"]),
            trust_env=bool(kwargs["trust_env"]),
            timeout=kwargs["timeout"],  # type: ignore[arg-type]
        )

    monkeypatch.setattr(acquire.httpx, "Client", factory)
    plans = {4: _plan(4)}
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, plans, exhausted=True)
    bundle_selected_batch(config, ApiWorld(plans).api_client(), store, clock=_clock())

    assert captured["follow_redirects"] is False
    assert captured["trust_env"] is False
    assert "auth" not in captured
    source = Path(acquire.__file__).read_text(encoding="utf-8")
    for banned in ("openai", "anthropic", AI_HOST, "seal_batch", "raster"):
        assert banned not in source


def test_review_next_remains_unimplemented() -> None:
    from transcription_review_agent.controller.run import review_next

    with pytest.raises(NotImplementedError):
        review_next(None)


def test_result_json_has_no_secret(tmp_path: Path) -> None:
    plans = {4: _plan(4)}
    result, _store, _world, _config_obj = _run(tmp_path, plans, exhausted=True)
    rendered = json.dumps(
        {
            "completed": result.completed,
            "batch_status": result.batch_status.value,
            "batch_id": result.batch_id,
            "document_ids": result.document_ids,
        }
    )
    assert SECRET not in rendered
