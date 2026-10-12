"""Discovery orchestration against a scripted read-only API. No live requests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx
import pytest

from transcription_review_agent.api_client.client import TranscriptionReviewApiClient
from transcription_review_agent.api_client.errors import (
    ApiInvalidJsonError,
    ApiProtocolError,
    ApiSchemaError,
    ApiTimeoutError,
    ApiTransportError,
    ScanQueryError,
    UnexpectedApiResponseError,
)
from transcription_review_agent.api_client.schemas import BlockingIntegrityCode
from transcription_review_agent.config import AgentConfig
from transcription_review_agent.constants import (
    BUNDLE_SCHEMA_VERSION,
    SCAN_PATH,
    SCAN_SCHEMA_VERSION,
)
from transcription_review_agent.controller.discovery import (
    DiscoveryResult,
    discover_next_batch,
)
from transcription_review_agent.errors import StateInvariantError
from transcription_review_agent.state.ledger import (
    DocumentNonEligibleEntry,
    DocumentNotFoundEntry,
    DocumentSelectedEntry,
    NoWorkEntry,
    RunBlockedEntry,
)
from transcription_review_agent.state.schema import BatchStatus, initial_state
from transcription_review_agent.state.store import StateStore
from transcription_review_agent.state.transitions import (
    begin_bundling,
    note_blocking,
    note_discovery_exhausted,
    select_reviewable,
)
from support import at

BASE_URL = "http://127.0.0.1:8000"
PRESIGN_SECRET = "PRESIGN-SECRET-DO-NOT-PERSIST"
TRANSCRIPTION_SECRET = "EXACT-TRANSCRIPTION-DO-NOT-PERSIST"
AI_HOST = "api.openai.com"

BundleKind = Literal[
    "reviewable",
    "non_eligible",
    "blocking",
    "not_found",
    "http",
    "timeout",
    "transport",
    "schema",
    "protocol",
    "invalid_json",
]
ScanFault = Literal["http", "timeout", "transport", "schema", "protocol", "query"]


@dataclass
class BundlePlan:
    kind: BundleKind
    code: str = "LANGUAGE_NOT_IN_SCOPE"
    message: str = "classified by the server"
    details: dict[str, object] | None = None
    archive_item_id: int = 900
    language: str = "en"
    status_code: int = 500


@dataclass
class ApiScript:
    """One scripted origin. Unknown scans and bundles fail the test."""

    pages: dict[int, list[int]]
    bundles: dict[int, list[BundlePlan]]
    limit: int = 50
    scan_faults: dict[int, ScanFault] = field(default_factory=dict)
    requests: list[httpx.Request] = field(default_factory=list)

    def client(self) -> TranscriptionReviewApiClient:
        return TranscriptionReviewApiClient(
            BASE_URL,
            5,
            transport=httpx.MockTransport(self.handle),
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method != "GET":
            raise AssertionError(f"unexpected method {request.method}")
        path = request.url.path
        if path == SCAN_PATH:
            return self._scan(request)
        marker = "/documents/"
        suffix = "/review-bundle"
        if marker in path and path.endswith(suffix):
            document_id = int(path.split(marker, 1)[1][: -len(suffix)])
            return self._bundle(document_id)
        raise AssertionError(f"unexpected path {path}")

    def _scan(self, request: httpx.Request) -> httpx.Response:
        after_id = int(request.url.params["after_id"])
        limit = int(request.url.params["limit"])
        if limit != self.limit:
            raise AssertionError(f"scan limit {limit} != {self.limit}")
        fault = self.scan_faults.get(after_id)
        if fault is not None:
            return _scan_fault_response(fault, after_id=after_id, limit=limit)
        if after_id not in self.pages:
            raise AssertionError(f"unexpected scan after_id={after_id}")
        return httpx.Response(
            200,
            json=_scan_body(
                after_id=after_id, limit=limit, document_ids=self.pages[after_id]
            ),
        )

    def _bundle(self, document_id: int) -> httpx.Response:
        planned = self.bundles.get(document_id)
        if not planned:
            raise AssertionError(f"unexpected review bundle {document_id}")
        plan = planned.pop(0)
        if plan.kind == "timeout":
            raise httpx.ReadTimeout("timed out")
        if plan.kind == "transport":
            raise httpx.ConnectError("connection refused")
        if plan.kind == "not_found":
            return httpx.Response(404, json={"detail": "missing"})
        if plan.kind == "http":
            return httpx.Response(plan.status_code, json={"detail": "nope"})
        if plan.kind == "invalid_json":
            return httpx.Response(200, content=b"not-json")
        if plan.kind == "schema":
            return httpx.Response(
                200,
                json={"schema_version": BUNDLE_SCHEMA_VERSION, "outcome": "REVIEWABLE"},
            )
        if plan.kind == "protocol":
            return httpx.Response(200, json=_reviewable_body(document_id + 1, plan))
        if plan.kind == "reviewable":
            return httpx.Response(200, json=_reviewable_body(document_id, plan))
        if plan.kind == "non_eligible":
            return httpx.Response(
                200,
                json=_classified_body(document_id, "NON_ELIGIBLE", plan),
            )
        if plan.kind == "blocking":
            return httpx.Response(
                200,
                json=_classified_body(document_id, "BLOCKING_INTEGRITY_ERROR", plan),
            )
        raise AssertionError(f"unknown bundle kind {plan.kind}")


def _scan_fault_response(
    fault: ScanFault, *, after_id: int, limit: int
) -> httpx.Response:
    if fault == "timeout":
        raise httpx.ReadTimeout("timed out")
    if fault == "transport":
        raise httpx.ConnectError("connection refused")
    if fault == "http":
        return httpx.Response(503, json={"detail": "unavailable"})
    if fault == "schema":
        return httpx.Response(200, json={"schema_version": SCAN_SCHEMA_VERSION})
    if fault == "protocol":
        body = _scan_body(after_id=after_id + 1, limit=limit, document_ids=[])
        body["after_id"] = after_id + 9
        body["exhausted"] = True
        return httpx.Response(200, json=body)
    if fault == "query":
        return httpx.Response(
            400,
            json={"error": "invalid_after_id", "detail": "after_id is not valid"},
        )
    raise AssertionError(f"unknown scan fault {fault}")


def _scan_body(
    *, after_id: int, limit: int, document_ids: list[int]
) -> dict[str, object]:
    if not document_ids:
        return {
            "schema_version": SCAN_SCHEMA_VERSION,
            "after_id": after_id,
            "limit": limit,
            "next_after_id": None,
            "exhausted": True,
            "items": [],
        }
    return {
        "schema_version": SCAN_SCHEMA_VERSION,
        "after_id": after_id,
        "limit": limit,
        "next_after_id": document_ids[-1],
        "exhausted": False,
        "items": [_candidate(document_id) for document_id in document_ids],
    }


def _candidate(document_id: int) -> dict[str, object]:
    return {
        "document_id": document_id,
        "archive_item_id": 1,
        "language": "en",
        "doc_type": "IMAGE",
        "item_type": "OCR_DOCUMENT",
        "upload_status": "UPLOADED",
        "processing_state_user": "READY",
        "visibility": "public",
        "expected_source_file_count": 1,
    }


def _reviewable_body(document_id: int, plan: BundlePlan) -> dict[str, object]:
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "outcome": "REVIEWABLE",
        "document": {
            "document_id": document_id,
            "archive_item_id": plan.archive_item_id,
            "language": plan.language,
            "doc_type": "IMAGE",
            "item_type": "OCR_DOCUMENT",
            "upload_status": "UPLOADED",
            "processing_state_user": "READY",
            "visibility": "public",
        },
        "transcription": {
            "text_result_id": 1,
            "result_type": "SOURCE_TEXT",
            "status": "NEEDS_REVIEW",
            "verification_status": "UNVERIFIED",
            "text": TRANSCRIPTION_SECRET,
        },
        "source_pages": [
            {
                "order_index": 0,
                "display_number": 1,
                "source_kind": "PRIMARY_FILE",
                "file_s3_key": "documents/source.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 12,
                "include_in_ocr": True,
                "upload_status": "UPLOADED",
                "download_url": f"https://files.example.test/{document_id}?X-Amz-Signature={PRESIGN_SECRET}",
                "download_url_expires_in": 3600,
            }
        ],
        "provenance": {
            "displayed_transcription_resolver": "resolve_displayed_transcription_result",
            "source_set": "primary_file",
            "presign_expires_in": 3600,
        },
    }


def _classified_body(
    document_id: int, outcome: str, plan: BundlePlan
) -> dict[str, object]:
    error: dict[str, object] = {"code": plan.code, "message": plan.message}
    if plan.details is not None:
        error["details"] = plan.details
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "outcome": outcome,
        "document": {
            "document_id": document_id,
            "archive_item_id": plan.archive_item_id,
            "language": "en",
            "doc_type": "IMAGE",
            "item_type": "OCR_DOCUMENT",
            "upload_status": "UPLOADED",
            "processing_state_user": "READY",
            "visibility": "public",
        },
        "error": error,
    }


def _config(tmp_path: Path, *, limit: int = 50) -> AgentConfig:
    return AgentConfig(
        api_base_url=BASE_URL,
        scan_page_limit=limit,
        state_dir=tmp_path / "state",
        batches_dir=tmp_path / "batches",
        request_timeout_seconds=5,
    )


def _clock():
    instant = {"n": 0}

    def now():
        instant["n"] += 1
        return at(instant["n"])

    return now


def _run(
    tmp_path: Path,
    script: ApiScript,
    *,
    limit: int | None = None,
    store: StateStore | None = None,
) -> tuple[DiscoveryResult, StateStore, ApiScript]:
    page_limit = script.limit if limit is None else limit
    config = _config(tmp_path, limit=page_limit)
    state_store = store if store is not None else StateStore(config.state_dir)
    result = discover_next_batch(
        config,
        script.client(),
        state_store,
        clock=_clock(),
    )
    return result, state_store, script


def _paths(requests: list[httpx.Request]) -> list[str]:
    return [request.url.path for request in requests]


def _scan_after_ids(requests: list[httpx.Request]) -> list[int]:
    return [
        int(request.url.params["after_id"])
        for request in requests
        if request.url.path == SCAN_PATH
    ]


def _bundle_ids(requests: list[httpx.Request]) -> list[int]:
    ids: list[int] = []
    for request in requests:
        path = request.url.path
        if path.endswith("/review-bundle"):
            ids.append(int(path.split("/documents/", 1)[1].split("/", 1)[0]))
    return ids


def _files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def _assert_no_secrets(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert PRESIGN_SECRET not in text
        assert TRANSCRIPTION_SECRET not in text
        assert "download_url" not in text
        assert AI_HOST not in text


def _assert_get_only_api(requests: list[httpx.Request]) -> None:
    assert requests
    for request in requests:
        assert request.method == "GET"
        assert request.url.host == "127.0.0.1"
        assert request.url.host != AI_HOST
        assert "files.example.test" not in str(request.url)
        assert request.url.path == SCAN_PATH or request.url.path.endswith(
            "/review-bundle"
        )


def test_skips_not_found_and_non_eligible_then_selects_reviewable(
    tmp_path: Path,
) -> None:
    script = ApiScript(
        pages={0: [4, 5, 6], 6: []},
        bundles={
            4: [BundlePlan("not_found")],
            5: [BundlePlan("non_eligible", code="NOT_OCR_DOCUMENT", message="not ocr")],
            6: [BundlePlan("reviewable", archive_item_id=777, language="fr")],
        },
    )
    result, store, script = _run(tmp_path, script)
    entries = store.read_ledger()

    assert _bundle_ids(script.requests) == [4, 5, 6]
    assert isinstance(entries[0], DocumentNotFoundEntry)
    assert entries[0].batch_id is None
    assert entries[0].document_id == 4
    assert isinstance(entries[1], DocumentNonEligibleEntry)
    assert entries[1].batch_id is None
    assert entries[1].code.value == "NOT_OCR_DOCUMENT"
    assert isinstance(entries[2], DocumentSelectedEntry)
    assert entries[2].archive_item_id == 777
    assert entries[2].language == "fr"
    assert result.selected_document_ids == (6,)
    assert result.current_batch_id == "batch-000001"
    state = store.load()
    assert state.last_scanned_document_id == 6
    assert state.documents[0].phase.value == "selected"
    assert state.documents[0].source_hashes == []
    assert state.documents[0].transcription_sha256 is None


def test_allocates_batch_only_on_first_reviewable(tmp_path: Path) -> None:
    script = ApiScript(
        pages={0: [1, 2, 3], 3: []},
        bundles={
            1: [BundlePlan("not_found")],
            2: [BundlePlan("reviewable", archive_item_id=20)],
            3: [BundlePlan("not_found")],
        },
    )
    result, store, _script = _run(tmp_path, script)
    entries = store.read_ledger()

    assert [type(entry) for entry in entries[:3]] == [
        DocumentNotFoundEntry,
        DocumentSelectedEntry,
        DocumentNotFoundEntry,
    ]
    assert entries[0].batch_id is None
    assert entries[1].batch_id == "batch-000001"
    assert entries[2].batch_id == "batch-000001"
    assert result.current_batch_id == "batch-000001"
    assert result.selected_document_ids == (2,)
    assert store.load().current_batch_document_ids == [2]
    assert all(entry.event_type != "batch_opened" for entry in entries)


def test_accumulates_exactly_five_reviewable_documents(tmp_path: Path) -> None:
    ids = [1, 2, 3, 4, 5, 6, 7]
    script = ApiScript(
        pages={0: ids},
        bundles={document_id: [BundlePlan("reviewable")] for document_id in ids},
    )
    result, store, script = _run(tmp_path, script)
    state = store.load()

    assert result.selected_document_ids == (1, 2, 3, 4, 5)
    assert result.batch_status is BatchStatus.ACCUMULATING
    assert result.discovery_exhausted is False
    assert result.blocked_document_id is None
    assert state.current_batch_id == "batch-000001"
    assert len(store.read_ledger()) == 5
    assert _bundle_ids(script.requests) == [1, 2, 3, 4, 5]
    assert _scan_after_ids(script.requests) == [0]


def test_does_not_fetch_later_candidates_after_the_fifth_selection(
    tmp_path: Path,
) -> None:
    script = ApiScript(
        pages={0: [10, 11, 12, 13, 14, 15, 16]},
        bundles={
            document_id: [BundlePlan("reviewable", archive_item_id=100 + document_id)]
            for document_id in range(10, 17)
        },
    )
    _result, store, script = _run(tmp_path, script)

    assert _bundle_ids(script.requests) == [10, 11, 12, 13, 14]
    assert 15 not in script.bundles or script.bundles[15]
    assert store.load().last_scanned_document_id == 14
    assert _scan_after_ids(script.requests) == [0]


def test_exhaustion_with_a_short_batch_records_discovery_exhausted(
    tmp_path: Path,
) -> None:
    script = ApiScript(
        pages={0: [2, 4], 4: []},
        bundles={
            2: [BundlePlan("reviewable")],
            4: [BundlePlan("non_eligible", code="NOT_READY_UPLOAD")],
        },
    )
    result, store, _script = _run(tmp_path, script)
    state = store.load()

    assert result.selected_document_ids == (2,)
    assert result.batch_status is BatchStatus.ACCUMULATING
    assert result.discovery_exhausted is True
    assert result.current_batch_id == "batch-000001"
    assert state.batch_status is BatchStatus.ACCUMULATING
    assert state.documents[0].phase.value == "selected"
    assert [entry.event_type for entry in store.read_ledger()] == [
        "document_selected",
        "document_non_eligible",
        "discovery_exhausted",
    ]
    assert _files(tmp_path / "batches") == set()


def test_exhaustion_with_zero_reviewable_documents_is_no_work(tmp_path: Path) -> None:
    script = ApiScript(
        pages={0: [3, 8], 8: []},
        bundles={
            3: [BundlePlan("not_found")],
            8: [BundlePlan("non_eligible", code="NOT_READY_PROCESSING")],
        },
    )
    result, store, _script = _run(tmp_path, script)
    state = store.load()
    no_work = store.read_ledger()[-1]

    assert isinstance(no_work, NoWorkEntry)
    assert no_work.batch_id is None
    assert result.batch_status is BatchStatus.NO_WORK
    assert result.current_batch_id is None
    assert result.selected_document_ids == ()
    assert result.discovery_exhausted is True
    assert result.blocked_document_id is None
    assert state.last_scanned_document_id == 8
    assert state.documents == []
    assert state.last_sealed_batch_id is None


def test_repeated_no_work_at_the_same_cursor_appends_nothing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    script = ApiScript(pages={0: []}, bundles={})
    first, _, first_script = _run(tmp_path, script, store=store)
    before_ledger = (config.state_dir / "ledger.jsonl").read_bytes()
    before_checkpoint = (config.state_dir / "controller_state.json").read_bytes()
    sequence = store.load().applied_ledger_sequence

    second_script = ApiScript(pages={0: []}, bundles={})
    second = discover_next_batch(config, second_script.client(), store, clock=_clock())

    assert first.batch_status is BatchStatus.NO_WORK
    assert second.batch_status is BatchStatus.NO_WORK
    assert second.current_batch_id is None
    assert store.load().applied_ledger_sequence == sequence
    assert len(store.read_ledger()) == 1
    assert (config.state_dir / "ledger.jsonl").read_bytes() == before_ledger
    assert (
        config.state_dir / "controller_state.json"
    ).read_bytes() == before_checkpoint
    assert _paths(second_script.requests) == [SCAN_PATH]
    assert _bundle_ids(second_script.requests) == []
    assert _scan_after_ids(first_script.requests) == [0]


def test_blocker_before_first_reviewable_holds_cursor_without_a_batch(
    tmp_path: Path,
) -> None:
    script = ApiScript(
        pages={0: [3, 8, 9]},
        bundles={
            3: [BundlePlan("not_found")],
            8: [
                BundlePlan(
                    "blocking",
                    code="MISSING_PRIMARY_SOURCE",
                    message="primary source is missing",
                    details={"order_index": 1},
                )
            ],
            9: [BundlePlan("reviewable")],
        },
    )
    result, store, script = _run(tmp_path, script)
    state = store.load()
    blocked = store.read_ledger()[-1]

    assert isinstance(blocked, RunBlockedEntry)
    assert blocked.batch_id is None
    assert blocked.cursor_before == 3
    assert blocked.cursor_after == 3
    assert result.batch_status is BatchStatus.BLOCKED
    assert result.current_batch_id is None
    assert result.selected_document_ids == ()
    assert result.blocked_document_id == 8
    assert result.discovery_exhausted is False
    assert state.last_scanned_document_id == 3
    assert state.blocked is not None
    assert state.blocked.code.value == "MISSING_PRIMARY_SOURCE"
    assert state.blocked.details == {"order_index": 1}
    assert _bundle_ids(script.requests) == [3, 8]


def test_blocker_after_selected_documents_preserves_the_batch(tmp_path: Path) -> None:
    script = ApiScript(
        pages={0: [2, 4, 6, 7]},
        bundles={
            2: [BundlePlan("reviewable", archive_item_id=22)],
            4: [BundlePlan("reviewable", archive_item_id=44)],
            6: [
                BundlePlan(
                    "blocking",
                    code="MISSING_SOURCE_FILE",
                    message="source file is missing",
                )
            ],
            7: [BundlePlan("reviewable")],
        },
    )
    result, store, script = _run(tmp_path, script)
    state = store.load()

    assert result.selected_document_ids == (2, 4)
    assert result.current_batch_id == "batch-000001"
    assert result.blocked_document_id == 6
    assert result.batch_status is BatchStatus.BLOCKED
    assert state.last_scanned_document_id == 4
    assert state.current_batch_document_ids == [2, 4]
    assert state.blocked is not None
    assert state.blocked.document_id == 6
    assert _bundle_ids(script.requests) == [2, 4, 6]
    blocked = store.read_ledger()[-1]
    assert isinstance(blocked, RunBlockedEntry)
    assert blocked.batch_id == "batch-000001"


def test_unchanged_blocker_retry_does_not_grow_ledger_or_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    plan = BundlePlan(
        "blocking",
        code="UNSUPPORTED_SOURCE_MIME",
        message="mime is unsupported",
        details={"mime_type": "image/tiff"},
    )
    first = ApiScript(
        pages={0: [11]},
        bundles={11: [plan]},
    )
    discover_next_batch(config, first.client(), store, clock=_clock())
    before_ledger = (config.state_dir / "ledger.jsonl").read_bytes()
    before_checkpoint = (config.state_dir / "controller_state.json").read_bytes()
    entries = len(store.read_ledger())

    retry = ApiScript(
        pages={0: [11, 12], 11: [12]},
        bundles={11: [plan], 12: [BundlePlan("reviewable")]},
    )
    result = discover_next_batch(config, retry.client(), store, clock=_clock())

    assert result.blocked_document_id == 11
    assert result.batch_status is BatchStatus.BLOCKED
    assert len(store.read_ledger()) == entries
    assert (config.state_dir / "ledger.jsonl").read_bytes() == before_ledger
    assert (
        config.state_dir / "controller_state.json"
    ).read_bytes() == before_checkpoint
    assert _paths(retry.requests) == [
        "/api/agent/v1/transcription-review/documents/11/review-bundle"
    ]


@pytest.mark.parametrize(
    ("kind", "expected_type"),
    [
        ("reviewable", DocumentSelectedEntry),
        ("non_eligible", DocumentNonEligibleEntry),
        ("not_found", DocumentNotFoundEntry),
    ],
)
def test_blocked_retry_resolves_and_continues(
    tmp_path: Path,
    kind: str,
    expected_type: type,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    opening = ApiScript(
        pages={0: [4, 10]},
        bundles={
            4: [BundlePlan("reviewable", archive_item_id=40)],
            10: [BundlePlan("blocking", code="MISSING_SOURCE_FILE", message="missing")],
        },
    )
    discover_next_batch(config, opening.client(), store, clock=_clock())
    assert store.load().batch_status is BatchStatus.BLOCKED

    if kind == "reviewable":
        resolved = BundlePlan("reviewable", archive_item_id=1000, language="fr")
    elif kind == "non_eligible":
        resolved = BundlePlan(
            "non_eligible", code="LANGUAGE_NOT_IN_SCOPE", message="skip"
        )
    else:
        resolved = BundlePlan("not_found")
    retry = ApiScript(
        pages={10: [12], 12: []},
        bundles={
            10: [resolved],
            12: [BundlePlan("reviewable", archive_item_id=1200)],
        },
    )
    result = discover_next_batch(config, retry.client(), store, clock=_clock())
    state = store.load()
    resolution = store.read_ledger()[2]

    assert isinstance(resolution, expected_type)
    assert resolution.document_id == 10
    assert resolution.batch_id == "batch-000001"
    assert state.blocked is None
    assert state.batch_status is BatchStatus.ACCUMULATING
    assert state.discovery_exhausted is True
    assert result.blocked_document_id is None
    assert _paths(retry.requests)[0].endswith("/documents/10/review-bundle")
    assert _scan_after_ids(retry.requests) == [10, 12]
    if kind == "reviewable":
        assert result.selected_document_ids == (4, 10, 12)
        assert state.last_scanned_document_id == 12
        assert isinstance(resolution, DocumentSelectedEntry)
        assert resolution.language == "fr"
        assert resolution.archive_item_id == 1000
    else:
        assert result.selected_document_ids == (4, 12)
        assert state.last_scanned_document_id == 12
        assert 10 not in state.current_batch_document_ids


def test_blocked_retry_before_a_batch_resolves_to_not_found_then_selects(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    opening = ApiScript(
        pages={0: [6]},
        bundles={
            6: [BundlePlan("blocking", code="SOURCE_URL_UNAVAILABLE", message="url")]
        },
    )
    first = discover_next_batch(config, opening.client(), store, clock=_clock())
    assert first.current_batch_id is None
    assert store.load().last_scanned_document_id is None

    retry = ApiScript(
        pages={6: [8], 8: []},
        bundles={
            6: [BundlePlan("not_found")],
            8: [BundlePlan("reviewable", archive_item_id=80)],
        },
    )
    result = discover_next_batch(config, retry.client(), store, clock=_clock())
    state = store.load()

    assert _paths(retry.requests)[0].endswith("/documents/6/review-bundle")
    assert _scan_after_ids(retry.requests) == [6, 8]
    assert result.selected_document_ids == (8,)
    assert result.current_batch_id == "batch-000001"
    assert result.discovery_exhausted is True
    assert state.last_scanned_document_id == 8
    assert isinstance(store.read_ledger()[1], DocumentNotFoundEntry)
    assert store.read_ledger()[1].batch_id is None
    assert isinstance(store.read_ledger()[2], DocumentSelectedEntry)


def test_changed_blocker_replaces_the_block_and_stops(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    opening = ApiScript(
        pages={0: [2, 9]},
        bundles={
            2: [BundlePlan("reviewable")],
            9: [
                BundlePlan(
                    "blocking", code="MISSING_SOURCE_FILE", message="first block"
                )
            ],
        },
    )
    discover_next_batch(config, opening.client(), store, clock=_clock())
    retry = ApiScript(
        pages={2: [9, 10]},
        bundles={
            9: [
                BundlePlan(
                    "blocking",
                    code="INCOMPLETE_MULTI_IMAGE_SOURCES",
                    message="counts differ",
                    details={"expected": 2, "actual": 1},
                )
            ],
            10: [BundlePlan("reviewable")],
        },
    )
    result = discover_next_batch(config, retry.client(), store, clock=_clock())
    state = store.load()
    replacement = store.read_ledger()[-1]

    assert isinstance(replacement, RunBlockedEntry)
    assert replacement.code.value == "INCOMPLETE_MULTI_IMAGE_SOURCES"
    assert replacement.batch_id == "batch-000001"
    assert replacement.cursor_after == 2
    assert result.blocked_document_id == 9
    assert result.selected_document_ids == (2,)
    assert result.batch_status is BatchStatus.BLOCKED
    assert state.last_scanned_document_id == 2
    assert state.blocked is not None
    assert state.blocked.message == "counts differ"
    assert _paths(retry.requests) == [
        "/api/agent/v1/transcription-review/documents/9/review-bundle"
    ]


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [
        ("http", UnexpectedApiResponseError),
        ("timeout", ApiTimeoutError),
        ("transport", ApiTransportError),
        ("schema", ApiSchemaError),
        ("protocol", ApiProtocolError),
        ("invalid_json", ApiInvalidJsonError),
    ],
)
def test_bundle_failure_does_not_advance_past_the_failed_document(
    tmp_path: Path,
    kind: BundleKind,
    error_type: type[Exception],
) -> None:
    script = ApiScript(
        pages={0: [4, 9]},
        bundles={
            4: [BundlePlan("not_found")],
            9: [BundlePlan(kind, status_code=502)],
        },
    )
    with pytest.raises(error_type):
        _run(tmp_path, script)
    store = StateStore(tmp_path / "state")
    state = store.load()
    assert state.last_scanned_document_id == 4
    assert [entry.document_id for entry in store.read_ledger()] == [4]
    assert state.current_batch_id is None


@pytest.mark.parametrize(
    "fault", ["http", "timeout", "transport", "schema", "protocol", "query"]
)
def test_scan_failure_does_not_advance_the_cursor(
    tmp_path: Path, fault: ScanFault
) -> None:
    script = ApiScript(
        pages={0: [4], 4: []},
        bundles={4: [BundlePlan("not_found")]},
        scan_faults={4: fault},
    )
    expected = {
        "http": UnexpectedApiResponseError,
        "timeout": ApiTimeoutError,
        "transport": ApiTransportError,
        "schema": ApiSchemaError,
        "protocol": ApiProtocolError,
        "query": ScanQueryError,
    }[fault]
    with pytest.raises(expected):
        _run(tmp_path, script)
    state = StateStore(tmp_path / "state").load()
    assert state.last_scanned_document_id == 4
    assert state.batch_status is BatchStatus.SCANNING


def test_opening_scan_failure_leaves_the_initial_checkpoint_unwritten(
    tmp_path: Path,
) -> None:
    script = ApiScript(pages={}, bundles={}, scan_faults={0: "http"})
    with pytest.raises(UnexpectedApiResponseError):
        _run(tmp_path, script)
    assert _files(tmp_path) == set()
    assert StateStore(tmp_path / "state").load() == initial_state()


def test_multi_page_scan_continues_in_order(tmp_path: Path) -> None:
    script = ApiScript(
        limit=2,
        pages={0: [1, 2], 2: [4, 5], 5: []},
        bundles={
            1: [BundlePlan("not_found")],
            2: [BundlePlan("non_eligible", code="NOT_OCR_DOCUMENT")],
            4: [BundlePlan("reviewable", archive_item_id=40, language="en")],
            5: [BundlePlan("reviewable", archive_item_id=50, language="fr")],
        },
    )
    result, store, script = _run(tmp_path, script, limit=2)
    state = store.load()

    assert _scan_after_ids(script.requests) == [0, 2, 5]
    assert _bundle_ids(script.requests) == [1, 2, 4, 5]
    assert all(
        int(request.url.params["limit"]) == 2
        for request in script.requests
        if request.url.path == SCAN_PATH
    )
    assert result.selected_document_ids == (4, 5)
    assert result.discovery_exhausted is True
    assert state.last_scanned_document_id == 5
    assert state.current_batch_id == "batch-000001"
    assert [entry.cursor_after for entry in store.read_ledger()[:4]] == [1, 2, 4, 5]


def test_durable_cursor_advances_only_from_document_outcomes(tmp_path: Path) -> None:
    script = ApiScript(
        pages={0: [1, 2, 3, 4, 5, 9]},
        bundles={
            document_id: [BundlePlan("reviewable", archive_item_id=100 + document_id)]
            for document_id in (1, 2, 3, 4, 5, 9)
        },
    )
    _result, store, script = _run(tmp_path, script)
    entries = store.read_ledger()

    assert [entry.cursor_after for entry in entries] == [1, 2, 3, 4, 5]
    assert entries[-1].cursor_before == 4
    assert store.load().last_scanned_document_id == 5
    assert _scan_after_ids(script.requests) == [0]
    assert _bundle_ids(script.requests) == [1, 2, 3, 4, 5]
    raw = (tmp_path / "state" / "ledger.jsonl").read_text(encoding="utf-8")
    assert '"cursor_after":9' not in raw.replace(" ", "")


def test_next_page_uses_next_after_id_without_writing_it_early(tmp_path: Path) -> None:
    script = ApiScript(
        limit=2,
        pages={0: [10, 30], 30: []},
        bundles={
            10: [BundlePlan("non_eligible", code="NOT_READY_UPLOAD")],
            30: [BundlePlan("reviewable", archive_item_id=300)],
        },
    )
    _result, store, script = _run(tmp_path, script, limit=2)
    entries = store.read_ledger()

    assert _scan_after_ids(script.requests) == [0, 30]
    assert [entry.cursor_after for entry in entries[:2]] == [10, 30]
    assert entries[0].cursor_after != 30
    assert store.load().last_scanned_document_id == 30


def test_no_source_downloads_ai_calls_or_payload_persistence(tmp_path: Path) -> None:
    script = ApiScript(
        pages={0: [7], 7: []},
        bundles={7: [BundlePlan("reviewable", archive_item_id=70)]},
    )
    result, _store, script = _run(tmp_path, script)

    _assert_get_only_api(script.requests)
    assert _files(tmp_path) == {"state/controller_state.json", "state/ledger.jsonl"}
    _assert_no_secrets(tmp_path)
    rendered = json.dumps(
        {
            "selected_document_ids": result.selected_document_ids,
            "batch_status": result.batch_status.value,
            "discovery_exhausted": result.discovery_exhausted,
            "blocked_document_id": result.blocked_document_id,
            "current_batch_id": result.current_batch_id,
        }
    )
    assert PRESIGN_SECRET not in rendered
    assert TRANSCRIPTION_SECRET not in rendered
    source = Path(
        "/home/yshif/vs-archive/vs-archive-transcription-review-agent/src/transcription_review_agent/controller/discovery.py"
    ).read_text(encoding="utf-8")
    for banned in ("openai", "anthropic", "download_url", AI_HOST):
        assert banned not in source


def test_crash_after_ledger_append_replays_through_the_existing_store(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class CrashAfterAppend(StateStore):
        def __init__(self, state_dir: Path) -> None:
            super().__init__(state_dir)
            self.commits = 0

        def write_checkpoint(self, state) -> None:
            self.commits += 1
            if self.commits == 2:
                raise RuntimeError("crash before checkpoint")
            super().write_checkpoint(state)

    store = CrashAfterAppend(config.state_dir)
    script = ApiScript(
        pages={0: [1, 2], 2: []},
        bundles={
            1: [BundlePlan("not_found")],
            2: [BundlePlan("reviewable", archive_item_id=22, language="fr")],
        },
    )
    with pytest.raises(RuntimeError, match="crash before checkpoint"):
        discover_next_batch(config, script.client(), store, clock=_clock())

    recovered = StateStore(config.state_dir).load()
    assert recovered.last_scanned_document_id == 2
    assert recovered.current_batch_id == "batch-000001"
    assert recovered.current_batch_document_ids == [2]
    assert recovered.applied_ledger_sequence == 2
    assert _files(tmp_path) == {"state/controller_state.json", "state/ledger.jsonl"}


def test_existing_open_batch_fills_to_five_without_a_new_batch_id(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    state = initial_state()
    for document_id in (1, 2, 3):
        state = store.commit(
            select_reviewable(
                state,
                document_id=document_id,
                archive_item_id=100 + document_id,
                language="en",
                occurred_at=at(document_id),
            )
        )
    script = ApiScript(
        pages={3: [4, 5, 6]},
        bundles={
            document_id: [BundlePlan("reviewable", archive_item_id=100 + document_id)]
            for document_id in (4, 5, 6)
        },
    )
    result = discover_next_batch(config, script.client(), store, clock=_clock())

    assert result.selected_document_ids == (1, 2, 3, 4, 5)
    assert result.current_batch_id == "batch-000001"
    assert result.discovery_exhausted is False
    assert _bundle_ids(script.requests) == [4, 5]
    assert _scan_after_ids(script.requests) == [3]
    assert store.load().last_scanned_document_id == 5


def test_reviewable_resolution_that_fills_the_batch_does_not_scan_onward(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    state = initial_state()
    for document_id in (1, 2, 3, 4):
        state = store.commit(
            select_reviewable(
                state,
                document_id=document_id,
                archive_item_id=100 + document_id,
                language="en",
                occurred_at=at(document_id),
            )
        )
    planned = note_blocking(
        state,
        document_id=8,
        code=BlockingIntegrityCode.MISSING_SOURCE_FILE,
        message="blocked fifth slot",
        occurred_at=at(5),
    )
    assert planned is not None
    state = store.commit(planned)
    script = ApiScript(
        pages={8: [9]},
        bundles={
            8: [BundlePlan("reviewable", archive_item_id=800)],
            9: [BundlePlan("reviewable")],
        },
    )
    result = discover_next_batch(config, script.client(), store, clock=_clock())

    assert result.selected_document_ids == (1, 2, 3, 4, 8)
    assert result.batch_status is BatchStatus.ACCUMULATING
    assert result.blocked_document_id is None
    assert result.discovery_exhausted is False
    assert _paths(script.requests) == [
        "/api/agent/v1/transcription-review/documents/8/review-bundle"
    ]


def test_discovery_does_not_run_while_bundling(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    state = store.commit(
        select_reviewable(
            initial_state(),
            document_id=1,
            archive_item_id=10,
            language="en",
            occurred_at=at(1),
        )
    )
    state = store.commit(note_discovery_exhausted(state, occurred_at=at(2)))
    store.commit(begin_bundling(state, occurred_at=at(3)))
    script = ApiScript(pages={1: [2]}, bundles={2: [BundlePlan("reviewable")]})
    before = len(store.read_ledger())

    with pytest.raises(StateInvariantError):
        discover_next_batch(config, script.client(), store, clock=_clock())

    assert script.requests == []
    assert len(store.read_ledger()) == before


def test_exhausted_open_batch_is_left_unchanged(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    state = store.commit(
        select_reviewable(
            initial_state(),
            document_id=4,
            archive_item_id=40,
            language="en",
            occurred_at=at(1),
        )
    )
    store.commit(note_discovery_exhausted(state, occurred_at=at(2)))
    before_ledger = (config.state_dir / "ledger.jsonl").read_bytes()
    before_checkpoint = (config.state_dir / "controller_state.json").read_bytes()
    script = ApiScript(pages={4: [8]}, bundles={8: [BundlePlan("reviewable")]})

    result = discover_next_batch(config, script.client(), store, clock=_clock())

    assert result.selected_document_ids == (4,)
    assert result.discovery_exhausted is True
    assert result.batch_status is BatchStatus.ACCUMULATING
    assert script.requests == []
    assert (config.state_dir / "ledger.jsonl").read_bytes() == before_ledger
    assert (
        config.state_dir / "controller_state.json"
    ).read_bytes() == before_checkpoint


def test_review_next_remains_unimplemented() -> None:
    from transcription_review_agent.controller.run import review_next

    with pytest.raises(NotImplementedError):
        review_next(None)
