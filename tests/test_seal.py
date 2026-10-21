"""Atomic batch sealing. No network and no AI provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import transcription_review_agent.bundling.seal as seal_module

from transcription_review_agent.bundling.hashing import sha256_bytes, sha256_text
from transcription_review_agent.bundling.layout import batch_manifest_path, document_dir
from transcription_review_agent.bundling.manifest import (
    BundleFile,
    DocumentBundleManifest,
    LocalProvenance,
    SourceArtifact,
    TranscriptionArtifact,
    publish_document_manifest,
)
from transcription_review_agent.bundling.seal import (
    _publish_new_manifest,
    seal_selected_batch,
)
from transcription_review_agent.bundling.verify import verify_batch_bundle
from transcription_review_agent.config import AgentConfig
from transcription_review_agent.errors import (
    BundleConflictError,
    BundleVerificationError,
    ImmutableBundleError,
    StateInvariantError,
)
from transcription_review_agent.state.schema import (
    BatchStatus,
    ControllerState,
    DocumentPhase,
    initial_state,
)
from transcription_review_agent.state.store import StateStore
from transcription_review_agent.state.transitions import (
    begin_bundling,
    mark_bundled,
    mark_bundling,
    note_discovery_exhausted,
    select_reviewable,
)
from support import at, source_record, text_digest

BASE_URL = "http://127.0.0.1:8000"


def _config(tmp_path: Path, *, batch_size: int = 5) -> AgentConfig:
    return AgentConfig(
        api_base_url=BASE_URL,
        batch_size=batch_size,
        state_dir=tmp_path / "state",
        batches_dir=tmp_path / "batches",
        request_timeout_seconds=5,
    )


def _clock(start: int = 100):
    instant = {"n": start}

    def now():
        instant["n"] += 1
        return at(instant["n"])

    return now


def _write_document(
    batches_dir: Path,
    document_id: int,
    *,
    language: str,
    batch_id: str = "batch-000001",
) -> None:
    text = f"text-{document_id}"
    text_bytes = text.encode("utf-8")
    source = f"page-{document_id}".encode()
    api = b'{"outcome":"REVIEWABLE"}\n'
    source_rel = "sources/0000_primary_file.jpg"
    root = document_dir(batches_dir, batch_id, document_id)
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
        archive_item_id=5000 + document_id,
        language=language,  # type: ignore[arg-type]
        batch_id=batch_id,
        created_at=at(1),
        transcription=TranscriptionArtifact(
            relative_path="transcription/transcription.txt",
            sha256=text_hash,
            size_bytes=len(text_bytes),
            text_result_id=document_id,
            result_type="SOURCE_TEXT",
            status="NEEDS_REVIEW",
            verification_status="UNVERIFIED",
        ),
        sources=[
            SourceArtifact(
                order_index=0,
                source_kind="PRIMARY_FILE",
                mime_type="image/jpeg",
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


def _seed(
    store: StateStore,
    batches_dir: Path,
    documents: list[tuple[int, str]],
    *,
    target: int,
    exhausted: bool,
    bundle_through: int | None = None,
) -> ControllerState:
    state = initial_state()
    for index, (document_id, language) in enumerate(documents, start=1):
        state = store.commit(
            select_reviewable(
                state,
                document_id=document_id,
                archive_item_id=5000 + document_id,
                language=language,  # type: ignore[arg-type]
                occurred_at=at(index),
                target_reviewable_count=target if index == 1 else None,
            )
        )
    if exhausted:
        state = store.commit(note_discovery_exhausted(state, occurred_at=at(30)))
    state = store.commit(begin_bundling(state, occurred_at=at(31)))
    limit = len(documents) if bundle_through is None else bundle_through
    for offset, (document_id, language) in enumerate(documents[:limit]):
        state = store.commit(
            mark_bundling(state, document_id=document_id, occurred_at=at(40 + offset))
        )
        _write_document(batches_dir, document_id, language=language)
        state = store.commit(
            mark_bundled(
                state,
                document_id=document_id,
                transcription_sha256=text_digest(f"text-{document_id}"),
                sources=[source_record(f"page-{document_id}".encode())],
                bundle_manifest_path=f"documents/{document_id}/document_manifest.json",
                occurred_at=at(50 + offset),
            )
        )
    return state


def _seal_count(store: StateStore) -> int:
    return sum(1 for entry in store.read_ledger() if entry.event_type == "batch_sealed")


def _manifest_temps(manifest_path: Path) -> list[Path]:
    prefix = f".{manifest_path.name}."
    fixed = f".{manifest_path.name}.tmp"
    return [
        candidate
        for candidate in manifest_path.parent.iterdir()
        if candidate.is_file()
        and (
            candidate.name == fixed
            or (candidate.name.startswith(prefix) and candidate.name.endswith(".tmp"))
        )
    ]


def test_short_batch_seals_with_document_manifest_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path, batch_size=1)
    store = StateStore(config.state_dir)
    _seed(
        store,
        config.batches_dir,
        [(7, "en"), (9, "fr")],
        target=5,
        exhausted=True,
    )
    result = seal_selected_batch(config, store, clock=_clock())
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    verified = verify_batch_bundle(manifest_path.parent)
    document_manifest = (
        document_dir(config.batches_dir, "batch-000001", 9) / "document_manifest.json"
    )

    assert result.batch_status is BatchStatus.SEALED
    assert result.document_ids == (7, 9)
    assert result.short_batch is True
    assert result.discovery_exhausted is True
    assert result.target_reviewable_count == 5
    assert result.manifest_sha256 == sha256_bytes(manifest_path.read_bytes())
    assert verified.documents[1].manifest_sha256 == sha256_bytes(document_manifest.read_bytes())
    assert verified.documents[1].document_id == 9
    assert store.load().batch_status is BatchStatus.SEALED
    assert _seal_count(store) == 1
    assert _manifest_temps(manifest_path) == []


def test_configured_full_and_short_targets_are_recorded_on_the_manifest(
    tmp_path: Path,
) -> None:
    full_root = tmp_path / "full"
    full = _config(full_root, batch_size=9)
    full_store = StateStore(full.state_dir)
    _seed(full_store, full.batches_dir, [(1, "en"), (2, "en")], target=2, exhausted=False)
    full_result = seal_selected_batch(full, full_store, clock=_clock())
    full_manifest = json.loads(
        batch_manifest_path(full.batches_dir, "batch-000001").read_text(encoding="utf-8")
    )
    assert full_result.short_batch is False
    assert full_result.discovery_exhausted is False
    assert full_manifest["target_reviewable_count"] == 2
    assert full_manifest["document_ids"] == [1, 2]

    short_root = tmp_path / "short"
    short = _config(short_root, batch_size=2)
    short_store = StateStore(short.state_dir)
    _seed(short_store, short.batches_dir, [(4, "en")], target=4, exhausted=True)
    short_result = seal_selected_batch(short, short_store, clock=_clock())
    assert short_result.target_reviewable_count == 4
    assert short_result.short_batch is True
    assert short_result.document_ids == (4,)


def test_batch_of_one_seals_as_a_full_batch(tmp_path: Path) -> None:
    config = _config(tmp_path, batch_size=5)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(6, "fr")], target=1, exhausted=False)
    result = seal_selected_batch(config, store, clock=_clock())
    assert result.target_reviewable_count == 1
    assert result.short_batch is False
    assert result.discovery_exhausted is False
    assert result.document_ids == (6,)


def test_repeated_seal_does_not_append_or_rewrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(3, "en")], target=5, exhausted=True)
    first = seal_selected_batch(config, store, clock=_clock())
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    before = manifest_path.read_bytes()
    ledger_before = store.ledger_path.read_bytes()

    second = seal_selected_batch(config, store, clock=_clock(200))

    assert second.manifest_sha256 == first.manifest_sha256
    assert manifest_path.read_bytes() == before
    assert store.ledger_path.read_bytes() == ledger_before
    assert _seal_count(store) == 1


def test_incomplete_batch_is_not_sealed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    state = _seed(
        store,
        config.batches_dir,
        [(1, "en"), (2, "en")],
        target=2,
        exhausted=False,
        bundle_through=1,
    )
    assert state.documents[1].phase is DocumentPhase.SELECTED
    ledger_before = store.ledger_path.read_bytes()

    with pytest.raises(StateInvariantError, match="bundled before seal"):
        seal_selected_batch(config, store, clock=_clock())

    assert not batch_manifest_path(config.batches_dir, "batch-000001").exists()
    assert store.ledger_path.read_bytes() == ledger_before
    assert store.load().batch_status is BatchStatus.BUNDLING


def test_extra_document_directory_is_not_sealed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(4, "en")], target=5, exhausted=True)
    extra = document_dir(config.batches_dir, "batch-000001", 99)
    extra.mkdir(parents=True)
    ledger_before = store.ledger_path.read_bytes()

    with pytest.raises(BundleConflictError, match="do not match the selected batch"):
        seal_selected_batch(config, store, clock=_clock())

    assert extra.is_dir()
    assert not batch_manifest_path(config.batches_dir, "batch-000001").exists()
    assert store.ledger_path.read_bytes() == ledger_before
    assert store.load().batch_status is BatchStatus.BUNDLING


def test_corrupt_document_bundle_is_not_sealed_or_repaired(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(4, "en"), (8, "en")], target=5, exhausted=True)
    text_path = (
        document_dir(config.batches_dir, "batch-000001", 4)
        / "transcription"
        / "transcription.txt"
    )
    text_path.write_bytes(b"tampered")
    ledger_before = store.ledger_path.read_bytes()

    with pytest.raises(BundleVerificationError, match="transcription bytes do not match"):
        seal_selected_batch(config, store, clock=_clock())

    assert text_path.read_bytes() == b"tampered"
    assert not batch_manifest_path(config.batches_dir, "batch-000001").exists()
    assert store.ledger_path.read_bytes() == ledger_before
    assert store.load().batch_status is BatchStatus.BUNDLING


def test_crash_before_ledger_seals_the_existing_manifest_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(2, "en")], target=5, exhausted=True)

    class CrashBeforeLedger(StateStore):
        def commit(self, entry):
            if entry.event_type == "batch_sealed":
                raise RuntimeError("crash before ledger")
            return super().commit(entry)

    with pytest.raises(RuntimeError, match="crash before ledger"):
        seal_selected_batch(config, CrashBeforeLedger(config.state_dir), clock=_clock())
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    published = manifest_path.read_bytes()
    assert _seal_count(store) == 0
    assert store.load().batch_status is BatchStatus.BUNDLING

    result = seal_selected_batch(config, store, clock=_clock(300))

    assert manifest_path.read_bytes() == published
    assert result.manifest_sha256 == sha256_bytes(published)
    assert _seal_count(store) == 1
    assert store.load().batch_status is BatchStatus.SEALED


def test_manifest_that_does_not_match_is_not_replaced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(2, "en")], target=5, exhausted=True)

    class CrashBeforeLedger(StateStore):
        def commit(self, entry):
            if entry.event_type == "batch_sealed":
                raise RuntimeError("crash before ledger")
            return super().commit(entry)

    with pytest.raises(RuntimeError, match="crash before ledger"):
        seal_selected_batch(config, CrashBeforeLedger(config.state_dir), clock=_clock())
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["documents"][0]["manifest_sha256"] = "ab" * 32
    tampered = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(tampered)

    with pytest.raises(BundleVerificationError, match="document manifest bytes do not match"):
        seal_selected_batch(config, store, clock=_clock(400))

    assert manifest_path.read_bytes() == tampered
    assert _seal_count(store) == 0
    assert store.load().batch_status is BatchStatus.BUNDLING


def test_crash_after_ledger_append_does_not_seal_twice(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(5, "en")], target=1, exhausted=False)
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")

    class CrashBeforeCheckpoint(StateStore):
        def write_checkpoint(self, state: ControllerState) -> None:
            if state.batch_status is BatchStatus.SEALED:
                raise RuntimeError("crash before checkpoint")
            super().write_checkpoint(state)

    with pytest.raises(RuntimeError, match="crash before checkpoint"):
        seal_selected_batch(
            config,
            CrashBeforeCheckpoint(config.state_dir),
            clock=_clock(),
        )
    published = manifest_path.read_bytes()
    recovered = StateStore(config.state_dir).load()
    assert recovered.batch_status is BatchStatus.SEALED
    assert _seal_count(store) == 1

    again = seal_selected_batch(config, store, clock=_clock(500))

    assert again.batch_status is BatchStatus.SEALED
    assert manifest_path.read_bytes() == published
    assert _seal_count(store) == 1


def test_missing_manifest_of_a_sealed_batch_is_not_recreated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(3, "en")], target=5, exhausted=True)
    seal_selected_batch(config, store, clock=_clock())
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    manifest_path.unlink()
    ledger_before = store.ledger_path.read_bytes()

    with pytest.raises(BundleConflictError, match="missing its batch manifest"):
        seal_selected_batch(config, store, clock=_clock(600))

    assert not manifest_path.exists()
    assert store.ledger_path.read_bytes() == ledger_before
    assert store.load().batch_status is BatchStatus.SEALED


def test_short_os_writes_publish_the_complete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(3, "en")], target=5, exhausted=True)
    real_write = seal_module.os.write

    def short_write(descriptor: int, data: bytes | memoryview) -> int:
        view = bytes(data)
        if len(view) > 2:
            return real_write(descriptor, view[:2])
        return real_write(descriptor, view)

    monkeypatch.setattr(seal_module.os, "write", short_write)
    result = seal_selected_batch(config, store, clock=_clock())
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    verified = verify_batch_bundle(manifest_path.parent)

    assert verified.document_ids == [3]
    assert result.manifest_sha256 == sha256_bytes(manifest_path.read_bytes())
    assert _manifest_temps(manifest_path) == []


def test_interrupted_temp_write_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(3, "en")], target=5, exhausted=True)
    real_write = seal_module.os.write

    def crash_during_write(descriptor: int, data: bytes | memoryview) -> int:
        view = bytes(data)
        real_write(descriptor, view[:1] if view else view)
        raise OSError("crash during manifest write")

    monkeypatch.setattr(seal_module.os, "write", crash_during_write)
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    with pytest.raises(OSError, match="crash during manifest write"):
        seal_selected_batch(config, store, clock=_clock())

    assert not manifest_path.exists()
    assert _manifest_temps(manifest_path) == []
    assert _seal_count(store) == 0

    monkeypatch.undo()
    result = seal_selected_batch(config, store, clock=_clock(200))
    assert result.document_ids == (3,)
    assert manifest_path.is_file()
    verify_batch_bundle(manifest_path.parent)
    assert _seal_count(store) == 1


def test_publish_leaves_an_existing_manifest_untouched(tmp_path: Path) -> None:
    path = tmp_path / "batch_manifest.json"
    path.write_bytes(b"already-sealed")
    stale = tmp_path / ".batch_manifest.json.tmp"
    stale.write_bytes(b"partial")

    with pytest.raises(ImmutableBundleError):
        _publish_new_manifest(path, '{"replacement": true}\n')

    assert path.read_bytes() == b"already-sealed"
    assert stale.read_bytes() == b"partial"


def test_abandoned_temp_is_removed_only_after_a_verified_seal(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.state_dir)
    _seed(store, config.batches_dir, [(3, "en")], target=5, exhausted=True)
    manifest_path = batch_manifest_path(config.batches_dir, "batch-000001")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    abandoned = manifest_path.parent / ".batch_manifest.json.99.abcd.tmp"
    abandoned.write_bytes(b"not-a-manifest")
    fixed = manifest_path.parent / ".batch_manifest.json.tmp"
    fixed.write_bytes(b"partial-old-temp")

    seal_selected_batch(config, store, clock=_clock())

    verified = verify_batch_bundle(manifest_path.parent)
    assert verified.document_ids == [3]
    assert manifest_path.read_bytes()[:1] == b"{"
    assert not abandoned.exists()
    assert not fixed.exists()
