"""Fixture-only checks for sealed original-image staging."""

from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.layout import document_dir
from transcription_review_agent.bundling.manifest import (
    BundleFile,
    DocumentBundleManifest,
    LocalProvenance,
    SourceArtifact,
    TranscriptionArtifact,
    publish_document_manifest,
)
from transcription_review_agent.errors import BundleVerificationError
from transcription_review_agent.review import original_image_stage
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
    ReviewPolicyError,
)
from transcription_review_agent.review.jpegutil import (
    encode_grayscale_dc_jpeg,
    jpeg_to_png,
)
from transcription_review_agent.review.models import (
    FORBIDDEN_INPUTS,
    PROVIDER_OUTPUT_SCHEMA_VERSION,
    WorkRequest,
    interface_for,
)
from transcription_review_agent.review.original_image_stage import (
    StageExecution,
    codex_staged_image,
    grok_staged_image,
    load_staged_original,
    resolve_provider_file,
    stage_sealed_document,
    stage_sealed_original,
)
from transcription_review_agent.review.paths import output_relative_path
from transcription_review_agent.review.pngutil import encode_png

BATCH_ID = "batch-000001"
DOCUMENT_ID = 7
TRANSCRIPTION = "The parish register names Henriette Duval on 3 May 1841."
S3_KEY = "s3://secret-bucket/private/page-0"
PRESIGNED = (
    b'{"download_url":"https://files.example.invalid/page'
    b'?X-Amz-Signature=secretvalue&X-Amz-Credential=ASIAEXAMPLE"}\n'
)
GPT_READING = "GPT-READING-SECRET"
GROK_READING = "GROK-READING-SECRET"
CREATED = datetime(2026, 11, 15, tzinfo=timezone.utc)
CODEX = (
    StageExecution("gpt", "codex_cli"),
    StageExecution("grok", "grok_composer"),
)
GPT_WORK = (
    StageExecution("gpt", "gpt_work"),
    StageExecution("grok", "grok_composer"),
)


def _work_requests(document_hash: str) -> tuple[WorkRequest, WorkRequest]:
    requests: list[WorkRequest] = []
    for provider in ("gpt", "grok"):
        task_id = f"{BATCH_ID}:{DOCUMENT_ID}:first_pass:r0:{provider}:a1"
        requests.append(
            WorkRequest(
                provider=provider,  # type: ignore[arg-type]
                provider_interface=interface_for(provider),
                task_id=task_id,
                job_id=f"job:{task_id}",
                batch_id=BATCH_ID,
                document_id=DOCUMENT_ID,
                stage="first_pass",
                round_index=0,
                attempt=1,
                source_bundle_sha256=document_hash,
                output_relative_path=output_relative_path("first_pass", provider, 0, 1),
                page_ids=["page_001"],
                forbidden_inputs=list(FORBIDDEN_INPUTS),
            )
        )
    return (requests[0], requests[1])


def _approved(batches: Path) -> tuple[WorkRequest, WorkRequest]:
    sealed = document_dir(batches, BATCH_ID, DOCUMENT_ID)
    digest = sha256_bytes((sealed / "document_manifest.json").read_bytes())
    return _work_requests(digest)


def _codex_root(stage_root: Path, executions: tuple[StageExecution, StageExecution]):
    if any(item.execution_mechanism == "codex_cli" for item in executions):
        return stage_root.parent / f"{stage_root.name}-codex"
    return None


def stage_call(**kwargs):
    executions = kwargs.get("executions", CODEX)
    stage_root = kwargs["stage_root"]
    kwargs.setdefault("work_requests", _approved(kwargs["batches_dir"]))
    kwargs.setdefault("codex_work_root", _codex_root(stage_root, executions))
    return stage_sealed_document(**kwargs)


def _png() -> bytes:
    return encode_png(2, 2, [bytearray(b"\x10\x20"), bytearray(b"\x30\x40")], 1)


def _jpeg() -> bytes:
    return encode_grayscale_dc_jpeg(8, 8, [0])


def _seal(
    batches: Path,
    image: bytes,
    *,
    mime: str = "image/png",
    transcription: str = TRANSCRIPTION,
    sources: list[tuple[bytes, str]] | None = None,
) -> Path:
    pages = sources if sources is not None else [(image, mime)]
    text = transcription.encode("utf-8")
    root = document_dir(batches, BATCH_ID, DOCUMENT_ID)
    payloads: dict[str, bytes] = {
        "api/review_bundle.json": PRESIGNED,
        "transcription/transcription.txt": text,
    }
    artifacts: list[SourceArtifact] = []
    files = [
        BundleFile(
            relative_path="api/review_bundle.json",
            sha256=sha256_bytes(PRESIGNED),
            size_bytes=len(PRESIGNED),
            role="api_review_bundle",
        ),
        BundleFile(
            relative_path="transcription/transcription.txt",
            sha256=sha256_bytes(text),
            size_bytes=len(text),
            role="transcription_text",
        ),
    ]
    for index, (payload, page_mime) in enumerate(pages):
        suffix = ".png" if page_mime == "image/png" else ".bin"
        if page_mime == "image/jpeg":
            suffix = ".jpg"
        elif page_mime == "image/webp":
            suffix = ".webp"
        relative = f"sources/{index:04d}_primary_file{suffix}"
        payloads[relative] = payload
        digest = sha256_bytes(payload)
        artifacts.append(
            SourceArtifact(
                order_index=index,
                source_kind="PRIMARY_FILE",
                mime_type=page_mime,
                relative_path=relative,
                sha256=digest,
                size_bytes=len(payload),
                file_s3_key=S3_KEY,
                include_in_ocr=True,
            )
        )
        files.append(
            BundleFile(
                relative_path=relative,
                sha256=digest,
                size_bytes=len(payload),
                role="source",
            )
        )
    for relative, payload in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "transcription_gpt.json").write_text(GPT_READING, encoding="utf-8")
    (root / "transcription_grok.json").write_text(GROK_READING, encoding="utf-8")
    manifest = DocumentBundleManifest(
        document_id=DOCUMENT_ID,
        archive_item_id=5000 + DOCUMENT_ID,
        language="fr",
        batch_id=BATCH_ID,
        created_at=CREATED,
        transcription=TranscriptionArtifact(
            relative_path="transcription/transcription.txt",
            sha256=sha256_bytes(text),
            size_bytes=len(text),
            text_result_id=DOCUMENT_ID,
            result_type="SOURCE_TEXT",
            status="NEEDS_REVIEW",
            verification_status="UNVERIFIED",
        ),
        sources=artifacts,
        provenance=LocalProvenance(
            displayed_transcription_resolver="resolve_displayed_transcription_result",
            source_set="primary_file",
            presign_expires_in=300,
        ),
        files=files,
    )
    publish_document_manifest(root / "document_manifest.json", manifest)
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _stage(
    tmp_path: Path,
    image: bytes,
    *,
    mime: str = "image/png",
    executions: tuple[StageExecution, StageExecution] = CODEX,
    transcription: str = TRANSCRIPTION,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    batches = tmp_path / "batches"
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    sealed = _seal(batches, image, mime=mime, transcription=transcription)
    before = _snapshot(sealed)
    digest = sha256_bytes((sealed / "document_manifest.json").read_bytes())
    stage = stage_call(
        batches_dir=batches,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        stage_root=tmp_path / "stage",
        executions=executions,
        reviews_dir=reviews,
        work_requests=_work_requests(digest),
        codex_work_root=_codex_root(tmp_path / "stage", executions),
    )
    assert _snapshot(sealed) == before
    return stage, sealed


def test_png_and_jpeg_originals_match_across_providers(tmp_path: Path) -> None:
    jpeg = _jpeg()
    derived = jpeg_to_png(jpeg)
    for image, mime, filename in (
        (_png(), "image/png", "original.png"),
        (jpeg, "image/jpeg", "original.jpg"),
    ):
        stage, sealed = _stage(tmp_path / mime.replace("/", "-"), image, mime=mime)
        assert stage.batch_id == BATCH_ID
        assert stage.document_id == DOCUMENT_ID
        assert stage.page_id == "page_001"
        assert stage.source_sha256 == sha256_bytes(image)
        assert stage.gpt.sha256 == stage.grok.sha256 == sha256_bytes(image)
        assert stage.gpt.byte_size == stage.grok.byte_size == len(image)
        assert stage.gpt.image_path.read_bytes() == image
        assert stage.grok.image_path.read_bytes() == image
        assert stage.gpt.image_path.name == filename
        assert derived not in {
            stage.gpt.image_path.read_bytes(),
            stage.grok.image_path.read_bytes(),
        }
        source = next(sealed.joinpath("sources").iterdir())
        assert source.read_bytes() == image
        for provider in (stage.gpt, stage.grok):
            record = json.loads(provider.manifest_path.read_text(encoding="utf-8"))
            assert record["schema_version"] == "original-image-stage-v1"
            assert record["sha256"] == sha256_bytes(image)
            assert record["byte_size"] == len(image)
            assert record["source_filename"] == source.name
            assert record["page_id"] == "page_001"
            assert record["document_id"] == DOCUMENT_ID
            assert record["staging_location"] == str(provider.directory)
            assert record["staging_location"].startswith("/tmp/")
            assert "provider_interface" not in record
            assert "created_at" not in record
            assert PROVIDER_OUTPUT_SCHEMA_VERSION == "review-provider-output-v1"


def test_second_stage_is_deterministic_and_does_not_rewrite(tmp_path: Path) -> None:
    stage, _sealed = _stage(tmp_path, _png())
    first = {
        provider: (stage.stage_root / provider / "manifest.json").read_bytes()
        for provider in ("gpt", "grok")
    }
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in stage.stage_root.rglob("*")
        if path.is_file()
    }
    again = stage_call(
        batches_dir=tmp_path / "batches",
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        stage_root=stage.stage_root,
        executions=CODEX,
        reviews_dir=tmp_path / "reviews",
    )
    assert again.source_sha256 == stage.source_sha256
    for provider, payload in first.items():
        path = stage.stage_root / provider / "manifest.json"
        assert path.read_bytes() == payload
        assert path.stat().st_mtime_ns == mtimes[path]
    gpt = json.loads(first["gpt"])
    grok = json.loads(first["grok"])
    assert gpt["execution_mechanism"] == "codex_cli"
    assert grok["execution_mechanism"] == "grok_composer"
    assert "gpt_work" not in first["gpt"].decode("utf-8")
    assert "codex_cli" not in first["grok"].decode("utf-8")
    prompt = stage.gpt.prompt_path.read_text(encoding="utf-8")
    assert "GPT_WORK" in prompt
    assert "gpt_work" not in prompt
    assert "execution_mechanism: codex_cli" in prompt


def test_gpt_work_is_not_labeled_codex(tmp_path: Path) -> None:
    stage, _sealed = _stage(tmp_path, _png(), executions=GPT_WORK)
    record = json.loads(stage.gpt.manifest_path.read_text(encoding="utf-8"))
    assert record["execution_mechanism"] == "gpt_work"
    assert "codex_cli" not in stage.gpt.manifest_path.read_text(encoding="utf-8")
    assert "codex_cli" not in stage.gpt.prompt_path.read_text(encoding="utf-8")
    with pytest.raises(ReviewPolicyError, match="codex_cli"):
        codex_staged_image(stage)
    assert grok_staged_image(stage) == stage.grok.image_path


def test_codex_consumer_receives_only_the_gpt_original(tmp_path: Path) -> None:
    image = _png()
    stage, _sealed = _stage(tmp_path, image)
    execution = stage.codex_execution
    assert execution is not None
    assert codex_staged_image(stage).read_bytes() == image
    assert codex_staged_image(stage).parent == execution.working_directory
    assert execution.working_directory.parent != stage.stage_root
    assert stage.grok.directory.parent != execution.working_directory.parent
    assert stage.grok.directory not in execution.working_directory.parents
    assert {path.name for path in execution.working_directory.parent.iterdir()} == {
        "cwd"
    }
    assert "grok" not in {path.name for path in execution.working_directory.iterdir()}
    assert grok_staged_image(stage).parent == stage.grok.directory


def test_staged_files_exclude_transcription_api_and_other_provider(
    tmp_path: Path,
) -> None:
    stage, _sealed = _stage(tmp_path, _jpeg(), mime="image/jpeg")
    forbidden = (
        TRANSCRIPTION.encode("utf-8"),
        PRESIGNED,
        S3_KEY.encode("utf-8"),
        GPT_READING.encode("utf-8"),
        GROK_READING.encode("utf-8"),
        b"transcription.txt",
        b"review_bundle.json",
        b".derived.png",
        b"secretvalue",
        b"presign_expires_in",
    )
    for provider in ("gpt", "grok"):
        directory = stage.stage_root / provider
        names = {path.name for path in directory.iterdir()}
        assert names == {
            "manifest.json",
            "prompt.txt",
            "work_request.json",
            "output_schema.json",
            "output_example.json",
            "original.jpg",
        }
        for path in directory.iterdir():
            data = path.read_bytes()
            for needle in forbidden:
                assert needle not in data
        prompt = (directory / "prompt.txt").read_text(encoding="utf-8")
        assert "review-provider-output-v1" in prompt
        assert "execution_mechanism" in prompt
    assert stage.codex_execution is not None
    for path in stage.codex_execution.working_directory.iterdir():
        data = path.read_bytes()
        for needle in forbidden:
            assert needle not in data
        assert path.name != "manifest.json"


def test_missing_or_modified_source_is_rejected(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    sealed = _seal(batches, _png())
    source = sealed / "sources" / "0000_primary_file.png"
    source.write_bytes(source.read_bytes() + b"\x00")
    with pytest.raises(BundleVerificationError, match="source bytes do not match"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=tmp_path / "stage",
            executions=CODEX,
        )
    assert not (tmp_path / "stage").exists()

    batches_missing = tmp_path / "missing-batches"
    sealed_missing = _seal(batches_missing, _png())
    (sealed_missing / "sources" / "0000_primary_file.png").unlink()
    with pytest.raises(BundleVerificationError, match="missing source"):
        stage_call(
            batches_dir=batches_missing,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=tmp_path / "missing-stage",
            executions=CODEX,
        )


def test_staged_mutation_is_detected(tmp_path: Path) -> None:
    stage, sealed = _stage(tmp_path, _png())
    image = stage.gpt.image_path
    image.write_bytes(image.read_bytes()[:-1] + b"\x00")
    with pytest.raises(ReviewIntegrityError, match="does not match the sealed source"):
        load_staged_original(stage.stage_root, sealed)
    source = sealed / "sources" / "0000_primary_file.png"
    source.write_bytes(_png() + b"\x01")
    with pytest.raises(BundleVerificationError, match="source bytes do not match"):
        load_staged_original(stage.stage_root, sealed)


def test_unsupported_formats_and_extra_pages_are_rejected(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    _seal(batches, b"RIFF", mime="image/webp")
    with pytest.raises(ReviewPolicyError, match="unsupported image format"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=tmp_path / "webp",
            executions=CODEX,
        )
    two = tmp_path / "two"
    _seal(
        two / "batches",
        _png(),
        sources=[(_png(), "image/png"), (_png(), "image/png")],
    )
    with pytest.raises(ReviewPolicyError, match="one page"):
        stage_call(
            batches_dir=two / "batches",
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=two / "stage",
            executions=CODEX,
        )
    corrupt = tmp_path / "corrupt"
    _seal(corrupt / "batches", b"\x89PNG\r\n\x1a\nnot-a-png", mime="image/png")
    with pytest.raises(ReviewPolicyError, match="PNG"):
        stage_call(
            batches_dir=corrupt / "batches",
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=corrupt / "stage",
            executions=CODEX,
        )


def test_transcription_inside_the_image_is_not_published(tmp_path: Path) -> None:
    image = _png() + TRANSCRIPTION.encode("utf-8")
    batches = tmp_path / "batches"
    _seal(batches, image)
    with pytest.raises(ReviewIntegrityError, match="archive transcription"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=tmp_path / "stage",
            executions=CODEX,
        )
    root = tmp_path / "stage"
    assert not (root / "gpt").exists()
    assert not (root / "grok").exists()
    assert not (root / ".partial").exists()


def test_credential_tamper_is_rejected_without_echoing_the_secret(
    tmp_path: Path,
) -> None:
    stage, sealed = _stage(tmp_path, _png())
    prompt = stage.gpt.prompt_path
    prompt.write_bytes(prompt.read_bytes() + b"\nX-Amz-Signature=secretvalue\n")
    with pytest.raises(ReviewIntegrityError, match="download credential") as caught:
        load_staged_original(stage.stage_root, sealed)
    assert "secretvalue" not in str(caught.value)


def test_cross_provider_paths_are_rejected(tmp_path: Path) -> None:
    stage, sealed = _stage(tmp_path, _png())
    with pytest.raises(ReviewPathError, match="provider directory"):
        resolve_provider_file(stage.stage_root, "gpt", "../grok/original.png")
    with pytest.raises(ReviewPathError, match="provider directory"):
        resolve_provider_file(stage.stage_root, "gpt", "/tmp/grok/original.png")
    image = stage.gpt.image_path
    image.unlink()
    image.symlink_to(stage.grok.image_path)
    with pytest.raises(ReviewIntegrityError, match="other provider"):
        load_staged_original(stage.stage_root, sealed)


def test_unsafe_stage_locations_are_rejected(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    sealed = _seal(batches, _png())
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    repo = Path(__file__).resolve().parents[1]
    before = _snapshot(sealed)
    cases = (
        repo / "stage-out",
        sealed,
        sealed.parent,
        reviews / "nested",
        Path("/tmp") / ".." / "..",
    )
    for destination in cases:
        with pytest.raises(ReviewPathError):
            stage_sealed_original(
                sealed,
                destination,
                CODEX,
                protected_roots=(reviews, batches),
            )
    assert _snapshot(sealed) == before


def test_mechanism_mismatch_is_rejected(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    _seal(batches, _png())
    with pytest.raises(ReviewPolicyError, match="does not match provider"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=tmp_path / "stage",
            executions=(
                StageExecution("gpt", "grok_composer"),
                StageExecution("grok", "grok_composer"),
            ),
        )
    with pytest.raises(ReviewPolicyError, match="does not match provider"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=tmp_path / "other",
            executions=(
                StageExecution("gpt", "codex_cli"),
                StageExecution("grok", "codex_cli"),
            ),
        )


def test_interrupted_partial_is_left_in_place(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    _seal(batches, _png())
    root = tmp_path / "stage"
    partial = root / ".partial" / "gpt"
    partial.mkdir(parents=True)
    evidence = partial / "original.png"
    evidence.write_bytes(_png())
    with pytest.raises(ReviewIntegrityError, match="interrupted"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=root,
            executions=CODEX,
        )
    assert evidence.read_bytes() == _png()
    assert not (root / "gpt").exists()
    assert not (root / "grok").exists()


def test_incomplete_provider_directory_is_not_replaced(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    _seal(batches, _png())
    root = tmp_path / "stage"
    evidence = root / "gpt" / "original.png"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(b"partial-evidence")
    with pytest.raises(ReviewIntegrityError, match="incomplete"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=root,
            executions=CODEX,
        )
    assert evidence.read_bytes() == b"partial-evidence"
    assert not (root / "grok").exists()


def test_failed_write_removes_only_the_unpublished_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = tmp_path / "batches"
    _seal(batches, _png())

    def fail(path: Path, payload: bytes) -> None:
        raise OSError(path.name)

    monkeypatch.setattr(original_image_stage, "_write_new", fail)
    root = tmp_path / "stage"
    with pytest.raises(OSError):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=root,
            executions=CODEX,
        )
    assert not (root / ".partial").exists()
    assert not (root / "gpt").exists()
    assert not (root / "grok").exists()


def test_interrupted_publish_is_not_a_completed_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = tmp_path / "batches"
    sealed = _seal(batches, _png())
    real = original_image_stage.os.rename
    calls = {"n": 0}

    def flaky(src: Path, dst: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("interrupted")
        real(src, dst)

    monkeypatch.setattr(original_image_stage.os, "rename", flaky)
    root = tmp_path / "stage"
    with pytest.raises(ReviewIntegrityError, match="incomplete"):
        stage_call(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=root,
            executions=CODEX,
        )
    assert (root / "gpt").is_dir()
    assert not (root / "grok").is_dir()
    assert (root / ".partial").exists()
    with pytest.raises(ReviewIntegrityError, match="interrupted"):
        load_staged_original(root, sealed)


def test_changed_stage_is_not_overwritten(tmp_path: Path) -> None:
    stage, _sealed = _stage(tmp_path, _png())
    prompt = stage.gpt.prompt_path
    prompt.write_bytes(prompt.read_bytes() + b"\nextra\n")
    with pytest.raises(ReviewIntegrityError, match="refusing to change"):
        stage_call(
            batches_dir=tmp_path / "batches",
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=stage.stage_root,
            executions=CODEX,
            reviews_dir=tmp_path / "reviews",
        )
    assert prompt.read_bytes().endswith(b"\nextra\n")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_instruction_package_matches_provider_output_identity(tmp_path: Path) -> None:
    stage, sealed = _stage(tmp_path, _png())
    digest = sha256_bytes((sealed / "document_manifest.json").read_bytes())
    request = json.loads(
        (stage.gpt.directory / "work_request.json").read_text(encoding="utf-8")
    )
    example = json.loads(
        (stage.gpt.directory / "output_example.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (stage.gpt.directory / "output_schema.json").read_text(encoding="utf-8")
    )
    assert stage.codex_execution is not None
    cwd_request = json.loads(
        stage.codex_execution.work_request_path.read_text(encoding="utf-8")
    )
    for payload in (request, cwd_request, example):
        assert payload["provider"] == "gpt"
        assert payload["provider_interface"] == "GPT_WORK"
        assert payload["task_id"] == f"{BATCH_ID}:{DOCUMENT_ID}:first_pass:r0:gpt:a1"
        assert payload["job_id"] == f"job:{payload['task_id']}"
        assert payload["batch_id"] == BATCH_ID
        assert payload["document_id"] == DOCUMENT_ID
        assert payload["attempt"] == 1
        assert payload["source_bundle_sha256"] == digest
        assert payload["stage"] == "first_pass"
    assert request["schema_version"] == "review-work-request-v1"
    assert example["schema_version"] == PROVIDER_OUTPUT_SCHEMA_VERSION
    assert example["pages"][0]["page_id"] == "page_001"
    assert schema["schema_version"] == PROVIDER_OUTPUT_SCHEMA_VERSION
    manifest = json.loads(stage.gpt.manifest_path.read_text(encoding="utf-8"))
    assert manifest["execution_mechanism"] == "codex_cli"
    assert "provider_interface" not in manifest
    assert manifest["sha256"] == sha256_bytes(_png())
    assert manifest["page_id"] == "page_001"


def test_private_permissions_are_created_and_tightened_on_reuse(
    tmp_path: Path,
) -> None:
    stage, _sealed = _stage(tmp_path, _png())
    assert stage.codex_execution is not None
    paths = (
        stage.stage_root,
        stage.gpt.directory,
        stage.grok.directory,
        stage.codex_execution.working_directory,
        stage.codex_execution.working_directory.parent,
    )
    files = (
        stage.gpt.image_path,
        stage.gpt.prompt_path,
        stage.grok.image_path,
        stage.codex_execution.image_path,
        stage.codex_execution.prompt_path,
    )
    for path in paths:
        assert _mode(path) == 0o700
    for path in files:
        assert _mode(path) == 0o600
    image = stage.gpt.image_path
    before = image.read_bytes()
    image.chmod(0o644)
    stage.gpt.directory.chmod(0o755)
    stage.codex_execution.image_path.chmod(0o666)
    again = stage_call(
        batches_dir=tmp_path / "batches",
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        stage_root=stage.stage_root,
        executions=CODEX,
        reviews_dir=tmp_path / "reviews",
    )
    assert image.read_bytes() == before
    assert _mode(image) == 0o600
    assert _mode(stage.gpt.directory) == 0o700
    assert again.codex_execution is not None
    assert _mode(again.codex_execution.image_path) == 0o600


def test_symlink_is_rejected_without_changing_the_target(tmp_path: Path) -> None:
    stage, sealed = _stage(tmp_path, _png())
    target = stage.grok.image_path
    target.chmod(0o644)
    image = stage.gpt.image_path
    image.unlink()
    image.symlink_to(target)
    with pytest.raises(ReviewIntegrityError, match="other provider"):
        load_staged_original(
            stage.stage_root,
            sealed,
            codex_work_root=stage.codex_execution.working_directory.parent
            if stage.codex_execution is not None
            else None,
        )
    assert image.is_symlink()
    assert _mode(target) == 0o644


def test_mismatched_work_request_is_rejected_before_publication(
    tmp_path: Path,
) -> None:
    batches = tmp_path / "batches"
    _seal(batches, _png())
    requests = list(_approved(batches))
    requests[0] = requests[0].model_copy(
        update={"task_id": "batch-000001:7:first_pass:r0:gpt:a2"}
    )
    with pytest.raises(ReviewIntegrityError, match="task identity"):
        stage_sealed_document(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            stage_root=tmp_path / "stage",
            executions=CODEX,
            work_requests=tuple(requests),
            codex_work_root=tmp_path / "codex-work",
        )
    assert not (tmp_path / "stage" / "gpt").exists()
    assert not (tmp_path / "codex-work").exists()
