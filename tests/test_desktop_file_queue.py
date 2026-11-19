"""Fixture checks for the Windows-visible desktop file queue."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.desktop_file_queue import (
    load_desktop_file_queue,
    publish_desktop_file_queue,
    resolve_desktop_queue_root,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
)
from transcription_review_agent.review.jpegutil import jpeg_to_png
from transcription_review_agent.review.models import FirstPassOutput
from transcription_review_agent.review.provider_output_contract import (
    TEMPLATE_PAGE_TEXT,
)
from test_original_image_stage import (
    BATCH_ID,
    DOCUMENT_ID,
    GPT_READING,
    GROK_READING,
    PRESIGNED,
    S3_KEY,
    TRANSCRIPTION,
    _approved,
    _jpeg,
    _png,
    _seal,
    _snapshot,
)

QUEUE = "D:/archive-queue/tasks"


def _mount(tmp_path: Path) -> Path:
    mount = tmp_path / "mnt"
    (mount / "d").mkdir(parents=True)
    return mount


def _publish(
    tmp_path: Path, image: bytes, *, mime: str = "image/png", queue: str = QUEUE
):
    batches = tmp_path / "batches"
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    sealed = _seal(batches, image, mime=mime)
    before = _snapshot(sealed)
    queue_root = publish_desktop_file_queue(
        batches_dir=batches,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=queue,
        work_requests=_approved(batches),
        reviews_dir=reviews,
        mount_root=_mount(tmp_path),
    )
    assert _snapshot(sealed) == before
    return queue_root, sealed


def _load(
    tmp_path: Path,
    sealed: Path,
    *,
    queue: Path | str = QUEUE,
    mount: Path | None = None,
):
    return load_desktop_file_queue(
        queue,
        sealed,
        _approved(tmp_path / "batches"),
        reviews_dir=tmp_path / "reviews",
        mount_root=mount or tmp_path / "mnt",
    )


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def test_drive_path_and_mount_path_publish_the_same_queue(tmp_path: Path) -> None:
    image = _png()
    published, sealed = _publish(tmp_path, image)
    mount = tmp_path / "mnt"
    local_root = mount / "d" / "archive-queue" / "tasks"
    again = publish_desktop_file_queue(
        batches_dir=tmp_path / "batches",
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=local_root,
        work_requests=_approved(tmp_path / "batches"),
        reviews_dir=tmp_path / "reviews",
        mount_root=mount,
    )
    assert published.windows_queue_root == "D:\\archive-queue\\tasks"
    assert again.windows_queue_root == published.windows_queue_root
    assert again.document_directory == published.document_directory
    assert published.gpt.sha256 == published.grok.sha256 == sha256_bytes(image)
    assert published.gpt.image_path.read_bytes() == image
    assert published.grok.image_path.read_bytes() == image
    assert published.gpt.output_directory != published.grok.output_directory
    assert list(published.gpt.output_directory.iterdir()) == []
    assert list(published.grok.output_directory.iterdir()) == []
    assert published.gpt.result_present is False
    assert not (published.document_directory / "first_pass").exists()
    assert published.gpt.windows_output_file == (
        "D:\\archive-queue\\tasks\\batch-000001\\document_7\\gpt"
        "\\output\\transcription_gpt.json"
    )
    assert published.grok.windows_output_file.endswith(
        "\\grok\\output\\transcription_grok.json"
    )
    loaded = _load(tmp_path, sealed, queue=local_root, mount=mount)
    assert loaded.source_bundle_sha256 == published.source_bundle_sha256
    assert "codex" not in published.gpt.prompt_path.read_text(encoding="utf-8")
    assert "cwd" not in {path.name for path in published.document_directory.iterdir()}


def test_provider_directories_keep_the_sealed_image_and_contract(
    tmp_path: Path,
) -> None:
    image = _jpeg()
    derived = jpeg_to_png(image)
    published, _sealed = _publish(tmp_path, image, mime="image/jpeg")
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
    )
    for task, other_name in (
        (published.gpt, "transcription_grok.json"),
        (published.grok, "transcription_gpt.json"),
    ):
        names = {path.name for path in task.directory.iterdir()}
        assert names == {
            "manifest.json",
            "prompt.txt",
            "work_request.json",
            "output_schema.json",
            "output_example.json",
            "original.jpg",
            "output",
        }
        assert task.image_path.read_bytes() != derived
        prompt = task.prompt_path.read_text(encoding="utf-8")
        assert task.windows_output_file in prompt
        assert "It is not a reading of this page." in prompt
        assert "desktop_queue.json" in prompt
        assert ".partial" in prompt
        assert other_name not in prompt
        example = task.directory.joinpath("output_example.json").read_text(
            encoding="utf-8"
        )
        assert TEMPLATE_PAGE_TEXT in example
        assert TRANSCRIPTION not in example
        for path in task.directory.iterdir():
            if path.name == "output":
                continue
            data = path.read_bytes()
            for needle in forbidden:
                assert needle not in data
            assert other_name.encode("utf-8") not in data
        record = json.loads(task.manifest_path.read_text(encoding="utf-8"))
        assert record["model_transcription_included"] is False
        assert record["archive_transcription_included"] is False
        assert record["other_provider_output_included"] is False
        assert record["execution_mechanism"] == (
            "gpt_work" if task.provider == "gpt" else "grok_composer"
        )
    index = json.loads(
        (published.document_directory / "desktop_queue.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["gpt_output_file"] == published.gpt.windows_output_file
    assert index["grok_output_file"] == published.grok.windows_output_file
    assert published.gpt.windows_directory not in published.grok.prompt_path.read_text(
        encoding="utf-8"
    )


def test_prompt_requires_the_example_shape_without_output_relative_path(
    tmp_path: Path,
) -> None:
    published, _sealed = _publish(tmp_path, _png())
    for task in (published.gpt, published.grok):
        prompt = task.prompt_path.read_text(encoding="utf-8")
        assert "Copy output_relative_path" not in prompt
        assert "Do not add output_relative_path to the JSON object." in prompt
        assert task.windows_output_file in prompt
        example = json.loads(
            task.directory.joinpath("output_example.json").read_text(encoding="utf-8")
        )
        schema = json.loads(
            task.directory.joinpath("output_schema.json").read_text(encoding="utf-8")
        )
        assert "output_relative_path" not in example
        assert "output_relative_path" not in schema["top_level_fields"]
        parsed = FirstPassOutput.model_validate(example)
        assert parsed.provider == task.provider
        with pytest.raises(ValueError):
            FirstPassOutput.model_validate(
                {**example, "output_relative_path": "first_pass/output/file.json"}
            )


def test_second_publish_reuses_tasks_and_preserves_a_result(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    result = published.gpt.result_path
    result.write_bytes(b'{"provider":"gpt","pages":[]}\n')
    result.chmod(0o644)
    published.gpt.image_path.chmod(0o644)
    before = {
        path: path.stat().st_mtime_ns
        for path in published.document_directory.rglob("*")
        if path.is_file()
    }
    result_bytes = result.read_bytes()
    again = publish_desktop_file_queue(
        batches_dir=tmp_path / "batches",
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        work_requests=_approved(tmp_path / "batches"),
        reviews_dir=tmp_path / "reviews",
        mount_root=tmp_path / "mnt",
    )
    assert again.gpt.result_present is True
    assert result.read_bytes() == result_bytes
    assert result.stat().st_mtime_ns == before[result]
    assert (
        published.gpt.image_path.stat().st_mtime_ns == before[published.gpt.image_path]
    )
    assert _mode(published.gpt.image_path) == 0o600
    assert _mode(published.gpt.output_directory) == 0o700
    loaded = _load(tmp_path, sealed)
    assert loaded.gpt.result_present is True
    assert loaded.gpt.result_path.read_bytes() == result_bytes


def test_missing_index_is_restored_without_rewriting_tasks(tmp_path: Path) -> None:
    published, _sealed = _publish(tmp_path, _png())
    index = published.document_directory / "desktop_queue.json"
    payload = index.read_bytes()
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in published.gpt.directory.rglob("*")
        if path.is_file()
    }
    index.unlink()
    again = publish_desktop_file_queue(
        batches_dir=tmp_path / "batches",
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        work_requests=_approved(tmp_path / "batches"),
        reviews_dir=tmp_path / "reviews",
        mount_root=tmp_path / "mnt",
    )
    assert (
        again.document_directory.joinpath("desktop_queue.json").read_bytes() == payload
    )
    for path, mtime in mtimes.items():
        assert path.stat().st_mtime_ns == mtime


def test_changed_task_is_not_overwritten(tmp_path: Path) -> None:
    published, _sealed = _publish(tmp_path, _png())
    prompt = published.gpt.prompt_path
    prompt.write_bytes(prompt.read_bytes() + b"\nextra\n")
    with pytest.raises(ReviewIntegrityError, match="refusing to change"):
        publish_desktop_file_queue(
            batches_dir=tmp_path / "batches",
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=_approved(tmp_path / "batches"),
            reviews_dir=tmp_path / "reviews",
            mount_root=tmp_path / "mnt",
        )
    assert prompt.read_bytes().endswith(b"\nextra\n")


def test_unexpected_result_and_sibling_symlink_stay_in_place(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    extra = published.gpt.output_directory / "notes.txt"
    extra.write_text("keep", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError, match="unexpected file"):
        _load(tmp_path, sealed)
    assert extra.read_text(encoding="utf-8") == "keep"
    extra.unlink()
    result = published.grok.result_path
    result.write_bytes(b'{"provider":"grok"}\n')
    image = published.gpt.image_path
    image.unlink()
    image.symlink_to(published.grok.image_path)
    with pytest.raises(ReviewIntegrityError, match="other provider"):
        _load(tmp_path, sealed)
    assert image.is_symlink()
    assert result.read_bytes() == b'{"provider":"grok"}\n'


def test_interrupted_partial_and_failed_write_are_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batches = tmp_path / "batches"
    _seal(batches, _png())
    mount = _mount(tmp_path)
    root = (
        mount / "d" / "archive-queue" / "tasks" / BATCH_ID / f"document_{DOCUMENT_ID}"
    )
    evidence = root / ".partial" / "gpt" / "original.png"
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes(_png())
    with pytest.raises(ReviewIntegrityError, match="interrupted"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=_approved(batches),
            reviews_dir=tmp_path / "reviews",
            mount_root=mount,
        )
    assert evidence.read_bytes() == _png()
    assert not (root / "gpt").exists()

    import transcription_review_agent.review.desktop_file_queue as queue

    def fail(path: Path, payload: bytes) -> None:
        raise OSError(path.name)

    monkeypatch.setattr(queue, "_write_new", fail)
    fresh = tmp_path / "fresh"
    mount_fresh = _mount(fresh)
    _seal(fresh / "batches", _png())
    failed = (
        mount_fresh
        / "d"
        / "archive-queue"
        / "tasks"
        / BATCH_ID
        / f"document_{DOCUMENT_ID}"
    )
    with pytest.raises(OSError):
        publish_desktop_file_queue(
            batches_dir=fresh / "batches",
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=_approved(fresh / "batches"),
            mount_root=mount_fresh,
        )
    assert not (failed / ".partial").exists()
    assert not (failed / "gpt").exists()


def test_unsafe_queue_locations_are_rejected(tmp_path: Path) -> None:
    mount = _mount(tmp_path)
    with pytest.raises(ReviewPathError, match="Windows-visible"):
        resolve_desktop_queue_root(Path("/tmp/desktop-queue"))
    with pytest.raises(ReviewPathError, match="not mounted"):
        resolve_desktop_queue_root(Path("Z:/queue/tasks"), mount_root=mount)
    with pytest.raises(ReviewPathError, match="not safe"):
        resolve_desktop_queue_root(Path("D:/queue/../sealed"), mount_root=mount)
    with pytest.raises(ReviewPathError, match="drive path"):
        resolve_desktop_queue_root("\\\\server\\share\\queue", mount_root=mount)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = mount / "d" / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ReviewIntegrityError, match="symlink"):
        resolve_desktop_queue_root(Path("D:/linked/tasks"), mount_root=mount)
    batches = mount / "d" / "batches"
    _seal(batches, _png())
    reviews = mount / "d" / "reviews"
    reviews.mkdir()
    with pytest.raises(ReviewPathError, match="sealed bundle"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root="D:/batches/batch-000001/documents/7/nested",
            work_requests=_approved(batches),
            mount_root=mount,
        )
    with pytest.raises(ReviewPathError, match="review tree"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root="D:/reviews/nested",
            work_requests=_approved(batches),
            reviews_dir=reviews,
            mount_root=mount,
        )


def test_transcription_inside_the_image_is_not_published(tmp_path: Path) -> None:
    image = _png() + TRANSCRIPTION.encode("utf-8")
    batches = tmp_path / "batches"
    _seal(batches, image)
    mount = _mount(tmp_path)
    with pytest.raises(ReviewIntegrityError, match="archive transcription"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=_approved(batches),
            mount_root=mount,
        )
    document = (
        mount / "d" / "archive-queue" / "tasks" / BATCH_ID / f"document_{DOCUMENT_ID}"
    )
    assert not (document / "gpt").exists()
    assert not (document / "grok").exists()
    assert not (document / ".partial").exists()


def test_mismatched_work_request_publishes_nothing(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    _seal(batches, _png())
    requests = list(_approved(batches))
    requests[0] = requests[0].model_copy(update={"attempt": 2})
    mount = _mount(tmp_path)
    with pytest.raises(ReviewIntegrityError, match="task identity"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=tuple(requests),
            mount_root=mount,
        )
    document = (
        mount / "d" / "archive-queue" / "tasks" / BATCH_ID / f"document_{DOCUMENT_ID}"
    )
    assert not document.exists()


def test_desktop_queue_does_not_invoke_a_model_or_desktop_app(tmp_path: Path) -> None:
    import ast

    import transcription_review_agent.review.desktop_file_queue as queue

    tree = ast.parse(Path(queue.__file__).read_text(encoding="utf-8"))
    banned = ("subprocess", "codex_image_transcription", "grok_desktop")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(alias.name for alias in node.names)]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names = [node.func.attr]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names = [node.func.id]
        else:
            continue
        assert not any(
            name == banned_name or name.endswith(f".{banned_name}")
            for name in names
            for banned_name in (*banned, "system", "Popen", "exec")
        )
    published, _sealed = _publish(tmp_path, _png())
    assert published.gpt.execution_mechanism == "gpt_work"
    assert published.grok.execution_mechanism == "grok_composer"
    assert os.listdir(published.gpt.output_directory) == []


@pytest.mark.parametrize(
    "name",
    [
        "prompt.txt",
        "work_request.json",
        "output_schema.json",
        "output_example.json",
        "manifest.json",
        "original.png",
    ],
)
def test_load_rejects_each_mutated_task_file(tmp_path: Path, name: str) -> None:
    published, sealed = _publish(tmp_path, _png())
    path = published.gpt.directory / name
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ReviewIntegrityError, match="does not match the sealed source"):
        _load(tmp_path, sealed)
    assert path.read_bytes() == original + b"\n"


@pytest.mark.parametrize("name", ["AGENTS.md", "page.png"])
def test_load_rejects_unexpected_task_files(tmp_path: Path, name: str) -> None:
    published, sealed = _publish(tmp_path, _png())
    extra = published.gpt.directory / name
    extra.write_bytes(b"extra")
    with pytest.raises(ReviewIntegrityError, match="unexpected file"):
        _load(tmp_path, sealed)
    assert extra.read_bytes() == b"extra"


def test_hardlink_to_the_sealed_image_or_sibling_is_rejected(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    sealed_image = sealed / "sources" / "0000_primary_file.png"
    image = published.gpt.image_path
    image.unlink()
    os.link(sealed_image, image)
    with pytest.raises(ReviewIntegrityError, match="hardlink"):
        _load(tmp_path, sealed)
    assert image.exists()
    assert sealed_image.read_bytes() == _png()
    image.unlink()
    os.link(published.grok.image_path, image)
    with pytest.raises(ReviewIntegrityError, match="hardlink"):
        _load(tmp_path, sealed)
    assert image.samefile(published.grok.image_path)


def test_reparse_attribute_is_rejected_when_windows_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcription_review_agent.review.desktop_file_queue as queue

    assert queue._has_reparse_attribute(0x400) is True
    assert queue._has_reparse_attribute(0x10) is False
    mount = _mount(tmp_path)
    linked = mount / "d" / "linked"
    linked.mkdir()
    seen: list[Path] = []

    def reject(path: Path) -> None:
        seen.append(path)
        if path.name == "linked":
            raise ReviewIntegrityError(
                "desktop queue path escapes through a symlink or reparse point"
            )

    monkeypatch.setattr(queue, "_reject_if_reparse", reject)
    with pytest.raises(ReviewIntegrityError, match="reparse point"):
        resolve_desktop_queue_root(Path("D:/linked/tasks"), mount_root=mount)
    assert any(path.name == "linked" for path in seen)

    monkeypatch.undo()
    monkeypatch.setattr(queue.os, "name", "nt")
    monkeypatch.setattr(queue, "_windows_file_attributes", lambda _path: 0x400)
    with pytest.raises(ReviewIntegrityError, match="reparse point"):
        queue._reject_if_reparse(linked)
    monkeypatch.setattr(queue, "_windows_file_attributes", lambda _path: None)
    with pytest.raises(ReviewPathError, match="cannot verify"):
        queue._reject_if_reparse(linked)


def test_interrupted_rename_leaves_the_partial_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcription_review_agent.review.desktop_file_queue as queue

    calls = {"count": 0}
    real_rename = os.rename

    def rename(source: Path, destination: Path) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("second rename")
        real_rename(source, destination)

    monkeypatch.setattr(queue.os, "rename", rename)
    batches = tmp_path / "batches"
    sealed = _seal(batches, _png())
    mount = _mount(tmp_path)
    document = (
        mount / "d" / "archive-queue" / "tasks" / BATCH_ID / f"document_{DOCUMENT_ID}"
    )
    with pytest.raises(OSError, match="second rename"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=_approved(batches),
            reviews_dir=tmp_path / "reviews",
            mount_root=mount,
        )
    assert (document / "gpt").is_dir()
    assert not (document / "grok").is_dir()
    assert (document / ".partial" / "grok").is_dir()
    assert not (document / "desktop_queue.json").exists()
    with pytest.raises(ReviewIntegrityError, match="interrupted"):
        _load(tmp_path, sealed, mount=mount)
    with pytest.raises(ReviewIntegrityError, match="interrupted"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=_approved(batches),
            reviews_dir=tmp_path / "reviews",
            mount_root=mount,
        )
    assert (document / "gpt" / "original.png").is_file()


def test_leftover_index_partial_blocks_load_and_publish(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    partial = published.document_directory / ".desktop_queue.json.partial"
    partial.write_bytes(b"{}")
    with pytest.raises(ReviewIntegrityError, match="interrupted"):
        _load(tmp_path, sealed)
    with pytest.raises(ReviewIntegrityError, match="interrupted"):
        publish_desktop_file_queue(
            batches_dir=tmp_path / "batches",
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=_approved(tmp_path / "batches"),
            reviews_dir=tmp_path / "reviews",
            mount_root=tmp_path / "mnt",
        )
    assert partial.read_bytes() == b"{}"
    assert (published.document_directory / "gpt" / "original.png").is_file()


def test_existing_parent_permissions_are_left_unchanged(tmp_path: Path) -> None:
    mount = _mount(tmp_path)
    tasks = mount / "d" / "archive-queue" / "tasks"
    tasks.mkdir(parents=True)
    tasks.chmod(0o755)
    tasks.parent.chmod(0o755)
    batches = tmp_path / "batches"
    _seal(batches, _png())
    (tmp_path / "reviews").mkdir()
    published = publish_desktop_file_queue(
        batches_dir=batches,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        work_requests=_approved(batches),
        reviews_dir=tmp_path / "reviews",
        mount_root=mount,
    )
    assert _mode(tasks) == 0o755
    assert _mode(tasks.parent) == 0o755
    assert _mode(published.document_directory) == 0o700


def test_concurrent_publication_converges(tmp_path: Path) -> None:
    import threading

    batches = tmp_path / "batches"
    _seal(batches, _png())
    mount = _mount(tmp_path)
    (tmp_path / "reviews").mkdir()
    results: list[object] = []
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            results.append(
                publish_desktop_file_queue(
                    batches_dir=batches,
                    batch_id=BATCH_ID,
                    document_id=DOCUMENT_ID,
                    queue_root=QUEUE,
                    work_requests=_approved(batches),
                    reviews_dir=tmp_path / "reviews",
                    mount_root=mount,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(results) == 2
    document = (
        mount / "d" / "archive-queue" / "tasks" / BATCH_ID / f"document_{DOCUMENT_ID}"
    )
    assert not (document / ".partial").exists()
    assert not (document / ".desktop_queue.json.partial").exists()
    assert (document / "desktop_queue.json").is_file()


def test_consistent_later_attempt_is_pinned(tmp_path: Path) -> None:
    from transcription_review_agent.review.paths import output_relative_path

    batches = tmp_path / "batches"
    _seal(batches, _png())
    requests = []
    for request in _approved(batches):
        task_id = (
            f"{request.batch_id}:{request.document_id}:first_pass:"
            f"r0:{request.provider}:a2"
        )
        requests.append(
            request.model_copy(
                update={
                    "attempt": 2,
                    "task_id": task_id,
                    "job_id": f"job:{task_id}",
                    "output_relative_path": output_relative_path(
                        "first_pass", request.provider, 0, 2
                    ),
                }
            )
        )
    mount = _mount(tmp_path)
    with pytest.raises(ReviewIntegrityError, match="attempt 1 only"):
        publish_desktop_file_queue(
            batches_dir=batches,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            work_requests=tuple(requests),
            mount_root=mount,
        )


def _republish(tmp_path: Path):
    return publish_desktop_file_queue(
        batches_dir=tmp_path / "batches",
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        work_requests=_approved(tmp_path / "batches"),
        reviews_dir=tmp_path / "reviews",
        mount_root=tmp_path / "mnt",
    )


@pytest.mark.parametrize(
    "relative",
    ["", "gpt", "grok", "gpt/output", "grok/output"],
)
def test_symlink_owned_directory_is_rejected_without_following_it(
    tmp_path: Path, relative: str
) -> None:
    """Linux symlink at each owned directory. This is not an NTFS junction."""

    published, sealed = _publish(tmp_path, _png())
    outside = tmp_path / "outside-task"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_bytes(b"keep")
    secret.chmod(0o644)
    document = published.document_directory
    path = document if relative == "" else document / relative
    if relative != "":
        kept = (
            published.grok.image_path
            if relative.startswith("gpt")
            else published.gpt.image_path
        )
        kept.chmod(0o644)
    else:
        kept = published.queue_root
    kept_mode = _mode(kept)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    path.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ReviewIntegrityError, match="reparse point"):
        _load(tmp_path, sealed)
    with pytest.raises(ReviewIntegrityError, match="reparse point"):
        _republish(tmp_path)
    assert secret.read_bytes() == b"keep"
    assert _mode(secret) == 0o644
    assert _mode(kept) == kept_mode
    assert path.is_symlink()
    assert [item.name for item in outside.iterdir()] == ["secret.txt"]


def test_mocked_windows_reparse_attribute_rejects_owned_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attribute check only. This does not create or resolve an NTFS junction."""

    import transcription_review_agent.review.desktop_file_queue as queue

    published, _sealed = _publish(tmp_path, _png())
    monkeypatch.setattr(queue.os, "name", "nt")
    monkeypatch.setattr(queue, "_windows_file_attributes", lambda _path: 0x400)
    with pytest.raises(ReviewIntegrityError, match="reparse point"):
        queue._reject_queue_directories(published.document_directory)
    monkeypatch.setattr(queue, "_windows_file_attributes", lambda _path: 0x10)
    queue._reject_queue_directories(published.document_directory)
    monkeypatch.setattr(queue, "_windows_file_attributes", lambda _path: None)
    with pytest.raises(ReviewPathError, match="cannot verify"):
        queue._reject_queue_directories(published.document_directory)


def test_lock_symlink_does_not_modify_its_target(tmp_path: Path) -> None:
    batches = tmp_path / "batches"
    sealed = _seal(batches, _png())
    mount = _mount(tmp_path)
    (tmp_path / "reviews").mkdir()
    root = mount / "d" / "archive-queue" / "tasks"
    root.mkdir(parents=True)
    target = tmp_path / "lock-target"
    target.write_bytes(b"do-not-touch")
    target.chmod(0o644)
    sealed_image = sealed / "sources" / "0000_primary_file.png"
    image_bytes = sealed_image.read_bytes()
    lock = root / f".desktop-queue-{BATCH_ID}-document-{DOCUMENT_ID}.lock"
    lock.symlink_to(target)
    with pytest.raises(ReviewIntegrityError, match="reparse point"):
        _republish(tmp_path)
    assert target.read_bytes() == b"do-not-touch"
    assert _mode(target) == 0o644
    assert lock.is_symlink()
    assert not (root / BATCH_ID).exists()
    lock.unlink()
    lock.symlink_to(sealed_image)
    with pytest.raises(ReviewIntegrityError, match="reparse point"):
        _republish(tmp_path)
    assert sealed_image.read_bytes() == image_bytes
    assert lock.is_symlink()


def test_publish_uses_a_regular_lock_file(tmp_path: Path) -> None:
    published, _sealed = _publish(tmp_path, _png())
    lock = (
        published.queue_root / f".desktop-queue-{BATCH_ID}-document-{DOCUMENT_ID}.lock"
    )
    assert lock.is_file()
    assert not stat.S_ISLNK(lock.lstat().st_mode)
    again = _republish(tmp_path)
    assert again.document_directory == published.document_directory
    assert lock.is_file()
    assert not lock.is_symlink()


def test_failed_validation_does_not_change_permissions(tmp_path: Path) -> None:
    published, sealed = _publish(tmp_path, _png())
    outside = tmp_path / "unrelated.bin"
    outside.write_bytes(b"keep")
    outside.chmod(0o644)
    image = published.gpt.image_path
    prompt = published.gpt.prompt_path
    result = published.grok.result_path
    result.write_bytes(b'{"provider":"grok"}\n')
    for path in (image, prompt, result, published.document_directory):
        path.chmod(0o644 if path.is_file() else 0o755)
    prompt.write_bytes(prompt.read_bytes() + b"\n")
    modes = {
        path: _mode(path)
        for path in (
            image,
            prompt,
            result,
            outside,
            published.document_directory,
            published.grok.directory,
            published.gpt.output_directory,
        )
    }
    result_bytes = result.read_bytes()
    with pytest.raises(ReviewIntegrityError, match="does not match the sealed source"):
        _load(tmp_path, sealed)
    with pytest.raises(ReviewIntegrityError, match="refusing to change"):
        _republish(tmp_path)
    assert {path: _mode(path) for path in modes} == modes
    assert prompt.read_bytes().endswith(b"\n")
    assert result.read_bytes() == result_bytes
    assert outside.read_bytes() == b"keep"
