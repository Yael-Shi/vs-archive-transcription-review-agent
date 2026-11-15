"""Publish isolated GPT Work and Grok Bot task directories on a Windows path.

This is the file queue between the review controller and the desktop apps.
It copies one sealed page into two task directories. It does not launch a
desktop app, call a model, or write a transcription. ``output/`` is the only
place a later desktop session may write, and this module never puts a file
there.

The queue root is a drive path such as ``D:\\archive-queue\\tasks`` or the
same directory under the WSL mount (``/mnt/d/archive-queue/tasks``). A WSL
``/tmp`` directory is not a queue root. Codex execution stays in
``original_image_stage`` and is not selected here.

Provider separation is two directories and two output paths. POSIX ``0700``
and ``0600`` modes are applied only to directories and files this module
creates under the document directory. Those modes are not a Windows ACL and
do not stop the desktop user from opening the other provider directory.
Instructions tell each app to stay in its own task and to stop unless the
parent ``desktop_queue.json`` index is present. That index is the publication
commit marker. A task tree without it is not published.

Recovery from an interrupted publication is manual. This module does not
delete task files, partial directories, or provider results. If both task
directories still match the sealed expectation and the index is the only
missing piece, a later publish writes that index and does not rewrite the
tasks. Any other partial state stays in place. An operator may remove the
document directory only after checking that neither ``output/`` directory
contains a result file, and then publish again.

On Linux, an NTFS junction is rejected when the filesystem presents it as a
symlink. Native Windows also rejects ``FILE_ATTRIBUTE_REPARSE_POINT`` and
fails closed when that attribute cannot be read. The same check covers the
document directory, both task directories, and both ``output/`` directories
before their contents are read or their permissions change. Hardlinks are
rejected. The lock file is opened without following a symlink or reparse point.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.layout import assert_safe_relative_path
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.config import FORBIDDEN_WRITE_ROOTS
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
)
from transcription_review_agent.review.models import WorkRequest
from transcription_review_agent.review.original_image_stage import (
    _PAGE_ID,
    _approved_requests,
    _prompt,
    _repo_root,
    _require_image,
    _sealed_page,
    _write_new,
)
from transcription_review_agent.review.paths import (
    archive_document_dir,
    output_filename,
    output_relative_path,
    require_batch_id,
    reviews_root,
)

QUEUE_SCHEMA_VERSION: Final[Literal["desktop-file-queue-v1"]] = "desktop-file-queue-v1"
TASK_SCHEMA_VERSION: Final[Literal["desktop-file-queue-task-v1"]] = (
    "desktop-file-queue-task-v1"
)
_PARTIAL = ".partial"
_INDEX_PARTIAL = ".desktop_queue.json.partial"
_INDEX_NAME = "desktop_queue.json"
_OUTPUT_DIR = "output"
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_PROVIDERS = ("gpt", "grok")
_SUPPORTED_ATTEMPT = 1
_REPARSE_ATTRIBUTE = 0x400
_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_TASK_FILES = (
    "manifest.json",
    "prompt.txt",
    "work_request.json",
    "output_schema.json",
    "output_example.json",
)
ProviderName = Literal["gpt", "grok"]
ExecutionMechanism = Literal["gpt_work", "grok_composer"]
_MECHANISM: dict[str, ExecutionMechanism] = {
    "gpt": "gpt_work",
    "grok": "grok_composer",
}


class DesktopTaskManifest(BaseModel):
    """Provenance for one desktop task. This file is not a model transcription."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["desktop-file-queue-task-v1"] = TASK_SCHEMA_VERSION
    provider: ProviderName
    execution_mechanism: ExecutionMechanism
    batch_id: str = Field(min_length=1)
    document_id: int = Field(ge=1)
    page_id: str = Field(pattern=r"^page_[0-9]{3}$")
    language: Literal["en", "fr"]
    source_filename: str = Field(min_length=1)
    sealed_relative_path: str = Field(min_length=1)
    mime_type: Literal["image/png", "image/jpeg"]
    byte_size: int = Field(ge=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    image_filename: Literal["original.png", "original.jpg"]
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    local_task_directory: str = Field(min_length=1)
    windows_task_directory: str = Field(min_length=1)
    windows_output_file: str = Field(min_length=1)
    queue_output_relative_path: str = Field(min_length=1)
    review_output_relative_path: str = Field(min_length=1)
    archive_transcription_included: Literal[False] = False
    api_payload_included: Literal[False] = False
    derived_image_included: Literal[False] = False
    other_provider_output_included: Literal[False] = False
    model_transcription_included: Literal[False] = False

    @field_validator("sealed_relative_path", "queue_output_relative_path")
    @classmethod
    def _relative(cls, value: str) -> str:
        return assert_safe_relative_path(value)

    @field_validator("local_task_directory")
    @classmethod
    def _local_absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("local task directory must be absolute")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> DesktopTaskManifest:
        if self.execution_mechanism != _MECHANISM[self.provider]:
            raise ValueError("execution mechanism does not match provider")
        expected_image = (
            "original.png" if self.mime_type == "image/png" else "original.jpg"
        )
        if self.image_filename != expected_image:
            raise ValueError("image filename does not match the sealed mime type")
        if Path(self.local_task_directory).name != self.provider:
            raise ValueError("local task directory does not name this provider")
        if _windows_name(self.windows_task_directory) != self.provider:
            raise ValueError("windows task directory does not name this provider")
        filename = output_filename("first_pass", self.provider)
        expected_windows = f"{self.windows_task_directory}\\{_OUTPUT_DIR}\\{filename}"
        if self.windows_output_file != expected_windows:
            raise ValueError("windows output file is outside this provider task")
        if self.queue_output_relative_path != f"{_OUTPUT_DIR}/{filename}":
            raise ValueError("queue output path is outside this provider task")
        if self.review_output_relative_path != output_relative_path(
            "first_pass", self.provider, 0, 1
        ):
            raise ValueError("review output path is not the first-pass output")
        if (
            "\\" not in self.windows_task_directory
            or "/" in self.windows_task_directory
        ):
            raise ValueError("windows task directory must be a drive path")
        return self


class DesktopQueueIndex(BaseModel):
    """Controller index. It is not placed inside either provider task."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["desktop-file-queue-v1"] = QUEUE_SCHEMA_VERSION
    batch_id: str = Field(min_length=1)
    document_id: int = Field(ge=1)
    page_id: str = Field(pattern=r"^page_[0-9]{3}$")
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    gpt_task_directory: str = Field(min_length=1)
    grok_task_directory: str = Field(min_length=1)
    gpt_output_file: str = Field(min_length=1)
    grok_output_file: str = Field(min_length=1)

    @model_validator(mode="after")
    def _separated(self) -> DesktopQueueIndex:
        if self.gpt_task_directory == self.grok_task_directory:
            raise ValueError("provider task directories are not independent")
        if self.gpt_output_file == self.grok_output_file:
            raise ValueError("provider output files are not independent")
        return self


@dataclass(frozen=True, slots=True)
class DesktopLocation:
    """One queue directory as both the local filesystem path and the Windows path."""

    local_path: Path
    windows_path: str
    drive_local: Path


@dataclass(frozen=True, slots=True)
class DesktopProviderTask:
    provider: ProviderName
    execution_mechanism: ExecutionMechanism
    directory: Path
    windows_directory: str
    image_path: Path
    prompt_path: Path
    manifest_path: Path
    output_directory: Path
    windows_output_file: str
    result_path: Path
    result_present: bool
    sha256: str


@dataclass(frozen=True, slots=True)
class DesktopFileQueue:
    queue_root: Path
    windows_queue_root: str
    document_directory: Path
    windows_document_directory: str
    batch_id: str
    document_id: int
    page_id: str
    source_sha256: str
    source_bundle_sha256: str
    gpt: DesktopProviderTask
    grok: DesktopProviderTask


def resolve_desktop_queue_root(
    queue_root: Path | str,
    *,
    mount_root: Path | None = None,
) -> DesktopLocation:
    """Map a drive path to the local directory a WSL or Windows process can write.

    ``mount_root`` defaults to ``/mnt``. Tests pass a temporary mount. The
    drive directory itself must already exist, so an unmounted letter is not
    created.
    """

    if os.name == "nt":
        letter, parts = _windows_parts(os.fspath(queue_root))
        drive = Path(f"{letter.upper()}:\\")
        local = drive.joinpath(*parts)
        _reject_reparse_chain(local)
        return DesktopLocation(local, _windows_join(letter, parts), drive)
    mount = Path("/mnt") if mount_root is None else Path(mount_root)
    _require_real_directory(mount, "desktop mount root is missing")
    text = os.fspath(queue_root)
    slash_form = text.replace("\\", "/")
    if slash_form.startswith("//"):
        raise ReviewPathError("desktop queue root must be a drive path")
    if _DRIVE.match(slash_form):
        letter, parts = _windows_parts(text)
        drive = _mounted_drive(mount, letter)
        local = drive.joinpath(*parts)
        _reject_reparse_chain(local)
        return DesktopLocation(local, _windows_join(letter, parts), drive)
    local_input = Path(text).expanduser()
    if not local_input.is_absolute():
        raise ReviewPathError("desktop queue root must be absolute")
    _reject_dot_parts(local_input)
    try:
        relative = local_input.relative_to(mount)
    except ValueError as exc:
        raise ReviewPathError(
            "desktop queue root must be a Windows-visible path"
        ) from exc
    if (
        not relative.parts
        or len(relative.parts[0]) != 1
        or not relative.parts[0].isalpha()
    ):
        raise ReviewPathError("desktop queue root must be a Windows-visible path")
    if len(relative.parts) < 2:
        raise ReviewPathError("desktop queue root must be below the drive root")
    letter = relative.parts[0]
    parts = relative.parts[1:]
    drive = _mounted_drive(mount, letter)
    local = drive.joinpath(*parts)
    _reject_reparse_chain(local)
    return DesktopLocation(local, _windows_join(letter, parts), drive)


def publish_desktop_file_queue(
    *,
    batches_dir: Path,
    batch_id: str,
    document_id: int,
    queue_root: Path | str,
    work_requests: Sequence[WorkRequest],
    reviews_dir: Path | None = None,
    mount_root: Path | None = None,
) -> DesktopFileQueue:
    """Publish GPT and Grok task directories. No model is called."""

    location = resolve_desktop_queue_root(queue_root, mount_root=mount_root)
    batches = reviews_root(batches_dir)
    sealed = archive_document_dir(batches, batch_id, document_id)
    protected: list[Path] = [batches]
    if reviews_dir is not None:
        protected.append(reviews_root(reviews_dir))
    return _publish_at(
        sealed,
        location,
        work_requests,
        protected_roots=protected,
        expected_batch_id=require_batch_id(batch_id),
        expected_document_id=document_id,
    )


def load_desktop_file_queue(
    queue_root: Path | str,
    sealed_document_dir: Path,
    work_requests: Sequence[WorkRequest],
    *,
    reviews_dir: Path | None = None,
    mount_root: Path | None = None,
) -> DesktopFileQueue:
    """Re-read a queue against the sealed source and the approved work requests.

    The on-disk manifest is not authority. Expected bytes are rebuilt and
    compared. Provider result files are left as they are.
    """

    location = resolve_desktop_queue_root(queue_root, mount_root=mount_root)
    sealed = Path(sealed_document_dir).expanduser()
    protected: list[Path] = []
    if reviews_dir is not None:
        protected.append(reviews_root(reviews_dir))
    prepared = _prepare(
        sealed,
        location,
        work_requests,
        protected_roots=protected,
        expected_batch_id=None,
        expected_document_id=None,
    )
    lock = _acquire_lock(_lock_path(location, prepared.batch_id, prepared.document_id))
    try:
        return _verified(prepared)
    finally:
        _release_lock(lock)


def _publish_at(
    sealed_document_dir: Path,
    location: DesktopLocation,
    work_requests: Sequence[WorkRequest],
    *,
    protected_roots: Sequence[Path],
    expected_batch_id: str,
    expected_document_id: int,
) -> DesktopFileQueue:
    prepared = _prepare(
        Path(sealed_document_dir),
        location,
        work_requests,
        protected_roots=protected_roots,
        expected_batch_id=expected_batch_id,
        expected_document_id=expected_document_id,
    )
    _ensure_existing_chain(location.drive_local, location.local_path)
    lock = _acquire_lock(_lock_path(location, prepared.batch_id, prepared.document_id))
    try:
        return _publish_locked(prepared)
    finally:
        _release_lock(lock)


def _publish_locked(prepared: _Prepared) -> DesktopFileQueue:
    document = prepared.document
    payloads = prepared.payloads
    _reject_queue_directories(document)
    if _tasks_match(document, payloads) and not (document / _INDEX_NAME).exists():
        _require_exact(
            document,
            payloads,
            sealed_image=prepared.sealed_image,
            require_index=False,
        )
        _write_index(document, _index_bytes(payloads))
        loaded = _verified(prepared)
        _privatize_document(document, payloads)
        return loaded
    if _published_matches(document, payloads):
        loaded = _verified(prepared)
        _privatize_document(document, payloads)
        return loaded
    _reject_partial_or_incomplete(document)
    _make_document(prepared.location, document)
    partial = document / _PARTIAL
    created = False
    try:
        os.mkdir(partial, _DIR_MODE)
        os.chmod(partial, _DIR_MODE)
        created = True
        for provider, files in payloads.items():
            directory = partial / provider
            _write_task(directory, files)
            _reject_task_leak(directory, prepared.transcription)
            _reject_owned_names(_read_task_files(directory), provider=provider)
            _privatize_task(directory, files)
        for provider in _PROVIDERS:
            os.rename(partial / provider, document / provider)
        os.rmdir(partial)
        created = False
        _write_index(document, _index_bytes(payloads))
    except Exception:
        published = any((document / provider).exists() for provider in _PROVIDERS)
        if created and not published:
            shutil.rmtree(partial, ignore_errors=True)
        raise
    return _verified(prepared)


@dataclass(frozen=True, slots=True)
class _Prepared:
    location: DesktopLocation
    sealed: Path
    sealed_image: Path
    document: Path
    payloads: dict[str, dict[str, bytes]]
    transcription: str
    image: bytes
    batch_id: str
    document_id: int
    protected_roots: tuple[Path, ...]


def _prepare(
    sealed_document_dir: Path,
    location: DesktopLocation,
    work_requests: Sequence[WorkRequest],
    *,
    protected_roots: Sequence[Path],
    expected_batch_id: str | None,
    expected_document_id: int | None,
) -> _Prepared:
    sealed = Path(sealed_document_dir).expanduser()
    _reject_reparse_chain(sealed)
    source, manifest, document_hash, transcription, image = _sealed_page(
        sealed,
        expected_batch_id=expected_batch_id,
        expected_document_id=expected_document_id,
    )
    if expected_batch_id is not None and manifest.batch_id != expected_batch_id:
        raise ReviewIntegrityError(
            "sealed document identity does not match the request"
        )
    if (
        expected_document_id is not None
        and manifest.document_id != expected_document_id
    ):
        raise ReviewIntegrityError(
            "sealed document identity does not match the request"
        )
    document = _document_dir(location, manifest.batch_id, manifest.document_id)
    protected = tuple(protected_roots)
    _reject_document_location(document, sealed, protected)
    requests = tuple(work_requests)
    _pin_attempt(requests)
    approved = _approved_requests(
        requests,
        manifest=manifest,
        document_hash=document_hash,
    )
    payloads = _payloads(
        document,
        location,
        requests=approved,
        source_filename=Path(source.relative_path).name,
        sealed_relative_path=source.relative_path,
        manifest_language=manifest.language,
        mime_type=source.mime_type,
        document_hash=document_hash,
        image=image,
    )
    _reject_secret_payloads(payloads)
    for provider, files in payloads.items():
        _reject_owned_names(files, provider=provider)
    sealed_image = sealed.joinpath(*Path(source.relative_path).parts)
    return _Prepared(
        location=location,
        sealed=sealed,
        sealed_image=sealed_image,
        document=document,
        payloads=payloads,
        transcription=transcription,
        image=image,
        batch_id=manifest.batch_id,
        document_id=manifest.document_id,
        protected_roots=protected,
    )


def _pin_attempt(requests: Sequence[WorkRequest]) -> None:
    for request in requests:
        if request.attempt == _SUPPORTED_ATTEMPT:
            continue
        task_id = (
            f"{request.batch_id}:{request.document_id}:{request.stage}:"
            f"r{request.round_index}:{request.provider}:a{request.attempt}"
        )
        if (
            request.task_id == task_id
            and request.job_id == f"job:{task_id}"
            and request.output_relative_path
            == output_relative_path(
                request.stage, request.provider, request.round_index, request.attempt
            )
        ):
            raise ReviewIntegrityError("desktop queue supports attempt 1 only")


def _verified(prepared: _Prepared) -> DesktopFileQueue:
    document = prepared.document
    if (document / _PARTIAL).exists() or (document / _INDEX_PARTIAL).exists():
        raise ReviewIntegrityError("interrupted desktop queue publication is present")
    _reject_document_location(document, prepared.sealed, prepared.protected_roots)
    _require_exact(document, prepared.payloads, sealed_image=prepared.sealed_image)
    return _assemble(prepared)


def _payloads(
    document: Path,
    location: DesktopLocation,
    *,
    requests: Mapping[str, WorkRequest],
    source_filename: str,
    sealed_relative_path: str,
    manifest_language: Literal["en", "fr"],
    mime_type: str,
    document_hash: str,
    image: bytes,
) -> dict[str, dict[str, bytes]]:
    from transcription_review_agent.review.first_pass_contract import (
        first_pass_schema_text,
    )
    from transcription_review_agent.review.provider_output_contract import (
        render_output_example,
    )

    image_name: Literal["original.png", "original.jpg"] = (
        "original.png" if mime_type == "image/png" else "original.jpg"
    )
    width, height = _require_image(image, mime_type)
    digest = sha256_bytes(image)
    windows_document = _windows_document(
        location.windows_path, requests["gpt"].batch_id, requests["gpt"].document_id
    )
    payloads: dict[str, dict[str, bytes]] = {}
    for provider in _PROVIDERS:
        request = requests[provider]
        mechanism = _MECHANISM[provider]
        windows_task = f"{windows_document}\\{provider}"
        filename = output_filename("first_pass", provider)
        windows_output = f"{windows_task}\\{_OUTPUT_DIR}\\{filename}"
        prompt = _desktop_prompt(
            request=request,
            execution_mechanism=mechanism,
            image_name=image_name,
            byte_size=len(image),
            image_sha256=digest,
            windows_output_file=windows_output,
        ).encode("utf-8")
        record = DesktopTaskManifest(
            provider=provider,  # type: ignore[arg-type]
            execution_mechanism=mechanism,
            batch_id=request.batch_id,
            document_id=request.document_id,
            page_id=_PAGE_ID,
            language=manifest_language,
            source_filename=source_filename,
            sealed_relative_path=sealed_relative_path,
            mime_type=mime_type,  # type: ignore[arg-type]
            byte_size=len(image),
            sha256=digest,
            width=width,
            height=height,
            source_bundle_sha256=document_hash,
            image_filename=image_name,
            prompt_sha256=sha256_bytes(prompt),
            local_task_directory=str(document / provider),
            windows_task_directory=windows_task,
            windows_output_file=windows_output,
            queue_output_relative_path=f"{_OUTPUT_DIR}/{filename}",
            review_output_relative_path=request.output_relative_path,
        )
        payloads[provider] = {
            image_name: image,
            "prompt.txt": prompt,
            "work_request.json": canonical_json(request).encode("utf-8"),
            "output_schema.json": first_pass_schema_text().encode("utf-8"),
            "output_example.json": render_output_example(request),
            "manifest.json": canonical_json(record).encode("utf-8"),
        }
    return payloads


def _desktop_prompt(
    *,
    request: WorkRequest,
    execution_mechanism: str,
    image_name: str,
    byte_size: int,
    image_sha256: str,
    windows_output_file: str,
) -> str:
    return (
        f"{_prompt(request=request, execution_mechanism=execution_mechanism, image_name=image_name, byte_size=byte_size, image_sha256=image_sha256)}"
        "Write the model transcription to this Windows file and to no other path:\n"
        f"{windows_output_file}\n"
        "This task directory does not already contain that transcription.\n"
        "output_example.json shows the JSON shape. It is not a reading of this page.\n"
        "Copy output_relative_path from work_request.json into the JSON object.\n"
        "That value identifies the later review file. It is not a second folder.\n"
        "Before transcribing, confirm that the parent directory contains "
        "desktop_queue.json.\n"
        "That index is the publication marker. If it is missing, or this "
        "directory is under .partial, stop and write nothing.\n"
        "Do not list or open the other provider directory.\n"
    )


def _assemble(prepared: _Prepared) -> DesktopFileQueue:
    document = prepared.document
    tasks: dict[str, DesktopProviderTask] = {}
    for provider, files in prepared.payloads.items():
        directory = document / provider
        sibling = document / _other(provider)
        record = DesktopTaskManifest.model_validate_json(files["manifest.json"])
        image_name = record.image_filename
        result_name = output_filename("first_pass", provider)
        result = _result_slot(directory / _OUTPUT_DIR, result_name, sibling)
        _reject_result_link(result, prepared.sealed_image, sibling)
        tasks[provider] = DesktopProviderTask(
            provider=provider,  # type: ignore[arg-type]
            execution_mechanism=record.execution_mechanism,
            directory=directory,
            windows_directory=record.windows_task_directory,
            image_path=directory / image_name,
            prompt_path=directory / "prompt.txt",
            manifest_path=directory / "manifest.json",
            output_directory=directory / _OUTPUT_DIR,
            windows_output_file=record.windows_output_file,
            result_path=result,
            result_present=result.is_file() and not _path_is_reparse(result),
            sha256=record.sha256,
        )
    _reject_shared_output(tasks["gpt"], tasks["grok"])
    windows_document = _windows_document(
        prepared.location.windows_path, prepared.batch_id, prepared.document_id
    )
    return DesktopFileQueue(
        queue_root=prepared.location.local_path,
        windows_queue_root=prepared.location.windows_path,
        document_directory=document,
        windows_document_directory=windows_document,
        batch_id=prepared.batch_id,
        document_id=prepared.document_id,
        page_id=_PAGE_ID,
        source_sha256=sha256_bytes(prepared.image),
        source_bundle_sha256=DesktopTaskManifest.model_validate_json(
            prepared.payloads["gpt"]["manifest.json"]
        ).source_bundle_sha256,
        gpt=tasks["gpt"],
        grok=tasks["grok"],
    )


def _require_exact(
    document: Path,
    payloads: Mapping[str, Mapping[str, bytes]],
    *,
    sealed_image: Path,
    require_index: bool = True,
) -> None:
    """Compare task bytes with the sealed expectation. Do not rewrite them."""

    _reject_queue_directories(document)
    if not document.is_dir():
        raise ReviewIntegrityError("incomplete desktop queue")
    _require_document_entries(document, require_index=require_index)
    for provider, expected in payloads.items():
        directory = document / provider
        sibling = document / _other(provider)
        present = _read_task_files(directory)
        if set(present) != set(expected):
            raise ReviewIntegrityError(
                "staged provider directory contains an unexpected file"
            )
        for name, payload in expected.items():
            path = directory / name
            _reject_hardlink(path, (sealed_image, sibling / name))
            if present[name] != payload:
                raise ReviewIntegrityError(
                    "staged task does not match the sealed source"
                )
        filename = output_filename("first_pass", provider)
        result = _result_slot(directory / _OUTPUT_DIR, filename, sibling)
        _reject_result_link(result, sealed_image, sibling)
    if not require_index:
        return
    index_path = document / _INDEX_NAME
    _reject_if_reparse(index_path)
    _reject_hardlink(index_path, (sealed_image,))
    if index_path.read_bytes() != _index_bytes(payloads):
        raise ReviewIntegrityError("staged task does not match the sealed source")


def _tasks_match(document: Path, payloads: Mapping[str, Mapping[str, bytes]]) -> bool:
    _reject_queue_directories(document)
    if not document.exists():
        return False
    if (document / _PARTIAL).exists() or (document / _INDEX_PARTIAL).exists():
        return False
    states = [(document / provider).exists() for provider in _PROVIDERS]
    if not any(states):
        return False
    if not all(states):
        raise ReviewIntegrityError("incomplete desktop queue")
    for provider, files in payloads.items():
        directory = document / provider
        if not _task_complete(directory):
            raise ReviewIntegrityError("incomplete desktop queue")
        present = _read_task_files(directory)
        if set(present) != set(files):
            raise ReviewIntegrityError(
                "staged provider directory contains an unexpected file"
            )
        if any(present[name] != payload for name, payload in files.items()):
            raise ReviewIntegrityError("refusing to change an existing desktop queue")
        filename = output_filename("first_pass", provider)
        _result_slot(directory / _OUTPUT_DIR, filename, document / _other(provider))
    return True


def _published_matches(
    document: Path, payloads: Mapping[str, Mapping[str, bytes]]
) -> bool:
    if not _tasks_match(document, payloads):
        return False
    index_path = document / _INDEX_NAME
    if not index_path.exists():
        return False
    _reject_if_reparse(index_path)
    if index_path.read_bytes() != _index_bytes(payloads):
        raise ReviewIntegrityError("refusing to change an existing desktop queue")
    _require_document_entries(document)
    return True


def _reject_partial_or_incomplete(document: Path) -> None:
    _reject_if_reparse(document)
    if not document.exists():
        return
    if not document.is_dir():
        raise ReviewPathError("desktop queue root is not a directory")
    if (document / _PARTIAL).exists() or (document / _INDEX_PARTIAL).exists():
        raise ReviewIntegrityError("interrupted desktop queue publication is present")
    for provider in _PROVIDERS:
        directory = document / provider
        if directory.exists() and not _task_complete(directory):
            raise ReviewIntegrityError("incomplete desktop queue")
    if (document / _INDEX_NAME).exists() and not (document / _INDEX_NAME).is_file():
        raise ReviewIntegrityError("incomplete desktop queue")
    extras = {path.name for path in document.iterdir()} - {
        *_PROVIDERS,
        _INDEX_NAME,
    }
    if extras:
        raise ReviewIntegrityError("incomplete desktop queue")


def _task_complete(directory: Path) -> bool:
    if directory.is_symlink() or _path_is_reparse(directory) or not directory.is_dir():
        return False
    names = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    output = directory / _OUTPUT_DIR
    return (
        set(_TASK_FILES) <= names
        and ("original.png" in names or "original.jpg" in names)
        and output.is_dir()
        and not _path_is_reparse(output)
    )


def _write_task(directory: Path, files: Mapping[str, bytes]) -> None:
    _make_private_dir(directory)
    for name, payload in files.items():
        _write_new(directory / name, payload)
    output = directory / _OUTPUT_DIR
    _make_private_dir(output)
    if any(output.iterdir()):
        raise ReviewIntegrityError("desktop output directory is not empty")


def _write_index(document: Path, payload: bytes) -> None:
    partial = document / _INDEX_PARTIAL
    if partial.exists() or (document / _INDEX_NAME).exists():
        raise ReviewIntegrityError("interrupted desktop queue publication is present")
    _write_new(partial, payload)
    os.rename(partial, document / _INDEX_NAME)


def _index_bytes(payloads: Mapping[str, Mapping[str, bytes]]) -> bytes:
    gpt = DesktopTaskManifest.model_validate_json(payloads["gpt"]["manifest.json"])
    grok = DesktopTaskManifest.model_validate_json(payloads["grok"]["manifest.json"])
    index = DesktopQueueIndex(
        batch_id=gpt.batch_id,
        document_id=gpt.document_id,
        page_id=gpt.page_id,
        source_sha256=gpt.sha256,
        source_bundle_sha256=gpt.source_bundle_sha256,
        gpt_task_directory=gpt.windows_task_directory,
        grok_task_directory=grok.windows_task_directory,
        gpt_output_file=gpt.windows_output_file,
        grok_output_file=grok.windows_output_file,
    )
    return canonical_json(index).encode("utf-8")


def _read_task_files(directory: Path) -> dict[str, bytes]:
    _reject_if_reparse(directory)
    if not directory.is_dir():
        raise ReviewIntegrityError("incomplete desktop queue")
    _reject_if_reparse(directory / _OUTPUT_DIR)
    files: dict[str, bytes] = {}
    for path in directory.iterdir():
        if path.name == _OUTPUT_DIR:
            _reject_if_reparse(path)
            if not path.is_dir():
                raise ReviewIntegrityError("desktop output directory escapes the task")
            continue
        if path.is_symlink() or _path_is_reparse(path):
            _reject_symlink_target(path, directory.parent / _other(directory.name))
        if not path.is_file():
            raise ReviewIntegrityError("staged file escapes the provider directory")
        _reject_hardlink(path, ())
        files[path.name] = path.read_bytes()
    return files


def _result_slot(output: Path, filename: str, sibling: Path) -> Path:
    _reject_if_reparse(output)
    if not output.is_dir():
        raise ReviewIntegrityError("desktop output directory escapes the task")
    if _path_inside(output.resolve(), sibling.resolve()):
        raise ReviewIntegrityError(
            "staged file resolves into the other provider directory"
        )
    entries = list(output.iterdir())
    result = output / filename
    if not entries:
        return result
    if len(entries) != 1 or entries[0].name != filename:
        raise ReviewIntegrityError(
            "desktop output directory contains an unexpected file"
        )
    if (
        entries[0].is_symlink()
        or not entries[0].is_file()
        or _path_is_reparse(entries[0])
    ):
        raise ReviewIntegrityError("desktop result escapes the output directory")
    _reject_hardlink(entries[0], ())
    if _path_inside(entries[0].resolve(), sibling.resolve()):
        raise ReviewIntegrityError(
            "staged file resolves into the other provider directory"
        )
    return entries[0]


def _reject_shared_output(gpt: DesktopProviderTask, grok: DesktopProviderTask) -> None:
    if gpt.output_directory == grok.output_directory:
        raise ReviewIntegrityError("provider output directories are not independent")
    if _overlaps(gpt.output_directory, grok.output_directory):
        raise ReviewIntegrityError("provider output directories are not independent")
    if _overlaps(gpt.directory, grok.directory):
        raise ReviewIntegrityError("provider task directories are not independent")
    shared = gpt.directory.parent / "first_pass" / _OUTPUT_DIR
    if shared.exists():
        raise ReviewIntegrityError("provider output directories are not independent")


def _reject_owned_names(files: Mapping[str, bytes], *, provider: str) -> None:
    own = output_filename("first_pass", provider).encode("utf-8")
    other = output_filename("first_pass", _other(provider)).encode("utf-8")
    forbidden = (
        b"transcription.txt",
        b"review_bundle.json",
        b".derived.png",
        other,
    )
    for name, payload in files.items():
        if name in {"original.png", "original.jpg"}:
            continue
        checked = payload.replace(own, b"")
        for needle in forbidden:
            if needle in checked:
                raise ReviewIntegrityError(
                    "staged manifest or prompt names a forbidden input"
                )


def _reject_secret_payloads(payloads: Mapping[str, Mapping[str, bytes]]) -> None:
    from transcription_review_agent.review.original_image_stage import (
        _reject_secret_text,
    )

    named = {
        provider: {
            name: payload
            for name, payload in files.items()
            if name != "original.png" and name != "original.jpg"
        }
        for provider, files in payloads.items()
    }
    _reject_secret_text(named)


def _reject_task_leak(directory: Path, transcription: str) -> None:
    needle = transcription.encode("utf-8") if len(transcription) >= 12 else b""
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if _OUTPUT_DIR in path.relative_to(directory).parts:
            continue
        if path.name == "transcription.txt":
            raise ReviewIntegrityError(
                "archive transcription file entered a provider bundle"
            )
        if needle and needle in path.read_bytes():
            raise ReviewIntegrityError(
                "archive transcription text entered a provider bundle"
            )


def _reject_document_location(
    document: Path, sealed: Path, protected_roots: Sequence[Path]
) -> None:
    logical = document
    resolved = document.resolve()
    sealed_paths = (sealed, sealed.resolve())
    if any(
        _overlaps(candidate, boundary)
        for candidate in (logical, resolved)
        for boundary in sealed_paths
    ):
        raise ReviewPathError("refusing to publish a queue inside a sealed bundle")
    protected = [_repo_root(), *FORBIDDEN_WRITE_ROOTS, *protected_roots]
    for candidate in protected:
        if candidate is None:
            continue
        root = Path(candidate).expanduser()
        boundaries = (root, root.resolve())
        if any(
            _overlaps(place, boundary)
            for place in (logical, resolved)
            for boundary in boundaries
        ):
            raise ReviewPathError(
                "refusing to publish a queue inside a repository, review tree, or sealed bundle"
            )


def _document_dir(location: DesktopLocation, batch_id: str, document_id: int) -> Path:
    safe_batch = require_batch_id(batch_id)
    if document_id < 1:
        raise ReviewPathError("document_id must be a positive integer")
    return location.local_path / safe_batch / f"document_{document_id}"


def _windows_document(windows_root: str, batch_id: str, document_id: int) -> str:
    return f"{windows_root}\\{require_batch_id(batch_id)}\\document_{document_id}"


def _make_document(location: DesktopLocation, document: Path) -> None:
    _ensure_existing_chain(location.drive_local, document)


def _ensure_existing_chain(start: Path, end: Path) -> None:
    """Create missing directories. Do not change the mode of an existing directory."""

    current = start
    relative = end.relative_to(start)
    for part in relative.parts:
        current = current / part
        _create_dir(current)


def _create_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        _reject_if_reparse(path)
        if path.is_symlink() or not path.is_dir():
            raise ReviewPathError("desktop queue root is not a directory")
        return
    try:
        os.mkdir(path, _DIR_MODE)
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise ReviewPathError("desktop queue root is not a directory") from None
        return
    os.chmod(path, _DIR_MODE)


def _make_private_dir(path: Path) -> None:
    _reject_if_reparse(path)
    if path.is_symlink():
        raise ReviewIntegrityError("desktop queue path escapes through a symlink")
    if not path.exists():
        os.mkdir(path, _DIR_MODE)
    elif not path.is_dir():
        raise ReviewPathError("desktop queue root is not a directory")
    os.chmod(path, _DIR_MODE)


def _privatize_document(
    document: Path, payloads: Mapping[str, Mapping[str, bytes]]
) -> None:
    _reject_queue_directories(document)
    _tighten(document, _DIR_MODE)
    for provider, files in payloads.items():
        _privatize_task(document / provider, files)
    index = document / _INDEX_NAME
    if index.exists():
        _tighten(index, _FILE_MODE)


def _privatize_task(directory: Path, files: Mapping[str, bytes]) -> None:
    _reject_if_reparse(directory)
    _reject_if_reparse(directory / _OUTPUT_DIR)
    _tighten(directory, _DIR_MODE)
    for name in files:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ReviewIntegrityError("staged file escapes the provider directory")
        _tighten(path, _FILE_MODE)
    output = directory / _OUTPUT_DIR
    _reject_if_reparse(output)
    if not output.is_dir():
        raise ReviewIntegrityError("desktop output directory escapes the task")
    _tighten(output, _DIR_MODE)


def _tighten(path: Path, target: int) -> None:
    _reject_if_reparse(path)
    if path.is_symlink():
        raise ReviewIntegrityError("desktop queue path escapes through a symlink")
    current = stat.S_IMODE(path.lstat().st_mode)
    if current == target:
        return
    if current & ~target:
        os.chmod(path, target)
        return
    raise ReviewIntegrityError("desktop queue permissions are not private")


def _windows_name(value: str) -> str:
    return value.rstrip("\\").rsplit("\\", 1)[-1]


def _windows_parts(text: str) -> tuple[str, tuple[str, ...]]:
    slash_form = text.replace("\\", "/").rstrip("/")
    if text.startswith("\\\\") or slash_form.startswith("//"):
        raise ReviewPathError("desktop queue root must be a drive path")
    match = _DRIVE.match(slash_form)
    if match is None:
        raise ReviewPathError("desktop queue root must be a drive path")
    raw_parts = match.group(2).split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ReviewPathError("desktop queue path is not safe")
    return match.group(1), tuple(raw_parts)


def _windows_join(letter: str, parts: Sequence[str]) -> str:
    return letter.upper() + ":\\" + "\\".join(parts)


def _mounted_drive(mount: Path, letter: str) -> Path:
    drive = mount / letter.lower()
    _require_real_directory(drive, "desktop drive is not mounted")
    return drive


def _require_real_directory(path: Path, message: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ReviewPathError(message)


def _reject_dot_parts(path: Path) -> None:
    if any(part in {".", ".."} for part in path.parts):
        raise ReviewPathError("desktop queue path is not safe")


def _reject_reparse_chain(path: Path) -> None:
    if not path.is_absolute():
        raise ReviewPathError("desktop queue root must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        if part in {".", ".."}:
            raise ReviewPathError("desktop queue path is not safe")
        current = current / part
        _reject_if_reparse(current)


def _reject_if_reparse(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if stat.S_ISLNK(path.lstat().st_mode):
        raise ReviewIntegrityError(
            "desktop queue path escapes through a symlink or reparse point"
        )
    if os.name != "nt":
        return
    attrs = _windows_file_attributes(path)
    if attrs is None:
        raise ReviewPathError("desktop queue cannot verify Windows reparse points")
    if _has_reparse_attribute(attrs):
        raise ReviewIntegrityError(
            "desktop queue path escapes through a symlink or reparse point"
        )


def _path_is_reparse(path: Path) -> bool:
    try:
        _reject_if_reparse(path)
    except ReviewIntegrityError:
        return True
    return False


def _has_reparse_attribute(attrs: int) -> bool:
    return bool(attrs & _REPARSE_ATTRIBUTE)


def _windows_file_attributes(path: Path) -> int | None:
    if os.name != "nt":
        return None
    import ctypes

    windll = getattr(ctypes, "windll")
    get = windll.kernel32.GetFileAttributesW
    get.argtypes = (ctypes.c_wchar_p,)
    get.restype = ctypes.c_uint32
    value = int(get(os.fspath(path)))
    if value == _INVALID_FILE_ATTRIBUTES:
        return None
    return value


def _reject_hardlink(path: Path, others: Sequence[Path]) -> None:
    if not path.exists() or path.is_symlink():
        return
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        return
    if info.st_nlink > 1:
        raise ReviewIntegrityError("desktop queue file is a hardlink")
    for other in others:
        if not other.exists() or other.is_symlink():
            continue
        other_info = other.lstat()
        if (
            stat.S_ISREG(other_info.st_mode)
            and other_info.st_dev == info.st_dev
            and other_info.st_ino == info.st_ino
        ):
            raise ReviewIntegrityError("desktop queue file is a hardlink")


def _reject_result_link(result: Path, sealed_image: Path, sibling: Path) -> None:
    if not result.exists():
        return
    targets: list[Path] = [sealed_image]
    if sibling.exists() and not sibling.is_symlink():
        targets.extend(path for path in sibling.rglob("*") if path.is_file())
    _reject_if_reparse(result)
    _reject_hardlink(result, tuple(targets))


def _lock_path(location: DesktopLocation, batch_id: str, document_id: int) -> Path:
    return location.local_path / (
        f".desktop-queue-{require_batch_id(batch_id)}-document-{document_id}.lock"
    )


def _acquire_lock(path: Path) -> int:
    _reject_if_reparse(path.parent)
    _reject_if_reparse(path)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ReviewIntegrityError(
                "desktop queue lock escapes through a symlink or reparse point"
            ) from exc
        raise ReviewPathError(
            "desktop queue publication requires an exclusive file lock"
        ) from exc
    try:
        _reject_opened_lock(path, fd)
        if os.name == "nt":
            import msvcrt

            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
            locking = getattr(msvcrt, "locking")
            locking(fd, getattr(msvcrt, "LK_LOCK"), 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
    except Exception as exc:
        os.close(fd)
        if isinstance(exc, (ReviewIntegrityError, ReviewPathError)):
            raise
        raise ReviewPathError(
            "desktop queue publication requires an exclusive file lock"
        ) from exc
    return fd


def _reject_opened_lock(path: Path, fd: int) -> None:
    """Confirm the opened descriptor is the lock file itself."""

    _reject_if_reparse(path)
    opened = os.fstat(fd)
    linked = path.lstat()
    if (
        stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_dev != linked.st_dev
        or opened.st_ino != linked.st_ino
    ):
        raise ReviewIntegrityError(
            "desktop queue lock escapes through a symlink or reparse point"
        )


def _release_lock(fd: int) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            locking = getattr(msvcrt, "locking")
            locking(fd, getattr(msvcrt, "LK_UNLCK"), 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _queue_directories(document: Path) -> tuple[Path, ...]:
    return (
        document,
        *(document / provider for provider in _PROVIDERS),
        *(document / provider / _OUTPUT_DIR for provider in _PROVIDERS),
    )


def _reject_queue_directories(document: Path) -> None:
    """Reject a symlink or reparse point at every owned directory level."""

    for path in _queue_directories(document):
        _reject_if_reparse(path)


def _require_document_entries(document: Path, *, require_index: bool = True) -> None:
    _reject_queue_directories(document)
    if not document.is_dir():
        raise ReviewIntegrityError("incomplete desktop queue")
    names = {path.name for path in document.iterdir()}
    expected = set(_PROVIDERS)
    if require_index:
        expected.add(_INDEX_NAME)
    if names != expected:
        raise ReviewIntegrityError("incomplete desktop queue")
    for provider in _PROVIDERS:
        path = document / provider
        _reject_if_reparse(path)
        if not path.is_dir():
            raise ReviewIntegrityError("staged file escapes the provider directory")
        _reject_if_reparse(path / _OUTPUT_DIR)


def _reject_symlink_target(path: Path, other: Path) -> None:
    if _path_inside(path.resolve(), other.resolve()):
        raise ReviewIntegrityError(
            "staged file resolves into the other provider directory"
        )
    raise ReviewIntegrityError("staged file escapes the provider directory")


def _contained(root: Path, candidate: Path, other: Path) -> Path:
    if candidate.is_symlink():
        _reject_symlink_target(candidate, other)
    resolved = candidate.resolve()
    if _path_inside(resolved, other.resolve()):
        raise ReviewIntegrityError(
            "staged file resolves into the other provider directory"
        )
    if not _path_inside(resolved, root.resolve()):
        raise ReviewIntegrityError("staged file escapes the provider directory")
    if not resolved.is_file():
        raise ReviewIntegrityError("incomplete desktop queue")
    return resolved


def _path_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or right in left.parents or left in right.parents


def _other(provider: str) -> str:
    if provider == "gpt":
        return "grok"
    if provider == "grok":
        return "gpt"
    raise ReviewPathError("provider must be gpt or grok")
