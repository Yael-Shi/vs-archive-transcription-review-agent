"""Stage one sealed original page for independent GPT and Grok first passes.

The copy under ``/tmp`` is the model-facing image. It is the sealed source
bytes, not the derived geometry PNG written at intake. Staging does not launch
a provider, upload an image, or change ``review-provider-output-v1``.

``execution_mechanism`` records how a later milestone may run the stage.
``codex_cli`` and ``gpt_work`` are different mechanisms for provider ``gpt``.
``grok_composer`` is the Grok Desktop composer attachment, and this module
does not import that adapter. The review output's ``provider_interface`` stays
``GPT_WORK`` or ``GROK`` and is copied from the approved work request.

Codex ``--sandbox read-only`` is not treated as sibling isolation. The Codex
sandbox documentation describes that mode as reads without writes, and the
Linux bubblewrap sandbox bind-mounts ``/`` read-only. The execution directory
prepared here is a separate tree whose parent contains only that directory.
That is the boundary this module enforces. It is not a mount namespace.
"""

from __future__ import annotations

import os
import stat
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from transcription_review_agent.bundling.acquire import _contains_download_credentials
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.layout import (
    TRANSCRIPTION_RELATIVE_PATH,
    assert_safe_relative_path,
)
from transcription_review_agent.bundling.manifest import (
    DocumentBundleManifest,
    SourceArtifact,
)
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.bundling.verify import verify_document_bundle
from transcription_review_agent.config import FORBIDDEN_WRITE_ROOTS
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.controller import (
    _instructions,
    _reject_transcription_leak,
)
from transcription_review_agent.review.crop import png_dimensions
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
    ReviewPolicyError,
)
from transcription_review_agent.review.first_pass_contract import first_pass_schema_text
from transcription_review_agent.review.jpegutil import jpeg_dimensions
from transcription_review_agent.review.models import (
    FORBIDDEN_INPUTS,
    PROVIDER_OUTPUT_SCHEMA_VERSION,
    FirstPassOutput,
    WorkRequest,
    interface_for,
)
from transcription_review_agent.review.paths import (
    archive_document_dir,
    output_relative_path,
    reviews_root,
)
from transcription_review_agent.review.provider_output_contract import (
    identity_from_request,
    render_output_example,
)

STAGE_SCHEMA_VERSION: Final[Literal["original-image-stage-v1"]] = (
    "original-image-stage-v1"
)
_TMP_ROOT = Path("/tmp")
_PARTIAL = ".partial"
_CODEX_CWD = "cwd"
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_PROVIDERS = ("gpt", "grok")
_PAGE_ID = "page_001"
_IMAGE_NAME = {"image/png": "original.png", "image/jpeg": "original.jpg"}
_ALLOWED_MECHANISM: dict[str, frozenset[str]] = {
    "gpt": frozenset({"codex_cli", "gpt_work"}),
    "grok": frozenset({"grok_composer"}),
}
ProviderName = Literal["gpt", "grok"]
ExecutionMechanism = Literal["codex_cli", "gpt_work", "grok_composer"]
ImageMime = Literal["image/png", "image/jpeg"]


class OriginalImageStageManifest(BaseModel):
    """Provenance for one provider's staged original. Not a review format."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["original-image-stage-v1"] = STAGE_SCHEMA_VERSION
    provider: ProviderName
    execution_mechanism: ExecutionMechanism
    batch_id: str = Field(min_length=1)
    document_id: int = Field(ge=1)
    page_id: str = Field(pattern=r"^page_[0-9]{3}$")
    reading_order: Literal[1] = 1
    language: Literal["en", "fr"]
    source_filename: str = Field(min_length=1)
    sealed_relative_path: str = Field(min_length=1)
    mime_type: ImageMime
    byte_size: int = Field(ge=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    image_filename: Literal["original.png", "original.jpg"]
    prompt_filename: Literal["prompt.txt"] = "prompt.txt"
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    staging_location: str = Field(min_length=1)
    archive_transcription_included: Literal[False] = False
    api_payload_included: Literal[False] = False
    derived_image_included: Literal[False] = False
    other_provider_output_included: Literal[False] = False

    @field_validator("sealed_relative_path")
    @classmethod
    def _sealed_path(cls, value: str) -> str:
        return assert_safe_relative_path(value)

    @field_validator("source_filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        if value != Path(value).name or value in {".", ".."}:
            raise ValueError("source filename must be a single path segment")
        return value

    @field_validator("staging_location")
    @classmethod
    def _absolute_location(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("staging location must be absolute")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> OriginalImageStageManifest:
        if self.execution_mechanism not in _ALLOWED_MECHANISM[self.provider]:
            raise ValueError("execution mechanism does not match provider")
        if self.image_filename != _IMAGE_NAME[self.mime_type]:
            raise ValueError("image filename does not match the sealed mime type")
        if Path(self.sealed_relative_path).name != self.source_filename:
            raise ValueError("source filename does not match the sealed path")
        if Path(self.staging_location).name != self.provider:
            raise ValueError("staging location does not name this provider")
        return self


@dataclass(frozen=True, slots=True)
class CodexExecutionLayout:
    """Working directory for a later Codex process. Codex is not launched.

    ``working_directory`` contains the GPT image and the first-pass instruction
    files. The Grok stage is not in that directory or its parent. File modes
    are ``0600`` and directory modes are ``0700``, which stop other users.
    They do not stop this user. Codex read-only mode is documented to allow
    reads, so this layout is a workflow boundary, not a sandbox.
    """

    working_directory: Path
    image_path: Path
    prompt_path: Path
    work_request_path: Path
    output_schema_path: Path
    output_example_path: Path


@dataclass(frozen=True, slots=True)
class StageExecution:
    """How one provider may later consume its stage. Nothing is launched."""

    provider: ProviderName
    execution_mechanism: ExecutionMechanism


@dataclass(frozen=True, slots=True)
class ProviderStage:
    provider: ProviderName
    execution_mechanism: ExecutionMechanism
    directory: Path
    image_path: Path
    prompt_path: Path
    manifest_path: Path
    sha256: str
    byte_size: int
    page_id: str
    source_filename: str


@dataclass(frozen=True, slots=True)
class OriginalImageStage:
    stage_root: Path
    batch_id: str
    document_id: int
    page_id: str
    source_sha256: str
    source_bundle_sha256: str
    gpt: ProviderStage
    grok: ProviderStage
    codex_execution: CodexExecutionLayout | None = None


def stage_sealed_document(
    *,
    batches_dir: Path,
    batch_id: str,
    document_id: int,
    stage_root: Path,
    executions: Sequence[StageExecution],
    work_requests: Sequence[WorkRequest],
    reviews_dir: Path | None = None,
    codex_work_root: Path | None = None,
) -> OriginalImageStage:
    """Stage the sealed page for GPT and Grok. The sealed bundle is only read."""

    batches = reviews_root(batches_dir)
    sealed = archive_document_dir(batches, batch_id, document_id)
    protected: list[Path] = [batches]
    if reviews_dir is not None:
        protected.append(reviews_root(reviews_dir))
    return stage_sealed_original(
        sealed,
        stage_root,
        executions,
        protected_roots=protected,
        expected_batch_id=batch_id,
        expected_document_id=document_id,
        work_requests=work_requests,
        codex_work_root=codex_work_root,
    )


def stage_sealed_original(
    sealed_document_dir: Path,
    stage_root: Path,
    executions: Sequence[StageExecution],
    *,
    protected_roots: Sequence[Path] = (),
    expected_batch_id: str | None = None,
    expected_document_id: int | None = None,
    work_requests: Sequence[WorkRequest] = (),
    codex_work_root: Path | None = None,
) -> OriginalImageStage:
    """Copy one sealed PNG or JPEG into separate GPT and Grok stages."""

    requested = _executions(executions)
    sealed = Path(sealed_document_dir).expanduser().resolve()
    root = _require_stage_root(stage_root, sealed, protected_roots)
    _require_codex_root_choice(requested, codex_work_root)
    work_root = _require_codex_root(codex_work_root, root, sealed, protected_roots)
    source, manifest, document_hash, transcription, sealed_bytes = _sealed_page(
        sealed,
        expected_batch_id=expected_batch_id,
        expected_document_id=expected_document_id,
    )
    requests = _approved_requests(
        work_requests,
        manifest=manifest,
        document_hash=document_hash,
    )
    payloads = _payloads(
        root,
        requested,
        requests=requests,
        source=source,
        manifest=manifest,
        document_hash=document_hash,
        image=sealed_bytes,
    )
    _reject_secret_text(payloads)
    if _published_matches(root, payloads):
        _privatize_stages(root, payloads)
        codex = _publish_codex_workdir(
            work_root,
            _codex_files(payloads),
            transcription,
            required=requested["gpt"] == "codex_cli",
        )
        return _finish(root, sealed, codex)
    _reject_partial_or_incomplete(root)
    _mkdir_private(root)
    partial = root / _PARTIAL
    created = False
    try:
        try:
            os.mkdir(partial, _DIR_MODE)
            os.chmod(partial, _DIR_MODE)
        except FileExistsError as exc:
            raise ReviewIntegrityError(
                "interrupted original-image staging is present"
            ) from exc
        created = True
        for provider, files in payloads.items():
            directory = partial / provider
            _mkdir_private(directory)
            for name, payload in files.items():
                _write_new(directory / name, payload)
            _reject_transcription_leak(directory, transcription)
            _reject_provider_layout(directory, provider, files)
            _privatize_directory(directory)
        _publish(root, partial)
    except Exception:
        published = any((root / provider).exists() for provider in _PROVIDERS)
        if created and not published:
            shutil.rmtree(partial, ignore_errors=True)
        raise
    codex = _publish_codex_workdir(
        work_root,
        _codex_files(payloads),
        transcription,
        required=requested["gpt"] == "codex_cli",
    )
    return _finish(root, sealed, codex)


def load_staged_original(
    stage_root: Path,
    sealed_document_dir: Path,
    *,
    codex_work_root: Path | None = None,
) -> OriginalImageStage:
    """Re-read a stage and the sealed source. Refuse mutation or a partial pair."""

    sealed = Path(sealed_document_dir).expanduser().resolve()
    root = _existing_stage_root(stage_root)
    if (root / _PARTIAL).exists():
        raise ReviewIntegrityError("interrupted original-image staging is present")
    preview = _finish(root, sealed, None, require_codex=False)
    if preview.gpt.execution_mechanism != "codex_cli":
        if codex_work_root is not None:
            raise ReviewPolicyError(
                "a codex working directory belongs only to execution_mechanism codex_cli"
            )
        return preview
    work_root = _require_codex_root(codex_work_root, root, sealed, ())
    files = _codex_files({"gpt": _instruction_bytes(preview.gpt.directory)})
    cwd = _publish_codex_workdir(
        work_root, files, _transcription(sealed), required=True
    )
    return _finish(root, sealed, cwd)


def _finish(
    root: Path,
    sealed: Path,
    codex: CodexExecutionLayout | None,
    *,
    require_codex: bool = True,
) -> OriginalImageStage:
    if (root / _PARTIAL).exists():
        raise ReviewIntegrityError("interrupted original-image staging is present")
    source, manifest, document_hash, transcription, sealed_bytes = _sealed_page(sealed)
    providers: dict[str, ProviderStage] = {}
    digests: list[str] = []
    for provider in _PROVIDERS:
        directory = _provider_dir(root, provider)
        if not directory.is_dir():
            raise ReviewIntegrityError("incomplete original-image staging")
        record = _load_provider(
            directory,
            provider,
            source_filename=Path(source.relative_path).name,
            sealed_relative_path=source.relative_path,
            document_hash=document_hash,
            sealed_bytes=sealed_bytes,
            transcription=transcription,
            batch_id=manifest.batch_id,
            document_id=manifest.document_id,
            other=_provider_dir(root, "grok" if provider == "gpt" else "gpt"),
        )
        if record.page_id != _PAGE_ID:
            raise ReviewIntegrityError("staged page identity does not match")
        if manifest.language != _manifest_file(directory).language:
            raise ReviewIntegrityError(
                "staged language does not match the sealed source"
            )
        providers[provider] = record
        digests.append(record.sha256)
    if len(set(digests)) != 1 or digests[0] != sha256_bytes(sealed_bytes):
        raise ReviewIntegrityError("staged originals do not share the sealed digest")
    gpt_record = _manifest_file(_provider_dir(root, "gpt"))
    grok_record = _manifest_file(_provider_dir(root, "grok"))
    if (
        gpt_record.batch_id != grok_record.batch_id
        or gpt_record.document_id != grok_record.document_id
        or gpt_record.page_id != grok_record.page_id
        or gpt_record.batch_id != manifest.batch_id
        or gpt_record.document_id != manifest.document_id
    ):
        raise ReviewIntegrityError("staged document identity does not match")
    if gpt_record.execution_mechanism == "codex_cli" and require_codex:
        _require_codex_layout(codex, providers["grok"].directory)
    elif codex is not None:
        raise ReviewIntegrityError(
            "codex working directory does not match execution_mechanism"
        )
    return OriginalImageStage(
        stage_root=root,
        batch_id=gpt_record.batch_id,
        document_id=gpt_record.document_id,
        page_id=gpt_record.page_id,
        source_sha256=digests[0],
        source_bundle_sha256=document_hash,
        gpt=providers["gpt"],
        grok=providers["grok"],
        codex_execution=codex,
    )


def codex_staged_image(stage: OriginalImageStage) -> Path:
    """Image path for a later Codex CLI ``-i`` argument. Codex is not launched.

    The path is inside the Codex working directory, not the paired provider
    stage. Pass ``stage.codex_execution.working_directory`` as the process
    working directory.
    """

    if stage.codex_execution is None or stage.gpt.execution_mechanism != "codex_cli":
        raise ReviewPolicyError(
            "codex image transcription requires execution_mechanism codex_cli"
        )
    return stage.codex_execution.image_path


def grok_staged_image(stage: OriginalImageStage) -> Path:
    """Image path for a later Grok attachment. Grok is not launched."""

    if stage.grok.execution_mechanism != "grok_composer":
        raise ReviewPolicyError(
            "grok image attachment requires execution_mechanism grok_composer"
        )
    return stage.grok.image_path


def resolve_provider_file(stage_root: Path, provider: str, relative: str) -> Path:
    """Resolve one file inside a provider stage. Refuse traversal and the sibling."""

    if provider not in _PROVIDERS:
        raise ReviewPathError("provider must be gpt or grok")
    try:
        safe = assert_safe_relative_path(relative)
    except ValueError as exc:
        raise ReviewPathError(
            "staged path is not inside the provider directory"
        ) from exc
    root = Path(stage_root).expanduser().resolve()
    directory = _provider_dir(root, provider)
    other = _provider_dir(root, "grok" if provider == "gpt" else "gpt")
    candidate = directory.joinpath(*Path(safe).parts)
    return _contained_file(directory, candidate, other)


def _executions(
    executions: Sequence[StageExecution],
) -> dict[str, ExecutionMechanism]:
    if len(executions) != 2:
        raise ReviewPolicyError("original-image staging prepares gpt and grok together")
    requested: dict[str, ExecutionMechanism] = {}
    for item in executions:
        if item.provider not in _PROVIDERS:
            raise ReviewPathError("provider must be gpt or grok")
        if item.provider in requested:
            raise ReviewPolicyError(
                "original-image staging prepares each provider once"
            )
        if item.execution_mechanism not in _ALLOWED_MECHANISM[item.provider]:
            raise ReviewPolicyError("execution mechanism does not match provider")
        requested[item.provider] = item.execution_mechanism
    if set(requested) != set(_PROVIDERS):
        raise ReviewPolicyError("original-image staging prepares gpt and grok together")
    return requested


def _existing_stage_root(stage_root: Path) -> Path:
    root = Path(stage_root).expanduser().resolve()
    tmp = _TMP_ROOT.resolve()
    if root == tmp or tmp not in root.parents:
        raise ReviewPathError("original-image stages must be created under /tmp")
    return root


def _require_stage_root(
    stage_root: Path, sealed: Path, protected_roots: Sequence[Path]
) -> Path:
    root = _existing_stage_root(stage_root)
    if root == sealed or _overlaps(root, sealed):
        raise ReviewPathError("refusing to stage inside a sealed bundle")
    protected = [_repo_root(), *FORBIDDEN_WRITE_ROOTS, *protected_roots]
    for candidate in protected:
        if candidate is None:
            continue
        forbidden = Path(candidate).expanduser().resolve()
        if _overlaps(root, forbidden):
            raise ReviewPathError(
                "refusing to stage inside a repository, review tree, or sealed bundle"
            )
    if root.exists() and not root.is_dir():
        raise ReviewPathError("original-image stage root is not a directory")
    return root


def _overlaps(left: Path, right: Path) -> bool:
    return left == right or right in left.parents or left in right.parents


def _repo_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        package = parent / "src" / "transcription_review_agent"
        if (parent / "pyproject.toml").is_file() and package.is_dir():
            return parent
    return None


def _sealed_page(
    sealed: Path,
    *,
    expected_batch_id: str | None = None,
    expected_document_id: int | None = None,
) -> tuple[SourceArtifact, DocumentBundleManifest, str, str, bytes]:
    if not sealed.is_dir():
        raise ReviewPathError("sealed document directory is missing")
    manifest = verify_document_bundle(sealed)
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
    if len(manifest.sources) != 1:
        raise ReviewPolicyError("original-image staging supports one page")
    source = manifest.sources[0]
    if source.mime_type not in _IMAGE_NAME:
        raise ReviewPolicyError("unsupported image format")
    relative = assert_safe_relative_path(source.relative_path)
    path = _contained_file(sealed, sealed.joinpath(*Path(relative).parts), None)
    data = path.read_bytes()
    if sha256_bytes(data) != source.sha256 or len(data) != source.size_bytes:
        raise ReviewIntegrityError("sealed source changed while staging")
    _require_image(data, source.mime_type)
    repeated = path.read_bytes()
    if repeated != data:
        raise ReviewIntegrityError("sealed source changed while staging")
    document_hash = sha256_bytes((sealed / "document_manifest.json").read_bytes())
    text_path = sealed / TRANSCRIPTION_RELATIVE_PATH
    try:
        transcription = text_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewIntegrityError(
            "sealed transcription text is not valid UTF-8"
        ) from exc
    return source, manifest, document_hash, transcription, data


def _require_image(data: bytes, mime_type: str) -> tuple[int, int]:
    try:
        if mime_type == "image/png":
            return png_dimensions(data)
        return jpeg_dimensions(data)
    except ValueError as exc:
        raise ReviewPolicyError(str(exc)) from exc


def _payloads(
    root: Path,
    requested: Mapping[str, ExecutionMechanism],
    *,
    requests: Mapping[str, WorkRequest],
    source: SourceArtifact,
    manifest: DocumentBundleManifest,
    document_hash: str,
    image: bytes,
) -> dict[str, dict[str, bytes]]:
    relative = source.relative_path
    filename = Path(relative).name
    mime_type: ImageMime = (
        "image/png" if source.mime_type == "image/png" else "image/jpeg"
    )
    width, height = _require_image(image, mime_type)
    image_name = _IMAGE_NAME[mime_type]
    digest = sha256_bytes(image)
    payloads: dict[str, dict[str, bytes]] = {}
    for provider in _PROVIDERS:
        mechanism = requested[provider]
        request = requests[provider]
        prompt = _prompt(
            request=request,
            execution_mechanism=mechanism,
            image_name=image_name,
            byte_size=len(image),
            image_sha256=digest,
        ).encode("utf-8")
        directory = root / provider
        record = OriginalImageStageManifest(
            provider=_provider_name(provider),
            execution_mechanism=mechanism,
            batch_id=manifest.batch_id,
            document_id=manifest.document_id,
            page_id=_PAGE_ID,
            language=manifest.language,
            source_filename=filename,
            sealed_relative_path=relative,
            mime_type=mime_type,
            byte_size=len(image),
            sha256=digest,
            width=width,
            height=height,
            source_bundle_sha256=document_hash,
            image_filename=_image_filename(image_name),
            prompt_sha256=sha256_bytes(prompt),
            staging_location=str(directory),
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


def _prompt(
    *,
    request: WorkRequest,
    execution_mechanism: str,
    image_name: str,
    byte_size: int,
    image_sha256: str,
) -> str:
    page_id = request.page_ids[0]
    return (
        f"{_instructions('first_pass')}"
        "The original page image in this directory is the authoritative source.\n"
        f"Image file: {image_name}\n"
        f"page_id: {page_id}\n"
        f"image_sha256: {image_sha256}\n"
        f"image_byte_size: {byte_size}\n"
        f"execution_mechanism: {execution_mechanism}\n"
        "execution_mechanism is not provider_interface.\n"
        f"provider_interface in the output is {request.provider_interface}.\n"
        f"The output schema_version is {PROVIDER_OUTPUT_SCHEMA_VERSION}.\n"
        "work_request.json, output_schema.json, and output_example.json are in "
        "this directory.\n"
    )


def _reject_secret_text(payloads: Mapping[str, Mapping[str, bytes]]) -> None:
    for files in payloads.values():
        for name, payload in files.items():
            if name in {
                "manifest.json",
                "prompt.txt",
                "work_request.json",
                "output_schema.json",
                "output_example.json",
            }:
                if _contains_download_credentials(payload):
                    raise ReviewIntegrityError(
                        "staged manifest or prompt contains a download credential"
                    )


def _published_matches(root: Path, payloads: Mapping[str, Mapping[str, bytes]]) -> bool:
    if (root / _PARTIAL).exists():
        return False
    states = [(root / provider).exists() for provider in _PROVIDERS]
    if not any(states):
        return False
    if not all(states):
        raise ReviewIntegrityError("incomplete original-image staging")
    for provider, files in payloads.items():
        directory = root / provider
        if directory.is_symlink() or not directory.is_dir():
            raise ReviewIntegrityError("incomplete original-image staging")
        present = {
            path.name: path.read_bytes()
            for path in directory.iterdir()
            if path.is_file() and not path.is_symlink()
        }
        if set(present) != set(files) or any(
            present[name] != files[name] for name in files
        ):
            raise ReviewIntegrityError(
                "refusing to change an existing original-image stage"
            )
    return True


def _reject_partial_or_incomplete(root: Path) -> None:
    if (root / _PARTIAL).exists():
        raise ReviewIntegrityError("interrupted original-image staging is present")
    for provider in _PROVIDERS:
        directory = root / provider
        if directory.exists() and not _directory_complete(directory):
            raise ReviewIntegrityError("incomplete original-image staging")


def _directory_complete(directory: Path) -> bool:
    if directory.is_symlink() or not directory.is_dir():
        return False
    names = {path.name for path in directory.iterdir() if path.is_file()}
    required = {
        "manifest.json",
        "prompt.txt",
        "work_request.json",
        "output_schema.json",
        "output_example.json",
    }
    return required <= names and ("original.png" in names or "original.jpg" in names)


def _publish(root: Path, partial: Path) -> None:
    try:
        for provider in _PROVIDERS:
            os.rename(partial / provider, root / provider)
        os.rmdir(partial)
    except OSError as exc:
        raise ReviewIntegrityError("incomplete original-image staging") from exc


def _provider_dir(root: Path, provider: str) -> Path:
    return root / provider


def _load_provider(
    directory: Path,
    provider: str,
    *,
    source_filename: str,
    sealed_relative_path: str,
    document_hash: str,
    sealed_bytes: bytes,
    transcription: str,
    batch_id: str,
    document_id: int,
    other: Path,
) -> ProviderStage:
    if directory.is_symlink():
        if _path_inside(directory.resolve(), other.resolve()):
            raise ReviewIntegrityError(
                "staged file resolves into the other provider directory"
            )
        raise ReviewIntegrityError("staged file escapes the provider directory")
    record = _manifest_file(directory)
    if record.provider != provider:
        raise ReviewIntegrityError("staged provider does not match its directory")
    if record.staging_location != str(directory):
        raise ReviewIntegrityError(
            "staged location does not match the provider directory"
        )
    if record.source_filename != source_filename:
        raise ReviewIntegrityError("staged source filename does not match")
    if record.sealed_relative_path != sealed_relative_path:
        raise ReviewIntegrityError("staged source path does not match")
    if record.source_bundle_sha256 != document_hash:
        raise ReviewIntegrityError(
            "staged bundle hash does not match the sealed source"
        )
    if record.byte_size != len(sealed_bytes) or record.sha256 != sha256_bytes(
        sealed_bytes
    ):
        raise ReviewIntegrityError("staged original does not match the sealed source")
    image_path = _contained_file(directory, directory / record.image_filename, other)
    prompt_path = _contained_file(directory, directory / record.prompt_filename, other)
    manifest_path = _contained_file(directory, directory / "manifest.json", other)
    image = image_path.read_bytes()
    prompt = prompt_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    _reject_secret_text(
        {provider: {"prompt.txt": prompt, "manifest.json": manifest_bytes}}
    )
    if image != sealed_bytes or sha256_bytes(image) != record.sha256:
        raise ReviewIntegrityError("staged original does not match the sealed source")
    if len(image) != record.byte_size:
        raise ReviewIntegrityError("staged original does not match the sealed source")
    if sha256_bytes(prompt) != record.prompt_sha256 or prompt != _prompt(
        request=_validated_work_request(
            directory,
            provider=provider,
            batch_id=batch_id,
            document_id=document_id,
            document_hash=document_hash,
            page_id=record.page_id,
        ),
        execution_mechanism=record.execution_mechanism,
        image_name=record.image_filename,
        byte_size=record.byte_size,
        image_sha256=record.sha256,
    ).encode("utf-8"):
        raise ReviewIntegrityError("staged prompt does not match its manifest")
    if manifest_bytes != canonical_json(record).encode("utf-8"):
        raise ReviewIntegrityError("staged manifest does not match its recorded bytes")
    names = {path.name for path in directory.iterdir() if not path.is_symlink()}
    expected = {
        record.image_filename,
        "prompt.txt",
        "work_request.json",
        "output_schema.json",
        "output_example.json",
        "manifest.json",
    }
    if names != expected or any(path.is_symlink() for path in directory.iterdir()):
        raise ReviewIntegrityError(
            "staged provider directory contains an unexpected file"
        )
    _reject_transcription_leak(directory, transcription)
    for forbidden in (
        b"transcription.txt",
        b"review_bundle.json",
        b".derived.png",
        b"transcription_gpt.json",
        b"transcription_grok.json",
    ):
        if forbidden in prompt or forbidden in manifest_bytes:
            raise ReviewIntegrityError(
                "staged manifest or prompt names a forbidden input"
            )
    return ProviderStage(
        provider=_provider_name(provider),
        execution_mechanism=record.execution_mechanism,
        directory=directory,
        image_path=image_path,
        prompt_path=prompt_path,
        manifest_path=manifest_path,
        sha256=record.sha256,
        byte_size=record.byte_size,
        page_id=record.page_id,
        source_filename=record.source_filename,
    )


def _manifest_file(directory: Path) -> OriginalImageStageManifest:
    path = directory / "manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ReviewIntegrityError("incomplete original-image staging")
    try:
        return OriginalImageStageManifest.model_validate_json(path.read_bytes())
    except ValueError as exc:
        raise ReviewIntegrityError("staged manifest is not readable") from exc


def _contained_file(root: Path, candidate: Path, other: Path | None) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if other is not None and _path_inside(resolved, other.resolve()):
        raise ReviewIntegrityError(
            "staged file resolves into the other provider directory"
        )
    if candidate.is_symlink() or not _path_inside(resolved, resolved_root):
        raise ReviewIntegrityError("staged file escapes the provider directory")
    if not resolved.is_file():
        raise ReviewIntegrityError("incomplete original-image staging")
    return resolved


def _path_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _provider_name(provider: str) -> ProviderName:
    if provider == "gpt":
        return "gpt"
    if provider == "grok":
        return "grok"
    raise ReviewPathError("provider must be gpt or grok")


def _image_filename(name: str) -> Literal["original.png", "original.jpg"]:
    if name == "original.png":
        return "original.png"
    if name == "original.jpg":
        return "original.jpg"
    raise ReviewIntegrityError("staged image filename does not match the sealed source")


def _reject_provider_layout(
    directory: Path, provider: str, files: Mapping[str, bytes]
) -> None:
    present = [path.name for path in directory.iterdir() if path.is_file()]
    if set(present) != set(files):
        raise ReviewIntegrityError("incomplete original-image staging")
    other = "grok" if provider == "gpt" else "gpt"
    for name, payload in files.items():
        written = (directory / name).read_bytes()
        if written != payload:
            raise ReviewIntegrityError("incomplete original-image staging")
        if other.encode("utf-8") in Path(name).name.encode("utf-8"):
            raise ReviewIntegrityError(
                "staged file resolves into the other provider directory"
            )


def _write_new(path: Path, payload: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, _FILE_MODE)
    view = memoryview(payload)
    try:
        os.fchmod(fd, _FILE_MODE)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ReviewIntegrityError("incomplete original-image staging")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _approved_requests(
    work_requests: Sequence[WorkRequest],
    *,
    manifest: DocumentBundleManifest,
    document_hash: str,
) -> dict[str, WorkRequest]:
    if len(work_requests) != 2:
        raise ReviewIntegrityError(
            "original-image staging requires the gpt and grok work requests"
        )
    found: dict[str, WorkRequest] = {}
    for request in work_requests:
        if request.provider in found:
            raise ReviewIntegrityError("work request provider is repeated")
        _check_request(
            request,
            provider=request.provider,
            batch_id=manifest.batch_id,
            document_id=manifest.document_id,
            document_hash=document_hash,
            page_id=_PAGE_ID,
        )
        found[request.provider] = request
    if set(found) != set(_PROVIDERS):
        raise ReviewIntegrityError(
            "original-image staging requires the gpt and grok work requests"
        )
    return found


def _check_request(
    request: WorkRequest,
    *,
    provider: str,
    batch_id: str,
    document_id: int,
    document_hash: str,
    page_id: str,
) -> None:
    if request.provider != provider:
        raise ReviewIntegrityError("work request provider does not match the stage")
    if request.provider_interface != interface_for(provider):
        raise ReviewIntegrityError(
            "work request provider_interface does not match provider"
        )
    if request.stage != "first_pass" or request.round_index != 0:
        raise ReviewIntegrityError("work request is not a first-pass request")
    if request.batch_id != batch_id or request.document_id != document_id:
        raise ReviewIntegrityError(
            "work request document identity does not match the sealed source"
        )
    if request.source_bundle_sha256 != document_hash:
        raise ReviewIntegrityError(
            "work request source hash does not match the sealed source"
        )
    if request.page_ids != [page_id]:
        raise ReviewIntegrityError(
            "work request page identity does not match the sealed source"
        )
    task_id = (
        f"{request.batch_id}:{request.document_id}:first_pass:"
        f"r{request.round_index}:{request.provider}:a{request.attempt}"
    )
    if request.task_id != task_id or request.job_id != f"job:{task_id}":
        raise ReviewIntegrityError("work request task identity does not match")
    if request.output_relative_path != output_relative_path(
        "first_pass", provider, 0, 1
    ):
        raise ReviewIntegrityError(
            "work request output path is not the first-pass output"
        )
    if list(request.forbidden_inputs) != list(FORBIDDEN_INPUTS):
        raise ReviewIntegrityError(
            "work request forbidden inputs do not match the first-pass contract"
        )
    example = FirstPassOutput.model_validate_json(render_output_example(request))
    identity = identity_from_request(request)
    for key, value in identity.items():
        if getattr(example, key) != value:
            raise ReviewIntegrityError(
                "output example identity does not match the work request"
            )
    if example.schema_version != "review-provider-output-v1":
        raise ReviewIntegrityError("output example is not review-provider-output-v1")
    if example.pages[0].page_id != page_id:
        raise ReviewIntegrityError(
            "output example page identity does not match the sealed source"
        )


def _validated_work_request(
    directory: Path,
    *,
    provider: str,
    batch_id: str,
    document_id: int,
    document_hash: str,
    page_id: str,
) -> WorkRequest:
    path = directory / "work_request.json"
    if not path.is_file() or path.is_symlink():
        raise ReviewIntegrityError("incomplete original-image staging")
    request = WorkRequest.model_validate_json(path.read_bytes())
    _check_request(
        request,
        provider=provider,
        batch_id=batch_id,
        document_id=document_id,
        document_hash=document_hash,
        page_id=page_id,
    )
    if path.read_bytes() != canonical_json(request).encode("utf-8"):
        raise ReviewIntegrityError("staged work request does not match its bytes")
    schema = directory / "output_schema.json"
    example = directory / "output_example.json"
    if schema.is_symlink() or example.is_symlink():
        raise ReviewIntegrityError("staged file escapes the provider directory")
    if schema.read_bytes() != first_pass_schema_text().encode("utf-8"):
        raise ReviewIntegrityError("staged output schema does not match the contract")
    if example.read_bytes() != render_output_example(request):
        raise ReviewIntegrityError(
            "staged output example does not match the work request"
        )
    return request


def _require_codex_root_choice(
    requested: Mapping[str, ExecutionMechanism],
    codex_work_root: Path | None,
) -> None:
    wants_codex = requested["gpt"] == "codex_cli"
    if wants_codex and codex_work_root is None:
        raise ReviewPolicyError("codex_cli staging requires a codex working directory")
    if not wants_codex and codex_work_root is not None:
        raise ReviewPolicyError(
            "a codex working directory belongs only to execution_mechanism codex_cli"
        )


def _require_codex_root(
    codex_work_root: Path | None,
    stage_root: Path,
    sealed: Path,
    protected_roots: Sequence[Path],
) -> Path | None:
    if codex_work_root is None:
        return None
    root = _require_stage_root(codex_work_root, sealed, protected_roots)
    if _overlaps(root, stage_root):
        raise ReviewPathError("codex working directory overlaps a provider stage")
    return root


def _codex_files(payloads: Mapping[str, Mapping[str, bytes]]) -> dict[str, bytes]:
    files = dict(payloads["gpt"])
    files.pop("manifest.json", None)
    return files


def _instruction_bytes(directory: Path) -> dict[str, bytes]:
    if directory.is_symlink() or not directory.is_dir():
        raise ReviewIntegrityError("incomplete original-image staging")
    files: dict[str, bytes] = {}
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ReviewIntegrityError("staged file escapes the provider directory")
        files[path.name] = path.read_bytes()
    return files


def _publish_codex_workdir(
    work_root: Path | None,
    files: Mapping[str, bytes],
    transcription: str,
    *,
    required: bool,
) -> CodexExecutionLayout | None:
    if not required:
        return None
    if work_root is None:
        raise ReviewPolicyError("codex_cli staging requires a codex working directory")
    _mkdir_private(work_root)
    if (work_root / _PARTIAL).exists():
        raise ReviewIntegrityError("interrupted original-image staging is present")
    cwd = work_root / _CODEX_CWD
    names = {path.name for path in work_root.iterdir()}
    if cwd.exists():
        if names != {_CODEX_CWD}:
            raise ReviewIntegrityError("incomplete original-image staging")
        _require_matching_files(cwd, files)
        _privatize_directory(cwd)
        _reject_transcription_leak(cwd, transcription)
        return _codex_layout(cwd, files)
    if names:
        raise ReviewIntegrityError("incomplete original-image staging")
    partial = work_root / _PARTIAL
    created = False
    try:
        os.mkdir(partial, _DIR_MODE)
        os.chmod(partial, _DIR_MODE)
        created = True
        for name, payload in files.items():
            _write_new(partial / name, payload)
        _reject_transcription_leak(partial, transcription)
        _privatize_directory(partial)
        os.rename(partial, cwd)
    except FileExistsError as exc:
        raise ReviewIntegrityError(
            "interrupted original-image staging is present"
        ) from exc
    except Exception:
        if created and not cwd.exists():
            shutil.rmtree(partial, ignore_errors=True)
        raise
    _privatize_directory(cwd)
    return _codex_layout(cwd, files)


def _require_matching_files(directory: Path, files: Mapping[str, bytes]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ReviewIntegrityError("incomplete original-image staging")
    present = {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if set(present) != set(files) or any(
        present[name] != payload for name, payload in files.items()
    ):
        raise ReviewIntegrityError(
            "refusing to change an existing original-image stage"
        )
    if any(path.is_symlink() for path in directory.iterdir()):
        raise ReviewIntegrityError("staged file escapes the provider directory")


def _codex_layout(cwd: Path, files: Mapping[str, bytes]) -> CodexExecutionLayout:
    image_name = next(name for name in files if name.startswith("original."))
    return CodexExecutionLayout(
        working_directory=cwd,
        image_path=cwd / image_name,
        prompt_path=cwd / "prompt.txt",
        work_request_path=cwd / "work_request.json",
        output_schema_path=cwd / "output_schema.json",
        output_example_path=cwd / "output_example.json",
    )


def _require_codex_layout(
    layout: CodexExecutionLayout | None, grok_directory: Path
) -> None:
    if layout is None:
        raise ReviewIntegrityError("incomplete original-image staging")
    cwd = layout.working_directory.resolve()
    grok = grok_directory.resolve()
    if (
        _overlaps(cwd, grok)
        or cwd.parent == grok.parent
        or grok in cwd.parents
        or cwd in grok.parents
    ):
        raise ReviewIntegrityError(
            "codex working directory contains the other provider stage"
        )
    if {path.name for path in cwd.parent.iterdir()} != {_CODEX_CWD}:
        raise ReviewIntegrityError(
            "codex working directory contains the other provider stage"
        )
    for path in cwd.rglob("*"):
        if path.is_symlink() or "grok" in path.relative_to(cwd).parts:
            raise ReviewIntegrityError(
                "codex working directory contains the other provider stage"
            )
        resolved = path.resolve()
        if resolved == grok or grok in resolved.parents:
            raise ReviewIntegrityError(
                "codex working directory contains the other provider stage"
            )


def _privatize_stages(root: Path, payloads: Mapping[str, Mapping[str, bytes]]) -> None:
    _mkdir_private(root)
    for provider in payloads:
        _privatize_directory(root / provider)


def _privatize_directory(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ReviewIntegrityError("incomplete original-image staging")
    _tighten(directory, _DIR_MODE)
    for path in directory.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ReviewIntegrityError("staged file escapes the provider directory")
        _tighten(path, _FILE_MODE)


def _tighten(path: Path, target: int) -> None:
    if path.is_symlink():
        raise ReviewIntegrityError("staged file escapes the provider directory")
    current = stat.S_IMODE(path.lstat().st_mode)
    if current == target:
        return
    if current & ~target:
        os.chmod(path, target)
        return
    raise ReviewIntegrityError("stage permissions are not private")


def _mkdir_private(path: Path) -> None:
    if path.is_symlink():
        raise ReviewIntegrityError("staged file escapes the provider directory")
    if not path.exists():
        os.mkdir(path, _DIR_MODE)
    elif not path.is_dir():
        raise ReviewPathError("original-image stage root is not a directory")
    os.chmod(path, _DIR_MODE)


def _transcription(sealed: Path) -> str:
    try:
        return (sealed / TRANSCRIPTION_RELATIVE_PATH).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewIntegrityError(
            "sealed transcription text is not valid UTF-8"
        ) from exc
