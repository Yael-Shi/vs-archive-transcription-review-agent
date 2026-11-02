"""Prepare first-pass worker tasks from one verified local image.

This is not discovery, bundling, sealing, or a review-controller run. It does
not write a batch manifest, an API review bundle, an archive transcription, or
a review ledger. ``source_bundle_sha256`` in the published work request is the
SHA-256 of the local image. The schema still requires a batch id and a
document id; those values are local labels, not archive identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.config import FORBIDDEN_WRITE_ROOTS
from transcription_review_agent.constants import SHA256_PATTERN
from transcription_review_agent.review.controller import (
    _parse_provider_file,
    _prepare_bundle,
    _work_request,
    _write_if_same,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
    ReviewResultMissing,
    ReviewUsageError,
)
from transcription_review_agent.review.models import (
    FirstPassOutput,
    SourceManifest,
    SourcePageRecord,
)
from transcription_review_agent.review.pngutil import decode_png
from transcription_review_agent.review.runtime import (
    DocumentReviewState,
    ExecutionState,
    PageAsset,
    ProviderRuntime,
    ProviderStatus,
    both_providers,
)

PILOT_ORIGIN: Final = "LOCAL_IMAGE_PILOT"
PILOT_BATCH_ID: Final = "batch-900002"
PILOT_DOCUMENT_ID: Final = 900002
PILOT_PAGE_ID: Final = "page_001"
PILOT_IMAGE_NAME = "page_001_original.png"
PILOT_02_SHA256 = "22febf496c60b61d6c8d8848a981f58be77935e42323dec1accc5a10e43b5e2e"
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_COLOR = {1: "gray", 3: "rgb", 4: "rgba"}
_NOTE = (
    "LOCAL_IMAGE_PILOT\n"
    "This task is not a sealed VS Archive bundle.\n"
    "No archive API payload and no archive transcription are included.\n"
    "source_bundle_sha256 is the SHA-256 of the local page image.\n"
    "batch-900002 and document 900002 are local schema labels, not archive ids.\n"
    "Write one JSON file at the output_relative_path in work_request.json.\n"
    "That file must validate as review-provider-output-v1 for stage first_pass.\n"
    "Do not submit this directory to review-next accept or finalize.\n"
)


class LocalImageProvenance(BaseModel):
    """Recorded origin of a local page. This is not a sealed bundle."""

    model_config = ConfigDict(extra="forbid")

    origin: Literal["LOCAL_IMAGE_PILOT"] = PILOT_ORIGIN
    sealed_archive_bundle: Literal[False] = False
    api_review_bundle_included: Literal[False] = False
    archive_transcription_included: Literal[False] = False
    local_source_path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    mime_type: Literal["image/png"] = "image/png"
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    bit_depth: Literal[8] = 8
    color_type: Literal["gray", "rgb", "rgba"]
    page_id: Literal["page_001"] = PILOT_PAGE_ID
    language: Literal["en", "fr"]
    schema_batch_id: Literal["batch-900002"] = PILOT_BATCH_ID
    schema_document_id: Literal[900002] = PILOT_DOCUMENT_ID
    source_bundle_sha256_is_image_sha256: Literal[True] = True


@dataclass(frozen=True, slots=True)
class LocalImagePilotPaths:
    destination: Path
    provenance: Path
    gpt_instructions: Path
    grok_instructions: Path
    shared_output: Path


def prepare_local_image_first_pass(
    image_path: Path,
    destination: Path,
    *,
    expected_sha256: str,
    language: Literal["en", "fr"],
) -> LocalImagePilotPaths:
    """Validate one PNG, then publish GPT Work and Grok first-pass tasks.

    The hash is checked before any destination file is created. A second call
    with the same image and destination leaves matching files in place.
    """

    source = Path(image_path).expanduser().resolve()
    target = _pilot_destination(destination)
    if not source.is_file():
        raise ReviewIntegrityError(f"local image is missing: {source}")
    data = source.read_bytes()
    digest = sha256_bytes(data)
    if digest != expected_sha256:
        raise ReviewIntegrityError(
            "local image SHA-256 does not match the expected digest"
        )
    try:
        width, height, _rows, bpp = decode_png(data)
    except ValueError as exc:
        raise ReviewIntegrityError(str(exc)) from exc
    color = _COLOR[bpp]
    provenance = LocalImageProvenance(
        local_source_path=str(source),
        sha256=digest,
        width=width,
        height=height,
        color_type=color,  # type: ignore[arg-type]
        language=language,
    )
    provenance_bytes = canonical_json(provenance).encode("utf-8")
    document = _memory_document(digest, width, height, language)
    source_manifest = SourceManifest(
        batch_id=PILOT_BATCH_ID,
        document_id=PILOT_DOCUMENT_ID,
        language=language,
        source_bundle_sha256=digest,
        pages=[
            SourcePageRecord(
                page_id=PILOT_PAGE_ID,
                reading_order=1,
                sha256=digest,
                width=width,
                height=height,
                filename=PILOT_IMAGE_NAME,
                mime_type="image/png",
            )
        ],
    )
    _write_if_same(target / "input" / PILOT_IMAGE_NAME, data)
    _write_if_same(
        target / "input" / "source_manifest.json",
        canonical_json(source_manifest).encode("utf-8"),
    )
    _write_if_same(target / "provenance.json", provenance_bytes)
    _write_if_same(
        target / "README.txt",
        _readme(source, digest, width, height, color).encode("utf-8"),
    )
    for provider in ("gpt", "grok"):
        request = _work_request(
            PILOT_BATCH_ID,
            document,
            provider=provider,
            stage="first_pass",
            round_index=0,
            attempt=1,
        )
        _prepare_bundle(
            target,
            document,
            request,
            "",
            comparison=None,
            gpt=None,
            grok=None,
            instruction_note=_NOTE,
            extra_files={"input/local_image_provenance.json": provenance_bytes},
        )
    _reject_archive_artifacts(target)
    return LocalImagePilotPaths(
        destination=target,
        provenance=target / "provenance.json",
        gpt_instructions=target
        / "first_pass"
        / "tasks"
        / "gpt"
        / "instructions_gpt.txt",
        grok_instructions=target
        / "first_pass"
        / "tasks"
        / "grok"
        / "instructions_grok.txt",
        shared_output=target / "first_pass" / "output",
    )


def check_local_image_first_pass_output(
    destination: Path,
    provider: str,
) -> FirstPassOutput:
    """Validate one dropped provider file against the published local task.

    This does not open a review ledger and does not call ``review-next``.
    """

    if provider not in {"gpt", "grok"}:
        raise ReviewUsageError("provider must be gpt or grok")
    target = _pilot_destination(destination)
    provenance = LocalImageProvenance.model_validate_json(
        (target / "provenance.json").read_bytes()
    )
    if provenance.origin != PILOT_ORIGIN or provenance.sealed_archive_bundle:
        raise ReviewIntegrityError(
            "local image provenance is not a LOCAL_IMAGE_PILOT record"
        )
    image = (target / "input" / PILOT_IMAGE_NAME).read_bytes()
    if sha256_bytes(image) != provenance.sha256:
        raise ReviewIntegrityError(
            "local image bytes changed after the pilot was prepared"
        )
    request_path = target / "first_pass" / "tasks" / provider / "work_request.json"
    request_bytes = request_path.read_bytes()
    from transcription_review_agent.review.models import WorkRequest

    request = WorkRequest.model_validate_json(request_bytes)
    if request.source_bundle_sha256 != provenance.sha256:
        raise ReviewIntegrityError("work request hash is not the local image hash")
    output = target / request.output_relative_path
    if not output.is_file():
        raise ReviewResultMissing(f"missing provider output {output.name}")
    document = _memory_document(
        provenance.sha256,
        provenance.width,
        provenance.height,
        provenance.language,
    )
    runtime = ProviderRuntime(
        provider=provider,  # type: ignore[arg-type]
        status=ProviderStatus.AWAITING_MANUAL_DISPATCH,
        attempt=request.attempt,
        task_id=request.task_id,
        job_id=request.job_id,
    )
    model = _parse_provider_file(
        output,
        batch_id=PILOT_BATCH_ID,
        provider=provider,
        stage="first_pass",
        document=document,
        runtime=runtime,
        round_index=0,
    )
    if not isinstance(model, FirstPassOutput):
        raise ReviewIntegrityError("local pilot output is not a first-pass file")
    return model


def _memory_document(
    digest: str,
    width: int,
    height: int,
    language: Literal["en", "fr"],
) -> DocumentReviewState:
    """In-memory page record. It is never written to a review ledger."""

    return DocumentReviewState(
        document_id=PILOT_DOCUMENT_ID,
        language=language,
        source_bundle_sha256=digest,
        archive_transcription_sha256=_EMPTY_SHA256,
        pages=[
            PageAsset(
                page_id=PILOT_PAGE_ID,
                reading_order=1,
                sha256=digest,
                width=width,
                height=height,
                mime_type="image/png",
                original_filename=PILOT_IMAGE_NAME,
            )
        ],
        first_pass=both_providers(),
        research=both_providers(),
        estimated_work_units=2,
        execution_state=ExecutionState.READY_FOR_PROVIDER,
    )


def _pilot_destination(destination: Path) -> Path:
    resolved = Path(destination).expanduser().resolve()
    for root in FORBIDDEN_WRITE_ROOTS:
        forbidden = root.expanduser().resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ReviewPathError(f"{resolved} is inside the VS Archive repository")
    repo = Path(__file__).resolve().parents[3]
    for relative in ("state", "outputs", "reviews"):
        forbidden = (repo / relative).resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ReviewPathError(f"{resolved} is inside agent controller state")
    return resolved


def _reject_archive_artifacts(root: Path) -> None:
    forbidden = {
        "transcription.txt",
        "review_bundle.json",
        "batch_manifest.json",
        "document_manifest.json",
        "ledger.jsonl",
        "state.json",
    }
    for path in root.rglob("*"):
        if path.is_file() and path.name in forbidden:
            raise ReviewIntegrityError(f"local pilot must not contain {path.name}")
        if path.is_dir() and path.name == "controller":
            raise ReviewIntegrityError(
                "local pilot must not contain a review controller"
            )


def _readme(source: Path, digest: str, width: int, height: int, color: str) -> str:
    return (
        "LOCAL_IMAGE_PILOT\n"
        "This directory is a local-image first-pass worker pilot.\n"
        "It is not a sealed VS Archive batch and it is not a review-next ledger.\n"
        "Do not pass it to review-next prepare, accept, or finalize.\n"
        f"Source: {source}\n"
        f"SHA-256: {digest}\n"
        f"Format: image/png 8-bit {color} {width}x{height}\n"
        "GPT Work instructions:\n"
        "  first_pass/tasks/gpt/instructions_gpt.txt\n"
        "Grok Bot instructions:\n"
        "  first_pass/tasks/grok/instructions_grok.txt\n"
        "Shared output directory:\n"
        "  first_pass/output/\n"
        "  transcription_gpt.json\n"
        "  transcription_grok.json\n"
        "After a provider writes its file, check it with:\n"
        "  python -m transcription_review_agent.review.local_image_pilot check \\\n"
        "    --destination <this directory> --provider gpt\n"
        "The same command with --provider grok checks the other file.\n"
        "A successful check validates review-provider-output-v1 only.\n"
        "It does not record a sealed review.\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Prepare or check a local-image pilot. No archive request is made."""

    parser = argparse.ArgumentParser(prog="local-image-pilot", add_help=False)
    parser.add_argument("operation", choices=("prepare", "check"))
    parser.add_argument("--image")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--language", choices=("en", "fr"))
    parser.add_argument("--provider", choices=("gpt", "grok"))
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise ReviewUsageError("local-image pilot usage was not recognized") from exc
    if args.operation == "prepare":
        if not args.image or not args.expected_sha256 or args.language is None:
            raise ReviewUsageError(
                "prepare requires --image, --expected-sha256, and --language"
            )
        prepare_local_image_first_pass(
            Path(args.image),
            Path(args.destination),
            expected_sha256=args.expected_sha256,
            language=args.language,
        )
        return 0
    if args.provider is None:
        raise ReviewUsageError("check requires --provider")
    check_local_image_first_pass_output(Path(args.destination), args.provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
