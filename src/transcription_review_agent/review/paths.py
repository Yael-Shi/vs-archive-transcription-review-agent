"""Review workspace paths and allowlists.

Archive bundles stay under ``batches_dir``. Review files stay under
``reviews_dir``. Provider output names are the only files imported from a
shared output directory.
"""

from __future__ import annotations

import re
from pathlib import Path

from transcription_review_agent.bundling.layout import assert_safe_relative_path
from transcription_review_agent.config import assert_outside_vs_archive
from transcription_review_agent.constants import BATCH_ID_PATTERN
from transcription_review_agent.review.errors import ReviewPathError

_BATCH_ID = re.compile(BATCH_ID_PATTERN)

OUTPUT_ALLOWLIST: dict[str, dict[str, str]] = {
    "first_pass": {
        "gpt": "transcription_gpt.json",
        "grok": "transcription_grok.json",
    },
    "challenge": {
        "gpt": "challenge_gpt.json",
        "grok": "challenge_grok.json",
    },
    "research": {
        "gpt": "research_gpt.json",
        "grok": "research_grok.json",
    },
}


def require_batch_id(batch_id: str) -> str:
    if _BATCH_ID.fullmatch(batch_id) is None:
        raise ReviewPathError("batch_id is not a sealed batch id")
    return batch_id


def require_provider(provider: str) -> str:
    if provider not in {"gpt", "grok"}:
        raise ReviewPathError("provider must be gpt or grok")
    return provider


def reviews_root(path: Path) -> Path:
    """Resolve the review workspace and refuse the VS Archive checkout."""

    try:
        return assert_outside_vs_archive(path)
    except ValueError as exc:
        raise ReviewPathError(str(exc)) from exc


def contained(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to stay inside ``root``."""

    root_resolved = root.resolve()
    resolved = candidate.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ReviewPathError(f"{candidate} is outside {root}")
    return resolved


def batch_root(reviews_dir: Path, batch_id: str) -> Path:
    return reviews_dir / f"batch_{require_batch_id(batch_id)}"


def document_root(reviews_dir: Path, batch_id: str, document_id: int) -> Path:
    if document_id < 1:
        raise ReviewPathError("document_id must be a positive integer")
    return batch_root(reviews_dir, batch_id) / "documents" / f"document_{document_id}"


def controller_dir(reviews_dir: Path, batch_id: str) -> Path:
    return batch_root(reviews_dir, batch_id) / "controller"


def archive_document_dir(batches_dir: Path, batch_id: str, document_id: int) -> Path:
    if document_id < 1:
        raise ReviewPathError("document_id must be a positive integer")
    return batches_dir / require_batch_id(batch_id) / "documents" / str(document_id)


def output_filename(stage: str, provider: str) -> str:
    try:
        return OUTPUT_ALLOWLIST[stage][require_provider(provider)]
    except KeyError as exc:
        raise ReviewPathError(f"no allowlisted output for {stage}") from exc


def stage_dir_name(stage: str, round_index: int) -> str:
    if stage == "first_pass":
        return "first_pass"
    if stage == "research":
        return "research"
    if stage == "challenge":
        if round_index < 1:
            raise ReviewPathError("challenge rounds start at 1")
        return f"challenge_round_{round_index:02d}"
    raise ReviewPathError(f"unknown stage {stage}")


def shared_output_dir(doc_root: Path, stage: str, round_index: int) -> Path:
    return doc_root / stage_dir_name(stage, round_index) / "output"


def output_relative_path(
    stage: str, provider: str, round_index: int, attempt: int = 1
) -> str:
    """Provider drop path. Attempt 1 keeps the historical shared output name.

    A recovered challenge attempt uses its own directory so the rejected file
    stays where the provider wrote it.
    """

    name = output_filename(stage, provider)
    directory = stage_dir_name(stage, round_index)
    if attempt == 1:
        relative = f"{directory}/output/{name}"
    elif attempt > 1:
        relative = f"{directory}/attempts/a{attempt:02d}/output/{name}"
    else:
        raise ReviewPathError("attempt must be positive")
    return assert_safe_relative_path(relative)


def assert_allowlisted_output_dir(directory: Path) -> None:
    """Reject any file whose name is not an allowlisted provider output."""

    if not directory.exists():
        return
    allowed = {
        name
        for stage_names in OUTPUT_ALLOWLIST.values()
        for name in stage_names.values()
    }
    for path in directory.iterdir():
        if not path.is_file():
            raise ReviewPathError(f"unexpected entry in provider output: {path.name}")
        if path.name not in allowed:
            raise ReviewPathError(f"provider output {path.name} is not allowlisted")
        if not (path.name.endswith("_gpt.json") or path.name.endswith("_grok.json")):
            raise ReviewPathError(
                f"provider output {path.name} must end in _gpt or _grok"
            )
