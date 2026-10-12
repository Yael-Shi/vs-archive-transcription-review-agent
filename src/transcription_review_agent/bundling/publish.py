"""Publish sealed manifest files.

Exclusive create is the immutability mechanism. A second publish of the same
manifest path fails even if the bytes would be identical.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from transcription_review_agent.errors import ImmutableBundleError


def canonical_json(model: BaseModel) -> str:
    """Stable pretty JSON with a trailing newline."""

    return model.model_dump_json(indent=2) + "\n"


def publish_exclusive(path: Path, payload: str) -> None:
    """Create ``path`` with ``O_EXCL``. Refuse an existing file."""

    if path.exists():
        raise ImmutableBundleError(
            f"refusing to overwrite completed bundle file {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload.encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise ImmutableBundleError(
            f"refusing to overwrite completed bundle file {path}"
        ) from exc
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def sealed_manifest_exists(path: Path) -> bool:
    """A manifest file on disk means that bundle is complete."""

    return path.is_file()
