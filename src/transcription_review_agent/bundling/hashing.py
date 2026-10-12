"""Local SHA-256 helpers.

Hashes are of exact UTF-8 text and exact downloaded bytes. Server-supplied
transcription hashes are informational; these digests are authoritative for
the local bundle.
"""

from __future__ import annotations

import hashlib


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``data``."""

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hash the exact UTF-8 encoding of ``text`` without adding a newline."""

    return sha256_bytes(text.encode("utf-8"))
