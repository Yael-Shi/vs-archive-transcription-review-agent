"""Minimum configuration for a read-only Pilot 01 consumer.

Values come from process environment variables passed in by the caller.
This module does not read ``.env`` files, the VS Archive checkout, or any
credential store.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from transcription_review_agent.constants import (
    DEFAULT_SCAN_LIMIT,
    MAX_SCAN_LIMIT,
    MIN_SCAN_LIMIT,
)
from transcription_review_agent.errors import ConfigError

API_BASE_URL_ENV = "TRANSCRIPTION_REVIEW_API_BASE_URL"
STATE_DIR_ENV = "TRANSCRIPTION_REVIEW_STATE_DIR"
BATCHES_DIR_ENV = "TRANSCRIPTION_REVIEW_BATCHES_DIR"
SCAN_PAGE_LIMIT_ENV = "TRANSCRIPTION_REVIEW_SCAN_PAGE_LIMIT"
REQUEST_TIMEOUT_ENV = "TRANSCRIPTION_REVIEW_REQUEST_TIMEOUT_SECONDS"

# The main Django checkout. This agent must not write inside it.
FORBIDDEN_WRITE_ROOTS = (Path("/home/yshif/vs-archive/vs-archive"),)


class AgentConfig(BaseModel):
    """Public API origin plus local paths. No credentials."""

    model_config = ConfigDict(extra="forbid")

    api_base_url: str
    scan_page_limit: int = Field(
        default=DEFAULT_SCAN_LIMIT,
        ge=MIN_SCAN_LIMIT,
        le=MAX_SCAN_LIMIT,
    )
    state_dir: Path
    batches_dir: Path
    request_timeout_seconds: float = Field(default=30.0, gt=0, le=120)

    @field_validator("api_base_url")
    @classmethod
    def _origin_only(cls, value: str) -> str:
        return normalize_api_base_url(value)

    @field_validator("state_dir", "batches_dir")
    @classmethod
    def _outside_vs_archive(cls, value: Path) -> Path:
        return assert_outside_vs_archive(value)


def normalize_api_base_url(value: str) -> str:
    """Accept an http(s) origin and reject credentials, queries, and paths."""

    raw = value.strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API base URL must be an http or https origin")
    if parsed.username or parsed.password:
        raise ValueError("API base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("API base URL must not contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("API base URL must be an origin without a path")
    return f"{parsed.scheme}://{parsed.netloc}"


def assert_outside_vs_archive(path: Path) -> Path:
    """Resolve ``path`` and refuse the VS Archive Django checkout."""

    resolved = path.expanduser().resolve()
    for root in FORBIDDEN_WRITE_ROOTS:
        forbidden = root.expanduser().resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError(
                f"{resolved} is inside the VS Archive repository; "
                "this agent must not write there"
            )
    return resolved


def load_config(
    environ: dict[str, str] | None = None,
    *,
    repo_root: Path | None = None,
) -> AgentConfig:
    """Load configuration from an explicit environment mapping.

    ``environ`` defaults to ``os.environ``. Dotenv files are never read.
    ``repo_root`` is the base for default ``state/`` and ``outputs/batches/``
    directories when the corresponding variables are unset. It defaults to
    the process working directory.
    """

    env = os.environ if environ is None else environ
    base_url = env.get(API_BASE_URL_ENV, "").strip()
    if not base_url:
        raise ConfigError(f"{API_BASE_URL_ENV} is required")

    root = Path.cwd() if repo_root is None else repo_root
    state_raw = env.get(STATE_DIR_ENV, "").strip()
    batches_raw = env.get(BATCHES_DIR_ENV, "").strip()
    state_dir = Path(state_raw) if state_raw else root / "state"
    batches_dir = Path(batches_raw) if batches_raw else root / "outputs" / "batches"

    try:
        scan_page_limit = _parse_int(
            env.get(SCAN_PAGE_LIMIT_ENV, str(DEFAULT_SCAN_LIMIT)),
            SCAN_PAGE_LIMIT_ENV,
        )
        timeout = _parse_float(
            env.get(REQUEST_TIMEOUT_ENV, "30"),
            REQUEST_TIMEOUT_ENV,
        )
        return AgentConfig(
            api_base_url=base_url,
            scan_page_limit=scan_page_limit,
            state_dir=state_dir,
            batches_dir=batches_dir,
            request_timeout_seconds=timeout,
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_float(raw: str, name: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
