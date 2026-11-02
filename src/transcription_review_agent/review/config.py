"""Limits for one local review run. No credentials and no API fallback."""

from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field

from transcription_review_agent.errors import ConfigError


MAX_PAGES_ENV = "TRANSCRIPTION_REVIEW_MAX_PAGES"
MAX_DOCUMENTS_ENV = "TRANSCRIPTION_REVIEW_MAX_DOCUMENTS"
MAX_CHALLENGE_ROUNDS_ENV = "TRANSCRIPTION_REVIEW_MAX_CHALLENGE_ROUNDS"


class ReviewConfig(BaseModel):
    """Page, document, and challenge caps. The default challenge cap is one."""

    model_config = ConfigDict(extra="forbid")

    max_pages: int = Field(default=50, ge=1, le=500)
    max_documents: int = Field(default=5, ge=1, le=200)
    max_challenge_rounds: int = Field(default=1, ge=1, le=2)


def load_review_config(environ: dict[str, str] | None = None) -> ReviewConfig:
    env = os.environ if environ is None else environ
    try:
        return ReviewConfig(
            max_pages=_parse_int(env.get(MAX_PAGES_ENV, "50"), MAX_PAGES_ENV),
            max_documents=_parse_int(
                env.get(MAX_DOCUMENTS_ENV, "5"),
                MAX_DOCUMENTS_ENV,
            ),
            max_challenge_rounds=_parse_int(
                env.get(MAX_CHALLENGE_ROUNDS_ENV, "1"),
                MAX_CHALLENGE_ROUNDS_ENV,
            ),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
