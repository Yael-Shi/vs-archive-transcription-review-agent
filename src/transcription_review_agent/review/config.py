"""Limits for one local review run. No credentials and no API fallback."""

from __future__ import annotations

import os
from pydantic import BaseModel, ConfigDict, Field

from transcription_review_agent.errors import ConfigError


MAX_PAGES_ENV = "TRANSCRIPTION_REVIEW_MAX_PAGES"
MAX_DOCUMENTS_ENV = "TRANSCRIPTION_REVIEW_MAX_DOCUMENTS"
MAX_CHALLENGE_ROUNDS_ENV = "TRANSCRIPTION_REVIEW_MAX_CHALLENGE_ROUNDS"
MAX_CHALLENGE_DISPUTES_ENV = "TRANSCRIPTION_REVIEW_MAX_CHALLENGE_DISPUTES"
MAX_DOCUMENT_CHALLENGE_DISPATCHES_ENV = (
    "TRANSCRIPTION_REVIEW_MAX_DOCUMENT_CHALLENGE_DISPATCHES"
)
MAX_BATCH_CHALLENGE_DISPATCHES_ENV = (
    "TRANSCRIPTION_REVIEW_MAX_BATCH_CHALLENGE_DISPATCHES"
)


class ReviewConfig(BaseModel):
    """Page, document, and challenge caps.

    The default challenge round cap is one. Dispatch budgets count provider
    tasks, not tokens or account usage. The defaults cover one round for both
    providers: 2 tasks on a document and 10 tasks in a five-document batch.
    Raising the round cap does not raise those task budgets.
    """

    model_config = ConfigDict(extra="forbid")

    max_pages: int = Field(default=50, ge=1, le=500)
    max_documents: int = Field(default=5, ge=1, le=200)
    max_challenge_rounds: int = Field(default=1, ge=1, le=2)
    max_challenge_disputes: int = Field(default=15, ge=1, le=200)
    max_document_challenge_dispatches: int = Field(default=2, ge=1, le=400)
    max_batch_challenge_dispatches: int = Field(default=10, ge=1, le=800)


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
            max_challenge_disputes=_parse_int(
                env.get(MAX_CHALLENGE_DISPUTES_ENV, "15"),
                MAX_CHALLENGE_DISPUTES_ENV,
            ),
            max_document_challenge_dispatches=_parse_int(
                env.get(MAX_DOCUMENT_CHALLENGE_DISPATCHES_ENV, "2"),
                MAX_DOCUMENT_CHALLENGE_DISPATCHES_ENV,
            ),
            max_batch_challenge_dispatches=_parse_int(
                env.get(MAX_BATCH_CHALLENGE_DISPATCHES_ENV, "10"),
                MAX_BATCH_CHALLENGE_DISPATCHES_ENV,
            ),
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
