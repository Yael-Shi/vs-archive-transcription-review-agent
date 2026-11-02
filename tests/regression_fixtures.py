"""Fixture loader for Track A slice A0 comparison regressions.

Two tiers (see ``tests/fixtures/regression/FIXTURE_POLICY.md``):

* committed synthetic fixtures, always available;
* an authorized archival pack kept outside Git and named by
  ``VS_ARCHIVAL_FIXTURE_DIR``.

Nothing here writes anywhere. Provider files are only read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcription_review_agent.review.compare import compare_transcriptions
from transcription_review_agent.review.models import ComparisonFile, FirstPassOutput

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "regression"
SYNTHETIC_DIR = FIXTURE_ROOT / "synthetic_pair"
POLICY_PATH = FIXTURE_ROOT / "fixture_policy.json"

PACK_DIR_ENV = "VS_ARCHIVAL_FIXTURE_DIR"
PACK_REQUIRED_ENV = "VS_ARCHIVAL_FIXTURES_REQUIRED"
PACK_MANIFEST_NAME = "archival_fixture_manifest.json"
PACK_SCHEMA = "a0-archival-pack-v1"

# Hashes are not an exact-reading test; they bind the fixture bytes only.
COMPARISON_SHA = "ab" * 32
DISPUTE_ID_RE = re.compile(r"dsp_[0-9a-f]{16}")


class ArchivalFixtureUnavailable(Exception):
    """An authorized archival run was not requested. Not a failure."""


class ArchivalFixtureError(Exception):
    """An authorized run was requested but the pack is missing or changed."""


def sha256_file_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expected_dispute_id(
    category: str, page_id: str, region_id: str | None, gpt_text: str, grok_text: str
) -> str:
    """The documented ID formula, restated independently of production code."""

    raw = f"{category}|{page_id}|{region_id or ''}|{gpt_text}|{grok_text}"
    return "dsp_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ProviderPair:
    gpt_bytes: bytes
    grok_bytes: bytes
    gpt: FirstPassOutput
    grok: FirstPassOutput

    def compare(self) -> ComparisonFile:
        return compare_transcriptions(
            self.gpt,
            self.grok,
            gpt_sha256=COMPARISON_SHA,
            grok_sha256=COMPARISON_SHA,
        )


def pair_from_bytes(gpt_bytes: bytes, grok_bytes: bytes) -> ProviderPair:
    return ProviderPair(
        gpt_bytes=gpt_bytes,
        grok_bytes=grok_bytes,
        gpt=FirstPassOutput.model_validate_json(gpt_bytes),
        grok=FirstPassOutput.model_validate_json(grok_bytes),
    )


def load_synthetic() -> tuple[ProviderPair, dict[str, Any]]:
    pair = pair_from_bytes(
        (SYNTHETIC_DIR / "gpt.json").read_bytes(),
        (SYNTHETIC_DIR / "grok.json").read_bytes(),
    )
    expected = json.loads(
        (SYNTHETIC_DIR / "expected_comparison.json").read_text(encoding="utf-8")
    )
    return pair, expected


def semantic_rows(comparison: ComparisonFile) -> dict[str, Any]:
    """The rows a regression binds: disputes and alignments, no hashes."""

    return {
        "disputes": [item.model_dump(mode="json") for item in comparison.disputes],
        "alignments": [item.model_dump(mode="json") for item in comparison.alignments],
        "page_originals": [
            item.model_dump(mode="json") for item in comparison.page_originals
        ],
        "unresolved_text_page_ids": list(comparison.unresolved_text_page_ids),
    }


# --------------------------------------------------------------------------
# Authorized archival pack
# --------------------------------------------------------------------------


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ArchivalDocument:
    document_id: str
    pair: ProviderPair
    expected: dict[str, Any]
    stale_dispute_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArchivalPack:
    root: Path
    authorized_by: str
    documents: dict[str, ArchivalDocument]


def archival_run_requested(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get(PACK_DIR_ENV)) or _truthy(env.get(PACK_REQUIRED_ENV))


def load_archival_pack(environ: dict[str, str] | None = None) -> ArchivalPack:
    """Load and hash-verify the authorized pack.

    Raises ``ArchivalFixtureUnavailable`` only when no authorized run was
    requested. Every problem after a request is an ``ArchivalFixtureError``.
    """

    env = os.environ if environ is None else environ
    if not archival_run_requested(env):
        raise ArchivalFixtureUnavailable(
            f"archival fixtures not authorized/unavailable: {PACK_DIR_ENV} is not set"
        )
    raw_dir = env.get(PACK_DIR_ENV)
    if not raw_dir:
        raise ArchivalFixtureError(
            f"{PACK_REQUIRED_ENV} is set but {PACK_DIR_ENV} is not"
        )
    root = Path(raw_dir).expanduser().resolve()
    if not root.is_dir():
        raise ArchivalFixtureError(f"archival fixture directory does not exist: {root}")
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise ArchivalFixtureError(
            "archival fixture directory must be outside the Git repository"
        )
    manifest_path = root / PACK_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArchivalFixtureError(
            f"archival fixture manifest is missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArchivalFixtureError(
            f"archival fixture manifest is unreadable: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != PACK_SCHEMA:
        raise ArchivalFixtureError(
            f"archival fixture manifest schema_version must be {PACK_SCHEMA}"
        )
    authorized_by = manifest.get("authorized_by")
    if not isinstance(authorized_by, str) or not authorized_by.strip():
        raise ArchivalFixtureError("archival fixture manifest must name authorized_by")
    stale = tuple(manifest.get("stale_dispute_ids") or ())
    raw_documents = manifest.get("documents")
    if not isinstance(raw_documents, dict) or not raw_documents:
        raise ArchivalFixtureError("archival fixture manifest lists no documents")

    documents: dict[str, ArchivalDocument] = {}
    for document_id, entry in raw_documents.items():
        try:
            documents[document_id] = ArchivalDocument(
                document_id=document_id,
                pair=pair_from_bytes(
                    _read_hashed(root, document_id, "gpt", entry["gpt"]),
                    _read_hashed(root, document_id, "grok", entry["grok"]),
                ),
                expected=entry["expected"],
                stale_dispute_ids=stale,
            )
        except ArchivalFixtureError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ArchivalFixtureError(
                f"document {document_id} entry is invalid: {exc}"
            ) from exc
    return ArchivalPack(root=root, authorized_by=authorized_by, documents=documents)


def _read_hashed(
    root: Path, document_id: str, side: str, entry: dict[str, str]
) -> bytes:
    path = (root / entry["path"]).resolve()
    if root != path and root not in path.parents:
        raise ArchivalFixtureError(
            f"document {document_id} {side} path escapes the pack: {entry['path']}"
        )
    if not path.is_file():
        raise ArchivalFixtureError(
            f"document {document_id} {side} fixture is missing: {path}"
        )
    data = path.read_bytes()
    if sha256_file_bytes(data) != entry["sha256"]:
        raise ArchivalFixtureError(
            f"document {document_id} {side} fixture changed: SHA-256 mismatch"
        )
    return data


def check_document_expectations(
    pair: ProviderPair,
    expected: dict[str, Any],
    stale_dispute_ids: tuple[str, ...] | list[str] = (),
) -> ComparisonFile:
    """Shared semantic checks for one document. Raises AssertionError."""

    gpt_before, grok_before = pair.gpt_bytes, pair.grok_bytes
    first = pair.compare()
    second = pair.compare()
    assert semantic_rows(first) == semantic_rows(second), (
        "comparison is not deterministic"
    )
    assert [d.dispute_id for d in first.disputes] == [
        d.dispute_id for d in second.disputes
    ]
    assert pair.gpt_bytes == gpt_before and pair.grok_bytes == grok_before

    gpt_pages = {page.page_id: page.diplomatic_transcription for page in pair.gpt.pages}
    grok_pages = {
        page.page_id: page.diplomatic_transcription for page in pair.grok.pages
    }
    for original in first.page_originals:
        assert original.gpt_diplomatic_transcription == gpt_pages.get(
            original.page_id, ""
        )
        assert original.grok_diplomatic_transcription == grok_pages.get(
            original.page_id, ""
        )

    ids = [item.dispute_id for item in first.disputes]
    assert len(ids) == len(set(ids)), "duplicate dispute IDs"
    for item in first.disputes:
        assert item.dispute_id == expected_dispute_id(
            item.category.value,
            item.page_id,
            item.region_id,
            item.gpt_text,
            item.grok_text,
        ), "dispute ID is not derived from its row text"

    if "disputes" in expected:
        assert len(first.disputes) == expected["disputes"]
    if "substantive" in expected:
        assert (
            sum(item.substantive for item in first.disputes) == expected["substantive"]
        )
    if "category_counts" in expected:
        counts = Counter(item.category.value for item in first.disputes)
        assert dict(counts) == expected["category_counts"]

    expected_ids = expected.get("dispute_ids")
    if expected_ids is not None:
        assert set(expected_ids) == set(ids), (
            "expected dispute IDs differ from regenerated IDs"
        )
        stale_only = set(stale_dispute_ids) - set(ids)
        assert not (stale_only & set(expected_ids)), (
            "stale dispute IDs were reused as expectations"
        )

    for row in expected.get("required_rows", []):
        matches = [
            item
            for item in first.disputes
            if item.category.value == row["category"]
            and row.get("gpt_contains", "") in item.gpt_text
            and row.get("grok_contains", "") in item.grok_text
            and (
                row.get("substantive") is None or item.substantive is row["substantive"]
            )
        ]
        assert matches, f"required row not found: {row['category']}"
    return first
