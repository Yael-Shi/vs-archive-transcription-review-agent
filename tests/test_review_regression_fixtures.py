"""Slice A0: always-on, privacy-safe comparison regressions.

Uses only committed fictional fixtures. No database, provider, archive API,
or operational-output access.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from regression_fixtures import (
    COMPARISON_SHA,
    DISPUTE_ID_RE,
    FIXTURE_ROOT,
    PACK_DIR_ENV,
    PACK_MANIFEST_NAME,
    PACK_REQUIRED_ENV,
    PACK_SCHEMA,
    POLICY_PATH,
    REPO_ROOT,
    SYNTHETIC_DIR,
    ArchivalFixtureError,
    ArchivalFixtureUnavailable,
    archival_run_requested,
    check_document_expectations,
    expected_dispute_id,
    load_archival_pack,
    load_synthetic,
    pair_from_bytes,
    semantic_rows,
    sha256_file_bytes,
)

from transcription_review_agent.review.compare import challengeable_disputes
from transcription_review_agent.review.models import DisputeCategory

ALLOWED_COMMITTED_FILES = {
    "FIXTURE_POLICY.md",
    "fixture_policy.json",
    "synthetic_pair/gpt.json",
    "synthetic_pair/grok.json",
    "synthetic_pair/expected_comparison.json",
}


# ---- policy record ------------------------------------------------------


def test_policy_record_keeps_archival_content_out_of_git_by_default() -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["decision"]["archival_content_in_git"] is False
    assert policy["decision"]["approved_by"] is None
    assert policy["repository"]["visibility"] in {"PUBLIC", "PRIVATE", "UNVERIFIED"}
    assert policy["decision"]["applies_to_documents"] == [104, 105]
    assert policy["storage"]["committed_tier"]["always_run"] is True
    assert policy["storage"]["authorized_tier"]["always_run"] is False
    if policy["decision"]["archival_content_in_git"] is False:
        assert policy["decision"]["approved_by"] is None


def test_committed_regression_directory_holds_only_the_approved_files() -> None:
    found = {
        path.relative_to(FIXTURE_ROOT).as_posix()
        for path in FIXTURE_ROOT.rglob("*")
        if path.is_file()
    }
    assert found == ALLOWED_COMMITTED_FILES


def test_committed_fixtures_name_no_archival_documents_or_stale_ids() -> None:
    _, expected = load_synthetic()
    allowed_ids = {item["dispute_id"] for item in expected["disputes"]}
    for path in FIXTURE_ROOT.rglob("*.json"):
        if path == POLICY_PATH:
            continue
        text = path.read_text(encoding="utf-8")
        assert set(DISPUTE_ID_RE.findall(text)) <= allowed_ids, path.name
    for name in ("gpt.json", "grok.json"):
        payload = json.loads((SYNTHETIC_DIR / name).read_text(encoding="utf-8"))
        assert payload["document_id"] >= 900000, (
            "synthetic fixtures must not use archival document IDs"
        )
        assert payload["batch_id"] == "batch-900001"


# ---- synthetic comparison regressions ----------------------------------


def test_synthetic_provider_bytes_are_bound_by_recorded_hashes() -> None:
    pair, expected = load_synthetic()
    assert sha256_file_bytes(pair.gpt_bytes) == expected["fixture_sha256"]["gpt.json"]
    assert sha256_file_bytes(pair.grok_bytes) == expected["fixture_sha256"]["grok.json"]


def test_synthetic_exact_text_is_preserved_in_originals_and_disputes() -> None:
    pair, _ = load_synthetic()
    comparison = pair.compare()
    gpt_pages = {page.page_id: page.diplomatic_transcription for page in pair.gpt.pages}
    grok_pages = {
        page.page_id: page.diplomatic_transcription for page in pair.grok.pages
    }
    assert [item.page_id for item in comparison.page_originals] == list(gpt_pages)
    for original in comparison.page_originals:
        assert original.gpt_diplomatic_transcription == gpt_pages[original.page_id]
        assert original.grok_diplomatic_transcription == grok_pages[original.page_id]
    # Wrap-only page keeps both different line breaks verbatim and is not disputed.
    assert (
        gpt_pages["page_001"]
        == "Letter from the fictional Harbour\nBoard, dated in spring."
    )
    assert (
        grok_pages["page_001"]
        == "Letter from the fictional\nHarbour Board, dated in spring."
    )
    assert [item for item in comparison.disputes if item.page_id == "page_001"] == []
    by_text = {(item.gpt_text, item.grok_text): item for item in comparison.disputes}
    assert (
        by_text[("quay is Corrupt", "quay is corrupt.")].category
        is DisputeCategory.PUNCTUATION
    )
    assert (
        by_text[("the house is damp", "the home is damp")].category
        is DisputeCategory.WORDING
    )
    assert (
        by_text[("Captain Ionnes Marlowe", "Captain Iannes Marlowe")].category
        is DisputeCategory.NAME_OR_ENTITY
    )
    assert (
        by_text[("dated 12 March 1812.", "dated 13 March 1812.")].category
        is DisputeCategory.DATE_OR_NUMBER
    )
    assert by_text[("and", "")].category is DisputeCategory.CANCELLATION_OR_INSERTION
    assert (
        by_text[("A closing line only one reader saw.", "")].category
        is DisputeCategory.MISSING_CONTENT
    )


def test_synthetic_rows_alignment_categories_and_ids_match_the_golden_file() -> None:
    pair, expected = load_synthetic()
    comparison = pair.compare()
    rows = semantic_rows(comparison)
    assert rows["disputes"] == expected["disputes"]
    assert rows["alignments"] == expected["alignments"]
    assert rows["unresolved_text_page_ids"] == expected["unresolved_text_page_ids"]
    assert {
        "disputes": len(comparison.disputes),
        "substantive": sum(item.substantive for item in comparison.disputes),
        "alignments": len(comparison.alignments),
    } == expected["counts"]


def test_synthetic_fixture_covers_each_category_alignment_kind_and_substantive_rule() -> (
    None
):
    pair, _ = load_synthetic()
    comparison = pair.compare()
    assert {item.category for item in comparison.disputes} == {
        DisputeCategory.MISSING_CONTENT,
        DisputeCategory.READING_ORDER,
        DisputeCategory.NAME_OR_ENTITY,
        DisputeCategory.DATE_OR_NUMBER,
        DisputeCategory.WORDING,
        DisputeCategory.CANCELLATION_OR_INSERTION,
        DisputeCategory.COVERAGE_UNCERTAINTY,
        DisputeCategory.PUNCTUATION,
    }
    assert {item.kind for item in comparison.alignments} == {
        "matched",
        "reordered",
        "missing",
    }
    challenged = challengeable_disputes(comparison)
    assert {item.category for item in challenged}.isdisjoint(
        {DisputeCategory.PUNCTUATION, DisputeCategory.COVERAGE_UNCERTAINTY}
    )
    assert len(challenged) == 6
    assert comparison.selected_by_confidence is False
    assert "winner" not in comparison.model_dump()


def test_synthetic_ids_are_stable_for_unchanged_input_and_follow_the_formula() -> None:
    pair, _ = load_synthetic()
    first = pair.compare()
    again = pair_from_bytes(pair.gpt_bytes, pair.grok_bytes).compare()
    assert [item.dispute_id for item in first.disputes] == [
        item.dispute_id for item in again.disputes
    ]
    assert semantic_rows(first) == semantic_rows(again)
    assert pair.gpt_bytes == (SYNTHETIC_DIR / "gpt.json").read_bytes()
    assert pair.grok_bytes == (SYNTHETIC_DIR / "grok.json").read_bytes()
    for item in first.disputes:
        assert item.dispute_id == expected_dispute_id(
            item.category.value,
            item.page_id,
            item.region_id,
            item.gpt_text,
            item.grok_text,
        )
    assert len({item.dispute_id for item in first.disputes}) == len(first.disputes)


def test_changed_text_changes_only_that_dispute_id_so_old_ids_cannot_be_reused() -> (
    None
):
    pair, _ = load_synthetic()
    before = {
        (item.page_id, item.region_id): item.dispute_id
        for item in pair.compare().disputes
    }
    edited = pair.grok_bytes.replace(b"the home is damp", b"the hovel is damp")
    assert edited != pair.grok_bytes
    after_comparison = pair_from_bytes(pair.gpt_bytes, edited).compare()
    after = {
        (item.page_id, item.region_id): item.dispute_id
        for item in after_comparison.disputes
    }
    changed = {key for key in before if before[key] != after.get(key)}
    assert changed == {("page_002", "r3")}
    assert before[("page_002", "r3")] not in after.values()


def test_synthetic_run_reads_without_writing(tmp_path: Path) -> None:
    before = {path: path.read_bytes() for path in SYNTHETIC_DIR.iterdir()}
    pair, _ = load_synthetic()
    pair.compare()
    assert {path: path.read_bytes() for path in SYNTHETIC_DIR.iterdir()} == before
    assert set(before) == {path for path in SYNTHETIC_DIR.iterdir()}


# ---- loader: unavailable versus failure --------------------------------


def _write_pack(
    root: Path, *, expected_override: dict | None = None, stale: list[str] | None = None
) -> Path:
    """Build a throwaway pack from the fictional fixtures (stand-in for 104/105)."""

    pair, expected = load_synthetic()
    (root / "900001").mkdir(parents=True)
    (root / "900001" / "gpt.json").write_bytes(pair.gpt_bytes)
    (root / "900001" / "grok.json").write_bytes(pair.grok_bytes)
    doc_expected = {
        "disputes": expected["counts"]["disputes"],
        "substantive": expected["counts"]["substantive"],
        "category_counts": {
            category: sum(
                1 for item in expected["disputes"] if item["category"] == category
            )
            for category in {item["category"] for item in expected["disputes"]}
        },
        "dispute_ids": [item["dispute_id"] for item in expected["disputes"]],
        "required_rows": [
            {
                "category": "DATE_OR_NUMBER",
                "gpt_contains": "12 March",
                "grok_contains": "13 March",
                "substantive": True,
            }
        ],
    }
    doc_expected.update(expected_override or {})
    manifest = {
        "schema_version": PACK_SCHEMA,
        "authorized_by": "test",
        "stale_dispute_ids": stale or [],
        "documents": {
            "900001": {
                "gpt": {
                    "path": "900001/gpt.json",
                    "sha256": sha256_file_bytes(pair.gpt_bytes),
                },
                "grok": {
                    "path": "900001/grok.json",
                    "sha256": sha256_file_bytes(pair.grok_bytes),
                },
                "expected": doc_expected,
            }
        },
    }
    (root / PACK_MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_unrequested_run_is_unavailable_not_a_failure() -> None:
    assert archival_run_requested({}) is False
    with pytest.raises(ArchivalFixtureUnavailable, match="not authorized/unavailable"):
        load_archival_pack({})
    with pytest.raises(ArchivalFixtureUnavailable):
        load_archival_pack({PACK_REQUIRED_ENV: "0"})


def test_requested_run_fails_when_pack_is_missing_or_incomplete(tmp_path: Path) -> None:
    with pytest.raises(ArchivalFixtureError, match="does not exist"):
        load_archival_pack({PACK_DIR_ENV: str(tmp_path / "absent")})
    with pytest.raises(ArchivalFixtureError, match="is not"):
        load_archival_pack({PACK_REQUIRED_ENV: "1"})
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ArchivalFixtureError, match="manifest is missing"):
        load_archival_pack({PACK_DIR_ENV: str(empty)})
    pack = _write_pack(tmp_path / "pack")
    (pack / "900001" / "grok.json").unlink()
    with pytest.raises(ArchivalFixtureError, match="fixture is missing"):
        load_archival_pack({PACK_DIR_ENV: str(pack)})


def test_requested_run_fails_when_a_fixture_changed(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    target = pack / "900001" / "gpt.json"
    target.write_bytes(target.read_bytes().replace(b"Harbour", b"Harbor"))
    with pytest.raises(ArchivalFixtureError, match="SHA-256 mismatch"):
        load_archival_pack({PACK_DIR_ENV: str(pack)})


def test_requested_run_refuses_a_pack_inside_the_repository() -> None:
    with pytest.raises(ArchivalFixtureError, match="outside the Git repository"):
        load_archival_pack({PACK_DIR_ENV: str(REPO_ROOT / "tests")})


def test_pack_path_cannot_escape_its_directory(tmp_path: Path) -> None:
    pack = _write_pack(tmp_path / "pack")
    manifest_path = pack / PACK_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"]["900001"]["gpt"]["path"] = "../outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArchivalFixtureError, match="escapes the pack"):
        load_archival_pack({PACK_DIR_ENV: str(pack)})


def test_valid_pack_passes_shared_checks_and_leaves_inputs_unchanged(
    tmp_path: Path,
) -> None:
    pack_dir = _write_pack(tmp_path / "pack")
    before = {path: path.read_bytes() for path in pack_dir.rglob("*") if path.is_file()}
    pack = load_archival_pack({PACK_DIR_ENV: str(pack_dir)})
    document = pack.documents["900001"]
    comparison = check_document_expectations(
        document.pair, document.expected, document.stale_dispute_ids
    )
    assert len(comparison.disputes) == 8
    assert {
        path: path.read_bytes() for path in pack_dir.rglob("*") if path.is_file()
    } == before


def test_shared_checks_reject_wrong_counts_and_stale_expectations(
    tmp_path: Path,
) -> None:
    pack = load_archival_pack(
        {
            PACK_DIR_ENV: str(
                _write_pack(tmp_path / "a", expected_override={"disputes": 9})
            )
        }
    )
    document = pack.documents["900001"]
    with pytest.raises(AssertionError):
        check_document_expectations(document.pair, document.expected)

    stale = "dsp_" + "0" * 16
    with_stale = _write_pack(
        tmp_path / "b",
        expected_override={"dispute_ids": [stale]},
        stale=[stale],
    )
    document = load_archival_pack({PACK_DIR_ENV: str(with_stale)}).documents["900001"]
    with pytest.raises(AssertionError):
        check_document_expectations(
            document.pair, document.expected, document.stale_dispute_ids
        )

    missing_row = _write_pack(
        tmp_path / "c",
        expected_override={
            "required_rows": [{"category": "WORDING", "gpt_contains": "absent"}]
        },
    )
    document = load_archival_pack({PACK_DIR_ENV: str(missing_row)}).documents["900001"]
    with pytest.raises(AssertionError, match="required row not found"):
        check_document_expectations(document.pair, document.expected)


def test_comparison_sha_constant_is_not_an_archival_hash() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", COMPARISON_SHA)
