"""The real Grok Bot first-pass file is invalid for a known set of causes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.controller import _instructions, _schema_text
from transcription_review_agent.review.first_pass_contract import (
    first_pass_instruction_note,
    first_pass_schema_text,
    write_schema_repair_request,
)
from transcription_review_agent.review.models import FirstPassOutput

FIXTURE = Path("tests/fixtures/provider_contracts/grok_bot_pilot02_first_pass.json")
ORIGINAL_SHA256 = "a59ae837ecc87e332bc4b52a7664588a1278325fdc9738777f4bbc7b72f710be"


def _groups(exc: ValidationError) -> Counter[str]:
    groups: Counter[str] = Counter()
    page_extras = {
        "pages.N.sha256",
        "pages.N.width",
        "pages.N.height",
        "pages.N.description",
        "pages.N.lines",
        "pages.N.text_diplomatic",
    }
    page_missing = {
        "pages.N.diplomatic_transcription",
        "pages.N.coverage",
        "pages.N.inaccessible_or_ambiguous",
    }
    for error in exc.errors():
        loc = ".".join(
            "N" if isinstance(part, int) else str(part) for part in error["loc"]
        )
        if loc.startswith("pages.N.alternative_readings"):
            groups["alternative_readings"] += 1
        elif loc.startswith("pages.N.regions"):
            groups["regions"] += 1
        elif loc.startswith("pages.N.attention_items"):
            groups["attention_items"] += 1
        elif loc.startswith("pages.N.cancellations"):
            groups["cancellations"] += 1
        elif loc.startswith("pages.N.unreadable_spans"):
            groups["unreadable_spans"] += 1
        elif loc in page_missing:
            groups["page_required_missing"] += 1
        elif loc in page_extras:
            groups["page_extras"] += 1
        elif not loc.startswith("pages."):
            groups["top_level_extras"] += 1
        else:
            groups[f"other:{loc}"] += 1
    return groups


def test_real_grok_output_fails_in_eight_groups() -> None:
    data = FIXTURE.read_bytes()
    assert sha256_bytes(data) == ORIGINAL_SHA256
    digest = sha256_bytes(data)
    payload = json.loads(data)
    with pytest.raises(ValidationError) as caught:
        FirstPassOutput.model_validate(payload)
    assert caught.value.error_count() == 214
    assert dict(_groups(caught.value)) == {
        "alternative_readings": 96,
        "regions": 39,
        "attention_items": 25,
        "cancellations": 17,
        "unreadable_spans": 15,
        "page_required_missing": 3,
        "page_extras": 6,
        "top_level_extras": 13,
    }
    assert digest == sha256_bytes(FIXTURE.read_bytes())


def test_grok_repair_request_preserves_evidence_without_rewriting(
    tmp_path: Path,
) -> None:
    before = FIXTURE.read_bytes()
    digest = sha256_bytes(before)
    request_path = write_schema_repair_request(FIXTURE, tmp_path / "repair")
    assert FIXTURE.read_bytes() == before
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["do_not_retranscribe"] is True
    assert request["original_sha256"] == digest
    assert request["write_repaired_output_to"].endswith(
        "transcription_grok.schema_repair.json"
    )
    extras = request["unmapped_values"]["region_extras"]
    assert extras[0]["type"] == "place_date_line"
    assert request["unmapped_values"]["page_extras"]["text_diplomatic"].startswith(
        "Seguein"
    )
    assert request["unmapped_values"]["top_level"]["coverage_report"][
        "regions_examined"
    ] == [
        "page_001.r_heading",
        "page_001.r_salutation",
        "page_001.r_body",
        "page_001.r_annotation_circle",
        "page_001.r_showthrough",
        "page_001.r_margins_edges",
    ]
    notes = request["lossless_renames_you_may_apply"]
    assert any("text_diplomatic" in note for note in notes)
    assert any("regions_examined" in note for note in notes)
    assert not (tmp_path / "repair" / "transcription_grok.schema_repair.json").exists()
    instructions = (
        tmp_path / "repair" / "grok_schema_repair_instructions.txt"
    ).read_text(encoding="utf-8")
    assert "Do not transcribe the page again." in instructions


def test_future_gpt_and_grok_packages_share_the_explicit_schema() -> None:
    schema = _schema_text("first_pass")
    assert schema == first_pass_schema_text()
    assert "text_diplomatic" in schema
    assert "examined_region_ids" in schema
    instructions = _instructions("first_pass")
    assert first_pass_instruction_note() in instructions
    assert "diplomatic_transcription" in instructions
