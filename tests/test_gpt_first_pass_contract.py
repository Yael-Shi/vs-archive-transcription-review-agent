"""The real GPT Work first-pass file is invalid for a known set of causes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.first_pass_contract import (
    FIRST_PASS_OUTPUT_SCHEMA,
    write_schema_repair_request,
)
from transcription_review_agent.review.models import FirstPassOutput

FIXTURE = Path("tests/fixtures/provider_contracts/gpt_work_pilot02_first_pass.json")
ORIGINAL_SHA256 = "b71599aece333382d145b40c6c23dca7f33e4510161e0a2912efc61da97255fd"


def _groups(exc: ValidationError) -> Counter[str]:
    groups: Counter[str] = Counter()
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
        elif loc == "pages.N.coverage":
            groups["page_coverage"] += 1
        elif loc.startswith("pages.N.cancellations"):
            groups["cancellations"] += 1
        elif loc.startswith("pages.N.marginal_notes"):
            groups["marginal_notes"] += 1
        elif loc == "pages.N.inaccessible_or_ambiguous":
            groups["page_inaccessible_or_ambiguous"] += 1
        elif loc in {"round_index", "language", "attention_items", "coverage_report"}:
            groups["top_level_extras"] += 1
        else:
            groups[f"other:{loc}"] += 1
    return groups


def test_real_gpt_output_fails_in_eight_groups() -> None:
    data = FIXTURE.read_bytes()
    assert sha256_bytes(data) == ORIGINAL_SHA256
    payload = json.loads(data)
    with pytest.raises(ValidationError) as caught:
        FirstPassOutput.model_validate(payload)
    assert caught.value.error_count() == 99
    assert dict(_groups(caught.value)) == {
        "alternative_readings": 54,
        "regions": 9,
        "attention_items": 21,
        "page_coverage": 1,
        "cancellations": 5,
        "marginal_notes": 4,
        "page_inaccessible_or_ambiguous": 1,
        "top_level_extras": 4,
    }


def test_schema_names_the_fields_the_stub_omitted() -> None:
    assert FIRST_PASS_OUTPUT_SCHEMA["schema_version"] == "review-provider-output-v1"
    assert FIRST_PASS_OUTPUT_SCHEMA["stage"] == "first_pass"
    page = FIRST_PASS_OUTPUT_SCHEMA["page_fields"]
    assert "reading_order" not in page["regions"][0]
    assert "order" in page["regions"][0]
    assert page["alternative_readings"][0]["readings"].startswith("array")
    assert "coverage" in page
    assert "region.reading_order" in FIRST_PASS_OUTPUT_SCHEMA["rejected_examples"]


def test_repair_request_does_not_rewrite_the_original(tmp_path: Path) -> None:
    before = FIXTURE.read_bytes()
    request_path = write_schema_repair_request(FIXTURE, tmp_path / "repair")
    assert FIXTURE.read_bytes() == before
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["do_not_retranscribe"] is True
    assert request["do_not_modify_original"] is True
    assert request["original_sha256"] == ORIGINAL_SHA256
    assert request["unmapped_values"]["region_types"] == [
        {"region_id": "page_001-r01", "region_type": "dateline"},
        {"region_id": "page_001-r02", "region_type": "salutation_and_body"},
        {"region_id": "page_001-r03", "region_type": "numbered_list"},
    ]
    assert "c'est le cas" in json.dumps(request["unmapped_values"]["cancellations"])
    assert not (tmp_path / "repair" / "transcription_gpt.schema_repair.json").exists()
    instructions = (
        tmp_path / "repair" / "gpt_schema_repair_instructions.txt"
    ).read_text(encoding="utf-8")
    assert "Do not transcribe the page again." in instructions
