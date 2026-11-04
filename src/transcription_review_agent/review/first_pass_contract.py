"""Explicit first-pass output contract.

The provider file remains ``review-provider-output-v1``. This module states
the fields that model accepts. It does not convert provider JSON and it does
not add fields to the model.
"""

from __future__ import annotations

import json
from pathlib import Path

from transcription_review_agent.bundling.hashing import sha256_bytes

FIRST_PASS_OUTPUT_SCHEMA: dict[str, object] = {
    "schema_version": "review-provider-output-v1",
    "stage": "first_pass",
    "note": (
        "Return one JSON object and no other fields. "
        "Extra fields are rejected. Do not include commands."
    ),
    "top_level_fields": {
        "schema_version": "review-provider-output-v1",
        "provider": "gpt or grok",
        "provider_interface": "GPT_WORK when provider is gpt; GROK when provider is grok",
        "task_id": "copy from work_request.json",
        "job_id": "copy from work_request.json",
        "batch_id": "copy from work_request.json",
        "document_id": "copy from work_request.json",
        "attempt": "copy from work_request.json",
        "source_bundle_sha256": "copy from work_request.json",
        "stage": "first_pass",
        "pages": "one object per page_id in work_request.json, in that order",
    },
    "page_fields": {
        "page_id": "page_001 style id from work_request.json",
        "reading_order": "integer starting at 1. This name is only for the page.",
        "diplomatic_transcription": "string",
        "regions": [
            {
                "region_id": "string",
                "order": "integer starting at 1. The field name is order, not reading_order.",
                "lines": [{"line_id": "string", "text": "string"}],
                "bbox": "optional {x, y, width, height} in page pixels, or omit it",
            }
        ],
        "unreadable_spans": [{"region_id": "string", "text_marker": "string"}],
        "alternative_readings": [
            {
                "region_id": "string",
                "readings": "array of at least two strings already read in that region",
            }
        ],
        "attention_items": [
            {
                "kind": "name, date, number, place, organization, event, or term",
                "text": "string already present in the reading",
                "region_id": "string",
            }
        ],
        "coverage": {
            "coverage_complete": "boolean. True only when every visible region was examined.",
            "unresolved_text_present": "boolean. May be true when coverage_complete is true.",
            "examined_region_ids": "array of region_id values",
            "inaccessible_or_ambiguous": "optional array of strings",
        },
        "cancellations": [{"region_id": "string", "text": "string"}],
        "insertions": [{"region_id": "string", "text": "string"}],
        "marginal_notes": [{"region_id": "string", "text": "string"}],
        "inaccessible_or_ambiguous": "array of strings",
    },
    "rejected_examples": [
        "region.reading_order",
        "region.region_type",
        "alternative_readings line_id, reading, alternatives, or reason",
        "attention_items attention_id, line_ids, category, or description",
        "cancellation_id, line_id, method, or certainty",
        "marginal_note_id, location, or description",
        "region.type, region.bbox_px, or region.line_ids",
        "page text_diplomatic, or a lines array beside regions",
        "unreadable span, reason, alternatives, or char_range_hint",
        "top-level round_index, language, origin, inputs_used, transcription_conventions, attention_items, or coverage_report",
    ],
}


def first_pass_schema_text() -> str:
    return json.dumps(FIRST_PASS_OUTPUT_SCHEMA, indent=2) + "\n"


def first_pass_instruction_note() -> str:
    return (
        "Follow output_schema.json exactly.\n"
        "On a page, the order field is named reading_order.\n"
        "On a region, the order field is named order.\n"
        "An alternative reading is region_id plus readings, an array of at least two strings.\n"
        "An attention item is kind, text, and region_id.\n"
        "kind is name, date, number, place, organization, event, or term.\n"
        "coverage is required on every page and includes examined_region_ids.\n"
        "A cancellation, insertion, or marginal note is region_id plus text.\n"
        "The page field is diplomatic_transcription, not text_diplomatic.\n"
        "Lines belong inside each region. A line is line_id plus text.\n"
        "An unreadable span is region_id plus text_marker.\n"
        "Do not add a field that output_schema.json does not name.\n"
        "Copy schema_version, provider, provider_interface, task_id, job_id,\n"
        "batch_id, document_id, attempt, source_bundle_sha256, and stage from\n"
        "work_request.json. output_example.json already contains those values.\n"
        "provider_interface is copied from work_request.json. It is GPT_WORK or GROK.\n"
        "Do not omit provider_interface, job_id, batch_id, document_id, attempt,\n"
        "or source_bundle_sha256. A missing field is invalid output.\n"
        "output_example.json is a shape template. Replace its template pages\n"
        "with your transcription. Do not submit the template unchanged.\n"
    )


def write_schema_repair_request(
    original_output: Path,
    destination: Path,
) -> Path:
    """Write a repair request beside a copy of the unmapped values.

    ``original_output`` is read and hashed. This function does not write that
    path. The repaired provider file is not created here.
    """

    source = Path(original_output).resolve()
    target = Path(destination).resolve()
    if target == source or target in source.parents:
        raise ValueError("repair request must not replace the original output")
    payload = json.loads(source.read_bytes())
    digest = sha256_bytes(source.read_bytes())
    provider = payload.get("provider")
    if provider not in {"gpt", "grok"}:
        raise ValueError("schema repair requires provider gpt or grok")
    repaired = target / f"transcription_{provider}.schema_repair.json"
    request = {
        "purpose": "schema_repair",
        "schema_version": "review-provider-output-v1",
        "stage": "first_pass",
        "do_not_retranscribe": True,
        "do_not_modify_original": True,
        "original_output_path": str(source),
        "original_sha256": digest,
        "write_repaired_output_to": str(repaired),
        "keep_unchanged": [
            "diplomatic_transcription",
            "every regions.lines text",
            "every region_id and line_id",
            "the work-request identity fields copied from work_request.json",
        ],
        "lossless_renames_you_may_apply": _lossless_renames(payload),
        "do_not_invent": [
            "coverage",
            "examined_region_ids",
            "inaccessible_or_ambiguous",
            "attention kind",
            "alternative readings that were not already recorded",
            "line text",
            "region ids",
        ],
        "fields_you_must_supply_from_the_existing_reading": [
            "page.coverage.examined_region_ids using only region_id values already present",
            "page.coverage.coverage_complete and unresolved_text_present, which may be copied from coverage_report when those booleans are already present",
            "page.inaccessible_or_ambiguous, using only statements already present, or an empty array if you already recorded none",
            "alternative_readings as region_id plus readings, using only strings already present in that item",
            "attention_items as kind, text, and region_id, or omit an item that has no allowed kind",
            "cancellations and marginal_notes as region_id plus text, using an existing region_id",
        ],
        "omit_from_the_repaired_file": FIRST_PASS_OUTPUT_SCHEMA["rejected_examples"],
        "unmapped_values": _unmapped_values(payload),
        "required_shape": FIRST_PASS_OUTPUT_SCHEMA,
    }
    target.mkdir(parents=True, exist_ok=True)
    request_path = target / "schema_repair_request.json"
    text = json.dumps(request, indent=2, ensure_ascii=False) + "\n"
    request_path.write_text(text, encoding="utf-8")
    (target / f"{provider}_schema_repair_instructions.txt").write_text(
        _repair_instructions(source, repaired),
        encoding="utf-8",
    )
    return request_path


def _repair_instructions(source: Path, repaired: Path) -> str:
    return (
        "Schema repair for an existing first-pass file. Do not transcribe the page again.\n"
        f"Read {source}\n"
        "Keep the diplomatic transcription and every line text unchanged.\n"
        "Do not add readings, region ids, coverage, or visual observations that are not already in that file.\n"
        "Write one review-provider-output-v1 JSON object to:\n"
        f"{repaired}\n"
        "Do not write to the original path.\n"
        "Use schema_repair_request.json for the required fields and the values that have no destination field.\n"
    )


def _lossless_renames(payload: dict[str, object]) -> list[str]:
    """Name copies that keep a value already stored. They are not applied here."""

    notes = [
        "Copy regions[].reading_order to regions[].order and omit reading_order when that integer is present.",
        "Leave page reading_order as it is.",
    ]
    page = _first_page(payload)
    if "text_diplomatic" in page and "diplomatic_transcription" not in page:
        notes.append(
            "Copy pages[].text_diplomatic to diplomatic_transcription. "
            "Do not rebuild that string from the lines."
        )
    regions = page.get("regions")
    lines = page.get("lines")
    if isinstance(lines, list) and lines and isinstance(regions, list):
        notes.append(
            "Move each existing line into the region that already cites its line_id. "
            "Copy text_diplomatic to text and keep line_id. "
            "bbox_px and a line reading_order have no schema field."
        )
    coverage = payload.get("coverage_report")
    if isinstance(coverage, dict) and {
        "coverage_complete",
        "unresolved_text_present",
        "regions_examined",
    } <= set(coverage):
        notes.append(
            "Copy coverage_report.coverage_complete and unresolved_text_present onto page.coverage. "
            "Copy regions_examined to examined_region_ids only when every value is already a region_id."
        )
    spans = page.get("unreadable_spans")
    if isinstance(spans, list) and spans and isinstance(lines, list):
        notes.append(
            "For an unreadable span, copy span to text_marker. "
            "Copy the region_id already stored on that line_id. "
            "Do not invent a marker."
        )
    return notes


def _first_page(payload: dict[str, object]) -> dict[str, object]:
    pages = payload.get("pages")
    if isinstance(pages, list) and pages and isinstance(pages[0], dict):
        return pages[0]
    return {}


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "provider",
    "provider_interface",
    "task_id",
    "job_id",
    "batch_id",
    "document_id",
    "attempt",
    "source_bundle_sha256",
    "stage",
    "pages",
}
_PAGE_FIELDS = {
    "page_id",
    "reading_order",
    "diplomatic_transcription",
    "regions",
    "unreadable_spans",
    "alternative_readings",
    "attention_items",
    "coverage",
    "cancellations",
    "insertions",
    "marginal_notes",
    "inaccessible_or_ambiguous",
}


def _unmapped_values(payload: dict[str, object]) -> dict[str, object]:
    """Copy values that have no first-pass field. Do not interpret them."""

    page = _first_page(payload)
    regions = page.get("regions")
    region_types = []
    region_extras = []
    if isinstance(regions, list):
        for region in regions:
            if not isinstance(region, dict):
                continue
            if "region_type" in region:
                region_types.append(
                    {
                        "region_id": region.get("region_id"),
                        "region_type": region.get("region_type"),
                    }
                )
            extra = {
                key: value
                for key, value in region.items()
                if key not in {"region_id", "lines"}
            }
            if extra:
                region_extras.append({"region_id": region.get("region_id"), **extra})
    return {
        "top_level": {
            key: value for key, value in payload.items() if key not in _TOP_LEVEL_FIELDS
        },
        "page_extras": {
            key: value for key, value in page.items() if key not in _PAGE_FIELDS
        },
        "region_types": region_types,
        "region_extras": region_extras,
        "alternative_readings": page.get("alternative_readings"),
        "attention_items": page.get("attention_items"),
        "unreadable_spans": page.get("unreadable_spans"),
        "cancellations": page.get("cancellations"),
        "insertions": page.get("insertions"),
        "marginal_notes": page.get("marginal_notes"),
    }
