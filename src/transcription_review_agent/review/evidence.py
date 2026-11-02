"""Supplementary evidence that FirstPassOutput cannot store.

The canonical provider file stays strict. Values that the repair could not
copy are read from the original JSON and kept with their field path. This
module does not write the original file and does not invent a region, a
line, or a reading.
"""

from __future__ import annotations

import json
from typing import Any

from transcription_review_agent.review.models import FirstPassOutput, SourceEvidence

_STRUCTURAL = {
    "pages",
    "regions",
    "lines",
    "alternative_readings",
    "unreadable_spans",
    "cancellations",
    "insertions",
    "marginal_notes",
    "attention_items",
    "coverage",
    "coverage_report",
}

_SKIP_SCALARS = {
    "coverage_complete",
    "unresolved_text_present",
    "reading_order",
    "order",
    "document_id",
    "attempt",
    "round_index",
}


def collect_source_evidence(
    original: dict[str, Any],
    repaired: FirstPassOutput,
    *,
    original_sha256: str,
) -> list[SourceEvidence]:
    """Return original values that are not already stored on ``repaired``.

    A string that appears unchanged in the repaired file is not repeated.
    Region and line ids are copied only from objects in ``original``.
    """

    preserved = _strings(repaired.model_dump(mode="json"))
    line_regions = _line_regions(original)
    found: list[SourceEvidence] = []
    _walk(
        original,
        "",
        {},
        line_regions,
        preserved,
        repaired.provider,
        original_sha256,
        found,
    )
    return found


def _strings(value: Any) -> set[str]:
    found: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, str):
            found.add(node)
        elif isinstance(node, dict):
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return found


def _line_regions(
    node: Any, current: str | None = None, found: dict[str, str] | None = None
) -> dict[str, str]:
    mapping = {} if found is None else found
    if isinstance(node, dict):
        region = (
            node.get("region_id") if isinstance(node.get("region_id"), str) else current
        )
        line_id = node.get("line_id")
        if isinstance(line_id, str) and line_id and isinstance(region, str) and region:
            mapping.setdefault(line_id, region)
        for value in node.values():
            _line_regions(
                value, region if isinstance(region, str) else current, mapping
            )
    elif isinstance(node, list):
        for value in node:
            _line_regions(value, current, mapping)
    return mapping


def _walk(
    node: Any,
    path: str,
    context: dict[str, str],
    line_regions: dict[str, str],
    preserved: set[str],
    provider: str,
    original_sha256: str,
    found: list[SourceEvidence],
) -> None:
    if isinstance(node, dict):
        local = _context(node, context, line_regions)
        for key, value in node.items():
            child = f"{path}.{key}" if path else key
            if _recurse(key, value):
                _walk(
                    value,
                    child,
                    local,
                    line_regions,
                    preserved,
                    provider,
                    original_sha256,
                    found,
                )
            else:
                _consider(
                    child,
                    key,
                    value,
                    local,
                    preserved,
                    provider,
                    original_sha256,
                    found,
                )
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            _walk(
                value,
                f"{path}[{index}]",
                context,
                line_regions,
                preserved,
                provider,
                original_sha256,
                found,
            )
        return
    key = path.rsplit(".", 1)[-1]
    key = key.split("[", 1)[0]
    _consider(path, key, node, context, preserved, provider, original_sha256, found)


def _recurse(key: str, value: Any) -> bool:
    if key in _STRUCTURAL:
        return True
    if isinstance(value, dict):
        return True
    if isinstance(value, list) and any(
        isinstance(item, (dict, list)) for item in value
    ):
        return True
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return True
    return False


def _context(
    node: dict[str, Any], context: dict[str, str], line_regions: dict[str, str]
) -> dict[str, str]:
    local = dict(context)
    page_id = node.get("page_id")
    if isinstance(page_id, str) and page_id:
        local["page_id"] = page_id
    region_id = node.get("region_id")
    if isinstance(region_id, str) and region_id:
        local["region_id"] = region_id
    line_id = node.get("line_id")
    if not isinstance(line_id, str) or not line_id:
        line_ids = node.get("line_ids")
        if (
            isinstance(line_ids, list)
            and len(line_ids) == 1
            and isinstance(line_ids[0], str)
            and line_ids[0]
        ):
            line_id = line_ids[0]
    if isinstance(line_id, str) and line_id:
        local["line_id"] = line_id
        if "region_id" not in local and line_id in line_regions:
            local["region_id"] = line_regions[line_id]
    return local


def _consider(
    path: str,
    key: str,
    value: Any,
    context: dict[str, str],
    preserved: set[str],
    provider: str,
    original_sha256: str,
    found: list[SourceEvidence],
) -> None:
    if value is None or value == "" or value == [] or value == {}:
        return
    if key in _SKIP_SCALARS:
        return
    if isinstance(value, str) and value in preserved:
        return
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item in preserved for item in value)
    ):
        return
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    found.append(
        SourceEvidence(
            provider=provider,  # type: ignore[arg-type]
            original_sha256=original_sha256,
            json_path=path,
            value=encoded,
            page_id=context.get("page_id"),
            region_id=context.get("region_id"),
            line_id=context.get("line_id"),
        )
    )
