"""Choose the challenge image window for one dispute.

A window is targeted only when a recorded rectangle places it. Rectangles
come from the accepted region box or from original provider geometry kept
as source evidence (``bbox`` or ``bbox_px``). They are never invented from
line order, text similarity, or alignment score. A missing, invalid, or
ambiguous rectangle stays a labeled full-page fallback.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from transcription_review_agent.review.coordinates import (
    corners_to_bbox,
    object_to_bbox,
)
from transcription_review_agent.review.crop import page_bbox
from transcription_review_agent.review.models import (
    BBox,
    Dispute,
    FallbackReason,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    SourceEvidence,
    TextRegion,
)

WindowKind = Literal["TARGETED", "FULL_PAGE_FALLBACK"]
_BOX_KEYS = frozenset({"bbox", "bbox_px"})
_COMPONENT_KEYS = frozenset({"x", "y", "width", "height"})


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    """The page rectangle a challenge crop may use."""

    target: BBox
    kind: WindowKind
    fallback_reason: FallbackReason | None
    coordinate_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind == "TARGETED" and self.fallback_reason is not None:
            raise ValueError("a targeted window has no fallback reason")
        if self.kind == "FULL_PAGE_FALLBACK" and self.fallback_reason is None:
            raise ValueError("a full-page fallback names its reason")


@dataclass(frozen=True, slots=True)
class _Hit:
    box: BBox
    source: str
    region_id: str | None
    line_id: str | None


class _Unusable(Exception):
    def __init__(self, reason: FallbackReason) -> None:
        self.reason = reason


def resolve_dispute_window(
    dispute: Dispute,
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    *,
    page_width: int,
    page_height: int,
    source_evidence: Sequence[SourceEvidence] = (),
) -> EvidenceWindow:
    """Return a targeted window, or a full-page fallback with a reason."""

    page = page_bbox(page_width, page_height)
    hits: list[_Hit] = []
    problems: list[FallbackReason] = []
    for provider, output in (("gpt", gpt), ("grok", grok)):
        _collect_provider(
            hits,
            problems,
            dispute,
            output,
            provider,
            page_width=page_width,
            page_height=page_height,
            source_evidence=source_evidence,
            region_ids=_cited_region_ids(dispute, provider),
        )
    if not hits and not problems and dispute.region_id is not None:
        already = set(dispute.gpt_region_ids) | set(dispute.grok_region_ids)
        if dispute.region_id not in already:
            for provider, output in (("gpt", gpt), ("grok", grok)):
                _collect_provider(
                    hits,
                    problems,
                    dispute,
                    output,
                    provider,
                    page_width=page_width,
                    page_height=page_height,
                    source_evidence=source_evidence,
                    region_ids=[dispute.region_id],
                )
    if problems:
        return _fallback(page, _combined_reason(problems), _sources(hits))
    if not hits:
        return _fallback(page, "NO_RECORDED_COORDINATES", ())
    target = _union([hit.box for hit in hits])
    if target == page:
        return _fallback(page, "RECORDED_BOX_IS_FULL_PAGE", _sources(hits))
    return EvidenceWindow(target, "TARGETED", None, _sources(hits))


def _collect_provider(
    hits: list[_Hit],
    problems: list[FallbackReason],
    dispute: Dispute,
    output: FirstPassOutput,
    provider: str,
    *,
    page_width: int,
    page_height: int,
    source_evidence: Sequence[SourceEvidence],
    region_ids: Sequence[str],
) -> None:
    if not region_ids:
        return
    try:
        hits.extend(
            _provider_hits(
                dispute,
                output,
                provider,
                page_width=page_width,
                page_height=page_height,
                source_evidence=source_evidence,
                region_ids=region_ids,
            )
        )
    except _Unusable as exc:
        problems.append(exc.reason)


def _fallback(
    page: BBox, reason: FallbackReason, sources: tuple[str, ...]
) -> EvidenceWindow:
    return EvidenceWindow(page, "FULL_PAGE_FALLBACK", reason, sources)


def _combined_reason(problems: Sequence[FallbackReason]) -> FallbackReason:
    if "AMBIGUOUS_COORDINATES" in problems:
        return "AMBIGUOUS_COORDINATES"
    if "INVALID_COORDINATES" in problems:
        return "INVALID_COORDINATES"
    return problems[0]


def _sources(hits: Sequence[_Hit]) -> tuple[str, ...]:
    found: list[str] = []
    for hit in hits:
        if hit.source not in found:
            found.append(hit.source)
    return tuple(found)


def _union(boxes: Sequence[BBox]) -> BBox:
    x = min(box.x for box in boxes)
    y = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BBox(x=x, y=y, width=right - x, height=bottom - y)


def _provider_hits(
    dispute: Dispute,
    output: FirstPassOutput,
    provider: str,
    *,
    page_width: int,
    page_height: int,
    source_evidence: Sequence[SourceEvidence],
    region_ids: Sequence[str],
) -> list[_Hit]:
    page = _page(output, dispute.page_id)
    regions = [
        region for region in _ordered(page) if region.region_id in set(region_ids)
    ]
    if not regions:
        return []
    lines = [(region.region_id, line) for region in regions for line in region.lines]
    line_owner = {line.line_id: region_id for region_id, line in lines}
    cited = {region.region_id for region in regions}
    recorded = _source_hits(
        source_evidence,
        provider,
        dispute.page_id,
        cited=cited,
        line_owner=line_owner,
        page_width=page_width,
        page_height=page_height,
    )
    for region in regions:
        if region.bbox is None:
            continue
        _require_inside(region.bbox, page_width, page_height)
        recorded.append(
            _Hit(
                region.bbox,
                f"accepted:{region.region_id}:bbox",
                region.region_id,
                None,
            )
        )
    return _select_hits(dispute, provider, regions, lines, recorded)


def _cited_region_ids(dispute: Dispute, provider: str) -> list[str]:
    listed = dispute.gpt_region_ids if provider == "gpt" else dispute.grok_region_ids
    return list(listed)


def _page(output: FirstPassOutput, page_id: str) -> PageTranscription:
    return next(item for item in output.pages if item.page_id == page_id)


def _ordered(page: PageTranscription) -> list[TextRegion]:
    return sorted(page.regions, key=lambda region: (region.order, region.region_id))


def _source_hits(
    evidence: Sequence[SourceEvidence],
    provider: str,
    page_id: str,
    *,
    cited: set[str],
    line_owner: dict[str, str],
    page_width: int,
    page_height: int,
) -> list[_Hit]:
    grouped: dict[tuple[str, str | None, str | None], dict[str, int]] = {}
    grouped_attached: dict[tuple[str, str | None, str | None], bool] = {}
    hits: list[_Hit] = []
    saw_attached = False
    for item in evidence:
        if item.provider != provider or item.page_id != page_id:
            continue
        parent, key = _split_path(item.json_path)
        attached = _attached(item, cited, line_owner)
        if key in _BOX_KEYS and attached:
            saw_attached = True
            hits.append(
                _hit_from_value(
                    item,
                    key,
                    line_owner,
                    page_width=page_width,
                    page_height=page_height,
                )
            )
            continue
        if key not in _COMPONENT_KEYS or not parent.endswith(".bbox"):
            continue
        identity = (parent, item.region_id, item.line_id)
        parts = grouped.setdefault(identity, {})
        grouped_attached[identity] = grouped_attached.get(identity, False) or attached
        number = _json_int(item.value)
        if not grouped_attached[identity]:
            continue
        if number is None or key in parts:
            raise _Unusable("INVALID_COORDINATES")
        parts[key] = number
    for identity, parts in grouped.items():
        if not grouped_attached.get(identity, False):
            continue
        saw_attached = True
        if set(parts) != _COMPONENT_KEYS:
            raise _Unusable("INVALID_COORDINATES")
        parent, region_id, line_id = identity
        box = _box_from_parts(parts, page_width=page_width, page_height=page_height)
        region = region_id or (line_owner.get(line_id) if line_id else None)
        hits.append(_Hit(box, f"source:{parent}", region, line_id))
    if saw_attached:
        declared = _declared_page_size(evidence, provider, page_id)
        if declared is not None and declared != (page_width, page_height):
            raise _Unusable("INVALID_COORDINATES")
    return hits


def _attached(
    item: SourceEvidence, cited: set[str], line_owner: dict[str, str]
) -> bool:
    if item.region_id in cited:
        return True
    return item.line_id is not None and item.line_id in line_owner


def _declared_page_size(
    evidence: Sequence[SourceEvidence], provider: str, page_id: str
) -> tuple[int, int] | None:
    sizes: dict[str, list[int | None]] = {}
    for item in evidence:
        if item.provider != provider or item.page_id != page_id:
            continue
        if item.region_id is not None or item.line_id is not None:
            continue
        parent, key = _split_path(item.json_path)
        if key not in {"width", "height"}:
            continue
        number = _json_int(item.value)
        if number is None:
            continue
        pair = sizes.setdefault(parent, [None, None])
        slot = 0 if key == "width" else 1
        if pair[slot] is not None and pair[slot] != number:
            raise _Unusable("AMBIGUOUS_COORDINATES")
        pair[slot] = number
    declared = [
        (pair[0], pair[1])
        for pair in sizes.values()
        if pair[0] is not None and pair[1] is not None
    ]
    if not declared:
        return None
    unique = set(declared)
    if len(unique) != 1:
        raise _Unusable("AMBIGUOUS_COORDINATES")
    width, height = declared[0]
    if width is None or height is None:
        return None
    return width, height


def _hit_from_value(
    item: SourceEvidence,
    key: str,
    line_owner: dict[str, str],
    *,
    page_width: int,
    page_height: int,
) -> _Hit:
    try:
        parsed: object = json.loads(item.value)
    except json.JSONDecodeError as exc:
        raise _Unusable("INVALID_COORDINATES") from exc
    if isinstance(parsed, dict):
        if key == "bbox_px":
            raise _Unusable("AMBIGUOUS_COORDINATES")
        box = _box_from_object(parsed, page_width=page_width, page_height=page_height)
    elif isinstance(parsed, list):
        if key != "bbox_px":
            raise _Unusable("AMBIGUOUS_COORDINATES")
        box = _box_from_corners(parsed, page_width=page_width, page_height=page_height)
    else:
        raise _Unusable("INVALID_COORDINATES")
    region = item.region_id or (
        line_owner.get(item.line_id) if item.line_id is not None else None
    )
    return _Hit(box, f"source:{item.json_path}", region, item.line_id)


def _box_from_object(
    parsed: dict[object, object], *, page_width: int, page_height: int
) -> BBox:
    box = object_to_bbox(parsed)
    if box is None:
        raise _Unusable("INVALID_COORDINATES")
    _require_inside(box, page_width, page_height)
    return box


def _box_from_corners(
    parsed: list[object], *, page_width: int, page_height: int
) -> BBox:
    """Read ``bbox_px`` as half-open corners ``[x0, y0, x1, y1]``."""

    box = corners_to_bbox(parsed)
    if box is None:
        raise _Unusable("INVALID_COORDINATES")
    _require_inside(box, page_width, page_height)
    return box


def _box_from_parts(
    parts: dict[str, int], *, page_width: int, page_height: int
) -> BBox:
    try:
        box = BBox(
            x=parts["x"],
            y=parts["y"],
            width=parts["width"],
            height=parts["height"],
        )
    except ValueError as exc:
        raise _Unusable("INVALID_COORDINATES") from exc
    _require_inside(box, page_width, page_height)
    return box


def _require_inside(box: BBox, page_width: int, page_height: int) -> None:
    if not page_bbox(page_width, page_height).contains(box):
        raise _Unusable("INVALID_COORDINATES")


def _select_hits(
    dispute: Dispute,
    provider: str,
    regions: Sequence[TextRegion],
    lines: Sequence[tuple[str, LineReading]],
    recorded: Sequence[_Hit],
) -> list[_Hit]:
    focused = _exact_line_hits(dispute, provider, lines, recorded)
    if focused is not None:
        _require_line_inside_region(regions, focused, recorded)
        return focused
    _reject_line_region_conflict(regions, recorded)
    if (
        lines
        and _all_lines_located(lines, recorded)
        and not _has_region_box(regions, recorded)
    ):
        return _line_hits(lines, recorded)
    if _line_geometry_is_partial(lines, recorded) and not _has_region_box(
        regions, recorded
    ):
        raise _Unusable("AMBIGUOUS_COORDINATES")
    chosen: list[_Hit] = []
    for region in regions:
        region_hits = [
            hit
            for hit in recorded
            if hit.line_id is None and hit.region_id == region.region_id
        ]
        if not region_hits:
            continue
        chosen.extend(_one_box(region_hits))
    return chosen


def _exact_line_hits(
    dispute: Dispute,
    provider: str,
    lines: Sequence[tuple[str, LineReading]],
    recorded: Sequence[_Hit],
) -> list[_Hit] | None:
    text = dispute.gpt_text if provider == "gpt" else dispute.grok_text
    if not text or not lines:
        return None
    matches: list[Sequence[tuple[str, LineReading]]] = []
    for start in range(len(lines)):
        for end in range(start + 1, len(lines) + 1):
            joined = "\n".join(line.text for _region_id, line in lines[start:end])
            if joined == text:
                matches.append(lines[start:end])
    if len(matches) != 1:
        return None
    chosen: list[_Hit] = []
    for _region_id, line in matches[0]:
        found = [hit for hit in recorded if hit.line_id == line.line_id]
        if not found:
            return None
        chosen.extend(_one_box(found))
    return chosen


def _line_geometry_is_partial(
    lines: Sequence[tuple[str, LineReading]], recorded: Sequence[_Hit]
) -> bool:
    located = {hit.line_id for hit in recorded if hit.line_id is not None}
    owned = {line.line_id for _region_id, line in lines}
    present = located & owned
    return bool(present) and present != owned


def _all_lines_located(
    lines: Sequence[tuple[str, LineReading]], recorded: Sequence[_Hit]
) -> bool:
    return all(
        any(hit.line_id == line.line_id for hit in recorded)
        for _region_id, line in lines
    )


def _line_hits(
    lines: Sequence[tuple[str, LineReading]], recorded: Sequence[_Hit]
) -> list[_Hit]:
    chosen: list[_Hit] = []
    for _region_id, line in lines:
        found = [hit for hit in recorded if hit.line_id == line.line_id]
        chosen.extend(_one_box(found))
    return chosen


def _has_region_box(regions: Sequence[TextRegion], recorded: Sequence[_Hit]) -> bool:
    cited = {region.region_id for region in regions}
    return any(hit.line_id is None and hit.region_id in cited for hit in recorded)


def _require_line_inside_region(
    regions: Sequence[TextRegion],
    line_hits: Sequence[_Hit],
    recorded: Sequence[_Hit],
) -> None:
    for region in regions:
        containers = [
            hit
            for hit in recorded
            if hit.line_id is None and hit.region_id == region.region_id
        ]
        if not containers:
            continue
        agreed = _one_box(containers)
        used = [hit for hit in line_hits if hit.region_id == region.region_id]
        if any(not agreed[0].box.contains(hit.box) for hit in used):
            raise _Unusable("AMBIGUOUS_COORDINATES")


def _reject_line_region_conflict(
    regions: Sequence[TextRegion], recorded: Sequence[_Hit]
) -> None:
    for region in regions:
        containers = [
            hit
            for hit in recorded
            if hit.line_id is None and hit.region_id == region.region_id
        ]
        lines = [
            hit
            for hit in recorded
            if hit.line_id is not None and hit.region_id == region.region_id
        ]
        if not containers or not lines:
            continue
        box = _one_box(containers)[0].box
        if any(not box.contains(hit.box) for hit in lines):
            raise _Unusable("AMBIGUOUS_COORDINATES")


def _one_box(hits: Sequence[_Hit]) -> list[_Hit]:
    boxes = {(hit.box.x, hit.box.y, hit.box.width, hit.box.height) for hit in hits}
    if len(boxes) != 1:
        raise _Unusable("AMBIGUOUS_COORDINATES")
    return list(hits)


def _split_path(path: str) -> tuple[str, str]:
    parent, separator, key = path.rpartition(".")
    if not separator:
        parent = ""
    key = key.split("[", 1)[0]
    return parent, key


def _json_int(value: str) -> int | None:
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, bool) or not isinstance(parsed, int):
        return None
    return parsed
