"""Copy recorded page rectangles into the first-pass region box.

``bbox_px`` is half-open corners ``[x0, y0, x1, y1]``. The supported region
field is ``bbox`` as ``{x, y, width, height}``. A value is copied only when
it is that shape, uses integers, and lies inside the sealed page. Nothing
here invents a box, clamps one, or matches text to a position.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from transcription_review_agent.review.crop import page_bbox
from transcription_review_agent.review.models import BBox


def corners_to_bbox(value: object) -> BBox | None:
    """Convert half-open ``[x0, y0, x1, y1]`` corners. ``None`` if not that form.

    The result is not clipped to a page. A reversed or empty span is rejected.
    """

    if not isinstance(value, list) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    x0, y0, x1, y1 = value
    if not all(isinstance(item, int) for item in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    try:
        return BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
    except ValueError:
        return None


def object_to_bbox(value: object) -> BBox | None:
    """Convert ``{x, y, width, height}`` integers. ``None`` if not that form."""

    if not isinstance(value, dict):
        return None
    if set(value) != {"x", "y", "width", "height"}:
        return None
    parts: dict[str, int] = {}
    for name in ("x", "y", "width", "height"):
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, int):
            return None
        parts[name] = number
    try:
        return BBox(
            x=parts["x"],
            y=parts["y"],
            width=parts["width"],
            height=parts["height"],
        )
    except ValueError:
        return None


def prepare_first_pass_coordinates(
    payload: Mapping[str, object],
    pages: Sequence[tuple[str, int, int]],
) -> dict[str, object]:
    """Return a copy whose valid recorded boxes are ``region.bbox``.

    ``pages`` is the sealed ``(page_id, width, height)`` list. The input
    mapping is not modified. ``bbox_px`` is removed from the copy so the
    supported file can validate. An invalid or conflicting box is omitted,
    not repaired.
    """

    copied = copy.deepcopy(dict(payload))
    sealed = {page_id: (width, height) for page_id, width, height in pages}
    page_list = copied.get("pages")
    if not isinstance(page_list, list):
        return copied
    for page in page_list:
        if not isinstance(page, dict):
            continue
        page_id = page.get("page_id")
        if not isinstance(page_id, str) or page_id not in sealed:
            _strip_page_coordinates(page)
            continue
        width, height = sealed[page_id]
        if not _declared_size_matches(page, width, height):
            _strip_page_coordinates(page)
            continue
        regions = page.get("regions")
        if not isinstance(regions, list):
            continue
        for region in regions:
            if isinstance(region, dict):
                _prepare_region(region, width, height)
    return copied


def _prepare_region(region: dict[str, object], width: int, height: int) -> None:
    kept, blocked = _recorded_box(
        region.get("bbox"), region.get("bbox_px"), width, height
    )
    region.pop("bbox_px", None)
    line_boxes = _line_boxes(region, width, height)
    if blocked or (kept is None and line_boxes is None):
        region.pop("bbox", None)
        return
    if kept is None:
        if not line_boxes:
            region.pop("bbox", None)
            return
        kept = _union(line_boxes)
        if not _inside(kept, width, height):
            region.pop("bbox", None)
            return
    region["bbox"] = kept.model_dump(mode="json")


def _line_boxes(
    region: dict[str, object], width: int, height: int
) -> list[BBox] | None:
    """Every line box, or ``None`` when they cannot fill a missing region box.

    ``None`` means "do not substitute". An empty list means the lines recorded
    no coordinates. A partial or invalid set is also ``None``. Coordinate
    fields are removed from lines because a line has no box field.
    """

    lines = region.get("lines")
    if not isinstance(lines, list):
        return []
    states: list[object] = []
    for line in lines:
        if not isinstance(line, dict):
            states.append("bad")
            continue
        present = "bbox" in line or "bbox_px" in line
        box, blocked = _recorded_box(
            line.get("bbox"), line.get("bbox_px"), width, height
        )
        line.pop("bbox", None)
        line.pop("bbox_px", None)
        if not present:
            states.append("absent")
        elif blocked or box is None:
            states.append("bad")
        else:
            states.append(box)
    if any(item == "bad" for item in states):
        return None
    boxes = [item for item in states if isinstance(item, BBox)]
    if not boxes:
        return []
    if len(boxes) != len(states):
        return None
    return boxes


def _recorded_box(
    bbox_value: object, px_value: object, width: int, height: int
) -> tuple[BBox | None, bool]:
    """Return the one safe box, and whether recorded geometry blocks a substitute.

    The second value is true when a field was a real rectangle that does not
    fit the page, or two rectangles disagree. Unparseable fields are ignored.
    """

    found: list[BBox] = []
    blocked = False
    for kind, value in (("bbox", bbox_value), ("bbox_px", px_value)):
        if value is None:
            continue
        parsed = object_to_bbox(value) if kind == "bbox" else corners_to_bbox(value)
        if parsed is None:
            continue
        if not _inside(parsed, width, height):
            blocked = True
            continue
        found.append(parsed)
    if blocked:
        return None, True
    if not found:
        return None, False
    first = found[0]
    if any(item != first for item in found[1:]):
        return None, True
    return first, False


def _declared_size_matches(page: Mapping[str, object], width: int, height: int) -> bool:
    declared_width = page.get("width")
    declared_height = page.get("height")
    if declared_width is None and declared_height is None:
        return True
    if isinstance(declared_width, bool) or isinstance(declared_height, bool):
        return False
    if not isinstance(declared_width, int) or not isinstance(declared_height, int):
        return False
    return (declared_width, declared_height) == (width, height)


def _strip_page_coordinates(page: dict[str, object]) -> None:
    regions = page.get("regions")
    if not isinstance(regions, list):
        return
    for region in regions:
        if not isinstance(region, dict):
            continue
        region.pop("bbox", None)
        region.pop("bbox_px", None)
        lines = region.get("lines")
        if not isinstance(lines, list):
            continue
        for line in lines:
            if isinstance(line, dict):
                line.pop("bbox", None)
                line.pop("bbox_px", None)


def _inside(box: BBox, width: int, height: int) -> bool:
    return page_bbox(width, height).contains(box)


def _union(boxes: Sequence[BBox]) -> BBox:
    x = min(box.x for box in boxes)
    y = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BBox(x=x, y=y, width=right - x, height=bottom - y)
