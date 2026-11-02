"""Crop windows on existing page pixels.

The crop is a rectangle inside the page. Making the rectangle larger adds
surrounding context from the same scan. It does not create visual detail.
"""

from __future__ import annotations

import struct

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.models import BBox, CropRecord
from transcription_review_agent.review.pngutil import crop_rows, decode_png, encode_png

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Read width and height from a PNG IHDR chunk."""

    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("page image must be a PNG")
    if len(data) < 24 or data[12:16] != b"IHDR":
        raise ValueError("PNG is missing an IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise ValueError("PNG dimensions must be positive")
    return width, height


def page_bbox(width: int, height: int) -> BBox:
    return BBox(x=0, y=0, width=width, height=height)


def build_crop(
    *,
    crop_id: str,
    dispute_id: str | None = None,
    page_id: str,
    page_sha256: str,
    page_width: int,
    page_height: int,
    target: BBox,
    margin_ratio: float = 0.25,
    window_kind: str = "TARGETED",
    group_id: str | None = None,
    dispute_ids: list[str] | None = None,
) -> CropRecord:
    """Enlarge ``target`` for context and prove the target lies inside the crop.

    A crop belongs to one dispute (``dispute_id``) or is shared by a visual
    evidence group (``group_id`` with its ``dispute_ids``).
    """

    page = page_bbox(page_width, page_height)
    if window_kind not in {"TARGETED", "FULL_PAGE_FALLBACK"}:
        raise ValueError("crop window kind is not recognized")
    if window_kind == "FULL_PAGE_FALLBACK" and target != page:
        raise ValueError(
            "a full-page fallback covers the page and is not a targeted crop"
        )
    if not page.contains(target):
        raise ValueError("target region is outside the page")
    crop = (
        page
        if window_kind == "FULL_PAGE_FALLBACK"
        else _enlarge(target, page, margin_ratio)
    )
    if not crop.contains(target):
        raise ValueError("crop does not contain the target region")
    return CropRecord(
        crop_id=crop_id,
        dispute_id=dispute_id,
        group_id=group_id,
        dispute_ids=dispute_ids,
        page_id=page_id,
        page_sha256=page_sha256,
        page_width=page_width,
        page_height=page_height,
        target_bbox=target,
        crop_bbox=crop,
        enlarged=crop.area > target.area,
        window_kind=window_kind,  # type: ignore[arg-type]
    )


def materialize_crop(record: CropRecord, page_png: bytes) -> tuple[CropRecord, bytes]:
    """Cut the crop rectangle out of the page and record its bytes."""

    assert_crop_contains_target(record)
    width, height, rows, bpp = decode_png(page_png)
    if width != record.page_width or height != record.page_height:
        raise ValueError("page image dimensions do not match the sealed page")
    cropped = crop_rows(
        rows,
        x=record.crop_bbox.x,
        y=record.crop_bbox.y,
        width=record.crop_bbox.width,
        height=record.crop_bbox.height,
        bpp=bpp,
        page_width=width,
        page_height=height,
    )
    encoded = encode_png(record.crop_bbox.width, record.crop_bbox.height, cropped, bpp)
    crop_width, crop_height, crop_pixels, crop_bpp = decode_png(encoded)
    if (crop_width, crop_height, crop_bpp) != (
        record.crop_bbox.width,
        record.crop_bbox.height,
        bpp,
    ):
        raise ValueError("encoded crop dimensions do not match the crop window")
    _assert_target_pixels_match(record, rows, crop_pixels, bpp)
    updated = record.model_copy(
        update={
            "crop_sha256": sha256_bytes(encoded),
            "crop_png_width": crop_width,
            "crop_png_height": crop_height,
        }
    )
    return updated, encoded


def _assert_target_pixels_match(
    record: CropRecord,
    source_rows: list[bytearray],
    crop_rows_pixels: list[bytearray],
    bpp: int,
) -> None:
    sample_x = record.target_bbox.x
    sample_y = record.target_bbox.y
    crop_x = sample_x - record.crop_bbox.x
    crop_y = sample_y - record.crop_bbox.y
    source = source_rows[sample_y][sample_x * bpp : (sample_x + 1) * bpp]
    cropped = crop_rows_pixels[crop_y][crop_x * bpp : (crop_x + 1) * bpp]
    if source != cropped:
        raise ValueError("crop pixels do not match the source page")


def assert_crop_contains_target(record: CropRecord) -> None:
    """Fail closed when a stored crop does not cover its target."""

    page = page_bbox(record.page_width, record.page_height)
    if not page.contains(record.crop_bbox):
        raise ValueError("crop is outside the page")
    if not record.crop_bbox.contains(record.target_bbox):
        raise ValueError("crop does not contain the target region")
    if record.represents_new_visual_detail or record.scale != 1:
        raise ValueError("a crop must not claim new visual detail")


def _enlarge(target: BBox, page: BBox, margin_ratio: float) -> BBox:
    margin_x = max(1, int(page.width * margin_ratio)) if target.area < page.area else 0
    margin_y = max(1, int(page.height * margin_ratio)) if target.area < page.area else 0
    if target.area >= page.area:
        return page
    x = max(page.x, target.x - margin_x)
    y = max(page.y, target.y - margin_y)
    right = min(page.x + page.width, target.x + target.width + margin_x)
    bottom = min(page.y + page.height, target.y + target.height + margin_y)
    crop = BBox(x=x, y=y, width=right - x, height=bottom - y)
    if not crop.contains(target):
        raise ValueError("enlarged crop missed the target region")
    return crop
