"""Pixel crops are cut from the page and invalid geometry is rejected."""

from __future__ import annotations

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.crop import build_crop, materialize_crop
from transcription_review_agent.review.models import BBox
from transcription_review_agent.review.pngutil import decode_png, encode_png

import pytest


def _page(
    width: int, height: int, colored: tuple[int, int, tuple[int, int, int]]
) -> bytes:
    rows = [bytearray(b"\x00" * (width * 3)) for _ in range(height)]
    x, y, color = colored
    rows[y][x * 3 : x * 3 + 3] = bytes(color)
    return encode_png(width, height, rows, 3)


def test_materialized_crop_keeps_target_pixels_and_records_provenance() -> None:
    page = _page(20, 20, (6, 6, (10, 20, 30)))
    record = build_crop(
        crop_id="crop_pixel",
        dispute_id="dsp_pixel",
        page_id="page_001",
        page_sha256="ab" * 32,
        page_width=20,
        page_height=20,
        target=BBox(x=4, y=4, width=4, height=4),
    )
    updated, encoded = materialize_crop(record, page)
    width, height, rows, _bpp = decode_png(encoded)
    assert (width, height) == (updated.crop_png_width, updated.crop_png_height)
    assert (width, height) == (record.crop_bbox.width, record.crop_bbox.height)
    assert width < 20 and height < 20
    crop_x = 6 - record.crop_bbox.x
    crop_y = 6 - record.crop_bbox.y
    assert bytes(rows[crop_y][crop_x * 3 : crop_x * 3 + 3]) == bytes((10, 20, 30))
    assert updated.crop_sha256 == sha256_bytes(encoded)
    assert updated.full_page_retained is True
    assert sha256_bytes(encoded) != sha256_bytes(page)


def test_invalid_target_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside the page"):
        build_crop(
            crop_id="crop_outside",
            dispute_id="dsp_out",
            page_id="page_001",
            page_sha256="ab" * 32,
            page_width=10,
            page_height=10,
            target=BBox(x=8, y=8, width=4, height=4),
        )
    record = build_crop(
        crop_id="crop_ok",
        dispute_id="dsp_ok",
        page_id="page_001",
        page_sha256="ab" * 32,
        page_width=10,
        page_height=10,
        target=BBox(x=1, y=1, width=2, height=2),
    )
    with pytest.raises(ValueError, match="dimensions do not match"):
        materialize_crop(record, _page(8, 8, (1, 1, (1, 2, 3))))
