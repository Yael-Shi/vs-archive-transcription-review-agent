"""Sealed JPEG pages stay original. Crops use a derived PNG."""

from __future__ import annotations

import json
import struct

import pytest

from test_review_workflow import BATCH_ID, DOCUMENT_ID, _controller, _snapshot
from transcription_review_agent.review.crop import build_crop, materialize_crop
from transcription_review_agent.review.jpegutil import (
    _AC_LUMINANCE_BITS,
    _AC_LUMINANCE_VALUES,
    _BitWriter,
    _DC_LUMINANCE_BITS,
    _DC_LUMINANCE_VALUES,
    _category,
    _code_for,
    _dht,
    _huffman,
    _marker,
    decode_jpeg,
    encode_grayscale_dc_jpeg,
    jpeg_dimensions,
    jpeg_to_png,
)
from transcription_review_agent.review.models import BBox
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.pngutil import decode_png


def test_sealed_jpeg_is_preserved_and_crops_use_a_derived_png(tmp_path) -> None:
    jpeg = encode_grayscale_dc_jpeg(16, 8, [0, 1])
    controller = _controller(tmp_path, pages=[jpeg], mime="image/jpeg")
    before = _snapshot(tmp_path / "batches")
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    assert _snapshot(tmp_path / "batches") == before

    doc_root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    original = (doc_root / "input" / "page_001.jpg").read_bytes()
    derived = (doc_root / "input" / "page_001.derived.png").read_bytes()
    assert original == jpeg
    assert derived == jpeg_to_png(jpeg)
    manifest = json.loads(
        (doc_root / "input" / "source_manifest.json").read_text(encoding="utf-8")
    )
    page = manifest["pages"][0]
    assert page["filename"] == "page_001.jpg"
    assert page["mime_type"] == "image/jpeg"
    assert page["sha256"] != page["derived_png_sha256"]
    assert page["derived_png_filename"] == "page_001.derived.png"
    _width, _height, rows, _bpp = decode_png(derived)
    assert bytes(rows[0][:8]) == b"\x80" * 8
    assert bytes(rows[0][8:]) == b"\x81" * 8

    record = build_crop(
        crop_id="crop_jpeg",
        dispute_id="dsp_jpeg",
        page_id="page_001",
        page_sha256=page["sha256"],
        page_width=16,
        page_height=8,
        target=BBox(x=8, y=0, width=8, height=8),
    )
    updated, crop = materialize_crop(record, derived)
    _crop_width, _crop_height, crop_rows, _crop_bpp = decode_png(crop)
    assert updated.crop_png_width == record.crop_bbox.width
    sample = 8 - record.crop_bbox.x
    assert crop_rows[0][sample] == 129
    assert original == jpeg


def test_unsupported_or_huge_jpeg_is_rejected() -> None:
    sampled = b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x08\x00\x08\x01\x01\x22\x00\xff\xd9"
    with pytest.raises(ValueError, match="4:4:4"):
        jpeg_dimensions(sampled)
    huge = b"\xff\xd8\xff\xc0\x00\x0b\x08\x7f\xff\x7f\xff\x01\x01\x11\x00\xff\xd9"
    with pytest.raises(ValueError, match="exceed the review limit"):
        jpeg_dimensions(huge)
    extended = b"\xff\xd8\xff\xc1\x00\x0b\x08\x00\x08\x00\x08\x01\x01\x11\x00\xff\xd9"
    with pytest.raises(ValueError, match="frame type is not supported"):
        jpeg_dimensions(extended)
    progressive_sampling = (
        b"\xff\xd8\xff\xc2\x00\x0b\x08\x00\x08\x00\x08\x01\x01\x33\x00\xff\xd9"
    )
    with pytest.raises(ValueError, match="sampling factors"):
        jpeg_dimensions(progressive_sampling)
    truncated = _progressive_grayscale(8, 8, scans=[(0, 0, 0, 0, b"")])[:-2]
    with pytest.raises(ValueError, match="truncated|missing a marker|missing a scan"):
        decode_jpeg(truncated)
    flat = encode_grayscale_dc_jpeg(8, 8, [0])
    width, _height, rows, _bpp = decode_jpeg(flat)
    assert width == 8
    assert bytes(rows[0]) == b"\x80" * 8


def test_progressive_dc_matches_baseline_pixels() -> None:
    for coefficient in (0, 1, -1):
        baseline = encode_grayscale_dc_jpeg(8, 8, [coefficient])
        high = coefficient >> 1 if coefficient >= 0 else -1
        low_bit = coefficient & 1
        progressive = _progressive_grayscale(
            8,
            8,
            scans=[
                (0, 0, 0, 1, _dc_entropy([high])),
                (0, 0, 1, 0, _raw_bits([low_bit])),
            ],
        )
        assert decode_jpeg(progressive) == decode_jpeg(baseline)
        derived = jpeg_to_png(progressive)
        assert jpeg_to_png(progressive) == derived
        width, height = jpeg_dimensions(progressive)
        assert (width, height) == png_dimensions_of(derived)


def test_progressive_ac_and_refinement_match_baseline_pixels() -> None:
    baseline = _baseline_grayscale_ac(8, 8, dc=0, ac_index=1, ac_value=1)
    initial = _progressive_grayscale(
        8,
        8,
        scans=[
            (0, 0, 0, 0, _dc_entropy([0])),
            (1, 63, 0, 0, _ac_entropy([(1, 1)])),
        ],
    )
    refined = _progressive_grayscale(
        8,
        8,
        scans=[
            (0, 0, 0, 0, _dc_entropy([0])),
            (1, 63, 0, 1, _ac_entropy([])),
            (1, 63, 1, 0, _ac_refine_new(run=0, positive=True)),
        ],
    )
    assert decode_jpeg(initial) == decode_jpeg(baseline)
    assert decode_jpeg(refined) == decode_jpeg(baseline)


def test_progressive_420_block_order_and_upsampling() -> None:
    # One MCU. Y DC order is left-to-right, top-to-bottom. Chroma DC is zero.
    image = _progressive_color(
        16,
        16,
        y_sampling=0x22,
        y_dc=[0, 1, 2, 3],
        cb_dc=[0],
        cr_dc=[0],
    )
    width, height, rows, bpp = decode_jpeg(image)
    assert (width, height, bpp) == (16, 16, 3)
    expected = {
        (0, 0): 128,
        (8, 0): 129,
        (0, 8): 130,
        (8, 8): 131,
    }
    for (origin_x, origin_y), level in expected.items():
        for y in range(origin_y, origin_y + 8):
            for x in range(origin_x, origin_x + 8):
                pixel = bytes(rows[y][x * 3 : x * 3 + 3])
                assert pixel == bytes((level, level, level))
    assert jpeg_to_png(image) == jpeg_to_png(image)


def png_dimensions_of(data: bytes) -> tuple[int, int]:
    width, height, _rows, _bpp = decode_png(data)
    return width, height


def _dc_table():
    return _huffman(_DC_LUMINANCE_BITS, _DC_LUMINANCE_VALUES)


def _ac_table():
    return _huffman(_AC_LUMINANCE_BITS, _AC_LUMINANCE_VALUES)


def _write_dc(
    writer: _BitWriter, values: list[int], table: dict[tuple[int, int], int]
) -> None:
    predictor = 0
    for value in values:
        diff = value - predictor
        predictor = value
        category, extra = _category(diff)
        writer.write(*_code_for(table, category))
        if category:
            writer.write(extra, category)


def _dc_entropy(values: list[int]) -> bytes:
    writer = _BitWriter()
    _write_dc(writer, values, _dc_table())
    return writer.finish()


def _raw_bits(bits: list[int]) -> bytes:
    writer = _BitWriter()
    for bit in bits:
        writer.write(bit, 1)
    return writer.finish()


def _ac_entropy(entries: list[tuple[int, int]]) -> bytes:
    writer = _BitWriter()
    table = _ac_table()
    cursor = 1
    for index, value in entries:
        run = index - cursor
        while run >= 16:
            writer.write(*_code_for(table, 0xF0))
            run -= 16
        category, extra = _category(value)
        writer.write(*_code_for(table, (run << 4) | category))
        if category:
            writer.write(extra, category)
        cursor = index + 1
    writer.write(*_code_for(table, 0x00))
    return writer.finish()


def _ac_refine_new(*, run: int, positive: bool) -> bytes:
    writer = _BitWriter()
    table = _ac_table()
    writer.write(*_code_for(table, (run << 4) | 1))
    writer.write(1 if positive else 0, 1)
    writer.write(*_code_for(table, 0x00))
    return writer.finish()


def _quant() -> bytes:
    return _marker(b"\xff\xdb", bytes([0]) + bytes([8] * 64))


def _frame(
    marker: bytes, width: int, height: int, components: list[tuple[int, int, int]]
) -> bytes:
    payload = struct.pack(">BHHB", 8, height, width, len(components))
    payload += b"".join(
        bytes((component_id, sampling, table))
        for component_id, sampling, table in components
    )
    return _marker(marker, payload)


def _scan(
    components: list[tuple[int, int]],
    ss: int,
    se: int,
    ah: int,
    al: int,
    entropy: bytes,
) -> bytes:
    payload = bytes([len(components)])
    payload += b"".join(bytes(item) for item in components)
    payload += bytes((ss, se, (ah << 4) | al))
    return _marker(b"\xff\xda", payload) + entropy


def _progressive_grayscale(
    width: int, height: int, scans: list[tuple[int, int, int, int, bytes]]
) -> bytes:
    parts = [
        b"\xff\xd8",
        _quant(),
        _frame(b"\xff\xc2", width, height, [(1, 0x11, 0)]),
        _dht(0x00, _DC_LUMINANCE_BITS, _DC_LUMINANCE_VALUES),
        _dht(0x10, _AC_LUMINANCE_BITS, _AC_LUMINANCE_VALUES),
    ]
    parts.extend(
        _scan([(1, 0x00)], ss, se, ah, al, entropy) for ss, se, ah, al, entropy in scans
    )
    parts.append(b"\xff\xd9")
    return b"".join(parts)


def _baseline_grayscale_ac(
    width: int, height: int, *, dc: int, ac_index: int, ac_value: int
) -> bytes:
    writer = _BitWriter()
    _write_dc(writer, [dc], _dc_table())
    table = _ac_table()
    category, extra = _category(ac_value)
    run = ac_index - 1
    writer.write(*_code_for(table, (run << 4) | category))
    if category:
        writer.write(extra, category)
    writer.write(*_code_for(table, 0x00))
    parts = [
        b"\xff\xd8",
        _quant(),
        _frame(b"\xff\xc0", width, height, [(1, 0x11, 0)]),
        _dht(0x00, _DC_LUMINANCE_BITS, _DC_LUMINANCE_VALUES),
        _dht(0x10, _AC_LUMINANCE_BITS, _AC_LUMINANCE_VALUES),
        _scan([(1, 0x00)], 0, 63, 0, 0, writer.finish()),
        b"\xff\xd9",
    ]
    return b"".join(parts)


def _progressive_color(
    width: int,
    height: int,
    *,
    y_sampling: int,
    y_dc: list[int],
    cb_dc: list[int],
    cr_dc: list[int],
) -> bytes:
    writer = _BitWriter()
    table = _dc_table()
    _write_dc(writer, y_dc, table)
    _write_dc(writer, cb_dc, table)
    _write_dc(writer, cr_dc, table)
    chroma = _marker(b"\xff\xdb", bytes([1]) + bytes([8] * 64))
    parts = [
        b"\xff\xd8",
        _quant(),
        chroma,
        _frame(
            b"\xff\xc2",
            width,
            height,
            [(1, y_sampling, 0), (2, 0x11, 1), (3, 0x11, 1)],
        ),
        _dht(0x00, _DC_LUMINANCE_BITS, _DC_LUMINANCE_VALUES),
        _scan([(1, 0x00), (2, 0x00), (3, 0x00)], 0, 0, 0, 0, writer.finish()),
        b"\xff\xd9",
    ]
    return b"".join(parts)
