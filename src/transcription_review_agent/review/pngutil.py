"""Minimal PNG decode and encode for 8-bit gray, RGB, and RGBA images.

Interlaced, paletted, and 16-bit images are rejected. This is enough to cut
a crop from a sealed page without inventing pixels.
"""

from __future__ import annotations

import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOR_BPP = {0: 1, 2: 3, 6: 4}


def decode_png(data: bytes) -> tuple[int, int, list[bytearray], int]:
    """Return width, height, unpacked rows, and bytes per pixel."""

    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("page image must be a PNG")
    width = height = None
    bit_depth = color_type = interlace = None
    idat = bytearray()
    offset = 8
    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("PNG chunk header is truncated")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + length
        if end + 4 > len(data):
            raise ValueError("PNG chunk is truncated")
        chunk = data[start:end]
        if tag == b"IHDR":
            if length < 13:
                raise ValueError("PNG IHDR is truncated")
            width, height, bit_depth, color_type, _compression, _filter, interlace = (
                struct.unpack(
                    ">IIBBBBB",
                    chunk[:13],
                )
            )
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"IEND":
            break
        offset = end + 4
    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("PNG is missing an IHDR chunk")
    if width < 1 or height < 1:
        raise ValueError("PNG dimensions must be positive")
    if bit_depth != 8 or color_type not in _COLOR_BPP:
        raise ValueError("only 8-bit gray, RGB, or RGBA PNGs can be cropped")
    if interlace != 0:
        raise ValueError("interlaced PNGs cannot be cropped")
    bpp = _COLOR_BPP[color_type]
    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as exc:
        raise ValueError("PNG image data could not be decoded") from exc
    stride = width * bpp
    expected = height * (stride + 1)
    if len(raw) < expected:
        raise ValueError("PNG image data is shorter than its dimensions")
    rows: list[bytearray] = []
    previous = bytearray(stride)
    cursor = 0
    for _y in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scan = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        rows.append(_unfilter(filter_type, scan, previous, bpp))
        previous = rows[-1]
    return width, height, rows, bpp


def encode_png(width: int, height: int, rows: list[bytearray], bpp: int) -> bytes:
    """Encode filter-0 rows. ``bpp`` selects gray, RGB, or RGBA."""

    if width < 1 or height < 1 or len(rows) != height:
        raise ValueError("PNG rows do not match the requested dimensions")
    color_type = {1: 0, 3: 2, 4: 6}.get(bpp)
    if color_type is None:
        raise ValueError("unsupported PNG bytes per pixel")
    raw = bytearray()
    for row in rows:
        if len(row) != width * bpp:
            raise ValueError("PNG row width does not match")
        raw.append(0)
        raw.extend(row)
    return _signature_and_chunks(width, height, color_type, zlib.compress(bytes(raw)))


def crop_rows(
    rows: list[bytearray],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    bpp: int,
    page_width: int,
    page_height: int,
) -> list[bytearray]:
    """Copy an existing rectangle. The rectangle must lie inside the page."""

    if width < 1 or height < 1 or x < 0 or y < 0:
        raise ValueError("crop target is outside the page")
    if x + width > page_width or y + height > page_height:
        raise ValueError("crop target is outside the page")
    if len(rows) != page_height:
        raise ValueError("page rows do not match the sealed height")
    cropped: list[bytearray] = []
    for row_index in range(y, y + height):
        row = rows[row_index]
        start = x * bpp
        end = start + width * bpp
        if end > len(row):
            raise ValueError("crop target is outside the page")
        cropped.append(bytearray(row[start:end]))
    return cropped


def _unfilter(
    filter_type: int, scan: bytearray, previous: bytearray, bpp: int
) -> bytearray:
    out = bytearray(len(scan))
    for index, value in enumerate(scan):
        left = out[index - bpp] if index >= bpp else 0
        up = previous[index]
        upper_left = previous[index - bpp] if index >= bpp else 0
        if filter_type == 0:
            blended = value
        elif filter_type == 1:
            blended = value + left
        elif filter_type == 2:
            blended = value + up
        elif filter_type == 3:
            blended = value + ((left + up) // 2)
        elif filter_type == 4:
            blended = value + _paeth(left, up, upper_left)
        else:
            raise ValueError("PNG filter is not supported")
        out[index] = blended & 255
    return out


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    distances = (
        (abs(estimate - left), left),
        (abs(estimate - up), up),
        (abs(estimate - upper_left), upper_left),
    )
    return min(distances)[1]


def _signature_and_chunks(
    width: int, height: int, color_type: int, compressed: bytes
) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def _chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
