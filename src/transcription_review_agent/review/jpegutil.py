"""JPEG dimensions and pixels for sealed page images.

8-bit baseline (SOF0) grayscale or 4:4:4, and 8-bit progressive (SOF2)
grayscale or YCbCr with sampling factors of 1 or 2, are accepted. The derived
PNG is produced from those pixels. Sealed JPEG bytes are never rewritten.
"""

from __future__ import annotations

import math
import struct
from typing import TypedDict

from transcription_review_agent.review.pngutil import encode_png


class _ProgressiveComponent(TypedDict):
    id: int
    h: int
    v: int
    table: int
    sample_width: int
    sample_height: int
    blocks_x: int
    blocks_y: int
    blocks: list[list[int]]


MAX_JPEG_EDGE = 16000
MAX_JPEG_PIXELS = 24_000_000

_ZIGZAG = (
    0,
    1,
    8,
    16,
    9,
    2,
    3,
    10,
    17,
    24,
    32,
    25,
    18,
    11,
    4,
    5,
    12,
    19,
    26,
    33,
    40,
    48,
    41,
    34,
    27,
    20,
    13,
    6,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    57,
    50,
    43,
    36,
    29,
    22,
    15,
    23,
    30,
    37,
    44,
    51,
    58,
    59,
    52,
    45,
    38,
    31,
    39,
    46,
    53,
    60,
    61,
    54,
    47,
    55,
    62,
    63,
)

# ITU T.81 Annex K.3 luminance tables.
_DC_LUMINANCE_BITS = (0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
_DC_LUMINANCE_VALUES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_AC_LUMINANCE_BITS = (0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D)
_AC_LUMINANCE_VALUES = (
    0x01,
    0x02,
    0x03,
    0x00,
    0x04,
    0x11,
    0x05,
    0x12,
    0x21,
    0x31,
    0x41,
    0x06,
    0x13,
    0x51,
    0x61,
    0x07,
    0x22,
    0x71,
    0x14,
    0x32,
    0x81,
    0x91,
    0xA1,
    0x08,
    0x23,
    0x42,
    0xB1,
    0xC1,
    0x15,
    0x52,
    0xD1,
    0xF0,
    0x24,
    0x33,
    0x62,
    0x72,
    0x82,
    0x09,
    0x0A,
    0x16,
    0x17,
    0x18,
    0x19,
    0x1A,
    0x25,
    0x26,
    0x27,
    0x28,
    0x29,
    0x2A,
    0x34,
    0x35,
    0x36,
    0x37,
    0x38,
    0x39,
    0x3A,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,
    0x49,
    0x4A,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,
    0x59,
    0x5A,
    0x63,
    0x64,
    0x65,
    0x66,
    0x67,
    0x68,
    0x69,
    0x6A,
    0x73,
    0x74,
    0x75,
    0x76,
    0x77,
    0x78,
    0x79,
    0x7A,
    0x83,
    0x84,
    0x85,
    0x86,
    0x87,
    0x88,
    0x89,
    0x8A,
    0x92,
    0x93,
    0x94,
    0x95,
    0x96,
    0x97,
    0x98,
    0x99,
    0x9A,
    0xA2,
    0xA3,
    0xA4,
    0xA5,
    0xA6,
    0xA7,
    0xA8,
    0xA9,
    0xAA,
    0xB2,
    0xB3,
    0xB4,
    0xB5,
    0xB6,
    0xB7,
    0xB8,
    0xB9,
    0xBA,
    0xC2,
    0xC3,
    0xC4,
    0xC5,
    0xC6,
    0xC7,
    0xC8,
    0xC9,
    0xCA,
    0xD2,
    0xD3,
    0xD4,
    0xD5,
    0xD6,
    0xD7,
    0xD8,
    0xD9,
    0xDA,
    0xE1,
    0xE2,
    0xE3,
    0xE4,
    0xE5,
    0xE6,
    0xE7,
    0xE8,
    0xE9,
    0xEA,
    0xF1,
    0xF2,
    0xF3,
    0xF4,
    0xF5,
    0xF6,
    0xF7,
    0xF8,
    0xF9,
    0xFA,
)
_DC_CHROMA_BITS = (0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
_DC_CHROMA_VALUES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
_AC_CHROMA_BITS = (0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77)
_AC_CHROMA_VALUES = (
    0x00,
    0x01,
    0x02,
    0x03,
    0x11,
    0x04,
    0x05,
    0x21,
    0x31,
    0x06,
    0x12,
    0x41,
    0x51,
    0x07,
    0x61,
    0x71,
    0x13,
    0x22,
    0x32,
    0x81,
    0x08,
    0x14,
    0x42,
    0x91,
    0xA1,
    0xB1,
    0xC1,
    0x09,
    0x23,
    0x33,
    0x52,
    0xF0,
    0x15,
    0x62,
    0x72,
    0xD1,
    0x0A,
    0x16,
    0x24,
    0x34,
    0xE1,
    0x25,
    0xF1,
    0x17,
    0x18,
    0x19,
    0x1A,
    0x26,
    0x27,
    0x28,
    0x29,
    0x2A,
    0x35,
    0x36,
    0x37,
    0x38,
    0x39,
    0x3A,
    0x43,
    0x44,
    0x45,
    0x46,
    0x47,
    0x48,
    0x49,
    0x4A,
    0x53,
    0x54,
    0x55,
    0x56,
    0x57,
    0x58,
    0x59,
    0x5A,
    0x63,
    0x64,
    0x65,
    0x66,
    0x67,
    0x68,
    0x69,
    0x6A,
    0x73,
    0x74,
    0x75,
    0x76,
    0x77,
    0x78,
    0x79,
    0x7A,
    0x82,
    0x83,
    0x84,
    0x85,
    0x86,
    0x87,
    0x88,
    0x89,
    0x8A,
    0x92,
    0x93,
    0x94,
    0x95,
    0x96,
    0x97,
    0x98,
    0x99,
    0x9A,
    0xA2,
    0xA3,
    0xA4,
    0xA5,
    0xA6,
    0xA7,
    0xA8,
    0xA9,
    0xAA,
    0xB2,
    0xB3,
    0xB4,
    0xB5,
    0xB6,
    0xB7,
    0xB8,
    0xB9,
    0xBA,
    0xC2,
    0xC3,
    0xC4,
    0xC5,
    0xC6,
    0xC7,
    0xC8,
    0xC9,
    0xCA,
    0xD2,
    0xD3,
    0xD4,
    0xD5,
    0xD6,
    0xD7,
    0xD8,
    0xD9,
    0xDA,
    0xE2,
    0xE3,
    0xE4,
    0xE5,
    0xE6,
    0xE7,
    0xE8,
    0xE9,
    0xEA,
    0xF2,
    0xF3,
    0xF4,
    0xF5,
    0xF6,
    0xF7,
    0xF8,
    0xF9,
    0xFA,
)


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Read width and height from a baseline or progressive frame marker."""

    width, height, _components, _marker = _sof0(data)
    return width, height


def decode_jpeg(data: bytes) -> tuple[int, int, list[bytearray], int]:
    """Return width, height, rows, and bytes per pixel for a supported JPEG."""

    if not data.startswith(b"\xff\xd8"):
        raise ValueError("page image must be a JPEG")
    width, height, frame, marker = _sof0(data)
    if marker == 0xC2:
        rows = _decode_progressive(data, width, height, frame)
    else:
        huffman, quant, scan, selected = _scan(data, frame)
        rows = _decode_scan(width, height, frame, huffman, quant, scan, selected)
    bpp = 1 if len(frame) == 1 else 3
    return width, height, rows, bpp


def jpeg_to_png(data: bytes) -> bytes:
    """Derive a PNG from JPEG pixels. The same bytes always yield the same PNG."""

    width, height, rows, bpp = decode_jpeg(data)
    return encode_png(width, height, rows, bpp)


def _decode_progressive(
    data: bytes,
    width: int,
    height: int,
    frame: list[tuple[int, int, int, int]],
) -> list[bytearray]:
    """Decode an 8-bit progressive frame into grayscale or RGB rows.

    Original bytes are only read. Restart intervals and sampling factors
    outside 1 or 2 are rejected.
    """

    hmax = max(item[1] >> 4 for item in frame)
    vmax = max(item[1] & 0x0F for item in frame)
    components: list[_ProgressiveComponent] = []
    for component_id, sampling, table, _index in frame:
        horizontal, vertical = sampling >> 4, sampling & 0x0F
        if hmax % horizontal or vmax % vertical:
            raise ValueError("JPEG sampling factors are inconsistent")
        sample_width = (width * horizontal + hmax - 1) // hmax
        sample_height = (height * vertical + vmax - 1) // vmax
        blocks_x = (sample_width + 7) // 8
        blocks_y = (sample_height + 7) // 8
        component: _ProgressiveComponent = {
            "id": component_id,
            "h": horizontal,
            "v": vertical,
            "table": table,
            "sample_width": sample_width,
            "sample_height": sample_height,
            "blocks_x": blocks_x,
            "blocks_y": blocks_y,
            "blocks": [[0] * 64 for _ in range(blocks_x * blocks_y)],
        }
        components.append(component)
    by_id = {int(item["id"]): item for item in components}
    mcus_x = (width + 8 * hmax - 1) // (8 * hmax)
    mcus_y = (height + 8 * vmax - 1) // (8 * vmax)
    tables: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    quant: dict[int, list[int]] = {}
    predictors = {int(item["id"]): 0 for item in components}
    saw_scan = False
    for marker, payload, entropy in _progressive_segments(data):
        if marker == 0xDB:
            _read_quant(payload, quant)
        elif marker == 0xC4:
            _read_huffman(payload, tables)
        elif marker == 0xDD:
            raise ValueError("JPEG restart intervals are not supported")
        elif marker == 0xDA:
            if entropy is None:
                raise ValueError("JPEG scan is truncated")
            _decode_progressive_scan(
                payload,
                entropy,
                by_id,
                tables,
                predictors,
                mcus_x,
                mcus_y,
            )
            saw_scan = True
    if not saw_scan:
        raise ValueError("JPEG is missing a scan")
    planes = {int(item["id"]): _render_component(item, quant) for item in components}
    if len(frame) == 1:
        return planes[frame[0][0]]
    return _upsample_rgb(planes, components, width, height, hmax, vmax)


def _progressive_segments(data: bytes) -> list[tuple[int, bytes, bytes | None]]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("page image must be a JPEG")
    found: list[tuple[int, bytes, bytes | None]] = []
    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            raise ValueError("JPEG marker is misaligned")
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            raise ValueError("JPEG marker is truncated")
        marker = data[index]
        index += 1
        if marker == 0xD9:
            break
        if marker in {0x01} or 0xD0 <= marker <= 0xD7:
            raise ValueError("JPEG restart intervals are not supported")
        payload = _segment_at(data, index)
        index += 2 + len(payload)
        if marker != 0xDA:
            found.append((marker, payload, None))
            continue
        entropy, index = _consume_scan(data, index)
        found.append((marker, payload, entropy))
    return found


def _consume_scan(data: bytes, start: int) -> tuple[bytes, int]:
    scan = bytearray()
    index = start
    while index < len(data):
        byte = data[index]
        if byte != 0xFF:
            scan.append(byte)
            index += 1
            continue
        if index + 1 >= len(data):
            raise ValueError("JPEG scan is truncated")
        nxt = data[index + 1]
        if nxt == 0x00:
            scan.append(0xFF)
            scan.append(0x00)
            index += 2
            continue
        return bytes(scan), index
    raise ValueError("JPEG scan is missing a marker")


def _decode_progressive_scan(
    payload: bytes,
    entropy: bytes,
    by_id: dict[int, _ProgressiveComponent],
    tables: dict[tuple[int, int], dict[tuple[int, int], int]],
    predictors: dict[int, int],
    mcus_x: int,
    mcus_y: int,
) -> None:
    if len(payload) < 4:
        raise ValueError("JPEG scan header is truncated")
    count = payload[0]
    if count < 1 or len(payload) < 1 + 2 * count + 3:
        raise ValueError("JPEG scan header is truncated")
    selected: list[tuple[int, int]] = []
    for index in range(count):
        component_id = payload[1 + index * 2]
        table_ids = payload[2 + index * 2]
        if component_id not in by_id:
            raise ValueError("JPEG scan names an unknown component")
        selected.append((component_id, table_ids))
    spectral_start = payload[1 + 2 * count]
    spectral_end = payload[2 + 2 * count]
    approximation = payload[3 + 2 * count]
    successive_high = approximation >> 4
    successive_low = approximation & 0x0F
    if (
        spectral_start > spectral_end
        or spectral_end > 63
        or successive_high > 13
        or successive_low > 13
    ):
        raise ValueError("JPEG progressive scan parameters are invalid")
    if spectral_start == 0 and spectral_end != 0:
        raise ValueError("JPEG progressive scan mixes DC and AC coefficients")
    if count > 1 and spectral_start != 0:
        raise ValueError("JPEG progressive AC scans must contain one component")
    bits = _Bits(entropy)
    eobrun = [0]
    if count > 1:
        if spectral_start != 0:
            raise ValueError("JPEG progressive AC scans must contain one component")
        _decode_interleaved_dc(
            bits,
            selected,
            by_id,
            tables,
            predictors,
            mcus_x,
            mcus_y,
            successive_high,
            successive_low,
        )
        return
    component_id, table_ids = selected[0]
    component = by_id[component_id]
    table_key = (0, table_ids >> 4) if spectral_start == 0 else (1, table_ids & 0x0F)
    try:
        huffman = tables[table_key]
    except KeyError as exc:
        raise ValueError("JPEG scan is missing a Huffman table") from exc
    blocks_x = int(component["blocks_x"])
    blocks_y = int(component["blocks_y"])
    stored = component["blocks"]
    assert isinstance(stored, list)
    for block_y in range(blocks_y):
        for block_x in range(blocks_x):
            block = stored[block_y * blocks_x + block_x]
            if spectral_start == 0:
                if successive_high == 0:
                    _decode_dc_initial(
                        bits, huffman, block, predictors, component_id, successive_low
                    )
                else:
                    _decode_dc_refine(bits, block, successive_low)
            elif successive_high == 0:
                _decode_ac_initial(
                    bits,
                    huffman,
                    block,
                    spectral_start,
                    spectral_end,
                    successive_low,
                    eobrun,
                )
            else:
                _decode_ac_refine(
                    bits,
                    huffman,
                    block,
                    spectral_start,
                    spectral_end,
                    successive_low,
                    eobrun,
                )


def _decode_interleaved_dc(
    bits: _Bits,
    selected: list[tuple[int, int]],
    by_id: dict[int, _ProgressiveComponent],
    tables: dict[tuple[int, int], dict[tuple[int, int], int]],
    predictors: dict[int, int],
    mcus_x: int,
    mcus_y: int,
    successive_high: int,
    successive_low: int,
) -> None:
    for _mcu_y in range(mcus_y):
        for _mcu_x in range(mcus_x):
            for component_id, table_ids in selected:
                component = by_id[component_id]
                try:
                    dc_table = tables[(0, table_ids >> 4)]
                except KeyError as exc:
                    raise ValueError("JPEG scan is missing a Huffman table") from exc
                horizontal = int(component["h"])
                vertical = int(component["v"])
                blocks_x = int(component["blocks_x"])
                blocks_y = int(component["blocks_y"])
                stored = component["blocks"]
                assert isinstance(stored, list)
                for block_y_offset in range(vertical):
                    for block_x_offset in range(horizontal):
                        block_x = _mcu_x * horizontal + block_x_offset
                        block_y = _mcu_y * vertical + block_y_offset
                        if block_x < blocks_x and block_y < blocks_y:
                            block = stored[block_y * blocks_x + block_x]
                        else:
                            block = [0] * 64
                        if successive_high == 0:
                            _decode_dc_initial(
                                bits,
                                dc_table,
                                block,
                                predictors,
                                component_id,
                                successive_low,
                            )
                        else:
                            _decode_dc_refine(bits, block, successive_low)


def _decode_dc_initial(
    bits: _Bits,
    table: dict[tuple[int, int], int],
    block: list[int],
    predictors: dict[int, int],
    component_id: int,
    successive_low: int,
) -> None:
    category = _symbol(bits, table)
    diff = _extend(bits.read(category) if category else 0, category) << successive_low
    predictors[component_id] += diff
    block[0] = predictors[component_id]


def _decode_dc_refine(bits: _Bits, block: list[int], successive_low: int) -> None:
    if bits.read(1):
        block[0] |= 1 << successive_low


def _decode_ac_initial(
    bits: _Bits,
    table: dict[tuple[int, int], int],
    block: list[int],
    spectral_start: int,
    spectral_end: int,
    successive_low: int,
    eobrun: list[int],
) -> None:
    if eobrun[0] > 0:
        eobrun[0] -= 1
        return
    index = spectral_start
    while index <= spectral_end:
        symbol = _symbol(bits, table)
        run = symbol >> 4
        size = symbol & 0x0F
        if size == 0:
            if run == 15:
                index += 16
                continue
            length = 1 << run
            if run:
                length += bits.read(run)
            eobrun[0] = length - 1
            return
        index += run
        if index > spectral_end:
            raise ValueError("JPEG AC coefficient is outside the block")
        block[index] = _extend(bits.read(size), size) << successive_low
        index += 1


def _decode_ac_refine(
    bits: _Bits,
    table: dict[tuple[int, int], int],
    block: list[int],
    spectral_start: int,
    spectral_end: int,
    successive_low: int,
    eobrun: list[int],
) -> None:
    point = 1 << successive_low
    negative = -point
    index = spectral_start
    if eobrun[0] == 0:
        while index <= spectral_end:
            symbol = _symbol(bits, table)
            run = symbol >> 4
            size = symbol & 0x0F
            if size == 0 and run < 15:
                length = 1 << run
                if run:
                    length += bits.read(run)
                while index <= spectral_end:
                    if block[index]:
                        _apply_ac_refinement(bits, block, index, point, negative)
                    index += 1
                eobrun[0] = length - 1
                return
            if size == 0:
                skipped = 0
                while skipped < 16:
                    if index > spectral_end:
                        raise ValueError("JPEG AC coefficient is outside the block")
                    if block[index]:
                        _apply_ac_refinement(bits, block, index, point, negative)
                    else:
                        skipped += 1
                    index += 1
                continue
            if size != 1:
                raise ValueError("JPEG AC refinement size is invalid")
            new_value = point if bits.read(1) else negative
            skipped = 0
            while True:
                if index > spectral_end:
                    raise ValueError("JPEG AC coefficient is outside the block")
                if block[index]:
                    _apply_ac_refinement(bits, block, index, point, negative)
                    index += 1
                    continue
                if skipped == run:
                    block[index] = new_value
                    index += 1
                    break
                skipped += 1
                index += 1
        return
    while index <= spectral_end:
        if block[index]:
            _apply_ac_refinement(bits, block, index, point, negative)
        index += 1
    eobrun[0] -= 1


def _apply_ac_refinement(
    bits: _Bits,
    block: list[int],
    index: int,
    point: int,
    negative: int,
) -> None:
    if bits.read(1) and (block[index] & point) == 0:
        block[index] += point if block[index] > 0 else negative


def _render_component(
    component: _ProgressiveComponent, quant: dict[int, list[int]]
) -> list[bytearray]:
    table_id = int(component["table"])
    if table_id not in quant:
        raise ValueError("JPEG component is missing a quantization table")
    table = quant[table_id]
    sample_width = int(component["sample_width"])
    sample_height = int(component["sample_height"])
    blocks_x = int(component["blocks_x"])
    stored = component["blocks"]
    assert isinstance(stored, list)
    plane = [bytearray(sample_width) for _ in range(sample_height)]
    for block_y in range(int(component["blocks_y"])):
        for block_x in range(blocks_x):
            spatial = _idct(_dequantize(stored[block_y * blocks_x + block_x], table))
            for y in range(8):
                target_y = block_y * 8 + y
                if target_y >= sample_height:
                    continue
                row = plane[target_y]
                for x in range(8):
                    target_x = block_x * 8 + x
                    if target_x >= sample_width:
                        continue
                    row[target_x] = spatial[y][x]
    return plane


def _upsample_rgb(
    planes: dict[int, list[bytearray]],
    components: list[_ProgressiveComponent],
    width: int,
    height: int,
    hmax: int,
    vmax: int,
) -> list[bytearray]:
    ordered = []
    for component in components:
        scale_x = hmax // int(component["h"])
        scale_y = vmax // int(component["v"])
        ordered.append(
            (
                planes[int(component["id"])],
                scale_x,
                scale_y,
                int(component["sample_width"]),
                int(component["sample_height"]),
            )
        )
    y_plane, cb_plane, cr_plane = ordered
    rows: list[bytearray] = []
    for y in range(height):
        row = bytearray()
        yy_row = _scaled_row(y_plane, y)
        cb_row = _scaled_row(cb_plane, y)
        cr_row = _scaled_row(cr_plane, y)
        for x in range(width):
            yy = yy_row[x // y_plane[1]]
            cb = cb_row[x // cb_plane[1]] - 128
            cr = cr_row[x // cr_plane[1]] - 128
            red = yy + 1.402 * cr
            green = yy - 0.344136 * cb - 0.714136 * cr
            blue = yy + 1.772 * cb
            for channel in (red, green, blue):
                snapped = round(channel, 6)
                if snapped <= 0:
                    row.append(0)
                elif snapped >= 255:
                    row.append(255)
                else:
                    row.append(int(math.floor(snapped + 0.5)))
        rows.append(row)
    return rows


def _scaled_row(
    plane: tuple[list[bytearray], int, int, int, int],
    y: int,
) -> bytearray:
    samples, _scale_x, scale_y, sample_width, sample_height = plane
    source_y = y // scale_y
    if source_y >= sample_height:
        raise ValueError("JPEG sampling does not cover the frame")
    row = samples[source_y]
    if len(row) != sample_width:
        raise ValueError("JPEG sampling does not cover the frame")
    return row


def encode_grayscale_dc_jpeg(width: int, height: int, dc_values: list[int]) -> bytes:
    """Write a baseline grayscale JPEG whose only nonzero coefficient is DC.

    ``dc_values`` are quantized DC coefficients, one per 8x8 block in row-major
    order. This is a fixture writer. Intake decodes; it does not call this.
    """

    if width < 8 or height < 8 or width % 8 or height % 8:
        raise ValueError("DC JPEG dimensions must be positive multiples of 8")
    blocks = (width // 8) * (height // 8)
    if len(dc_values) != blocks:
        raise ValueError("DC JPEG needs one coefficient per block")
    quant = bytes([8] * 64)
    parts = [
        b"\xff\xd8",
        _marker(b"\xff\xdb", b"\x00" + quant),
        _marker(
            b"\xff\xc0",
            struct.pack(">BHHB", 8, height, width, 1) + bytes((1, 0x11, 0)),
        ),
        _dht(0x00, _DC_LUMINANCE_BITS, _DC_LUMINANCE_VALUES),
        _dht(0x10, _AC_LUMINANCE_BITS, _AC_LUMINANCE_VALUES),
        _marker(b"\xff\xda", bytes((1, 1, 0x00, 0, 63, 0))),
        _dc_scan(
            dc_values,
            _huffman(_DC_LUMINANCE_BITS, _DC_LUMINANCE_VALUES),
            _huffman(_AC_LUMINANCE_BITS, _AC_LUMINANCE_VALUES),
        ),
        b"\xff\xd9",
    ]
    return b"".join(parts)


def _sof0(data: bytes) -> tuple[int, int, list[tuple[int, int, int, int]], int]:
    found = None
    for marker, payload in _markers(data):
        if marker in {0xC1, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            raise ValueError("JPEG frame type is not supported")
        if marker not in {0xC0, 0xC2}:
            continue
        if found is not None:
            raise ValueError("JPEG contains more than one frame")
        if len(payload) < 8:
            raise ValueError("JPEG frame header is truncated")
        precision = payload[0]
        height, width, count = struct.unpack(">HHB", payload[1:6])
        if precision != 8:
            raise ValueError("JPEG precision must be 8-bit")
        _reject_huge(width, height)
        if count not in {1, 3} or len(payload) < 6 + 3 * count:
            raise ValueError("JPEG components must be grayscale or RGB")
        components = []
        for index in range(count):
            start = 6 + index * 3
            component_id, sampling, table = payload[start : start + 3]
            horizontal, vertical = sampling >> 4, sampling & 0x0F
            if marker == 0xC0:
                if sampling != 0x11:
                    raise ValueError("JPEG sampling other than 4:4:4 is not supported")
            elif horizontal not in {1, 2} or vertical not in {1, 2}:
                raise ValueError("JPEG sampling factors must be 1 or 2")
            components.append((component_id, sampling, table, index))
        found = (width, height, components, marker)
    if found is None:
        raise ValueError("JPEG is missing a frame")
    return found


def _scan(
    data: bytes,
    frame: list[tuple[int, int, int, int]],
) -> tuple[
    dict[tuple[int, int], dict[tuple[int, int], int]],
    dict[int, list[int]],
    bytes,
    list[tuple[int, int]],
]:
    tables: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
    quant: dict[int, list[int]] = {}
    scan = None
    selected: list[tuple[int, int]] = []
    for marker, payload in _markers(data):
        if marker == 0xDB:
            _read_quant(payload, quant)
        elif marker == 0xC4:
            _read_huffman(payload, tables)
        elif marker == 0xDD:
            raise ValueError("JPEG restart intervals are not supported")
        elif marker == 0xDA:
            if len(payload) < 4:
                raise ValueError("JPEG scan header is truncated")
            count = payload[0]
            if count != len(frame) or len(payload) < 1 + 2 * count + 3:
                raise ValueError("JPEG scan components do not match the frame")
            frame_ids = {item[0] for item in frame}
            for index in range(count):
                component_id = payload[1 + index * 2]
                table_ids = payload[2 + index * 2]
                if component_id not in frame_ids:
                    raise ValueError("JPEG scan names an unknown component")
                selected.append((component_id, table_ids))
            spectral = payload[1 + 2 * count : 4 + 2 * count]
            if tuple(spectral) != (0, 63, 0):
                raise ValueError("progressive JPEG scans are not supported")
            scan = _entropy(data, _marker_end(data, marker, payload))
            break
    if scan is None:
        raise ValueError("JPEG is missing a scan")
    for component in frame:
        if component[2] not in quant:
            raise ValueError("JPEG component is missing a quantization table")
    return tables, quant, scan, selected


def _decode_scan(
    width: int,
    height: int,
    frame: list[tuple[int, int, int, int]],
    tables: dict[tuple[int, int], dict[tuple[int, int], int]],
    quant: dict[int, list[int]],
    scan: bytes,
    selected: list[tuple[int, int]],
) -> list[bytearray]:
    bits = _Bits(scan)
    predictors = {component[0]: 0 for component in frame}
    by_id = {component[0]: component for component in frame}
    mcus_x = (width + 7) // 8
    mcus_y = (height + 7) // 8
    planes = {
        component[0]: [bytearray(width) for _ in range(height)] for component in frame
    }
    for mcu_y in range(mcus_y):
        for mcu_x in range(mcus_x):
            decoded: dict[int, list[list[int]]] = {}
            for component_id, table_ids in selected:
                component = by_id[component_id]
                try:
                    dc_table = tables[(0, table_ids >> 4)]
                    ac_table = tables[(1, table_ids & 0x0F)]
                except KeyError as exc:
                    raise ValueError("JPEG scan is missing a Huffman table") from exc
                block = _decode_block(
                    bits, dc_table, ac_table, predictors, component_id
                )
                decoded[component_id] = _idct(_dequantize(block, quant[component[2]]))
            _paint(planes, decoded, mcu_x, mcu_y, width, height)
    if len(frame) == 1:
        return planes[frame[0][0]]
    return _rgb(planes, frame, width, height)


def _markers(data: bytes) -> list[tuple[int, bytes]]:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("page image must be a JPEG")
    found: list[tuple[int, bytes]] = []
    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            raise ValueError("JPEG marker is misaligned")
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            raise ValueError("JPEG marker is truncated")
        marker = data[index]
        index += 1
        if marker == 0xD9:
            break
        if marker == 0xDA:
            found.append((marker, _segment_at(data, index)))
            break
        if marker in {0x01} or 0xD0 <= marker <= 0xD7:
            continue
        payload = _segment_at(data, index)
        found.append((marker, payload))
        index += 2 + len(payload)
    return found


def _segment_at(data: bytes, index: int) -> bytes:
    if index + 2 > len(data):
        raise ValueError("JPEG segment is truncated")
    length = struct.unpack(">H", data[index : index + 2])[0]
    if length < 2 or index + length > len(data):
        raise ValueError("JPEG segment is truncated")
    return data[index + 2 : index + length]


def _marker_end(data: bytes, marker: int, payload: bytes) -> int:
    del marker
    # The scan header length includes the two length bytes, which were excluded
    # from payload. Locate the header by its payload.
    needle = struct.pack(">H", len(payload) + 2) + payload
    start = data.find(needle)
    if start < 0:
        raise ValueError("JPEG scan header was not found")
    return start + len(needle)


def _entropy(data: bytes, start: int) -> bytes:
    scan = bytearray()
    index = start
    while index < len(data):
        byte = data[index]
        if byte != 0xFF:
            scan.append(byte)
            index += 1
            continue
        if index + 1 >= len(data):
            raise ValueError("JPEG scan is truncated")
        nxt = data[index + 1]
        if nxt == 0x00:
            scan.append(0xFF)
            scan.append(0x00)
            index += 2
            continue
        if nxt == 0xD9:
            return bytes(scan)
        raise ValueError("unsupported marker inside a JPEG scan")
    raise ValueError("JPEG scan is missing EOI")


def _read_quant(payload: bytes, quant: dict[int, list[int]]) -> None:
    index = 0
    while index < len(payload):
        info = payload[index]
        index += 1
        precision = info >> 4
        table_id = info & 0x0F
        size = 128 if precision else 64
        if precision != 0 or index + size > len(payload):
            raise ValueError("JPEG quantization table is not 8-bit")
        quant[table_id] = list(payload[index : index + size])
        index += size


def _read_huffman(
    payload: bytes, tables: dict[tuple[int, int], dict[tuple[int, int], int]]
) -> None:
    index = 0
    while index < len(payload):
        info = payload[index]
        index += 1
        klass = info >> 4
        table_id = info & 0x0F
        if index + 16 > len(payload):
            raise ValueError("JPEG Huffman table is truncated")
        counts = tuple(payload[index : index + 16])
        index += 16
        total = sum(counts)
        if index + total > len(payload):
            raise ValueError("JPEG Huffman table is truncated")
        symbols = tuple(payload[index : index + total])
        index += total
        tables[(klass, table_id)] = _huffman(counts, symbols)


def _huffman(
    counts: tuple[int, ...], symbols: tuple[int, ...]
) -> dict[tuple[int, int], int]:
    table: dict[tuple[int, int], int] = {}
    code = 0
    cursor = 0
    for length, count in enumerate(counts, start=1):
        for _ in range(count):
            if cursor >= len(symbols):
                raise ValueError("JPEG Huffman table is truncated")
            table[(code, length)] = symbols[cursor]
            cursor += 1
            code += 1
        code <<= 1
    return table


def _reject_huge(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ValueError("JPEG dimensions must be positive")
    if (
        width > MAX_JPEG_EDGE
        or height > MAX_JPEG_EDGE
        or width * height > MAX_JPEG_PIXELS
    ):
        raise ValueError("JPEG dimensions exceed the review limit")


def _marker(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">H", len(payload) + 2) + payload


def _dht(info: int, counts: tuple[int, ...], symbols: tuple[int, ...]) -> bytes:
    return _marker(b"\xff\xc4", bytes((info, *counts, *symbols)))


def _dc_scan(
    dc_values: list[int],
    dc_table: dict[tuple[int, int], int],
    ac_table: dict[tuple[int, int], int],
) -> bytes:
    writer = _BitWriter()
    predictor = 0
    eob = _code_for(ac_table, 0x00)
    for value in dc_values:
        diff = value - predictor
        predictor = value
        category, extra = _category(diff)
        writer.write(*_code_for(dc_table, category))
        if category:
            writer.write(extra, category)
        writer.write(*eob)
    return writer.finish()


def _code_for(table: dict[tuple[int, int], int], symbol: int) -> tuple[int, int]:
    for (code, length), found in table.items():
        if found == symbol:
            return code, length
    raise ValueError("JPEG Huffman table has no symbol")


def _category(diff: int) -> tuple[int, int]:
    if diff == 0:
        return 0, 0
    magnitude = abs(diff)
    category = magnitude.bit_length()
    extra = diff if diff > 0 else diff + (1 << category) - 1
    return category, extra


class _BitWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.acc = 0
        self.bits = 0

    def write(self, value: int, length: int) -> None:
        for shift in range(length - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> shift) & 1)
            self.bits += 1
            if self.bits == 8:
                self.buffer.append(self.acc)
                if self.acc == 0xFF:
                    self.buffer.append(0x00)
                self.acc = 0
                self.bits = 0

    def finish(self) -> bytes:
        if self.bits:
            pad = 8 - self.bits
            self.acc = (self.acc << pad) | ((1 << pad) - 1)
            self.buffer.append(self.acc)
            if self.acc == 0xFF:
                self.buffer.append(0x00)
        return bytes(self.buffer)


class _Bits:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.index = 0
        self.acc = 0
        self.left = 0

    def read(self, count: int) -> int:
        value = 0
        for _ in range(count):
            if self.left == 0:
                if self.index >= len(self.data):
                    raise ValueError("JPEG scan is truncated")
                byte = self.data[self.index]
                self.index += 1
                if byte == 0xFF:
                    if self.index >= len(self.data) or self.data[self.index] != 0x00:
                        raise ValueError("JPEG scan byte stuffing is invalid")
                    self.index += 1
                self.acc = byte
                self.left = 8
            self.left -= 1
            value = (value << 1) | ((self.acc >> self.left) & 1)
        return value


def _symbol(bits: _Bits, table: dict[tuple[int, int], int]) -> int:
    code = 0
    for length in range(1, 17):
        code = (code << 1) | bits.read(1)
        found = table.get((code, length))
        if found is not None:
            return found
    raise ValueError("JPEG Huffman code is not in the table")


def _extend(value: int, category: int) -> int:
    if category == 0:
        return 0
    if value < (1 << (category - 1)):
        return value - (1 << category) + 1
    return value


def _decode_block(
    bits: _Bits,
    dc_table: dict[tuple[int, int], int],
    ac_table: dict[tuple[int, int], int],
    predictors: dict[int, int],
    component_id: int,
) -> list[int]:
    block = [0] * 64
    category = _symbol(bits, dc_table)
    diff = _extend(bits.read(category) if category else 0, category)
    predictors[component_id] += diff
    block[0] = predictors[component_id]
    index = 1
    while index < 64:
        symbol = _symbol(bits, ac_table)
        if symbol == 0x00:
            break
        if symbol == 0xF0:
            index += 16
            continue
        index += symbol >> 4
        size = symbol & 0x0F
        if index >= 64 or size == 0:
            raise ValueError("JPEG AC coefficient is outside the block")
        block[index] = _extend(bits.read(size), size)
        index += 1
    return block


def _dequantize(block: list[int], table: list[int]) -> list[list[int]]:
    natural = [[0] * 8 for _ in range(8)]
    for index, slot in enumerate(_ZIGZAG):
        natural[slot // 8][slot % 8] = block[index] * table[index]
    return natural


_IDCT_SCALE = tuple((1 / math.sqrt(2)) if freq == 0 else 1.0 for freq in range(8))
_IDCT_COS = tuple(
    tuple(math.cos(((2 * sample) + 1) * freq * math.pi / 16) for freq in range(8))
    for sample in range(8)
)
# Separable weights. W[sample][freq] = scale(freq) * cos((2*sample+1)*freq*pi/16).
_IDCT_W = tuple(
    tuple(_IDCT_SCALE[freq] * _IDCT_COS[sample][freq] for freq in range(8))
    for sample in range(8)
)


def _project_row(src: list[int]) -> list[float] | None:
    s0, s1, s2, s3, s4, s5, s6, s7 = src
    if not (s0 or s1 or s2 or s3 or s4 or s5 or s6 or s7):
        return None
    projected: list[float] = []
    for weights in _IDCT_W:
        projected.append(
            weights[0] * s0
            + weights[1] * s1
            + weights[2] * s2
            + weights[3] * s3
            + weights[4] * s4
            + weights[5] * s5
            + weights[6] * s6
            + weights[7] * s7
        )
    return projected


def _clamp_sample(value: float) -> int:
    snapped = round(value, 6)
    if snapped <= 0:
        return 0
    if snapped >= 255:
        return 255
    return int(math.floor(snapped + 0.5))


def _idct(block: list[list[int]]) -> list[list[int]]:
    """Inverse DCT used for both baseline and progressive pixels.

    This is the separable form of the JPEG cosine transform. A DC-only block
    still lands on the exact sample the baseline fixtures require.
    """

    if not any(block[0][1:]) and all(not any(row) for row in block[1:]):
        return [[_clamp_sample(block[0][0] / 8.0 + 128.0)] * 8 for _ in range(8)]
    horizontal = [_project_row(row) for row in block]
    zero = (0.0,) * 8
    r0, r1, r2, r3, r4, r5, r6, r7 = (
        row if row is not None else zero for row in horizontal
    )
    spatial: list[list[int]] = []
    for weights in _IDCT_W:
        a, b, c, d, e, f, g, h = weights
        spatial.append(
            [
                _clamp_sample(
                    (
                        a * r0[x]
                        + b * r1[x]
                        + c * r2[x]
                        + d * r3[x]
                        + e * r4[x]
                        + f * r5[x]
                        + g * r6[x]
                        + h * r7[x]
                    )
                    / 4.0
                    + 128.0
                )
                for x in range(8)
            ]
        )
    return spatial


def _paint(
    planes: dict[int, list[bytearray]],
    decoded: dict[int, list[list[int]]],
    mcu_x: int,
    mcu_y: int,
    width: int,
    height: int,
) -> None:
    for component_id, block in decoded.items():
        plane = planes[component_id]
        for y in range(8):
            target_y = mcu_y * 8 + y
            if target_y >= height:
                continue
            for x in range(8):
                target_x = mcu_x * 8 + x
                if target_x >= width:
                    continue
                plane[target_y][target_x] = block[y][x]


def _rgb(
    planes: dict[int, list[bytearray]],
    frame: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> list[bytearray]:
    y_id, cb_id, cr_id = (item[0] for item in frame)
    rows: list[bytearray] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            yy = planes[y_id][y][x]
            cb = planes[cb_id][y][x] - 128
            cr = planes[cr_id][y][x] - 128
            red = yy + 1.402 * cr
            green = yy - 0.344136 * cb - 0.714136 * cr
            blue = yy + 1.772 * cb
            for channel in (red, green, blue):
                snapped = round(channel, 6)
                if snapped <= 0:
                    row.append(0)
                elif snapped >= 255:
                    row.append(255)
                else:
                    row.append(int(math.floor(snapped + 0.5)))
        rows.append(row)
    return rows
