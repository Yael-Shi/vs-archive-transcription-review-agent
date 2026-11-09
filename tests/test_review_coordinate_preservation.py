"""Recorded boxes survive first-pass acceptance and become a real crop."""

from __future__ import annotations

import json
from pathlib import Path

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _controller,
    _first_pass_body,
    _page,
    _png,
    _snapshot,
    _write_output,
)
from transcription_review_agent.review.coordinates import (
    corners_to_bbox,
    prepare_first_pass_coordinates,
)
from transcription_review_agent.review.models import BBox
from transcription_review_agent.review.paths import document_root

PAGE = ("page_001", 20, 20)


def test_half_open_corners_convert_without_clamping() -> None:
    assert corners_to_bbox([4, 5, 9, 8]) == BBox(x=4, y=5, width=5, height=3)
    assert corners_to_bbox([0, 0, 1, 1]) == BBox(x=0, y=0, width=1, height=1)
    assert corners_to_bbox([16, 0, 20, 4]) == BBox(x=16, y=0, width=4, height=4)
    assert corners_to_bbox([18, 0, 24, 4]) == BBox(x=18, y=0, width=6, height=4)
    for value in (
        [1, 2, 3],
        [1, 2, 3, 4.0],
        [5, 1, 5, 4],
        [8, 1, 4, 4],
        [True, 0, 2, 2],
    ):
        assert corners_to_bbox(value) is None
    prepared = prepare_first_pass_coordinates(
        {
            "pages": [
                {
                    "page_id": "page_001",
                    "regions": [
                        {"region_id": "r1", "bbox_px": [18, 0, 24, 4]},
                        {
                            "region_id": "r2",
                            "bbox": {"x": 1, "y": 1, "width": 2, "height": 2},
                            "bbox_px": [4, 4, 8, 8],
                        },
                    ],
                }
            ]
        },
        [PAGE],
    )
    regions = prepared["pages"][0]["regions"]  # type: ignore[index]
    assert "bbox" not in regions[0]
    assert "bbox_px" not in regions[0]
    assert "bbox" not in regions[1]
    assert "bbox_px" not in regions[1]


def test_line_boxes_fill_a_region_only_when_every_line_has_one() -> None:
    prepared = prepare_first_pass_coordinates(
        {
            "pages": [
                {
                    "page_id": "page_001",
                    "regions": [
                        {
                            "region_id": "r1",
                            "lines": [
                                {"line_id": "a", "bbox_px": [1, 1, 5, 3]},
                                {"line_id": "b", "bbox_px": [1, 4, 6, 7]},
                            ],
                        }
                    ],
                }
            ]
        },
        [PAGE],
    )
    region = prepared["pages"][0]["regions"][0]  # type: ignore[index]
    assert region["bbox"] == {"x": 1, "y": 1, "width": 5, "height": 6}
    assert "bbox_px" not in region["lines"][0]
    partial = prepare_first_pass_coordinates(
        {
            "pages": [
                {
                    "page_id": "page_001",
                    "regions": [
                        {
                            "region_id": "r1",
                            "lines": [
                                {"line_id": "a", "bbox_px": [1, 1, 5, 3]},
                                {"line_id": "b", "text": "no box"},
                            ],
                        }
                    ],
                }
            ]
        },
        [PAGE],
    )
    missed = partial["pages"][0]["regions"][0]  # type: ignore[index]
    assert "bbox" not in missed
    mismatched = prepare_first_pass_coordinates(
        {
            "pages": [
                {
                    "page_id": "page_001",
                    "width": 30,
                    "height": 40,
                    "regions": [{"region_id": "r1", "bbox_px": [1, 1, 5, 5]}],
                }
            ]
        },
        [PAGE],
    )
    assert "bbox" not in mismatched["pages"][0]["regions"][0]  # type: ignore[index]


def _publish_pair(
    tmp_path: Path, gpt_box: object, grok_box: object
) -> tuple[Path, bytes, bytes]:
    controller = _controller(tmp_path, pages=[_png(20, 20)])
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    sealed = _snapshot(tmp_path / "batches")
    gpt_request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    grok_request = controller.publish(BATCH_ID, DOCUMENT_ID, "grok")
    gpt_body = _first_pass_body(gpt_request, [_page("page_001", 1, "Ionnes")])
    grok_body = _first_pass_body(grok_request, [_page("page_001", 1, "Iannes")])
    _set_box(gpt_body, gpt_box)
    _set_box(grok_body, grok_box)
    gpt_path = _write_output(controller, gpt_request, gpt_body)
    grok_path = _write_output(controller, grok_request, grok_body)
    gpt_bytes = gpt_path.read_bytes()
    grok_bytes = grok_path.read_bytes()
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt")
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok")
    assert gpt_path.read_bytes() == gpt_bytes
    assert grok_path.read_bytes() == grok_bytes
    controller.compare(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    assert _snapshot(tmp_path / "batches") == sealed
    root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    return root, gpt_bytes, grok_bytes


def _set_box(body: dict[str, object], box: object) -> None:
    pages = body["pages"]
    assert isinstance(pages, list)
    regions = pages[0]["regions"]
    assert isinstance(regions, list)
    region = regions[0]
    assert isinstance(region, dict)
    region.pop("bbox", None)
    if box is not None:
        region["bbox_px"] = box


def _crop(root: Path) -> dict[str, object]:
    crops = list((root / "challenge_round_01" / "input" / "crops").glob("*.json"))
    assert len(crops) == 1
    return json.loads(crops[0].read_text(encoding="utf-8"))


def _inputs_match(root: Path) -> None:
    stage = root / "challenge_round_01" / "tasks"

    def files(provider: str) -> dict[str, bytes]:
        folder = stage / provider / "input"
        return {
            path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*")
            if path.is_file()
        }

    assert files("gpt") == files("grok")


def test_valid_bbox_px_becomes_a_targeted_challenge_crop(tmp_path: Path) -> None:
    root, gpt_bytes, _grok_bytes = _publish_pair(tmp_path, [4, 5, 9, 8], [4, 5, 9, 8])
    payload = _crop(root)
    assert payload["window_kind"] == "TARGETED"
    assert payload["page_id"] == "page_001"
    assert payload["target_bbox"] == {"x": 4, "y": 5, "width": 5, "height": 3}
    assert payload["crop_bbox"]["width"] < 20
    assert "bbox_px" in gpt_bytes.decode("utf-8")
    assert (root / "challenge_round_01" / "input" / "page_001.png").is_file()
    _inputs_match(root)


def test_malformed_and_absent_coordinates_stay_full_page(tmp_path: Path) -> None:
    outside = _crop(
        _publish_pair(tmp_path / "outside", [18, 0, 24, 4], [18, 0, 24, 4])[0]
    )
    assert outside["window_kind"] == "FULL_PAGE_FALLBACK"
    assert outside["fallback_reason"] == "NO_RECORDED_COORDINATES"
    assert outside["target_bbox"] == {"x": 0, "y": 0, "width": 20, "height": 20}
    assert outside["crop_bbox"] == outside["target_bbox"]
    absent = _crop(_publish_pair(tmp_path / "absent", None, None)[0])
    assert absent["window_kind"] == "FULL_PAGE_FALLBACK"
    assert absent["fallback_reason"] == "NO_RECORDED_COORDINATES"
    short = _crop(_publish_pair(tmp_path / "short", [1, 2, 3], [1, 2, 3])[0])
    assert short["window_kind"] == "FULL_PAGE_FALLBACK"
    assert short["target_bbox"]["width"] == 20
