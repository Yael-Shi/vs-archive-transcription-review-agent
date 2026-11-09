"""Focused challenge crops use recorded coordinates and label every fallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _controller,
    _deliver_first_pass,
    _first_pass_body,
    _png,
    _snapshot,
    _write_output,
)
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.jpegutil import encode_grayscale_dc_jpeg
from transcription_review_agent.review.models import (
    BBox,
    CoverageReport,
    Dispute,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    SourceEvidence,
    TextRegion,
)
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.pngutil import decode_png, encode_png
from transcription_review_agent.review.windows import resolve_dispute_window

SHA = "ab" * 32
PILOT = Path("/mnt/c/Users/User/vs-archive-transcription-review-agent/desktop_pilot")


def _coverage(region_ids: list[str]) -> CoverageReport:
    return CoverageReport(
        coverage_complete=True,
        unresolved_text_present=False,
        examined_region_ids=region_ids,
    )


def _output(
    provider: str, regions: list[TextRegion], *, page_id: str = "page_001"
) -> FirstPassOutput:
    interface = "GPT_WORK" if provider == "gpt" else "GROK"
    page = PageTranscription(
        page_id=page_id,
        reading_order=1,
        diplomatic_transcription="\n".join(
            line.text for region in regions for line in region.lines
        ),
        regions=regions,
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=[],
        coverage=_coverage([region.region_id for region in regions]),
        cancellations=[],
        insertions=[],
        marginal_notes=[],
        inaccessible_or_ambiguous=[],
    )
    return FirstPassOutput(
        provider=provider,  # type: ignore[arg-type]
        provider_interface=interface,  # type: ignore[arg-type]
        task_id=f"task-{provider}",
        job_id=f"job-{provider}",
        batch_id="batch-000001",
        document_id=101,
        attempt=1,
        source_bundle_sha256=SHA,
        pages=[page],
    )


def _region(
    region_id: str,
    text: str,
    *,
    order: int = 1,
    bbox: BBox | None = None,
    line_id: str | None = None,
) -> TextRegion:
    return TextRegion(
        region_id=region_id,
        order=order,
        lines=[LineReading(line_id=line_id or f"{region_id}-l1", text=text)],
        bbox=bbox,
    )


def _dispute(
    gpt_text: str,
    grok_text: str,
    *,
    gpt_regions: list[str] | None = None,
    grok_regions: list[str] | None = None,
) -> Dispute:
    return Dispute(
        dispute_id="dsp_test",
        category="WORDING",  # type: ignore[arg-type]
        page_id="page_001",
        region_id=(gpt_regions or ["r1"])[0],
        gpt_region_ids=gpt_regions or ["r1"],
        grok_region_ids=grok_regions or ["r1"],
        gpt_text=gpt_text,
        grok_text=grok_text,
        substantive=True,
    )


def _box_evidence(
    provider: str,
    path: str,
    value: object,
    *,
    region_id: str | None = "r1",
    line_id: str | None = None,
) -> SourceEvidence:
    return SourceEvidence(
        provider=provider,  # type: ignore[arg-type]
        original_sha256=SHA,
        json_path=path,
        value=json.dumps(value, separators=(",", ":")),
        page_id="page_001",
        region_id=region_id,
        line_id=line_id,
    )


def _window(
    gpt: FirstPassOutput,
    grok: FirstPassOutput,
    dispute: Dispute,
    evidence: list[SourceEvidence] | None = None,
    *,
    width: int = 20,
    height: int = 20,
):
    return resolve_dispute_window(
        dispute,
        gpt,
        grok,
        page_width=width,
        page_height=height,
        source_evidence=evidence or [],
    )


def test_accepted_region_box_is_a_focused_crop_on_that_page() -> None:
    box = BBox(x=4, y=6, width=5, height=3)
    gpt = _output("gpt", [_region("r1", "Ionnes", bbox=box)])
    grok = _output("grok", [_region("r1", "Iannes", bbox=box)])
    window = _window(gpt, grok, _dispute("Ionnes", "Iannes"))
    assert window.kind == "TARGETED"
    assert window.fallback_reason is None
    assert window.target == box
    assert window.coordinate_sources == ("accepted:r1:bbox",)


def test_line_bbox_px_uses_exact_text_and_keeps_the_edge() -> None:
    gpt = _output("gpt", [_region("r1", "Ionnes")])
    grok = _output("grok", [_region("r1", "Iannes")])
    evidence = [
        _box_evidence(
            "gpt",
            "pages[0].regions[0].lines[0].bbox_px",
            [16, 0, 20, 4],
            line_id="r1-l1",
        ),
        _box_evidence(
            "grok",
            "pages[0].regions[0].lines[0].bbox_px",
            [16, 0, 20, 4],
            line_id="r1-l1",
        ),
    ]
    window = _window(gpt, grok, _dispute("Ionnes", "Iannes"), evidence)
    assert window.kind == "TARGETED"
    assert window.target == BBox(x=16, y=0, width=4, height=4)
    assert window.target.x + window.target.width == 20


def test_outside_invalid_and_ambiguous_boxes_stay_full_page() -> None:
    gpt = _output("gpt", [_region("r1", "Ionnes")])
    grok = _output("grok", [_region("r1", "Iannes")])
    dispute = _dispute("Ionnes", "Iannes")
    missing = _window(gpt, grok, dispute)
    assert missing.kind == "FULL_PAGE_FALLBACK"
    assert missing.fallback_reason == "NO_RECORDED_COORDINATES"
    assert missing.target == BBox(x=0, y=0, width=20, height=20)
    assert missing.coordinate_sources == ()

    outside = _window(
        gpt,
        grok,
        dispute,
        [
            _box_evidence(
                "gpt",
                "pages[0].regions[0].bbox_px",
                [18, 0, 24, 4],
            )
        ],
    )
    assert outside.kind == "FULL_PAGE_FALLBACK"
    assert outside.fallback_reason == "INVALID_COORDINATES"
    assert outside.target == missing.target

    malformed = _window(
        gpt,
        grok,
        dispute,
        [_box_evidence("gpt", "pages[0].regions[0].bbox_px", [1, 2, "3", 4])],
    )
    assert malformed.fallback_reason == "INVALID_COORDINATES"

    listed = _window(
        gpt,
        grok,
        dispute,
        [_box_evidence("gpt", "pages[0].regions[0].bbox", [4, 4, 8, 8])],
    )
    assert listed.kind == "FULL_PAGE_FALLBACK"
    assert listed.fallback_reason == "AMBIGUOUS_COORDINATES"

    conflict = _window(
        _output(
            "gpt", [_region("r1", "Ionnes", bbox=BBox(x=1, y=1, width=2, height=2))]
        ),
        grok,
        dispute,
        [
            _box_evidence(
                "gpt",
                "pages[0].regions[0].bbox_px",
                [8, 8, 12, 12],
            )
        ],
    )
    assert conflict.fallback_reason == "AMBIGUOUS_COORDINATES"
    assert conflict.target == missing.target


def test_disagreement_between_providers_is_their_union() -> None:
    gpt = _output(
        "gpt", [_region("r1", "Ionnes", bbox=BBox(x=2, y=2, width=4, height=4))]
    )
    grok = _output(
        "grok", [_region("r1", "Iannes", bbox=BBox(x=6, y=6, width=4, height=3))]
    )
    window = _window(gpt, grok, _dispute("Ionnes", "Iannes"))
    assert window.kind == "TARGETED"
    assert window.target == BBox(x=2, y=2, width=8, height=7)


def test_a_full_page_box_is_not_labeled_a_focused_crop() -> None:
    page = BBox(x=0, y=0, width=20, height=20)
    gpt = _output("gpt", [_region("r1", "Ionnes", bbox=page)])
    grok = _output("grok", [_region("r1", "Iannes")])
    window = _window(gpt, grok, _dispute("Ionnes", "Iannes"))
    assert window.kind == "FULL_PAGE_FALLBACK"
    assert window.fallback_reason == "RECORDED_BOX_IS_FULL_PAGE"


def test_similar_text_without_a_line_match_does_not_invent_a_crop() -> None:
    gpt = _output(
        "gpt",
        [
            TextRegion(
                region_id="r1",
                order=1,
                lines=[
                    LineReading(line_id="r1-l1", text="alpha extra"),
                    LineReading(line_id="r1-l2", text="beta"),
                ],
            )
        ],
    )
    grok = _output("grok", [_region("r1", "alpha")])
    evidence = [
        _box_evidence(
            "gpt",
            "pages[0].regions[0].lines[0].bbox_px",
            [1, 1, 5, 3],
            line_id="r1-l1",
        )
    ]
    window = _window(gpt, grok, _dispute("alpha", "alpha"), evidence)
    assert window.kind == "FULL_PAGE_FALLBACK"
    assert window.fallback_reason == "AMBIGUOUS_COORDINATES"
    assert window.target == BBox(x=0, y=0, width=20, height=20)


def test_declared_page_size_is_not_scaled_onto_the_sealed_page() -> None:
    gpt = _output("gpt", [_region("r1", "Ionnes")])
    grok = _output("grok", [_region("r1", "Iannes")])
    evidence = [
        _box_evidence("gpt", "pages[0].width", 30, region_id=None),
        _box_evidence("gpt", "pages[0].height", 40, region_id=None),
        _box_evidence("gpt", "pages[0].regions[0].bbox_px", [1, 1, 5, 5]),
    ]
    window = _window(gpt, grok, _dispute("Ionnes", "Iannes"), evidence)
    assert window.kind == "FULL_PAGE_FALLBACK"
    assert window.fallback_reason == "INVALID_COORDINATES"


def test_crop_pixels_keep_orientation_and_stop_at_the_page_edge() -> None:
    width, height = 12, 6
    rows = [bytearray([0, 0, 0] * width) for _ in range(height)]
    rows[1][10 * 3 : 11 * 3] = bytes((9, 8, 7))
    page = encode_png(width, height, rows, 3)
    from transcription_review_agent.review.crop import build_crop, materialize_crop

    record = build_crop(
        crop_id="crop_edge",
        dispute_id="dsp_edge",
        page_id="page_001",
        page_sha256=SHA,
        page_width=width,
        page_height=height,
        target=BBox(x=8, y=0, width=4, height=2),
    )
    updated, encoded = materialize_crop(record, page)
    crop_width, crop_height, crop_rows, _bpp = decode_png(encoded)
    assert (crop_width, crop_height) == (
        updated.crop_png_width,
        updated.crop_png_height,
    )
    assert crop_width > crop_height
    assert crop_width < width or updated.crop_bbox.x == 0
    sample_x = 10 - updated.crop_bbox.x
    sample_y = 1 - updated.crop_bbox.y
    assert bytes(crop_rows[sample_y][sample_x * 3 : sample_x * 3 + 3]) == bytes(
        (9, 8, 7)
    )
    assert updated.crop_bbox.x + updated.crop_bbox.width <= width
    assert updated.crop_bbox.y + updated.crop_bbox.height <= height
    with pytest.raises(ValueError, match="outside the page"):
        build_crop(
            crop_id="crop_past",
            dispute_id="dsp_past",
            page_id="page_001",
            page_sha256=SHA,
            page_width=width,
            page_height=height,
            target=BBox(x=10, y=0, width=4, height=2),
        )


def _task_inputs(root: Path) -> dict[str, dict[str, bytes]]:
    stage = root / "challenge_round_01" / "tasks"
    found = {}
    for provider in ("gpt", "grok"):
        folder = stage / provider / "input"
        found[provider] = {
            path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*")
            if path.is_file()
        }
    return found


def test_both_providers_get_the_same_focused_crop_and_the_page(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, pages=[_png(20, 20)])
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    sealed = _snapshot(tmp_path / "batches")
    gpt_regions = [
        _region("r1", "alpha one", bbox=BBox(x=2, y=2, width=4, height=3)),
        _region("r2", "beta one", order=2, bbox=BBox(x=2, y=12, width=4, height=3)),
    ]
    grok_regions = [
        _region("r1", "alpha two", bbox=BBox(x=2, y=2, width=4, height=3)),
        _region("r2", "beta two", order=2, bbox=BBox(x=8, y=12, width=4, height=3)),
    ]
    _deliver_first_pass(controller, "gpt", [_page_of(gpt_regions)])
    _deliver_first_pass(controller, "grok", [_page_of(grok_regions)])
    controller.compare(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    assert _snapshot(tmp_path / "batches") == sealed
    root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    inputs = _task_inputs(root)
    assert inputs["gpt"] == inputs["grok"]
    crops = sorted((root / "challenge_round_01" / "input" / "crops").glob("*.json"))
    assert len(crops) >= 2
    targets = []
    for path in crops:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["page_id"] == "page_001"
        assert payload["window_kind"] == "TARGETED"
        assert "fallback_reason" not in payload
        assert payload["crop_bbox"]["width"] < 20
        assert path.with_suffix(".png").is_file()
        targets.append(payload["target_bbox"])
    assert len({json.dumps(item, sort_keys=True) for item in targets}) >= 2
    page = (root / "challenge_round_01" / "input" / "page_001.png").read_bytes()
    assert page == (root / "input" / "page_001.png").read_bytes()
    assert list((root / "challenge_round_01" / "input").glob("page_001.png"))


def test_fallback_names_the_reason_and_does_not_duplicate_the_page(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, pages=[_png(20, 20)])
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    regions = [
        _region("r1", "alpha one"),
        _region("r2", "beta one", order=2),
    ]
    other = [
        _region("r1", "alpha two"),
        _region("r2", "beta two", order=2),
    ]
    _deliver_first_pass(controller, "gpt", [_page_of(regions)])
    _deliver_first_pass(controller, "grok", [_page_of(other)])
    controller.compare(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    assert _task_inputs(root)["gpt"] == _task_inputs(root)["grok"]
    crops = list((root / "challenge_round_01" / "input" / "crops").glob("*.json"))
    assert len(crops) >= 2
    page = (root / "challenge_round_01" / "input" / "page_001.png").read_bytes()
    assert not list((root / "challenge_round_01" / "input" / "crops").glob("*.png"))
    for path in crops:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["window_kind"] == "FULL_PAGE_FALLBACK"
        assert payload["fallback_reason"] == "NO_RECORDED_COORDINATES"
        assert payload["reuses_page_pixels"] is True
        assert payload["image_filename"] == "page_001.png"
        assert payload["target_bbox"] == {"x": 0, "y": 0, "width": 20, "height": 20}
        assert payload["crop_sha256"] == sha256_bytes(page)
        assert "CROP" not in payload["window_kind"]


def test_source_bbox_px_is_shared_without_rewriting_the_accepted_file(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path, pages=[_png(20, 20)])
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    page = _page_of([_region("r1", "Ionnes")])
    other = _page_of([_region("r1", "Iannes")])
    gpt_request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    grok_request = controller.publish(BATCH_ID, DOCUMENT_ID, "grok")
    gpt_output = _write_output(
        controller, gpt_request, _first_pass_body(gpt_request, [page])
    )
    _write_output(controller, grok_request, _first_pass_body(grok_request, [other]))
    source = {
        "provider": "gpt",
        "pages": [
            {
                "page_id": "page_001",
                "regions": [
                    {
                        "region_id": "r1",
                        "lines": [
                            {
                                "line_id": "r1-l1",
                                "text": "Ionnes",
                                "bbox_px": [4, 5, 9, 8],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    source_path = tmp_path / "original_gpt.json"
    source_bytes = (json.dumps(source) + "\n").encode("utf-8")
    source_path.write_bytes(source_bytes)
    accepted = gpt_output.read_bytes()
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", source_json=source_path)
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok")
    assert gpt_output.read_bytes() == accepted
    assert source_path.read_bytes() == source_bytes
    controller.compare(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    assert _task_inputs(root)["gpt"] == _task_inputs(root)["grok"]
    payload = json.loads(
        next(
            (root / "challenge_round_01" / "input" / "crops").glob("*.json")
        ).read_text(encoding="utf-8")
    )
    assert payload["window_kind"] == "TARGETED"
    assert payload["target_bbox"] == {"x": 4, "y": 5, "width": 5, "height": 3}
    assert payload["page_id"] == "page_001"
    assert any("bbox_px" in item for item in payload["coordinate_sources"])
    assert (root / "challenge_round_01" / "input" / "page_001.png").read_bytes() == (
        root / "input" / "page_001.png"
    ).read_bytes()


def test_jpeg_crop_uses_derived_pixels_and_keeps_the_original(tmp_path: Path) -> None:
    jpeg = encode_grayscale_dc_jpeg(16, 8, [0, 1])
    controller = _controller(tmp_path, pages=[jpeg], mime="image/jpeg")
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    box = BBox(x=8, y=0, width=8, height=8)
    _deliver_first_pass(
        controller, "gpt", [_page_of([_region("r1", "Ionnes", bbox=box)])]
    )
    _deliver_first_pass(
        controller, "grok", [_page_of([_region("r1", "Iannes", bbox=box)])]
    )
    controller.compare(BATCH_ID, DOCUMENT_ID)
    controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    challenge = root / "challenge_round_01" / "input"
    assert (challenge / "page_001.jpg").read_bytes() == jpeg
    assert (root / "input" / "page_001.jpg").read_bytes() == jpeg
    payload = json.loads(next((challenge / "crops").glob("*.json")).read_text())
    assert payload["window_kind"] == "TARGETED"
    assert payload["source_mime_type"] == "image/jpeg"
    crop = (challenge / "crops" / f"{payload['crop_id']}.png").read_bytes()
    crop_width, crop_height, rows, _bpp = decode_png(crop)
    assert crop_width == payload["crop_bbox"]["width"]
    assert crop_height == payload["crop_bbox"]["height"]
    assert crop_width > crop_height or crop_height == 8
    sample = 8 - payload["crop_bbox"]["x"]
    assert rows[0][sample] == 129
    assert _task_inputs(root)["gpt"] == _task_inputs(root)["grok"]


def _page_of(regions: list[TextRegion]) -> PageTranscription:
    return PageTranscription(
        page_id="page_001",
        reading_order=1,
        diplomatic_transcription="\n".join(
            line.text for region in regions for line in region.lines
        ),
        regions=regions,
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=[],
        coverage=_coverage([region.region_id for region in regions]),
        cancellations=[],
        insertions=[],
        marginal_notes=[],
        inaccessible_or_ambiguous=[],
    )


def test_document_108_has_no_recorded_coordinates_and_is_not_rewritten() -> None:
    paths = [
        PILOT / "first_pass_gpt.json",
        PILOT / "transcription_grok.json",
        PILOT / "doc108_comparison" / "selected_challenge_disputes.json",
        PILOT
        / "_doc108_challenge_r1_bundle"
        / "input"
        / "crops"
        / "crop_dsp_1592b280614a6ad2.json",
        PILOT / "doc108_final" / "review_report.json",
    ]
    if any(not path.is_file() for path in paths):
        pytest.skip("document 108 review files are not on this machine")
    before = {path: path.read_bytes() for path in paths}
    gpt = FirstPassOutput.model_validate_json(before[paths[0]])
    grok = FirstPassOutput.model_validate_json(before[paths[1]])
    selected = json.loads(before[paths[2]])
    if gpt.document_id != 108 or grok.document_id != 108:
        pytest.skip("provider files are not document 108")
    disputes = [Dispute.model_validate(item) for item in selected["selected_disputes"]]
    assert len(disputes) == 8
    windows = [
        resolve_dispute_window(
            dispute,
            gpt,
            grok,
            page_width=1500,
            page_height=2000,
        )
        for dispute in disputes
    ]
    assert {item.kind for item in windows} == {"FULL_PAGE_FALLBACK"}
    assert {item.fallback_reason for item in windows} == {"NO_RECORDED_COORDINATES"}
    assert all(item.coordinate_sources == () for item in windows)
    published = json.loads(before[paths[3]])
    assert published["window_kind"] == "FULL_PAGE_FALLBACK"
    assert published["page_id"] == "page_005"
    assert {path: path.read_bytes() for path in paths} == before
