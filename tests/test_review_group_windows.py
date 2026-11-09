"""Grouped challenge preparation when members do not share one window label."""

from __future__ import annotations

import json
from pathlib import Path

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _controller,
    _first_pass_body,
    _png,
    _write_output,
)
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.models import (
    ApparatusNote,
    CoverageReport,
    DisputeCategory,
    LineReading,
    PageTranscription,
    TextRegion,
)


def _page(text_tail: str, *, note: str | None) -> PageTranscription:
    return PageTranscription(
        page_id="page_001",
        reading_order=1,
        diplomatic_transcription=f"one\n{text_tail}",
        regions=[
            TextRegion(
                region_id="r1",
                order=1,
                lines=[
                    LineReading(line_id="l1", text="one"),
                    LineReading(line_id="l2", text=text_tail),
                ],
                bbox=None,
            )
        ],
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=[],
        coverage=CoverageReport(
            coverage_complete=True,
            unresolved_text_present=False,
            examined_region_ids=["r1"],
        ),
        cancellations=[],
        insertions=[],
        marginal_notes=[]
        if note is None
        else [ApparatusNote(region_id="r1", text=note)],
        inaccessible_or_ambiguous=[],
    )


def _publish_group(tmp_path: Path, line_box: list[int]) -> tuple[Path, dict[str, str]]:
    controller = _controller(tmp_path, pages=[_png(20, 20)])
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    gpt_request = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt")
    grok_request = controller.publish(BATCH_ID, DOCUMENT_ID, "grok")
    gpt_body = _first_pass_body(gpt_request, [_page("Alpha extra", note="Alpha")])
    grok_body = _first_pass_body(grok_request, [_page("Alfa extra", note=None)])
    gpt_path = _write_output(controller, gpt_request, gpt_body)
    _write_output(controller, grok_request, grok_body)
    source = tmp_path / "gpt_original.json"
    original = json.loads(gpt_path.read_text(encoding="utf-8"))
    original["pages"][0]["regions"][0]["lines"][1]["bbox_px"] = line_box
    source.write_text(json.dumps(original), encoding="utf-8")
    before = gpt_path.read_bytes()
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", source_json=source)
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok")
    assert gpt_path.read_bytes() == before
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    names: dict[str, str] = {}
    for item in comparison.disputes:
        if not item.substantive:
            continue
        if item.category is DisputeCategory.CANCELLATION_OR_INSERTION:
            names["note"] = item.dispute_id
        else:
            names["word"] = item.dispute_id
    assert set(names) == {"note", "word"}
    selected = [names["word"], names["note"]]
    controller.publish(
        BATCH_ID,
        DOCUMENT_ID,
        "gpt",
        stage="challenge",
        selected_dispute_ids=selected,
    )
    controller.publish(
        BATCH_ID,
        DOCUMENT_ID,
        "grok",
        stage="challenge",
        selected_dispute_ids=selected,
    )
    root = (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "challenge_round_01"
    )
    gpt_files = _files(root / "tasks" / "gpt" / "input")
    grok_files = _files(root / "tasks" / "grok" / "input")
    assert gpt_files == grok_files
    return root / "input", names


def _files(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in folder.rglob("*")
        if path.is_file()
    }


def _crops(stage_input: Path) -> list[dict[str, object]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((stage_input / "crops").glob("*.json"))
    ]


def test_different_fallback_reasons_share_one_full_page_group(tmp_path: Path) -> None:
    stage_input, names = _publish_group(tmp_path, [0, 0, 20, 20])
    crops = _crops(stage_input)
    assert len(crops) == 1
    crop = crops[0]
    assert crop["window_kind"] == "FULL_PAGE_FALLBACK"
    assert crop["page_id"] == "page_001"
    assert crop["target_bbox"] == {"x": 0, "y": 0, "width": 20, "height": 20}
    assert "fallback_reason" not in crop
    recorded = {
        item["dispute_id"]: item["fallback_reason"]
        for item in crop["member_fallbacks"]  # type: ignore[index]
    }
    assert recorded == {
        names["word"]: "RECORDED_BOX_IS_FULL_PAGE",
        names["note"]: "AMBIGUOUS_COORDINATES",
    }
    assert set(crop["dispute_ids"]) == set(names.values())  # type: ignore[arg-type]
    published = json.loads((stage_input / "disputes.json").read_text(encoding="utf-8"))
    assert published["groups"][0]["window_kind"] == "FULL_PAGE_FALLBACK"
    raw = json.dumps(crop)
    assert "gpt" not in raw and "grok" not in raw


def test_targeted_and_full_page_members_stay_separate_crops(tmp_path: Path) -> None:
    stage_input, _names = _publish_group(tmp_path, [1, 1, 5, 5])
    crops = _crops(stage_input)
    assert len(crops) == 2
    by_kind = {item["window_kind"]: item for item in crops}
    assert set(by_kind) == {"TARGETED", "FULL_PAGE_FALLBACK"}
    assert by_kind["TARGETED"]["target_bbox"] == {
        "x": 1,
        "y": 1,
        "width": 4,
        "height": 4,
    }
    assert "fallback_reason" not in by_kind["TARGETED"]
    assert "member_fallbacks" not in by_kind["TARGETED"]
    assert by_kind["FULL_PAGE_FALLBACK"]["target_bbox"] == {
        "x": 0,
        "y": 0,
        "width": 20,
        "height": 20,
    }
    assert by_kind["FULL_PAGE_FALLBACK"]["fallback_reason"] == "AMBIGUOUS_COORDINATES"
    published = json.loads((stage_input / "disputes.json").read_text(encoding="utf-8"))
    assert {item["window_kind"] for item in published["groups"]} == {
        "TARGETED",
        "FULL_PAGE_FALLBACK",
    }
