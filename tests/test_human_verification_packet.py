"""Read-only human-verification packet."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
)
from transcription_review_agent.review.human_verification_packet import (
    build_human_verification_packet,
    phase1_band,
    render_packet_html,
    select_phase1_queue,
)
from transcription_review_agent.review.learning_export import (
    DocumentSummary,
    EvidenceRefRecord,
    LearningCandidate,
    LearningCandidateExport,
    OmissionCounts,
    PageRef,
)
from transcription_review_agent.review.models import BBox, CropRecord, DisputeCategory
from transcription_review_agent.review.models import DISPUTE_PRIORITY

PILOT_ROOTS = (
    Path("outputs/reviews_track_a_pilot_104"),
    Path("outputs/reviews_track_a_pilot_105"),
    Path("outputs/reviews_track_a_pilot_108"),
)
BATCHES = Path("outputs/batches")
BATCH = "batch-000001"


def _sha(seed: bytes) -> str:
    return hashlib.sha256(seed).hexdigest()


def _page(filename: str, data: bytes, *, order: int = 1) -> PageRef:
    return PageRef(
        page_id="page_001",
        reading_order=order,
        sha256=_sha(data),
        filename=filename,
        mime_type="image/jpeg",
        width=100,
        height=200,
    )


def _candidate(
    ident: str,
    *,
    document_id: int = 104,
    category: str = "WORDING",
    gpt: str = "left",
    grok: str = "right",
    band: str = "unresolved",
    page: PageRef | None = None,
    dispute_id: str = "dsp_1",
    closure: str | None = "SELECTED_UNRESOLVED",
    role: str = "FULL_DISPUTE",
    related: str | None = None,
    bbox: BBox | None = None,
    span: str | None = None,
    evidence: list[EvidenceRefRecord] | None = None,
    selected: str | None = None,
    superseded_text: str | None = None,
) -> LearningCandidate:
    image = page or _page("page_001.jpg", b"page")
    supported = band == "supported"
    return LearningCandidate(
        candidate_id="lc_" + ident,
        batch_id=BATCH,
        document_id=document_id,
        source_bundle_sha256=_sha(b"bundle"),
        page_id=image.page_id,
        page_image_sha256=image.sha256,
        page_image_filename=image.filename,
        page_mime_type=image.mime_type,
        derived_png_sha256=image.derived_png_sha256,
        gpt_region_ids=[],
        grok_region_ids=[],
        dispute_id=dispute_id,
        dispute_category=category,  # type: ignore[arg-type]
        learning_category=(
            "MODEL_SUPPORTED_READING" if supported else "UNRESOLVED_DISAGREEMENT"
        ),
        model_supported=supported,
        reading_role=role,  # type: ignore[arg-type]
        gpt_text=gpt,
        grok_text=grok,
        dispute_gpt_text=gpt,
        dispute_grok_text=grok,
        model_selected_reading=selected,
        superseded_text=superseded_text,
        image_span_status=span or ("RECORDED_REGION_BBOX" if bbox else "UNAVAILABLE"),  # type: ignore[arg-type]
        image_span_limitation="recorded limitation",
        image_bbox=bbox,
        reconciled_status="AUTOMATICALLY_SUPPORTED" if supported else "UNRESOLVED",
        closure_status=None if supported else closure,
        related_candidate_id=related,
        evidence=evidence or [],
        review_report_sha256=_sha(b"report"),
        comparison_sha256=_sha(b"comparison"),
        gpt_output_sha256=_sha(b"gpt"),
        grok_output_sha256=_sha(b"grok"),
    )


def _summary(page: PageRef, document_id: int = 104) -> DocumentSummary:
    return DocumentSummary(
        batch_id=BATCH,
        document_id=document_id,
        source_bundle_sha256=_sha(b"bundle"),
        review_report_sha256=_sha(b"report"),
        comparison_sha256=_sha(b"comparison"),
        quality="UNRESOLVED_CONTENT",
        review_completion="completed",
        sealed_bundle_checked=False,
        candidate_count=1,
        model_supported_count=0,
        unresolved_count=1,
        rejected_or_superseded_count=0,
        pages=[page],
    )


def _export(
    candidates: list[LearningCandidate], pages: list[tuple[int, PageRef]]
) -> LearningCandidateExport:
    return LearningCandidateExport(
        limitations=["Every candidate is NOT_VERIFIED."],
        documents=[_summary(page, document_id) for document_id, page in pages],
        candidates=candidates,
        omitted=OmissionCounts(),
    )


def _plant(root: Path, document_id: int, filename: str, data: bytes) -> None:
    directory = (
        root / f"batch_{BATCH}" / "documents" / f"document_{document_id}" / "input"
    )
    directory.mkdir(parents=True)
    (directory / filename).write_bytes(data)


def _dead_pid() -> int:
    process = subprocess.Popen(["/bin/true"])
    pid = process.pid
    process.wait()
    return pid


def _file_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_bytes(path.read_bytes())
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def test_queue_orders_unresolved_before_supported() -> None:
    page = _page("page_001.jpg", b"page")
    unresolved = _candidate("a" * 20, page=page, category="WORDING")
    supported = _candidate(
        "b" * 20,
        page=page,
        band="supported",
        category="MISSING_CONTENT",
        dispute_id="dsp_supported",
    )
    wording = _candidate(
        "c" * 20,
        page=page,
        category="WORDING",
        dispute_id="dsp_other",
        closure="NOT_SELECTED",
    )
    queued = select_phase1_queue(
        _export([supported, wording, unresolved], [(104, page)]),
        document_ids=[104],
    )
    assert [item.band for item in queued] == [0, 1]
    assert [item.candidate.candidate_id for item in queued] == [
        unresolved.candidate_id,
        supported.candidate_id,
    ]
    assert phase1_band(wording) is None


def test_queue_sorts_within_a_band() -> None:
    short = _page("page_001.jpg", b"short")
    late = PageRef(
        page_id="page_002",
        reading_order=2,
        sha256=_sha(b"late"),
        filename="page_002.jpg",
        mime_type="image/jpeg",
        width=100,
        height=200,
    )
    early_page_late_order = PageRef(
        page_id="page_009",
        reading_order=1,
        sha256=_sha(b"early"),
        filename="page_009.jpg",
        mime_type="image/jpeg",
        width=100,
        height=200,
    )
    rows = [
        _candidate(
            "1" * 20,
            document_id=108,
            category="WORDING",
            gpt="a long wording",
            grok="a long wording too",
            page=short,
            dispute_id="dsp_w",
        ),
        _candidate(
            "2" * 20,
            document_id=105,
            category="NAME_OR_ENTITY",
            gpt="Ann",
            grok="Anne",
            page=late,
            dispute_id="dsp_n2",
        ),
        _candidate(
            "3" * 20,
            document_id=104,
            category="NAME_OR_ENTITY",
            gpt="Ann",
            grok="Anne",
            page=early_page_late_order,
            dispute_id="dsp_n1",
        ),
        _candidate(
            "4" * 20,
            document_id=105,
            category="MISSING_CONTENT",
            gpt="gap",
            grok="missing gap text",
            page=short,
            dispute_id="dsp_m",
        ),
        _candidate(
            "5" * 20,
            document_id=104,
            category="NAME_OR_ENTITY",
            span="RECORDED_REGION_BBOX",
            bbox=BBox(x=1, y=1, width=2, height=2),
            page=short,
            dispute_id="dsp_box",
        ),
    ]
    export = LearningCandidateExport(
        limitations=["Every candidate is NOT_VERIFIED."],
        documents=[
            _summary(short, 104).model_copy(
                update={"pages": [short, early_page_late_order]}
            ),
            _summary(short, 105).model_copy(update={"pages": [short, late]}),
            _summary(short, 108),
        ],
        candidates=rows,
        omitted=OmissionCounts(),
    )
    queued = select_phase1_queue(export, document_ids=[104, 105, 108])
    assert [item.candidate.dispute_id for item in queued] == [
        "dsp_box",
        "dsp_m",
        "dsp_n1",
        "dsp_n2",
        "dsp_w",
    ]
    priorities = [
        DISPUTE_PRIORITY[DisputeCategory(item.candidate.dispute_category)]
        for item in queued
        if item.candidate.image_bbox is None
    ]
    assert priorities == sorted(priorities)


def test_queue_skips_other_rows_and_does_not_pad() -> None:
    page = _page("page_001.jpg", b"page")
    primary = _candidate("a" * 20, page=page)
    superseded = _candidate(
        "b" * 20,
        page=page,
        role="SUPERSEDED_READING",
        related=primary.candidate_id,
        dispute_id="dsp_old",
        superseded_text="old reading",
        band="rejected",
    )
    other_document = _candidate("c" * 20, document_id=7, page=page, dispute_id="dsp_7")
    queued = select_phase1_queue(
        _export([superseded, other_document, primary], [(104, page), (7, page)]),
        document_ids=[104, 105, 108],
    )
    assert len(queued) == 1
    assert queued[0].superseded[0].candidate_id == superseded.candidate_id
    both = _candidate(
        "d" * 20,
        page=page,
        band="supported",
        dispute_id="dsp_both",
    ).model_copy(update={"closure_status": "SELECTED_UNRESOLVED"})
    assert phase1_band(both) == 0


def test_missing_page_image_writes_nothing(tmp_path: Path) -> None:
    page = _page("page_001.jpg", b"page")
    export = _export([_candidate("a" * 20, page=page)], [(104, page)])
    output = tmp_path / "packet"
    with pytest.raises(ReviewIntegrityError, match="missing page image"):
        build_human_verification_packet(
            [tmp_path / "reviews"],
            output_dir=output,
            document_ids=[104],
            export=export,
        )
    assert not output.exists()


def test_page_hash_mismatch_and_escape_write_nothing(tmp_path: Path) -> None:
    page = _page("page_001.jpg", b"expected")
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", b"different")
    export = _export([_candidate("a" * 20, page=page)], [(104, page)])
    with pytest.raises(ReviewIntegrityError, match="hash does not match"):
        build_human_verification_packet(
            [reviews],
            output_dir=tmp_path / "packet",
            document_ids=[104],
            export=export,
        )
    hostile = page.model_copy(update={"filename": "../page_001.jpg"})
    escaped = _candidate("b" * 20, page=hostile)
    with pytest.raises(ReviewPathError, match="one path segment"):
        build_human_verification_packet(
            [reviews],
            output_dir=tmp_path / "packet-2",
            document_ids=[104],
            export=_export([escaped], [(104, hostile)]),
        )
    assert not (tmp_path / "packet").exists()


def test_html_escapes_model_text_and_draws_only_a_recorded_box() -> None:
    page = _page("page_001.jpg", b"page")
    hostile = '<script>alert(1)</script> & "quotes"'
    plain = _candidate("a" * 20, page=page, gpt=hostile, grok="ok")
    located = _candidate(
        "b" * 20,
        page=page,
        dispute_id="dsp_box",
        bbox=BBox(x=10, y=20, width=30, height=40),
        category="NAME_OR_ENTITY",
    )
    export = _export([plain, located], [(104, page)])
    queued = select_phase1_queue(export, document_ids=[104])
    rendered = render_packet_html(
        queued,
        limitations=["limit <tag>"],
        export_sha256=_sha(b"export"),
        windows_index_path=None,
        image_paths={page.sha256: f"images/{page.sha256}.jpg"},
        crop_paths={},
    )
    assert hostile not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&amp;" in rendered
    assert "limit &lt;tag&gt;" in rendered
    assert 'class="box"' in rendered
    assert (
        "left:10.000000%;top:10.000000%;width:30.000000%;height:20.000000%" in rendered
    )
    assert rendered.count('class="box"') == 1
    assert rendered.count("NOT_VERIFIED") >= 2
    assert "<form" not in rendered
    assert (
        "writer" not in rendered.lower()
        or "Writer identity is not recorded" in rendered
    )


def test_packet_is_deterministic_and_copies_each_page_once(tmp_path: Path) -> None:
    data = b"same-page"
    page = _page("page_001.jpg", data)
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    derived = (
        reviews
        / f"batch_{BATCH}"
        / "documents"
        / "document_104"
        / "input"
        / "page_001.png"
    )
    derived.write_bytes(b"derived-png")
    page = page.model_copy(update={"derived_png_sha256": _sha(b"derived-png")})
    first = _candidate("a" * 20, page=page, gpt="one")
    second = _candidate(
        "b" * 20,
        page=page,
        band="supported",
        dispute_id="dsp_2",
        gpt="two",
    )
    decoy = (
        reviews
        / f"batch_{BATCH}"
        / "documents"
        / "document_104"
        / "challenge_round_01"
        / "input"
        / "crops"
        / "crop_full.png"
    )
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"full-page-crop")
    first = first.model_copy(
        update={
            "evidence": [
                EvidenceRefRecord(
                    provider="gpt",
                    kind="FULL_PAGE_FALLBACK",
                    evidence_id="crop_full",
                    page_id="page_001",
                    assessment="SUPPORTS_DECISION",
                    window_kind="FULL_PAGE_FALLBACK",
                    crop_sha256=_sha(b"full-page-crop"),
                )
            ]
        }
    )
    export = _export([first, second], [(104, page)])
    before = _tree(reviews)
    left = build_human_verification_packet(
        [reviews],
        output_dir=tmp_path / "left",
        document_ids=[104],
        export=export,
        windows_index_path=r"\\wsl.localhost\Ubuntu\tmp\packet\index.html",
    )
    right = build_human_verification_packet(
        [reviews],
        output_dir=tmp_path / "right",
        document_ids=[104],
        export=export,
        windows_index_path=r"\\wsl.localhost\Ubuntu\tmp\packet\index.html",
    )
    assert left.model_dump() == right.model_dump()
    assert _tree(tmp_path / "left") == _tree(tmp_path / "right")
    assert before == _tree(reviews)
    images = list((tmp_path / "left" / "images").iterdir())
    assert len(images) == 1
    assert sha256_bytes(images[0].read_bytes()) == page.sha256
    html_text = (tmp_path / "left" / "index.html").read_text(encoding="utf-8")
    assert "Full-page fallback. This is not a word crop." in html_text
    assert "crop_full.png" not in html_text
    assert "NOT_VERIFIED" in html_text
    assert left.decision_ledger_written is False
    assert left.human_verified_count == 0
    manifest = json.loads(
        (tmp_path / "left" / "packet_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["decision_ledger_written"] is False
    assert [item["candidate_id"] for item in manifest["cases"]] == [
        first.candidate_id,
        second.candidate_id,
    ]


def test_targeted_crop_is_copied_and_a_missing_crop_fails(tmp_path: Path) -> None:
    data = b"page-bytes"
    page = _page("page_001.jpg", data)
    crop_bytes = b"crop-bytes"
    crop_hash = _sha(crop_bytes)
    evidence = EvidenceRefRecord(
        provider="grok",
        kind="CROP",
        evidence_id="crop_word",
        page_id="page_001",
        assessment="SUPPORTS_DECISION",
        window_kind="TARGETED",
        crop_sha256=crop_hash,
    )
    candidate = _candidate("a" * 20, page=page, evidence=[evidence])
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    crops = (
        reviews
        / f"batch_{BATCH}"
        / "documents"
        / "document_104"
        / "challenge_round_01"
        / "input"
        / "crops"
    )
    crops.mkdir(parents=True)
    task_copy = (
        reviews
        / f"batch_{BATCH}"
        / "documents"
        / "document_104"
        / "challenge_round_01"
        / "tasks"
        / "gpt"
        / "input"
        / "crops"
    )
    task_copy.mkdir(parents=True)
    (task_copy / "crop_word.png").write_bytes(crop_bytes)
    record = CropRecord(
        crop_id="crop_word",
        dispute_id="dsp_1",
        page_id="page_001",
        page_sha256=page.sha256,
        page_width=100,
        page_height=200,
        target_bbox=BBox(x=1, y=1, width=4, height=4),
        crop_bbox=BBox(x=0, y=0, width=8, height=8),
        enlarged=False,
        window_kind="TARGETED",
        coordinate_sources=["region.bbox"],
        image_filename="crop_word.png",
        crop_sha256=crop_hash,
        crop_png_width=8,
        crop_png_height=8,
    )
    (crops / "crop_word.json").write_text(record.model_dump_json(), encoding="utf-8")
    (crops / "crop_word.png").write_bytes(crop_bytes)
    export = _export([candidate], [(104, page)])
    manifest = build_human_verification_packet(
        [reviews],
        output_dir=tmp_path / "packet",
        document_ids=[104],
        export=export,
    )
    copied = tmp_path / "packet" / "images" / f"{crop_hash}.png"
    assert copied.is_file()
    assert sha256_bytes(copied.read_bytes()) == crop_hash
    assert any(item.role == "RECORDED_TARGETED_CROP" for item in manifest.images)
    html_text = (tmp_path / "packet" / "index.html").read_text(encoding="utf-8")
    assert f"images/{crop_hash}.png" in html_text
    (crops / "crop_word.png").unlink()
    with pytest.raises(ReviewIntegrityError, match="missing targeted crop image"):
        build_human_verification_packet(
            [reviews],
            output_dir=tmp_path / "missing-crop",
            document_ids=[104],
            export=export,
        )
    assert not (tmp_path / "missing-crop").exists()


def test_output_inside_reviews_is_refused(tmp_path: Path) -> None:
    data = b"page"
    page = _page("page_001.jpg", data)
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    before = _tree(reviews)
    with pytest.raises(ReviewPathError, match="inside a review"):
        build_human_verification_packet(
            [reviews],
            output_dir=reviews / "packet",
            document_ids=[104],
            export=_export([_candidate("a" * 20, page=page)], [(104, page)]),
        )
    assert before == _tree(reviews)


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    data = b"page"
    page = _page("page_001.jpg", data)
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    output = tmp_path / "packet"
    manifest = build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        dry_run=True,
        export=_export([_candidate("a" * 20, page=page)], [(104, page)]),
    )
    assert manifest.case_count == 1
    assert not output.exists()


def test_pilot_packet_matches_live_queue(tmp_path: Path) -> None:
    if not all(path.is_dir() for path in PILOT_ROOTS) or not BATCHES.is_dir():
        pytest.skip("completed pilot reviews are not in this checkout")
    before = {str(path): _tree(path) for path in (*PILOT_ROOTS, BATCHES)}
    output = tmp_path / "pilot"
    manifest = build_human_verification_packet(
        PILOT_ROOTS,
        output_dir=output,
        batches_dir=BATCHES,
        document_ids=[104, 105, 108],
        windows_index_path=None,
    )
    again = build_human_verification_packet(
        PILOT_ROOTS,
        output_dir=tmp_path / "pilot-2",
        batches_dir=BATCHES,
        document_ids=[104, 105, 108],
        windows_index_path=None,
    )
    after = {str(path): _tree(path) for path in (*PILOT_ROOTS, BATCHES)}
    assert before == after
    assert manifest.model_dump() == again.model_dump()
    assert manifest.band_0_count == 14
    assert manifest.band_1_count == 7
    assert manifest.case_count == 21
    assert manifest.human_verified_count == 0
    assert manifest.decision_ledger_written is False
    assert all(
        item.human_verification_status == "NOT_VERIFIED" for item in manifest.cases
    )
    assert all(not item.image_bbox_recorded for item in manifest.cases)
    assert [item.band for item in manifest.cases] == [0] * 14 + [1] * 7
    priorities = [
        DISPUTE_PRIORITY[DisputeCategory(item.dispute_category)]
        for item in manifest.cases
        if item.band == 0
    ]
    assert priorities == sorted(priorities)
    html_text = (output / "index.html").read_text(encoding="utf-8")
    assert 'class="box"' not in html_text
    assert 'class="mark human"' not in html_text
    assert "<form" not in html_text
    assert "ol.jump" not in html_text
    assert html_text.count("<article ") == 21
    assert html_text.count('<span class="status">NOT_VERIFIED</span>') == 21
    assert html_text.count('data-pilot="1"') == 3
    assert html_text.count('<p class="loc-status">לא סומן מיקום.</p>') == 21
    images = list((output / "images").iterdir())
    assert len(images) == len({item.page_image_sha256 for item in manifest.cases})
    for image in images:
        assert sha256_bytes(image.read_bytes()) == image.stem
    from transcription_review_agent.review.learning_export import (
        export_learning_candidates,
    )

    export = export_learning_candidates(
        PILOT_ROOTS,
        batches_dir=BATCHES,
        document_ids=[104, 105, 108],
        dry_run=True,
    )
    queued = select_phase1_queue(export, document_ids=[104, 105, 108])
    for item in queued:
        assert item.candidate.candidate_id in html_text
        for text in (item.candidate.gpt_text, item.candidate.grok_text):
            if any(mark in text for mark in "<>&"):
                assert text not in html_text
    assert not (output / "ledger.jsonl").exists()


def test_unsaved_marks_warn_and_export_does_not_claim_a_saved_file() -> None:
    page = _page("page_001.jpg", b"page")
    queued = select_phase1_queue(
        _export([_candidate("a" * 20, page=page)], [(104, page)]),
        document_ids=[104],
    )
    rendered = render_packet_html(
        queued,
        limitations=["limit"],
        export_sha256=_sha(b"export"),
        windows_index_path=None,
        image_paths={page.sha256: f"images/{page.sha256}.jpg"},
        crop_paths={},
    )
    assert "beforeunload" in rendered
    assert "זה לא אישור שהקובץ נשמר" in rendered
    assert "הוא העותק שנשמר" not in rendered
    assert "<script" in rendered and "</script>" in rendered


def test_packet_publish_refuses_unknown_content_and_keeps_sidecars(
    tmp_path: Path,
) -> None:
    from transcription_review_agent.review.localization import (
        LocalizationMark,
        LocalizationSidecar,
    )

    data = b"page"
    page = _page("page_002.jpg", data).model_copy(update={"page_id": "page_002"})
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_002.jpg", data)
    candidate = _candidate(
        "a" * 20,
        document_id=104,
        category="NAME_OR_ENTITY",
        page=page,
        dispute_id="dsp_" + "ab" * 8,
    )
    export = _export([candidate], [(104, page)])
    output = tmp_path / "packet"
    sidecar = LocalizationSidecar(
        marks=[
            LocalizationMark(
                candidate_id=candidate.candidate_id,
                document_id=104,
                page_id="page_002",
                page_image_sha256=page.sha256,
                page_width=100,
                page_height=200,
                status="NO_LOCATION",
            )
        ]
    )
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=export,
        localization=sidecar,
    )
    exported = output / "localization.json"
    original_export = exported.read_bytes()
    exported.write_bytes(original_export + b"\n")
    index_before = (output / "index.html").read_bytes()
    with pytest.raises(ReviewIntegrityError, match="manually exported sidecar"):
        build_human_verification_packet(
            [reviews],
            output_dir=output,
            document_ids=[104],
            export=export,
            localization=sidecar,
        )
    assert exported.read_bytes() == original_export + b"\n"
    assert (output / "index.html").read_bytes() == index_before
    (output / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError, match="unexpected file"):
        build_human_verification_packet(
            [reviews],
            output_dir=output,
            document_ids=[104],
            export=export,
        )
    assert (output / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert exported.read_bytes() == original_export + b"\n"
    unknown = tmp_path / "notes"
    unknown.mkdir()
    (unknown / "notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError, match="not a verification packet"):
        build_human_verification_packet(
            [reviews],
            output_dir=unknown,
            document_ids=[104],
            export=export,
        )
    assert (unknown / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert not (unknown / "index.html").exists()
    backup = tmp_path / f".{output.name}.previous"
    backup.mkdir()
    (backup / "keep.txt").write_text("keep", encoding="utf-8")
    (output / "notes.txt").unlink()
    exported.write_bytes(original_export)
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=export,
    )
    assert (backup / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert exported.read_bytes() == original_export


def test_failed_packet_update_restores_the_previous_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcription_review_agent.review.human_verification_packet as packet_module

    data = b"page"
    page = _page("page_001.jpg", data)
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    export = _export([_candidate("a" * 20, page=page)], [(104, page)])
    output = tmp_path / "packet"
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=export,
    )
    index = output / "index.html"
    before = index.read_bytes()
    manifest = (output / "packet_manifest.json").read_bytes()
    real_replace = packet_module.os.replace

    failed = False

    def fail_manifest(src: object, dst: object) -> None:
        nonlocal failed
        if str(dst).endswith("packet_manifest.json") and not failed:
            failed = True
            raise OSError("manifest replace failed")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr(packet_module.os, "replace", fail_manifest)
    with pytest.raises(OSError, match="manifest replace failed"):
        build_human_verification_packet(
            [reviews],
            output_dir=output,
            document_ids=[104],
            export=export,
            windows_index_path="CHANGED-PATH",
        )
    assert index.read_bytes() == before
    assert (output / "packet_manifest.json").read_bytes() == manifest
    assert b"CHANGED-PATH" not in index.read_bytes()
    assert not list(tmp_path.glob(".packet-rollback-*"))


def test_short_os_writes_publish_the_complete_packet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcription_review_agent.review.human_verification_packet as packet_module

    data = b"page-bytes"
    page = _page("page_001.jpg", data)
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    export = _export(
        [_candidate("a" * 20, page=page, gpt="<script>alert(1)</script>")],
        [(104, page)],
    )
    real_write = packet_module.os.write

    def short_write(descriptor: int, payload: bytes | memoryview) -> int:
        view = bytes(payload)
        if len(view) > 2:
            return real_write(descriptor, view[:2])
        return real_write(descriptor, view)

    monkeypatch.setattr(packet_module.os, "write", short_write)
    output = tmp_path / "packet"
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=export,
    )
    html_text = (output / "index.html").read_text(encoding="utf-8")
    assert html_text.startswith("<!DOCTYPE html>")
    assert html_text.rstrip().endswith("</html>")
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_text
    manifest = json.loads((output / "packet_manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 1
    copied = output / "images" / f"{page.sha256}.jpg"
    assert copied.read_bytes() == data


def test_failed_restore_keeps_a_recoverable_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcription_review_agent.review.human_verification_packet as packet_module

    data = b"page"
    page = _page("page_001.jpg", data)
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    export = _export([_candidate("a" * 20, page=page)], [(104, page)])
    output = tmp_path / "packet"
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=export,
    )
    index_before = (output / "index.html").read_bytes()
    sidecar = output / "localization.json"
    sidecar.write_text('{"marks": []}\n', encoding="utf-8")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    def fail_install(*_args: object, **_kwargs: object) -> None:
        (output / "index.html").write_bytes(b"damaged")
        raise OSError("install failed")

    def fail_restore(*_args: object, **_kwargs: object) -> None:
        raise OSError("restore failed")

    monkeypatch.setattr(packet_module, "_install_generated", fail_install)
    monkeypatch.setattr(packet_module, "_restore_snapshot", fail_restore)
    with pytest.raises(ReviewIntegrityError, match="recovery backup kept at") as caught:
        build_human_verification_packet(
            [reviews],
            output_dir=output,
            document_ids=[104],
            export=export,
        )
    backup = Path(str(caught.value).rsplit("recovery backup kept at ", 1)[1])
    assert backup.is_dir()
    assert (backup / "index.html").read_bytes() == index_before
    assert sidecar.read_text(encoding="utf-8") == '{"marks": []}\n'
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not (backup / "localization.json").exists()


def test_abandoned_temps_are_removed_without_deleting_other_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transcription_review_agent.review.human_verification_packet as packet_module

    source = tmp_path / "incoming.html"
    source.write_bytes(b"replacement")
    packet = tmp_path / "packet"
    packet.mkdir()
    (packet / "index.html").write_bytes(b"old")
    (packet / "packet_manifest.json").write_bytes(b"{}")
    dead = _dead_pid()
    stale = packet / f".index.html.{dead}.abcdefabcdef.tmp"
    stale.write_bytes(b"partial")
    image_name = (
        ".0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
        f".jpg.{dead}.abcdefabcdef.tmp"
    )
    image_stale = packet / "images" / image_name
    image_stale.parent.mkdir()
    image_stale.write_bytes(b"partial-image")
    kept = packet / ".notes.txt.9.abcdefabcdef.tmp"
    kept.write_bytes(b"keep")
    sidecar = packet / "localization.json"
    sidecar.write_text("human\n", encoding="utf-8")
    reads = 0
    real_read = Path.read_bytes

    def counting(self: Path) -> bytes:
        nonlocal reads
        if self == source:
            reads += 1
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    packet_module._discard_abandoned_temps([stale, image_stale])
    packet_module._replace_file(source, packet / "index.html")
    assert reads == 1
    assert (packet / "index.html").read_bytes() == b"replacement"
    assert not stale.exists()
    assert not image_stale.exists()
    assert kept.read_bytes() == b"keep"
    assert sidecar.read_text(encoding="utf-8") == "human\n"
    assert (packet / "packet_manifest.json").read_bytes() == b"{}"


def test_packet_validation_precedes_temp_cleanup(tmp_path: Path) -> None:
    data = b"page"
    page = _page("page_001.jpg", data)
    reviews = tmp_path / "reviews"
    _plant(reviews, 104, "page_001.jpg", data)
    export = _export([_candidate("a" * 20, page=page)], [(104, page)])
    dead = _dead_pid()
    temp_name = f".index.html.{dead}.abcdefabcdef.tmp"

    unknown = tmp_path / "unrelated"
    unknown.mkdir()
    (unknown / temp_name).write_bytes(b"foreign temp")
    (unknown / "notes.txt").write_text("keep", encoding="utf-8")
    before_unknown = _file_bytes(unknown)
    with pytest.raises(ReviewIntegrityError, match="not a verification packet"):
        build_human_verification_packet(
            [reviews],
            output_dir=unknown,
            document_ids=[104],
            export=export,
        )
    assert _file_bytes(unknown) == before_unknown

    output = tmp_path / "packet"
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=export,
    )
    sidecar = output / "localization.json"
    sidecar.write_text("human export\n", encoding="utf-8")
    notes = output / "notes.txt"
    notes.write_text("keep", encoding="utf-8")
    blocked = output / temp_name
    blocked.write_bytes(b"abandoned but blocked")
    before_blocked = _file_bytes(output)
    with pytest.raises(ReviewIntegrityError, match="unexpected file"):
        build_human_verification_packet(
            [reviews],
            output_dir=output,
            document_ids=[104],
            export=export,
        )
    assert _file_bytes(output) == before_blocked

    notes.unlink()
    held = subprocess.Popen(["/bin/sleep", "30"])
    live = output / f".index.html.{held.pid}.abcdefabcdef.tmp"
    live.write_bytes(b"in progress")
    try:
        with pytest.raises(ReviewIntegrityError, match="may still be active"):
            build_human_verification_packet(
                [reviews],
                output_dir=output,
                document_ids=[104],
                export=export,
                windows_index_path="SHOULD-NOT-LAND",
            )
        assert live.read_bytes() == b"in progress"
        assert sidecar.read_text(encoding="utf-8") == "human export\n"
        assert b"SHOULD-NOT-LAND" not in (output / "index.html").read_bytes()
    finally:
        held.kill()
        held.wait()
    live.unlink()
    blocked.unlink()

    abandoned = output / f".index.html.{_dead_pid()}.abcdefabcdef.tmp"
    abandoned.write_bytes(b"abandoned")
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=export,
        windows_index_path="REBUILT",
    )
    assert not abandoned.exists()
    assert sidecar.read_text(encoding="utf-8") == "human export\n"
    assert b"REBUILT" in (output / "index.html").read_bytes()
