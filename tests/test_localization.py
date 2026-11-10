"""Manual localization sidecar for the three pilot disputes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.errors import ReviewIntegrityError
from transcription_review_agent.review.human_verification_packet import (
    build_human_verification_packet,
    render_packet_html,
    select_phase1_queue,
)
from transcription_review_agent.review.learning_export import (
    DocumentSummary,
    LearningCandidate,
    LearningCandidateExport,
    OmissionCounts,
    PageRef,
)
from transcription_review_agent.review.localization import (
    LOCALIZATION_SCHEMA,
    NORMALIZED_SCALE,
    CaseBinding,
    LocalizationMark,
    LocalizationSidecar,
    NormalizedBox,
    export_sidecar,
    load_sidecar,
)

BATCH = "batch-000001"


def _sha(seed: bytes) -> str:
    return hashlib.sha256(seed).hexdigest()


def _page(page_id: str) -> PageRef:
    return PageRef(
        page_id=page_id,
        reading_order=int(page_id.removeprefix("page_")),
        sha256=_sha(page_id.encode()),
        filename=f"{page_id}.jpg",
        mime_type="image/jpeg",
        width=100,
        height=200,
    )


def _candidate(
    ident: str,
    page: PageRef,
    *,
    document_id: int,
    category: str,
) -> LearningCandidate:
    return LearningCandidate(
        candidate_id="lc_" + ident,
        batch_id=BATCH,
        document_id=document_id,
        source_bundle_sha256=_sha(b"bundle"),
        page_id=page.page_id,
        page_image_sha256=page.sha256,
        page_image_filename=page.filename,
        page_mime_type=page.mime_type,
        derived_png_sha256=None,
        gpt_region_ids=[],
        grok_region_ids=[],
        dispute_id="dsp_" + ident[:8],
        dispute_category=category,  # type: ignore[arg-type]
        learning_category="UNRESOLVED_DISAGREEMENT",
        model_supported=False,
        reading_role="FULL_DISPUTE",
        gpt_text="alpha",
        grok_text="beta",
        dispute_gpt_text="alpha",
        dispute_grok_text="beta",
        image_span_status="UNAVAILABLE",
        image_span_limitation="recorded limitation",
        reconciled_status="UNRESOLVED",
        closure_status="SELECTED_UNRESOLVED",
        evidence=[],
        review_report_sha256=_sha(b"report"),
        comparison_sha256=_sha(b"comparison"),
        gpt_output_sha256=_sha(b"gpt"),
        grok_output_sha256=_sha(b"grok"),
    )


def _export(rows: list[tuple[LearningCandidate, PageRef]]) -> LearningCandidateExport:
    documents = []
    seen: set[tuple[int, str]] = set()
    for candidate, page in rows:
        key = (candidate.document_id, page.page_id)
        if key in seen:
            continue
        seen.add(key)
        documents.append(
            DocumentSummary(
                batch_id=BATCH,
                document_id=candidate.document_id,
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
        )
    return LearningCandidateExport(
        limitations=["Every candidate is NOT_VERIFIED."],
        documents=documents,
        candidates=[candidate for candidate, _page in rows],
        omitted=OmissionCounts(),
    )


def _binding(candidate: LearningCandidate, *, pilot: bool) -> CaseBinding:
    return CaseBinding(
        candidate_id=candidate.candidate_id,
        document_id=candidate.document_id,
        page_id=candidate.page_id,
        page_image_sha256=candidate.page_image_sha256,
        page_width=100,
        page_height=200,
        dispute_category=candidate.dispute_category,
        pilot=pilot,
    )


def _mark(candidate: LearningCandidate, **updates: object) -> LocalizationMark:
    payload: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "document_id": candidate.document_id,
        "page_id": candidate.page_id,
        "page_image_sha256": candidate.page_image_sha256,
        "page_width": 100,
        "page_height": 200,
        "status": "HUMAN_MARKED",
        "bbox": {"x": 10_000, "y": 20_000, "width": 30_000, "height": 40_000},
    }
    payload.update(updates)
    return LocalizationMark.model_validate(payload)


def _cases() -> tuple[LearningCandidate, LearningCandidate, LearningCandidate, PageRef]:
    page = _page("page_002")
    name = _candidate("a" * 20, page, document_id=104, category="NAME_OR_ENTITY")
    date = _candidate("b" * 20, page, document_id=104, category="DATE_OR_NUMBER")
    other = _candidate("c" * 20, _page("page_001"), document_id=105, category="WORDING")
    return name, date, other, page


def test_round_trip_keeps_marks_and_refuses_verification() -> None:
    name, date, other, _page = _cases()
    sidecar = LocalizationSidecar(
        marks=[
            _mark(name),
            _mark(date, status="UNCLEAR", bbox=None),
            _mark(
                other,
                status="NO_LOCATION",
                bbox=None,
                document_id=105,
                page_id="page_001",
                page_image_sha256=other.page_image_sha256,
            ),
        ]
    )
    bindings = [
        _binding(name, pilot=True),
        _binding(other, pilot=True),
    ]
    # The date row is on the same page but is not a pilot, so it cannot ride along.
    with pytest.raises(ReviewIntegrityError, match="not one of the three pilot"):
        load_sidecar(export_sidecar(sidecar), bindings)

    kept = LocalizationSidecar(
        marks=[
            _mark(name),
            _mark(
                other,
                status="NO_LOCATION",
                bbox=None,
                document_id=105,
                page_id="page_001",
                page_image_sha256=other.page_image_sha256,
            ),
        ]
    )
    loaded = load_sidecar(export_sidecar(kept), bindings)
    assert loaded == kept
    assert loaded.schema_version == LOCALIZATION_SCHEMA
    assert loaded.human_verified_count == 0
    assert loaded.decision_ledger_written is False
    assert all(
        item.human_verification_status == "NOT_VERIFIED" for item in loaded.marks
    )
    browser = {
        "schema_version": "dispute-localization-v1",
        "packet_role": "MANUAL_LOCALIZATION_SIDECAR",
        "decision_ledger_written": False,
        "human_verified_count": 0,
        "marks": [kept.marks[0].model_dump(mode="json")],
    }
    assert (
        load_sidecar(json.dumps(browser), [_binding(name, pilot=True)]).marks[0]
        == kept.marks[0]
    )
    with pytest.raises(ValidationError):
        _mark(name, human_verification_status="VERIFIED")
    with pytest.raises(ValidationError):
        _mark(name, eligible_for_authoritative_learning=True)


def test_wrong_case_page_and_coordinates_are_rejected() -> None:
    name, date, _other, page = _cases()
    bindings = [_binding(name, pilot=True), _binding(date, pilot=False)]
    with pytest.raises(ReviewIntegrityError, match="not one of the three pilot"):
        load_sidecar(
            export_sidecar(LocalizationSidecar(marks=[_mark(date)])),
            bindings,
        )
    with pytest.raises(ReviewIntegrityError, match="does not match the original page"):
        load_sidecar(
            export_sidecar(
                LocalizationSidecar(
                    marks=[_mark(name, page_image_sha256=_sha(b"other-page"))]
                )
            ),
            bindings,
        )
    with pytest.raises(ReviewIntegrityError, match="does not match the original page"):
        load_sidecar(
            export_sidecar(
                LocalizationSidecar(marks=[_mark(name, page_id="page_009")])
            ),
            bindings,
        )
    full_page = _mark(name).model_dump(mode="json")
    full_page["bbox"] = {
        "x": 0,
        "y": 0,
        "width": NORMALIZED_SCALE,
        "height": NORMALIZED_SCALE,
        "scale": NORMALIZED_SCALE,
    }
    with pytest.raises(ReviewIntegrityError, match="does not match the schema"):
        load_sidecar(
            json.dumps(
                {
                    "schema_version": "dispute-localization-v1",
                    "packet_role": "MANUAL_LOCALIZATION_SIDECAR",
                    "human_verified_count": 0,
                    "decision_ledger_written": False,
                    "marks": [full_page],
                }
            ),
            bindings,
        )
    with pytest.raises(ValidationError):
        NormalizedBox(x=900_000, y=0, width=200_000, height=1)
    with pytest.raises(ValidationError):
        NormalizedBox(x=0, y=0, width=0, height=1)
    with pytest.raises(ReviewIntegrityError, match="does not match the schema"):
        load_sidecar('{"schema_version":"other"}', bindings)
    assert page.sha256 == name.page_image_sha256


def test_rendered_box_stays_on_the_selected_case_and_escapes_text() -> None:
    name, date, _other, page = _cases()
    queued = select_phase1_queue(
        _export([(name, page), (date, page)]),
        document_ids=[104],
    )
    sidecar = LocalizationSidecar(marks=[_mark(name)])
    hostile = "<script>alert(1)</script>"
    rendered = render_packet_html(
        queued,
        limitations=[hostile],
        export_sha256=_sha(b"export"),
        windows_index_path=None,
        image_paths={page.sha256: f"images/{page.sha256}.jpg"},
        crop_paths={},
        localization=sidecar,
    )
    assert hostile not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    name_html, date_html = _articles(rendered)
    assert 'data-pilot="1"' in name_html
    assert 'class="mark human"' in name_html
    assert 'data-provenance="HUMAN_MARKED"' in name_html
    assert "NOT_VERIFIED" in name_html
    assert 'data-pilot="0"' in date_html
    assert 'class="mark human"' not in date_html
    assert "לא סומן מיקום." in date_html
    assert "Unclear location" not in date_html
    assert rendered.count('class="mark human"') == 1
    assert "<form" not in rendered


def test_build_writes_sidecar_outside_reviews(tmp_path: Path) -> None:
    name, _date, _other, page = _cases()
    reviews = tmp_path / "reviews"
    directory = reviews / f"batch_{BATCH}" / "documents" / "document_104" / "input"
    directory.mkdir(parents=True)
    (directory / page.filename).write_bytes(b"page-bytes")
    # The planted file must hash to the page record.
    digest = hashlib.sha256(b"page-bytes").hexdigest()
    page = page.model_copy(update={"sha256": digest})
    name = name.model_copy(update={"page_image_sha256": digest})
    before = _tree(reviews)
    output = tmp_path / "packet"
    sidecar = LocalizationSidecar(marks=[_mark(name, page_image_sha256=digest)])
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=_export([(name, page)]),
        localization=sidecar,
    )
    assert _tree(reviews) == before
    assert not (reviews / "localization.json").exists()
    stored = (output / "localization.json").read_text(encoding="utf-8")
    loaded = load_sidecar(
        stored,
        [_binding(name, pilot=True)],
    )
    assert loaded.marks[0].status == "HUMAN_MARKED"
    assert loaded.marks[0].bbox is not None
    html_text = (output / "index.html").read_text(encoding="utf-8")
    assert 'class="mark human"' in html_text
    assert html_text.count('<span class="status">NOT_VERIFIED</span>') == 1
    assert sha256_bytes((output / "images" / f"{digest}.jpg").read_bytes()) == digest


def _articles(rendered: str) -> tuple[str, str]:
    parts = [part.split("</article>", 1)[0] for part in rendered.split("<article ")[1:]]
    name = next(part for part in parts if 'data-category="NAME_OR_ENTITY"' in part)
    date = next(part for part in parts if 'data-category="DATE_OR_NUMBER"' in part)
    return name, date


def _tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_bytes(path.read_bytes())
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }
