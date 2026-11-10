"""Visual proposal sidecar for the three pilot disputes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
)
from transcription_review_agent.review.human_verification_packet import (
    build_human_verification_packet,
    load_confirmed_location_displays,
    main,
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
    LocalizationMark,
    LocalizationSidecar,
)
from transcription_review_agent.review.visual_localization import (
    VISUAL_PROPOSAL_SCHEMA,
    ProposalBinding,
    ProposalProvenance,
    VisualProposal,
    VisualProposalSidecar,
    export_visual_proposals,
    load_visual_proposals,
    normalize_measured_pixels,
    prepare_request_sha256,
    unexecuted_placeholder,
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
    dispute: str,
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
        dispute_id=dispute,
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


def _binding(candidate: LearningCandidate, *, pilot: bool) -> ProposalBinding:
    return ProposalBinding(
        candidate_id=candidate.candidate_id,
        dispute_id=candidate.dispute_id,
        document_id=candidate.document_id,
        page_id=candidate.page_id,
        page_image_sha256=candidate.page_image_sha256,
        page_width=100,
        page_height=200,
        dispute_category=candidate.dispute_category,
        pilot=pilot,
    )


def _box() -> dict[str, int]:
    return {
        "x": 10_000,
        "y": 20_000,
        "width": 30_000,
        "height": 40_000,
        "scale": 1_000_000,
    }


def _proposal(candidate: LearningCandidate, **updates: object) -> VisualProposal:
    provenance = {
        "image_sha256": candidate.page_image_sha256,
        "method_version": "visual-proposal-pilot-1",
        "model_id": "example",
        "tool": "mock",
        "request_sha256": _sha(b"request"),
        "executed": True,
    }
    payload: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "dispute_id": candidate.dispute_id,
        "document_id": candidate.document_id,
        "page_id": candidate.page_id,
        "page_image_sha256": candidate.page_image_sha256,
        "page_width": 100,
        "page_height": 200,
        "dispute_category": candidate.dispute_category,
        "status": "PROPOSED",
        "method": "MODEL_PROPOSAL",
        "provenance": provenance,
        "boxes": [_box()],
        "explanation": "unverified suggestion",
        "nearby_text_anchor": "nearby words",
    }
    payload.update(updates)
    if "page_image_sha256" in updates and "provenance" not in updates:
        provenance["image_sha256"] = updates["page_image_sha256"]
        payload["provenance"] = provenance
    return VisualProposal.model_validate(payload)


def _cases() -> tuple[LearningCandidate, LearningCandidate, PageRef]:
    page = _page("page_002")
    name = _candidate(
        "a" * 20,
        page,
        document_id=104,
        category="NAME_OR_ENTITY",
        dispute="dsp_" + "a" * 16,
    )
    date = _candidate(
        "b" * 20,
        page,
        document_id=104,
        category="DATE_OR_NUMBER",
        dispute="dsp_" + "b" * 16,
    )
    return name, date, page


def test_request_hash_ignores_pixels_and_changes_with_the_reading() -> None:
    shared = {
        "candidate_id": "lc_" + "a" * 20,
        "dispute_id": "dsp_" + "a" * 16,
        "document_id": 104,
        "page_id": "page_002",
        "page_image_sha256": _sha(b"page"),
        "gpt_text": "Eli",
        "grok_text": "Elie",
    }
    assert prepare_request_sha256(**shared) == prepare_request_sha256(**shared)
    changed = dict(shared)
    changed["gpt_text"] = "other"
    assert prepare_request_sha256(**shared) != prepare_request_sha256(**changed)


def test_measured_pixels_stay_inside_the_page() -> None:
    box = normalize_measured_pixels(
        x=10, y=20, width=30, height=40, page_width=100, page_height=200
    )
    assert box.model_dump() == {
        "x": 100_000,
        "y": 100_000,
        "width": 300_000,
        "height": 200_000,
        "scale": 1_000_000,
    }
    with pytest.raises(ReviewIntegrityError, match="full-page"):
        normalize_measured_pixels(
            x=0, y=0, width=100, height=200, page_width=100, page_height=200
        )
    with pytest.raises(ReviewIntegrityError, match="degenerate"):
        normalize_measured_pixels(
            x=0, y=0, width=0, height=10, page_width=100, page_height=200
        )
    with pytest.raises(ReviewIntegrityError, match="outside"):
        normalize_measured_pixels(
            x=90, y=0, width=20, height=10, page_width=100, page_height=200
        )


def test_statuses_boxes_and_human_fields_are_rejected() -> None:
    name, _date, _page = _cases()
    other = _box()
    other["x"] = 50_000
    loaded = load_visual_proposals(
        export_visual_proposals(
            VisualProposalSidecar(
                results=[
                    _proposal(name, status="AMBIGUOUS", boxes=[_box(), other]),
                ]
            )
        ),
        [_binding(name, pilot=True)],
    )
    assert loaded.schema_version == VISUAL_PROPOSAL_SCHEMA
    assert loaded.results[0].location_authority == "MODEL_PROPOSED"
    assert loaded.results[0].human_verification_status == "NOT_VERIFIED"
    assert loaded.human_verified_count == 0
    with pytest.raises(ValidationError):
        _proposal(name, status="NOT_FOUND", boxes=[_box()])
    with pytest.raises(ValidationError):
        _proposal(name, status="PROPOSED", boxes=[])
    with pytest.raises(ValidationError):
        _proposal(name, status="AMBIGUOUS", boxes=[_box()])
    with pytest.raises(ValidationError):
        _proposal(name, status="ERROR", boxes=[], explanation=None)
    with pytest.raises(ValidationError):
        _proposal(name, boxes=[_box(), _box()])
    with pytest.raises(ValidationError):
        _proposal(name, status="HUMAN_MARKED")
    with pytest.raises(ValidationError):
        _proposal(name, method="HUMAN_MARK")
    with pytest.raises(ValidationError):
        _proposal(name, location_authority="HUMAN_MARKED")
    with pytest.raises(ValidationError):
        _proposal(name, human_verification_status="VERIFIED")
    with pytest.raises(ValidationError):
        ProposalProvenance(
            image_sha256=name.page_image_sha256,
            method_version="visual-proposal-pilot-1",
            model_id="example",
            tool="mock",
            request_sha256=_sha(b"request"),
            executed=True,
            human_marked=True,
        )


def test_identity_wrong_page_and_duplicates_are_rejected() -> None:
    name, date, _page = _cases()
    bindings = [_binding(name, pilot=True), _binding(date, pilot=False)]
    with pytest.raises(ReviewIntegrityError, match="not one of the three pilot"):
        load_visual_proposals(
            export_visual_proposals(VisualProposalSidecar(results=[_proposal(date)])),
            bindings,
        )
    with pytest.raises(ReviewIntegrityError, match="does not match the original page"):
        load_visual_proposals(
            export_visual_proposals(
                VisualProposalSidecar(
                    results=[_proposal(name, page_image_sha256=_sha(b"other-page"))]
                )
            ),
            bindings,
        )
    with pytest.raises(ReviewIntegrityError, match="does not match the original page"):
        load_visual_proposals(
            export_visual_proposals(
                VisualProposalSidecar(results=[_proposal(name, page_id="page_009")])
            ),
            bindings,
        )
    with pytest.raises(ReviewIntegrityError, match="does not match the original page"):
        load_visual_proposals(
            export_visual_proposals(
                VisualProposalSidecar(
                    results=[_proposal(name, dispute_id="dsp_" + "c" * 16)]
                )
            ),
            bindings,
        )
    with pytest.raises(ReviewIntegrityError, match="repeat a candidate"):
        load_visual_proposals(
            export_visual_proposals(
                VisualProposalSidecar(results=[_proposal(name), _proposal(name)])
            ),
            bindings,
        )
    with pytest.raises(ReviewIntegrityError, match="does not match the schema"):
        load_visual_proposals('{"schema_version":"other"}', bindings)
    with pytest.raises(ReviewIntegrityError, match="different learning export"):
        load_visual_proposals(
            export_visual_proposals(
                VisualProposalSidecar(
                    export_sha256=_sha(b"other-export"),
                    results=[_proposal(name)],
                )
            ),
            bindings,
            export_sha256=_sha(b"export"),
        )


def test_unexecuted_placeholder_has_no_box_and_round_trips() -> None:
    name, _date, _page = _cases()
    binding = _binding(name, pilot=True)
    placeholder = unexecuted_placeholder(
        binding,
        gpt_text="alpha",
        grok_text="",
        reason="No vision execution was run.",
    )
    assert placeholder.status == "ERROR"
    assert placeholder.method == "UNEXECUTED"
    assert placeholder.boxes == []
    assert placeholder.provenance.executed is False
    loaded = load_visual_proposals(
        export_visual_proposals(VisualProposalSidecar(results=[placeholder])),
        [binding],
    )
    again = export_visual_proposals(loaded)
    assert again == export_visual_proposals(
        VisualProposalSidecar.model_validate_json(again)
    )


def test_rendered_suggestion_stays_off_the_other_dispute_and_escapes() -> None:
    name, date, page = _cases()
    queued = select_phase1_queue(
        _export([(name, page), (date, page)]),
        document_ids=[104],
    )
    hostile = "<script>alert(1)</script>"
    second = _box()
    second["y"] = 80_000
    proposals = VisualProposalSidecar(
        results=[
            _proposal(
                name,
                status="AMBIGUOUS",
                boxes=[_box(), second],
                explanation=hostile,
                nearby_text_anchor=hostile,
            )
        ]
    )
    mark = LocalizationMark.model_validate(
        {
            "candidate_id": name.candidate_id,
            "document_id": 104,
            "page_id": page.page_id,
            "page_image_sha256": page.sha256,
            "page_width": 100,
            "page_height": 200,
            "status": "HUMAN_MARKED",
            "bbox": _box(),
        }
    )
    before = mark.model_dump()
    rendered = render_packet_html(
        queued,
        limitations=["limit"],
        export_sha256=_sha(b"export"),
        windows_index_path=None,
        image_paths={page.sha256: f"images/{page.sha256}.jpg"},
        crop_paths={},
        localization=LocalizationSidecar(marks=[mark]),
        visual_proposals=proposals,
    )
    assert mark.model_dump() == before
    assert hostile not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    name_html, date_html = _articles(rendered)
    assert name_html.count('data-provenance="MODEL_PROPOSED"') == 2
    assert "הצעה 1" in name_html
    assert "הצעה 2" in name_html
    assert "GPT קרא: «alpha»" in name_html
    assert "Grok קרא: «beta»" in name_html
    assert "אין כאן קביעה איזו קריאה נכונה." in name_html
    assert 'data-provenance="HUMAN_MARKED"' in name_html
    assert "NOT_VERIFIED" in name_html
    assert "זה לא אימות של התמלול" in name_html
    assert name_html.count('class="inspect-region"') == 2
    assert "<h3>GPT</h3>" in name_html
    assert "<h3>Grok</h3>" in name_html
    assert 'data-provenance="MODEL_PROPOSED"' not in date_html
    assert "proposal-status" not in date_html
    assert "לא סומן מיקום." in date_html

    missing = VisualProposalSidecar(
        results=[_proposal(name, status="NOT_FOUND", boxes=[], explanation=None)]
    )
    not_found = render_packet_html(
        queued,
        limitations=["limit"],
        export_sha256=_sha(b"export"),
        windows_index_path=None,
        image_paths={page.sha256: f"images/{page.sha256}.jpg"},
        crop_paths={},
        visual_proposals=missing,
    )
    found_name, _found_date = _articles(not_found)
    assert "לא נמצא אזור מתאים" in found_name
    assert 'class="proposal"' not in found_name
    assert 'class="inspect-region"' not in found_name


def test_confirmed_location_is_drawn_only_for_the_matching_pilot() -> None:
    name, date, page = _cases()
    queued = select_phase1_queue(
        _export([(name, page), (date, page)]),
        document_ids=[104],
    )
    payload = {
        "locations": [
            {
                "candidate_id": name.candidate_id,
                "dispute_id": name.dispute_id,
                "document_id": name.document_id,
                "page_id": name.page_id,
                "page_image_sha256": name.page_image_sha256,
                "page_width": 100,
                "page_height": 200,
                "boxes": [_box()],
                "location_authority": "HUMAN_CONFIRMED_LOCATION",
                "confirmation": "USER_VISUAL_CONFIRMATION",
                "human_verification_status": "NOT_VERIFIED",
                "transcription_verified": False,
            }
        ]
    }
    rows = load_confirmed_location_displays(json.dumps(payload))
    rendered = render_packet_html(
        queued,
        limitations=["limit"],
        export_sha256=_sha(b"export"),
        windows_index_path=None,
        image_paths={page.sha256: f"images/{page.sha256}.jpg"},
        crop_paths={},
        confirmed_locations={name.candidate_id: rows[0]},
    )
    name_html, date_html = _articles(rendered)
    assert name_html.count('data-provenance="HUMAN_CONFIRMED_LOCATION"') == 1
    assert 'data-provenance="HUMAN_CONFIRMED_LOCATION"' not in date_html
    assert "NOT_VERIFIED" in name_html
    assert "מיקום שאושר 1" in name_html
    verified = json.loads(json.dumps(payload))
    verified["locations"][0]["transcription_verified"] = True
    with pytest.raises(ReviewIntegrityError, match="display contract"):
        load_confirmed_location_displays(json.dumps(verified))
    outside = rows[0].model_copy(
        update={
            "candidate_id": date.candidate_id,
            "dispute_id": date.dispute_id,
        }
    )
    with pytest.raises(ReviewIntegrityError, match="outside the three pilot"):
        render_packet_html(
            queued,
            limitations=["limit"],
            export_sha256=_sha(b"export"),
            windows_index_path=None,
            image_paths={page.sha256: f"images/{page.sha256}.jpg"},
            crop_paths={},
            confirmed_locations={date.candidate_id: outside},
        )


def test_build_writes_proposals_outside_reviews_and_refuses_a_protected_path(
    tmp_path: Path,
) -> None:
    name, _date, page = _cases()
    reviews = tmp_path / "reviews"
    directory = reviews / f"batch_{BATCH}" / "documents" / "document_104" / "input"
    directory.mkdir(parents=True)
    (directory / page.filename).write_bytes(b"page-bytes")
    digest = hashlib.sha256(b"page-bytes").hexdigest()
    page = page.model_copy(update={"sha256": digest})
    name = name.model_copy(update={"page_image_sha256": digest})
    before = _tree(reviews)
    output = tmp_path / "packet"
    proposals = VisualProposalSidecar(
        results=[_proposal(name, page_image_sha256=digest)]
    )
    human = LocalizationSidecar(
        marks=[
            LocalizationMark.model_validate(
                {
                    "candidate_id": name.candidate_id,
                    "document_id": 104,
                    "page_id": "page_002",
                    "page_image_sha256": digest,
                    "page_width": 100,
                    "page_height": 200,
                    "status": "HUMAN_MARKED",
                    "bbox": _box(),
                }
            )
        ]
    )
    original_human = human.model_dump()
    build_human_verification_packet(
        [reviews],
        output_dir=output,
        document_ids=[104],
        export=_export([(name, page)]),
        localization=human,
        visual_proposals=proposals,
    )
    assert human.model_dump() == original_human
    assert _tree(reviews) == before
    assert not (reviews / "visual_proposals.json").exists()
    stored = (output / "visual_proposals.json").read_text(encoding="utf-8")
    loaded = load_visual_proposals(stored, [_binding(name, pilot=True)])
    assert loaded.results[0].status == "PROPOSED"
    assert loaded.results[0].method == "MODEL_PROPOSAL"
    html_text = (output / "index.html").read_text(encoding="utf-8")
    assert 'data-provenance="MODEL_PROPOSED"' in html_text
    assert 'data-provenance="HUMAN_MARKED"' in html_text
    assert name.image_bbox is None
    protected = reviews / "visual_proposals.json"
    protected.write_text(stored, encoding="utf-8")
    with pytest.raises(ReviewPathError, match="refusing to read a visual proposal"):
        main(
            [
                "--reviews-dir",
                str(reviews),
                "--output-dir",
                str(tmp_path / "elsewhere"),
                "--document-id",
                "104",
                "--visual-proposals",
                str(protected),
                "--dry-run",
            ]
        )


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


def test_review_screen_is_hebrew_and_keeps_reading_direction() -> None:
    name, date, page = _cases()
    name = name.model_copy(update={"gpt_text": "ligne un\nligne deux", "grok_text": ""})
    queued = select_phase1_queue(
        _export([(name, page), (date, page)]),
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
    assert '<html lang="he" dir="rtl">' in rendered
    assert "unicode-bidi: plaintext" in rendered
    assert 'id="fit-page"' in rendered
    assert 'id="focus-region"' in rendered
    assert 'id="view-status"' in rendered
    assert 'id="prev"' in rendered and 'id="next"' in rendered
    name_html, date_html = _articles(rendered)
    primary, _technical = name_html.split('<details class="technical">', 1)
    assert "NOT_VERIFIED" not in primary
    assert "ligne un\nligne deux" in primary
    assert "GPT כלל את הקטע «ligne un ligne deux»" in primary
    assert "Grok לא כלל קטע" in primary
    assert "הקריאה הנכונה" not in primary
    assert "לא נמסרה קריאה." in primary
    assert "אין מיקום מסומן" in date_html
    assert 'id="approve"' not in rendered
    assert ">אשר<" not in rendered
    assert "<form" not in rendered


def test_focus_prefers_confirmed_location_over_a_model_proposal() -> None:
    name, _date, page = _cases()
    queued = select_phase1_queue(
        _export([(name, page)]),
        document_ids=[104],
    )
    proposal = VisualProposalSidecar(results=[_proposal(name)])
    confirmed = load_confirmed_location_displays(
        json.dumps(
            {
                "locations": [
                    {
                        "candidate_id": name.candidate_id,
                        "dispute_id": name.dispute_id,
                        "document_id": name.document_id,
                        "page_id": name.page_id,
                        "page_image_sha256": name.page_image_sha256,
                        "page_width": 100,
                        "page_height": 200,
                        "boxes": [_box()],
                        "location_authority": "HUMAN_CONFIRMED_LOCATION",
                        "confirmation": "USER_VISUAL_CONFIRMATION",
                        "human_verification_status": "NOT_VERIFIED",
                        "transcription_verified": False,
                    }
                ]
            }
        )
    )
    rendered = render_packet_html(
        queued,
        limitations=["limit"],
        export_sha256=_sha(b"export"),
        windows_index_path=None,
        image_paths={page.sha256: f"images/{page.sha256}.jpg"},
        crop_paths={},
        visual_proposals=proposal,
        confirmed_locations={name.candidate_id: confirmed[0]},
    )
    listener = rendered.split('getElementById("focus-region")', 1)[1][:700]
    confirmed_at = listener.index('querySelector(".confirmed")')
    proposal_at = listener.index('querySelector(".proposal")')
    page_at = listener.index("fitPage(current)")
    assert confirmed_at < proposal_at < page_at
    article = rendered.split("<article ", 1)[1]
    assert 'data-kind="confirmed"' in article
    assert 'data-kind="proposal"' in article
    assert "הקריאה לא אומתה." in rendered
    assert "התמלול אומת" not in rendered


def test_invalid_payload_is_not_applied(tmp_path: Path) -> None:
    name, _date, _page = _cases()
    raw = json.dumps(
        {
            "schema_version": VISUAL_PROPOSAL_SCHEMA,
            "results": [{"candidate_id": "nope"}],
        }
    )
    with pytest.raises(ReviewIntegrityError, match="does not match the schema"):
        load_visual_proposals(raw, [_binding(name, pilot=True)])
    assert not (tmp_path / "visual_proposals.json").exists()
