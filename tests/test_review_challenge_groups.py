"""A2: structural grouping of selected challenge disputes.

Synthetic fixtures only. No database, network, provider, or archive API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    NOW,
    _close_challenge,
    _controller,
    _deliver_first_pass,
    _snapshot,
    _write_output,
)
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review import (
    assemble,
    controller as controller_module,
    spans,
)
from transcription_review_agent.review.adjudicate import (
    EvidenceBindingError,
    check_evidence_binding,
)
from transcription_review_agent.review.challenge_groups import (
    group_challenge_disputes,
    member_evidence_scope,
    required_decision_ids,
)
from transcription_review_agent.review.compare import (
    challengeable_disputes,
    compare_transcriptions,
)
from transcription_review_agent.review.controller import (
    AnonymousChallengeFile,
    AnonymousChallengeItem,
    ReviewController,
)
from transcription_review_agent.review.crop import assert_crop_contains_target
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewOutputRejected,
)
from transcription_review_agent.review.models import (
    ApparatusNote,
    BBox,
    ChallengeDecision,
    ChallengeOutput,
    ComparisonFile,
    CoverageReport,
    CropRecord,
    Dispute,
    DisputeCategory,
    EvidenceReference,
    FirstPassOutput,
    LineReading,
    PageTranscription,
    RegionAlignment,
    TextRegion,
    WorkRequest,
)
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.runtime import (
    WorkPublishedEntry,
    provider_runtime,
)

GPT_TEXT = {
    "page_001": ["Alpha one", "Bravo two", "Charlie three"],
    "page_002": ["Echo five"],
}
GROK_TEXT = {
    "page_001": ["Alfa one", "Bravo twa", "Charly three"],
    "page_002": ["Echo fyve"],
}
# Only the GPT side records a marginal note on r1. That is a second substantive
# dispute on the same aligned span as the r1 wording dispute.
GPT_NOTES = [ApparatusNote(region_id="r1", text="one")]


def _page(
    page_id: str,
    order: int,
    texts: list[str],
    *,
    notes: list[ApparatusNote] | None = None,
    no_bbox: tuple[int, ...] = (),
) -> PageTranscription:
    regions = [
        TextRegion(
            region_id=f"r{index + 1}",
            order=index + 1,
            lines=[LineReading(line_id=f"r{index + 1}-l1", text=text)],
            bbox=None
            if index in no_bbox
            else BBox(x=1 + 3 * index, y=1, width=3, height=3),
        )
        for index, text in enumerate(texts)
    ]
    return PageTranscription(
        page_id=page_id,
        reading_order=order,
        diplomatic_transcription="\n".join(texts),
        regions=regions,
        unreadable_spans=[],
        alternative_readings=[],
        attention_items=[],
        coverage=CoverageReport(
            coverage_complete=True,
            unresolved_text_present=False,
            examined_region_ids=[region.region_id for region in regions],
        ),
        cancellations=[],
        insertions=[],
        marginal_notes=[] if notes is None else notes,
        inaccessible_or_ambiguous=[],
    )


def _pages(provider: str, no_bbox: tuple[int, ...] = ()) -> list[PageTranscription]:
    text = GPT_TEXT if provider == "gpt" else GROK_TEXT
    notes = GPT_NOTES if provider == "gpt" else None
    return [
        _page("page_001", 1, text["page_001"], notes=notes, no_bbox=no_bbox),
        _page("page_002", 2, text["page_002"]),
    ]


def _first_pass(provider: str, pages: list[PageTranscription]) -> FirstPassOutput:
    return FirstPassOutput(
        provider=provider,  # type: ignore[arg-type]
        provider_interface="GPT_WORK" if provider == "gpt" else "GROK",
        task_id="task",
        job_id="job",
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        attempt=1,
        source_bundle_sha256="a" * 64,
        pages=pages,
    )


def _comparison(no_bbox: tuple[int, ...] = ()) -> ComparisonFile:
    return compare_transcriptions(
        _first_pass("gpt", _pages("gpt", no_bbox)),
        _first_pass("grok", _pages("grok", no_bbox)),
        gpt_sha256="b" * 64,
        grok_sha256="c" * 64,
    )


def _labels(comparison: ComparisonFile) -> dict[str, str]:
    """Stable names for the five substantive disputes of the fixture."""

    names: dict[str, str] = {}
    for item in comparison.disputes:
        if not item.substantive:
            continue
        if item.page_id == "page_002":
            names["p2"] = item.dispute_id
        elif item.category is DisputeCategory.CANCELLATION_OR_INSERTION:
            names["note_r1"] = item.dispute_id
        else:
            names[f"word_{item.region_id}"] = item.dispute_id
    assert set(names) == {"word_r1", "note_r1", "word_r2", "word_r3", "p2"}
    return names


def _window_of(
    comparison: ComparisonFile, no_bbox_regions: frozenset[str] = frozenset()
):
    def window(dispute: Dispute):
        region = dispute.gpt_region_ids[0]
        if dispute.page_id == "page_001" and region in no_bbox_regions:
            return BBox(x=0, y=0, width=20, height=20), "FULL_PAGE_FALLBACK"
        index = int(region[1:]) - 1
        return BBox(x=1 + 3 * index, y=1, width=3, height=3), "TARGETED"

    return window


def _select(
    comparison: ComparisonFile, names: dict[str, str], *keys: str
) -> list[Dispute]:
    wanted = {names[key] for key in keys}
    return [
        item for item in challengeable_disputes(comparison) if item.dispute_id in wanted
    ]


# --- the shared span helper ---------------------------------------------------


def test_final_assembly_uses_the_shared_span_helper_and_has_no_duplicate() -> None:
    assert assemble.same_span is spans.same_span
    assert not hasattr(assemble, "_same_span")
    assert not hasattr(assemble, "_note_words_in")
    assert not hasattr(assemble, "_words")


def test_same_span_rule_is_the_assembly_rule() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    by_id = {item.dispute_id: item for item in comparison.disputes}
    first = next(
        item
        for item in comparison.alignments
        if item.gpt_region_ids == ["r1"] and item.page_id == "page_001"
    )
    other = next(
        item for item in comparison.alignments if item.gpt_region_ids == ["r2"]
    )
    assert spans.same_span(by_id[names["word_r1"]], first)
    assert spans.same_span(by_id[names["note_r1"]], first)
    assert not spans.same_span(by_id[names["word_r2"]], first)
    assert not spans.same_span(by_id[names["word_r1"]], other)


def test_link_alignment_is_structural_and_stays_on_the_page() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    by_id = {item.dispute_id: item for item in comparison.disputes}
    word = spans.link_alignment(by_id[names["word_r1"]], comparison.alignments)
    note = spans.link_alignment(by_id[names["note_r1"]], comparison.alignments)
    assert word is not None and note is not None
    assert word.alignment_id == note.alignment_id
    page_two = spans.link_alignment(by_id[names["p2"]], comparison.alignments)
    assert page_two is not None and page_two.page_id == "page_002"
    # Same region IDs on two pages are still two different spans.
    assert page_two.alignment_id != word.alignment_id
    # A dispute never links to an alignment of another page.
    assert spans.link_alignment(by_id[names["p2"]], [word]) is None


def test_link_alignment_refuses_ambiguity_and_reading_order() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    word = next(
        item for item in comparison.disputes if item.dispute_id == names["word_r1"]
    )
    first = next(
        item
        for item in comparison.alignments
        if item.page_id == "page_001" and item.gpt_region_ids == ["r1"]
    )
    twin = first.model_copy(
        update={"alignment_id": "aln_twin", "gpt_text": "other", "grok_text": "text"}
    )
    # Two spans carry identical region IDs: the exact text decides.
    assert spans.link_alignment(word, [twin, first]) == first
    # Neither text matches: no link rather than a guess.
    unmatched = word.model_copy(update={"gpt_text": "zzz", "grok_text": "yyy"})
    assert (
        spans.link_alignment(
            unmatched,
            [twin, first.model_copy(update={"alignment_id": "aln_b", "gpt_text": "q"})],
        )
        is None
    )
    ordering = word.model_copy(update={"category": DisputeCategory.READING_ORDER})
    assert spans.link_alignment(ordering, comparison.alignments) is None


def test_legacy_dispute_without_region_lists_links_only_when_unique() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    word = next(
        item for item in comparison.disputes if item.dispute_id == names["word_r2"]
    )
    legacy = word.model_copy(update={"gpt_region_ids": [], "grok_region_ids": []})
    linked = spans.link_alignment(legacy, comparison.alignments)
    assert linked is not None and linked.gpt_region_ids == ["r2"]
    # The same region split over two spans is ambiguous, so it is not linked.
    split = [
        RegionAlignment(
            alignment_id=f"aln_{index}",
            page_id="page_001",
            kind="matched",
            gpt_region_ids=["r2"],
            grok_region_ids=[f"x{index}"],
            gpt_text="a",
            grok_text="b",
            sequence=index,
        )
        for index in range(2)
    ]
    assert spans.link_alignment(legacy, split) is None
    assert spans.span_key(legacy, split).basis == "UNLINKED"


# --- pure grouping ------------------------------------------------------------


def test_two_selected_disputes_on_one_span_form_one_group() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    selected = _select(comparison, names, "word_r1", "note_r1")
    groups = group_challenge_disputes(
        selected, comparison.alignments, _window_of(comparison)
    )
    assert len(groups) == 1
    group = groups[0]
    assert set(group.dispute_ids) == {names["word_r1"], names["note_r1"]}
    assert group.shares_crop and group.basis == "ALIGNED_SPAN"
    assert group.crop_id == f"crop_{group.group_id}"
    assert group.group_id.startswith("grp_") and len(group.group_id) == 20
    assert required_decision_ids(groups) == list(group.dispute_ids)


def test_different_spans_and_pages_stay_separate() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    selected = _select(
        comparison, names, "word_r1", "note_r1", "word_r2", "word_r3", "p2"
    )
    groups = group_challenge_disputes(
        selected, comparison.alignments, _window_of(comparison)
    )
    assert len(groups) == 4
    assert sorted(len(item.dispute_ids) for item in groups) == [1, 1, 1, 2]
    # Nothing crosses a page: every group has one page, and the page-two group is alone.
    by_page = {item.page_id: item for item in groups if item.page_id == "page_002"}
    assert by_page["page_002"].dispute_ids == (names["p2"],)
    # Disputes on neighbouring spans of one page do not merge.
    singles = [
        item for item in groups if not item.shares_crop and item.page_id == "page_001"
    ]
    assert {item.dispute_ids[0] for item in singles} == {
        names["word_r2"],
        names["word_r3"],
    }
    # Every selected ID is in exactly one group.
    flat = required_decision_ids(groups)
    assert len(flat) == len(set(flat)) == 5


def test_selecting_one_member_does_not_add_its_neighbor() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    both = group_challenge_disputes(
        _select(comparison, names, "word_r1", "note_r1"),
        comparison.alignments,
        _window_of(comparison),
    )[0]
    alone = group_challenge_disputes(
        _select(comparison, names, "word_r1"),
        comparison.alignments,
        _window_of(comparison),
    )
    assert len(alone) == 1
    assert alone[0].dispute_ids == (names["word_r1"],)
    assert names["note_r1"] not in required_decision_ids(alone)
    # The same span keeps the same group ID whichever neighbours were selected.
    assert alone[0].group_id == both.group_id
    # A group of one keeps the crop name an ungrouped dispute always had.
    assert alone[0].crop_id == f"crop_{names['word_r1']}"
    assert both.crop_id != alone[0].crop_id


def test_group_ids_and_order_are_deterministic() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    selected = _select(
        comparison, names, "word_r1", "note_r1", "word_r2", "word_r3", "p2"
    )
    first = group_challenge_disputes(
        selected, comparison.alignments, _window_of(comparison)
    )
    second = group_challenge_disputes(
        list(selected), list(comparison.alignments), _window_of(comparison)
    )
    assert first == second
    # Groups follow the comparison's dispute order, by first member.
    order = [item.dispute_id for item in challengeable_disputes(comparison)]
    firsts = [order.index(item.dispute_ids[0]) for item in first]
    assert firsts == sorted(firsts)
    for group in first:
        positions = [order.index(item) for item in group.dispute_ids]
        assert positions == sorted(positions)
    # IDs depend on structure and window, not on dispute text or provider names.
    assert len({item.group_id for item in first}) == len(first)
    # An unrelated selection change leaves the other groups' IDs alone.
    fewer = group_challenge_disputes(
        _select(comparison, names, "word_r2", "p2"),
        comparison.alignments,
        _window_of(comparison),
    )
    ids = {item.dispute_ids[0]: item.group_id for item in first}
    assert {item.dispute_ids[0]: item.group_id for item in fewer} == {
        names["word_r2"]: ids[names["word_r2"]],
        names["p2"]: ids[names["p2"]],
    }


def test_a_duplicate_selection_is_refused() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    selected = _select(comparison, names, "word_r1")
    with pytest.raises(ValueError, match="more than once"):
        group_challenge_disputes(
            selected + selected, comparison.alignments, _window_of(comparison)
        )


def test_disputes_with_different_windows_are_split_not_merged() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    selected = _select(comparison, names, "word_r1", "note_r1")

    def window(dispute: Dispute):
        if dispute.dispute_id == names["note_r1"]:
            return BBox(x=0, y=0, width=20, height=20), "FULL_PAGE_FALLBACK"
        return BBox(x=1, y=1, width=3, height=3), "TARGETED"

    groups = group_challenge_disputes(selected, comparison.alignments, window)
    assert len(groups) == 2
    assert {item.window_kind for item in groups} == {"TARGETED", "FULL_PAGE_FALLBACK"}
    assert all(not item.shares_crop for item in groups)


def test_full_page_fallback_group_is_labeled_and_cited_as_fallback() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    selected = _select(comparison, names, "word_r1", "note_r1")
    groups = group_challenge_disputes(
        selected, comparison.alignments, _window_of(comparison, frozenset({"r1"}))
    )
    assert len(groups) == 1
    group = groups[0]
    assert group.window_kind == "FULL_PAGE_FALLBACK"
    assert group.evidence_kind == "FULL_PAGE_FALLBACK"
    scope = member_evidence_scope(group, names["word_r1"])
    assert ("FULL_PAGE_FALLBACK", group.crop_id) in scope.allowed
    assert ("CROP", group.crop_id) not in scope.allowed


def test_reading_order_dispute_is_a_page_level_group_of_its_own() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    order = Dispute(
        dispute_id="dsp_order",
        category=DisputeCategory.READING_ORDER,
        page_id="page_001",
        gpt_region_ids=["r1", "r2", "r3"],
        grok_region_ids=["r3", "r2", "r1"],
        gpt_text="r1,r2,r3",
        grok_text="r3,r2,r1",
        substantive=True,
    )
    other_order = order.model_copy(update={"dispute_id": "dsp_order_two"})
    selected = [order, other_order, *_select(comparison, names, "word_r1")]

    def window(dispute: Dispute):
        if dispute.category is DisputeCategory.READING_ORDER:
            return BBox(x=1, y=1, width=9, height=3), "TARGETED"
        return BBox(x=1, y=1, width=3, height=3), "TARGETED"

    groups = group_challenge_disputes(selected, comparison.alignments, window)
    assert [item.basis for item in groups] == ["PAGE_READING_ORDER", "ALIGNED_SPAN"]
    assert groups[0].dispute_ids == ("dsp_order", "dsp_order_two")
    # A span dispute on the same page is not pulled into the page-level group.
    assert names["word_r1"] not in groups[0].dispute_ids
    assert groups[0].alignment_id is None


def test_unlinked_dispute_is_never_grouped() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    word = next(
        item for item in comparison.disputes if item.dispute_id == names["word_r1"]
    )
    stray_one = word.model_copy(
        update={
            "dispute_id": "dsp_a",
            "gpt_region_ids": ["r9"],
            "grok_region_ids": ["r9"],
        }
    )
    stray_two = word.model_copy(
        update={
            "dispute_id": "dsp_b",
            "gpt_region_ids": ["r9"],
            "grok_region_ids": ["r9"],
        }
    )
    groups = group_challenge_disputes(
        [stray_one, stray_two],
        comparison.alignments,
        lambda _d: (BBox(x=1, y=1, width=3, height=3), "TARGETED"),
    )
    assert [item.basis for item in groups] == ["UNLINKED", "UNLINKED"]
    assert [item.dispute_ids for item in groups] == [("dsp_a",), ("dsp_b",)]


def test_group_evidence_scope_supports_the_a3_binding_rule() -> None:
    comparison = _comparison()
    names = _labels(comparison)
    selected = _select(comparison, names, "word_r1", "note_r1", "word_r2")
    groups = group_challenge_disputes(
        selected, comparison.alignments, _window_of(comparison)
    )
    shared = next(item for item in groups if item.shares_crop)
    single = next(item for item in groups if not item.shares_crop)
    scope = member_evidence_scope(shared, names["note_r1"])
    assert scope.dispute_id == names["note_r1"] and scope.page_id == "page_001"
    assert scope.allowed == frozenset(
        {("GROUP", shared.group_id), ("CROP", shared.crop_id)}
    )
    with pytest.raises(EvidenceBindingError) as raised:
        member_evidence_scope(shared, names["word_r2"])
    assert raised.value.code == "SCOPE_MISMATCH"

    def decision(dispute_id: str, kind: str, evidence_id: str) -> ChallengeDecision:
        return ChallengeDecision(
            dispute_id=dispute_id,
            decision="CANDIDATE_X",
            visual_evidence="The crop shows candidate X.",
            evidence_ref=EvidenceReference(
                kind=kind, evidence_id=evidence_id, page_id="page_001"
            ),  # type: ignore[arg-type]
            evidence_assessment="SUPPORTS_DECISION",
        )

    for kind, evidence_id in (("GROUP", shared.group_id), ("CROP", shared.crop_id)):
        ok = check_evidence_binding(
            decision(names["word_r1"], kind, evidence_id),
            scope.__class__(names["word_r1"], "page_001", scope.allowed),
            provider="gpt",
        )
        assert (ok.kind, ok.evidence_id) == (kind, evidence_id)
    # The other span's crop is not evidence for this dispute.
    with pytest.raises(EvidenceBindingError) as other:
        check_evidence_binding(
            decision(names["word_r1"], "CROP", single.crop_id),
            scope.__class__(names["word_r1"], "page_001", scope.allowed),
            provider="gpt",
        )
    assert other.value.code == "REFERENCE_NOT_IN_DISPUTE"


# --- controller: grouped challenge input ---------------------------------------


def _document(controller: ReviewController):
    return controller._require_document(BATCH_ID, DOCUMENT_ID)[1]


def _root(controller: ReviewController) -> Path:
    return document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)


def _setup(tmp_path: Path, no_bbox: tuple[int, ...] = ()):
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    for provider in ("gpt", "grok"):
        _deliver_first_pass(controller, provider, _pages(provider, no_bbox))
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    return controller, comparison, _labels(comparison)


def _publish(controller: ReviewController, provider: str, **kwargs) -> Path:
    return controller.publish(
        BATCH_ID, DOCUMENT_ID, provider, stage="challenge", **kwargs
    )


def _answer(
    controller,
    request_path: Path,
    ids: list[str],
    *,
    per_id: dict[str, str] | None = None,
):
    request = WorkRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    challenge = ChallengeOutput(
        provider=request.provider,
        provider_interface=request.provider_interface,
        task_id=request.task_id,
        job_id=request.job_id,
        batch_id=request.batch_id,
        document_id=request.document_id,
        attempt=request.attempt,
        source_bundle_sha256=request.source_bundle_sha256,
        round_index=request.round_index,
        decisions=[
            ChallengeDecision(
                dispute_id=dispute_id,
                decision=(per_id or {}).get(dispute_id, "CANDIDATE_X"),  # type: ignore[arg-type]
                visual_evidence="The crop shows the first candidate.",
            )
            for dispute_id in ids
        ],
    )
    return _write_output(controller, request_path, challenge.model_dump(mode="json"))


def _input(controller: ReviewController, round_index: int = 1) -> Path:
    return _root(controller) / f"challenge_round_{round_index:02d}" / "input"


def _published(controller: ReviewController) -> AnonymousChallengeFile:
    return AnonymousChallengeFile.model_validate_json(
        (_input(controller) / "disputes.json").read_text(encoding="utf-8")
    )


def test_two_selected_disputes_share_one_group_one_crop_and_two_decisions(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    chosen = [names["word_r1"], names["note_r1"]]
    gpt_request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    grok_request = _publish(controller, "grok")

    published = _published(controller)
    assert published.schema_version == "review-challenge-input-v3"
    assert published.challenge_contract == "EVIDENCE_BOUND_V1"
    assert len(published.groups or []) == 1
    group = (published.groups or [])[0]
    assert set(group.dispute_ids) == set(chosen)
    assert published.required_decision_ids == [
        item.dispute_id for item in published.disputes
    ]
    assert set(published.required_decision_ids or []) == set(chosen)
    assert group.decisions_independent is True

    # One shared visual evidence source for both disputes.
    assert {item.crop_id for item in published.disputes} == {group.crop_id}
    assert {item.group_id for item in published.disputes} == {group.group_id}
    crops = sorted(path.name for path in (_input(controller) / "crops").iterdir())
    assert crops == [f"{group.crop_id}.json", f"{group.crop_id}.png"]
    for provider in ("gpt", "grok"):
        task_crops = sorted(
            path.name
            for path in (
                _root(controller)
                / "challenge_round_01"
                / "tasks"
                / provider
                / "input"
                / "crops"
            ).iterdir()
        )
        assert task_crops == crops

    # The crop record names the group and its disputes, not one dispute.
    crop_path = _input(controller) / "crops" / f"{group.crop_id}.json"
    crop_payload = json.loads(crop_path.read_text(encoding="utf-8"))
    assert "dispute_id" not in crop_payload
    assert crop_payload["group_id"] == group.group_id
    assert crop_payload["dispute_ids"] == list(group.dispute_ids)
    crop = CropRecord.model_validate(crop_payload)
    assert_crop_contains_target(crop)
    assert crop.window_kind == "TARGETED" and crop.scale == 1
    assert (
        crop.represents_new_visual_detail is False and crop.full_page_retained is True
    )

    # Source hashes and evidence references travel with the group.
    page = next(
        item for item in _document(controller).pages if item.page_id == "page_001"
    )
    assert group.page_sha256 == page.sha256 == crop.page_sha256
    png = (_input(controller) / "crops" / f"{group.crop_id}.png").read_bytes()
    assert group.crop_sha256 == crop.crop_sha256 == sha256_bytes(png)
    assert [
        (item.kind, item.evidence_id, item.page_id) for item in group.evidence_refs
    ] == [
        ("GROUP", group.group_id, "page_001"),
        ("CROP", group.crop_id, "page_001"),
    ]

    # Two decisions are required, one per dispute; grouping does not merge them.
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == chosen
    _answer(controller, gpt_request, chosen, per_id={chosen[1]: "CANDIDATE_Y"})
    controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    _answer(
        controller,
        grok_request,
        list(reversed(chosen)),
        per_id={chosen[1]: "CANDIDATE_Y"},
    )
    controller.accept(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    decisions = {
        item.dispute_id: item for item in _document(controller).reconciled_decisions
    }
    assert set(decisions) == set(chosen)
    assert decisions[chosen[0]].decision == "CANDIDATE_X"
    assert decisions[chosen[1]].decision == "CANDIDATE_Y"


def test_selected_ids_equal_required_decision_ids_whatever_the_grouping(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    everything = [item.dispute_id for item in challengeable_disputes(comparison)]
    _publish(controller, "gpt", selected_dispute_ids=everything)
    published = _published(controller)
    assert set(published.required_decision_ids or []) == set(everything)
    assert len(published.required_decision_ids or []) == len(everything)
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == everything
    grouped_ids = [
        item for group in published.groups or [] for item in group.dispute_ids
    ]
    assert sorted(grouped_ids) == sorted(everything)


def test_different_spans_and_pages_are_separate_groups_with_separate_crops(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    everything = [item.dispute_id for item in challengeable_disputes(comparison)]
    _publish(controller, "gpt", selected_dispute_ids=everything)
    published = _published(controller)
    groups = published.groups or []
    assert len(groups) == 4
    assert sorted(len(item.dispute_ids) for item in groups) == [1, 1, 1, 2]
    assert (
        len({item.group_id for item in groups})
        == len({item.crop_id for item in groups})
        == 4
    )
    page_two = next(item for item in groups if item.page_id == "page_002")
    assert page_two.dispute_ids == [names["p2"]]
    assert page_two.crop_id == f"crop_{names['p2']}"
    crop_files = sorted(
        path.name for path in (_input(controller) / "crops").glob("*.json")
    )
    assert crop_files == sorted(f"{item.crop_id}.json" for item in groups)


def test_selecting_one_member_does_not_select_or_show_its_neighbor(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    _publish(controller, "gpt", selected_dispute_ids=[names["word_r1"]])
    published = _published(controller)
    assert [item.dispute_id for item in published.disputes] == [names["word_r1"]]
    assert published.required_decision_ids == [names["word_r1"]]
    group = (published.groups or [])[0]
    assert group.dispute_ids == [names["word_r1"]]
    assert group.crop_id == f"crop_{names['word_r1']}"
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == [
        names["word_r1"]
    ]
    text = (_input(controller) / "disputes.json").read_text(encoding="utf-8")
    assert names["note_r1"] not in text
    crop_payload = json.loads(
        (_input(controller) / "crops" / f"{group.crop_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert crop_payload["dispute_id"] == names["word_r1"]
    assert "group_id" not in crop_payload and "dispute_ids" not in crop_payload
    # The group keeps its identity when the neighbor is selected later.
    groups_both = group_challenge_disputes(
        _select(comparison, names, "word_r1", "note_r1"),
        comparison.alignments,
        _window_of(comparison),
    )
    assert groups_both[0].group_id == group.group_id


def test_group_ids_order_and_bytes_are_stable_across_runs(tmp_path: Path) -> None:
    runs = []
    for name in ("one", "two"):
        controller, comparison, names = _setup(tmp_path / name)
        everything = [item.dispute_id for item in challengeable_disputes(comparison)]
        _publish(controller, "gpt", selected_dispute_ids=list(reversed(everything)))
        _publish(controller, "grok")
        runs.append(_snapshot(_input(controller)))
    assert runs[0] == runs[1]
    assert "disputes.json" in runs[0]


def test_recomputed_groups_match_the_published_groups(tmp_path: Path) -> None:
    """A3 can rebuild the evidence scope of any dispute from comparison and selection alone."""

    controller, comparison, names = _setup(tmp_path)
    everything = [item.dispute_id for item in challengeable_disputes(comparison)]
    _publish(controller, "gpt", selected_dispute_ids=everything)
    published = _published(controller)
    gpt = controller_module._load_output(
        _root(controller), _document(controller), "gpt"
    )
    grok = controller_module._load_output(
        _root(controller), _document(controller), "grok"
    )
    assert gpt is not None and grok is not None
    pages = {item.page_id: item for item in _document(controller).pages}

    def window(dispute: Dispute):
        page = pages[dispute.page_id]
        return controller_module._target_window(
            dispute, gpt, grok, page.width, page.height
        )

    recomputed = group_challenge_disputes(
        challengeable_disputes(comparison), comparison.alignments, window
    )
    assert [
        (item.group_id, list(item.dispute_ids), item.crop_id) for item in recomputed
    ] == [
        (item.group_id, item.dispute_ids, item.crop_id)
        for item in published.groups or []
    ]
    for group in recomputed:
        for dispute_id in group.dispute_ids:
            scope = member_evidence_scope(group, dispute_id)
            assert ("GROUP", group.group_id) in scope.allowed


def test_provider_output_must_decide_every_member_by_its_exact_id(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    chosen = [names["word_r1"], names["note_r1"]]
    gpt_request = _publish(controller, "gpt", selected_dispute_ids=chosen)

    # One decision for a two-member group is a missing decision.
    _answer(controller, gpt_request, [chosen[0]])
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")


def test_group_or_crop_ids_are_not_decision_ids(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    chosen = [names["word_r1"], names["note_r1"]]
    request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    group = (_published(controller).groups or [])[0]
    _answer(controller, request, [group.group_id])
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")


def test_neighbor_or_extra_decision_is_rejected(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    request = _publish(controller, "gpt", selected_dispute_ids=[names["word_r1"]])
    _answer(controller, request, [names["word_r1"], names["note_r1"]])
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")


def test_accepted_output_hash_and_ids_are_exact(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    chosen = [names["word_r1"], names["note_r1"]]
    requests = {
        provider: _publish(controller, provider, selected_dispute_ids=chosen)
        for provider in ("gpt", "grok")
    }
    for provider, request in requests.items():
        written = _answer(controller, request, chosen)
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
        runtime = provider_runtime(
            _document(controller), stage="challenge", provider=provider, round_index=1
        )
        assert runtime.output_sha256 == sha256_bytes(written.read_bytes())
        accepted = ChallengeOutput.model_validate_json(written.read_bytes())
        assert [item.dispute_id for item in accepted.decisions] == chosen
    assert [
        item.dispute_id for item in _document(controller).reconciled_decisions
    ] == chosen


def test_candidates_are_exact_and_provider_anonymous(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    everything = [item.dispute_id for item in challengeable_disputes(comparison)]
    _publish(controller, "gpt", selected_dispute_ids=everything)
    _publish(controller, "grok")
    by_id = {item.dispute_id: item for item in comparison.disputes}
    published = _published(controller)
    for item in published.disputes:
        assert item.candidate_x == by_id[item.dispute_id].gpt_text
        assert item.candidate_y == by_id[item.dispute_id].grok_text
    # The note dispute has an empty Y side. It stays empty, not "missing".
    note = next(
        item for item in published.disputes if item.dispute_id == names["note_r1"]
    )
    assert (note.candidate_x, note.candidate_y) == ("one", "")
    # No provider name in any provider-visible file or key of the challenge input.
    for provider in ("gpt", "grok"):
        task = _root(controller) / "challenge_round_01" / "tasks" / provider
        for path in (task / "input").rglob("*.json"):
            payload = path.read_text(encoding="utf-8").lower()
            for token in ("gpt", "grok", "gpt_work"):
                assert f'"{token}' not in payload, (path.name, token)
                assert f"{token}_" not in payload, (path.name, token)
    raw = (_input(controller) / "disputes.json").read_text(encoding="utf-8")
    keys = set()

    def walk(value) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(json.loads(raw))
    assert not any("gpt" in key.lower() or "grok" in key.lower() for key in keys)


def test_group_crop_covers_every_member_and_page_pixels_are_unchanged(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    chosen = [names["word_r1"], names["note_r1"]]
    before = _snapshot(tmp_path / "batches")
    _publish(controller, "gpt", selected_dispute_ids=chosen)
    group = (_published(controller).groups or [])[0]
    crop = CropRecord.model_validate_json(
        (_input(controller) / "crops" / f"{group.crop_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert crop.crop_bbox.contains(crop.target_bbox)
    assert crop.crop_bbox.area > crop.target_bbox.area
    assert crop.derived_png_sha256 is None or len(crop.derived_png_sha256) == 64
    page = (_input(controller) / "page_001.png").read_bytes()
    assert page == (_root(controller) / "input" / "page_001.png").read_bytes()
    assert sha256_bytes(page) == group.page_sha256
    assert _snapshot(tmp_path / "batches") == before


def test_shared_full_page_fallback_is_labeled(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path, no_bbox=(0,))
    chosen = [names["word_r1"], names["note_r1"]]
    request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    published = _published(controller)
    group = (published.groups or [])[0]
    assert set(group.dispute_ids) == set(chosen)
    assert group.window_kind == "FULL_PAGE_FALLBACK"
    assert [(item.kind, item.evidence_id) for item in group.evidence_refs] == [
        ("GROUP", group.group_id),
        ("FULL_PAGE_FALLBACK", group.crop_id),
    ]
    crop_payload = json.loads(
        (_input(controller) / "crops" / f"{group.crop_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert crop_payload["window_kind"] == "FULL_PAGE_FALLBACK"
    assert crop_payload["crop_bbox"] == {"x": 0, "y": 0, "width": 20, "height": 20}
    assert (crop_payload["crop_png_width"], crop_payload["crop_png_height"]) == (20, 20)
    assert json.loads(request.read_text(encoding="utf-8"))["stage"] == "challenge"
    text = (
        _root(controller)
        / "challenge_round_01"
        / "tasks"
        / "gpt"
        / "instructions_gpt.txt"
    ).read_text(encoding="utf-8")
    assert "FULL_PAGE_FALLBACK means no region box was usable" in text


def test_grouped_instructions_extend_the_legacy_text(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    _publish(
        controller, "gpt", selected_dispute_ids=[names["word_r1"], names["note_r1"]]
    )
    text = (
        _root(controller)
        / "challenge_round_01"
        / "tasks"
        / "gpt"
        / "instructions_gpt.txt"
    ).read_text(encoding="utf-8")
    legacy = controller_module._instructions("challenge")
    assert text.startswith(legacy)
    assert "needs its own decision" in text
    assert "gpt" not in text.lower().replace("gpt_work", "")  # provider-anonymous


def test_replay_and_sealed_bytes_are_unchanged_by_grouping(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    chosen = [names["word_r1"], names["note_r1"]]
    sealed_before = _snapshot(tmp_path / "batches")
    gpt_request = _publish(controller, "gpt", selected_dispute_ids=chosen)
    checkpoint = (
        controller.reviews_dir / f"batch_{BATCH_ID}" / "controller" / "state.json"
    )
    live = checkpoint.read_text(encoding="utf-8")
    ledger = (checkpoint.parent / "ledger.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in ledger.splitlines()]
    published_events = [
        item
        for item in events
        if item["event_type"] == "work_published" and item["stage"] == "challenge"
    ]
    # Grouping adds nothing to the ledger: it is derived from comparison and selection.
    assert not any("group" in key for event in published_events for key in event)

    checkpoint.unlink()
    resumed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    assert _document(resumed).challenge_rounds[0].selected_dispute_ids == chosen
    assert checkpoint.read_text(encoding="utf-8") == live
    grok_request = _publish(resumed, "grok")
    assert _published(resumed).required_decision_ids == chosen
    for provider, request in (("gpt", gpt_request), ("grok", grok_request)):
        _answer(resumed, request, chosen)
        resumed.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    final = checkpoint.read_text(encoding="utf-8")
    checkpoint.unlink()
    replayed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=lambda: NOW,
    )
    assert [
        item.dispute_id for item in _document(replayed).reconciled_decisions
    ] == chosen
    assert checkpoint.read_text(encoding="utf-8") == final
    assert _snapshot(tmp_path / "batches") == sealed_before


def test_grouping_does_not_change_the_report_or_final_text_for_a_full_round(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    everything = [item.dispute_id for item in challengeable_disputes(comparison)]
    requests = {
        provider: _publish(controller, provider, selected_dispute_ids=everything)
        for provider in ("gpt", "grok")
    }
    for provider, request in requests.items():
        _answer(controller, request, everything)
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    _close_challenge(controller)
    paths = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    assert [item["dispute_id"] for item in report["decisions"]] == everything
    assert report["production_writeback"] is False


# --- legacy compatibility --------------------------------------------------------


LEGACY_ITEM = {
    "dispute_id": "dsp_0123456789abcdef",
    "page_id": "page_001",
    "crop_id": "crop_dsp_0123456789abcdef",
    "candidate_x": "Alpha",
    "candidate_y": "Alfa",
    "request_visual_evidence": True,
}


def test_previously_published_ungrouped_input_still_parses_and_reserializes_exactly() -> (
    None
):
    raw = (
        json.dumps(
            {"schema_version": "review-challenge-input-v1", "disputes": [LEGACY_ITEM]},
            indent=2,
        )
        + "\n"
    )
    parsed = AnonymousChallengeFile.model_validate_json(raw)
    assert parsed.groups is None and parsed.required_decision_ids is None
    assert parsed.disputes[0].group_id is None
    from transcription_review_agent.bundling.publish import canonical_json

    assert canonical_json(parsed) == raw


def test_previously_published_single_dispute_crop_record_is_byte_stable(
    tmp_path: Path,
) -> None:
    from transcription_review_agent.bundling.publish import canonical_json
    from transcription_review_agent.review.crop import build_crop

    record = build_crop(
        crop_id="crop_dsp_old",
        dispute_id="dsp_old",
        page_id="page_001",
        page_sha256="ab" * 32,
        page_width=20,
        page_height=20,
        target=BBox(x=4, y=4, width=4, height=4),
    )
    text = canonical_json(record)
    payload = json.loads(text)
    assert payload["dispute_id"] == "dsp_old"
    assert "group_id" not in payload and "dispute_ids" not in payload
    assert list(payload)[:3] == ["crop_id", "dispute_id", "page_id"]
    assert canonical_json(CropRecord.model_validate_json(text)) == text


def test_ungrouped_layout_is_not_regrouped_for_a_round_published_before_a2(
    tmp_path: Path,
) -> None:
    """A round that an earlier version published keeps its exact ungrouped bytes."""

    controller, comparison, names = _setup(tmp_path)
    wanted = {names["word_r1"], names["note_r1"], names["word_r2"]}
    chosen = [
        item.dispute_id
        for item in challengeable_disputes(comparison)
        if item.dispute_id in wanted
    ]
    # Simulate the earlier version for the first provider: an ungrouped task
    # and a ledger publication with no round contract.
    document = _document(controller)
    root = _root(controller)
    gpt = controller_module._load_output(root, document, "gpt")
    grok = controller_module._load_output(root, document, "grok")
    assert gpt is not None and grok is not None
    controller_module._write_challenge_input(
        root,
        document,
        comparison,
        gpt,
        grok,
        1,
        selection=chosen,
        layout="UNGROUPED",
        challenge_contract=None,
    )
    request = controller_module._work_request(
        BATCH_ID,
        document,
        provider="gpt",
        stage="challenge",
        round_index=1,
        attempt=1,
    )
    controller_module._prepare_bundle(
        root,
        document,
        request,
        controller_module._archive_text(controller.batches_dir, BATCH_ID, DOCUMENT_ID),
        comparison=comparison,
        gpt=gpt,
        grok=grok,
        challenge_selection=chosen,
        challenge_contract=None,
        challenge_already_published=True,
    )
    state = controller._store(BATCH_ID).load()
    assert state is not None
    controller._store(BATCH_ID).commit(
        WorkPublishedEntry(
            sequence=state.applied_ledger_sequence + 1,
            occurred_at=NOW,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            provider="gpt",
            stage="challenge",
            round_index=1,
            attempt=1,
            task_id=request.task_id,
            job_id=request.job_id,
            output_relative_path=request.output_relative_path,
            selected_dispute_ids=chosen,
            selection_comparison_sha256=document.comparison_sha256,
        )
    )
    gpt_request = root / "challenge_round_01" / "tasks" / "gpt" / "work_request.json"
    legacy_bytes = _snapshot(_input(controller))
    legacy = _published(controller)
    assert legacy.schema_version == "review-challenge-input-v1"
    assert legacy.groups is None and legacy.required_decision_ids is None
    assert [item.dispute_id for item in legacy.disputes] == chosen
    assert [item.crop_id for item in legacy.disputes] == [
        f"crop_{item}" for item in chosen
    ]
    assert all(item.group_id is None for item in legacy.disputes)
    legacy_text = (
        _root(controller)
        / "challenge_round_01"
        / "tasks"
        / "gpt"
        / "instructions_gpt.txt"
    ).read_text(encoding="utf-8")
    assert legacy_text == controller_module._instructions("challenge")

    # The current code publishes the second provider without regrouping.
    grok_request = _publish(controller, "grok")
    assert _snapshot(_input(controller)) == legacy_bytes
    grok_text = (
        _root(controller)
        / "challenge_round_01"
        / "tasks"
        / "grok"
        / "instructions_grok.txt"
    ).read_text(encoding="utf-8")
    assert grok_text == controller_module._instructions("challenge")
    for provider in ("gpt", "grok"):
        listed = json.loads(
            (
                _root(controller)
                / "challenge_round_01"
                / "tasks"
                / provider
                / "input"
                / "disputes.json"
            ).read_text(encoding="utf-8")
        )
        assert "groups" not in listed and "required_decision_ids" not in listed

    # The ungrouped round still accepts and reconciles every selected dispute.
    for provider, request in (("gpt", gpt_request), ("grok", grok_request)):
        _answer(controller, request, chosen)
        controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")
    assert [
        item.dispute_id for item in _document(controller).reconciled_decisions
    ] == chosen


def test_an_unreadable_published_input_fails_closed(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    _publish(
        controller, "gpt", selected_dispute_ids=[names["word_r1"], names["note_r1"]]
    )
    (_input(controller) / "disputes.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError):
        _publish(controller, "grok")


def test_a_changed_grouped_input_is_not_overwritten(tmp_path: Path) -> None:
    controller, comparison, names = _setup(tmp_path)
    _publish(
        controller, "gpt", selected_dispute_ids=[names["word_r1"], names["note_r1"]]
    )
    group = (_published(controller).groups or [])[0]
    crop_path = _input(controller) / "crops" / f"{group.crop_id}.json"
    crop_path.write_text(crop_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError):
        _publish(controller, "grok")


def test_full_set_round_without_selection_is_grouped_but_unchanged_in_meaning(
    tmp_path: Path,
) -> None:
    controller, comparison, names = _setup(tmp_path)
    everything = [item.dispute_id for item in challengeable_disputes(comparison)]
    _publish(controller, "gpt")
    assert _document(controller).challenge_rounds[0].selected_dispute_ids == everything
    published = _published(controller)
    assert [item.dispute_id for item in published.disputes] == everything
    assert set(published.required_decision_ids or []) == set(everything)


def test_legacy_reconciled_records_and_events_still_parse() -> None:
    from transcription_review_agent.review.models import ReconciledDecision

    record = ReconciledDecision.model_validate(
        {"dispute_id": "dsp_old", "decision": "CANDIDATE_X", "agreed": True}
    )
    assert record.result_class is None and record.independently_verified is False


# --- schema validation --------------------------------------------------------


def _group_payload(**changes) -> dict:
    base = {
        "group_id": "grp_" + "0" * 16,
        "page_id": "page_001",
        "page_sha256": "a" * 64,
        "basis": "ALIGNED_SPAN",
        "dispute_ids": ["dsp_a", "dsp_b"],
        "crop_id": "crop_grp_" + "0" * 16,
        "crop_sha256": "b" * 64,
        "window_kind": "TARGETED",
        "evidence_refs": [
            {"kind": "GROUP", "evidence_id": "grp_" + "0" * 16, "page_id": "page_001"},
            {
                "kind": "CROP",
                "evidence_id": "crop_grp_" + "0" * 16,
                "page_id": "page_001",
            },
        ],
    }
    base.update(changes)
    return base


def _item(dispute_id: str, **changes) -> dict:
    base = {
        "dispute_id": dispute_id,
        "page_id": "page_001",
        "crop_id": "crop_grp_" + "0" * 16,
        "group_id": "grp_" + "0" * 16,
        "candidate_x": "x",
        "candidate_y": "y",
    }
    base.update(changes)
    return base


def _v2(**changes) -> dict:
    base = {
        "schema_version": "review-challenge-input-v2",
        "disputes": [_item("dsp_a"), _item("dsp_b")],
        "groups": [_group_payload()],
        "required_decision_ids": ["dsp_a", "dsp_b"],
    }
    base.update(changes)
    return base


def test_valid_grouped_input_round_trips() -> None:
    parsed = AnonymousChallengeFile.model_validate(_v2())
    assert [item.group_id for item in parsed.disputes] == ["grp_" + "0" * 16] * 2


@pytest.mark.parametrize(
    "changes",
    [
        {"required_decision_ids": ["dsp_a"]},
        {"required_decision_ids": ["dsp_b", "dsp_a"]},
        {"required_decision_ids": None},
        {"groups": None},
        {"groups": [_group_payload(dispute_ids=["dsp_a"])]},
        {"groups": [_group_payload(dispute_ids=["dsp_a", "dsp_b", "dsp_c"])]},
        {"groups": [_group_payload(), _group_payload(group_id="grp_" + "1" * 16)]},
        {"groups": [_group_payload(crop_id="crop_other")]},
        {"groups": [_group_payload(page_id="page_002")]},
        {"disputes": [_item("dsp_a"), _item("dsp_a")]},
        {"disputes": [_item("dsp_a"), _item("dsp_b", group_id="grp_" + "9" * 16)]},
        {"disputes": [_item("dsp_a"), _item("dsp_b", crop_id="crop_other")]},
    ],
)
def test_inconsistent_grouped_input_is_rejected(changes) -> None:
    with pytest.raises(ValidationError):
        AnonymousChallengeFile.model_validate(_v2(**changes))


def test_ungrouped_input_cannot_carry_group_data() -> None:
    legacy = {
        "schema_version": "review-challenge-input-v1",
        "disputes": [_item("dsp_a")],
    }
    with pytest.raises(ValidationError):
        AnonymousChallengeFile.model_validate(legacy)
    with pytest.raises(ValidationError):
        AnonymousChallengeFile.model_validate(
            {
                "schema_version": "review-challenge-input-v1",
                "disputes": [LEGACY_ITEM],
                "groups": [],
            }
        )


def test_evidence_group_requires_both_references() -> None:
    with pytest.raises(ValidationError):
        AnonymousChallengeFile.model_validate(
            _v2(
                groups=[
                    _group_payload(evidence_refs=[_group_payload()["evidence_refs"][0]])
                ]
            )
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"dispute_id": "dsp_a"},
        {"dispute_id": None, "group_id": None, "dispute_ids": None},
        {"dispute_id": None, "group_id": "grp_x", "dispute_ids": None},
        {"dispute_id": None, "group_id": None, "dispute_ids": ["a", "b"]},
        {"dispute_id": None, "group_id": "grp_x", "dispute_ids": ["a"]},
        {"dispute_id": None, "group_id": "grp_x", "dispute_ids": ["a", "a"]},
    ],
)
def test_crop_record_belongs_to_one_dispute_or_one_group(changes) -> None:
    from transcription_review_agent.review.crop import build_crop

    arguments = dict(
        crop_id="crop_x",
        dispute_id="dsp_one",
        page_id="page_001",
        page_sha256="ab" * 32,
        page_width=20,
        page_height=20,
        target=BBox(x=4, y=4, width=4, height=4),
    )
    arguments.update(changes)
    if changes == {"dispute_id": "dsp_a"}:
        build_crop(**arguments)  # the ordinary single-dispute form is valid
        return
    with pytest.raises(ValidationError):
        build_crop(**arguments)


def test_challenge_item_omits_an_absent_group_and_keeps_empty_candidates() -> None:
    item = AnonymousChallengeItem(
        dispute_id="dsp_a",
        page_id="page_001",
        crop_id="crop_dsp_a",
        candidate_x="",
        candidate_y="",
    )
    payload = json.loads(item.model_dump_json())
    assert "group_id" not in payload
    assert payload["candidate_x"] == "" and payload["candidate_y"] == ""
