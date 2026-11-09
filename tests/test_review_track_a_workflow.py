"""Slice A6: local Track A workflow, from fixture bytes to a merged proposal.

The committed synthetic pair is the always-on document. An authorized
104/105 pack is optional and is read only when ``VS_ARCHIVAL_FIXTURE_DIR``
requests it; those bytes are never written into the repository. Nothing
here calls a provider, a database, or an archive API, and nothing writes
under operational ``outputs/``, state, ledgers, or source bundles.

Dispute IDs are taken from the comparison of the bytes under test. They
are not copied from a document 104 ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from regression_fixtures import (
    FIXTURE_ROOT,
    POLICY_PATH,
    SYNTHETIC_DIR,
    ArchivalFixtureError,
    ArchivalFixtureUnavailable,
    expected_dispute_id,
    load_archival_pack,
    load_synthetic,
    check_document_expectations,
)
from test_review_challenge_groups import _labels, _pages
from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    NOW,
    OCR,
    _png,
    _seal,
    _snapshot,
    _write_output,
)
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.adjudicate import reconcile_challenge
from transcription_review_agent.review.assemble import (
    build_report,
    classify_quality,
    render_transcription,
)
from transcription_review_agent.review.challenge_groups import (
    group_challenge_disputes,
    member_evidence_scope,
    required_decision_ids,
)
from transcription_review_agent.review.config import ReviewConfig
from transcription_review_agent.review.controller import (
    AnonymousChallengeFile,
    ReviewController,
)
from transcription_review_agent.review.errors import ReviewPolicyError
from transcription_review_agent.review.formatting import format_unresolved_alternatives
from transcription_review_agent.review.models import (
    BBox,
    ChallengeDecision,
    ChallengeOutput,
    EvidenceReference,
    WorkRequest,
)
from transcription_review_agent.review.paths import (
    batch_root,
    controller_dir,
    document_root,
)
from transcription_review_agent.review.spans import link_alignment, span_key

REQUIRED_ARCHIVAL_DOCUMENTS = ("104", "105")


def _files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _controller(
    tmp_path: Path, page_count: int, *, rounds: int = 1
) -> ReviewController:
    batches = tmp_path / "batches"
    _seal(batches, [_png(20, 20) for _ in range(page_count)])
    return ReviewController(
        batches_dir=batches,
        reviews_dir=tmp_path / "reviews",
        config=ReviewConfig(
            max_challenge_rounds=rounds,
            max_document_challenge_dispatches=2 * rounds,
        ),
        clock=lambda: NOW,
    )


def _document(controller: ReviewController):
    return controller._require_document(BATCH_ID, DOCUMENT_ID)[1]


def _rebind(request_path: Path, pages) -> dict[str, object]:
    request = WorkRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    return {
        "provider": request.provider,
        "provider_interface": request.provider_interface,
        "task_id": request.task_id,
        "job_id": request.job_id,
        "batch_id": request.batch_id,
        "document_id": request.document_id,
        "attempt": request.attempt,
        "source_bundle_sha256": request.source_bundle_sha256,
        "pages": [page.model_dump(mode="json") for page in pages],
    }


def _deliver_pages(controller: ReviewController, provider: str, pages) -> None:
    request_path = controller.publish(BATCH_ID, DOCUMENT_ID, provider)
    _write_output(controller, request_path, _rebind(request_path, pages))
    controller.accept(BATCH_ID, DOCUMENT_ID, provider)


def _ids_match_rows(comparison) -> None:
    seen: list[str] = []
    for item in comparison.disputes:
        digest = expected_dispute_id(
            item.category.value,
            item.page_id,
            item.region_id,
            item.gpt_text,
            item.grok_text,
        )
        assert item.dispute_id == digest
        seen.append(digest)
    assert len(seen) == len(set(seen))


def _one(comparison, category: str):
    matches = [
        item
        for item in comparison.disputes
        if item.category.value == category and item.substantive
    ]
    assert len(matches) == 1, category
    return matches[0]


def _published(
    controller: ReviewController, round_index: int = 1
) -> AnonymousChallengeFile:
    path = (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / f"challenge_round_{round_index:02d}"
        / "input"
        / "disputes.json"
    )
    return AnonymousChallengeFile.model_validate_json(path.read_text(encoding="utf-8"))


def _answer(
    controller: ReviewController,
    request_path: Path,
    choices: dict[str, tuple[str, str]],
) -> None:
    """Write one evidence-bound file. ``choices`` maps a dispute to label and assessment."""

    request = WorkRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    published = _published(controller, request.round_index or 1)
    decisions = []
    for item in published.disputes:
        label, assessment = choices[item.dispute_id]
        group = next(
            group
            for group in published.groups or []
            if item.dispute_id in group.dispute_ids
        )
        decisions.append(
            ChallengeDecision(
                dispute_id=item.dispute_id,
                decision=label,  # type: ignore[arg-type]
                visual_evidence=f"{request.provider} examined {item.dispute_id}",
                evidence_ref=group.evidence_refs[0],
                evidence_assessment=assessment,  # type: ignore[arg-type]
            )
        )
    output = ChallengeOutput(
        provider=request.provider,
        provider_interface=request.provider_interface,
        task_id=request.task_id,
        job_id=request.job_id,
        batch_id=request.batch_id,
        document_id=request.document_id,
        attempt=request.attempt,
        source_bundle_sha256=request.source_bundle_sha256,
        round_index=request.round_index,
        decisions=decisions,
    )
    _write_output(controller, request_path, output.model_dump(mode="json"))
    controller.accept(BATCH_ID, DOCUMENT_ID, request.provider, stage="challenge")


def _both(
    controller: ReviewController,
    selected: list[str],
    choices_for,
    *,
    evidence: str | None = None,
    evidence_path: Path | None = None,
) -> None:
    gpt = controller.publish(
        BATCH_ID,
        DOCUMENT_ID,
        "gpt",
        stage="challenge",
        evidence=evidence,
        evidence_path=evidence_path,
        selected_dispute_ids=selected,
    )
    grok = controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    for request in (gpt, grok):
        provider = WorkRequest.model_validate_json(
            request.read_text(encoding="utf-8")
        ).provider
        _answer(controller, request, choices_for(provider))


def _close(controller: ReviewController, mode: str) -> None:
    controller.close_challenge(
        BATCH_ID,
        DOCUMENT_ID,
        mode=mode,
        reason=f"Operator closed the local proposal as {mode}.",
    )


def _assert_local_artifacts(
    controller: ReviewController,
    paths: dict[str, Path],
    *,
    sealed: dict[str, bytes],
    comparison_sha: str,
) -> None:
    report = json.loads(paths["review_report.json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["review_manifest.json"].read_text(encoding="utf-8"))
    header = json.loads(
        (batch_root(controller.reviews_dir, BATCH_ID) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["production_writeback"] is False
    assert manifest["production_writeback"] is False
    assert header["production_writeback"] is False
    assert (
        manifest["source_bundle_sha256"] == _document(controller).source_bundle_sha256
    )
    assert manifest["comparison_sha256"] == comparison_sha
    comparison = (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "comparison"
        / "comparison.json"
    )
    assert sha256_bytes(comparison.read_bytes()) == comparison_sha
    for artifact in manifest["artifacts"]:
        assert sha256_bytes(paths[artifact["name"]].read_bytes()) == artifact["sha256"]
    for decision in report["decisions"]:
        assert decision["independently_verified"] is False
        assert decision["eligible_for_writer_memory"] is False
        assert decision["eligible_for_publication"] is False
    assert OCR.encode("utf-8") not in paths["proposed_transcription.txt"].read_bytes()
    source = (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "input"
        / "source_manifest.json"
    )
    assert b"archive_transcription_included" in source.read_bytes()
    assert (
        json.loads(source.read_text(encoding="utf-8"))["archive_transcription_included"]
        is False
    )
    assert _snapshot(controller.batches_dir) == sealed


def _replay(controller: ReviewController) -> ReviewController:
    ledger = controller_dir(controller.reviews_dir, BATCH_ID) / "ledger.jsonl"
    before = ledger.read_bytes()
    checkpoint = controller_dir(controller.reviews_dir, BATCH_ID) / "state.json"
    checkpoint.unlink()
    resumed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        config=controller.config,
        clock=lambda: NOW,
    )
    resumed.status(BATCH_ID, DOCUMENT_ID)
    assert ledger.read_bytes() == before
    return resumed


def _version_bytes(controller: ReviewController, version: int) -> dict[str, bytes]:
    directory = (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "final"
        / "versions"
        / f"v{version:03d}"
    )
    return {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }


def _prepare_synthetic(tmp_path: Path, *, rounds: int = 1):
    before = _files(SYNTHETIC_DIR)
    pair, expected = load_synthetic()
    controller = _controller(tmp_path, len(pair.gpt.pages), rounds=rounds)
    sealed = _snapshot(controller.batches_dir)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_pages(controller, "gpt", pair.gpt.pages)
    _deliver_pages(controller, "grok", pair.grok.pages)
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    assert [item.dispute_id for item in comparison.disputes] == [
        item.dispute_id for item in pair.compare().disputes
    ]
    assert [item.dispute_id for item in comparison.disputes] == [
        item["dispute_id"] for item in expected["disputes"]
    ]
    _ids_match_rows(comparison)
    assert _files(SYNTHETIC_DIR) == before
    return controller, comparison, sealed


def _page(text: str, page_id: str) -> str:
    marker = f"@@ {page_id}\n"
    start = text.index(marker) + len(marker)
    rest = text[start:]
    end = rest.find("\n@@ ")
    return rest if end < 0 else rest[:end]


# --- committed synthetic lifecycle ------------------------------------------


def test_zero_round_completion_uses_current_fixture_ids_and_keeps_alternatives(
    tmp_path: Path,
) -> None:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["decision"]["archival_content_in_git"] is False
    assert list(FIXTURE_ROOT.rglob("*.png")) == []
    assert list(FIXTURE_ROOT.rglob("*.jpg")) == []

    controller, comparison, sealed = _prepare_synthetic(tmp_path)
    comparison_sha = _document(controller).comparison_sha256
    with pytest.raises(ReviewPolicyError, match="explicitly closed"):
        controller.finalize(BATCH_ID, DOCUMENT_ID)
    provisional = controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)
    saved = _version_bytes(controller, 1)
    assert (
        json.loads(provisional["review_report.json"].read_text(encoding="utf-8"))[
            "review_completion"
        ]
        == "interim_provisional"
    )
    assert _document(controller).challenge_closure is None

    _close(controller, "NO_CHALLENGE_REQUESTED")
    completed = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert _version_bytes(controller, 1) == saved
    assert _version_bytes(controller, 2) == {
        name: path.read_bytes() for name, path in completed.items()
    }
    report = json.loads(completed["review_report.json"].read_text(encoding="utf-8"))
    text = completed["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert report["review_completion"] == "completed"
    # The fixture's last page is an incomplete coverage record, so that
    # structured fact outranks unresolved content. The alternatives remain.
    assert report["quality"] == "INCOMPLETE_COVERAGE"
    assert report["coverage_gaps"] == ["page_006"]
    assert report["decisions"] == []
    assert report["challenge_closure"]["mode"] == "NO_CHALLENGE_REQUESTED"
    assert report["challenge_closure"]["round_count"] == 0
    assert [item["status"] for item in report["challenge_closure"]["unresolved"]] == [
        "NOT_CHALLENGED"
    ] * len(_document(controller).substantive_dispute_ids)
    date = _one(comparison, "DATE_OR_NUMBER")
    name = _one(comparison, "NAME_OR_ENTITY")
    wording = _one(comparison, "WORDING")
    missing = _one(comparison, "MISSING_CONTENT")
    cancelled = _one(comparison, "CANCELLATION_OR_INSERTION")
    order = _one(comparison, "READING_ORDER")
    assert format_unresolved_alternatives(date.gpt_text, date.grok_text) in text
    assert format_unresolved_alternatives(name.gpt_text, name.grok_text) in text
    assert format_unresolved_alternatives(wording.gpt_text, wording.grok_text) in text
    assert format_unresolved_alternatives(missing.gpt_text, missing.grok_text) in text
    assert "youth and the middle" in _page(text, cancelled.page_id)
    assert (
        format_unresolved_alternatives(cancelled.gpt_text, cancelled.grok_text)
        not in text
    )
    gpt_order = next(
        page for page in comparison.page_originals if page.page_id == order.page_id
    )
    assert (
        format_unresolved_alternatives(
            gpt_order.gpt_diplomatic_transcription,
            gpt_order.grok_diplomatic_transcription,
        )
        in text
    )
    assert "dated [12 / 13] March 1812." in text
    resumed = _replay(controller)
    assert resumed.status(BATCH_ID, DOCUMENT_ID).finalized is True
    assert _document(resumed).challenge_closure.mode == "NO_CHALLENGE_REQUESTED"
    assert _document(resumed).reconciled_decisions == []
    _assert_local_artifacts(
        resumed, completed, sealed=sealed, comparison_sha=comparison_sha
    )


def test_partial_selection_supports_one_reading_and_leaves_the_rest_unresolved(
    tmp_path: Path,
) -> None:
    controller, comparison, sealed = _prepare_synthetic(tmp_path)
    comparison_sha = _document(controller).comparison_sha256
    comparison_bytes = (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "comparison"
        / "comparison.json"
    ).read_bytes()
    provisional = controller.finalize(BATCH_ID, DOCUMENT_ID, provisional=True)
    saved = {name: path.read_bytes() for name, path in provisional.items()}

    date = _one(comparison, "DATE_OR_NUMBER")
    name = _one(comparison, "NAME_OR_ENTITY")
    wording = _one(comparison, "WORDING")
    cancelled = _one(comparison, "CANCELLATION_OR_INSERTION")
    missing = _one(comparison, "MISSING_CONTENT")
    order = _one(comparison, "READING_ORDER")
    wanted = {
        date.dispute_id,
        name.dispute_id,
        wording.dispute_id,
        cancelled.dispute_id,
    }
    selected = [
        item.dispute_id for item in comparison.disputes if item.dispute_id in wanted
    ]
    specs = {
        date.dispute_id: ("CANDIDATE_X", "SUPPORTS_DECISION"),
        name.dispute_id: ("CANDIDATE_X", "INSUFFICIENT"),
        cancelled.dispute_id: ("CANDIDATE_Y", "SUPPORTS_DECISION"),
    }

    def choices(provider: str) -> dict[str, tuple[str, str]]:
        result = dict(specs)
        label = "CANDIDATE_X" if provider == "gpt" else "CANDIDATE_Y"
        result[wording.dispute_id] = (label, "SUPPORTS_DECISION")
        return result

    gpt = controller.publish(
        BATCH_ID,
        DOCUMENT_ID,
        "gpt",
        stage="challenge",
        selected_dispute_ids=list(reversed(selected)),
    )
    with pytest.raises(ReviewPolicyError, match="frozen"):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "grok",
            stage="challenge",
            selected_dispute_ids=[date.dispute_id],
        )
    grok = controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    published = _published(controller)
    assert [item.dispute_id for item in published.disputes] == selected
    assert published.required_decision_ids == selected
    assert [len(group.dispute_ids) for group in published.groups or []] == [1, 1, 1, 1]
    assert all(group.decisions_independent is True for group in published.groups or [])
    assert {item.dispute_id for item in published.disputes}.isdisjoint(
        {missing.dispute_id, order.dispute_id}
    )
    for item in published.disputes:
        dispute = next(
            row for row in comparison.disputes if row.dispute_id == item.dispute_id
        )
        assert item.candidate_x == dispute.gpt_text
        assert item.candidate_y == dispute.grok_text
    for request in (gpt, grok):
        provider = WorkRequest.model_validate_json(
            request.read_text(encoding="utf-8")
        ).provider
        _answer(controller, request, choices(provider))

    by_id = {
        item.dispute_id: item for item in _document(controller).reconciled_decisions
    }
    assert by_id[date.dispute_id].result_class == "AUTOMATICALLY_SUPPORTED"
    assert by_id[date.dispute_id].selected_reading == date.gpt_text
    assert by_id[date.dispute_id].basis == "DUAL_MODEL_VISUAL_SUPPORT"
    assert by_id[name.dispute_id].result_class == "UNRESOLVED"
    assert by_id[name.dispute_id].selected_reading is None
    assert by_id[name.dispute_id].basis == "INSUFFICIENT_EVIDENCE"
    assert by_id[name.dispute_id].gpt_visual_evidence.startswith("gpt examined")
    assert by_id[name.dispute_id].grok_visual_evidence.startswith("grok examined")
    assert by_id[wording.dispute_id].result_class == "UNRESOLVED"
    assert by_id[wording.dispute_id].basis == "PROVIDER_DISAGREEMENT"
    assert by_id[cancelled.dispute_id].result_class == "AUTOMATICALLY_SUPPORTED"
    assert by_id[cancelled.dispute_id].selected_reading == cancelled.grok_text == ""
    for item in by_id.values():
        assert item.independently_verified is False
        assert item.eligible_for_writer_memory is False
        assert item.eligible_for_publication is False

    _close(controller, "SELECTED_ROUNDS_COMPLETE")
    completed = controller.finalize(BATCH_ID, DOCUMENT_ID)
    assert saved == {name: path.read_bytes() for name, path in provisional.items()}
    assert _version_bytes(controller, 1) == saved
    text = completed["proposed_transcription.txt"].read_text(encoding="utf-8")
    report = json.loads(completed["review_report.json"].read_text(encoding="utf-8"))
    assert report["review_completion"] == "completed"
    assert report["quality"] == "INCOMPLETE_COVERAGE"
    assert date.gpt_text in text
    assert date.grok_text not in text
    assert format_unresolved_alternatives(name.gpt_text, name.grok_text) in text
    assert format_unresolved_alternatives(wording.gpt_text, wording.grok_text) in text
    assert format_unresolved_alternatives(missing.gpt_text, missing.grok_text) in text
    assert (
        format_unresolved_alternatives(cancelled.gpt_text, cancelled.grok_text)
        not in text
    )
    assert "youth and the middle" in _page(text, cancelled.page_id)
    closure = {
        item["dispute_id"]: item["status"]
        for item in report["challenge_closure"]["unresolved"]
    }
    assert closure[name.dispute_id] == "SELECTED_UNRESOLVED"
    assert closure[wording.dispute_id] == "SELECTED_UNRESOLVED"
    assert closure[missing.dispute_id] == "NOT_SELECTED"
    assert closure[order.dispute_id] == "NOT_SELECTED"
    assert date.dispute_id not in closure
    assert (
        document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
        / "comparison"
        / "comparison.json"
    ).read_bytes() == comparison_bytes
    resumed = _replay(controller)
    assert [
        item.dispute_id for item in _document(resumed).reconciled_decisions
    ] == selected
    assert _document(resumed).challenge_rounds[0].selected_dispute_ids == selected
    _assert_local_artifacts(
        resumed, completed, sealed=sealed, comparison_sha=comparison_sha
    )


def test_provider_failure_closes_the_selection_without_a_winner(tmp_path: Path) -> None:
    controller, comparison, sealed = _prepare_synthetic(tmp_path)
    date = _one(comparison, "DATE_OR_NUMBER")
    name = _one(comparison, "NAME_OR_ENTITY")
    selected = [
        item.dispute_id
        for item in comparison.disputes
        if item.dispute_id in {date.dispute_id, name.dispute_id}
    ]
    controller.publish(
        BATCH_ID,
        DOCUMENT_ID,
        "gpt",
        stage="challenge",
        selected_dispute_ids=selected,
    )
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="quota", stage="challenge"
    )
    with pytest.raises(ReviewPolicyError, match="cannot be finalized"):
        controller.finalize(BATCH_ID, DOCUMENT_ID)
    _close(controller, "PROVIDER_TERMINAL_FAILURE")
    completed = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(completed["review_report.json"].read_text(encoding="utf-8"))
    text = completed["proposed_transcription.txt"].read_text(encoding="utf-8")
    closure = {
        item["dispute_id"]: item["status"]
        for item in report["challenge_closure"]["unresolved"]
    }
    assert report["review_completion"] == "completed"
    assert report["decisions"] == []
    assert report["challenge_closure"]["mode"] == "PROVIDER_TERMINAL_FAILURE"
    assert report["challenge_closure"]["failed_round_indexes"] == [1]
    assert closure[date.dispute_id] == "PROVIDER_FAILURE"
    assert closure[name.dispute_id] == "PROVIDER_FAILURE"
    assert closure[_one(comparison, "WORDING").dispute_id] == "NOT_SELECTED"
    assert format_unresolved_alternatives(date.gpt_text, date.grok_text) in text
    assert format_unresolved_alternatives(name.gpt_text, name.grok_text) in text
    assert "[12 / 13]" in text
    resumed = _replay(controller)
    assert _document(resumed).challenge_closure.mode == "PROVIDER_TERMINAL_FAILURE"
    assert _document(resumed).reconciled_decisions == []
    _assert_local_artifacts(
        resumed,
        completed,
        sealed=sealed,
        comparison_sha=_document(resumed).comparison_sha256,
    )


def test_shared_group_members_stay_independent_across_later_rounds(
    tmp_path: Path,
) -> None:
    """Two disputes of one span keep their own results, then a later round stops."""

    controller = _controller(tmp_path, 2, rounds=2)
    sealed = _snapshot(controller.batches_dir)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    for provider in ("gpt", "grok"):
        _deliver_pages(controller, provider, _pages(provider))
    comparison = controller.compare(BATCH_ID, DOCUMENT_ID)
    _ids_match_rows(comparison)
    # Page 2's only region is also named r1. The same-span pair is the page 1
    # wording dispute plus the marginal note on that region, as in the A2 fixture.
    names = _labels(comparison)
    by_id = {item.dispute_id: item for item in comparison.disputes}
    first = by_id[names["word_r1"]]
    note = by_id[names["note_r1"]]
    second = by_id[names["word_r2"]]
    third = by_id[names["word_r3"]]
    assert first.page_id == note.page_id == "page_001"
    assert second.page_id == "page_001" and second.region_id == "r2"
    assert span_key(first, comparison.alignments) == span_key(
        note, comparison.alignments
    )
    assert span_key(first, comparison.alignments) != span_key(
        second, comparison.alignments
    )

    shared_ids = {first.dispute_id, note.dispute_id}

    def first_round(_provider: str) -> dict[str, tuple[str, str]]:
        return {
            first.dispute_id: ("CANDIDATE_X", "SUPPORTS_DECISION"),
            note.dispute_id: ("CANDIDATE_Y", "SUPPORTS_DECISION"),
        }

    _both(
        controller,
        [
            item.dispute_id
            for item in comparison.disputes
            if item.dispute_id in shared_ids
        ],
        first_round,
    )
    published = _published(controller, 1)
    assert len(published.groups or []) == 1
    group = (published.groups or [])[0]
    assert group.dispute_ids == [
        item.dispute_id for item in comparison.disputes if item.dispute_id in shared_ids
    ]
    assert group.decisions_independent is True
    assert second.dispute_id not in group.dispute_ids
    standing = {
        item.dispute_id: item for item in _document(controller).reconciled_decisions
    }
    assert standing[first.dispute_id].selected_reading == first.gpt_text
    assert standing[note.dispute_id].selected_reading == note.grok_text
    assert standing[first.dispute_id].result_class == "AUTOMATICALLY_SUPPORTED"
    assert standing[note.dispute_id].result_class == "AUTOMATICALLY_SUPPORTED"
    assert standing[first.dispute_id].independently_verified is False

    with pytest.raises(
        ReviewPolicyError, match="no repeated challenge without new evidence"
    ):
        controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    evidence = tmp_path / "new-crop.png"
    evidence.write_bytes(_png(8, 8))
    with pytest.raises(ReviewPolicyError, match="already resolved"):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="challenge",
            evidence="new_crop",
            evidence_path=evidence,
            selected_dispute_ids=[first.dispute_id],
        )

    def second_round(_provider: str) -> dict[str, tuple[str, str]]:
        return {second.dispute_id: ("CANDIDATE_X", "SUPPORTS_DECISION")}

    _both(
        controller,
        [second.dispute_id],
        second_round,
        evidence="new_crop",
        evidence_path=evidence,
    )
    with pytest.raises(ReviewPolicyError, match="challenge round cap is reached"):
        controller.publish(
            BATCH_ID,
            DOCUMENT_ID,
            "gpt",
            stage="challenge",
            evidence="new_crop",
            evidence_path=evidence,
        )
    later = _published(controller, 2)
    assert [item.dispute_id for item in later.disputes] == [second.dispute_id]
    kept = {
        item.dispute_id: item for item in _document(controller).reconciled_decisions
    }
    assert kept[first.dispute_id].selected_reading == first.gpt_text
    assert kept[second.dispute_id].selected_reading == second.gpt_text
    assert kept[second.dispute_id].basis == "DUAL_MODEL_VISUAL_SUPPORT"

    _close(controller, "ROUND_CAP")
    completed = controller.finalize(BATCH_ID, DOCUMENT_ID)
    report = json.loads(completed["review_report.json"].read_text(encoding="utf-8"))
    text = completed["proposed_transcription.txt"].read_text(encoding="utf-8")
    assert report["review_completion"] == "completed"
    assert report["quality"] == "UNRESOLVED_CONTENT"
    assert report["challenge_closure"]["mode"] == "ROUND_CAP"
    assert report["challenge_closure"]["max_challenge_rounds"] == 2
    # The two supported members chose different sides of one span, so neither
    # side is dropped. The later round's own span is the exact supported side.
    assert format_unresolved_alternatives(first.gpt_text, first.grok_text) in text
    assert "[Alpha / Alfa] one" in text
    assert second.gpt_text in text
    assert second.grok_text not in text
    assert format_unresolved_alternatives(third.gpt_text, third.grok_text) in text
    unresolved = {
        item["dispute_id"]: item["status"]
        for item in report["challenge_closure"]["unresolved"]
    }
    assert unresolved[third.dispute_id] == "NOT_SELECTED"
    assert first.dispute_id not in unresolved
    resumed = _replay(controller)
    decided = {first.dispute_id, note.dispute_id, second.dispute_id}
    assert [item.dispute_id for item in _document(resumed).reconciled_decisions] == [
        item.dispute_id for item in comparison.disputes if item.dispute_id in decided
    ]
    assert _document(resumed).challenge_closure.mode == "ROUND_CAP"
    _assert_local_artifacts(
        resumed,
        completed,
        sealed=sealed,
        comparison_sha=_document(resumed).comparison_sha256,
    )


# --- optional authorized 104/105 pack ----------------------------------------


def _archival_pack():
    try:
        return load_archival_pack()
    except ArchivalFixtureUnavailable as exc:
        pytest.skip(str(exc))
    except ArchivalFixtureError as exc:
        pytest.fail(
            f"authorized archival fixture run requested but unusable: {exc}",
            pytrace=False,
        )


@pytest.mark.parametrize("document_id", REQUIRED_ARCHIVAL_DOCUMENTS)
def test_authorized_archival_workflow_uses_regenerated_ids(document_id: str) -> None:
    """Exact readings from an outside pack. No images or text are written to Git."""

    pack = _archival_pack()
    document = pack.documents.get(document_id)
    if document is None:
        pytest.fail(
            f"authorized pack does not contain document {document_id}", pytrace=False
        )
    before = _files(pack.root)
    synthetic_before = _files(SYNTHETIC_DIR)
    comparison = check_document_expectations(
        document.pair, document.expected, document.stale_dispute_ids
    )
    _ids_match_rows(comparison)
    substantive = [item for item in comparison.disputes if item.substantive]
    pool = [item for item in substantive if item.category.value != "READING_ORDER"]
    if len(pool) < 2:
        pool = substantive
    if len(pool) < 3:
        pytest.fail(
            f"document {document_id} does not have an unselected substantive dispute",
            pytrace=False,
        )
    spans: dict[object, list] = {}
    for item in pool:
        spans.setdefault(span_key(item, comparison.alignments), []).append(item)
    shared = next((items for items in spans.values() if len(items) >= 2), None)
    chosen = shared[:2] if shared is not None else pool[:2]
    chosen_ids = {item.dispute_id for item in chosen}
    current_ids = {item.dispute_id for item in comparison.disputes}
    assert not ((set(document.stale_dispute_ids) - current_ids) & chosen_ids)

    def window(_dispute):
        return BBox(x=0, y=0, width=1, height=1), "FULL_PAGE_FALLBACK"

    groups = group_challenge_disputes(chosen, comparison.alignments, window)
    again = group_challenge_disputes(chosen, comparison.alignments, window)
    assert [group.group_id for group in groups] == [group.group_id for group in again]
    assert required_decision_ids(groups) == [item.dispute_id for item in chosen]
    assert (len(groups) == 1) is (shared is not None)
    neighbor = next(item for item in pool if item.dispute_id not in chosen_ids)
    assert neighbor.dispute_id not in required_decision_ids(groups)

    decisions = []
    for index, dispute in enumerate(chosen):
        group = next(item for item in groups if dispute.dispute_id in item.dispute_ids)
        scope = member_evidence_scope(group, dispute.dispute_id)
        assessment = "SUPPORTS_DECISION" if index == 0 else "INSUFFICIENT"
        left = ChallengeDecision(
            dispute_id=dispute.dispute_id,
            decision="CANDIDATE_X",
            visual_evidence=f"gpt pack {dispute.dispute_id}",
            evidence_ref=EvidenceReference(
                kind="GROUP",
                evidence_id=group.group_id,
                page_id=group.page_id,
            ),
            evidence_assessment=assessment,  # type: ignore[arg-type]
        )
        right = left.model_copy(
            update={"visual_evidence": f"grok pack {dispute.dispute_id}"}
        )
        decisions.append(reconcile_challenge(dispute, left, right, scope))
    supported, unresolved = decisions
    assert supported.result_class == "AUTOMATICALLY_SUPPORTED"
    assert supported.selected_reading == chosen[0].gpt_text
    assert supported.independently_verified is False
    assert supported.eligible_for_writer_memory is False
    assert supported.eligible_for_publication is False
    assert unresolved.result_class == "UNRESOLVED"
    assert unresolved.selected_reading is None
    assert unresolved.basis == "INSUFFICIENT_EVIDENCE"

    def render() -> str:
        return render_transcription(
            page_ids=[page.page_id for page in document.pair.gpt.pages],
            gpt=document.pair.gpt,
            grok=document.pair.grok,
            comparison=comparison,
            decisions=decisions,
        )

    text = render()
    assert text == render()
    for dispute in chosen:
        block = _page(text, dispute.page_id)
        order = next(
            (
                item
                for item in comparison.disputes
                if item.substantive
                and item.category.value == "READING_ORDER"
                and item.page_id == dispute.page_id
            ),
            None,
        )
        if order is not None and order.dispute_id not in chosen_ids:
            original = next(
                item
                for item in comparison.page_originals
                if item.page_id == dispute.page_id
            )
            assert original.gpt_diplomatic_transcription in block
            assert original.grok_diplomatic_transcription in block
            continue
        alignment = link_alignment(dispute, comparison.alignments)
        gpt_side = dispute.gpt_text or ("" if alignment is None else alignment.gpt_text)
        grok_side = dispute.grok_text or (
            "" if alignment is None else alignment.grok_text
        )
        if gpt_side:
            assert gpt_side in block or (
                alignment is not None and alignment.gpt_text in block
            )
        if grok_side:
            assert grok_side in block or (
                alignment is not None and alignment.grok_text in block
            )
        elif "∅" not in block:
            assert alignment is not None and alignment.grok_text in block
    quality = classify_quality(
        single_provider=False,
        gpt=document.pair.gpt,
        grok=document.pair.grok,
        comparison=comparison,
        decisions=decisions,
    )
    report = build_report(
        batch_id=BATCH_ID,
        document_id=int(document_id),
        quality=quality,
        page_ids=[page.page_id for page in document.pair.gpt.pages],
        single_provider=False,
        succeeded_provider=None,
        comparison=comparison,
        decisions=decisions,
        identifications=[],
        review_completion="completed",
        provider_failure_count=0,
        estimated_work_units=0,
        gpt=document.pair.gpt,
        grok=document.pair.grok,
    )
    assert report.production_writeback is False
    assert quality in {"UNRESOLVED_CONTENT", "INCOMPLETE_COVERAGE"}
    assert _files(pack.root) == before
    assert _files(SYNTHETIC_DIR) == synthetic_before
