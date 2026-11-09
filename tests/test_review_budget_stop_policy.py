"""Dispatch budgets and the round-1 dispute cap.

Selections, evidence kinds, reconciliation, and closure modes stay as they
are. These tests cover only the scheduling limits. No provider, database,
or archive call.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest

from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _controller,
    _deliver_first_pass,
    _first_pass_body,
    _page,
    _png,
    _write_output,
)
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.manifest import (
    BatchDocumentRef,
    BatchManifest,
    DocumentBundleManifest,
    publish_batch_manifest,
    publish_document_manifest,
)
from transcription_review_agent.errors import ConfigError
from transcription_review_agent.review.config import ReviewConfig, load_review_config
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.errors import (
    ReviewLimitError,
    ReviewOutputRejected,
    ReviewPolicyError,
)
from transcription_review_agent.review.models import (
    ChallengeDecision,
    ChallengeOutput,
    ComparisonFile,
    Dispute,
    DisputeCategory,
    WorkRequest,
)
from transcription_review_agent.review.paths import document_root
from transcription_review_agent.review.runtime import (
    batch_challenge_dispatches,
    document_challenge_dispatches,
    schedule_challenge_disputes,
)

SHA = "ab" * 32


def _comparison(disputes: list[Dispute]) -> ComparisonFile:
    return ComparisonFile(
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        source_bundle_sha256=SHA,
        gpt_output_sha256=SHA,
        grok_output_sha256=SHA,
        page_originals=[],
        disputes=disputes,
        unresolved_text_page_ids=[],
    )


def _dispute(dispute_id: str, category: DisputeCategory) -> Dispute:
    return Dispute(
        dispute_id=dispute_id,
        category=category,
        page_id="page_001",
        gpt_text="left",
        grok_text="right",
        substantive=category is not DisputeCategory.PUNCTUATION,
    )


def _open_compared(
    tmp_path: Path,
    *,
    pages: list[bytes] | None = None,
    gpt: list | None = None,
    grok: list | None = None,
    config: ReviewConfig | None = None,
) -> ReviewController:
    controller = _controller(tmp_path, pages)
    if config is not None:
        controller.config = config
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    _deliver_first_pass(
        controller,
        "gpt",
        gpt
        or [
            _page("page_001", 1, "Alpha one"),
            _page("page_002", 2, "Beta two"),
        ],
    )
    _deliver_first_pass(
        controller,
        "grok",
        grok
        or [
            _page("page_001", 1, "Alfa one"),
            _page("page_002", 2, "Beta two"),
        ],
    )
    controller.compare(BATCH_ID, DOCUMENT_ID)
    return controller


def _publish(controller: ReviewController, provider: str, **kwargs) -> Path:
    return controller.publish(
        BATCH_ID, DOCUMENT_ID, provider, stage="challenge", **kwargs
    )


def _document(controller: ReviewController):
    return controller._require_document(BATCH_ID, DOCUMENT_ID)[1]


def _state(controller: ReviewController):
    return controller._require_document(BATCH_ID, DOCUMENT_ID)[0]


def _root(controller: ReviewController) -> Path:
    return document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)


def _accept_unresolved(
    controller: ReviewController, provider: str, request_path: Path
) -> None:
    request = WorkRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    selected = _document(controller).challenge_rounds[-1].selected_dispute_ids or []
    payload = ChallengeOutput(
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
                decision="UNRESOLVED",
                visual_evidence="The ink is unclear.",
            )
            for dispute_id in selected
        ],
    ).model_dump(mode="json")
    _write_output(controller, request_path, payload)
    controller.accept(BATCH_ID, DOCUMENT_ID, provider, stage="challenge")


def test_defaults_count_tasks_not_tokens() -> None:
    config = load_review_config({})
    assert config.max_challenge_rounds == 1
    assert config.max_challenge_disputes == 15
    assert config.max_document_challenge_dispatches == 2
    assert config.max_batch_challenge_dispatches == 10
    raised = load_review_config(
        {
            "TRANSCRIPTION_REVIEW_MAX_CHALLENGE_ROUNDS": "2",
            "TRANSCRIPTION_REVIEW_MAX_CHALLENGE_DISPUTES": "15",
            "TRANSCRIPTION_REVIEW_MAX_DOCUMENT_CHALLENGE_DISPATCHES": "4",
            "TRANSCRIPTION_REVIEW_MAX_BATCH_CHALLENGE_DISPATCHES": "20",
        }
    )
    assert raised.max_challenge_rounds == 2
    assert raised.max_document_challenge_dispatches == 4
    assert raised.max_batch_challenge_dispatches == 20
    with pytest.raises(ConfigError):
        load_review_config({"TRANSCRIPTION_REVIEW_MAX_CHALLENGE_DISPUTES": "many"})


def test_priority_then_comparison_order_caps_without_dropping_punctuation() -> None:
    comparison = _comparison(
        [
            _dispute("word-later", DisputeCategory.WORDING),
            _dispute("missing", DisputeCategory.MISSING_CONTENT),
            _dispute("word-earlier-in-file", DisputeCategory.WORDING),
            _dispute("comma", DisputeCategory.PUNCTUATION),
        ]
    )
    assert schedule_challenge_disputes(comparison, limit=2, resolved_ids=[]) == [
        "missing",
        "word-later",
    ]
    assert schedule_challenge_disputes(
        comparison, limit=15, resolved_ids=["missing"]
    ) == ["word-later", "word-earlier-in-file"]


def test_sixteen_disputes_auto_select_fifteen_and_explicit_sixteen_is_refused(
    tmp_path: Path,
) -> None:
    page_count = 16
    controller = _open_compared(
        tmp_path,
        pages=[_png(20, 20) for _ in range(page_count)],
        gpt=[
            _page(f"page_{index:03d}", index, f"Alpha {index}")
            for index in range(1, 17)
        ],
        grok=[
            _page(f"page_{index:03d}", index, f"Alfa {index}") for index in range(1, 17)
        ],
    )
    substantive = list(_document(controller).substantive_dispute_ids)
    assert len(substantive) == 16
    before = _root(controller)
    with pytest.raises(ReviewPolicyError, match="at most 15"):
        _publish(controller, "gpt", selected_dispute_ids=substantive)
    assert not (before / "challenge_round_01").exists()
    explicit = _publish(controller, "gpt", selected_dispute_ids=substantive[:15])
    assert explicit.name == "work_request.json"
    assert (
        _document(controller).challenge_rounds[0].selected_dispute_ids
        == substantive[:15]
    )
    # A second provider joins the frozen set. A different subset is refused.
    with pytest.raises(ReviewPolicyError, match="frozen"):
        _publish(controller, "grok", selected_dispute_ids=substantive[:14])
    _publish(controller, "grok")
    stored = _document(controller).challenge_rounds[0].selected_dispute_ids
    assert stored == substantive[:15]
    listed = json.loads(
        (before / "challenge_round_01" / "input" / "disputes.json").read_text(
            encoding="utf-8"
        )
    )
    assert [item["dispute_id"] for item in listed["disputes"]] == substantive[:15]
    assert substantive[15] not in stored


def test_omitted_round_one_selection_is_the_same_frozen_set_for_both_providers(
    tmp_path: Path,
) -> None:
    controller = _open_compared(tmp_path)
    gpt_request = _publish(controller, "gpt")
    grok_request = _publish(controller, "grok")
    selected = _document(controller).challenge_rounds[0].selected_dispute_ids
    assert selected == list(_document(controller).substantive_dispute_ids)
    assert len(selected or []) == 1
    assert _document(controller).challenge_rounds[0].dispatch_reservation == 2
    assert document_challenge_dispatches(_document(controller)) == 2
    for request in (gpt_request, grok_request):
        listed = json.loads(
            (request.parent / "input" / "disputes.json").read_text(encoding="utf-8")
        )
        assert [item["dispute_id"] for item in listed["disputes"]] == selected


def test_document_budget_of_one_opens_no_round(tmp_path: Path) -> None:
    for first in ("gpt", "grok"):
        controller = _open_compared(
            tmp_path / first,
            config=ReviewConfig(max_document_challenge_dispatches=1),
        )
        with pytest.raises(ReviewLimitError, match="both providers"):
            _publish(controller, first)
        document = _document(controller)
        assert document.challenge_rounds == []
        assert document.challenge_closure is None
        assert document.finalized is False
        assert not (_root(controller) / "challenge_round_01").exists()
        assert document_challenge_dispatches(document) == 0


def test_batch_budget_is_independent_of_the_document_budget(tmp_path: Path) -> None:
    controller = _open_compared(
        tmp_path,
        config=ReviewConfig(
            max_document_challenge_dispatches=4,
            max_batch_challenge_dispatches=1,
        ),
    )
    with pytest.raises(ReviewLimitError, match="both providers"):
        _publish(controller, "gpt")
    assert batch_challenge_dispatches(_state(controller)) == 0
    assert _document(controller).challenge_rounds == []
    assert _document(controller).challenge_closure is None
    assert not (_root(controller) / "challenge_round_01").exists()


def test_awaiting_retry_and_recovery_do_not_take_another_allocation(
    tmp_path: Path,
) -> None:
    controller = _open_compared(
        tmp_path,
        config=ReviewConfig(max_document_challenge_dispatches=2),
    )
    first = _publish(controller, "gpt")
    second = _publish(controller, "gpt")
    assert second == first
    assert document_challenge_dispatches(_document(controller)) == 2
    controller.record_failure(
        BATCH_ID, DOCUMENT_ID, "gpt", kind="transient", stage="challenge"
    )
    retried = _publish(controller, "gpt")
    assert retried.name == "work_request.json"
    assert document_challenge_dispatches(_document(controller)) == 2
    target = (
        _root(controller)
        / json.loads(retried.read_text(encoding="utf-8"))["output_relative_path"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"provider":"nope"}', encoding="utf-8")
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    recovered = controller.recover_invalid_output(
        BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge"
    )
    assert recovered.name == "work_request.json"
    assert document_challenge_dispatches(_document(controller)) == 2
    _publish(controller, "grok")
    assert document_challenge_dispatches(_document(controller)) == 2
    assert _document(controller).finalized is False
    assert _document(controller).challenge_closure is None


def test_round_two_needs_an_explicit_selection_and_new_evidence(tmp_path: Path) -> None:
    controller = _open_compared(
        tmp_path,
        config=ReviewConfig(
            max_challenge_rounds=2,
            max_document_challenge_dispatches=4,
        ),
    )
    for provider in ("gpt", "grok"):
        _accept_unresolved(controller, provider, _publish(controller, provider))
    dispute_id = _document(controller).substantive_dispute_ids[0]
    evidence = tmp_path / "new-crop.png"
    evidence.write_bytes(_png(8, 8))
    page = _root(controller) / "input" / "page_001.png"
    with pytest.raises(ReviewPolicyError, match="explicit dispute selection"):
        _publish(
            controller,
            "gpt",
            evidence="new_crop",
            evidence_path=evidence,
        )
    with pytest.raises(ReviewPolicyError, match=dispute_id):
        _publish(
            controller,
            "gpt",
            evidence="different_source_scan",
            evidence_path=page,
            selected_dispute_ids=[dispute_id],
        )
    assert "already examined" in _repeat_message(controller, dispute_id, page)
    opened = _publish(
        controller,
        "gpt",
        evidence="new_crop",
        evidence_path=evidence,
        selected_dispute_ids=[dispute_id],
    )
    assert "challenge_round_02" in opened.as_posix()
    challenge = _document(controller).challenge_rounds[1]
    assert challenge.selected_dispute_ids == [dispute_id]
    assert challenge.evidence_identity == f"new_crop:{challenge.evidence_sha256}"
    assert (
        challenge.evidence_identity
        != _document(controller).challenge_rounds[0].evidence_identity
    )
    grok_request = _publish(
        controller,
        "grok",
        evidence="new_crop",
        evidence_path=evidence,
        selected_dispute_ids=[dispute_id],
    )
    _accept_unresolved(controller, "gpt", opened)
    _accept_unresolved(controller, "grok", grok_request)
    assert document_challenge_dispatches(_document(controller)) == 4
    more = tmp_path / "third.png"
    more.write_bytes(_png(9, 9))
    with pytest.raises(ReviewPolicyError, match="challenge round cap is reached"):
        _publish(
            controller,
            "gpt",
            evidence="another_occurrence",
            evidence_path=more,
            selected_dispute_ids=[dispute_id],
        )
    assert _document(controller).challenge_closure is None


def _repeat_message(controller: ReviewController, dispute_id: str, page: Path) -> str:
    try:
        _publish(
            controller,
            "gpt",
            evidence="different_source_scan",
            evidence_path=page,
            selected_dispute_ids=[dispute_id],
        )
    except ReviewPolicyError as exc:
        return str(exc)
    raise AssertionError("same-page evidence was accepted")


def test_opening_a_round_past_the_budget_writes_no_round(tmp_path: Path) -> None:
    controller = _open_compared(
        tmp_path,
        config=ReviewConfig(max_challenge_rounds=2),
    )
    for provider in ("gpt", "grok"):
        _accept_unresolved(controller, provider, _publish(controller, provider))
    evidence = tmp_path / "new-crop.png"
    evidence.write_bytes(_png(8, 8))
    dispute_id = _document(controller).substantive_dispute_ids[0]
    with pytest.raises(ReviewLimitError, match="document challenge dispatch budget"):
        _publish(
            controller,
            "gpt",
            evidence="new_crop",
            evidence_path=evidence,
            selected_dispute_ids=[dispute_id],
        )
    assert not (_root(controller) / "challenge_round_02").exists()
    assert len(_document(controller).challenge_rounds) == 1
    assert _document(controller).challenge_closure is None


def test_legacy_round_without_evidence_identity_still_replays(tmp_path: Path) -> None:
    controller = _open_compared(tmp_path)
    _publish(controller, "gpt")
    directory = controller.reviews_dir / f"batch_{BATCH_ID}" / "controller"
    lines = []
    for line in (directory / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        item.pop("evidence_identity", None)
        item.pop("dispatch_reservation", None)
        lines.append(json.dumps(item, separators=(",", ":"), sort_keys=True))
    (directory / "ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    for document in state["documents"]:
        for challenge in document["challenge_rounds"]:
            challenge.pop("evidence_identity", None)
            challenge.pop("dispatch_reservation", None)
    (directory / "state.json").write_text(json.dumps(state), encoding="utf-8")
    resumed = ReviewController(
        batches_dir=controller.batches_dir,
        reviews_dir=controller.reviews_dir,
        clock=controller.clock,
    )
    document = _document(resumed)
    assert document.challenge_rounds[0].evidence_identity is None
    assert document.challenge_rounds[0].dispatch_reservation is None
    assert document.challenge_rounds[0].selected_dispute_ids
    assert document_challenge_dispatches(document) == 1
    _publish(resumed, "grok")
    legacy = _document(resumed).challenge_rounds[0]
    assert legacy.evidence_identity is None
    assert legacy.dispatch_reservation is None
    assert document_challenge_dispatches(_document(resumed)) == 2


def _seal_second_document(batches: Path) -> int:
    """Add document 102 to a sealed batch before any review document is opened."""

    new_id = DOCUMENT_ID + 1
    batch = batches / BATCH_ID
    source = batch / "documents" / str(DOCUMENT_ID)
    destination = batch / "documents" / str(new_id)
    shutil.copytree(source, destination)
    manifest = DocumentBundleManifest.model_validate_json(
        (destination / "document_manifest.json").read_bytes()
    )
    (destination / "document_manifest.json").unlink()
    publish_document_manifest(
        destination / "document_manifest.json",
        manifest.model_copy(
            update={
                "document_id": new_id,
                "archive_item_id": manifest.archive_item_id + 1,
            }
        ),
    )
    batch_manifest = BatchManifest.model_validate_json(
        (batch / "batch_manifest.json").read_bytes()
    )
    (batch / "batch_manifest.json").unlink()
    published = destination / "document_manifest.json"
    publish_batch_manifest(
        batch / "batch_manifest.json",
        batch_manifest.model_copy(
            update={
                "target_reviewable_count": 2,
                "document_ids": [DOCUMENT_ID, new_id],
                "documents": [
                    *batch_manifest.documents,
                    BatchDocumentRef(
                        document_id=new_id,
                        manifest_relative_path=(
                            f"documents/{new_id}/document_manifest.json"
                        ),
                        manifest_sha256=sha256_bytes(published.read_bytes()),
                    ),
                ],
            }
        ),
    )
    return new_id


def _compare_document(controller: ReviewController, document_id: int) -> None:
    controller.open_sealed_document(BATCH_ID, document_id)
    pages = [
        _page("page_001", 1, "Alpha one"),
        _page("page_002", 2, "Beta two"),
    ]
    other = [
        _page("page_001", 1, "Alfa one"),
        _page("page_002", 2, "Beta two"),
    ]
    for provider, provider_pages in (("gpt", pages), ("grok", other)):
        request_path = controller.publish(BATCH_ID, document_id, provider)
        request = json.loads(request_path.read_text(encoding="utf-8"))
        target = (
            document_root(controller.reviews_dir, BATCH_ID, document_id)
            / request["output_relative_path"]
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_first_pass_body(request_path, provider_pages)),
            encoding="utf-8",
        )
        controller.accept(BATCH_ID, document_id, provider)
    controller.compare(BATCH_ID, document_id)


def _open_two(tmp_path: Path, config: ReviewConfig) -> tuple[ReviewController, int]:
    controller = _controller(tmp_path)
    second = _seal_second_document(controller.batches_dir)
    controller.config = config
    _compare_document(controller, DOCUMENT_ID)
    _compare_document(controller, second)
    return controller, second


def test_grok_first_uses_the_same_pair_reservation(tmp_path: Path) -> None:
    controller = _open_compared(tmp_path)
    grok_request = _publish(controller, "grok")
    assert document_challenge_dispatches(_document(controller)) == 2
    assert _document(controller).challenge_rounds[0].dispatch_reservation == 2
    gpt_request = _publish(controller, "gpt")
    assert document_challenge_dispatches(_document(controller)) == 2
    assert grok_request.parent.name == "grok"
    assert gpt_request.parent.name == "gpt"


def test_provider_failure_keeps_the_reserved_pair(tmp_path: Path) -> None:
    controller, second = _open_two(
        tmp_path,
        ReviewConfig(max_batch_challenge_dispatches=2),
    )
    request = _publish(controller, "gpt")
    target = (
        _root(controller)
        / json.loads(request.read_text(encoding="utf-8"))["output_relative_path"]
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"provider":"nope"}', encoding="utf-8")
    with pytest.raises(ReviewOutputRejected):
        controller.accept(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    _publish(controller, "grok")
    assert document_challenge_dispatches(_document(controller)) == 2
    assert _document(controller).challenge_closure is None
    with pytest.raises(ReviewLimitError, match="both providers"):
        controller.publish(BATCH_ID, second, "gpt", stage="challenge")
    assert not (
        document_root(controller.reviews_dir, BATCH_ID, second) / "challenge_round_01"
    ).exists()


def test_competing_documents_share_one_batch_reservation(tmp_path: Path) -> None:
    controller, second = _open_two(
        tmp_path,
        ReviewConfig(max_batch_challenge_dispatches=2),
    )
    _publish(controller, "gpt")
    with pytest.raises(ReviewLimitError, match="batch challenge dispatch budget"):
        controller.publish(BATCH_ID, second, "gpt", stage="challenge")
    assert not (
        document_root(controller.reviews_dir, BATCH_ID, second) / "challenge_round_01"
    ).exists()
    _publish(controller, "grok")
    assert batch_challenge_dispatches(_state(controller)) == 2
    assert _document(controller).challenge_rounds[0].dispatch_reservation == 2


def test_interleaved_document_preparation_cannot_exceed_the_batch_budget(
    tmp_path: Path,
) -> None:
    controller, second = _open_two(
        tmp_path,
        ReviewConfig(max_batch_challenge_dispatches=2),
    )
    barrier = threading.Barrier(2)
    outcomes: dict[int, BaseException | None] = {}

    def publish(document_id: int) -> None:
        barrier.wait()
        try:
            controller.publish(BATCH_ID, document_id, "gpt", stage="challenge")
        except BaseException as exc:  # noqa: BLE001 - the test records the race
            outcomes[document_id] = exc
        else:
            outcomes[document_id] = None

    threads = [
        threading.Thread(target=publish, args=(document_id,))
        for document_id in (DOCUMENT_ID, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = [item for item in outcomes.values() if item is not None]
    assert len(failures) == 1
    assert isinstance(failures[0], ReviewLimitError)
    assert "both providers" in str(failures[0])
    assert batch_challenge_dispatches(_state(controller)) == 2
    winner = next(
        document_id for document_id, error in outcomes.items() if error is None
    )
    controller.publish(BATCH_ID, winner, "grok", stage="challenge")
    assert batch_challenge_dispatches(_state(controller)) == 2
