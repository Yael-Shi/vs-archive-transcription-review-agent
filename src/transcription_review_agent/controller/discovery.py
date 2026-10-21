"""Discovery orchestration for one Pilot 01 scan pass.

Loads the checkpoint, fetches scan pages and review bundles, and commits the
existing ledger transitions. It does not download sources, call an AI
provider, persist review-bundle payloads, or start bundling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from transcription_review_agent.api_client.client import (
    ReviewBundleNotFound,
    TranscriptionReviewApiClient,
)
from transcription_review_agent.api_client.errors import ApiProtocolError
from transcription_review_agent.api_client.schemas import (
    ApiError,
    BlockingIntegrityCode,
    BundleOutcome,
    NonEligibleCode,
    ReviewBundleResponse,
)
from transcription_review_agent.config import AgentConfig
from transcription_review_agent.constants import SCAN_PATH
from transcription_review_agent.errors import StateInvariantError
from transcription_review_agent.state.schema import BatchStatus, ControllerState
from transcription_review_agent.state.store import StateStore
from transcription_review_agent.state.transitions import (
    note_blocking,
    note_discovery_exhausted,
    note_no_work,
    note_non_eligible,
    note_not_found,
    select_reviewable,
)

Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Outcome of one discovery run. No bundle payload and no presigned URL."""

    selected_document_ids: tuple[int, ...]
    batch_status: BatchStatus
    discovery_exhausted: bool
    blocked_document_id: int | None
    current_batch_id: str | None


def discover_next_batch(
    config: AgentConfig,
    api_client: TranscriptionReviewApiClient,
    state_store: StateStore,
    *,
    clock: Clock | None = None,
) -> DiscoveryResult:
    """Scan forward from the durable cursor and commit discovery transitions.

    ``review_bundle`` is called once per visited candidate. The scan page's
    ``next_after_id`` is only the cursor for the next GET. It is never written
    into ``last_scanned_document_id``. That field moves only when a per-document
    transition is committed.
    """

    now = _utc_now if clock is None else clock
    state = state_store.load()
    if state.batch_status is BatchStatus.BUNDLING:
        raise StateInvariantError("discovery does not run while bundling")

    if state.batch_status is BatchStatus.BLOCKED:
        state, still_blocked = _revisit_blocked_document(
            state,
            api_client=api_client,
            state_store=state_store,
            now=now,
            batch_size=config.batch_size,
        )
        if still_blocked or _batch_is_full(state):
            return _result(state)
    elif _batch_is_full(state) or _open_batch_already_exhausted(state):
        return _result(state)

    return _scan_forward(
        state,
        config=config,
        api_client=api_client,
        state_store=state_store,
        now=now,
    )


def _scan_forward(
    state: ControllerState,
    *,
    config: AgentConfig,
    api_client: TranscriptionReviewApiClient,
    state_store: StateStore,
    now: Clock,
) -> DiscoveryResult:
    """Visit scan pages until the batch is full, discovery ends, or a stop hits.

    ``scan_after`` is the pagination argument for the next GET. It starts as
    the durable cursor and then follows ``next_after_id``. Those two values
    are not assigned to each other.
    """

    scan_after = state.last_scanned_document_id
    while True:
        page = api_client.scan(after_id=scan_after, limit=config.scan_page_limit)
        for candidate in page.items:
            state, stop = _visit_document(
                state,
                document_id=candidate.document_id,
                api_client=api_client,
                state_store=state_store,
                now=now,
                batch_size=config.batch_size,
            )
            if stop or _batch_is_full(state):
                return _result(state)
        if page.exhausted:
            return _record_exhaustion(state, state_store=state_store, now=now)
        if page.next_after_id is None:
            raise ApiProtocolError(
                "GET scan returned a page that is not exhausted and has no next_after_id",
                method="GET",
                path=SCAN_PATH,
                status_code=200,
            )
        scan_after = page.next_after_id


def _revisit_blocked_document(
    state: ControllerState,
    *,
    api_client: TranscriptionReviewApiClient,
    state_store: StateStore,
    now: Clock,
    batch_size: int,
) -> tuple[ControllerState, bool]:
    """Re-fetch only the blocked document. Does not scan past it."""

    blocked = state.blocked
    if blocked is None:
        raise StateInvariantError("blocked discovery requires a blocked document")
    updated, stop = _visit_document(
        state,
        document_id=blocked.document_id,
        api_client=api_client,
        state_store=state_store,
        now=now,
        batch_size=batch_size,
    )
    return updated, stop


def _visit_document(
    state: ControllerState,
    *,
    document_id: int,
    api_client: TranscriptionReviewApiClient,
    state_store: StateStore,
    now: Clock,
    batch_size: int,
) -> tuple[ControllerState, bool]:
    """Commit one existing transition. The second value is true when the run stops."""

    bundle = api_client.review_bundle(document_id)
    if isinstance(bundle, ReviewBundleNotFound):
        entry = note_not_found(state, document_id=document_id, occurred_at=now())
        return state_store.commit(entry), False

    if bundle.outcome is BundleOutcome.NON_ELIGIBLE:
        entry = note_non_eligible(
            state,
            document_id=document_id,
            code=_non_eligible_code(bundle),
            occurred_at=now(),
        )
        return state_store.commit(entry), False

    if bundle.outcome is BundleOutcome.REVIEWABLE:
        archive_item_id, language = _reviewable_selection(bundle)
        entry = select_reviewable(
            state,
            document_id=document_id,
            archive_item_id=archive_item_id,
            language=language,
            occurred_at=now(),
            target_reviewable_count=batch_size,
        )
        return state_store.commit(entry), False

    if bundle.outcome is BundleOutcome.BLOCKING_INTEGRITY_ERROR:
        error = _require_error(bundle)
        planned = note_blocking(
            state,
            document_id=document_id,
            code=_blocking_code(error),
            message=error.message,
            details=error.details,
            occurred_at=now(),
        )
        if planned is None:
            return state, True
        return state_store.commit(planned), True

    raise StateInvariantError(f"unexpected bundle outcome {bundle.outcome}")


def _record_exhaustion(
    state: ControllerState,
    *,
    state_store: StateStore,
    now: Clock,
) -> DiscoveryResult:
    if state.batch_status is BatchStatus.ACCUMULATING:
        if state.discovery_exhausted:
            return _result(state)
        entry = note_discovery_exhausted(state, occurred_at=now())
        return _result(state_store.commit(entry))
    if state.batch_status is BatchStatus.NO_WORK:
        return _result(state)
    entry = note_no_work(state, occurred_at=now())
    return _result(state_store.commit(entry))


def _reviewable_selection(
    bundle: ReviewBundleResponse,
) -> tuple[int, Literal["en", "fr"]]:
    archive_item_id = bundle.document.archive_item_id
    language = bundle.document.language
    if language == "en":
        selected_language: Literal["en", "fr"] = "en"
    elif language == "fr":
        selected_language = "fr"
    else:
        raise StateInvariantError("REVIEWABLE bundle cannot be selected")
    if archive_item_id is None:
        raise StateInvariantError("REVIEWABLE bundle cannot be selected")
    return archive_item_id, selected_language


def _non_eligible_code(bundle: ReviewBundleResponse) -> NonEligibleCode:
    try:
        return NonEligibleCode(_require_error(bundle).code)
    except ValueError as exc:
        raise StateInvariantError("NON_ELIGIBLE bundle has an unknown code") from exc


def _blocking_code(error: ApiError) -> BlockingIntegrityCode:
    try:
        return BlockingIntegrityCode(error.code)
    except ValueError as exc:
        raise StateInvariantError("blocking bundle has an unknown code") from exc


def _require_error(bundle: ReviewBundleResponse) -> ApiError:
    error = bundle.error
    if error is None:
        raise StateInvariantError("classified bundle is missing its error")
    return error


def _batch_is_full(state: ControllerState) -> bool:
    return (
        state.batch_status is BatchStatus.ACCUMULATING
        and len(state.documents) >= state.batch_target_count
    )


def _open_batch_already_exhausted(state: ControllerState) -> bool:
    return state.batch_status is BatchStatus.ACCUMULATING and state.discovery_exhausted


def _result(state: ControllerState) -> DiscoveryResult:
    blocked = state.blocked
    return DiscoveryResult(
        selected_document_ids=tuple(state.current_batch_document_ids),
        batch_status=state.batch_status,
        discovery_exhausted=state.discovery_exhausted,
        blocked_document_id=None if blocked is None else blocked.document_id,
        current_batch_id=state.current_batch_id,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
