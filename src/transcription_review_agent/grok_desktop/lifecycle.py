"""Local lifecycle for resources this adapter created.

Remote deletion is intentionally absent. ``DELETED`` is reserved for a later
check that a provider listing no longer contains one owned id. Nothing in
this module calls Grok or Codex.

States
------
ACTIVE
    The resource exists and work is still open.
RETAINED
    Required outputs and provenance were verified, and a retain reason says
    the resource is still needed for a challenge or recovery.
ELIGIBLE_FOR_CLEANUP
    Outputs stay verified and the retain reason has been cleared. A future
    dry-run may name this one id. This module does not perform that dry-run.
DELETED
    Reserved. Not reachable here.
FAILED_CREATE, FAILED_SEND, FAILED_DELETE
    Terminal until an operator records a new resource. No automatic retry.

A Grok bot, a Grok chat, and a Codex exec are different kinds. Making one
eligible does not make another eligible. A chat reset is not a bot deletion.
"""

from __future__ import annotations

from transcription_review_agent.grok_desktop.registry import (
    ResourceRecord,
    ResourceState,
)


class LifecycleError(Exception):
    """The requested lifecycle move is not allowed."""


_TRANSITIONS: dict[ResourceState, frozenset[ResourceState]] = {
    ResourceState.ACTIVE: frozenset(
        {ResourceState.RETAINED, ResourceState.FAILED_CREATE, ResourceState.FAILED_SEND}
    ),
    ResourceState.RETAINED: frozenset({ResourceState.ELIGIBLE_FOR_CLEANUP}),
    ResourceState.ELIGIBLE_FOR_CLEANUP: frozenset(),
    ResourceState.DELETED: frozenset(),
    ResourceState.FAILED_CREATE: frozenset(),
    ResourceState.FAILED_SEND: frozenset(),
    ResourceState.FAILED_DELETE: frozenset(),
}


def can_transition(current: ResourceState, target: ResourceState) -> bool:
    return target in _TRANSITIONS[current]


def cleanup_eligible(record: ResourceRecord) -> bool:
    """True only after outputs are verified and retention has been released."""

    return (
        record.state is ResourceState.ELIGIBLE_FOR_CLEANUP
        and record.outputs_verified
        and record.retain_reason == ""
    )


def apply_transition(
    record: ResourceRecord,
    target: ResourceState,
    *,
    retain_reason: str = "",
) -> ResourceRecord:
    """Return the next record. Does not contact a provider."""

    if not can_transition(record.state, target):
        raise LifecycleError(f"{record.state.value} cannot move to {target.value}")
    if target is ResourceState.RETAINED:
        if not record.outputs_verified or not retain_reason.strip():
            raise LifecycleError("retention requires verified outputs and a reason")
        return _copy(record, state=target, retain_reason=retain_reason.strip())
    if target is ResourceState.ELIGIBLE_FOR_CLEANUP:
        if not record.outputs_verified:
            raise LifecycleError("cleanup requires verified outputs")
        return _copy(record, state=target, retain_reason="")
    if target in {ResourceState.FAILED_CREATE, ResourceState.FAILED_SEND}:
        return _copy(record, state=target)
    raise LifecycleError(f"{target.value} is not available")


def note_outputs_verified(record: ResourceRecord) -> ResourceRecord:
    if record.state is not ResourceState.ACTIVE:
        raise LifecycleError("only an active resource can record verified outputs")
    return _copy(record, outputs_verified=True)


def _copy(
    record: ResourceRecord,
    *,
    state: ResourceState | None = None,
    retain_reason: str | None = None,
    outputs_verified: bool | None = None,
) -> ResourceRecord:
    return ResourceRecord(
        local_id=record.local_id,
        provider=record.provider,
        kind=record.kind,
        provider_id=record.provider_id,
        source_document_id=record.source_document_id,
        state=record.state if state is None else state,
        ownership_evidence=record.ownership_evidence,
        outputs_verified=(
            record.outputs_verified if outputs_verified is None else outputs_verified
        ),
        retain_reason=record.retain_reason if retain_reason is None else retain_reason,
    )


def remote_delete_available() -> bool:
    """Grok ``deleteAgents`` exists in 0.79.4 and is not wired up."""

    return False
