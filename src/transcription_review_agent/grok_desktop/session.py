"""One Grok send, with no automatic retry after an uncertain delivery."""

from __future__ import annotations

import time
from typing import Any, Protocol

from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)
from transcription_review_agent.grok_desktop.transcript import (
    MAX_REPLY,
    PREVIEW_CHARS,
    as_list,
    classify_preflight,
    focus_expression,
    greeting_view,
    judge_turn,
    poll_expression,
    preflight_expression,
    select_expression,
    submit_expression,
    validate_request,
)


class CdpSession(Protocol):
    def evaluate(self, expression: str) -> Any: ...

    def insert_text(self, text: str) -> None: ...

    def close(self) -> None: ...


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SteadyClock:
    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def run_exchange(
    session: CdpSession,
    request: ExchangeRequest,
    clock: Clock,
    *,
    poll_timeout: float = 90.0,
    poll_interval: float = 0.5,
) -> ExchangeResult:
    invalid = validate_request(
        agent_id=request.agent_id,
        bot_name=request.bot_name,
        message=request.message,
        request_id=request.request_id,
    )
    if invalid:
        return _finish(ExchangeStatus.ERROR, False, request, reason=invalid)

    seen: list[tuple[str, ...]] = []
    submitted = False
    submit_dispatched = False
    try:
        selected = session.evaluate(select_expression(request.agent_id))
        if not isinstance(selected, dict) or selected.get("ok") is not True:
            reason = "bot_missing"
            if isinstance(selected, dict) and selected.get("reason"):
                reason = str(selected["reason"])
            return _finish(ExchangeStatus.ERROR, False, request, reason=reason)
        clock.sleep(1.5)
        preflight_js = preflight_expression(
            request.agent_id, request.bot_name, request.request_id
        )
        preflight = session.evaluate(preflight_js)
        refusal = classify_preflight(preflight)
        if refusal:
            return _finish(ExchangeStatus.ERROR, False, request, reason=refusal)
        clock.sleep(1.0)
        confirm = session.evaluate(preflight_js)
        confirm_refusal = classify_preflight(confirm)
        if confirm_refusal:
            return _finish(ExchangeStatus.ERROR, False, request, reason=confirm_refusal)
        if (
            not isinstance(preflight, dict)
            or not isinstance(confirm, dict)
            or greeting_view(preflight) != greeting_view(confirm)
        ):
            return _finish(
                ExchangeStatus.ERROR, False, request, reason="greeting_unstable"
            )
        focused = session.evaluate(focus_expression())
        if not isinstance(focused, dict) or focused.get("ok") is not True:
            return _finish(ExchangeStatus.ERROR, False, request, reason="focus_failed")
        try:
            session.insert_text(request.message)
        except Exception:
            return _finish(
                ExchangeStatus.UNCERTAIN,
                False,
                request,
                reason="insert_unconfirmed",
            )
        submit_dispatched = True
        clicked = session.evaluate(submit_expression(request.request_id))
        submit_dispatched = False
        if not isinstance(clicked, dict) or clicked.get("ok") is not True:
            reason = "submit_not_clicked"
            if isinstance(clicked, dict) and clicked.get("reason"):
                reason = str(clicked["reason"])
            return _finish(ExchangeStatus.ERROR, False, request, reason=reason)
        submitted = True
        deadline = clock.now() + poll_timeout
        last_poll: dict[str, Any] | None = None
        saw_active = False
        confirmed_settled: str | None = None
        while True:
            if clock.now() >= deadline:
                return _timeout(request, seen, last_poll)
            poll = session.evaluate(
                poll_expression(request.agent_id, request.request_id)
            )
            if not isinstance(poll, dict):
                return _finish(
                    ExchangeStatus.UNCERTAIN,
                    True,
                    request,
                    reason="poll_invalid",
                    states_seen=tuple(seen),
                )
            last_poll = poll
            user_hits = int(poll.get("userHits") or 0)
            states = tuple(as_list(poll.get("states")))
            seen.append(states)
            if len(seen) > 40:
                del seen[0]
            structure = poll.get("replyStructure")
            if structure == "inconclusive":
                return _finish(
                    ExchangeStatus.UNCERTAIN,
                    True,
                    request,
                    reason="reply_inconclusive",
                    states_seen=tuple(seen),
                )
            reply_value = poll.get("reply")
            raw_reply = (
                reply_value
                if structure == "prose" and isinstance(reply_value, str)
                else ""
            )
            reply_chars = poll.get("replyChars")
            reported_over_limit = poll.get("truncated") is True or (
                isinstance(reply_chars, int) and reply_chars > MAX_REPLY
            )
            if len(raw_reply) > MAX_REPLY or reported_over_limit:
                return _finish(
                    ExchangeStatus.ERROR,
                    True,
                    request,
                    reason=(
                        "reply_oversized"
                        if len(raw_reply) > MAX_REPLY
                        else "reply_incomplete"
                    ),
                    states_seen=tuple(seen),
                )
            reply = raw_reply
            if user_hits > 1:
                return _finish(
                    ExchangeStatus.UNCERTAIN,
                    True,
                    request,
                    reason="multiple_user_turns",
                    states_seen=tuple(seen),
                )
            phase, refusal = judge_turn(poll, request.agent_id)
            if refusal:
                return _finish(
                    ExchangeStatus.UNCERTAIN,
                    True,
                    request,
                    reason=refusal,
                    states_seen=tuple(seen),
                )
            if phase == "failed":
                return _finish(
                    ExchangeStatus.FAILED,
                    True,
                    request,
                    reason="generation_failed",
                    reply=reply,
                    raw_reply=raw_reply,
                    states_seen=tuple(seen),
                )
            if phase == "blocked":
                return _finish(
                    ExchangeStatus.BLOCKED,
                    True,
                    request,
                    reason="generation_blocked",
                    reply=reply,
                    raw_reply=raw_reply,
                    states_seen=tuple(seen),
                )
            if phase == "finished" and user_hits == 1 and reply:
                return _finish(
                    ExchangeStatus.SUCCESS,
                    True,
                    request,
                    reply=reply,
                    raw_reply=raw_reply,
                    states_seen=tuple(seen),
                )
            if phase == "finished" and user_hits == 1:
                return _finish(
                    ExchangeStatus.UNCERTAIN,
                    True,
                    request,
                    reason="empty_reply",
                    raw_reply=raw_reply,
                    states_seen=tuple(seen),
                )
            if phase == "running":
                saw_active = True
                confirmed_settled = None
            elif phase == "settled" and user_hits == 1 and reply:
                if saw_active or confirmed_settled == reply:
                    return _finish(
                        ExchangeStatus.SUCCESS,
                        True,
                        request,
                        reply=reply,
                        raw_reply=raw_reply,
                        states_seen=tuple(seen),
                    )
                confirmed_settled = reply
            else:
                confirmed_settled = None
            clock.sleep(poll_interval)
    except TimeoutError:
        if submitted or submit_dispatched:
            if submit_dispatched:
                return _finish(
                    ExchangeStatus.UNCERTAIN,
                    True,
                    request,
                    reason="submit_unconfirmed",
                    states_seen=tuple(seen),
                )
            return _timeout(request, seen, last_poll)
        return _finish(
            ExchangeStatus.TIMEOUT,
            False,
            request,
            reason="timeout_before_send",
            states_seen=tuple(seen),
        )
    except Exception:
        if submitted or submit_dispatched:
            return _finish(
                ExchangeStatus.UNCERTAIN,
                True,
                request,
                reason="session_lost_after_send" if submitted else "submit_unconfirmed",
                states_seen=tuple(seen),
            )
        raise


def _timeout(
    request: ExchangeRequest,
    seen: list[tuple[str, ...]],
    poll: dict[str, Any] | None,
) -> ExchangeResult:
    """Keep the last turn snapshot. The status stays uncertain and is not a retry."""

    reply = ""
    detail = ""
    if isinstance(poll, dict):
        text = poll.get("reply")
        # Only a turn-attributed prose snapshot is stored. An unattributed
        # string stays out of the reply field so it cannot be read as success.
        if (
            poll.get("attribution") == "turn"
            and poll.get("replyStructure") == "prose"
            and isinstance(text, str)
        ):
            reply = text
        chars = poll.get("replyChars")
        shown = chars if isinstance(chars, int) else len(reply)
        detail = (
            f"attribution={poll.get('attribution') or ''};"
            f"streaming={str(poll.get('streaming') is True).lower()};"
            f"pending={str(poll.get('pending') is True).lower()};"
            f"failed={str(poll.get('failed') is True).lower()};"
            f"busy={str(poll.get('busy') is True).lower()};"
            f"replyChars={shown}"
        )
    return _finish(
        ExchangeStatus.UNCERTAIN,
        True,
        request,
        reason="timeout_after_send",
        reply=reply,
        raw_reply=reply,
        states_seen=tuple(seen),
        detail=detail,
    )


def _finish(
    status: ExchangeStatus,
    sent: bool,
    request: ExchangeRequest,
    *,
    reason: str = "",
    reply: str = "",
    raw_reply: str = "",
    states_seen: tuple[tuple[str, ...], ...] = (),
    detail: str = "",
) -> ExchangeResult:
    return ExchangeResult(
        status=status,
        sent=sent,
        request_id=request.request_id,
        agent_id=request.agent_id,
        reply=reply,
        raw_reply=raw_reply,
        reply_preview=reply[:PREVIEW_CHARS],
        reason=reason,
        states_seen=states_seen,
        detail=detail,
    )
