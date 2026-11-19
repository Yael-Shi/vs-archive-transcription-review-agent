"""Submit one Codex Desktop turn and read its attributed reply."""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from typing import Any

from transcription_review_agent.codex_desktop.focus import focus_verdict
from transcription_review_agent.codex_desktop.framing import FrameError
from transcription_review_agent.codex_desktop.journal import AttemptJournal
from transcription_review_agent.codex_desktop.models import (
    CodexRequest,
    CodexResult,
    CodexStatus,
    FocusObservation,
)
from transcription_review_agent.codex_desktop.pipe import CodexTransport, PipeError

_NAMESPACE = "codex_app"
_TURN_PAGE_LIMIT = 5
_MAX_TURN_PAGES = 20


class CodexProtocolError(ValueError):
    """A Desktop response cannot be attributed to the requested turn."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "ambiguous_turn_attribution",
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.recoverable = recoverable


def new_call_id() -> str:
    """One tools/call envelope id. This is not a Desktop turn id."""

    return uuid.uuid4().hex


def tool_call_params(
    request: CodexRequest,
    *,
    tool: str,
    arguments: dict[str, Any],
    call_id: str,
) -> dict[str, Any]:
    """Address the stable thread with a fresh tool call id.

    ``turnId`` here is only the app-tool envelope. The model turn id comes
    back from ``read_thread``.
    """

    if not call_id.strip():
        raise CodexProtocolError("tool call id is required")
    return {
        "arguments": arguments,
        "callId": call_id,
        "callerSource": "codex",
        "namespace": _NAMESPACE,
        "threadId": request.thread_id,
        "tool": tool,
        "turnId": f"{call_id}-envelope",
    }


def _user_message_text(turn: dict[str, Any]) -> str:
    items = turn.get("items")
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "userMessage":
            continue
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
        if isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts)


def _final_answer(turn: dict[str, Any]) -> str | None:
    items = turn.get("items")
    if not isinstance(items, list):
        return None
    finals = [
        item.get("text")
        for item in items
        if isinstance(item, dict)
        and item.get("type") == "agentMessage"
        and item.get("phase") == "final_answer"
        and isinstance(item.get("text"), str)
    ]
    if not finals:
        return None
    text = finals[-1]
    return text if isinstance(text, str) else None


_DELEGATION_INPUT = re.compile(r"<input>(.*?)</input>", re.DOTALL)


def _parent_turn_id(turn: dict[str, Any]) -> str:
    for key in ("parent_turn_id", "parentTurnId"):
        value = turn.get(key)
        if isinstance(value, str):
            return value
    return ""


def _delegation_inputs(turn: dict[str, Any]) -> list[str]:
    """Submitted prompt text carried on the send tool, not the assistant reply."""

    items = turn.get("items")
    if not isinstance(items, list):
        return []
    found: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "functionCallOutput":
            continue
        if item.get("name") != "send_message_to_thread":
            continue
        output = item.get("output")
        text = output if isinstance(output, str) else ""
        if isinstance(output, dict) and isinstance(output.get("text"), str):
            text = output["text"]
        found.extend(_DELEGATION_INPUT.findall(text))
    return found


def _turn_matches(turn: dict[str, Any], request_id: str, dispatch_turn_id: str) -> bool:
    if dispatch_turn_id and _parent_turn_id(turn) == dispatch_turn_id:
        return True
    if request_id in _user_message_text(turn):
        return True
    return any(request_id in item for item in _delegation_inputs(turn))


def attributed_reply(
    thread_body: dict[str, Any],
    request_id: str,
    *,
    dispatch_turn_id: str = "",
) -> tuple[str, str] | None:
    """Return the Desktop turn id and final answer bound to this send.

    A turn matches the recorded send envelope, a user message that contains
    the request id, or the ``send_message_to_thread`` delegation input that
    contains it. Assistant text, commands, and file paths do not match.
    """

    turns = thread_body.get("turns")
    if not isinstance(turns, list):
        return None
    matches = [
        turn
        for turn in turns
        if isinstance(turn, dict) and _turn_matches(turn, request_id, dispatch_turn_id)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise CodexProtocolError("more than one Desktop turn matches the send")
    turn = matches[0]
    turn_id = turn.get("id")
    if not isinstance(turn_id, str) or not turn_id:
        raise CodexProtocolError("matching user turn has no Desktop turn id")
    if turn.get("status") != "completed":
        return None
    reply = _final_answer(turn)
    if reply is None:
        raise CodexProtocolError(
            "completed turn has no final answer", reason="incomplete_output"
        )
    return turn_id, reply


def parse_thread_body(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("success") is not True:
        raise CodexProtocolError("read_thread was not successful", recoverable=True)
    items = result.get("contentItems")
    if not isinstance(items, list) or not items:
        raise CodexProtocolError("read_thread returned no content", recoverable=True)
    first = items[0]
    if not isinstance(first, dict) or not isinstance(first.get("text"), str):
        raise CodexProtocolError("read_thread content is not text", recoverable=True)
    try:
        body = json.loads(first["text"])
    except json.JSONDecodeError as exc:
        raise CodexProtocolError(
            "read_thread body is not JSON", recoverable=True
        ) from exc
    if not isinstance(body, dict):
        raise CodexProtocolError("read_thread body is not an object", recoverable=True)
    return body


def _next_cursor(body: dict[str, Any]) -> str | None:
    """Return the next page cursor, or None when this page ends the thread.

    ``read_thread`` reports ``page.hasMore`` and ``page.nextCursor``. A page
    that claims more turns without a cursor cannot be searched safely.
    """

    page = body.get("page")
    if not isinstance(page, dict) or page.get("hasMore") is not True:
        return None
    cursor = page.get("nextCursor")
    if not isinstance(cursor, str) or not cursor.strip():
        raise CodexProtocolError("read_thread page has no cursor")
    return cursor


class CodexDesktopClient:
    def __init__(
        self,
        transport: CodexTransport,
        journal: AttemptJournal,
        *,
        completion_timeout_s: float = 180,
        poll_s: float = 1,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._transport = transport
        self._journal = journal
        self._completion_timeout_s = completion_timeout_s
        self._poll_s = poll_s
        self._sleep = sleeper
        self._clock = clock

    def send(self, request: CodexRequest) -> CodexResult:
        problem = _validate(request)
        if problem:
            return CodexResult(
                status=CodexStatus.ERROR,
                sent=False,
                request_id=request.request_id,
                thread_id=request.thread_id,
                reason=problem,
            )
        replay = self._journal.reserve(request)
        if replay is not None:
            if replay.sent and replay.status in {
                CodexStatus.UNCERTAIN,
                CodexStatus.TIMEOUT,
            }:
                return self._recover(request, replay)
            return replay
        params = self._params(
            request,
            tool="send_message_to_thread",
            arguments={
                "hostId": "local",
                "prompt": request.message,
                "threadId": request.thread_id,
            },
        )
        envelope = str(params["turnId"])
        try:
            call = self._transport.call("tools/call", params, timeout_s=30)
        except (PipeError, FrameError, TimeoutError, OSError):
            result = _result(
                request,
                CodexStatus.UNCERTAIN,
                True,
                reason="send_uncertain",
                dispatch_turn_id=envelope,
            )
            self._journal.finish(result)
            return result
        verdict = focus_verdict(call.focus)
        response = call.response
        if "error" in response or _tool_failed(response):
            status = (
                CodexStatus.UNCERTAIN if _ambiguous(response) else CodexStatus.ERROR
            )
            sent = status is CodexStatus.UNCERTAIN
            result = _result(
                request,
                status,
                sent,
                reason=_rpc_reason(response),
                dispatch_turn_id=envelope,
                focus=call.focus if verdict == "preserved" else call.focus,
            )
            result = _with_focus(result, verdict)
            self._journal.finish(result)
            return result
        if verdict == "codex_activated":
            result = _result(
                request,
                CodexStatus.BLOCKED,
                True,
                reason="codex_window_activated",
                dispatch_turn_id=envelope,
                focus_preserved=False,
            )
            self._journal.finish(result)
            return result
        return self._await_reply(
            request,
            focus_ok=verdict == "preserved",
            dispatch_turn_id=envelope,
        )

    def _recover(self, request: CodexRequest, prior: CodexResult) -> CodexResult:
        """Read an already submitted turn. This does not send again."""

        return self._await_reply(
            request,
            focus_ok=prior.focus_preserved is not False,
            replayed=True,
            prior=prior,
            dispatch_turn_id=prior.dispatch_turn_id,
        )

    def _await_reply(
        self,
        request: CodexRequest,
        *,
        focus_ok: bool,
        replayed: bool = False,
        prior: CodexResult | None = None,
        dispatch_turn_id: str = "",
    ) -> CodexResult:
        deadline = self._clock() + self._completion_timeout_s
        while True:
            try:
                body, read_focus_ok = self._read(request)
            except _FocusBlocked:
                result = _result(
                    request,
                    CodexStatus.BLOCKED,
                    True,
                    reason="codex_window_activated",
                    replayed=replayed,
                    dispatch_turn_id=dispatch_turn_id,
                    focus_preserved=False,
                )
                self._journal.finish(result)
                return result
            except CodexProtocolError as exc:
                if not exc.recoverable:
                    result = _result(
                        request,
                        CodexStatus.ERROR,
                        True,
                        reason=exc.reason,
                        replayed=replayed,
                        dispatch_turn_id=dispatch_turn_id,
                        focus_preserved=focus_ok,
                    )
                    self._journal.finish(result)
                    return result
                if self._clock() >= deadline:
                    if prior is not None and prior.status is CodexStatus.UNCERTAIN:
                        return prior
                    result = _result(
                        request,
                        CodexStatus.UNCERTAIN,
                        True,
                        reason="completion_uncertain",
                        replayed=replayed,
                        dispatch_turn_id=dispatch_turn_id,
                        focus_preserved=focus_ok,
                    )
                    self._journal.finish(result)
                    return result
                self._sleep(self._poll_s)
                continue
            except (PipeError, FrameError, TimeoutError, OSError, json.JSONDecodeError):
                if self._clock() >= deadline:
                    if prior is not None and prior.status is CodexStatus.UNCERTAIN:
                        return prior
                    result = _result(
                        request,
                        CodexStatus.UNCERTAIN,
                        True,
                        reason="completion_uncertain",
                        replayed=replayed,
                        dispatch_turn_id=dispatch_turn_id,
                        focus_preserved=focus_ok,
                    )
                    self._journal.finish(result)
                    return result
                self._sleep(self._poll_s)
                continue
            focus_ok = focus_ok and read_focus_ok
            try:
                matched = attributed_reply(
                    body, request.request_id, dispatch_turn_id=dispatch_turn_id
                )
            except CodexProtocolError as exc:
                result = _result(
                    request,
                    CodexStatus.ERROR,
                    True,
                    reason=exc.reason,
                    replayed=replayed,
                    dispatch_turn_id=dispatch_turn_id,
                    focus_preserved=focus_ok,
                )
                self._journal.finish(result)
                return result
            if matched is not None:
                turn_id, reply = matched
                result = _result(
                    request,
                    CodexStatus.SUCCESS,
                    True,
                    reply=reply,
                    raw_reply=reply,
                    turn_id=turn_id,
                    reason="recovered" if replayed else "",
                    replayed=replayed,
                    dispatch_turn_id=dispatch_turn_id,
                    focus_preserved=focus_ok,
                )
                self._journal.finish(result)
                return result
            if self._clock() >= deadline:
                result = _result(
                    request,
                    CodexStatus.TIMEOUT,
                    True,
                    reason="completion_timeout",
                    replayed=replayed,
                    dispatch_turn_id=dispatch_turn_id,
                    focus_preserved=focus_ok,
                )
                self._journal.finish(result)
                return result
            self._sleep(self._poll_s)

    def _params(
        self, request: CodexRequest, *, tool: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        return tool_call_params(
            request, tool=tool, arguments=arguments, call_id=new_call_id()
        )

    def _read(self, request: CodexRequest) -> tuple[dict[str, Any], bool]:
        """Read newest-first pages until the thread ends or the page bound.

        A missing cursor, a repeated cursor, or a page bound that still
        reports more turns fails closed. The newest page is not assumed to
        be the submitted turn.
        """

        cursor: str | None = None
        seen_cursors: set[str] = set()
        seen_ids: set[str] = set()
        turns: list[object] = []
        focus_ok = True
        for _page in range(_MAX_TURN_PAGES):
            body, preserved = self._read_page(request, cursor)
            focus_ok = focus_ok and preserved
            page_turns = body.get("turns")
            if not isinstance(page_turns, list):
                raise CodexProtocolError(
                    "read_thread returned no turns", recoverable=True
                )
            for turn in page_turns:
                if isinstance(turn, dict) and isinstance(turn.get("id"), str):
                    turn_id = turn["id"]
                    if turn_id in seen_ids:
                        raise CodexProtocolError("read_thread repeated a turn id")
                    seen_ids.add(turn_id)
                turns.append(turn)
            nxt = _next_cursor(body)
            if nxt is None:
                merged = dict(body)
                merged["turns"] = turns
                return merged, focus_ok
            if nxt in seen_cursors:
                raise CodexProtocolError("read_thread repeated a page cursor")
            seen_cursors.add(nxt)
            cursor = nxt
        raise CodexProtocolError("read_thread page bound ended before the thread")

    def _read_page(
        self, request: CodexRequest, cursor: str | None
    ) -> tuple[dict[str, Any], bool]:
        arguments: dict[str, Any] = {
            "hostId": "local",
            "includeOutputs": True,
            "maxOutputCharsPerItem": 16000,
            "threadId": request.thread_id,
            "turnLimit": _TURN_PAGE_LIMIT,
        }
        if cursor is not None:
            arguments["cursor"] = cursor
        call = self._transport.call(
            "tools/call",
            self._params(request, tool="read_thread", arguments=arguments),
            timeout_s=20,
        )
        if focus_verdict(call.focus) == "codex_activated":
            raise _FocusBlocked
        response = call.response
        if "error" in response:
            raise PipeError(_rpc_reason(response))
        result = response.get("result")
        if not isinstance(result, dict):
            raise CodexProtocolError("tools/call result is missing", recoverable=True)
        preserved = focus_verdict(call.focus) == "preserved"
        return parse_thread_body(result), preserved


class _FocusBlocked(Exception):
    """The Desktop window became the foreground during a pipe call."""


def _validate(request: CodexRequest) -> str:
    if not request.thread_id.strip() or not request.request_id.strip():
        return "thread_id and request_id are required"
    if request.request_id not in request.message:
        return "request_id must occur inside the message"
    return ""


def _tool_failed(response: dict[str, Any]) -> bool:
    result = response.get("result")
    return isinstance(result, dict) and result.get("success") is False


def _ambiguous(response: dict[str, Any]) -> bool:
    error = response.get("error")
    if not isinstance(error, dict):
        return False
    return error.get("code") == -32000


def _rpc_reason(response: dict[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return str(error["message"])
    return "desktop tool call failed"


def _with_focus(result: CodexResult, verdict: str) -> CodexResult:
    if verdict != "codex_activated":
        return CodexResult(
            status=result.status,
            sent=result.sent,
            request_id=result.request_id,
            thread_id=result.thread_id,
            reply=result.reply,
            raw_reply=result.raw_reply,
            turn_id=result.turn_id,
            dispatch_turn_id=result.dispatch_turn_id,
            reason=result.reason,
            replayed=result.replayed,
            focus_preserved=verdict == "preserved",
        )
    return CodexResult(
        status=CodexStatus.BLOCKED,
        sent=result.sent,
        request_id=result.request_id,
        thread_id=result.thread_id,
        reason="codex_window_activated",
        dispatch_turn_id=result.dispatch_turn_id,
        focus_preserved=False,
    )


def _result(
    request: CodexRequest,
    status: CodexStatus,
    sent: bool,
    *,
    reply: str = "",
    raw_reply: str = "",
    turn_id: str = "",
    reason: str = "",
    replayed: bool = False,
    dispatch_turn_id: str = "",
    focus_preserved: bool | None = None,
    focus: FocusObservation | None = None,
) -> CodexResult:
    del focus
    return CodexResult(
        status=status,
        sent=sent,
        request_id=request.request_id,
        thread_id=request.thread_id,
        reply=reply,
        raw_reply=raw_reply,
        turn_id=turn_id,
        dispatch_turn_id=dispatch_turn_id,
        reason=reason,
        replayed=replayed,
        focus_preserved=focus_preserved,
    )
