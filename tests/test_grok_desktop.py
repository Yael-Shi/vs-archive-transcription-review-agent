"""Mocked Grok Desktop adapter coverage. These tests do not launch Grok."""

from __future__ import annotations

import inspect
import io
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from transcription_review_agent.grok_desktop.adapter import GrokDesktopAdapter
from transcription_review_agent.grok_desktop.lifecycle import (
    LifecycleError,
    apply_transition,
    can_transition,
    cleanup_eligible,
    note_outputs_verified,
    remote_delete_available,
)
from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)
from transcription_review_agent.grok_desktop.profile import (
    ISOLATED_MARKER,
    ProfileError,
    assert_isolated_profile,
    local_read_path,
    read_permission,
)
from transcription_review_agent.grok_desktop.registry import (
    DuplicateResourceError,
    RegistryError,
    ResourceKind,
    ResourceRecord,
    ResourceRegistry,
    ResourceState,
)
from transcription_review_agent.grok_desktop.session import run_exchange
from transcription_review_agent.grok_desktop.transcript import (
    MAX_REPLY,
    PREVIEW_CHARS,
    classify_preflight,
    completion_phase,
    poll_expression,
    preflight_expression,
    structural_message_text,
)
from transcription_review_agent.grok_desktop.transport import (
    ASSETS,
    HOST_READY_TIMEOUT_SECONDS,
    HOST_SHUTDOWN_TIMEOUT_SECONDS,
    PowershellCdpSession,
    ProcessIdentity,
    TransportError,
    classify_host_failure,
    decode_protocol_line,
    host_command,
    windows_host_env,
)

AGENT = "2efb9d31-fab8-4d9e-9be1-95b94f6bcdfa"
OTHER = "5286584c-cd62-43b6-a86a-fad5cac56fc9"
NAME = "VS Archive Integration Test 01"
REQUEST_ID = "INTEGRATION-01-B7D2"
MESSAGE = "Reply with exactly INTEGRATION-01-B7D2 and nothing else."


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


def greeting(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "active": True,
        "nameMatch": True,
        "composerEmpty": True,
        "userCount": 0,
        "userRows": 0,
        "assistantCount": 1,
        "msgCount": 1,
        "tagged": False,
        "state": "idle",
        "transcript": "ready",
        "roles": ["assistant"],
        "streaming": False,
        "pending": False,
        "failed": False,
        "busy": False,
        "greetingTruncated": False,
        "greeting": "hello",
    }
    value.update(overrides)
    return value


class FakeSession:
    def __init__(
        self,
        polls: list[object] | None = None,
        *,
        select: object | None = None,
        preflight: object | None = None,
        focus: object | None = None,
        submit: object | None = None,
        insert_error: BaseException | None = None,
    ) -> None:
        self.polls = list(polls or [])
        self.select = {"ok": True} if select is None else select
        self.preflight = greeting() if preflight is None else preflight
        self.focus = {"ok": True} if focus is None else focus
        self.submit = {"ok": True} if submit is None else submit
        self.insert_error = insert_error
        self.evals: list[str] = []
        self.inserts: list[str] = []
        self.closed = 0

    def evaluate(self, expression: str) -> object:
        self.evals.append(expression)
        if "grok-desktop:select" in expression:
            if isinstance(self.select, Exception):
                raise self.select
            return self.select
        if "grok-desktop:preflight" in expression:
            if isinstance(self.preflight, list):
                if not self.preflight:
                    raise AssertionError("preflight exhausted")
                return self.preflight.pop(0)
            return self.preflight
        if "grok-desktop:focus" in expression:
            return self.focus
        if "grok-desktop:submit" in expression:
            if isinstance(self.submit, Exception):
                raise self.submit
            return self.submit
        if "grok-desktop:poll" in expression:
            if not self.polls:
                raise TimeoutError("poll")
            item = self.polls.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        raise AssertionError(expression)

    def insert_text(self, text: str) -> None:
        self.inserts.append(text)
        if self.insert_error is not None:
            raise self.insert_error

    def close(self) -> None:
        self.closed += 1


def request(**overrides: object) -> ExchangeRequest:
    payload: dict[str, object] = {
        "agent_id": AGENT,
        "bot_name": NAME,
        "message": MESSAGE,
        "request_id": REQUEST_ID,
    }
    payload.update(overrides)
    return ExchangeRequest(
        agent_id=str(payload["agent_id"]),
        bot_name=str(payload["bot_name"]),
        message=str(payload["message"]),
        request_id=str(payload["request_id"]),
        source_document_id=(
            None
            if "source_document_id" not in payload
            else (
                None
                if payload["source_document_id"] is None
                else str(payload["source_document_id"])
            )
        ),
    )


def steps(session: FakeSession) -> list[str]:
    found: list[str] = []
    for expression in session.evals:
        for name in ("select", "preflight", "focus", "submit", "poll"):
            if f"grok-desktop:{name}" in expression:
                found.append(name)
    return found


def turn_poll(states: object, reply: str = "", **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "userHits": 1,
        "activeAgent": AGENT,
        "attribution": "turn",
        "states": states,
        "unrelatedStates": [],
        "truncated": False,
        "reply": reply,
        "replyChars": len(reply),
        "replyStructure": "prose",
    }
    payload.update(extra)
    return payload


def success_poll(reply: str | None = None) -> dict[str, object]:
    text = reply if reply is not None else REQUEST_ID
    return turn_poll(["idle", "finished", "idle"], text)


def test_correct_bot_extracts_reply_without_chrome() -> None:
    session = FakeSession(polls=[success_poll()])
    result = run_exchange(session, request(), Clock())
    assert result.status is ExchangeStatus.SUCCESS
    assert result.sent is True
    assert result.reply == REQUEST_ID
    assert result.raw_reply == REQUEST_ID
    assert result.reply_preview == REQUEST_ID
    assert session.inserts == [MESSAGE]
    assert steps(session).count("submit") == 1


def test_wrong_bot_and_missing_bot_do_not_send() -> None:
    wrong = FakeSession(preflight=greeting(nameMatch=False))
    wrong_result = run_exchange(wrong, request(), Clock())
    assert wrong_result.status is ExchangeStatus.ERROR
    assert wrong_result.reason == "wrong_bot"
    assert wrong.inserts == []

    missing = FakeSession(select={"ok": False, "reason": "missing"})
    missing_result = run_exchange(missing, request(agent_id=OTHER), Clock())
    assert missing_result.reason == "missing"
    assert steps(missing) == ["select"]


def test_not_idle_and_focus_failure_stop_before_send() -> None:
    idle = run_exchange(
        FakeSession(preflight=greeting(state="thinking")), request(), Clock()
    )
    assert idle.reason == "not_idle"
    assert idle.sent is False
    focused = FakeSession(focus={"ok": False})
    assert run_exchange(focused, request(), Clock()).reason == "focus_failed"
    assert focused.inserts == []


def test_unexpected_draft_stops_before_send() -> None:
    session = FakeSession(preflight=greeting(composerEmpty=False))
    result = run_exchange(session, request(), Clock())
    assert result.reason == "unexpected_draft"
    assert result.sent is False
    assert "focus" not in steps(session)


def test_greeting_is_allowed_and_prior_user_turns_are_not() -> None:
    assert classify_preflight(greeting()) is None
    session = FakeSession(
        preflight=greeting(
            userCount=1, roles=["assistant", "user"], msgCount=2, assistantCount=1
        )
    )
    result = run_exchange(session, request(), Clock())
    assert result.reason == "prior_user_turns"
    assert session.inserts == []
    rows = run_exchange(FakeSession(preflight=greeting(userRows=1)), request(), Clock())
    assert rows.reason == "prior_user_turns"
    assert classify_preflight(greeting(roles="assistant")) is None
    hidden_user = classify_preflight(
        greeting(roles=["assistant", "user"], assistantCount=2, msgCount=2)
    )
    assert hidden_user == "prior_user_turns"


def test_one_or_more_stable_greetings_can_be_sent_once() -> None:
    single = greeting()
    assert classify_preflight(single) is None
    several = greeting(
        assistantCount=2,
        msgCount=2,
        roles=["assistant", "assistant"],
        greeting="היי\nלפי השם",
    )
    assert classify_preflight(several) is None
    expression = preflight_expression(AGENT, NAME, REQUEST_ID)
    assert "msgs.slice(0, 6)" not in expression
    assert "greeting: greeting" in expression
    assert "data-pending" in expression
    assert "data-failed" in expression
    assert "aria-busy" in expression
    assert "aria-label" in expression
    assert "transcript = 'virtualized'" in expression
    assert "transcript = 'ambiguous'" in expression
    sent = run_exchange(
        FakeSession(preflight=several, polls=[success_poll(REQUEST_ID)]),
        request(),
        Clock(),
    )
    assert sent.status is ExchangeStatus.SUCCESS
    assert sent.sent is True


def test_changing_incomplete_and_unknown_greetings_do_not_send() -> None:
    changed = run_exchange(
        FakeSession(preflight=[greeting(greeting="one"), greeting(greeting="two")]),
        request(),
        Clock(),
    )
    assert changed.reason == "greeting_unstable"
    assert changed.sent is False
    streaming = run_exchange(
        FakeSession(preflight=greeting(streaming=True)),
        request(),
        Clock(),
    )
    assert streaming.reason == "greeting_incomplete"
    assert streaming.sent is False
    missing_flag = greeting()
    del missing_flag["busy"]
    assert classify_preflight(missing_flag) == "greeting_incomplete"
    assert classify_preflight(greeting(pending=True)) == "greeting_incomplete"
    assert classify_preflight(greeting(failed=True)) == "greeting_incomplete"
    assert classify_preflight(greeting(busy=True)) == "greeting_incomplete"
    assert (
        classify_preflight(
            greeting(roles=["assistant", "system"], assistantCount=2, msgCount=2)
        )
        == "unknown_role"
    )
    assert classify_preflight(greeting(transcript="virtualized")) == (
        "virtualized_transcript"
    )
    assert (
        classify_preflight(greeting(transcript="ambiguous")) == "ambiguous_transcript"
    )


def test_prior_tag_and_unidentified_transcript_stop() -> None:
    tagged = run_exchange(
        FakeSession(preflight=greeting(tagged=True)),
        request(),
        Clock(),
    )
    assert tagged.reason == "prior_test_activity"
    empty = run_exchange(
        FakeSession(preflight=greeting(assistantCount=0, msgCount=0, roles=[])),
        request(),
        Clock(),
    )
    assert empty.reason == "greeting_not_identified"


def test_submit_happens_once_across_completion_states() -> None:
    session = FakeSession(
        polls=[
            turn_poll(["thinking"]),
            turn_poll(["writing", "idle"]),
            success_poll("INTEGRATION-01-B7D2"),
        ]
    )
    result = run_exchange(session, request(), Clock())
    assert result.status is ExchangeStatus.SUCCESS
    assert result.states_seen[0] == ("thinking",)
    assert result.states_seen[1] == ("writing", "idle")
    assert "finished" in result.states_seen[2]
    assert steps(session).count("submit") == 1
    assert len(session.inserts) == 1


def test_timeout_before_send_is_not_uncertain() -> None:
    session = FakeSession(select=TimeoutError("cdp"))
    result = run_exchange(session, request(), Clock())
    assert result.status is ExchangeStatus.TIMEOUT
    assert result.sent is False
    assert result.reason == "timeout_before_send"
    assert session.inserts == []


def test_timeout_after_send_is_uncertain() -> None:
    session = FakeSession(polls=[turn_poll(["thinking"])])
    result = run_exchange(
        session, request(), Clock(), poll_timeout=1.0, poll_interval=0.5
    )
    assert result.status is ExchangeStatus.UNCERTAIN
    assert result.sent is True
    assert result.reason == "timeout_after_send"
    assert steps(session).count("submit") == 1
    assert steps(session).count("poll") == 2


def test_failed_and_blocked_generation_stop() -> None:
    failed = run_exchange(
        FakeSession(polls=[turn_poll(["finished", "failed"], "x")]),
        request(),
        Clock(),
    )
    assert failed.status is ExchangeStatus.FAILED
    assert failed.sent is True
    blocked = run_exchange(
        FakeSession(polls=[turn_poll(["blocked"])]),
        request(),
        Clock(),
    )
    assert blocked.status is ExchangeStatus.BLOCKED


def test_submit_timeout_is_uncertain_and_is_not_retried() -> None:
    session = FakeSession(submit=TimeoutError("submit"))
    result = run_exchange(session, request(), Clock())
    assert result.status is ExchangeStatus.UNCERTAIN
    assert result.sent is True
    assert result.reason == "submit_unconfirmed"
    assert steps(session).count("submit") == 1
    assert "poll" not in steps(session)


def test_lost_session_after_send_stays_uncertain() -> None:
    result = run_exchange(
        FakeSession(polls=[RuntimeError("pipe")]),
        request(),
        Clock(),
    )
    assert result.status is ExchangeStatus.UNCERTAIN
    assert result.sent is True
    assert result.reason == "session_lost_after_send"


def test_multiple_user_turns_and_empty_finished_reply_are_uncertain() -> None:
    many = run_exchange(
        FakeSession(polls=[turn_poll(["finished"], REQUEST_ID, userHits=2)]),
        request(),
        Clock(),
    )
    assert many.status is ExchangeStatus.UNCERTAIN
    assert many.reason == "multiple_user_turns"
    empty = run_exchange(
        FakeSession(polls=[turn_poll(["finished"], "")]),
        request(),
        Clock(),
    )
    assert empty.status is ExchangeStatus.UNCERTAIN
    assert empty.reason == "empty_reply"
    assert empty.reply == ""
    assert empty.raw_reply == ""


def test_single_finished_state_from_powershell_still_completes() -> None:
    result = run_exchange(
        FakeSession(polls=[turn_poll("finished", REQUEST_ID)]),
        request(),
        Clock(),
    )
    assert result.status is ExchangeStatus.SUCCESS
    assert result.states_seen == (("finished",),)
    assert result.reply == REQUEST_ID


def test_insert_timeout_is_uncertain_without_a_second_submit() -> None:
    session = FakeSession(insert_error=TimeoutError("insert"))
    result = run_exchange(session, request(), Clock())
    assert result.status is ExchangeStatus.UNCERTAIN
    assert result.reason == "insert_unconfirmed"
    assert result.sent is False
    assert "submit" not in steps(session)


def test_completion_phase_priority() -> None:
    assert completion_phase(["idle", "finished", "idle"]) == "finished"
    assert completion_phase(["finished", "failed"]) == "failed"
    assert completion_phase(["finished", "blocked"]) == "blocked"
    assert completion_phase(["thinking"]) == "running"
    assert completion_phase(["finished", "thinking"]) == "running"
    assert completion_phase(["finished", "new-running-state"]) == "unknown"
    assert completion_phase(["idle"]) == "idle"


def test_structural_reply_keeps_literal_text_and_skips_chrome_nodes() -> None:
    prose = (
        f"thinking\n14:19\n\n  {NAME}  \nDéjà  vu, la deuxième ligne.\nThe third line."
    )
    html = (
        '<div class="sand-message" data-role="assistant">'
        f"<span hidden>{NAME}</span><time>14:19</time>"
        f'<div class="sand-message-prose">{prose}</div>'
        '<div class="sand-activity-mark" data-activity="thinking" '
        'data-grok-state="thinking">thinking</div></div>'
    )
    text, kind = structural_message_text(html)
    assert kind == "prose"
    assert text == prose
    mixed = (
        '<div class="sand-message" data-role="assistant">'
        f"{NAME}\n14:19\nthinking\n{REQUEST_ID}</div>"
    )
    assert structural_message_text(mixed) == ("", "inconclusive")
    several = (
        '<div class="sand-message" data-role="assistant">'
        '<div class="sand-message-prose">one</div>'
        '<div class="sand-message-prose">two</div></div>'
    )
    assert structural_message_text(several) == ("", "inconclusive")


def _prose(inner: str) -> str:
    return (
        '<div class="sand-message" data-role="assistant">'
        f'<div class="sand-message-prose">{inner}</div></div>'
    )


def test_paragraph_boundaries_match_the_prose_walker() -> None:
    assert structural_message_text(_prose("<p>Bonjour</p><p>Mon ami</p>")) == (
        "Bonjour\n\nMon ami",
        "prose",
    )
    assert structural_message_text(_prose("<p>Bonjour</p><p></p><p>Mon ami</p>")) == (
        "Bonjour\n\n\n\nMon ami",
        "prose",
    )
    assert structural_message_text(_prose("<div>Bonjour</div><div>Mon ami</div>")) == (
        "Bonjour\n\nMon ami",
        "prose",
    )
    assert structural_message_text(
        _prose("<p>Bon<strong>jour</strong> <em>ami</em></p>")
    ) == ("Bonjour ami", "prose")
    assert structural_message_text(_prose("<p>Line<br>next</p>")) == (
        "Line\nnext",
        "prose",
    )
    assert structural_message_text(_prose("<p>  spaced \n text  </p>")) == (
        "  spaced \n text  ",
        "prose",
    )
    assert structural_message_text(_prose("<p></p>")) == ("", "prose")
    assert structural_message_text(_prose("<div>Bonjour<p>X</p></div>"))[1] == (
        "inconclusive"
    )
    assert structural_message_text(_prose("<p>A</p><br><p>B</p>"))[1] == "inconclusive"
    assert structural_message_text(_prose("<table><tr><td>A</td></tr></table>")) == (
        "",
        "inconclusive",
    )
    expression = poll_expression(AGENT, REQUEST_ID)
    assert "P:1" in expression
    assert "blocks.join('\\n\\n')" in expression
    assert "full == null" in expression


def test_adapter_closes_session_and_does_not_register_manual_bots(
    tmp_path: Path,
) -> None:
    session = FakeSession(polls=[success_poll()])
    registry = ResourceRegistry(tmp_path / "registry.json")

    adapter = GrokDesktopAdapter(
        registry=registry,
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: session,
        check_profile=lambda: None,
        clock=Clock(),
    )
    result = adapter.exchange(request(source_document_id="doc-1"))
    assert result.status is ExchangeStatus.SUCCESS
    assert session.closed == 1
    assert registry._resources == {}
    saved = (tmp_path / "diag" / f"{REQUEST_ID}.json").read_text(encoding="utf-8")
    assert MESSAGE not in saved
    assert REQUEST_ID in saved
    assert "raw_reply" in saved
    assert saved.count(REQUEST_ID) >= 1

    def refuse() -> FakeSession:
        raise AssertionError("replay must not launch")

    replay_adapter = GrokDesktopAdapter(
        registry=ResourceRegistry(tmp_path / "registry.json"),
        diagnostics_dir=tmp_path / "diag",
        open_session=refuse,
        check_profile=lambda: None,
        clock=Clock(),
    )
    replay = replay_adapter.exchange(request(source_document_id="doc-1"))
    assert replay.replayed is True
    assert replay.status is ExchangeStatus.SUCCESS
    assert replay.reply == REQUEST_ID


def test_started_attempt_survives_restart_without_another_send(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    first = ResourceRegistry(path)
    assert first.begin_attempt(request()) is None

    def refuse() -> FakeSession:
        raise AssertionError("launch")

    adapter = GrokDesktopAdapter(
        registry=ResourceRegistry(path),
        diagnostics_dir=tmp_path / "diag",
        open_session=refuse,
        check_profile=lambda: None,
    )
    result = adapter.exchange(request())
    assert result.replayed is True
    assert result.status is ExchangeStatus.UNCERTAIN
    assert result.sent is True


def test_duplicate_provider_id_is_rejected(tmp_path: Path) -> None:
    registry = ResourceRegistry(tmp_path / "registry.json")
    registry.register_owned(
        local_id="local-1",
        provider="grok",
        kind=ResourceKind.GROK_BOT,
        provider_id=AGENT,
        ownership_evidence="created by the isolated pilot",
        source_document_id="doc-9",
    )
    with pytest.raises(DuplicateResourceError):
        registry.register_owned(
            local_id="local-2",
            provider="grok",
            kind=ResourceKind.GROK_BOT,
            provider_id=AGENT,
            ownership_evidence="again",
        )
    with pytest.raises(DuplicateResourceError):
        registry.register_owned(
            local_id="local-1",
            provider="grok",
            kind=ResourceKind.GROK_CHAT,
            provider_id="chat-other",
            ownership_evidence="same local id",
        )
    reopened = ResourceRegistry(tmp_path / "registry.json")
    assert reopened.get("local-1").source_document_id == "doc-9"
    assert reopened.get("local-1").state is ResourceState.ACTIVE


def test_later_registry_write_keeps_an_inflight_attempt(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    holder = ResourceRegistry(path)
    other = ResourceRegistry(path)
    assert holder.begin_attempt(request()) is None
    other.register_owned(
        local_id="owned-1",
        provider="grok",
        kind=ResourceKind.GROK_BOT,
        provider_id=AGENT,
        ownership_evidence="created by this adapter",
        source_document_id="doc-3",
    )
    reloaded = ResourceRegistry(path)
    replay = reloaded.begin_attempt(request())
    assert replay is not None
    assert replay.replayed is True
    assert replay.sent is True
    assert reloaded.get("owned-1").source_document_id == "doc-3"


def test_lifecycle_requires_verified_outputs_and_does_not_delete(
    tmp_path: Path,
) -> None:
    registry = ResourceRegistry(tmp_path / "registry.json")
    bot = registry.register_owned(
        local_id="bot-1",
        provider="grok",
        kind=ResourceKind.GROK_BOT,
        provider_id=AGENT,
        ownership_evidence="adapter create",
    )
    chat = registry.register_owned(
        local_id="chat-1",
        provider="grok",
        kind=ResourceKind.GROK_CHAT,
        provider_id="chat-owned",
        ownership_evidence="adapter create",
    )
    assert cleanup_eligible(bot) is False
    with pytest.raises(LifecycleError):
        apply_transition(bot, ResourceState.ELIGIBLE_FOR_CLEANUP)
    with pytest.raises(LifecycleError):
        apply_transition(bot, ResourceState.RETAINED, retain_reason="challenge")
    verified = note_outputs_verified(bot)
    retained = apply_transition(
        verified, ResourceState.RETAINED, retain_reason="challenge"
    )
    assert cleanup_eligible(retained) is False
    eligible = apply_transition(retained, ResourceState.ELIGIBLE_FOR_CLEANUP)
    assert cleanup_eligible(eligible) is True
    assert cleanup_eligible(chat) is False
    verified_stored = registry.replace(verified)
    retained_stored = registry.replace(retained)
    stored = registry.replace(eligible)
    assert verified_stored.outputs_verified is True
    assert retained_stored.state is ResourceState.RETAINED
    assert stored.state is ResourceState.ELIGIBLE_FOR_CLEANUP
    hijacked = ResourceRecord(
        local_id=bot.local_id,
        provider=bot.provider,
        kind=ResourceKind.GROK_CHAT,
        provider_id="someone-else",
        source_document_id="other-doc",
        state=ResourceState.DELETED,
        ownership_evidence="rewritten",
        outputs_verified=True,
        retain_reason="",
    )
    with pytest.raises(RegistryError):
        registry.replace(hijacked)
    jumped = ResourceRecord(
        local_id=chat.local_id,
        provider=chat.provider,
        kind=chat.kind,
        provider_id=chat.provider_id,
        source_document_id=chat.source_document_id,
        state=ResourceState.ELIGIBLE_FOR_CLEANUP,
        ownership_evidence=chat.ownership_evidence,
        outputs_verified=True,
        retain_reason="",
    )
    with pytest.raises(RegistryError):
        registry.replace(jumped)
    reopened = ResourceRegistry(tmp_path / "registry.json")
    assert cleanup_eligible(reopened.get("bot-1")) is True
    assert reopened.get("bot-1").provider_id == AGENT
    assert reopened.get("bot-1").kind is ResourceKind.GROK_BOT
    assert reopened.get("bot-1").ownership_evidence == "adapter create"
    assert reopened.get("chat-1").state is ResourceState.ACTIVE
    assert remote_delete_available() is False
    assert (
        can_transition(ResourceState.ELIGIBLE_FOR_CLEANUP, ResourceState.DELETED)
        is False
    )


def test_profile_guard_and_bad_request_do_not_launch(tmp_path: Path) -> None:
    isolated = Path(r"C:\Users\User\AppData\Local\Grok Bot Isolated Pilot")
    assert_isolated_profile(
        isolated,
        marker_text=ISOLATED_MARKER,
        pilot_settings='{"localToolPermission": "never"}',
        live_settings='{"localToolPermission": "ask"}',
    )
    with pytest.raises(ProfileError):
        assert_isolated_profile(
            Path(r"C:\Users\User\AppData\Roaming\Grok Bot"),
            marker_text=ISOLATED_MARKER,
            pilot_settings='{"localToolPermission": "never"}',
            live_settings='{"localToolPermission": "ask"}',
        )
    with pytest.raises(ProfileError):
        assert_isolated_profile(
            isolated,
            marker_text=ISOLATED_MARKER,
            pilot_settings='{"localToolPermission": "always"}',
            live_settings='{"localToolPermission": "ask"}',
        )
    assert (
        read_permission('{"localToolPermission":"ask","localToolPermission":"never"}')
        == "unreadable"
    )

    def refuse() -> FakeSession:
        raise AssertionError("launch")

    adapter = GrokDesktopAdapter(
        registry=ResourceRegistry(tmp_path / "registry.json"),
        diagnostics_dir=tmp_path / "diag",
        open_session=refuse,
        check_profile=lambda: (_ for _ in ()).throw(ProfileError("blocked")),
    )
    with pytest.raises(ProfileError):
        adapter.exchange(request())
    idle = GrokDesktopAdapter(
        registry=ResourceRegistry(tmp_path / "other.json"),
        diagnostics_dir=tmp_path / "diag",
        open_session=refuse,
        check_profile=lambda: None,
    )
    rejected = idle.exchange(request(message="no tag"))
    assert rejected.reason == "bad_message"
    assert not (tmp_path / "other.json").exists()


def test_shutdown_after_pre_send_timeout(tmp_path: Path) -> None:
    session = FakeSession(select=TimeoutError("before"))
    adapter = GrokDesktopAdapter(
        registry=ResourceRegistry(tmp_path / "registry.json"),
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: session,
        check_profile=lambda: None,
        clock=Clock(),
    )
    result = adapter.exchange(request())
    assert result.status is ExchangeStatus.TIMEOUT
    assert result.sent is False
    assert session.closed == 1

    def refuse() -> FakeSession:
        raise AssertionError("timeout must not send again")

    replay = GrokDesktopAdapter(
        registry=ResourceRegistry(tmp_path / "registry.json"),
        diagnostics_dir=tmp_path / "diag",
        open_session=refuse,
        check_profile=lambda: None,
    ).exchange(request())
    assert replay.replayed is True
    assert replay.status is ExchangeStatus.TIMEOUT
    assert replay.sent is False


def test_host_reuses_pipe_harness_without_a_tcp_port() -> None:
    host = (ASSETS / "pipe-host.ps1").read_text(encoding="utf-8")
    harness = (ASSETS / "harness.ps1").read_text(encoding="utf-8")
    assert "harness.ps1" in host
    assert "--remote-debugging-pipe" in host
    assert "remote-debugging-port" not in host
    assert "deleteAgents" not in host
    assert "function Send-Cdp" in harness
    assert "function New-DebugPipes" in harness
    assert "RedirectStandardInput" not in host
    assert "RedirectStandardOutput = $true" in host
    assert "RedirectStandardError = $true" in host
    assert "System.IO.StreamWriter" in host
    assert "[Console]::Out.WriteLine" not in host
    assert "[Console]::In.ReadLine" not in host
    assert "Read-ProtocolLine" in host
    assert ".child-out" in host
    assert "Write-Output $Json" not in host
    assert "remote-debugging-pipe" in host
    command = host_command(
        profile_dir=Path(r"C:\Users\User\AppData\Local\Grok Bot Isolated Pilot"),
        exe_path=Path(r"C:\Users\User\AppData\Local\Programs\Grok Bot\Grok Bot.exe"),
        live_settings=Path(r"C:\Users\User\.grokbot\settings.json"),
        log_path=Path(r"C:\Users\User\AppData\Local\Temp\grok-desktop-host.log"),
    )
    assert command[0] == "powershell.exe"
    assert "-ConfirmLaunch" in command
    assert "remote-debugging-port" not in " ".join(command)


def test_protocol_round_trips_unicode_code_points() -> None:
    sample = {
        "hebrew": "היי יעל! נעים להכיר, אני כאן ומוכן לעבודה.",
        "french": "déjà vu — œuvre",
        "arabic": "مرحبا بالعالم",
        "punct": "«שלום» … –",
        "multiline": "one\ntwo\n  three  \n",
        "nbsp": "a\u00a0b",
        "wave": "👋",
    }
    raw = json.dumps({"value": sample}, ensure_ascii=False).encode("utf-8")
    assert "ה".encode() in raw
    assert b"\\u05d4" not in raw.lower()
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ASSETS / "pipe-host.ps1"),
            "-ProtocolSelfTest",
        ],
        input=raw.decode("utf-8") + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    for name, expected in sample.items():
        actual = payload["value"][name]
        assert [ord(char) for char in actual] == [ord(char) for char in expected]


def test_unknown_child_stop_is_a_real_false_return() -> None:
    harness = (ASSETS / "harness.ps1").read_text(encoding="utf-8")
    walker = harness.split("function Update-OwnedChildren", 1)[1].split(
        "function Test-MayStop", 1
    )[0]
    assert walker.count("$null = Save-OwnedLedger $State") == 3
    assert "\n        Save-OwnedLedger $State\n" not in walker
    host = (ASSETS / "pipe-host.ps1").read_text(encoding="utf-8")
    assert "if (-not (Update-OwnedChildren $state)) { throw 'owned children' }" in host
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "function Save { return $true }; "
            "function Bad { Save; return $false }; "
            "function Good { $null = Save; return $false }; "
            "if (-not (Bad)) { 'bad-aborted' } else { 'bad-continued' }; "
            "if (-not (Good)) { 'good-aborted' } else { 'good-continued' }",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["bad-continued", "good-aborted"]


def test_review_pipeline_does_not_reference_the_desktop_adapter() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "transcription_review_agent"
    for path in root.rglob("*.py"):
        if "grok_desktop" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "grok_desktop" not in text


def test_windows_profile_path_is_readable_from_wsl() -> None:
    import os

    mapped = local_read_path(
        Path(r"C:\Users\User\AppData\Local\Grok Bot Isolated Pilot\pilot-marker.txt")
    )
    if os.name == "nt":
        assert "Grok Bot Isolated Pilot" in str(mapped)
    else:
        assert (
            mapped.as_posix()
            == "/mnt/c/Users/User/AppData/Local/Grok Bot Isolated Pilot/pilot-marker.txt"
        )


def test_wslenv_shares_confirmation_and_preserves_existing_entries() -> None:
    import os

    parent_confirm = os.environ.get("GROK_ISO_CONFIRM")
    parent_wslenv = os.environ.get("WSLENV")
    shared = windows_host_env(
        {
            "PATH": "/usr/bin",
            "WSLENV": "FOO/p:BAR",
            "GROK_ISO_CONFIRM": "0",
        },
        wsl=True,
    )
    assert shared["GROK_ISO_CONFIRM"] == "1"
    assert shared["PATH"] == "/usr/bin"
    assert shared["WSLENV"] == "FOO/p:BAR:GROK_ISO_CONFIRM/w"
    one_way = windows_host_env(
        {"WSLENV": "GROK_ISO_CONFIRM/u:OTHER/p"},
        wsl=True,
    )
    assert one_way["WSLENV"] == "GROK_ISO_CONFIRM/uw:OTHER/p"
    already = windows_host_env({"WSLENV": "FOO:GROK_ISO_CONFIRM/w:BAR"}, wsl=True)
    assert already["WSLENV"] == "FOO:GROK_ISO_CONFIRM/w:BAR"
    plain = windows_host_env({"WSLENV": "GROK_ISO_CONFIRM"}, wsl=True)
    assert plain["WSLENV"] == "GROK_ISO_CONFIRM"
    assert plain["GROK_ISO_CONFIRM"] == "1"
    translated = windows_host_env(
        {"WSLENV": "FOO/p:GROK_ISO_CONFIRM/p:BAR/l", "GROK_ISO_CONFIRM": "1"},
        wsl=True,
    )
    assert translated["WSLENV"] == "FOO/p:GROK_ISO_CONFIRM/w:BAR/l"
    assert translated["GROK_ISO_CONFIRM"] == "1"
    listed = windows_host_env({"WSLENV": "GROK_ISO_CONFIRM/lp:KEEP/u"}, wsl=True)
    assert listed["WSLENV"] == "GROK_ISO_CONFIRM/w:KEEP/u"
    combined = windows_host_env({"WSLENV": "GROK_ISO_CONFIRM/wp:OTHER"}, wsl=True)
    assert combined["WSLENV"] == "GROK_ISO_CONFIRM/w:OTHER"
    reverse_path = windows_host_env({"WSLENV": "GROK_ISO_CONFIRM/up"}, wsl=True)
    assert reverse_path["WSLENV"] == "GROK_ISO_CONFIRM/w"
    assert reverse_path["GROK_ISO_CONFIRM"] == "1"
    duplicate = windows_host_env(
        {"WSLENV": "GROK_ISO_CONFIRM/u:GROK_ISO_CONFIRM/p"},
        wsl=True,
    )
    assert duplicate["WSLENV"] == "GROK_ISO_CONFIRM/uw"
    assert duplicate["GROK_ISO_CONFIRM"] == "1"
    native = windows_host_env({"WSLENV": "FOO/p"}, wsl=False)
    assert native["GROK_ISO_CONFIRM"] == "1"
    assert native["WSLENV"] == "FOO/p"
    assert os.environ.get("GROK_ISO_CONFIRM") == parent_confirm
    assert os.environ.get("WSLENV") == parent_wslenv
    source = Path(windows_host_env.__code__.co_filename).read_text(encoding="utf-8")
    assert "env=windows_host_env()" in source


def test_update_suppression_is_only_on_the_isolated_child() -> None:
    host = (ASSETS / "pipe-host.ps1").read_text(encoding="utf-8")
    harness = (ASSETS / "harness.ps1").read_text(encoding="utf-8")
    assert "$psi.EnvironmentVariables['SAND_DISABLE_UPDATES'] = '1'" in host
    assert "$env:SAND_DISABLE_UPDATES" not in host
    assert "SAND_DISABLE_UPDATES" not in harness
    parent = windows_host_env({"PATH": "/usr/bin"}, wsl=False)
    assert "SAND_DISABLE_UPDATES" not in parent
    preserved = windows_host_env(
        {"PATH": "/usr/bin", "SAND_DISABLE_UPDATES": "0"},
        wsl=False,
    )
    assert preserved["SAND_DISABLE_UPDATES"] == "0"
    check = harness[harness.index("function Test-BaselineIntegrity") :]
    check = check[: check.index("function Publish-BaselineIntegrity")]
    assert "Stop-Process" not in check
    assert "baseline integrity failed" in check
    assert "baseline integrity intact" in harness
    assert "Publish-BaselineIntegrity $state" in host


def test_missing_confirmation_still_prevents_launch() -> None:
    host = (ASSETS / "pipe-host.ps1").read_text(encoding="utf-8")
    harness = (ASSETS / "harness.ps1").read_text(encoding="utf-8")
    assert "return ($ConfirmLaunch -and $ConfirmEnv -eq '1')" in harness
    assert "$env:GROK_ISO_CONFIRM" in host
    assert "Write-Refusal 'launch not confirmed'" in host
    assert "Write-Refusal 'normal profile'" in host
    command = host_command(
        profile_dir=Path(r"C:\Users\User\AppData\Local\Grok Bot Isolated Pilot"),
        exe_path=Path(r"C:\Users\User\AppData\Local\Programs\Grok Bot\Grok Bot.exe"),
        live_settings=Path(r"C:\Users\User\.grokbot\settings.json"),
        log_path=Path(r"C:\Users\User\AppData\Local\Temp\grok-desktop-host.log"),
    )
    assert "-ConfirmLaunch" in command
    assert "GROK_ISO_CONFIRM" not in command
    evidence = command[command.index("-EvidenceId") + 1]
    assert re.fullmatch(r"[0-9a-f]{16}", evidence)


def test_early_host_failure_is_a_sanitized_code() -> None:
    assert (
        classify_host_failure(TransportError("launch not confirmed"))
        == "launch_not_confirmed"
    )
    assert classify_host_failure(TransportError("normal profile")) == "normal_profile"
    assert classify_host_failure(TransportError("marker missing")) == "marker_missing"
    assert (
        classify_host_failure(TransportError("permission preflight mismatch"))
        == "permission_mismatch"
    )
    assert classify_host_failure(FileNotFoundError("powershell.exe")) == (
        "powershell_not_found"
    )
    assert classify_host_failure(TransportError(r"C:\Users\User\secret")) == (
        "host_startup_failed"
    )
    assert classify_host_failure(OSError("startup")) == "host_startup_failed"


def test_failed_session_open_does_not_send_or_retry(tmp_path: Path) -> None:
    opened: list[str] = []

    def open_session() -> FakeSession:
        opened.append("open")
        raise TransportError("launch not confirmed")

    request_id = "OPEN-FAIL-7K2M"
    probe = request(request_id=request_id, message=f"NO-SEND {request_id}")
    registry = tmp_path / "registry.json"
    result = GrokDesktopAdapter(
        registry=ResourceRegistry(registry),
        diagnostics_dir=tmp_path / "diag",
        open_session=open_session,
        check_profile=lambda: None,
    ).exchange(probe)
    assert result.status is ExchangeStatus.ERROR
    assert result.sent is False
    assert result.reason == "session_open_failed:launch_not_confirmed"
    assert result.replayed is False
    assert opened == ["open"]

    def refuse() -> FakeSession:
        raise AssertionError("failed open must not launch again")

    replay = GrokDesktopAdapter(
        registry=ResourceRegistry(registry),
        diagnostics_dir=tmp_path / "diag",
        open_session=refuse,
        check_profile=lambda: None,
    ).exchange(probe)
    assert replay.replayed is True
    assert replay.sent is False
    assert replay.reason == "session_open_failed:launch_not_confirmed"


def test_poll_expression_scopes_the_turn_and_keeps_the_reply() -> None:
    expression = poll_expression(AGENT, REQUEST_ID)
    assert AGENT in expression
    assert "sand-transcript-row" in expression
    assert 'data-active="true"' in expression
    assert "sand-message-prose" in expression
    assert "replyStructure" in expression
    assert "nodeValue" in expression
    assert "truncated" in expression
    assert "innerText" not in expression
    assert ".slice(0, 400)" not in expression


def test_long_multiline_reply_is_preserved_and_preview_is_short() -> None:
    french = "La première ligne reste entière.\nDéjà  vu, la deuxième ligne."
    english = "The third line is still the same sentence."
    body = "\n".join([french, english] * 8)
    assert len(body) > 400
    result = run_exchange(
        FakeSession(polls=[success_poll(body)]), request(), Clock(), poll_timeout=2
    )
    assert result.status is ExchangeStatus.SUCCESS
    assert result.reply == body
    assert result.raw_reply == body
    assert result.reply_preview == body[:PREVIEW_CHARS]
    assert len(result.reply_preview) == PREVIEW_CHARS
    assert "\n" in result.reply
    assert "Déjà  vu" in result.reply


def test_oversized_and_truncated_replies_are_not_success() -> None:
    oversized = run_exchange(
        FakeSession(polls=[turn_poll(["finished"], "x" * (MAX_REPLY + 1))]),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert oversized.status is ExchangeStatus.ERROR
    assert oversized.reason == "reply_oversized"
    assert oversized.reply == ""
    assert oversized.reply_preview == ""
    incomplete = run_exchange(
        FakeSession(
            polls=[
                turn_poll(
                    ["finished"],
                    "",
                    truncated=True,
                    replyChars=MAX_REPLY + 20,
                )
            ]
        ),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert incomplete.status is ExchangeStatus.ERROR
    assert incomplete.reason == "reply_incomplete"
    assert incomplete.reply == ""
    assert incomplete.sent is True


def _settled(reply: str, **extra: object) -> dict[str, object]:
    return turn_poll(
        [],
        reply,
        streaming=False,
        pending=False,
        failed=False,
        busy=False,
        **extra,
    )


def test_streaming_pause_and_changing_prose_do_not_complete() -> None:
    paused = run_exchange(
        FakeSession(
            polls=[
                turn_poll(["thinking"], "partial", streaming=True, pending=False),
                turn_poll(["thinking"], "partial", streaming=True, pending=False),
            ]
        ),
        request(),
        Clock(),
        poll_timeout=1,
        poll_interval=0.5,
    )
    assert paused.status is ExchangeStatus.UNCERTAIN
    assert paused.reason == "timeout_after_send"
    assert paused.reply == "partial"
    assert "streaming=true" in paused.detail
    assert "attribution=turn" in paused.detail

    growing = run_exchange(
        FakeSession(
            polls=[
                _settled("partial"),
                _settled("partial line"),
                _settled("partial line done"),
            ]
        ),
        request(),
        Clock(),
        poll_timeout=1,
        poll_interval=0.5,
    )
    assert growing.status is ExchangeStatus.UNCERTAIN
    assert growing.reply == "partial line"
    assert growing.reason == "timeout_after_send"


def test_turn_settles_only_after_active_generation_or_a_second_settled_poll() -> None:
    transition = run_exchange(
        FakeSession(
            polls=[
                turn_poll(
                    ["thinking"],
                    "partial",
                    streaming=True,
                    pending=False,
                    failed=False,
                    busy=False,
                ),
                _settled(REQUEST_ID),
            ]
        ),
        request(),
        Clock(),
    )
    assert transition.status is ExchangeStatus.SUCCESS
    assert transition.reply == REQUEST_ID

    once = run_exchange(
        FakeSession(polls=[_settled(REQUEST_ID), turn_poll(["thinking"], REQUEST_ID)]),
        request(),
        Clock(),
        poll_timeout=1,
    )
    assert once.status is ExchangeStatus.UNCERTAIN
    assert once.reason == "timeout_after_send"
    assert once.reply == REQUEST_ID

    confirmed = run_exchange(
        FakeSession(polls=[_settled(REQUEST_ID), _settled(REQUEST_ID)]),
        request(),
        Clock(),
    )
    assert confirmed.status is ExchangeStatus.SUCCESS
    assert confirmed.reply == REQUEST_ID
    assert confirmed.states_seen == ((), ())


def test_sidebar_finished_and_missing_turn_signals_do_not_complete() -> None:
    sidebar = run_exchange(
        FakeSession(
            polls=[
                _settled(REQUEST_ID, unrelatedStates=["finished"]),
                turn_poll(["idle"], REQUEST_ID, unrelatedStates=["finished"]),
            ]
        ),
        request(),
        Clock(),
        poll_timeout=1,
    )
    assert sidebar.status is ExchangeStatus.UNCERTAIN
    assert sidebar.reason == "timeout_after_send"
    assert sidebar.reply == REQUEST_ID
    assert "finished" not in sidebar.states_seen[0]

    missing = run_exchange(
        FakeSession(polls=[turn_poll([], REQUEST_ID), turn_poll([], REQUEST_ID)]),
        request(),
        Clock(),
        poll_timeout=1,
    )
    assert missing.status is ExchangeStatus.UNCERTAIN
    assert missing.reply == REQUEST_ID

    pending = run_exchange(
        FakeSession(
            polls=[turn_poll(["finished"], REQUEST_ID, pending=True, streaming=False)]
        ),
        request(),
        Clock(),
        poll_timeout=1,
    )
    assert pending.status is ExchangeStatus.UNCERTAIN
    assert pending.reason == "timeout_after_send"
    assert "pending=true" in pending.detail


def test_poll_expression_bounds_the_turn_to_transcript_rows() -> None:
    expression = poll_expression(AGENT, REQUEST_ID)
    assert "closest('.sand-transcript-row')" in expression
    assert "nextElementSibling" in expression
    assert "parent.children" not in expression
    assert "sand-virtual-transcript__inset" in expression
    assert "hasAttribute('aria-label')" in expression
    assert "data-pending" in expression
    assert "aria-busy" in expression
    assert "data-failed" in expression
    assert "streaming: streaming" in expression


_INSET_POLLS = r"""
const fs = require("fs");
const vm = require("vm");
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
function createElement(tag) {
  const el = { nodeType: 1, tagName: String(tag).toUpperCase(), parentElement: null,
    childNodes: [], attrs: {}, className: "", nodeValue: null };
  el.classList = { contains(name) {
    return (" " + el.className + " ").indexOf(" " + name + " ") >= 0; } };
  el.getAttribute = (name) => Object.prototype.hasOwnProperty.call(el.attrs, name)
    ? el.attrs[name] : null;
  el.hasAttribute = (name) => Object.prototype.hasOwnProperty.call(el.attrs, name);
  el.setAttribute = (name, value) => { el.attrs[name] = String(value); };
  Object.defineProperty(el, "children", { get() {
    return el.childNodes.filter((node) => node.nodeType === 1); } });
  Object.defineProperty(el, "nextElementSibling", { get() {
    if (!el.parentElement) return null;
    const kids = el.parentElement.children;
    const index = kids.indexOf(el);
    return index >= 0 && index + 1 < kids.length ? kids[index + 1] : null; } });
  Object.defineProperty(el, "textContent", { get() {
    let out = "";
    for (const child of el.childNodes) {
      out += child.nodeType === 3 ? (child.nodeValue || "") : child.textContent;
    }
    return out; } });
  el.closest = (selector) => {
    let cur = el;
    while (cur) { if (matches(cur, selector)) return cur; cur = cur.parentElement; }
    return null;
  };
  el.contains = (other) => {
    let cur = other;
    while (cur) { if (cur === el) return true; cur = cur.parentElement; }
    return false;
  };
  el.querySelectorAll = (selector) => {
    const found = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (matches(child, selector)) found.push(child);
        walk(child);
      }
    };
    walk(el);
    return found;
  };
  el.append = (...nodes) => {
    for (const node of nodes) {
      if (typeof node === "string") {
        el.childNodes.push({ nodeType: 3, nodeValue: node, parentElement: el, childNodes: [] });
      } else { node.parentElement = el; el.childNodes.push(node); }
    }
    return el;
  };
  return el;
}
function matches(el, selector) {
  if (!el || el.nodeType !== 1) return false;
  if (selector === ".sand-transcript-row") return el.classList.contains("sand-transcript-row");
  if (selector === ".sand-message-prose") return el.classList.contains("sand-message-prose");
  if (selector === "[data-grok-state]") return el.hasAttribute("data-grok-state");
  if (selector === ".sand-message[data-role]") {
    return el.classList.contains("sand-message") && el.hasAttribute("data-role");
  }
  if (selector === "button[data-agent-id][data-active=\"true\"]") {
    return el.tagName === "BUTTON" && el.hasAttribute("data-agent-id")
      && el.getAttribute("data-active") === "true";
  }
  throw new Error("selector " + selector);
}
function row(role) {
  const el = createElement("div");
  el.className = "sand-virtual-transcript__row sand-transcript-row";
  if (role) el.setAttribute("data-role", role);
  return el;
}
function assistantMessage() {
  const prose = createElement("div");
  prose.className = "sand-message-prose";
  prose.append(createElement("p").append(payload.tag));
  const message = createElement("div");
  message.className = "sand-message";
  message.setAttribute("data-role", "assistant");
  return message.append(prose);
}
function taggedUser() {
  const message = createElement("div");
  message.className = "sand-message";
  message.setAttribute("data-role", "user");
  message.append(payload.tag);
  return row("user").append(message);
}
function assistantRow() { return row(null).append(assistantMessage()); }
function inset() {
  const el = createElement("div");
  el.className = "sand-virtual-transcript__inset sand-10l6tqk";
  return el;
}
function unknown() {
  const el = createElement("div");
  el.className = "sand-unknown-sibling";
  return el;
}
function documentFor(children) {
  const root = createElement("div");
  root.className = "sand-virtual-transcript__inner";
  const button = createElement("button");
  button.setAttribute("data-agent-id", payload.agent);
  button.setAttribute("data-active", "true");
  root.append(button, ...children);
  return {
    querySelector: (selector) => root.querySelectorAll(selector)[0] || null,
    querySelectorAll: (selector) => root.querySelectorAll(selector),
  };
}
function run(children) {
  return vm.runInNewContext(payload.expression, { document: documentFor(children) });
}
const laterUser = row("user").append((() => {
  const message = createElement("div");
  message.className = "sand-message";
  message.setAttribute("data-role", "user");
  message.append("later user");
  return message;
})());
const hidden = row(null).append(payload.tag);
const results = {
  trailing_inset: run([taggedUser(), assistantRow(), inset()]),
  duplicated_assistants: run([taggedUser(), assistantRow(), assistantRow(), inset()]),
  unknown_sibling: run([taggedUser(), assistantRow(), unknown()]),
  inset_between_rows: run([taggedUser(), assistantRow(), inset(), assistantRow()]),
  next_user_row: run([taggedUser(), assistantRow(), laterUser]),
  missing_assistant: run([taggedUser(), inset()]),
  virtualized_assistant: run([hidden]),
};
process.stdout.write(JSON.stringify(results));
"""


def test_virtual_transcript_inset_bounds_only_a_trailing_rendered_list() -> None:
    expression = poll_expression(AGENT, REQUEST_ID)
    completed = subprocess.run(
        ["node", "-e", _INSET_POLLS],
        input=json.dumps({"expression": expression, "tag": REQUEST_ID, "agent": AGENT}),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    results = json.loads(completed.stdout)
    trailing = results["trailing_inset"]
    assert trailing["attribution"] == "turn"
    assert trailing["replyStructure"] == "prose"
    assert trailing["reply"] == REQUEST_ID
    assert trailing["streaming"] is False
    assert trailing["pending"] is False
    assert trailing["failed"] is False
    assert trailing["busy"] is False
    assert trailing["states"] == []
    duplicated = results["duplicated_assistants"]
    assert duplicated["attribution"] == "turn"
    assert duplicated["replyStructure"] == "inconclusive"
    assert duplicated["reply"] == ""
    assert results["unknown_sibling"]["attribution"] == "ambiguous"
    assert results["unknown_sibling"]["reply"] == ""
    assert results["inset_between_rows"]["attribution"] == "ambiguous"
    assert results["inset_between_rows"]["reply"] == ""
    delimited = results["next_user_row"]
    assert delimited["attribution"] == "turn"
    assert delimited["replyStructure"] == "prose"
    assert delimited["reply"] == REQUEST_ID
    assert "later user" not in delimited["reply"]
    missing = results["missing_assistant"]
    assert missing["attribution"] == "turn"
    assert missing["replyStructure"] == "absent"
    assert missing["reply"] == ""
    virtualized = results["virtualized_assistant"]
    assert virtualized["attribution"] == "virtualized"
    assert virtualized["reply"] == ""
    assert virtualized["userHits"] == 0


def test_unrelated_streaming_and_wrong_bot_states_do_not_complete() -> None:
    stale = run_exchange(
        FakeSession(polls=[turn_poll(["idle"], "", unrelatedStates=["finished"])] * 8),
        request(),
        Clock(),
        poll_timeout=1,
    )
    assert stale.status is ExchangeStatus.UNCERTAIN
    assert stale.reason == "timeout_after_send"
    assert stale.reply == ""
    streaming = run_exchange(
        FakeSession(
            polls=[
                turn_poll(["finished", "thinking"], "partial line"),
                TimeoutError("still streaming"),
            ]
        ),
        request(),
        Clock(),
        poll_timeout=5,
    )
    assert streaming.status is ExchangeStatus.UNCERTAIN
    assert streaming.reply == "partial line"
    assert streaming.reason == "timeout_after_send"
    virtualized = run_exchange(
        FakeSession(
            polls=[
                turn_poll(
                    ["finished"],
                    REQUEST_ID,
                    attribution="virtualized",
                    userHits=0,
                )
            ]
        ),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert virtualized.reason == "virtualized_turn"
    assert virtualized.reply == ""
    ambiguous = run_exchange(
        FakeSession(
            polls=[turn_poll(["finished"], REQUEST_ID, attribution="ambiguous")]
        ),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert ambiguous.reason == "ambiguous_completion"
    assert ambiguous.reply == ""
    wrong_bot = run_exchange(
        FakeSession(polls=[turn_poll(["finished"], REQUEST_ID, activeAgent=OTHER)]),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert wrong_bot.reason == "completion_wrong_bot"
    assert wrong_bot.reply == ""
    finished = run_exchange(
        FakeSession(
            polls=[
                turn_poll(["finished", "thinking"], ""),
                success_poll(REQUEST_ID),
            ]
        ),
        request(),
        Clock(),
        poll_timeout=5,
    )
    assert finished.status is ExchangeStatus.SUCCESS
    assert finished.reply == REQUEST_ID


def test_same_request_id_with_different_work_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    session = FakeSession(polls=[success_poll()])
    first = GrokDesktopAdapter(
        registry=ResourceRegistry(path),
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: session,
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(request())
    assert first.status is ExchangeStatus.SUCCESS

    def refuse() -> FakeSession:
        raise AssertionError("a mismatched request id must not launch")

    adapter = GrokDesktopAdapter(
        registry=ResourceRegistry(path),
        diagnostics_dir=tmp_path / "diag",
        open_session=refuse,
        check_profile=lambda: None,
    )
    mismatch = adapter.exchange(
        request(message=f"Autre message {REQUEST_ID}", source_document_id="doc-x")
    )
    assert mismatch.status is ExchangeStatus.ERROR
    assert mismatch.reason == "request_id_reused"
    assert mismatch.replayed is True
    assert mismatch.sent is False
    other_bot = adapter.exchange(request(bot_name="Another Integration Bot"))
    assert other_bot.reason == "request_id_reused"
    replay = adapter.exchange(request())
    assert replay.replayed is True
    assert replay.status is ExchangeStatus.SUCCESS
    assert replay.reply == REQUEST_ID


def test_registry_rejects_corruption_and_survives_concurrent_writers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.json"
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def own() -> None:
        barrier.wait()
        try:
            ResourceRegistry(path).register_owned(
                local_id="owned-1",
                provider="grok",
                kind=ResourceKind.GROK_BOT,
                provider_id=AGENT,
                ownership_evidence="created by this adapter",
            )
        except Exception as exc:
            errors.append(exc)

    def attempt() -> None:
        barrier.wait()
        try:
            started = ResourceRegistry(path).begin_attempt(request())
            if started is not None:
                errors.append(RegistryError("attempt was already present"))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=own), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    loaded = ResourceRegistry(path)
    assert loaded.get("owned-1").provider_id == AGENT
    replay = loaded.begin_attempt(request())
    assert replay is not None and replay.sent is True and replay.replayed is True
    leftover = path.with_suffix(path.suffix + ".tmp")
    leftover.write_text("{partial", encoding="utf-8")
    assert ResourceRegistry(path).get("owned-1").provider_id == AGENT
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["resources"].append(dict(payload["resources"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryError):
        ResourceRegistry(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RegistryError):
        ResourceRegistry(path)
    with pytest.raises(RegistryError):
        loaded.begin_attempt(request(request_id="INTEGRATION-01-C8E3"))


def test_baseline_close_failure_is_reported_without_resending(tmp_path: Path) -> None:
    class BaselineClose(FakeSession):
        def close(self) -> None:
            super().close()
            raise TransportError("baseline_integrity_failed", cleanup="closed")

    path = tmp_path / "registry.json"
    session = BaselineClose(polls=[success_poll()])
    adapter = GrokDesktopAdapter(
        registry=ResourceRegistry(path),
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: session,
        check_profile=lambda: None,
        clock=Clock(),
    )
    with pytest.raises(TransportError, match="baseline_integrity_failed"):
        adapter.exchange(request())
    assert session.closed == 1
    stored = json.loads(path.read_text(encoding="utf-8"))["attempts"][0]["result"]
    assert stored["status"] == "success"
    assert stored["reply"] == REQUEST_ID
    assert stored["sent"] is True
    assert stored["cleanup"] == "closed"
    assert stored["baseline"] == "failed"
    reloaded = ResourceRegistry(path).begin_attempt(request())
    assert reloaded is not None
    assert reloaded.status is ExchangeStatus.SUCCESS
    assert reloaded.reply == REQUEST_ID
    assert reloaded.baseline == "failed"
    diagnostic = json.loads(
        (tmp_path / "diag" / f"{REQUEST_ID}.json").read_text(encoding="utf-8")
    )
    assert diagnostic["result"]["baseline"] == "failed"
    assert diagnostic["result"]["cleanup"] == "closed"

    def refuse() -> FakeSession:
        raise AssertionError("baseline failure must not send again")

    with pytest.raises(TransportError, match="baseline_integrity_failed") as replayed:
        GrokDesktopAdapter(
            registry=ResourceRegistry(path),
            diagnostics_dir=tmp_path / "diag",
            open_session=refuse,
            check_profile=lambda: None,
        ).exchange(request())
    assert replayed.value.cleanup == "closed"
    assert session.closed == 1

    intact_path = tmp_path / "intact.json"
    intact = FakeSession(polls=[success_poll()])
    intact_result = GrokDesktopAdapter(
        registry=ResourceRegistry(intact_path),
        diagnostics_dir=tmp_path / "intact-diag",
        open_session=lambda: intact,
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(request())
    assert intact_result.status is ExchangeStatus.SUCCESS
    assert intact_result.reply == REQUEST_ID
    assert intact_result.cleanup == "closed"
    assert intact_result.baseline == "intact"
    intact_replay = GrokDesktopAdapter(
        registry=ResourceRegistry(intact_path),
        diagnostics_dir=tmp_path / "intact-diag",
        open_session=refuse,
        check_profile=lambda: None,
    ).exchange(request())
    assert intact_replay.replayed is True
    assert intact_replay.status is ExchangeStatus.SUCCESS
    assert intact_replay.baseline == "intact"
    legacy = json.loads(intact_path.read_text(encoding="utf-8"))
    del legacy["attempts"][0]["result"]["baseline"]
    intact_path.write_text(json.dumps(legacy), encoding="utf-8")
    legacy_result = ResourceRegistry(intact_path).begin_attempt(request())
    assert legacy_result is not None
    assert legacy_result.status is ExchangeStatus.SUCCESS
    assert legacy_result.baseline == "unrecorded"
    assert (
        GrokDesktopAdapter(
            registry=ResourceRegistry(intact_path),
            diagnostics_dir=tmp_path / "intact-diag",
            open_session=refuse,
            check_profile=lambda: None,
        )
        .exchange(request())
        .baseline
        == "unrecorded"
    )


def test_close_failure_keeps_the_stored_success(tmp_path: Path) -> None:
    class Closing(FakeSession):
        def close(self) -> None:
            super().close()
            raise OSError("pipe close failed")

    path = tmp_path / "registry.json"
    session = Closing(polls=[success_poll()])
    result = GrokDesktopAdapter(
        registry=ResourceRegistry(path),
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: session,
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(request())
    assert result.status is ExchangeStatus.SUCCESS
    assert result.reply == REQUEST_ID
    assert result.cleanup == "failed"
    assert result.baseline == "unrecorded"
    assert session.closed == 1
    stored = ResourceRegistry(path).begin_attempt(request())
    assert stored is not None
    assert stored.status is ExchangeStatus.SUCCESS
    assert stored.cleanup == "failed"
    assert stored.replayed is True
    assert stored.reason != "attempt_already_started"


def test_cleanup_is_recorded_beside_the_exchange_outcome(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    clean = FakeSession(polls=[success_poll()])
    result = GrokDesktopAdapter(
        registry=ResourceRegistry(path),
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: clean,
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(request())
    assert result.status is ExchangeStatus.SUCCESS
    assert result.sent is True
    assert result.cleanup == "closed"
    assert result.baseline == "intact"
    reloaded = ResourceRegistry(path).begin_attempt(request())
    assert reloaded is not None
    assert reloaded.replayed is True
    assert reloaded.status is ExchangeStatus.SUCCESS
    assert reloaded.cleanup == "closed"
    diagnostic = json.loads(
        (tmp_path / "diag" / f"{REQUEST_ID}.json").read_text(encoding="utf-8")
    )
    assert diagnostic["result"]["cleanup"] == "closed"

    uncertain_path = tmp_path / "uncertain.json"

    class UncertainClose(FakeSession):
        def close(self) -> None:
            super().close()
            raise TransportError("host_cleanup_uncertain", cleanup="uncertain")

    uncertain = UncertainClose(polls=[success_poll()])
    uncertain_result = GrokDesktopAdapter(
        registry=ResourceRegistry(uncertain_path),
        diagnostics_dir=tmp_path / "uncertain-diag",
        open_session=lambda: uncertain,
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(request())
    assert uncertain_result.status is ExchangeStatus.SUCCESS
    assert uncertain_result.reply == REQUEST_ID
    assert uncertain_result.cleanup == "uncertain"
    assert uncertain_result.baseline == "unrecorded"
    uncertain_stored = ResourceRegistry(uncertain_path).begin_attempt(request())
    assert uncertain_stored is not None
    assert uncertain_stored.replayed is True
    assert uncertain_stored.cleanup == "uncertain"

    def refuse_uncertain() -> FakeSession:
        raise AssertionError("uncertain cleanup must not send again")

    assert (
        GrokDesktopAdapter(
            registry=ResourceRegistry(uncertain_path),
            diagnostics_dir=tmp_path / "uncertain-diag",
            open_session=refuse_uncertain,
            check_profile=lambda: None,
        )
        .exchange(request())
        .cleanup
        == "uncertain"
    )

    failed_path = tmp_path / "failed.json"

    class FailedClose(FakeSession):
        def close(self) -> None:
            super().close()
            raise TransportError("host_cleanup_uncertain", cleanup="uncertain")

    failed_session = FailedClose(polls=[])
    failed_result = GrokDesktopAdapter(
        registry=ResourceRegistry(failed_path),
        diagnostics_dir=tmp_path / "failed-diag",
        open_session=lambda: failed_session,
        check_profile=lambda: None,
        clock=Clock(),
        poll_timeout=1,
    ).exchange(request())
    assert failed_result.status is ExchangeStatus.UNCERTAIN
    assert failed_result.sent is True
    assert failed_result.reason == "timeout_after_send"
    assert failed_result.cleanup == "uncertain"
    failed_stored = json.loads(failed_path.read_text(encoding="utf-8"))["attempts"][0][
        "result"
    ]
    assert failed_stored["status"] == "uncertain"
    assert failed_stored["sent"] is True
    assert failed_stored["cleanup"] == "uncertain"
    assert failed_session.closed == 1


def test_cleanup_write_failure_is_not_reported_as_closed(tmp_path: Path) -> None:
    class FlakyRegistry(ResourceRegistry):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.writes = 0

        def finish_attempt(self, result: ExchangeResult) -> None:
            self.writes += 1
            if self.writes >= 2:
                raise RegistryError("disk")
            super().finish_attempt(result)

    path = tmp_path / "registry.json"
    registry = FlakyRegistry(path)
    result = GrokDesktopAdapter(
        registry=registry,
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: FakeSession(polls=[success_poll()]),
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(request())
    assert result.status is ExchangeStatus.SUCCESS
    assert result.cleanup == "unrecorded"
    stored = json.loads(path.read_text(encoding="utf-8"))["attempts"][0]["result"]
    assert stored["status"] == "success"
    assert stored["cleanup"] == "unrecorded"


def test_protocol_reader_skips_child_logs(tmp_path: Path) -> None:
    assert decode_protocol_line("electron stdout noise\n") is None
    with pytest.raises(TransportError):
        decode_protocol_line("{not-json\n")
    script = tmp_path / "host.py"
    script.write_text(
        "import json, sys\n"
        "print('child log', flush=True)\n"
        "print('stderr noise', file=sys.stderr, flush=True)\n"
        "print(json.dumps({'ready': True}), flush=True)\n"
        "for line in sys.stdin:\n"
        "    command = json.loads(line)\n"
        "    if command.get('op') == 'close':\n"
        "        break\n"
        "    print('renderer noise', flush=True)\n"
        "    print(json.dumps({'id': command['id'], 'ok': True,"
        " 'value': {'ok': True}}), flush=True)\n",
        encoding="utf-8",
    )
    session = PowershellCdpSession([sys.executable, str(script)], timeout=5)
    try:
        assert session.evaluate("document.body") == {"ok": True}
    finally:
        with pytest.raises(TransportError, match="host_cleanup_uncertain"):
            session.close()


def test_unknown_and_unattributed_states_do_not_complete() -> None:
    unknown = run_exchange(
        FakeSession(polls=[turn_poll(["finished", "new-running-state"], REQUEST_ID)]),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert unknown.status is ExchangeStatus.UNCERTAIN
    assert unknown.reason == "unknown_completion_state"
    assert unknown.reply == ""
    missing = run_exchange(
        FakeSession(
            polls=[turn_poll(["finished"], REQUEST_ID, attribution="missing")] * 6
        ),
        request(),
        Clock(),
        poll_timeout=1,
    )
    assert missing.status is ExchangeStatus.UNCERTAIN
    assert missing.reason == "timeout_after_send"
    assert missing.reply == ""
    virtualized = run_exchange(
        FakeSession(
            polls=[
                turn_poll(
                    ["finished"],
                    REQUEST_ID,
                    attribution="virtualized",
                    userHits=0,
                )
            ]
        ),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert virtualized.reason == "virtualized_turn"
    assert virtualized.reply == ""
    changed = run_exchange(
        FakeSession(polls=[turn_poll(["thinking"], ""), success_poll(REQUEST_ID)]),
        request(),
        Clock(),
        poll_timeout=5,
    )
    assert changed.status is ExchangeStatus.SUCCESS
    assert changed.reply == REQUEST_ID
    inconclusive = run_exchange(
        FakeSession(
            polls=[
                turn_poll(
                    ["finished"],
                    f"{NAME}\n14:19\nthinking",
                    replyStructure="inconclusive",
                )
            ]
        ),
        request(),
        Clock(),
        poll_timeout=2,
    )
    assert inconclusive.reason == "reply_inconclusive"
    assert inconclusive.reply == ""
    assert inconclusive.raw_reply == ""


def test_registry_write_failures_keep_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    path = tmp_path / "registry.json"

    def own(local_id: str, provider_id: str) -> None:
        ResourceRegistry(path).register_owned(
            local_id=local_id,
            provider="grok",
            kind=ResourceKind.GROK_BOT,
            provider_id=provider_id,
            ownership_evidence="created by this adapter",
        )

    real_write = os.write

    def short(fd: int, data: bytes) -> int:
        if len(data) > 1:
            return real_write(fd, data[:1])
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", short)
    own("owned-1", AGENT)
    monkeypatch.setattr(os, "write", real_write)
    original = path.read_bytes()
    assert ResourceRegistry(path).get("owned-1").provider_id == AGENT

    def interrupt(fd: int, data: bytes) -> int:
        if len(data) > 8:
            real_write(fd, data[:4])
            raise OSError("interrupted")
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", interrupt)
    with pytest.raises(RegistryError):
        own("owned-2", OTHER)
    monkeypatch.setattr(os, "write", real_write)
    assert path.read_bytes() == original
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert ResourceRegistry(path).get("owned-1").provider_id == AGENT

    real_fsync = os.fsync

    def fail_file_sync(fd: int) -> None:
        raise OSError("fsync")

    monkeypatch.setattr(os, "fsync", fail_file_sync)
    with pytest.raises(RegistryError):
        own("owned-2", OTHER)
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert path.read_bytes() == original

    calls = {"n": 0}

    def fail_directory_sync(fd: int) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("directory")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_sync)
    with pytest.raises(RegistryError, match="directory sync"):
        own("owned-2", OTHER)
    monkeypatch.setattr(os, "fsync", real_fsync)
    reloaded = ResourceRegistry(path)
    assert reloaded.get("owned-2").provider_id == OTHER
    json.loads(path.read_text(encoding="utf-8"))

    def fail_replace(src: str, dst: str) -> None:
        raise OSError("replace")

    before = path.read_bytes()
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(RegistryError, match="replace"):
        own("owned-3", "11111111-1111-1111-1111-111111111111")
    monkeypatch.setattr(os, "replace", os.replace)
    assert path.read_bytes() == before
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RegistryError, match="corrupt"):
        ResourceRegistry(path)


class _Pipe:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _QuietStdout(_Pipe):
    def __iter__(self) -> _QuietStdout:
        return self

    def __next__(self) -> str:
        raise StopIteration


class _BlockedStdout(_Pipe):
    def __init__(self) -> None:
        super().__init__()
        self.released = threading.Event()

    def __iter__(self) -> _BlockedStdout:
        return self

    def __next__(self) -> str:
        self.released.wait(30)
        raise StopIteration

    def close(self) -> None:
        super().close()
        self.released.set()


class _Host:
    def __init__(
        self,
        stdout: _Pipe,
        *,
        sticky: bool = False,
        exit_on_stop: bool = False,
    ) -> None:
        self.stdin = _Pipe()
        self.stdout = stdout
        self.stderr = io.StringIO()
        self.sticky = sticky
        self.exit_on_stop = exit_on_stop
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        if not self.sticky:
            self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        if not self.sticky:
            self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if (
            self.exit_on_stop
            and self.stdin.closed
            and self.returncode is None
            and not self.sticky
        ):
            self.returncode = 0
        if self.returncode is None:
            raise subprocess.TimeoutExpired("owned-host", timeout or 0)
        return self.returncode


def _open_host(
    host: _Host,
    *,
    command: list[str] | None = None,
    lookup: object = None,
    stop: object = None,
    **kwargs: object,
) -> PowershellCdpSession:
    def popen(given: list[str], **_kwargs: object) -> _Host:
        return host

    extra: dict[str, object] = {}
    if lookup is not None:
        extra["identity_lookup"] = lookup
    if stop is not None:
        extra["process_stop"] = stop
    return PowershellCdpSession(  # type: ignore[arg-type]
        command or ["owned-host"],
        popen=popen,
        **extra,
        **kwargs,
    )


def test_startup_failures_stop_only_the_owned_host(tmp_path: Path) -> None:
    bystander = _Host(_QuietStdout())
    fast = {"timeout": 0.2, "cleanup_timeout": 0.05, "shutdown_timeout": 0.05}
    eof = _Host(_QuietStdout())
    eof.returncode = 0
    with pytest.raises(TransportError, match="host closed the pipe") as eof_error:
        _open_host(eof, **fast)
    assert eof_error.value.cleanup == "uncertain"
    assert "host_cleanup_uncertain" in str(eof_error.value)
    assert eof.terminated is False and eof.killed is False
    assert eof.stdin.closed and eof.stdout.closed
    assert bystander.terminated is False and bystander.killed is False

    malformed = _Host(_QuietStdout())
    malformed.stdout = io.StringIO("{not-json\n")
    with pytest.raises(TransportError, match="malformed json") as malformed_error:
        _open_host(malformed, **fast)
    assert malformed_error.value.cleanup == "uncertain"
    assert malformed.terminated is True
    assert malformed.poll() is not None
    assert "host_cleanup_uncertain" in str(malformed_error.value)

    refused = _Host(_QuietStdout(), exit_on_stop=True)
    refused.stdout = io.StringIO('{"ready": false, "error": "launch not confirmed"}\n')
    refusal_log = tmp_path / "refuse.log"
    _write_phase(refusal_log, _RUN, "prelaunch")
    with pytest.raises(TransportError, match="launch not confirmed") as refused_error:
        _open_host(
            refused,
            command=_run_command(refusal_log),
            **fast,
        )
    assert refused_error.value.cleanup == "closed"
    assert refused.terminated is False
    assert refused.poll() == 0
    assert classify_host_failure(refused_error.value) == "launch_not_confirmed"

    blocked = _Host(_BlockedStdout())
    with pytest.raises(TransportError, match="startup timeout") as timeout_error:
        _open_host(blocked, **fast)
    assert timeout_error.value.cleanup == "uncertain"
    assert blocked.terminated is True
    assert blocked.poll() is not None
    assert classify_host_failure(timeout_error.value) == (
        "host_startup_failed:host_cleanup_uncertain"
    )

    stuck = _Host(_BlockedStdout(), sticky=True)
    with pytest.raises(TransportError, match="host_cleanup_uncertain") as stuck_error:
        _open_host(stuck, **fast)
    assert stuck_error.value.cleanup == "uncertain"
    assert stuck.terminated is True and stuck.killed is True
    assert stuck.poll() is None
    assert stuck.stdin.closed
    assert classify_host_failure(stuck_error.value) == (
        "host_startup_failed:host_cleanup_uncertain"
    )


def test_thread_start_failure_stops_the_owned_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _Host(_QuietStdout())

    def fail_start(self: threading.Thread) -> None:
        raise RuntimeError("reader")

    monkeypatch.setattr(
        "transcription_review_agent.grok_desktop.transport.threading.Thread.start",
        fail_start,
    )
    with pytest.raises(TransportError, match="host_startup_failed") as error:
        _open_host(host, timeout=0.2, cleanup_timeout=0.05, shutdown_timeout=0.05)
    assert error.value.cleanup == "uncertain"
    assert host.terminated is True
    assert host.poll() is not None
    assert host.stdin.closed and host.stdout.closed and host.stderr.closed


def test_ready_timeout_covers_the_host_startup_budget() -> None:
    host = (ASSETS / "pipe-host.ps1").read_text(encoding="utf-8")
    harness = (ASSETS / "harness.ps1").read_text(encoding="utf-8")
    child_ready = int(
        re.search(r"\$readyUntil = \(Get-Date\)\.AddSeconds\((\d+)\)", host).group(1)
    )
    attempts = int(re.search(r"foreach \(\$attempt in 1\.\.(\d+)\)", host).group(1))
    waits = [int(value) for value in re.findall(r"Wait-CdpId \d+ (\d+)", host)]
    browser_close = int(
        re.search(r"\$closeUntil = \(Get-Date\)\.AddSeconds\((\d+)\)", host).group(1)
    )
    stop_wait = int(re.search(r"StopWaitMs = (\d+)", harness).group(1)) / 1000
    budget = child_ready + attempts * waits[0] + waits[1]
    signature = inspect.signature(PowershellCdpSession.__init__)
    assert signature.parameters["timeout"].default == HOST_READY_TIMEOUT_SECONDS
    assert (
        signature.parameters["shutdown_timeout"].default
        == HOST_SHUTDOWN_TIMEOUT_SECONDS
    )
    assert HOST_READY_TIMEOUT_SECONDS >= budget
    assert HOST_SHUTDOWN_TIMEOUT_SECONDS >= browser_close + stop_wait
    assert host.count("Test-StopRequested") >= 3
    assert "Stop-OwnedProcesses $state" in host
    assert harness.count("Save-OwnedLedger") >= 4
    assert "ledger unpublished" in harness
    assert "return $false" in harness
    launch = host.index("[void]$proc.Start()")
    assert "Publish-RunPhase 'launching'" in host[launch - 80 : launch]
    assert host.index("Publish-RunPhase 'prelaunch'") < launch
    assert "$text -eq $EvidenceId" in host


def _sleeper() -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_sleepers(*procs: subprocess.Popen[str]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=2)


def _record(
    pid: int, *, ticks: str = "100", executable: str | None = None
) -> dict[str, object]:
    return {
        "pid": pid,
        "creationTicks": ticks,
        "executable": executable or sys.executable,
    }


_RUN = "0123456789abcdef"
_STALE_RUN = "aaaaaaaaaaaaaaaa"


def _run_command(log: Path, evidence: str = _RUN) -> list[str]:
    return ["host", "-LogPath", str(log), "-EvidenceId", evidence]


def _write_phase(log: Path, evidence: str, phase: str, **extra: object) -> Path:
    path = Path(f"{log}.run.{evidence}.json")
    body: dict[str, object] = {"evidenceId": evidence, "phase": phase}
    body.update(extra)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _write_ledger(log: Path, payload: dict[str, object]) -> None:
    _write_phase(log, _RUN, "owned", **payload)


class _Children:
    def __init__(self) -> None:
        self.stopped: list[int] = []
        self.procs: dict[int, subprocess.Popen[str]] = {}
        self.identity: dict[int, ProcessIdentity | None] = {}
        self.fail_stop = False

    def add(self, proc: subprocess.Popen[str], *, ticks: str = "100") -> None:
        self.procs[proc.pid] = proc
        self.identity[proc.pid] = ProcessIdentity(proc.pid, ticks, sys.executable)

    def lookup(self, pid: int) -> ProcessIdentity | None:
        if pid not in self.identity:
            raise AssertionError(f"unexpected lookup {pid}")
        proc = self.procs.get(pid)
        if proc is not None and proc.poll() is not None:
            return None
        return self.identity[pid]

    def stop(self, pid: int) -> None:
        if self.fail_stop:
            raise OSError("cleanup failed")
        self.stopped.append(pid)
        proc = self.procs[pid]
        proc.terminate()
        proc.wait(timeout=2)


def test_baseline_integrity_failure_does_not_stop_baseline(tmp_path: Path) -> None:
    child = _sleeper()
    baseline = _sleeper()
    tracked = _Children()
    try:
        tracked.add(child)
        tracked.add(baseline)
        log = tmp_path / "host.log"
        _write_ledger(
            log,
            {
                "expectedExe": sys.executable,
                "controllerPid": 1,
                "baseline": [baseline.pid],
                "baselineIntegrity": "failed",
                "owned": [_record(child.pid), _record(baseline.pid)],
            },
        )
        host = _Host(_BlockedStdout())
        with pytest.raises(TransportError, match="baseline_integrity_failed") as error:
            _open_host(
                host,
                command=_run_command(log),
                lookup=tracked.lookup,
                stop=tracked.stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert error.value.cleanup == "closed"
        assert classify_host_failure(error.value).endswith("baseline_integrity_failed")
        assert child.poll() is not None
        assert baseline.poll() is None
        assert baseline.pid not in tracked.stopped
        assert tracked.stopped == [child.pid]
    finally:
        _stop_sleepers(child, baseline)


def test_startup_deadline_reaps_only_the_recorded_child(tmp_path: Path) -> None:
    child = _sleeper()
    baseline = _sleeper()
    unrelated = _sleeper()
    tracked = _Children()
    try:
        tracked.add(child)
        tracked.add(baseline)
        log = tmp_path / "host.log"
        _write_ledger(
            log,
            {
                "expectedExe": sys.executable,
                "controllerPid": 1,
                "baseline": [baseline.pid],
                "owned": [
                    _record(child.pid),
                    _record(baseline.pid),
                    _record(1),
                ],
            },
        )
        host = _Host(_BlockedStdout())
        with pytest.raises(TransportError, match="startup timeout") as error:
            _open_host(
                host,
                command=_run_command(log),
                lookup=tracked.lookup,
                stop=tracked.stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert error.value.cleanup == "closed"
        assert host.terminated is True
        assert Path(str(log) + ".stop").read_text(encoding="utf-8") == _RUN
        assert child.poll() is not None
        assert baseline.poll() is None
        assert unrelated.poll() is None
        assert tracked.stopped == [child.pid]
    finally:
        _stop_sleepers(child, baseline, unrelated)


def test_exited_host_reaps_a_child_that_is_still_running(tmp_path: Path) -> None:
    child = _sleeper()
    tracked = _Children()
    try:
        tracked.add(child)
        log = tmp_path / "host.log"
        _write_ledger(
            log,
            {
                "expectedExe": sys.executable,
                "controllerPid": 1,
                "baseline": [],
                "owned": [_record(child.pid)],
            },
        )
        host = _Host(_QuietStdout())
        host.returncode = 0
        with pytest.raises(TransportError, match="host closed the pipe") as error:
            _open_host(
                host,
                command=_run_command(log),
                lookup=tracked.lookup,
                stop=tracked.stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert error.value.cleanup == "closed"
        assert host.terminated is False and host.killed is False
        assert child.poll() is not None
        assert tracked.stopped == [child.pid]
    finally:
        _stop_sleepers(child)


def test_child_that_already_exited_is_not_signaled(tmp_path: Path) -> None:
    child = _sleeper()
    child.terminate()
    child.wait(timeout=2)
    tracked = _Children()
    tracked.add(child)
    log = tmp_path / "host.log"
    _write_ledger(
        log,
        {
            "expectedExe": sys.executable,
            "controllerPid": 1,
            "baseline": [],
            "owned": [_record(child.pid)],
        },
    )
    host = _Host(_QuietStdout())
    host.returncode = 0
    with pytest.raises(TransportError, match="host closed the pipe") as error:
        _open_host(
            host,
            command=_run_command(log),
            lookup=tracked.lookup,
            stop=tracked.stop,
            timeout=0.2,
            shutdown_timeout=0.05,
            cleanup_timeout=0.05,
        )
    assert error.value.cleanup == "closed"
    assert tracked.stopped == []
    assert host.terminated is False


def test_cleanup_failure_and_uncertain_ownership_are_explicit(tmp_path: Path) -> None:
    remaining = _sleeper()
    reused = _sleeper()
    tracked = _Children()
    try:
        tracked.add(remaining)
        tracked.fail_stop = True
        log = tmp_path / "fail.log"
        _write_ledger(
            log,
            {
                "expectedExe": sys.executable,
                "controllerPid": 1,
                "baseline": [],
                "owned": [_record(remaining.pid)],
            },
        )
        host = _Host(_BlockedStdout())
        with pytest.raises(TransportError, match="host_cleanup_uncertain") as failed:
            _open_host(
                host,
                command=_run_command(log),
                lookup=tracked.lookup,
                stop=tracked.stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert failed.value.cleanup == "uncertain"
        assert remaining.poll() is None
        assert tracked.stopped == []

        mismatched = _Children()
        mismatched.add(reused, ticks="999")
        other = tmp_path / "reuse.log"
        _write_ledger(
            other,
            {
                "expectedExe": sys.executable,
                "controllerPid": 1,
                "baseline": [],
                "owned": [_record(reused.pid, ticks="100")],
            },
        )
        second = _Host(_QuietStdout())
        second.returncode = 0
        with pytest.raises(
            TransportError, match="host_cleanup_uncertain"
        ) as reused_error:
            _open_host(
                second,
                command=_run_command(other),
                lookup=mismatched.lookup,
                stop=mismatched.stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert reused_error.value.cleanup == "uncertain"
        assert mismatched.stopped == []
        assert reused.poll() is None
        assert second.terminated is False
    finally:
        _stop_sleepers(remaining, reused)


def test_missing_ledger_after_a_forced_stop_is_uncertain(tmp_path: Path) -> None:
    bystander = _sleeper()
    try:
        log = tmp_path / "host.log"
        host = _Host(_BlockedStdout())

        def lookup(pid: int) -> ProcessIdentity | None:
            raise AssertionError(f"lookup {pid}")

        def stop(pid: int) -> None:
            raise AssertionError(f"stop {pid}")

        with pytest.raises(TransportError, match="host_cleanup_uncertain") as error:
            _open_host(
                host,
                command=_run_command(log),
                lookup=lookup,
                stop=stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert error.value.cleanup == "uncertain"
        assert host.terminated is True
        assert bystander.poll() is None
        assert Path(str(log) + ".stop").read_text(encoding="utf-8") == _RUN
        assert not Path(str(log) + ".owned.json").exists()
    finally:
        _stop_sleepers(bystander)


def test_unexpected_exit_without_a_ledger_is_uncertain(tmp_path: Path) -> None:
    child = _sleeper()
    try:
        log = tmp_path / "host.log"
        _write_phase(log, _RUN, "launching")
        host = _Host(_QuietStdout())
        host.returncode = 0

        def lookup(pid: int) -> ProcessIdentity | None:
            raise AssertionError(f"lookup {pid}")

        def stop(pid: int) -> None:
            raise AssertionError(f"stop {pid}")

        with pytest.raises(TransportError, match="host_cleanup_uncertain") as error:
            _open_host(
                host,
                command=_run_command(log),
                lookup=lookup,
                stop=stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert error.value.cleanup == "uncertain"
        assert host.terminated is False and host.killed is False
        assert child.poll() is None
        assert (
            Path(f"{log}.run.{_RUN}.json").read_text(encoding="utf-8").find("launching")
            >= 0
        )
    finally:
        _stop_sleepers(child)


def test_confirmed_prelaunch_refusal_reports_closed(tmp_path: Path) -> None:
    log = tmp_path / "host.log"
    _write_phase(log, _RUN, "prelaunch")
    host = _Host(_QuietStdout(), exit_on_stop=True)
    host.stdout = io.StringIO('{"ready": false, "error": "launch not confirmed"}\n')
    with pytest.raises(TransportError, match="launch not confirmed") as error:
        _open_host(
            host,
            command=_run_command(log),
            timeout=0.2,
            shutdown_timeout=0.05,
            cleanup_timeout=0.05,
        )
    assert error.value.cleanup == "closed"
    assert classify_host_failure(error.value) == "launch_not_confirmed"
    assert host.terminated is False


def test_incomplete_ledger_is_uncertain_and_stops_nothing(tmp_path: Path) -> None:
    child = _sleeper()
    try:
        log = tmp_path / "host.log"
        _write_phase(log, _RUN, "owned", expectedExe=sys.executable)
        host = _Host(_QuietStdout())
        host.returncode = 0

        def stop(pid: int) -> None:
            raise AssertionError(f"stop {pid}")

        with pytest.raises(TransportError, match="host_cleanup_uncertain") as error:
            _open_host(
                host,
                command=_run_command(log),
                lookup=lambda pid: ProcessIdentity(pid, "100", sys.executable),
                stop=stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert error.value.cleanup == "uncertain"
        assert child.poll() is None
        assert host.terminated is False
    finally:
        _stop_sleepers(child)


def test_stale_sidecars_are_not_current_run_evidence(tmp_path: Path) -> None:
    stale_child = _sleeper()
    try:
        log = tmp_path / "host.log"
        stale_stop = Path(str(log) + ".stop")
        stale_stop.write_text(_STALE_RUN, encoding="utf-8")
        stale_owned = Path(str(log) + ".owned.json")
        stale_record = {
            "evidenceId": _STALE_RUN,
            "phase": "owned",
            "expectedExe": sys.executable,
            "controllerPid": 1,
            "baseline": [],
            "owned": [_record(stale_child.pid)],
        }
        stale_owned.write_text(json.dumps(stale_record), encoding="utf-8")
        stale_run = _write_phase(
            log,
            _STALE_RUN,
            "owned",
            expectedExe=sys.executable,
            controllerPid=1,
            baseline=[],
            owned=[_record(stale_child.pid)],
        )
        stale_bytes = stale_run.read_bytes()
        owned_bytes = stale_owned.read_bytes()
        _write_phase(log, _RUN, "prelaunch")
        host = _Host(_QuietStdout())
        host.stdout = io.StringIO('{"ready": false, "error": "launch not confirmed"}\n')
        host.returncode = 0

        def stop(pid: int) -> None:
            raise AssertionError(f"stop {pid}")

        with pytest.raises(TransportError, match="launch not confirmed") as error:
            _open_host(
                host,
                command=_run_command(log),
                stop=stop,
                timeout=0.2,
                shutdown_timeout=0.05,
                cleanup_timeout=0.05,
            )
        assert error.value.cleanup == "closed"
        assert classify_host_failure(error.value) == "launch_not_confirmed"
        assert stale_child.poll() is None
        assert stale_run.read_bytes() == stale_bytes
        assert stale_owned.read_bytes() == owned_bytes
        assert stale_stop.read_text(encoding="utf-8") == _STALE_RUN
        assert host.terminated is False
    finally:
        _stop_sleepers(stale_child)
