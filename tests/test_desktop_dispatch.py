"""Automatic desktop dispatch with scripted transports.

The Codex client and the Grok adapter are the real classes with their real
exactly-once journals. Only the pipe and the CDP session are replaced, and the
scripted "model" is a function of the message it receives. No application is
launched and no model is called.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.codex_desktop.client import CodexDesktopClient
from transcription_review_agent.codex_desktop.journal import AttemptJournal
from transcription_review_agent.codex_desktop.pipe import PipeCall, PipeError
from transcription_review_agent.desktop_orchestration import __main__ as orchestration
from transcription_review_agent.desktop_orchestration.senders import (
    CodexDesktopSender,
    GrokDesktopSender,
    windows_attachment,
)
from transcription_review_agent.grok_desktop.adapter import GrokDesktopAdapter
from transcription_review_agent.grok_desktop.registry import ResourceRegistry
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.desktop_dispatch import (
    DispatchReply,
    DispatchTask,
    PreparedSend,
    dispatch_task,
    reply_delivery_prompt,
)
from transcription_review_agent.review.desktop_e2e import register_desktop_first_pass
from transcription_review_agent.review.document_run import (
    DocumentRun,
    advance_document,
    run_payload,
)
from transcription_review_agent.review.errors import ReviewIntegrityError
from test_codex_desktop import THREAD, _focus
from test_desktop_e2e import QUEUE, _fixture, _mount
from test_document_run import (
    GPT_TEXT,
    GROK_TEXT,
    _challenge_file,
    _controller,
    _queue_document,
)
from test_grok_desktop import AGENT, NAME, Clock, greeting, turn_poll
from test_review_workflow import BATCH_ID, DOCUMENT_ID, NOW, OCR


def _request_id(message: str) -> str:
    found = re.search(r"request_id: (\S+)", message)
    assert found is not None
    return found.group(1)


class Models:
    """Scripted model replies, chosen by the request id a message carries."""

    def __init__(self, tmp_path: Path, controller: ReviewController) -> None:
        self.tmp_path = tmp_path
        registered = register_desktop_first_pass(
            controller,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            mount_root=_mount(tmp_path),
        )
        self.requests = {item.provider: item for item in registered.work_requests}
        self.first_pass = {
            "gpt": _fixture(self.requests["gpt"], GPT_TEXT).decode("utf-8"),
            "grok": _fixture(self.requests["grok"], GROK_TEXT).decode("utf-8"),
        }
        self.decision = {"gpt": "CANDIDATE_X", "grok": "CANDIDATE_Y"}
        self.override: dict[str, str | None] = {}
        self.asked: list[str] = []

    def reply(self, message: str) -> str | None:
        request_id = _request_id(message)
        self.asked.append(request_id)
        if request_id in self.override:
            return self.override[request_id]
        provider = "gpt" if ".gpt." in request_id else "grok"
        if ".first_pass." in request_id:
            return self.first_pass[provider]
        root = (
            _queue_document(self.tmp_path).parent
            / f"document_{DOCUMENT_ID}_challenge_round_01"
            / provider
        )
        return _challenge_file(root, self.decision[provider]).decode("utf-8")

    def id_for(self, provider: str, stage: str) -> str:
        task = self.requests[provider].task_id
        if stage == "challenge":
            task = task.replace(":first_pass:r0:", ":challenge:r1:")
        return task.replace(":", ".")


class FakeCodex:
    """A Codex thread. ``send`` records the turn the way Desktop would."""

    def __init__(self, models: Models) -> None:
        self.models = models
        self.sent: list[str] = []
        self.turns: list[dict[str, object]] = []
        self.break_pipe_after_send = False
        self.steal_focus = False

    def call(
        self, method: str, params: dict[str, object], *, timeout_s: float
    ) -> PipeCall:
        del method, timeout_s
        arguments = params["arguments"]
        assert isinstance(arguments, dict)
        if params["tool"] == "send_message_to_thread":
            prompt = str(arguments["prompt"])
            self.sent.append(prompt)
            reply = self.models.reply(prompt)
            items: list[dict[str, object]] = [
                {
                    "type": "userMessage",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            if reply is not None:
                items.append(
                    {"type": "agentMessage", "phase": "final_answer", "text": reply}
                )
            self.turns.append(
                {
                    "id": f"turn-{len(self.turns) + 1}",
                    "status": "completed" if reply is not None else "inProgress",
                    "items": items,
                }
            )
            if self.break_pipe_after_send:
                raise PipeError("pipe closed after the send")
            focus = _focus(same=not self.steal_focus, codex=self.steal_focus)
            return PipeCall(response=_ok({"threadId": THREAD}), focus=focus)
        body = {
            "thread": {"id": THREAD, "cwd": r"C:\work", "status": {"type": "idle"}},
            "turns": list(reversed(self.turns)),
        }
        return PipeCall(response=_ok(body), focus=_focus())

    def complete(self, index: int, reply: str) -> None:
        turn = self.turns[index]
        turn["status"] = "completed"
        items = turn["items"]
        assert isinstance(items, list)
        items.append({"type": "agentMessage", "phase": "final_answer", "text": reply})


def _ok(body: dict[str, object]) -> dict[str, object]:
    return {
        "id": "1",
        "jsonrpc": "2.0",
        "result": {
            "success": True,
            "contentItems": [{"type": "inputText", "text": json.dumps(body)}],
        },
    }


class GrokRig:
    """Opens scripted CDP sessions and records what the composer received."""

    def __init__(self, models: Models) -> None:
        self.models = models
        self.opened = 0
        self.messages: list[str] = []
        self.attachments: list[tuple[str, ...]] = []
        self.refuse_bot = False

    def open(self) -> GrokSession:
        self.opened += 1
        return GrokSession(self)


class GrokSession:
    def __init__(self, rig: GrokRig) -> None:
        self.rig = rig
        self.staged: list[str] = []
        self.message = ""

    def evaluate(self, expression: str) -> object:
        if "grok-desktop:select" in expression:
            return {"ok": not self.rig.refuse_bot}
        if "grok-desktop:preflight" in expression:
            return greeting()
        if "grok-desktop:file-input" in expression:
            return {"ok": True, "count": 1, "multiple": True, "disabled": False}
        if "grok-desktop:composer-files" in expression:
            return {
                "ok": True,
                "files": [],
                "chips": list(self.staged),
                "labels": list(self.staged),
            }
        if "grok-desktop:focus" in expression:
            return {"ok": True}
        if "grok-desktop:submit" in expression:
            return {"ok": True}
        if "grok-desktop:poll" in expression:
            reply = self.rig.models.reply(self.message)
            if reply is None:
                raise TimeoutError("poll")
            return turn_poll(["idle", "finished", "idle"], reply)
        raise AssertionError(expression)

    def insert_text(self, text: str) -> None:
        self.message = text
        self.rig.messages.append(text)

    def set_input_files(self, selector: str, files: tuple[str, ...]) -> None:
        del selector
        self.rig.attachments.append(tuple(files))
        self.staged = [item.replace("\\", "/").rsplit("/", 1)[-1] for item in files]

    def close(self) -> None:
        pass


class Rig:
    def __init__(self, tmp_path: Path, controller: ReviewController) -> None:
        self.tmp_path = tmp_path
        self.controller = controller
        self.models = Models(tmp_path, controller)
        self.codex = FakeCodex(self.models)
        self.grok = GrokRig(self.models)
        self.windows_paths: list[str] = []

    def senders(self) -> dict[str, object]:
        clock = {"now": 0.0}

        def sleep(seconds: float) -> None:
            clock["now"] += seconds

        client = CodexDesktopClient(
            self.codex,
            AttemptJournal(self.tmp_path / "codex-attempts.json"),
            completion_timeout_s=1,
            poll_s=1,
            sleeper=sleep,
            clock=lambda: clock["now"],
        )
        adapter = GrokDesktopAdapter(
            registry=ResourceRegistry(self.tmp_path / "grok-registry.json"),
            diagnostics_dir=self.tmp_path / "grok-diagnostics",
            open_session=self.grok.open,
            check_profile=lambda: None,
            clock=Clock(),
        )

        def attachment(task: DispatchTask, path: Path) -> str:
            self.windows_paths.append(windows_attachment(task, path))
            return str(path)

        return {
            "gpt": CodexDesktopSender(client, thread_id=THREAD),
            "grok": GrokDesktopSender(
                adapter, agent_id=AGENT, bot_name=NAME, attachment_path=attachment
            ),
        }

    def advance(self, controller: ReviewController | None = None, **kwargs: object):
        return advance_document(
            controller or self.controller,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            preservation_dir=self.tmp_path / "raw",
            mount_root=_mount(self.tmp_path),
            senders=self.senders(),  # type: ignore[arg-type]
            **kwargs,  # type: ignore[arg-type]
        )

    def restarted(self) -> ReviewController:
        return ReviewController(
            batches_dir=self.controller.batches_dir,
            reviews_dir=self.controller.reviews_dir,
            config=self.controller.config,
            clock=lambda: NOW,
        )


def _rig(tmp_path: Path) -> Rig:
    return Rig(tmp_path, _controller(tmp_path))


def _challenge_root(tmp_path: Path) -> Path:
    return _queue_document(tmp_path).parent / (
        f"document_{DOCUMENT_ID}_challenge_round_01"
    )


def _dispatch(result: DocumentRun) -> dict[tuple[str, str], str]:
    return {
        (str(item["provider"]), str(item["stage"])): str(item["outcome"])
        for item in result.dispatch
    }


def test_one_command_reaches_the_pending_review_page(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    pending_dir = tmp_path / "pending"
    result = rig.advance(packet_dir=pending_dir)
    assert result.stopped_at == "waiting_for_human"
    assert result.blocker == "human_challenge_closure"
    assert result.finalized is False
    assert result.challenge_reconciled is True
    assert result.providers == {"gpt": "SUCCEEDED", "grok": "SUCCEEDED"}
    assert _dispatch(result) == {
        ("gpt", "first_pass"): "installed",
        ("grok", "first_pass"): "installed",
        ("gpt", "challenge"): "installed",
        ("grok", "challenge"): "installed",
    }
    assert len(rig.codex.sent) == 2
    assert len(rig.grok.messages) == 2
    assert all(item["decision"] == "UNRESOLVED" for item in result.decisions)
    assert result.packet_dir == str(pending_dir)
    page = (pending_dir / "index.html").read_text(encoding="utf-8")
    assert "Pending human review" in page
    assert "No reading is approved" in page
    assert GPT_TEXT in page and GROK_TEXT in page
    assert OCR not in page
    payload = run_payload(result)
    assert len(payload["dispatch"]) == 4  # type: ignore[arg-type]
    assert json.dumps(payload)  # serialisable


def test_raw_replies_and_provenance_are_preserved_exactly(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.advance(packet_dir=tmp_path / "pending")
    document = _queue_document(tmp_path)
    raw_dir = tmp_path / "raw" / "dispatch"
    for provider in ("gpt", "grok"):
        expected = rig.models.first_pass[provider].encode("utf-8")
        request_id = rig.models.id_for(provider, "first_pass")
        output = document / provider / "output" / f"transcription_{provider}.json"
        raw = raw_dir / "first_pass" / provider / f"{request_id}.raw.txt"
        assert output.read_bytes() == expected
        assert raw.read_bytes() == expected
        provenance = json.loads(
            raw.with_name(f"{request_id}.provenance.json").read_text(encoding="utf-8")
        )
        task = document / provider
        image = next(task.glob("original.*")).read_bytes()
        assert provenance["evidence_sha256"] == [sha256_bytes(image)]
        assert provenance["raw_reply_sha256"] == sha256_bytes(expected)
        assert provenance["task_id"] == rig.models.requests[provider].task_id
        assert provenance["job_id"] == rig.models.requests[provider].job_id
        assert provenance["source_bundle_sha256"] == (
            rig.models.requests[provider].source_bundle_sha256
        )
        assert provenance["provider"] == provider
    codex = json.loads(
        (
            raw_dir
            / "first_pass"
            / "gpt"
            / f"{rig.models.id_for('gpt', 'first_pass')}.provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert codex["mechanism"] == "codex_app_tools_pipe"
    assert codex["transport"]["turn_id"] == "turn-1"
    assert codex["transport"]["thread_id"] == THREAD
    grok = json.loads(
        (
            raw_dir
            / "first_pass"
            / "grok"
            / f"{rig.models.id_for('grok', 'first_pass')}.provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert grok["mechanism"] == "grok_composer_file_input"
    assert grok["transport"]["cleanup"] == "closed"
    assert grok["attachment_sha256"]
    challenge = json.loads(
        (
            raw_dir
            / "challenge_round_01"
            / "grok"
            / f"{rig.models.id_for('grok', 'challenge')}.provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert challenge["round_index"] == 1
    page = (_challenge_root(tmp_path) / "grok" / "input" / "page_001.png").read_bytes()
    assert sha256_bytes(page) in challenge["evidence_sha256"]


def test_prompts_match_each_providers_interface(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.advance(packet_dir=tmp_path / "pending")
    first_codex, challenge_codex = rig.codex.sent
    document = _queue_document(tmp_path)
    for message in (first_codex, challenge_codex):
        assert "Delivery override" in message
        assert "Do not write any file" in message
        assert "Write the model transcription" not in message
        assert "Write one JSON file and no other path" not in message
        assert "desktop_queue.json" not in message
        assert "Identity values, copy exactly" in message
    assert "Transcribe every page image" in first_codex
    assert "Candidate readings are anonymous" in (
        document.parent
        / f"document_{DOCUMENT_ID}_challenge_round_01"
        / "gpt"
        / "instructions_gpt.txt"
    ).read_text(encoding="utf-8")
    assert "D:\\archive-queue\\tasks\\batch-000001\\document_101\\gpt" in first_codex
    packet = (
        document.parent
        / f"document_{DOCUMENT_ID}_dispatch_packets"
        / "grok"
        / "first_pass_packet.txt"
    ).read_text(encoding="utf-8")
    assert "Transcribe every page image" in packet
    assert "output_schema.json" not in packet
    assert "work_request.json" not in packet
    assert "Write one JSON file" not in packet
    assert '"stage": "first_pass"' in packet
    assert "Shape template, not a reading" in packet
    assert rig.models.requests["grok"].task_id in packet
    first_grok = rig.grok.messages[0]
    assert len(first_grok) <= 2000
    assert "first_pass_packet.txt" in first_grok
    names = [
        item.replace("\\", "/").rsplit("/", 1)[-1] for item in rig.grok.attachments[0]
    ]
    assert names == ["original.png", "first_pass_packet.txt"]
    challenge_names = [
        item.replace("\\", "/").rsplit("/", 1)[-1] for item in rig.grok.attachments[1]
    ]
    assert "page_001.png" in challenge_names
    assert challenge_names[-1] == "challenge_packet.txt"
    assert all(
        item.startswith("D:\\archive-queue\\tasks\\") for item in rig.windows_paths
    )
    assert not any("document_101\\gpt" in item for item in rig.windows_paths)


def test_providers_never_see_each_other(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.advance(packet_dir=tmp_path / "pending")
    queue = _queue_document(tmp_path)
    grok_visible = list(rig.grok.messages)
    for packet in (queue.parent).rglob("*_packet.txt"):
        grok_visible.append(packet.read_text(encoding="utf-8"))
    gpt_ids = (
        rig.models.requests["gpt"].task_id,
        rig.models.id_for("gpt", "first_pass"),
        rig.models.id_for("gpt", "challenge"),
        "\\gpt",
        "/gpt",
    )
    for text in grok_visible:
        for needle in gpt_ids:
            assert needle not in text
    # A first pass is independent: neither provider sees the other's reading.
    # A challenge shows both readings, anonymously, by design.
    first_pass_texts = [
        rig.grok.messages[0],
        (
            queue.parent
            / f"document_{DOCUMENT_ID}_dispatch_packets"
            / "grok"
            / "first_pass_packet.txt"
        ).read_text(encoding="utf-8"),
    ]
    for text in first_pass_texts:
        assert GPT_TEXT not in text
    assert GROK_TEXT not in rig.codex.sent[0]
    for text in rig.codex.sent:
        for needle in (
            rig.models.requests["grok"].task_id,
            rig.models.id_for("grok", "first_pass"),
            rig.models.id_for("grok", "challenge"),
            "\\grok",
            "/grok",
        ):
            assert needle not in text
    for text in (*rig.codex.sent, *grok_visible):
        assert OCR not in text
    # The Grok send is chat plus attachments only: no task path to read.
    for text in rig.grok.messages:
        assert "Write one JSON file" not in text
        assert not re.search(r"[A-Za-z]:\\", text)


def test_second_run_after_completion_sends_nothing(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    first = rig.advance(packet_dir=tmp_path / "pending")
    sent = (len(rig.codex.sent), len(rig.grok.messages), rig.grok.opened)
    again = rig.advance(rig.restarted(), packet_dir=tmp_path / "pending")
    assert (len(rig.codex.sent), len(rig.grok.messages), rig.grok.opened) == sent
    assert again.dispatch == ()
    assert again.ledger_sequence == first.ledger_sequence
    assert again.stopped_at == "waiting_for_human"
    assert again.finalized is False


def test_interrupted_send_resumes_without_a_second_send(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.codex.break_pipe_after_send = True
    first = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(first)[("gpt", "first_pass")] == "uncertain"
    assert _dispatch(first)[("grok", "first_pass")] == "installed"
    assert first.stopped_at == "waiting_for_first_pass"
    assert first.blocker == "dispatch_uncertain"
    assert first.providers == {"gpt": "AWAITING_MANUAL_DISPATCH", "grok": "SUCCEEDED"}
    assert [item["executable"] for item in first.execution] == [True, True]
    assert len(rig.codex.sent) == 1
    # The reply arrives later. A new process reads the same thread.
    rig.codex.break_pipe_after_send = False
    rig.codex.complete(0, rig.models.first_pass["gpt"])
    grok_opened = rig.grok.opened
    resumed = rig.advance(rig.restarted(), packet_dir=tmp_path / "pending")
    # The first pass was sent once. The only new Codex send is the challenge.
    assert len(rig.codex.sent) == 2
    assert [_request_id(item) for item in rig.codex.sent] == [
        rig.models.id_for("gpt", "first_pass"),
        rig.models.id_for("gpt", "challenge"),
    ]
    assert _dispatch(resumed)[("gpt", "first_pass")] == "installed"
    assert ("grok", "first_pass") not in _dispatch(resumed)
    assert resumed.stopped_at == "waiting_for_human"
    assert resumed.challenge_reconciled is True
    assert resumed.finalized is False
    assert rig.grok.opened == grok_opened + 1  # only the Grok challenge


def test_codex_timeout_is_recovered_not_resent(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    gpt_first = rig.models.id_for("gpt", "first_pass")
    rig.models.override[gpt_first] = None
    first = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(first)[("gpt", "first_pass")] == "uncertain"
    assert first.blocker == "dispatch_uncertain"
    rig.codex.complete(0, rig.models.first_pass["gpt"])
    resumed = rig.advance(rig.restarted(), packet_dir=tmp_path / "pending")
    assert _dispatch(resumed)[("gpt", "first_pass")] == "installed"
    assert [_request_id(item) for item in rig.codex.sent].count(gpt_first) == 1
    assert resumed.stopped_at == "waiting_for_human"


def test_invalid_reply_is_preserved_and_never_resent(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    gpt_first = rig.models.id_for("gpt", "first_pass")
    rig.models.override[gpt_first] = "Here is the transcription: not JSON"
    first = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(first)[("gpt", "first_pass")] == "invalid_reply"
    assert first.blocker == "dispatch_invalid_reply"
    assert first.providers["gpt"] == "AWAITING_MANUAL_DISPATCH"
    assert first.stopped_at == "waiting_for_first_pass"
    raw = tmp_path / "raw" / "dispatch" / "first_pass" / "gpt" / f"{gpt_first}.raw.txt"
    assert raw.read_text(encoding="utf-8") == "Here is the transcription: not JSON"
    gpt_output = _queue_document(tmp_path) / "gpt" / "output"
    assert not (gpt_output / "transcription_gpt.json").exists()
    sent = len(rig.codex.sent)
    again = rig.advance(rig.restarted(), packet_dir=tmp_path / "pending")
    assert len(rig.codex.sent) == sent
    assert _dispatch(again)[("gpt", "first_pass")] == "invalid_reply"
    assert again.ledger_sequence == first.ledger_sequence


def test_reply_for_another_task_is_not_installed(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    grok_first = rig.models.id_for("grok", "first_pass")
    wrong = json.loads(rig.models.first_pass["grok"])
    wrong["task_id"] = rig.models.requests["gpt"].task_id
    rig.models.override[grok_first] = json.dumps(wrong)
    result = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(result)[("grok", "first_pass")] == "invalid_reply"
    assert result.blocker == "dispatch_invalid_reply"
    assert not (
        _queue_document(tmp_path) / "grok" / "output" / "transcription_grok.json"
    ).exists()


def test_template_placeholder_reply_is_not_accepted(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    example = (_queue_document(tmp_path) / "gpt" / "output_example.json").read_text(
        encoding="utf-8"
    )
    rig.models.override[rig.models.id_for("gpt", "first_pass")] = example
    result = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(result)[("gpt", "first_pass")] == "invalid_reply"


def test_grok_uncertain_and_not_sent_are_not_launched_again(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    grok_first = rig.models.id_for("grok", "first_pass")
    rig.models.override[grok_first] = None
    first = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(first)[("grok", "first_pass")] == "uncertain"
    assert first.blocker == "dispatch_uncertain"
    opened = rig.grok.opened
    again = rig.advance(rig.restarted(), packet_dir=tmp_path / "pending")
    assert rig.grok.opened == opened
    assert _dispatch(again)[("grok", "first_pass")] == "uncertain"
    assert again.stopped_at == "waiting_for_first_pass"
    assert len(rig.grok.messages) == 1


def test_grok_refusal_before_send_is_recorded_not_retried(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.grok.refuse_bot = True
    first = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(first)[("grok", "first_pass")] == "not_sent"
    assert first.blocker == "dispatch_not_sent"
    assert rig.grok.messages == []
    rig.grok.refuse_bot = False
    opened = rig.grok.opened
    again = rig.advance(rig.restarted(), packet_dir=tmp_path / "pending")
    assert rig.grok.opened == opened
    assert _dispatch(again)[("grok", "first_pass")] == "not_sent"


def test_codex_focus_theft_stops_all_dispatch(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    rig.codex.steal_focus = True
    result = rig.advance(packet_dir=tmp_path / "pending")
    assert _dispatch(result) == {("gpt", "first_pass"): "blocked"}
    assert result.blocker == "dispatch_blocked"
    assert rig.grok.opened == 0
    assert len(rig.codex.sent) == 1
    again = rig.advance(rig.restarted(), packet_dir=tmp_path / "pending")
    assert len(rig.codex.sent) == 1
    assert _dispatch(again) == {("gpt", "first_pass"): "blocked"}
    assert rig.grok.opened == 0


def test_tampered_sealed_image_is_refused_before_any_send(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    image = next((_queue_document(tmp_path) / "gpt").glob("original.*"))
    image.write_bytes(image.read_bytes() + b"x")
    with pytest.raises(ReviewIntegrityError):
        rig.advance(packet_dir=tmp_path / "pending")
    assert rig.codex.sent == []
    assert rig.grok.opened == 0


def test_provider_reply_cannot_change_the_sealed_page(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    queue = _queue_document(tmp_path)

    class Tamper:
        def __init__(self, inner: CodexDesktopSender) -> None:
            self.inner = inner

        def prepare(self, task: DispatchTask) -> PreparedSend:
            return self.inner.prepare(task)

        def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
            reply = self.inner.send(task, prepared)
            image = next(task.task_dir.glob("original.*"))
            image.write_bytes(b"changed during the turn")
            return reply

    senders = rig.senders()
    senders["gpt"] = Tamper(senders["gpt"])  # type: ignore[arg-type]
    with pytest.raises(ReviewIntegrityError):
        advance_document(
            rig.controller,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            preservation_dir=tmp_path / "raw",
            mount_root=_mount(tmp_path),
            senders=senders,  # type: ignore[arg-type]
        )
    assert not (queue / "gpt" / "output" / "transcription_gpt.json").exists()
    raw = list((tmp_path / "raw" / "dispatch").rglob("*.raw.txt"))
    assert len(raw) == 1  # the reply is kept as evidence, never installed


class _Stub:
    def __init__(self, message: str, attachments: tuple[Path, ...] = ()) -> None:
        self.message = message
        self.attachments = attachments
        self.sent = 0

    def prepare(self, task: DispatchTask) -> PreparedSend:
        return PreparedSend(
            request_id=task.request_id,
            message=self.message.replace("{id}", task.request_id),
            attachments=self.attachments,
            mechanism="stub",
        )

    def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
        del task, prepared
        self.sent += 1
        return DispatchReply("reply", "{}", "", {})


def _task(tmp_path: Path) -> DispatchTask:
    rig = _rig(tmp_path)
    queue = _queue_document(tmp_path)
    request = rig.models.requests["grok"]
    directory = queue / "grok"
    image = next(directory.glob("original.*"))
    return DispatchTask(
        request=request,
        task_dir=directory,
        windows_task_dir="D:\\q\\grok",
        output_path=directory / "output" / "transcription_grok.json",
        packet_dir=tmp_path / "packets",
        windows_packet_dir="D:\\packets",
        evidence_sha256=(sha256_bytes(image.read_bytes()),),
        forbidden_texts=(str(queue / "gpt"), "D:\\q\\gpt", "SECRET ARCHIVE TEXT"),
    )


@pytest.mark.parametrize(
    "message",
    [
        "request_id: {id} see D:\\q\\gpt\\output",
        "request_id: {id} read SECRET ARCHIVE TEXT",
        "no correlation tag at all",
    ],
)
def test_unsafe_message_is_refused_before_the_send(
    tmp_path: Path, message: str
) -> None:
    task = _task(tmp_path)
    stub = _Stub(message)
    with pytest.raises(ReviewIntegrityError):
        dispatch_task(
            stub,
            task,
            preservation_dir=tmp_path / "raw",
            validate=lambda _raw, _request: None,
        )
    assert stub.sent == 0


def test_attachment_outside_the_task_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path)
    other = next((task.task_dir.parent / "gpt").glob("original.*"))
    stub = _Stub("request_id: {id}", (other,))
    with pytest.raises(ReviewIntegrityError):
        dispatch_task(
            stub,
            task,
            preservation_dir=tmp_path / "raw",
            validate=lambda _raw, _request: None,
        )
    assert stub.sent == 0


def test_preserved_reply_is_reused_without_calling_the_sender(tmp_path: Path) -> None:
    task = _task(tmp_path)
    stub = _Stub("request_id: {id}")
    first = dispatch_task(
        stub,
        task,
        preservation_dir=tmp_path / "raw",
        validate=lambda _raw, _request: "rejected",
    )
    assert first.outcome == "invalid_reply"
    assert stub.sent == 1
    second = dispatch_task(
        stub,
        task,
        preservation_dir=tmp_path / "raw",
        validate=lambda _raw, _request: "rejected",
    )
    assert second.outcome == "invalid_reply"
    assert stub.sent == 1
    assert first.raw_sha256 == second.raw_sha256 == sha256_bytes(b"{}")


def test_delivery_rewrite_refuses_a_drifted_prompt() -> None:
    with pytest.raises(ReviewIntegrityError):
        reply_delivery_prompt("Transcribe the page.\n", "first_pass")
    with pytest.raises(ReviewIntegrityError):
        reply_delivery_prompt("Challenge text without delivery lines.\n", "challenge")


def test_controller_stops_before_closure_and_publication(tmp_path: Path) -> None:
    rig = _rig(tmp_path)
    result = rig.advance(packet_dir=tmp_path / "pending")
    status = rig.controller.status(BATCH_ID, DOCUMENT_ID)
    assert status.finalized is False
    assert status.challenge_closure is None
    assert result.blocker == "human_challenge_closure"
    options = {action.dest for action in orchestration._parser()._actions}
    assert not options & {"finalize", "closure_mode", "reason"}


def test_without_senders_nothing_is_sent_and_reporting_is_unchanged(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path)
    result = advance_document(
        rig.controller,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        queue_root=QUEUE,
        preservation_dir=tmp_path / "raw",
        mount_root=_mount(tmp_path),
    )
    assert result.stopped_at == "waiting_for_first_pass"
    assert result.blocker == "desktop_first_pass"
    assert result.dispatch == ()
    assert [item["executable"] for item in result.execution] == [False, False]
    assert rig.codex.sent == [] and rig.grok.opened == 0


def test_live_command_refuses_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(_args: object) -> object:
        raise AssertionError("senders must not be built")

    monkeypatch.setattr(orchestration, "build_senders", explode)
    base = [
        "run",
        "--batches-dir",
        str(tmp_path / "b"),
        "--reviews-dir",
        str(tmp_path / "r"),
        "--batch-id",
        BATCH_ID,
        "--document-id",
        str(DOCUMENT_ID),
        "--queue-root",
        QUEUE,
        "--preservation-dir",
        str(tmp_path / "raw"),
        "--packet-dir",
        str(tmp_path / "pending"),
    ]
    for name in (
        "DESKTOP_DISPATCH_CONFIRM",
        "GROK_ISO_CONFIRM",
        "GROK_IMAGE_SUBMIT",
    ):
        monkeypatch.delenv(name, raising=False)
    assert orchestration.main(base) == 2
    assert "not confirmed" in capsys.readouterr().err
    monkeypatch.setenv("DESKTOP_DISPATCH_CONFIRM", "1")
    assert orchestration.main(base) == 2
    assert "grok launch not confirmed" in capsys.readouterr().err
    monkeypatch.setenv("GROK_ISO_CONFIRM", "1")
    monkeypatch.setenv("GROK_IMAGE_SUBMIT", "1")
    assert orchestration.main(base) == 2
    assert "grok needs" in capsys.readouterr().err
    assert orchestration.main([*base, "--providers", "gpt"]) == 2
    assert "codex needs" in capsys.readouterr().err


def test_orchestration_has_no_forbidden_execution_paths() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "transcription_review_agent"
    text = "".join(
        path.read_text(encoding="utf-8")
        for path in (root / "desktop_orchestration").glob("*.py")
    ) + (root / "review" / "desktop_dispatch.py").read_text(encoding="utf-8")
    for banned in (
        "codex exec",
        "CodexCliRunner",
        "app-server",
        "openai",
        "api.x.ai",
        "SetForegroundWindow",
        "localToolPermission",
    ):
        assert banned not in text
    adapter = (root / "grok_desktop" / "adapter.py").read_text(encoding="utf-8")
    assert "assert_isolated_profile" in adapter


# --- evidence integrity across resume -------------------------------------


def _blocked_markers(root: Path) -> list[Path]:
    return sorted((root / "dispatch").rglob("*.integrity_blocked.json"))


def test_integrity_failure_is_durable_across_restart_and_restore(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path)
    image = next((_queue_document(tmp_path) / "gpt").glob("original.*"))
    original = image.read_bytes()

    class Tamper:
        def __init__(self, inner: CodexDesktopSender) -> None:
            self.inner = inner

        def prepare(self, task: DispatchTask) -> PreparedSend:
            return self.inner.prepare(task)

        def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
            reply = self.inner.send(task, prepared)
            image.write_bytes(b"changed during the turn")
            return reply

    senders = rig.senders()
    senders["gpt"] = Tamper(senders["gpt"])  # type: ignore[arg-type]
    with pytest.raises(ReviewIntegrityError):
        advance_document(
            rig.controller,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            preservation_dir=tmp_path / "raw",
            mount_root=_mount(tmp_path),
            senders=senders,  # type: ignore[arg-type]
        )
    assert len(rig.codex.sent) == 1
    markers = _blocked_markers(tmp_path / "raw")
    assert len(markers) == 1
    raw_files = list((tmp_path / "raw" / "dispatch").rglob("*.raw.txt"))
    assert len(raw_files) == 1  # kept for audit

    image.write_bytes(original)  # the evidence is whole again
    restarted = rig.restarted()
    result = rig.advance(controller=restarted, packet_dir=tmp_path / "pending")
    assert _dispatch(result)[("gpt", "first_pass")] == "integrity_blocked"
    assert result.providers.get("gpt") != "SUCCEEDED"
    assert not (
        _queue_document(tmp_path) / "gpt" / "output" / "transcription_gpt.json"
    ).exists()
    assert len(rig.codex.sent) == 1  # no second send
    assert rig.grok.opened == 0  # nothing else is dispatched past the block
    assert result.finalized is False
    assert result.packet_dir is None

    again = rig.advance(controller=rig.restarted(), packet_dir=tmp_path / "pending")
    assert _dispatch(again)[("gpt", "first_pass")] == "integrity_blocked"
    assert len(rig.codex.sent) == 1


def test_restored_image_does_not_unblock_a_marked_attempt(tmp_path: Path) -> None:
    task = _task(tmp_path)
    image = next(task.task_dir.glob("original.*"))
    original = image.read_bytes()

    class Mutating(_Stub):
        def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
            image.write_bytes(b"changed")
            return super().send(task, prepared)

    stub = Mutating("request_id: {id}")
    with pytest.raises(ReviewIntegrityError):
        dispatch_task(
            stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: None
        )
    image.write_bytes(original)
    installs: list[bytes] = []

    def validate(raw: bytes, _request: object) -> None:
        installs.append(raw)

    record = dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=validate
    )
    assert record.outcome == "integrity_blocked"
    assert installs == [] and stub.sent == 1
    assert not task.output_path.exists()


def test_integrity_marker_that_is_a_symlink_still_blocks(tmp_path: Path) -> None:
    task = _task(tmp_path)
    stub = _Stub("request_id: {id}")
    dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: "no"
    )
    marker = next((tmp_path / "raw" / "dispatch").rglob("*.raw.txt")).with_name(
        f"{task.request_id}.integrity_blocked.json"
    )
    marker.symlink_to(tmp_path / "nowhere")
    record = dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: None
    )
    assert record.outcome == "integrity_blocked"
    assert stub.sent == 1


# --- provenance verification ----------------------------------------------


def _first_dispatch(tmp_path: Path) -> tuple[DispatchTask, _Stub, Path]:
    task = _task(tmp_path)
    stub = _Stub("request_id: {id}")
    dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: "no"
    )
    directory = next((tmp_path / "raw" / "dispatch").rglob("*.raw.txt")).parent
    return task, stub, directory


def _resume(task: DispatchTask, stub: _Stub, tmp_path: Path) -> object:
    return dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: "no"
    )


def _edit(path: Path, **changes: object) -> None:
    body = json.loads(path.read_text(encoding="utf-8"))
    body.update(changes)
    path.write_text(json.dumps(body), encoding="utf-8")


@pytest.mark.parametrize(
    "changes",
    [
        {"raw_reply_sha256": "0" * 64},
        {"raw_reply_bytes": 99},
        {"message_sha256": "1" * 64},
        {"request_id": "someone.else"},
        {"task_id": "other:task"},
        {"evidence_sha256": ["2" * 64]},
        {"attachment_sha256": ["3" * 64]},
        {"batch_id": "other-batch"},
        {"source_bundle_sha256": "4" * 64},
        {"schema_version": "bogus"},
    ],
)
def test_inconsistent_provenance_is_rejected_and_kept(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    provenance = directory / f"{task.request_id}.provenance.json"
    _edit(provenance, **changes)
    tampered = provenance.read_bytes()
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert provenance.read_bytes() == tampered  # never silently replaced
    assert stub.sent == 1


@pytest.mark.parametrize("content", [b"{not json", b"[]", b"", b"\xff\xfe"])
def test_corrupt_provenance_is_rejected_and_kept(
    tmp_path: Path, content: bytes
) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    provenance = directory / f"{task.request_id}.provenance.json"
    provenance.write_bytes(content)
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert provenance.read_bytes() == content
    assert stub.sent == 1


def test_missing_key_in_provenance_is_rejected(tmp_path: Path) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    provenance = directory / f"{task.request_id}.provenance.json"
    body = json.loads(provenance.read_text(encoding="utf-8"))
    del body["message_sha256"]
    provenance.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert stub.sent == 1


def test_symlinked_provenance_is_rejected(tmp_path: Path) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    provenance = directory / f"{task.request_id}.provenance.json"
    real = tmp_path / "elsewhere.json"
    real.write_bytes(provenance.read_bytes())
    provenance.unlink()
    provenance.symlink_to(real)
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert provenance.is_symlink()
    assert stub.sent == 1


def test_symlinked_preserved_reply_is_rejected_without_a_send(tmp_path: Path) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    raw = directory / f"{task.request_id}.raw.txt"
    real = tmp_path / "elsewhere.txt"
    real.write_bytes(raw.read_bytes())
    raw.unlink()
    raw.symlink_to(real)
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert stub.sent == 1


def test_altered_preserved_reply_is_rejected(tmp_path: Path) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    raw = directory / f"{task.request_id}.raw.txt"
    raw.write_bytes(b'{"changed": true}')
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert stub.sent == 1


def test_drifted_message_does_not_reuse_a_preserved_reply(tmp_path: Path) -> None:
    task, stub, _directory = _first_dispatch(tmp_path)
    drifted = _Stub("request_id: {id} and a different instruction")
    with pytest.raises(ReviewIntegrityError):
        _resume(task, drifted, tmp_path)
    assert drifted.sent == 0


def test_preserved_reply_without_any_record_is_rejected(tmp_path: Path) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    (directory / f"{task.request_id}.provenance.json").unlink()
    (directory / f"{task.request_id}.attempt.json").unlink()
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert stub.sent == 1
    assert not (directory / f"{task.request_id}.provenance.json").exists()


def test_crash_between_reply_and_provenance_is_accounted_and_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transcription_review_agent.review import desktop_dispatch

    task = _task(tmp_path)
    stub = _Stub("request_id: {id}")

    def crash(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("process died")

    monkeypatch.setattr(desktop_dispatch, "_publish_provenance", crash)
    with pytest.raises(RuntimeError):
        dispatch_task(
            stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: "no"
        )
    monkeypatch.undo()
    directory = next((tmp_path / "raw" / "dispatch").rglob("*.raw.txt")).parent
    assert (directory / f"{task.request_id}.raw.txt").is_file()
    assert not (directory / f"{task.request_id}.provenance.json").exists()

    record = _resume(task, stub, tmp_path)
    # Provenance is accounted for from the attempt record, but the reply was
    # never verified after the send, so it is not usable. No resend.
    assert record.outcome == "integrity_blocked"  # type: ignore[attr-defined]
    assert stub.sent == 1
    body = json.loads(
        (directory / f"{task.request_id}.provenance.json").read_text(encoding="utf-8")
    )
    assert body["recovered_after_interruption"] is True
    assert body["raw_reply_sha256"] == sha256_bytes(b"{}")
    assert (directory / f"{task.request_id}.raw.txt").read_bytes() == b"{}"
    assert _resume(task, stub, tmp_path).outcome == "integrity_blocked"  # type: ignore[attr-defined]
    assert stub.sent == 1


def test_attempt_record_for_another_request_blocks_recovery(tmp_path: Path) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    (directory / f"{task.request_id}.provenance.json").unlink()
    _edit(directory / f"{task.request_id}.attempt.json", message_sha256="5" * 64)
    with pytest.raises(ReviewIntegrityError):
        _resume(task, stub, tmp_path)
    assert stub.sent == 1
    assert not (directory / f"{task.request_id}.provenance.json").exists()


def test_send_leaves_a_matching_attempt_record_before_the_reply(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    seen: list[bool] = []

    class Watching(_Stub):
        def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
            directory = tmp_path / "raw" / "dispatch"
            seen.append(any(directory.rglob("*.attempt.json")))
            seen.append(any(directory.rglob("*.raw.txt")))
            return super().send(task, prepared)

    dispatch_task(
        Watching("request_id: {id}"),
        task,
        preservation_dir=tmp_path / "raw",
        validate=lambda *_: "no",
    )
    assert seen == [True, False]


# --- review page for a document with no dispute ---------------------------


def test_agreeing_first_passes_still_reach_a_pending_review_page(
    tmp_path: Path,
) -> None:
    rig = _rig(tmp_path)
    rig.models.first_pass["grok"] = _fixture(
        rig.models.requests["grok"], GPT_TEXT
    ).decode("utf-8")
    pending = tmp_path / "pending"
    result = rig.advance(packet_dir=pending)

    assert result.stopped_at == "waiting_for_human"
    assert result.blocker == "human_finalize"
    assert result.finalized is False
    assert result.providers == {"gpt": "SUCCEEDED", "grok": "SUCCEEDED"}
    assert result.disputes == ()
    assert _dispatch(result) == {
        ("gpt", "first_pass"): "installed",
        ("grok", "first_pass"): "installed",
    }
    assert result.packet_dir == str(pending)  # a packet, though nothing disputed
    assert all(".challenge." not in item for item in rig.models.asked)

    page = (pending / "index.html").read_text(encoding="utf-8")
    assert "Pending human review" in page
    assert "No human verification or approval has occurred" in page
    assert "No reading is approved" in page
    assert "The two first-pass readings agree" in page
    assert "No challenge or reconciliation was run" in page
    assert "GPT" in page and "Grok" in page
    assert GPT_TEXT in page
    assert OCR not in page
    assert page.index("<img") < page.index("GPT")  # image first
    image = pending / "page.png"
    sealed = next((_queue_document(tmp_path) / "gpt").glob("original.*"))
    assert image.read_bytes() == sealed.read_bytes()
    assert {item.name for item in pending.iterdir()} == {"index.html", "page.png"}

    status = rig.controller.status(BATCH_ID, DOCUMENT_ID)
    assert status.finalized is False
    assert status.challenge_closure is None
    _state, document = rig.controller._require_document(BATCH_ID, DOCUMENT_ID)
    assert not document.challenge_rounds
    assert not document.reconciled_decisions

    # resuming sends nothing more and leaves the page in place
    sent = (len(rig.codex.sent), len(rig.grok.messages))
    again = rig.advance(controller=rig.restarted(), packet_dir=pending)
    assert again.blocker == "human_finalize"
    assert (len(rig.codex.sent), len(rig.grok.messages)) == sent


# --- durable post-send verification ---------------------------------------


def _crash_after_preservation(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """Make the post-send evidence check die the way a killed process would."""

    from transcription_review_agent.review import desktop_dispatch

    state = {"sent": False}
    real = desktop_dispatch._require_evidence

    def check(task: DispatchTask) -> None:
        if state["sent"]:
            raise RuntimeError("process died before the post-send check")
        real(task)

    monkeypatch.setattr(desktop_dispatch, "_require_evidence", check)
    return state


def test_crash_before_post_send_verification_is_not_trusted_after_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _task(tmp_path)
    image = next(task.task_dir.glob("original.*"))
    original = image.read_bytes()
    state = _crash_after_preservation(monkeypatch)

    class Crashing(_Stub):
        def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
            reply = super().send(task, prepared)
            image.write_bytes(b"changed while the process was away")
            state["sent"] = True
            return reply

    stub = Crashing("request_id: {id}")
    with pytest.raises(RuntimeError):
        dispatch_task(
            stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: None
        )
    monkeypatch.undo()
    directory = next((tmp_path / "raw" / "dispatch").rglob("*.raw.txt")).parent
    assert (directory / f"{task.request_id}.raw.txt").is_file()
    assert (directory / f"{task.request_id}.provenance.json").is_file()
    assert not (directory / f"{task.request_id}.verified.json").exists()

    image.write_bytes(original)  # restored: this proves nothing about the run
    accepted: list[bytes] = []

    def validate(raw: bytes, _request: object) -> None:
        accepted.append(raw)

    record = dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=validate
    )
    assert record.outcome == "integrity_blocked"
    assert record.installed_path is None
    assert accepted == [] and stub.sent == 1
    assert not task.output_path.exists()
    assert (directory / f"{task.request_id}.raw.txt").read_bytes() == b"{}"
    assert (directory / f"{task.request_id}.integrity_blocked.json").is_file()
    again = dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=validate
    )
    assert again.outcome == "integrity_blocked"
    assert accepted == [] and stub.sent == 1


def test_crash_before_post_send_verification_blocks_the_whole_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = _rig(tmp_path)
    image = next((_queue_document(tmp_path) / "gpt").glob("original.*"))
    original = image.read_bytes()
    state = _crash_after_preservation(monkeypatch)

    class Crash:
        def __init__(self, inner: CodexDesktopSender) -> None:
            self.inner = inner

        def prepare(self, task: DispatchTask) -> PreparedSend:
            return self.inner.prepare(task)

        def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
            reply = self.inner.send(task, prepared)
            image.write_bytes(b"changed while the process was away")
            state["sent"] = True
            return reply

    senders = rig.senders()
    senders["gpt"] = Crash(senders["gpt"])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        advance_document(
            rig.controller,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            queue_root=QUEUE,
            preservation_dir=tmp_path / "raw",
            mount_root=_mount(tmp_path),
            senders=senders,  # type: ignore[arg-type]
        )
    monkeypatch.undo()
    assert len(rig.codex.sent) == 1

    image.write_bytes(original)
    result = rig.advance(controller=rig.restarted(), packet_dir=tmp_path / "pending")
    assert _dispatch(result)[("gpt", "first_pass")] == "integrity_blocked"
    assert result.providers.get("gpt") != "SUCCEEDED"
    assert len(rig.codex.sent) == 1
    assert rig.grok.opened == 0
    assert not (
        _queue_document(tmp_path) / "gpt" / "output" / "transcription_gpt.json"
    ).exists()
    assert result.packet_dir is None


def test_verified_reply_is_installed_on_resume_without_a_send(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    stub = _Stub("request_id: {id}")
    dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: "no"
    )
    directory = next((tmp_path / "raw" / "dispatch").rglob("*.raw.txt")).parent
    assert (directory / f"{task.request_id}.verified.json").is_file()
    record = dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: None
    )
    assert record.outcome == "installed"
    assert stub.sent == 1
    assert task.output_path.read_bytes() == b"{}"


@pytest.mark.parametrize("damage", ["delete", "corrupt", "wrong_hash", "symlink"])
def test_missing_or_unusable_verification_record_fails_closed(
    tmp_path: Path, damage: str
) -> None:
    task, stub, directory = _first_dispatch(tmp_path)
    verified = directory / f"{task.request_id}.verified.json"
    if damage == "delete":
        verified.unlink()
    elif damage == "corrupt":
        verified.write_bytes(b"{nope")
    elif damage == "wrong_hash":
        _edit(verified, raw_reply_sha256="0" * 64)
    else:
        real = tmp_path / "elsewhere.json"
        real.write_bytes(verified.read_bytes())
        verified.unlink()
        verified.symlink_to(real)
    record = dispatch_task(
        stub, task, preservation_dir=tmp_path / "raw", validate=lambda *_: None
    )
    assert record.outcome == "integrity_blocked"
    assert stub.sent == 1
    assert not task.output_path.exists()
    assert (directory / f"{task.request_id}.raw.txt").read_bytes() == b"{}"


# --- multi-page agreement packet ------------------------------------------


def _two_page_controller(tmp_path: Path) -> ReviewController:
    from test_review_workflow import (
        _agreeing_pages,
        _controller as workflow_controller,
        _deliver_first_pass,
        _png,
    )

    controller = workflow_controller(tmp_path, [_png(20, 20), _png(30, 20)])
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    for provider in ("gpt", "grok"):
        _deliver_first_pass(controller, provider, _agreeing_pages())
    controller.compare(BATCH_ID, DOCUMENT_ID)
    return controller


def _sealed_pages(controller: ReviewController) -> list[tuple[str, Path, str]]:
    from transcription_review_agent.review.paths import document_root

    _state, document = controller._require_document(BATCH_ID, DOCUMENT_ID)
    root = document_root(controller.reviews_dir, BATCH_ID, DOCUMENT_ID)
    return [
        (item.page_id, root / "input" / item.original_filename, item.sha256)
        for item in document.pages
    ]


def test_agreement_packet_shows_every_sealed_page(tmp_path: Path) -> None:
    from transcription_review_agent.review.human_verification_packet import (
        build_pending_agreement_packet,
    )

    controller = _two_page_controller(tmp_path)
    output = tmp_path / "pending"
    build_pending_agreement_packet(
        controller, batch_id=BATCH_ID, document_id=DOCUMENT_ID, output_dir=output
    )
    pages = _sealed_pages(controller)
    assert len(pages) == 2
    assert {item.name for item in output.iterdir()} == {
        "index.html",
        "page_001.png",
        "page_002.png",
    }
    for page_id, source, sha in pages:
        copied = output / f"{page_id}.png"
        assert copied.read_bytes() == source.read_bytes()
        assert sha256_bytes(copied.read_bytes()) == sha
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "2 page(s)" in html
    assert "No human verification or approval has occurred" in html
    assert html.count("<img") == 2
    assert "src='page_001.png'" in html and "src='page_002.png'" in html
    for page_id, _source, sha in pages:
        assert sha in html
    # each reading sits in the section of its own page
    first = html.index("id='page_001'")
    second = html.index("id='page_002'")
    assert first < second
    one, two = html[first:second], html[second:]
    assert "The visible line is secure." in one
    assert "The visible line is secure." not in two
    assert "Second page remains." in two
    assert "Second page remains." not in one
    assert one.count("<pre>") == 2 and two.count("<pre>") == 2
    # the ledger was not touched
    status = controller.status(BATCH_ID, DOCUMENT_ID)
    assert status.finalized is False


def test_agreement_packet_refuses_a_missing_page_image(tmp_path: Path) -> None:
    from transcription_review_agent.review.human_verification_packet import (
        build_pending_agreement_packet,
    )

    controller = _two_page_controller(tmp_path)
    _page_id, source, _sha = _sealed_pages(controller)[1]
    source.unlink()
    output = tmp_path / "pending"
    with pytest.raises(ReviewIntegrityError, match="page_002"):
        build_pending_agreement_packet(
            controller, batch_id=BATCH_ID, document_id=DOCUMENT_ID, output_dir=output
        )
    assert not output.exists()  # no page that silently omits an image


def test_agreement_packet_refuses_a_page_hash_mismatch(tmp_path: Path) -> None:
    from transcription_review_agent.review.human_verification_packet import (
        build_pending_agreement_packet,
    )

    controller = _two_page_controller(tmp_path)
    _page_id, source, _sha = _sealed_pages(controller)[1]
    source.write_bytes(source.read_bytes() + b"x")
    output = tmp_path / "pending"
    with pytest.raises(ReviewIntegrityError, match="page_002"):
        build_pending_agreement_packet(
            controller, batch_id=BATCH_ID, document_id=DOCUMENT_ID, output_dir=output
        )
    assert not output.exists()


def test_agreement_packet_refuses_a_hash_mismatch_on_the_first_page(
    tmp_path: Path,
) -> None:
    from transcription_review_agent.review.human_verification_packet import (
        build_pending_agreement_packet,
    )

    controller = _two_page_controller(tmp_path)
    _page_id, source, _sha = _sealed_pages(controller)[0]
    source.write_bytes(b"not the sealed page")
    with pytest.raises(ReviewIntegrityError, match="page_001"):
        build_pending_agreement_packet(
            controller,
            batch_id=BATCH_ID,
            document_id=DOCUMENT_ID,
            output_dir=tmp_path / "pending",
        )
