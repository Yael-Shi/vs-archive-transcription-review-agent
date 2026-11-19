"""Codex Desktop pipe client, focus rule, and challenge handoff. No live turn."""

from __future__ import annotations

import ast
import base64
import json
import subprocess
from pathlib import Path

import pytest

from transcription_review_agent.codex_desktop.challenge import (
    accept_installed_challenge,
    challenge_desktop_request,
    install_challenge_reply,
    plan_codex_challenge,
)
from transcription_review_agent.codex_desktop.client import (
    CodexDesktopClient,
    attributed_reply,
    tool_call_params,
)
from transcription_review_agent.codex_desktop.focus import (
    focus_verdict,
    is_codex_desktop,
)
from transcription_review_agent.codex_desktop.framing import (
    FrameError,
    decode_frame,
    encode_frame,
)
from transcription_review_agent.codex_desktop.journal import AttemptJournal
from transcription_review_agent.codex_desktop.models import (
    CodexRequest,
    FocusObservation,
    ForegroundWindow,
)
from transcription_review_agent.codex_desktop.journal import request_fingerprint
from transcription_review_agent.codex_desktop.pipe import (
    HELPER,
    PipeCall,
    PipeEndpoint,
    PipeError,
    PowershellCodexTransport,
    parse_response_bytes,
    select_endpoint,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewUsageError,
)
from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.models import BBox, CropRecord, WorkRequest
from test_review_workflow import (
    BATCH_ID,
    DOCUMENT_ID,
    _controller,
    _deliver_first_pass,
    _page,
)

THREAD = "01a15e54-908c-7762-9bf6-ae96160f4156"
REQUEST = "ipc-test-01"
CODEX = r"C:\Program Files\WindowsApps\OpenAI.Codex_26\app\ChatGPT.exe"


class Scripted:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(
        self, method: str, params: dict[str, object], *, timeout_s: float
    ) -> PipeCall:
        del timeout_s
        self.calls.append((method, params))
        item = self.results.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, PipeCall)
        return item


def _window(
    hwnd: int = 7, pid: int = 11, executable: str = r"C:\Windows\explorer.exe"
) -> ForegroundWindow:
    return ForegroundWindow(hwnd=hwnd, pid=pid, executable=executable)


def _focus(*, same: bool = True, codex: bool = False) -> FocusObservation:
    before = _window()
    if same:
        after = before
    elif codex:
        after = _window(hwnd=99, pid=50, executable=CODEX)
    else:
        after = _window(hwnd=8, pid=12, executable=r"C:\Windows\notepad.exe")
    return FocusObservation(before=before, after=after)


def _send_ok() -> PipeCall:
    return PipeCall(
        response={
            "id": "1",
            "jsonrpc": "2.0",
            "result": {
                "success": True,
                "contentItems": [
                    {"type": "inputText", "text": json.dumps({"threadId": THREAD})}
                ],
            },
        },
        focus=_focus(),
    )


def _read(text: str, *, request_id: str = REQUEST) -> PipeCall:
    body = {
        "thread": {"id": THREAD, "cwd": r"C:\work", "status": {"type": "idle"}},
        "turns": [
            {
                "id": "turn-1",
                "status": "completed",
                "items": [
                    {
                        "type": "userMessage",
                        "content": [{"type": "text", "text": request_id}],
                    },
                    {"type": "agentMessage", "phase": "final_answer", "text": text},
                ],
            }
        ],
    }
    return PipeCall(
        response={
            "id": "2",
            "jsonrpc": "2.0",
            "result": {
                "success": True,
                "contentItems": [{"type": "inputText", "text": json.dumps(body)}],
            },
        },
        focus=_focus(),
    )


def _request(message: str | None = None) -> CodexRequest:
    text = message or f"Reply with the tag {REQUEST} and nothing else."
    return CodexRequest(thread_id=THREAD, request_id=REQUEST, message=text)


def test_frames_round_trip_and_reject_a_partial_body() -> None:
    frame = encode_frame({"jsonrpc": "2.0", "id": "a", "method": "tools/list"})
    payload, rest = decode_frame(frame + b"tail")
    assert payload["method"] == "tools/list"
    assert rest == b"tail"
    with pytest.raises(FrameError):
        decode_frame(frame[:6])


def test_pipe_owner_must_be_the_one_desktop_process() -> None:
    trusted = PipeEndpoint(name=r"\\.\pipe\codex-app-tools-1", pid=4, executable=CODEX)
    other = PipeEndpoint(
        name=r"\\.\pipe\codex-app-tools-2", pid=5, executable=r"C:\other\ChatGPT.exe"
    )
    assert select_endpoint([other, trusted]) == trusted
    with pytest.raises(PipeError):
        select_endpoint([trusted, trusted])
    assert is_codex_desktop(CODEX)
    assert focus_verdict(_focus()) == "preserved"
    assert focus_verdict(_focus(same=False, codex=True)) == "codex_activated"
    assert focus_verdict(_focus(same=False)) == "foreground_changed"


def test_helper_does_not_activate_a_window() -> None:
    source = HELPER.read_text(encoding="utf-8")
    assert "responseBase64" in source
    assert "ConvertFrom-Json" not in source.split("frameBase64", 1)[-1]
    for banned in (
        "SetForegroundWindow",
        "ShowWindow",
        "SendInput",
        "keybd_event",
        "mouse_event",
    ):
        assert banned not in source
    tree = ast.parse(
        Path("src/transcription_review_agent/codex_desktop/client.py").read_text(
            encoding="utf-8"
        )
    )
    imported = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(
        name.startswith("transcription_review_agent.grok_desktop") for name in imported
    )


def test_send_is_attributed_and_a_second_call_does_not_send(tmp_path: Path) -> None:
    transport = Scripted([_send_ok(), _read("CODEX_DESKTOP_IPC_OK")])
    client = CodexDesktopClient(
        transport,
        AttemptJournal(tmp_path / "attempts.json"),
        sleeper=lambda _s: None,
        clock=lambda: 0,
    )
    first = client.send(_request())
    assert first.status.value == "success"
    assert first.sent is True
    assert first.turn_id == "turn-1"
    assert first.reply == "CODEX_DESKTOP_IPC_OK"
    assert first.focus_preserved is True
    assert transport.calls[0][1]["tool"] == "send_message_to_thread"
    assert transport.calls[0][1]["threadId"] == THREAD
    again = client.send(_request())
    assert again.replayed is True
    assert again.reply == "CODEX_DESKTOP_IPC_OK"
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1


def _fast_client(tmp_path: Path, transport: Scripted) -> CodexDesktopClient:
    clock = {"now": 0.0}

    def advance() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    return CodexDesktopClient(
        transport,
        AttemptJournal(tmp_path / "attempts.json"),
        completion_timeout_s=1,
        poll_s=1,
        sleeper=sleep,
        clock=advance,
    )


def test_uncertain_send_is_not_repeated(tmp_path: Path) -> None:
    transport = Scripted(
        [
            PipeError("closed"),
            _read("no match", request_id="other"),
            _read("no match", request_id="other"),
        ]
    )
    client = _fast_client(tmp_path, transport)
    failed = client.send(_request())
    assert failed.status.value == "uncertain"
    assert failed.sent is True
    recovered = client.send(_request())
    assert recovered.status.value == "timeout"
    assert recovered.replayed is True
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1
    assert [params["tool"] for _method, params in transport.calls].count(
        "read_thread"
    ) == 2


def test_ambiguous_rpc_error_is_not_resent(tmp_path: Path) -> None:
    transport = Scripted(
        [
            PipeCall(
                response={
                    "id": "1",
                    "jsonrpc": "2.0",
                    "error": {"code": -32000, "message": "failed"},
                },
                focus=_focus(),
            ),
            _read("no match", request_id="other"),
            _read("no match", request_id="other"),
        ]
    )
    client = _fast_client(tmp_path, transport)
    result = client.send(_request())
    assert result.status.value == "uncertain"
    replay = client.send(_request())
    assert replay.replayed is True
    assert replay.status.value == "timeout"
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1


def test_focus_theft_blocks_without_another_send(tmp_path: Path) -> None:
    transport = Scripted(
        [PipeCall(response=_send_ok().response, focus=_focus(same=False, codex=True))]
    )
    client = CodexDesktopClient(transport, AttemptJournal(tmp_path / "attempts.json"))
    result = client.send(_request())
    assert result.status.value == "blocked"
    assert result.sent is True
    assert result.focus_preserved is False
    replay = client.send(_request())
    assert replay.replayed is True
    assert len(transport.calls) == 1


def test_completion_timeout_stays_sent(tmp_path: Path) -> None:
    transport = Scripted(
        [
            _send_ok(),
            _read("no match", request_id="other"),
            _read("no match", request_id="other"),
            _read("no match", request_id="other"),
            _read("no match", request_id="other"),
        ]
    )
    clock = {"now": 0.0}

    def advance() -> float:
        return clock["now"]

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    client = CodexDesktopClient(
        transport,
        AttemptJournal(tmp_path / "attempts.json"),
        completion_timeout_s=1,
        poll_s=1,
        sleeper=sleep,
        clock=advance,
    )
    result = client.send(_request())
    assert result.status.value == "timeout"
    assert result.sent is True
    assert client.send(_request()).replayed is True
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1
    recovery = [
        params for _method, params in transport.calls if params["tool"] == "read_thread"
    ]
    assert len(recovery) == 4
    assert len({params["callId"] for params in recovery}) == 4


def test_two_matching_turns_are_not_guessed() -> None:
    body = {
        "turns": [
            {
                "id": "a",
                "status": "completed",
                "items": [
                    {"type": "agentMessage", "phase": "final_answer", "text": REQUEST}
                ],
            },
            {
                "id": "b",
                "status": "completed",
                "items": [
                    {"type": "agentMessage", "phase": "final_answer", "text": REQUEST}
                ],
            },
        ]
    }
    assert attributed_reply(body, REQUEST) is None
    user = {
        "turns": [
            {
                "id": "a",
                "status": "completed",
                "items": [
                    {
                        "type": "userMessage",
                        "content": [{"type": "text", "text": REQUEST}],
                    },
                    {"type": "agentMessage", "phase": "final_answer", "text": "one"},
                ],
            },
            {
                "id": "b",
                "status": "completed",
                "items": [
                    {
                        "type": "userMessage",
                        "content": [{"type": "text", "text": f"also {REQUEST}"}],
                    },
                    {"type": "agentMessage", "phase": "final_answer", "text": "two"},
                ],
            },
        ]
    }
    with pytest.raises(ValueError, match="more than one Desktop turn"):
        attributed_reply(user, REQUEST)
    first = tool_call_params(
        _request(), tool="read_thread", arguments={"threadId": THREAD}, call_id="one"
    )
    second = tool_call_params(
        _request(), tool="read_thread", arguments={"threadId": THREAD}, call_id="two"
    )
    assert first["callId"] == "one"
    assert second["callId"] == "two"
    assert first["turnId"] != second["turnId"]
    assert first["threadId"] == THREAD
    assert REQUEST not in first["callId"]


def _task(tmp_path: Path, *, workspace: Path | None = None) -> tuple[Path, Path]:
    root = workspace or tmp_path / "desktop_pilot"
    task = root / "queue" / "gpt"
    image = task / "input" / "page_001.png"
    image.parent.mkdir(parents=True)
    page_bytes = b"\x89PNG\r\n\x1a\n"
    crop_bytes = b"\x89PNG\r\n\x1a\ncrop"
    image.write_bytes(page_bytes)
    (task / "input" / "crops").mkdir()
    (task / "input" / "crops" / "crop.png").write_bytes(crop_bytes)
    record = CropRecord(
        crop_id="crop_page",
        dispute_id="dsp_1",
        page_id="page_001",
        page_sha256=sha256_bytes(page_bytes),
        page_width=8,
        page_height=8,
        target_bbox=BBox(x=1, y=1, width=4, height=4),
        crop_bbox=BBox(x=0, y=0, width=8, height=8),
        enlarged=False,
        window_kind="TARGETED",
        coordinate_sources=["region.bbox"],
        image_filename="crop.png",
        crop_sha256=sha256_bytes(crop_bytes),
        crop_png_width=8,
        crop_png_height=8,
    )
    (task / "input" / "crops" / "crop_page.json").write_text(
        record.model_dump_json(), encoding="utf-8"
    )
    request = {
        "schema_version": "review-work-request-v1",
        "provider": "gpt",
        "provider_interface": "GPT_WORK",
        "task_id": "batch-000903:1:challenge:r1:gpt:a1",
        "job_id": "job:batch-000903:1:challenge:r1:gpt:a1",
        "batch_id": "batch-000903",
        "document_id": 1,
        "stage": "challenge",
        "round_index": 1,
        "attempt": 1,
        "source_bundle_sha256": "a" * 64,
        "dispatch": "manual",
        "unattended_api": False,
        "output_relative_path": "challenge_round_01/output/challenge_gpt.json",
        "page_ids": ["page_001"],
        "forbidden_inputs": [],
    }
    (task / "work_request.json").write_text(json.dumps(request), encoding="utf-8")
    (task / "prompt.txt").write_text(
        "Read the sealed page and crop. Do not use the other provider.\n",
        encoding="utf-8",
    )
    return task, root


def test_wsl_mount_is_inside_the_windows_workspace() -> None:
    from transcription_review_agent.codex_desktop.challenge import _inside

    workspace = Path(
        r"C:\Users\User\vs-archive-transcription-review-agent\desktop_pilot"
    )
    image = Path(
        "/mnt/c/Users/User/vs-archive-transcription-review-agent/"
        "desktop_pilot/image_probe_03/page.png"
    )
    assert _inside(workspace, image)
    assert not _inside(workspace, Path("/mnt/c/Users/User/other/page.png"))


def test_images_must_sit_inside_the_bound_workspace(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    plan = plan_codex_challenge(
        task, thread_id=THREAD, workspace=workspace, transport_request_id="codex-img-01"
    )
    assert len(plan.images) == 2
    assert all(str(path).startswith(str(workspace)) for path in plan.images)
    assert "codex-img-01" in plan.message
    with pytest.raises(ReviewIntegrityError):
        plan_codex_challenge(
            task,
            thread_id=THREAD,
            workspace=tmp_path / "elsewhere",
            transport_request_id="codex-img-01",
        )


def test_invalid_challenge_json_is_preserved_and_not_installed(tmp_path: Path) -> None:
    task, _workspace = _task(tmp_path)
    request = WorkRequest.model_validate_json(
        (task / "work_request.json").read_text(encoding="utf-8")
    )
    saved = install_challenge_reply(
        reviews_dir=tmp_path / "reviews",
        work_request=request,
        raw=b"not json",
        preservation_dir=tmp_path / "raw",
    )
    assert saved.accepted is False
    assert saved.raw_path.read_bytes() == b"not json"
    assert saved.review_output_path is None
    with pytest.raises(ReviewUsageError):
        accept_installed_challenge(
            None,  # type: ignore[arg-type]
            saved,
            batch_id="batch-000903",
            document_id=1,
            round_index=1,
        )


def test_valid_challenge_is_accepted_without_finalizing(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    controller.open_sealed_document(BATCH_ID, DOCUMENT_ID)
    bbox = BBox(x=4, y=4, width=4, height=4)
    _deliver_first_pass(
        controller,
        "gpt",
        [
            _page("page_001", 1, "Ionnes", name="Ionnes", bbox=bbox),
            _page("page_002", 2, "Same."),
        ],
    )
    _deliver_first_pass(
        controller,
        "grok",
        [
            _page("page_001", 1, "Iannes", name="Iannes", bbox=bbox),
            _page("page_002", 2, "Same."),
        ],
    )
    controller.compare(BATCH_ID, DOCUMENT_ID)
    gpt_path = controller.publish(BATCH_ID, DOCUMENT_ID, "gpt", stage="challenge")
    controller.publish(BATCH_ID, DOCUMENT_ID, "grok", stage="challenge")
    request = WorkRequest.model_validate_json(gpt_path.read_text(encoding="utf-8"))
    disputes = json.loads(
        (gpt_path.parent / "input" / "disputes.json").read_text(encoding="utf-8")
    )
    dispute_id = disputes["required_decision_ids"][0]
    ref = disputes["groups"][0]["evidence_refs"][0]
    raw = json.dumps(
        {
            "schema_version": "review-provider-output-v1",
            "provider": "gpt",
            "provider_interface": "GPT_WORK",
            "task_id": request.task_id,
            "job_id": request.job_id,
            "batch_id": request.batch_id,
            "document_id": request.document_id,
            "attempt": request.attempt,
            "source_bundle_sha256": request.source_bundle_sha256,
            "stage": "challenge",
            "round_index": request.round_index,
            "decisions": [
                {
                    "dispute_id": dispute_id,
                    "decision": "UNRESOLVED",
                    "third_reading": None,
                    "visual_evidence": "The crop does not settle the letters.",
                    "evidence_ref": ref,
                    "evidence_assessment": "INSUFFICIENT",
                }
            ],
        }
    ).encode("utf-8")
    saved = install_challenge_reply(
        reviews_dir=controller.reviews_dir,
        work_request=request,
        raw=raw,
        preservation_dir=tmp_path / "raw",
        disputes=disputes,
    )
    assert saved.accepted is True
    assert saved.review_output_path is not None
    assert saved.review_output_path.read_bytes() == raw
    status = accept_installed_challenge(
        controller,
        saved,
        batch_id=BATCH_ID,
        document_id=DOCUMENT_ID,
        round_index=1,
    )
    assert status.finalized is False
    assert status.ledger_sequence > 0
    grok_output = saved.review_output_path.with_name("challenge_grok.json")
    assert not grok_output.exists()


def test_omitted_user_message_matches_the_send_delegation() -> None:
    request_id = "challenge_gpt.json"
    reply = "Created and validated challenge_gpt.json."
    body = {
        "turns": [
            {
                "id": "01a1f037-a1ed-7c02-b8d7-66b41724b750",
                "status": "completed",
                "items": [
                    {
                        "type": "functionCallOutput",
                        "id": "fco_01a1f037-a1ee-7c72-a340-17e2e1228a33",
                        "name": "send_message_to_thread",
                        "namespace": "codex_app",
                        "output": {
                            "text": (
                                "<codex_delegation>\n"
                                "  <source_thread_id>01a15e54-908c-7762-9bf6-ae96160f4156</source_thread_id>\n"
                                f"  <input>Read the sealed page.\n{request_id}\n</input>\n"
                                "</codex_delegation>"
                            ),
                            "truncated": False,
                        },
                    },
                    {
                        "type": "commandExecution",
                        "command": f"Get-Content {request_id}",
                    },
                    {"type": "agentMessage", "phase": "final_answer", "text": reply},
                ],
            },
            {
                "id": "newer-unrelated",
                "status": "completed",
                "items": [
                    {
                        "type": "agentMessage",
                        "phase": "final_answer",
                        "text": f"The tag {request_id} is only in this reply.",
                    }
                ],
            },
        ]
    }
    turn_id, text = attributed_reply(body, request_id)
    assert turn_id == "01a1f037-a1ed-7c02-b8d7-66b41724b750"
    assert text == reply
    by_parent = {
        "turns": [
            {
                "id": "01a1f037-a1ed-7c02-b8d7-66b41724b750",
                "status": "completed",
                "parent_turn_id": "codex-desktop-turn-challenge-gpt-json",
                "items": [
                    {"type": "agentMessage", "phase": "final_answer", "text": reply}
                ],
            },
            body["turns"][1],
        ]
    }
    assert attributed_reply(
        by_parent,
        request_id,
        dispatch_turn_id="codex-desktop-turn-challenge-gpt-json",
    ) == ("01a1f037-a1ed-7c02-b8d7-66b41724b750", reply)


def test_large_raw_reply_keeps_unicode_bytes() -> None:
    text = "HARBOUR " + ("שלום " * 4000) + '"quote"'
    raw = json.dumps({"result": {"text": text}}, ensure_ascii=False).encode("utf-8")
    assert parse_response_bytes(raw)["result"]["text"] == text


def test_changed_image_is_not_the_same_attempt(tmp_path: Path) -> None:
    task, root = _task(tmp_path)
    plan = plan_codex_challenge(
        task, thread_id=THREAD, workspace=root, transport_request_id=REQUEST
    )
    bound = challenge_desktop_request(plan)
    assert bound.image_sha256 == plan.image_sha256
    plain = request_fingerprint(_request())
    assert "image_sha256" not in json.dumps(
        {"message": _request().message, "request_id": REQUEST, "thread_id": THREAD}
    )
    assert plain == request_fingerprint(_request())
    (task / "input" / "page_001.png").write_bytes(b"\x89PNG\r\n\x1a\nchanged")
    with pytest.raises(ReviewIntegrityError):
        challenge_desktop_request(plan)
    journal = AttemptJournal(tmp_path / "attempts.json")
    transport = Scripted([_send_ok(), _read("ok")])
    client = CodexDesktopClient(
        transport, journal, sleeper=lambda _s: None, clock=lambda: 0
    )
    sent = client.send(bound)
    assert sent.status.value == "success"
    changed = CodexRequest(
        thread_id=bound.thread_id,
        request_id=bound.request_id,
        message=bound.message,
        image_sha256=("f" * 64,),
    )
    blocked = client.send(changed)
    assert blocked.status.value == "error"
    assert blocked.reason == "request_id_reused"
    assert blocked.sent is False
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1


def _thread_page(
    turns: list[dict[str, object]], *, cursor: str | None, has_more: bool
) -> PipeCall:
    body = {
        "turns": turns,
        "page": {
            "order": "newest_first",
            "limit": 5,
            "hasMore": has_more,
            "nextCursor": cursor,
        },
    }
    return PipeCall(
        response={
            "id": "2",
            "jsonrpc": "2.0",
            "result": {
                "success": True,
                "contentItems": [{"type": "inputText", "text": json.dumps(body)}],
            },
        },
        focus=_focus(),
    )


def test_older_turn_is_found_on_the_next_page(tmp_path: Path) -> None:
    decoys = [
        {
            "id": f"new-{index}",
            "status": "completed",
            "items": [
                {
                    "type": "agentMessage",
                    "phase": "final_answer",
                    "text": f"The tag {REQUEST} is only in this reply.",
                }
            ],
        }
        for index in range(5)
    ]
    older = {
        "id": "turn-older",
        "status": "completed",
        "items": [
            {
                "type": "userMessage",
                "content": [{"type": "text", "text": REQUEST}],
            },
            {"type": "agentMessage", "phase": "final_answer", "text": "older reply"},
        ],
    }
    transport = Scripted(
        [
            _send_ok(),
            _thread_page(decoys, cursor="older-page", has_more=True),
            _thread_page([older], cursor=None, has_more=False),
        ]
    )
    client = CodexDesktopClient(
        transport,
        AttemptJournal(tmp_path / "attempts.json"),
        sleeper=lambda _s: None,
        clock=lambda: 0,
    )
    result = client.send(_request())
    assert result.status.value == "success"
    assert result.turn_id == "turn-older"
    assert result.reply == "older reply"
    reads = [
        params for _method, params in transport.calls if params["tool"] == "read_thread"
    ]
    assert "cursor" not in reads[0]["arguments"]
    assert reads[1]["arguments"]["cursor"] == "older-page"
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1


def test_failed_read_is_recovered_without_another_send(tmp_path: Path) -> None:
    failed = PipeCall(
        response={
            "id": "2",
            "jsonrpc": "2.0",
            "result": {"success": False, "contentItems": []},
        },
        focus=_focus(),
    )
    transport = Scripted([_send_ok(), failed, failed])
    client = _fast_client(tmp_path, transport)
    result = client.send(_request())
    assert result.status.value == "uncertain"
    assert result.reason == "completion_uncertain"
    assert result.sent is True
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1
    assert [params["tool"] for _method, params in transport.calls].count(
        "read_thread"
    ) == 2


def test_a_page_without_a_cursor_is_not_attributed(tmp_path: Path) -> None:
    transport = Scripted([_send_ok(), _thread_page([], cursor=None, has_more=True)])
    client = CodexDesktopClient(
        transport,
        AttemptJournal(tmp_path / "attempts.json"),
        sleeper=lambda _s: None,
        clock=lambda: 0,
    )
    result = client.send(_request())
    assert result.status.value == "error"
    assert result.reason == "ambiguous_turn_attribution"
    assert result.sent is True
    assert [params["tool"] for _method, params in transport.calls].count(
        "send_message_to_thread"
    ) == 1


def test_response_id_must_match_the_request() -> None:
    body = json.dumps({"jsonrpc": "2.0", "id": "mine", "result": {}}).encode()
    assert parse_response_bytes(body, expected_id="mine")["id"] == "mine"
    for raw in (
        json.dumps({"jsonrpc": "2.0", "id": "other", "result": {}}).encode(),
        json.dumps({"jsonrpc": "2.0", "result": {}}).encode(),
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}).encode(),
        json.dumps({"jsonrpc": "2.0", "id": None, "result": {}}).encode(),
    ):
        with pytest.raises(FrameError, match="response id"):
            parse_response_bytes(raw, expected_id="mine")


def test_manifest_hash_is_required_before_dispatch(tmp_path: Path) -> None:
    task, root = _task(tmp_path)
    plan = plan_codex_challenge(
        task, thread_id=THREAD, workspace=root, transport_request_id=REQUEST
    )
    record = json.loads(
        (task / "input" / "crops" / "crop_page.json").read_text(encoding="utf-8")
    )
    record["crop_sha256"] = "ab" * 32
    (task / "input" / "crops" / "crop_page.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(ReviewIntegrityError, match="task manifest"):
        challenge_desktop_request(plan)


def test_transport_rejects_a_mismatched_response_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, input, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        request = json.loads(input)
        sent = json.loads(base64.b64decode(request["frameBase64"]))
        assert isinstance(sent["id"], str) and sent["id"]
        raw = json.dumps(
            {"jsonrpc": "2.0", "id": "not-the-request", "result": {}}
        ).encode()
        focus = {
            "before": {"hwnd": 1, "pid": 2, "executable": r"C:\cursor\Cursor.exe"},
            "after": {"hwnd": 1, "pid": 2, "executable": r"C:\cursor\Cursor.exe"},
        }
        stdout = json.dumps(
            {
                "ok": True,
                "responseBase64": base64.b64encode(raw).decode("ascii"),
                "focus": focus,
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        "transcription_review_agent.codex_desktop.pipe.subprocess.run", fake_run
    )
    with pytest.raises(FrameError, match="response id"):
        PowershellCodexTransport().call(
            "tools/call", {"tool": "read_thread"}, timeout_s=1
        )
