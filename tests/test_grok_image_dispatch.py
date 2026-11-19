"""Image attachment is prepared and tested without launching Grok."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from transcription_review_agent.grok_desktop.adapter import GrokDesktopAdapter
from transcription_review_agent.grok_desktop.__main__ import main
from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)
from transcription_review_agent.grok_desktop.registry import ResourceRegistry
from transcription_review_agent.grok_desktop.session import (
    run_attachment_preflight,
    run_exchange,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewUsageError,
)
from transcription_review_agent.grok_desktop.image_dispatch import (
    challenge_review_status,
    plan_grok_image_dispatch,
    preserve_grok_reply,
)
from test_grok_desktop import AGENT, NAME, REQUEST_ID, Clock, FakeSession, success_poll

SHA = "bd6795147000693795d2c0ca1993521c48970d615c68a91a467ac01a29ae4a59"


class ImageSession(FakeSession):
    def __init__(
        self,
        *args: object,
        show_chip: bool = True,
        composer_reads: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.file_sets: list[tuple[str, tuple[str, ...]]] = []
        self.show_chip = show_chip
        self.composer_reads = list(composer_reads or [])
        self.composer_observations: list[dict[str, object]] = []

    def evaluate(self, expression: str) -> object:
        if "grok-desktop:file-input" in expression:
            return {
                "ok": True,
                "count": 1,
                "multiple": True,
                "accept": "",
                "disabled": False,
            }
        if "grok-desktop:composer-files" in expression:
            if self.composer_reads:
                observed = self.composer_reads.pop(0)
            elif not self.file_sets or not self.show_chip:
                observed = {
                    "ok": True,
                    "fileCount": 0,
                    "files": [],
                    "chips": [],
                    "labels": [],
                }
            else:
                # The live app clears the input and leaves image chip text empty.
                observed = {
                    "ok": True,
                    "fileCount": 0,
                    "files": [],
                    "chips": [""],
                    "labels": ["page.png"],
                }
            self.composer_observations.append(observed)
            return observed
        return super().evaluate(expression)

    def set_input_files(self, selector: str, files: tuple[str, ...]) -> None:
        self.file_sets.append((selector, files))


def _request(path: Path) -> dict[str, object]:
    return {
        "schema_version": "review-work-request-v1",
        "provider": "grok",
        "provider_interface": "GROK",
        "task_id": "batch-000903.1.challenge.r1.grok.a1".replace(".", ":"),
        "job_id": "job:batch-000903:1:challenge:r1:grok:a1",
        "batch_id": "batch-000903",
        "document_id": 1,
        "stage": "challenge",
        "round_index": 1,
        "attempt": 1,
        "source_bundle_sha256": SHA,
        "dispatch": "manual",
        "unattended_api": False,
        "output_relative_path": "challenge_round_01/output/challenge_grok.json",
        "page_ids": ["page_001"],
        "forbidden_inputs": ["archive_transcription_of_this_document"],
    }


def _task(tmp_path: Path, *, prompt: str, stage: str = "challenge") -> Path:
    task = tmp_path / "grok"
    (task / "input" / "crops").mkdir(parents=True)
    (task / "input" / "page_001.png").write_bytes(b"png-page")
    (task / "input" / "crops" / "crop_dsp.png").write_bytes(b"png-crop")
    payload = _request(task)
    payload["stage"] = stage
    if stage == "first_pass":
        payload["round_index"] = 0
        payload["task_id"] = "batch-000903:1:first_pass:r0:grok:a1"
        payload["job_id"] = "job:batch-000903:1:first_pass:r0:grok:a1"
        payload["output_relative_path"] = "first_pass/output/transcription_grok.json"
        (task / "original.png").write_bytes(b"png-original")
    (task / "work_request.json").write_text(json.dumps(payload), encoding="utf-8")
    (task / "prompt.txt").write_text(prompt, encoding="utf-8")
    return task


def _exchange(image: Path) -> ExchangeRequest:
    return ExchangeRequest(
        agent_id=AGENT,
        bot_name=NAME,
        message=f"Read the attached image.\nrequest_id: {REQUEST_ID}\n",
        request_id=REQUEST_ID,
        attachments=(str(image),),
    )


def test_short_prompt_keeps_full_text_and_attaches_images(tmp_path: Path) -> None:
    prompt = "Transcribe the sealed image.\nDo not use an archive transcription.\n"
    plan = plan_grok_image_dispatch(_task(tmp_path, prompt=prompt))
    assert prompt in plan.message
    assert plan.request_id in plan.message
    assert plan.prompt_attached is False
    assert [path.name for path in plan.attachments] == ["page_001.png", "crop_dsp.png"]
    assert "png-page" not in plan.message


def test_long_prompt_is_attached_whole_and_not_truncated(tmp_path: Path) -> None:
    prompt = "INSTRUCTION " + ("word " * 800)
    plan = plan_grok_image_dispatch(_task(tmp_path, prompt=prompt, stage="first_pass"))
    assert len(plan.message) <= 2000
    assert prompt not in plan.message
    assert prompt[:40] not in plan.message
    assert plan.prompt_attached is True
    prompt_file = next(path for path in plan.attachments if path.name == "prompt.txt")
    assert prompt_file.read_text(encoding="utf-8") == prompt
    assert any(path.name == "original.png" for path in plan.attachments)
    assert plan.stage == "first_pass"


def test_other_provider_text_is_not_read_into_the_message(tmp_path: Path) -> None:
    task = _task(tmp_path, prompt="Read only this grok prompt.\n")
    sibling = tmp_path / "gpt" / "transcription.txt"
    sibling.parent.mkdir()
    sibling.write_text("WEBSITE TRANSCRIPTION LEAK", encoding="utf-8")
    plan = plan_grok_image_dispatch(task)
    assert "WEBSITE TRANSCRIPTION LEAK" not in plan.message
    assert all("gpt" not in path.parts for path in plan.attachments)


def test_gpt_task_is_refused(tmp_path: Path) -> None:
    task = _task(tmp_path, prompt="prompt\n")
    payload = json.loads((task / "work_request.json").read_text(encoding="utf-8"))
    payload["provider"] = "gpt"
    payload["provider_interface"] = "GPT_WORK"
    (task / "work_request.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReviewUsageError):
        plan_grok_image_dispatch(task)


def test_cleared_input_and_blank_chip_text_confirm_by_aria_label(
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession(
        polls=[success_poll("raw model json")],
        composer_reads=[
            {"ok": True, "fileCount": 0, "files": [], "chips": [], "labels": []},
            {
                "ok": True,
                "fileCount": 0,
                "files": [],
                "chips": [""],
                "labels": ["page.png"],
            },
        ],
    )
    result = run_exchange(session, _exchange(image), Clock())
    assert result.status is ExchangeStatus.SUCCESS
    assert result.sent is True
    assert len(session.file_sets) == 1
    assert session.inserts == [_exchange(image).message]


def test_filename_in_the_input_or_chip_text_is_not_acceptance(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    occupied = {
        "ok": True,
        "fileCount": 1,
        "files": ["page.png"],
        "chips": ["page.png"],
        "labels": [""],
    }
    session = ImageSession(
        composer_reads=[
            {"ok": True, "fileCount": 0, "files": [], "chips": [], "labels": []},
            *([occupied] * 20),
        ]
    )
    result = run_exchange(session, _exchange(image), Clock())
    assert result.status is ExchangeStatus.ERROR
    assert result.sent is False
    assert result.reason == "attachment_not_in_composer"
    assert session.inserts == []
    assert len(session.file_sets) == 1


def test_existing_same_filename_is_refused_before_assignment(
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    present = {
        "ok": True,
        "fileCount": 0,
        "files": [],
        "chips": [""],
        "labels": ["page.png"],
    }
    exchange = ImageSession(composer_reads=[present])
    result = run_exchange(exchange, _exchange(image), Clock())
    assert result.status is ExchangeStatus.ERROR
    assert result.sent is False
    assert result.reason == "attachment_already_present"
    assert exchange.file_sets == []
    assert exchange.inserts == []

    preflight = ImageSession(composer_reads=[dict(present)])
    checked = run_attachment_preflight(
        preflight, (str(image),), agent_id=AGENT, bot_name=NAME, clock=Clock()
    )
    assert checked.ready is False
    assert checked.submitted is False
    assert checked.reason == "attachment_already_present"
    assert checked.labels == ("page.png",)
    assert preflight.file_sets == []
    assert preflight.inserts == []


def test_pending_file_selection_is_refused_before_assignment(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession(
        composer_reads=[
            {
                "ok": True,
                "fileCount": 1,
                "files": ["page.png"],
                "chips": [],
                "labels": [],
            }
        ]
    )
    result = run_exchange(session, _exchange(image), Clock())
    assert result.status is ExchangeStatus.ERROR
    assert result.sent is False
    assert result.reason == "attachment_already_present"
    assert session.file_sets == []
    assert session.inserts == []


def test_missing_chip_does_not_insert_or_submit(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession(show_chip=False)
    result = run_exchange(session, _exchange(image), Clock())
    assert result.status is ExchangeStatus.ERROR
    assert result.sent is False
    assert result.reason == "attachment_not_in_composer"
    assert session.inserts == []
    assert session.file_sets == [("input.sand-prompt-file-input", (str(image),))]


def test_visible_chip_submits_once_and_keeps_the_raw_reply(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession(polls=[success_poll("raw model json")])
    result = run_exchange(session, _exchange(image), Clock())
    assert result.status is ExchangeStatus.SUCCESS
    assert result.raw_reply == "raw model json"
    assert session.inserts == [_exchange(image).message]
    assert len(session.file_sets) == 1


def test_timeout_after_submit_stays_uncertain(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession(polls=[])
    result = run_exchange(session, _exchange(image), Clock())
    assert result.status is ExchangeStatus.UNCERTAIN
    assert result.sent is True
    assert len(session.file_sets) == 1


def test_preflight_selects_the_bot_and_does_not_insert(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession()
    result = run_attachment_preflight(
        session, (str(image),), agent_id=AGENT, bot_name=NAME
    )
    assert result.ready is True
    assert result.submitted is False
    assert result.files == ()
    assert result.chips == ("",)
    assert result.labels == ("page.png",)
    assert session.inserts == []
    wrong = ImageSession(preflight={"ok": False, "active": True, "nameMatch": False})
    refused = run_attachment_preflight(
        wrong, (str(image),), agent_id=AGENT, bot_name=NAME
    )
    assert refused.ready is False
    assert refused.reason == "wrong_bot"
    assert wrong.file_sets == []


def test_same_request_is_not_sent_twice(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession(polls=[success_poll("raw model json")])
    registry = ResourceRegistry(tmp_path / "registry.json")
    adapter = GrokDesktopAdapter(
        registry=registry,
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: session,
        check_profile=lambda: None,
        clock=Clock(),
    )
    first = adapter.exchange(_exchange(image))
    assert first.status is ExchangeStatus.SUCCESS

    def refuse() -> ImageSession:
        raise AssertionError("replay must not launch")

    replay = GrokDesktopAdapter(
        registry=ResourceRegistry(tmp_path / "registry.json"),
        diagnostics_dir=tmp_path / "diag2",
        open_session=refuse,
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(_exchange(image))
    assert replay.replayed is True
    assert replay.raw_reply == "raw model json"
    assert len(session.file_sets) == 1


def test_invalid_reply_is_preserved_and_not_promoted(tmp_path: Path) -> None:
    task = _task(tmp_path, prompt="Transcribe.\n")
    saved = preserve_grok_reply(task, "not json", tmp_path / "raw")
    assert saved.accepted is False
    assert saved.raw_path.read_text(encoding="utf-8") == "not json"
    assert saved.output_path is None
    assert not (task / "output" / "challenge_grok.json").exists()


def test_valid_challenge_reply_is_stored_unchanged(tmp_path: Path) -> None:
    task = _task(tmp_path, prompt="Transcribe.\n")
    reply = json.dumps(
        {
            "schema_version": "review-provider-output-v1",
            "provider": "grok",
            "provider_interface": "GROK",
            "task_id": "batch-000903:1:challenge:r1:grok:a1",
            "job_id": "job:batch-000903:1:challenge:r1:grok:a1",
            "batch_id": "batch-000903",
            "document_id": 1,
            "attempt": 1,
            "source_bundle_sha256": SHA,
            "stage": "challenge",
            "round_index": 1,
            "decisions": [
                {
                    "dispute_id": "dsp_23f02d5e3014d3f0",
                    "decision": "UNRESOLVED",
                    "visual_evidence": "The crop does not settle the numeral.",
                }
            ],
        }
    )
    saved = preserve_grok_reply(task, reply, tmp_path / "raw")
    assert saved.accepted is True
    assert saved.output_path is not None
    assert saved.output_path.read_text(encoding="utf-8") == reply
    assert saved.raw_path.read_bytes() == reply.encode("utf-8")
    again = preserve_grok_reply(task, reply, tmp_path / "raw")
    assert again.accepted is True
    with pytest.raises(ReviewIntegrityError):
        preserve_grok_reply(task, reply + " ", tmp_path / "raw")


def test_live_flags_refuse_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROK_ISO_CONFIRM", "1")
    monkeypatch.delenv("GROK_ATTACH_PREFLIGHT", raising=False)
    monkeypatch.delenv("GROK_IMAGE_SUBMIT", raising=False)
    common = [
        "--agent-id",
        AGENT,
        "--bot-name",
        NAME,
        "--request-id",
        REQUEST_ID,
        "--registry",
        str(tmp_path / "registry.json"),
        "--diagnostics",
        str(tmp_path / "diag"),
        "--log",
        str(tmp_path / "host.log"),
        "--image",
        str(tmp_path / "page.png"),
    ]
    assert main([*common, "--preflight-attachment"]) == 2
    assert main([*common, "--message", "request_id: " + REQUEST_ID]) == 2


_RULES = (
    "Inspect the page image and the crop window. The crop is existing pixels,\n"
    "not new detail.\n"
    "Candidate readings are anonymous.\n"
    "Choose CANDIDATE_X, CANDIDATE_Y, THIRD_READING, or UNRESOLVED.\n"
    "Cite visual evidence. Do not change a reading only to force agreement.\n"
    "Write one JSON file. Dispatch is manual. No API is assumed.\n"
    "Copy schema_version and stage from\n"
    "work_request.json. output_example.json already contains those values.\n"
    "output_example.json is a shape template. Replace its template text with\n"
    "your own reading. Do not submit the template unchanged.\n"
)

_DISPUTES = {
    "schema_version": "review-challenge-input-v3",
    "disputes": [
        {
            "dispute_id": "dsp_23f02d5e3014d3f0",
            "page_id": "page_001",
            "candidate_x": "BLUE HARBOUR OPENS AT 12",
            "candidate_y": "BLUE HARBOR OPENS AT 42",
        }
    ],
    "required_decision_ids": ["dsp_23f02d5e3014d3f0"],
}

_CROP = {
    "crop_id": "crop_dsp_23f02d5e3014d3f0",
    "window_kind": "TARGETED",
    "represents_new_visual_detail": False,
}

_SCHEMA = {
    "note": (
        "Return one JSON object and no other fields. "
        "Copy every identity field from work_request.json. "
        "output_example.json is a shape template."
    ),
    "decisions": {"dispute_id": "exact required decision ID"},
}

_EXAMPLE = {
    "schema_version": "review-provider-output-v1",
    "provider": "grok",
    "stage": "challenge",
    "decisions": [
        {
            "dispute_id": "dsp_23f02d5e3014d3f0",
            "decision": "UNRESOLVED",
            "visual_evidence": "TEMPLATE ONLY. Replace this sentence.",
        }
    ],
}


def _published(tmp_path: Path, *, instructions: str, prompt: str) -> Path:
    task = _task(tmp_path, prompt=prompt)
    (task / "instructions_grok.txt").write_text(instructions, encoding="utf-8")
    (task / "input" / "disputes.json").write_text(
        json.dumps(_DISPUTES), encoding="utf-8"
    )
    (task / "input" / "crops" / "crop_dsp.json").write_text(
        json.dumps(_CROP), encoding="utf-8"
    )
    (task / "output_schema.json").write_text(json.dumps(_SCHEMA), encoding="utf-8")
    (task / "output_example.json").write_text(json.dumps(_EXAMPLE), encoding="utf-8")
    return task


def test_chat_challenge_keeps_rules_and_drops_filesystem_instructions(
    tmp_path: Path,
) -> None:
    prompt = (
        "Read C:\\task\\instructions_grok.txt\n"
        "Write one JSON file and no other path:\n"
        "C:\\task\\output\\challenge_grok.json\n"
    )
    task = _published(tmp_path, instructions=_RULES, prompt=prompt)
    before = (task / "prompt.txt").read_bytes()
    plan = plan_grok_image_dispatch(
        task,
        packet_dir=tmp_path / "packet",
        transport_request_id="batch-000903.1.challenge.r1.grok.a2",
    )
    assert (task / "prompt.txt").read_bytes() == before
    assert not (task / "challenge_packet.txt").exists()
    assert plan.prompt_attached is False
    facing = plan.message
    packet = ""
    if plan.packet_attached:
        packet_path = next(
            path for path in plan.attachments if path.name == "challenge_packet.txt"
        )
        packet = packet_path.read_text(encoding="utf-8")
        assert hashlib.sha256(packet.encode("utf-8")).hexdigest() == plan.packet_sha256
        assert plan.packet_sha256 in plan.message
        assert "challenge_packet.txt" in plan.message
        facing = plan.message + "\n" + packet
    assert "C:\\task" not in facing
    assert "Write one JSON file" not in facing
    assert "work_request.json" not in facing
    assert "shell" not in facing.lower()
    assert "Candidate readings are anonymous." in facing
    assert "Cite visual evidence." in facing
    assert "Do not change a reading only to force agreement." in facing
    assert "BLUE HARBOUR OPENS AT 12" in facing
    assert "BLUE HARBOR OPENS AT 42" in facing
    assert "dsp_23f02d5e3014d3f0" in facing
    assert "TARGETED" in facing
    assert "batch-000903:1:challenge:r1:grok:a1" in facing
    assert "bd6795147000693795d2c0ca1993521c48970d615c68a91a467ac01a29ae4a59" in facing
    assert plan.request_id in plan.message
    assert [path.name for path in plan.attachments if path.suffix == ".png"] == [
        "page_001.png",
        "crop_dsp.png",
    ]
    assert "WEBSITE" not in facing


def test_large_challenge_packet_is_attached_outside_the_task(tmp_path: Path) -> None:
    task = _published(tmp_path, instructions=_RULES * 8, prompt="Read C:\\secret\\prompt.txt\n")
    prompt_bytes = (task / "prompt.txt").read_bytes()
    with pytest.raises(ReviewUsageError, match="already submitted"):
        plan_grok_image_dispatch(task)
    with pytest.raises(ReviewUsageError, match="no packet directory"):
        plan_grok_image_dispatch(
            task,
            transport_request_id="batch-000903.1.challenge.r1.grok.a2",
        )
    with pytest.raises(ReviewUsageError, match="already submitted"):
        plan_grok_image_dispatch(
            task,
            packet_dir=tmp_path / "outside",
            transport_request_id="batch-000903.1.challenge.r1.grok.a1",
        )
    with pytest.raises(ReviewIntegrityError):
        plan_grok_image_dispatch(
            task,
            packet_dir=task / "packet",
            transport_request_id="batch-000903.1.challenge.r1.grok.a2",
        )
    plan = plan_grok_image_dispatch(
        task,
        packet_dir=tmp_path / "outside",
        transport_request_id="batch-000903.1.challenge.r1.grok.a2",
    )
    assert (task / "prompt.txt").read_bytes() == prompt_bytes
    assert plan.packet_attached is True
    assert plan.prompt_attached is False
    assert "prompt.txt" not in plan.message
    assert "C:\\secret" not in plan.message
    assert len(plan.message) <= 2000
    packet = (tmp_path / "outside" / "challenge_packet.txt").read_text(encoding="utf-8")
    assert packet.count("Candidate readings are anonymous.") == 8
    assert "Read C:\\secret\\prompt.txt" not in packet


def test_other_provider_file_stays_out_of_the_chat_packet(tmp_path: Path) -> None:
    task = _published(tmp_path, instructions=_RULES, prompt="grok only\n")
    sibling = tmp_path / "gpt" / "transcription.txt"
    sibling.parent.mkdir()
    sibling.write_text("WEBSITE TRANSCRIPTION LEAK", encoding="utf-8")
    plan = plan_grok_image_dispatch(
        task,
        packet_dir=tmp_path / "packet",
        transport_request_id="batch-000903.1.challenge.r1.grok.a2",
    )
    visible = plan.message
    for path in plan.attachments:
        if path.suffix == ".txt":
            visible += path.read_text(encoding="utf-8")
    assert "WEBSITE TRANSCRIPTION LEAK" not in visible
    assert all("gpt" not in path.parts for path in plan.attachments)


def test_prose_reply_is_transport_success_and_not_review_success(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, prompt="Transcribe.\n")
    prose = (
        "On it. I'm checking that challenge_queue.json exists first, "
        "then I'll read the instructions, disputes, and crops."
    )
    saved = preserve_grok_reply(task, prose, tmp_path / "raw")
    status = challenge_review_status(
        ExchangeResult(
            status=ExchangeStatus.SUCCESS,
            sent=True,
            request_id=REQUEST_ID,
            agent_id=AGENT,
            reply=prose,
            raw_reply=prose,
        ),
        saved,
    )
    assert status == "transport_only"
    assert saved.accepted is False
    assert saved.raw_path.read_text(encoding="utf-8") == prose
    assert not (task / "output" / "challenge_grok.json").exists()


def test_valid_challenge_json_is_review_acceptance(tmp_path: Path) -> None:
    task = _task(tmp_path, prompt="Transcribe.\n")
    reply = json.dumps(
        {
            "schema_version": "review-provider-output-v1",
            "provider": "grok",
            "provider_interface": "GROK",
            "task_id": "batch-000903:1:challenge:r1:grok:a1",
            "job_id": "job:batch-000903:1:challenge:r1:grok:a1",
            "batch_id": "batch-000903",
            "document_id": 1,
            "attempt": 1,
            "source_bundle_sha256": SHA,
            "stage": "challenge",
            "round_index": 1,
            "decisions": [
                {
                    "dispute_id": "dsp_23f02d5e3014d3f0",
                    "decision": "CANDIDATE_Y",
                    "third_reading": None,
                    "visual_evidence": "The crop shows HARBOR and 42.",
                    "evidence_ref": {
                        "kind": "CROP",
                        "evidence_id": "crop_dsp_23f02d5e3014d3f0",
                        "page_id": "page_001",
                    },
                    "evidence_assessment": "SUPPORTS_DECISION",
                }
            ],
        }
    )
    saved = preserve_grok_reply(task, reply, tmp_path / "raw")
    assert challenge_review_status(
        ExchangeResult(
            status=ExchangeStatus.SUCCESS,
            sent=True,
            request_id="batch-000903.1.challenge.r1.grok.a1",
            agent_id=AGENT,
            raw_reply=reply,
        ),
        saved,
    ) == "accepted"
    assert saved.output_path is not None
    assert saved.output_path.read_bytes() == reply.encode("utf-8")


def test_submitted_request_is_not_sent_again_after_invalid_review(
    tmp_path: Path,
) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    session = ImageSession(polls=[success_poll("not json")])
    registry = tmp_path / "registry.json"
    adapter = GrokDesktopAdapter(
        registry=ResourceRegistry(registry),
        diagnostics_dir=tmp_path / "diag",
        open_session=lambda: session,
        check_profile=lambda: None,
        clock=Clock(),
    )
    first = adapter.exchange(_exchange(image))
    assert first.status is ExchangeStatus.SUCCESS
    assert first.sent is True
    task = _task(tmp_path / "task", prompt="Transcribe.\n")
    saved = preserve_grok_reply(task, first.raw_reply, tmp_path / "raw")
    assert challenge_review_status(first, saved) == "transport_only"

    def refuse() -> ImageSession:
        raise AssertionError("an accepted send must not be submitted again")

    replay = GrokDesktopAdapter(
        registry=ResourceRegistry(registry),
        diagnostics_dir=tmp_path / "diag2",
        open_session=refuse,
        check_profile=lambda: None,
        clock=Clock(),
    ).exchange(_exchange(image))
    assert replay.replayed is True
    assert replay.sent is True
    assert len(session.file_sets) == 1


def test_host_script_sets_files_without_a_tcp_port() -> None:
    script = Path(
        "src/transcription_review_agent/grok_desktop/assets/pipe-host.ps1"
    ).read_text(encoding="utf-8")
    assert "DOM.setFileInputFiles" in script
    assert "input.sand-prompt-file-input" in script
    assert "remote-debugging-port" not in script
