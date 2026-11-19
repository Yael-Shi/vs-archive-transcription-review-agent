"""The desktop execution gate refuses image-grounded dispatch without launching."""

from __future__ import annotations

import ast
from pathlib import Path

from transcription_review_agent.grok_desktop.models import ExchangeRequest
from transcription_review_agent.grok_desktop.profile import assert_isolated_profile
from transcription_review_agent.review.desktop_execution import (
    assess_review_dispatch,
    review_dispatch_blocked,
)


def test_image_grounded_review_tasks_are_not_executable() -> None:
    for stage in ("first_pass", "challenge"):
        gpt, grok = review_dispatch_blocked(stage)
        assert gpt.executable is False
        assert grok.executable is False
        assert gpt.evidence_class == "documented"
        assert grok.evidence_class == "documented"
        assert gpt.mechanism == "codex_app_tools_pipe"
        assert grok.mechanism == "grok_composer_file_input"
        assert "codex-app-tools" in gpt.reason
        assert "not unattended" in gpt.reason
        assert "one completed execution" in gpt.reason
        assert "sand-prompt-attachment" in grok.reason
        assert "unattended dispatch" in grok.reason
        assert assess_review_dispatch("gpt", stage) == gpt


def test_gate_does_not_launch_or_call_the_grok_adapter() -> None:
    source = Path(
        "src/transcription_review_agent/review/desktop_execution.py"
    ).read_text(encoding="utf-8")
    banned = {
        "subprocess",
        "live_adapter",
        "PowershellCdpSession",
        "exchange",
        "codex_image_transcription",
        "openai",
    }
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(alias.name for alias in node.names)]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names = [node.func.attr]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names = [node.func.id]
        else:
            continue
        assert not any(name in banned for name in names)


def test_existing_grok_contract_has_no_image_or_local_execution() -> None:
    fields = set(ExchangeRequest.__dataclass_fields__)
    assert "image" not in fields
    assert fields == {
        "agent_id",
        "attachments",
        "bot_name",
        "message",
        "request_id",
        "source_document_id",
    }
    profile = Path("src/transcription_review_agent/grok_desktop/profile.py").read_text(
        encoding="utf-8"
    )
    assert 'read_permission(pilot_settings) != "never"' in profile
    assert "assert_isolated_profile" in profile
    transcript = Path(
        "src/transcription_review_agent/grok_desktop/transcript.py"
    ).read_text(encoding="utf-8")
    assert "_MAX_MESSAGE = 2000" in transcript
    assert assert_isolated_profile.__doc__
