"""Provider senders for automatic desktop dispatch.

Each sender adapts one published task to its own interface and makes the one
send through an existing adapter. Neither sender opens an application by
itself, repairs a reply, or writes a review file. Both are handed an already
constructed adapter, so tests use scripted transports.

Codex Desktop reads the published task by absolute path and replies in the
thread. The reply, not a file the model writes, is the result. Grok Desktop's
isolated profile has no local execution, so its instructions travel in the
chat and the sealed images are the attachments.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from transcription_review_agent.codex_desktop.client import CodexDesktopClient
from transcription_review_agent.codex_desktop.models import (
    CodexRequest,
    CodexResult,
    CodexStatus,
)
from transcription_review_agent.grok_desktop.adapter import GrokDesktopAdapter
from transcription_review_agent.grok_desktop.image_dispatch import (
    plan_grok_first_pass_chat,
    plan_grok_image_dispatch,
)
from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)
from transcription_review_agent.grok_desktop.transport import TransportError
from transcription_review_agent.review.desktop_dispatch import (
    DispatchReply,
    DispatchTask,
    PreparedSend,
    identity_values,
    reply_delivery_prompt,
)
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPolicyError,
)

CODEX_MECHANISM = "codex_app_tools_pipe"
GROK_MECHANISM = "grok_composer_file_input"

AttachmentPath = Callable[[DispatchTask, Path], str]


class CodexDesktopSender:
    """Send one GPT task to an existing Codex Desktop thread."""

    mechanism = CODEX_MECHANISM

    def __init__(
        self,
        client: CodexDesktopClient,
        *,
        thread_id: str,
        workspace: str | None = None,
    ) -> None:
        self._client = client
        self._thread_id = thread_id
        self._workspace = workspace

    def prepare(self, task: DispatchTask) -> PreparedSend:
        if task.provider != "gpt":
            raise ReviewPolicyError("the Codex sender handles only gpt tasks")
        if self._workspace is not None and not _within(
            self._workspace, task.windows_task_dir
        ):
            raise ReviewIntegrityError(
                "the published task is outside the Codex thread workspace"
            )
        prompt = (task.task_dir / "prompt.txt").read_text(encoding="utf-8")
        identity = json.dumps(identity_values(task.request), indent=2, sort_keys=True)
        message = (
            f"{reply_delivery_prompt(prompt, task.stage)}"
            f"This task directory is {task.windows_task_dir}. Every file named "
            "above is under it. Open files by absolute path from there only.\n"
            f"request_id: {task.request_id}\n"
            f"Identity values, copy exactly:\n{identity}\n"
            "Reply with exactly one JSON object as the final answer. No "
            "preamble, no Markdown code fence, no other text.\n"
            "Delivery override: where any task file says to write one JSON "
            "file, return that JSON object as the final answer instead. Do not "
            "write any file.\n"
        )
        return PreparedSend(
            request_id=task.request_id,
            message=message,
            attachments=(),
            mechanism=self.mechanism,
        )

    def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
        result = self._client.send(
            CodexRequest(
                thread_id=self._thread_id,
                request_id=prepared.request_id,
                message=prepared.message,
                image_sha256=task.evidence_sha256,
            )
        )
        return _codex_reply(result, self._thread_id)


class GrokDesktopSender:
    """Send one Grok task to the isolated Grok Desktop profile."""

    mechanism = GROK_MECHANISM

    def __init__(
        self,
        adapter: GrokDesktopAdapter,
        *,
        agent_id: str,
        bot_name: str,
        attachment_path: AttachmentPath | None = None,
    ) -> None:
        self._adapter = adapter
        self._agent_id = agent_id
        self._bot_name = bot_name
        self._attachment_path = attachment_path or windows_attachment

    def prepare(self, task: DispatchTask) -> PreparedSend:
        if task.provider != "grok":
            raise ReviewPolicyError("the Grok sender handles only grok tasks")
        if task.stage == "challenge":
            if not (task.task_dir / "input" / "disputes.json").is_file():
                raise ReviewIntegrityError("the Grok challenge task has no disputes")
            plan = plan_grok_image_dispatch(task.task_dir, packet_dir=task.packet_dir)
        else:
            plan = plan_grok_first_pass_chat(task.task_dir, packet_dir=task.packet_dir)
        return PreparedSend(
            request_id=plan.request_id,
            message=plan.message,
            attachments=plan.attachments,
            mechanism=self.mechanism,
        )

    def send(self, task: DispatchTask, prepared: PreparedSend) -> DispatchReply:
        request = ExchangeRequest(
            agent_id=self._agent_id,
            bot_name=self._bot_name,
            message=prepared.message,
            request_id=prepared.request_id,
            source_document_id=f"{task.request.batch_id}.{task.request.document_id}",
            attachments=tuple(
                self._attachment_path(task, path) for path in prepared.attachments
            ),
        )
        try:
            result = self._adapter.exchange(request)
        except TransportError as exc:
            return DispatchReply(
                status="uncertain",
                raw_reply="",
                reason=str(exc),
                transport={"interface": GROK_MECHANISM, "agent_id": self._agent_id},
            )
        return _grok_reply(result)


def windows_attachment(task: DispatchTask, path: Path) -> str:
    """The Windows path of a file under this task or its packet folder."""

    resolved = path.resolve()
    for root, windows in (
        (task.task_dir.resolve(), task.windows_task_dir),
        (task.packet_dir.resolve(), task.windows_packet_dir),
    ):
        if root == resolved.parent or root in resolved.parents:
            relative = resolved.relative_to(root)
            return windows + "".join(f"\\{part}" for part in relative.parts)
    raise ReviewIntegrityError("attachment is outside this provider's task")


def _within(workspace: str, windows_path: str) -> bool:
    root = workspace.replace("/", "\\").rstrip("\\").casefold()
    return windows_path.casefold().startswith(root + "\\")


def _codex_reply(result: CodexResult, thread_id: str) -> DispatchReply:
    transport: dict[str, object] = {
        "interface": CODEX_MECHANISM,
        "thread_id": thread_id,
        "status": result.status.value,
        "turn_id": result.turn_id,
        "dispatch_turn_id": result.dispatch_turn_id,
        "focus_preserved": result.focus_preserved,
    }
    reason = result.reason or result.status.value
    if result.status is CodexStatus.SUCCESS:
        return DispatchReply(
            "reply", result.raw_reply or result.reply, reason, transport
        )
    if result.status is CodexStatus.BLOCKED:
        return DispatchReply("blocked", "", reason, transport)
    if not result.sent:
        return DispatchReply("not_sent", "", reason, transport)
    return DispatchReply("uncertain", "", reason, transport)


def _grok_reply(result: ExchangeResult) -> DispatchReply:
    transport: dict[str, object] = {
        "interface": GROK_MECHANISM,
        "agent_id": result.agent_id,
        "status": result.status.value,
        "cleanup": result.cleanup,
        "baseline": result.baseline,
    }
    reason = result.reason or result.status.value
    if result.status is ExchangeStatus.SUCCESS and result.sent:
        return DispatchReply(
            "reply", result.raw_reply or result.reply, reason, transport
        )
    if result.status is ExchangeStatus.BLOCKED and result.sent:
        return DispatchReply("blocked", "", reason, transport)
    if not result.sent:
        return DispatchReply("not_sent", "", reason, transport)
    return DispatchReply("uncertain", "", reason, transport)
