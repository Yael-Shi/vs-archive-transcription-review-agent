"""Whether a review task can be dispatched to a desktop worker.

Publishing a task folder is not dispatch. This module does not launch an
application, open a pipe, or send a message. One completed live challenge is
not a repeatable adapter run and is not unattended dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transcription_review_agent.review.errors import ReviewUsageError

ProviderName = Literal["gpt", "grok"]
EvidenceClass = Literal[
    "proven",
    "documented",
    "live_test_required",
    "unsupported",
]

_GROK_REASON = (
    "One authorized Grok Desktop turn submitted an image-grounded challenge and "
    "the controller accepted that file. That is one completed execution. It does "
    "not show that composer attachment through .sand-prompt-attachment is a "
    "repeatable adapter, and it does not open unattended dispatch. Local "
    "execution stays never and the normal profile stays untouched."
)
_CODEX_REASON = (
    "One authorized Codex Desktop turn read a sealed page through the "
    "codex-app-tools pipe and wrote a challenge file. A later combined send "
    "and read_thread completed one short turn on that adapter. Each is one "
    "completed execution. This is not unattended dispatch: the controller "
    "does not send Desktop turns. The Codex CLI, SDK, and app-server are "
    "not used."
)


@dataclass(frozen=True, slots=True)
class ExecutionAssessment:
    """One provider's ability to run one image-grounded review stage."""

    provider: ProviderName
    stage: str
    executable: bool
    evidence_class: EvidenceClass
    mechanism: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_class": self.evidence_class,
            "executable": self.executable,
            "mechanism": self.mechanism,
            "provider": self.provider,
            "reason": self.reason,
            "stage": self.stage,
        }


_AUTOMATIC_REASON = {
    "gpt": (
        "A Codex Desktop sender is configured for this run. The controller sends "
        "the task once through the codex-app-tools pipe, preserves the raw reply, "
        "and installs it only when it validates. This orchestration has not been "
        "run live. The Codex CLI, SDK, and app-server are not used."
    ),
    "grok": (
        "A Grok Desktop sender is configured for this run. The controller sends "
        "the task once through the isolated profile, with the sealed page "
        "attached and the instructions in the chat. Local execution stays never. "
        "This orchestration has not been run live."
    ),
}


def assess_review_dispatch(
    provider: str, stage: str, *, automatic: bool = False
) -> ExecutionAssessment:
    """Return the dispatch boundary. This call has no side effects.

    ``automatic`` states that the caller configured a sender for this provider.
    It does not launch anything and does not show that a live run succeeded.
    """

    if provider not in {"gpt", "grok"}:
        raise ReviewUsageError("provider must be gpt or grok")
    if stage not in {"first_pass", "challenge"}:
        raise ReviewUsageError("desktop dispatch covers first_pass and challenge")
    if automatic:
        return ExecutionAssessment(
            provider=provider,  # type: ignore[arg-type]
            stage=stage,
            executable=True,
            evidence_class="live_test_required",
            mechanism=(
                "codex_app_tools_pipe"
                if provider == "gpt"
                else "grok_composer_file_input"
            ),
            reason=_AUTOMATIC_REASON[provider],
        )
    if provider == "grok":
        return ExecutionAssessment(
            provider="grok",
            stage=stage,
            executable=False,
            evidence_class="documented",
            mechanism="grok_composer_file_input",
            reason=_GROK_REASON,
        )
    return ExecutionAssessment(
        provider="gpt",
        stage=stage,
        executable=False,
        evidence_class="documented",
        mechanism="codex_app_tools_pipe",
        reason=_CODEX_REASON,
    )


def review_dispatch_blocked(
    stage: str,
) -> tuple[ExecutionAssessment, ExecutionAssessment]:
    """Assess both workers for one image-grounded stage."""

    return (
        assess_review_dispatch("gpt", stage),
        assess_review_dispatch("grok", stage),
    )
