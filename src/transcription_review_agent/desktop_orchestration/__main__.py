"""One command: advance a sealed document through both desktop providers.

``python -m transcription_review_agent.desktop_orchestration run ...`` registers
the first-pass tasks, sends each missing task once, accepts valid replies,
compares, publishes and sends challenges, reconciles, and writes the pending
human-review page. It stops there. It has no closure, finalize, or publish
option.

Importing this module launches nothing. A run sends real turns, so it refuses
to start unless ``DESKTOP_DISPATCH_CONFIRM=1`` is set. Selecting Grok also
needs ``GROK_ISO_CONFIRM=1`` and ``GROK_IMAGE_SUBMIT=1``, the same gates the
manual Grok entry uses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from transcription_review_agent.codex_desktop.client import CodexDesktopClient
from transcription_review_agent.codex_desktop.journal import (
    AttemptJournal,
    JournalError,
)
from transcription_review_agent.codex_desktop.pipe import PowershellCodexTransport
from transcription_review_agent.desktop_orchestration.senders import (
    CodexDesktopSender,
    GrokDesktopSender,
)
from transcription_review_agent.grok_desktop.adapter import DesktopPaths, live_adapter
from transcription_review_agent.grok_desktop.profile import ProfileError
from transcription_review_agent.review.controller import ReviewController
from transcription_review_agent.review.desktop_dispatch import DesktopSender
from transcription_review_agent.review.document_run import (
    advance_document,
    run_payload,
)
from transcription_review_agent.review.errors import ReviewError

_DEFAULT_PROFILE = r"C:\Users\User\AppData\Local\Grok Bot Isolated Pilot"
_DEFAULT_EXE = r"C:\Users\User\AppData\Local\Programs\Grok Bot\Grok Bot.exe"
_DEFAULT_LIVE = r"C:\Users\User\.grokbot\settings.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="desktop-orchestration")
    parser.add_argument("operation", choices=("run",))
    parser.add_argument("--batches-dir", required=True, type=Path)
    parser.add_argument("--reviews-dir", required=True, type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--document-id", required=True, type=int)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--preservation-dir", required=True, type=Path)
    parser.add_argument("--packet-dir", required=True, type=Path)
    parser.add_argument("--mount-root", type=Path)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("gpt", "grok"),
        default=["gpt", "grok"],
    )
    parser.add_argument("--codex-thread-id")
    parser.add_argument("--codex-journal", type=Path)
    parser.add_argument("--codex-workspace")
    parser.add_argument("--codex-timeout-s", type=float, default=600.0)
    parser.add_argument("--grok-agent-id")
    parser.add_argument("--grok-bot-name")
    parser.add_argument("--grok-registry", type=Path)
    parser.add_argument("--grok-diagnostics", type=Path)
    parser.add_argument("--grok-log", type=Path)
    parser.add_argument("--grok-profile", type=Path, default=Path(_DEFAULT_PROFILE))
    parser.add_argument("--grok-exe", type=Path, default=Path(_DEFAULT_EXE))
    parser.add_argument("--grok-live-settings", type=Path, default=Path(_DEFAULT_LIVE))
    parser.add_argument("--grok-timeout-s", type=float, default=300.0)
    return parser


def _refusal(args: argparse.Namespace, env: dict[str, str]) -> str | None:
    if env.get("DESKTOP_DISPATCH_CONFIRM") != "1":
        return "live dispatch not confirmed"
    if "grok" in args.providers:
        if env.get("GROK_ISO_CONFIRM") != "1":
            return "grok launch not confirmed"
        if env.get("GROK_IMAGE_SUBMIT") != "1":
            return "grok image submit not authorized"
        needed = (
            args.grok_agent_id,
            args.grok_bot_name,
            args.grok_registry,
            args.grok_diagnostics,
            args.grok_log,
        )
        if any(item is None for item in needed):
            return "grok needs agent id, bot name, registry, diagnostics, and log"
    if "gpt" in args.providers and (
        not args.codex_thread_id or args.codex_journal is None
    ):
        return "codex needs a thread id and an attempt journal"
    return None


def build_senders(args: argparse.Namespace) -> dict[str, DesktopSender]:
    """Construct the live senders. No application is contacted until a send."""

    senders: dict[str, DesktopSender] = {}
    if "gpt" in args.providers:
        senders["gpt"] = CodexDesktopSender(
            CodexDesktopClient(
                PowershellCodexTransport(),
                AttemptJournal(args.codex_journal),
                completion_timeout_s=args.codex_timeout_s,
            ),
            thread_id=args.codex_thread_id,
            workspace=args.codex_workspace,
        )
    if "grok" in args.providers:
        senders["grok"] = GrokDesktopSender(
            live_adapter(
                DesktopPaths(
                    profile_dir=args.grok_profile,
                    exe_path=args.grok_exe,
                    live_settings=args.grok_live_settings,
                    registry_path=args.grok_registry,
                    diagnostics_dir=args.grok_diagnostics,
                    log_path=args.grok_log,
                ),
                poll_timeout=args.grok_timeout_s,
            ),
            agent_id=args.grok_agent_id,
            bot_name=args.grok_bot_name,
        )
    return senders


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    refusal = _refusal(args, dict(os.environ))
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return 2
    from transcription_review_agent.review.config import load_review_config

    controller = ReviewController(
        batches_dir=args.batches_dir,
        reviews_dir=args.reviews_dir,
        config=load_review_config(),
    )
    try:
        result = advance_document(
            controller,
            batch_id=args.batch_id,
            document_id=args.document_id,
            queue_root=args.queue_root,
            preservation_dir=args.preservation_dir,
            mount_root=args.mount_root,
            packet_dir=args.packet_dir,
            senders=build_senders(args),
        )
    except (ReviewError, ProfileError, JournalError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(run_payload(result), sort_keys=True))
    return 0 if result.stopped_at == "waiting_for_human" else 3


if __name__ == "__main__":
    raise SystemExit(main())
