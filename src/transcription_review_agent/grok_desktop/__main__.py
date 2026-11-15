"""Manual acceptance entry. Importing it does not launch Grok."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from transcription_review_agent.grok_desktop.adapter import DesktopPaths, live_adapter
from transcription_review_agent.grok_desktop.models import ExchangeRequest

_DEFAULT_PROFILE = Path(r"C:\Users\User\AppData\Local\Grok Bot Isolated Pilot")
_DEFAULT_EXE = Path(r"C:\Users\User\AppData\Local\Programs\Grok Bot\Grok Bot.exe")
_DEFAULT_LIVE = Path(r"C:\Users\User\.grokbot\settings.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Send one message to an isolated Grok bot."
    )
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--bot-name", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--source-document-id")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=_DEFAULT_PROFILE)
    parser.add_argument("--exe", type=Path, default=_DEFAULT_EXE)
    parser.add_argument("--live-settings", type=Path, default=_DEFAULT_LIVE)
    args = parser.parse_args(argv)
    if os.environ.get("GROK_ISO_CONFIRM") != "1":
        print("launch not confirmed", file=sys.stderr)
        return 2
    adapter = live_adapter(
        DesktopPaths(
            profile_dir=args.profile,
            exe_path=args.exe,
            live_settings=args.live_settings,
            registry_path=args.registry,
            diagnostics_dir=args.diagnostics,
            log_path=args.log,
        )
    )
    result = adapter.exchange(
        ExchangeRequest(
            agent_id=args.agent_id,
            bot_name=args.bot_name,
            message=args.message,
            request_id=args.request_id,
            source_document_id=args.source_document_id,
        )
    )
    print(json.dumps(result.to_public_dict(), indent=2))
    return 0 if result.status.value == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
