"""Adapter entry that stays outside the review pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from transcription_review_agent.grok_desktop.models import (
    ExchangeRequest,
    ExchangeResult,
    ExchangeStatus,
)
from transcription_review_agent.grok_desktop.profile import (
    assert_isolated_profile,
    load_profile_text,
)
from transcription_review_agent.grok_desktop.registry import ResourceRegistry
from transcription_review_agent.grok_desktop.session import (
    AttachmentPreflight,
    Clock,
    CdpSession,
    SteadyClock,
    run_attachment_preflight,
    run_exchange,
)
from transcription_review_agent.grok_desktop.transcript import validate_request
from transcription_review_agent.grok_desktop.transport import (
    PowershellCdpSession,
    TransportError,
    classify_host_failure,
    host_command,
)


class DesktopPaths:
    def __init__(
        self,
        *,
        profile_dir: Path,
        exe_path: Path,
        live_settings: Path,
        registry_path: Path,
        diagnostics_dir: Path,
        log_path: Path,
    ) -> None:
        self.profile_dir = profile_dir
        self.exe_path = exe_path
        self.live_settings = live_settings
        self.registry_path = registry_path
        self.diagnostics_dir = diagnostics_dir
        self.log_path = log_path


class GrokDesktopAdapter:
    def __init__(
        self,
        *,
        registry: ResourceRegistry,
        diagnostics_dir: Path,
        open_session: Callable[[], CdpSession],
        check_profile: Callable[[], None],
        clock: Clock | None = None,
        poll_timeout: float = 90.0,
    ) -> None:
        self._registry = registry
        self._diagnostics_dir = diagnostics_dir
        self._open_session = open_session
        self._check_profile = check_profile
        self._clock = clock or SteadyClock()
        self._poll_timeout = poll_timeout

    def attachment_preflight(
        self, paths: tuple[str, ...], *, agent_id: str, bot_name: str
    ) -> AttachmentPreflight:
        """Open the isolated app, set the composer files, and do not submit.

        The caller has to authorize the launch. This does not reserve a
        request id and does not insert a message.
        """

        self._check_profile()
        session = self._open_session()
        try:
            return run_attachment_preflight(
                session, paths, agent_id=agent_id, bot_name=bot_name
            )
        finally:
            session.close()

    def exchange(self, request: ExchangeRequest) -> ExchangeResult:
        self._check_profile()
        invalid = validate_request(
            agent_id=request.agent_id,
            bot_name=request.bot_name,
            message=request.message,
            request_id=request.request_id,
        )
        if invalid:
            return _error(request, reason=invalid)
        replay = self._registry.begin_attempt(request)
        if replay is not None:
            if replay.baseline == "failed":
                raise TransportError(
                    "baseline_integrity_failed",
                    cleanup=replay.cleanup,
                )
            return replay
        try:
            session = self._open_session()
        except Exception as exc:
            result = _error(
                request,
                reason=f"session_open_failed:{classify_host_failure(exc)}",
            )
            self._persist(request, result)
            return result
        result = _error(request, reason="exchange_failed")
        close_error: BaseException | None = None
        persist_error: BaseException | None = None
        try:
            try:
                result = run_exchange(
                    session,
                    request,
                    self._clock,
                    poll_timeout=self._poll_timeout,
                )
            except Exception as exc:
                result = _error(request, reason=f"exchange_failed:{type(exc).__name__}")
            result = replace(result, cleanup="unrecorded")
            try:
                self._persist(request, result)
            except Exception as exc:
                persist_error = exc
        finally:
            try:
                session.close()
            except Exception as exc:
                close_error = exc
        result = replace(
            result,
            cleanup=_cleanup_state(close_error),
            baseline=_baseline_state(close_error),
        )
        try:
            self._registry.finish_attempt(result)
        except Exception:
            result = replace(
                result,
                cleanup=(
                    "unrecorded" if result.cleanup == "closed" else result.cleanup
                ),
                baseline=(
                    "unrecorded" if result.baseline == "intact" else result.baseline
                ),
            )
        else:
            try:
                self._write_diagnostics(request, result)
            except Exception:
                pass
        if close_error is not None and "baseline_integrity_failed" in str(close_error):
            raise close_error
        if persist_error is not None and result.cleanup == "unrecorded":
            raise persist_error
        return result

    def _persist(self, request: ExchangeRequest, result: ExchangeResult) -> None:
        self._registry.finish_attempt(result)
        self._write_diagnostics(request, result)

    def _write_diagnostics(
        self, request: ExchangeRequest, result: ExchangeResult
    ) -> None:
        self._diagnostics_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(
            char if char.isalnum() or char in "-_." else "_"
            for char in request.request_id
        )
        payload = {
            "request_id": request.request_id,
            "agent_id": request.agent_id,
            "source_document_id": request.source_document_id,
            "message_sha256": hashlib.sha256(
                request.message.encode("utf-8")
            ).hexdigest(),
            "result": result.to_public_dict(),
        }
        path = self._diagnostics_dir / f"{safe_name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def live_adapter(paths: DesktopPaths) -> GrokDesktopAdapter:
    """Open the isolated profile. Does nothing until ``exchange`` is called."""

    def check_profile() -> None:
        marker = load_profile_text(paths.profile_dir / "pilot-marker.txt")
        pilot = load_profile_text(paths.profile_dir / "sand-data" / "settings.json")
        live = load_profile_text(paths.live_settings)
        assert_isolated_profile(
            paths.profile_dir,
            marker_text=marker,
            pilot_settings=pilot,
            live_settings=live,
        )

    def open_session() -> CdpSession:
        command = host_command(
            profile_dir=paths.profile_dir,
            exe_path=paths.exe_path,
            live_settings=paths.live_settings,
            log_path=paths.log_path,
        )
        return PowershellCdpSession(command)

    return GrokDesktopAdapter(
        registry=ResourceRegistry(paths.registry_path),
        diagnostics_dir=paths.diagnostics_dir,
        open_session=open_session,
        check_profile=check_profile,
    )


def _baseline_state(exc: BaseException | None) -> str:
    """Record baseline integrity separately from owned-process cleanup.

    A normal close observed no baseline failure and is ``intact``. An explicit
    baseline failure stays ``failed`` even when owned cleanup is ``closed``.
    Any other close error was not a baseline observation and stays
    ``unrecorded``.
    """

    if exc is not None and "baseline_integrity_failed" in str(exc):
        return "failed"
    if exc is None:
        return "intact"
    return "unrecorded"


def _cleanup_state(exc: BaseException | None) -> str:
    """Classify process cleanup without changing the exchange outcome.

    ``closed`` is a normal return. ``uncertain`` is an explicit host cleanup
    doubt. Any other exception is ``failed``. A missing record stays
    ``unrecorded`` and is never treated as success.
    """

    if exc is None:
        return "closed"
    text = str(exc)
    marker = getattr(exc, "cleanup", "")
    if marker == "uncertain" or "host_cleanup_uncertain" in text:
        return "uncertain"
    if marker == "closed" and "baseline_integrity_failed" in text:
        return "closed"
    return "failed"


def _error(request: ExchangeRequest, *, reason: str) -> ExchangeResult:
    return ExchangeResult(
        status=ExchangeStatus.ERROR,
        sent=False,
        request_id=request.request_id,
        agent_id=request.agent_id,
        reason=reason,
    )
