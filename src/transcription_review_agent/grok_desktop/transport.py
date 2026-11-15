"""Pipe transport. The CDP bytes are handled by the proven PowerShell harness."""

from __future__ import annotations

import json
import os
import queue
import re
import secrets
import subprocess
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NoReturn

from transcription_review_agent.grok_desktop.profile import local_read_path

ASSETS = Path(__file__).resolve().parent / "assets"
# pipe-host.ps1 waits 25s for the child, then up to three 30s discovery
# reads and one 30s attach before it can say ready. The session deadline
# stays past that sum so a host still inside those waits is not killed.
HOST_CHILD_READY_SECONDS = 25
HOST_DISCOVERY_ATTEMPTS = 3
HOST_DISCOVERY_WAIT_SECONDS = 30
HOST_ATTACH_WAIT_SECONDS = 30
HOST_BROWSER_CLOSE_SECONDS = 15
HOST_STOP_WAIT_SECONDS = 2
HOST_STARTUP_BUDGET_SECONDS = (
    HOST_CHILD_READY_SECONDS
    + HOST_DISCOVERY_ATTEMPTS * HOST_DISCOVERY_WAIT_SECONDS
    + HOST_ATTACH_WAIT_SECONDS
)
HOST_READY_TIMEOUT_SECONDS = HOST_STARTUP_BUDGET_SECONDS + 15
HOST_SHUTDOWN_TIMEOUT_SECONDS = HOST_BROWSER_CLOSE_SECONDS + HOST_STOP_WAIT_SECONDS + 3


class TransportError(Exception):
    """The desktop host failed before a reliable CDP result was available."""

    def __init__(self, message: str, *, cleanup: str = "closed") -> None:
        super().__init__(message)
        self.cleanup = cleanup


class ProcessIdentity:
    """One process at one creation time. A reused PID does not match."""

    def __init__(self, pid: int, creation_ticks: str, executable: str) -> None:
        self.pid = pid
        self.creation_ticks = creation_ticks
        self.executable = executable


def _same_exe(left: str, right: str) -> bool:
    return left.casefold() == right.casefold()


_EVIDENCE_ID = re.compile(r"[0-9a-f]{16}\Z")


def new_evidence_id() -> str:
    """One id for this launch. Sidecars from any other id are not its evidence."""

    return secrets.token_hex(8)


def _evidence_id_from_command(command: list[str] | None) -> str | None:
    if not command:
        return None
    for index, part in enumerate(command):
        if part == "-EvidenceId" and index + 1 < len(command):
            value = command[index + 1]
            if _EVIDENCE_ID.fullmatch(value):
                return value
    return None


def _log_path_from_command(command: list[str] | None) -> Path | None:
    if not command:
        return None
    for index, part in enumerate(command):
        if part == "-LogPath" and index + 1 < len(command):
            return Path(command[index + 1])
    return None


def _as_int_list(value: Any) -> list[int] | None:
    if value is None:
        return []
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        return None
    numbers: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        numbers.append(item)
    return numbers


def _as_records(value: Any) -> list[Any] | None:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return value
    return None


def _run_evidence_path(log_path: Path, evidence_id: str) -> Path:
    return local_read_path(Path(f"{log_path}.run.{evidence_id}.json"))


def load_run_evidence(
    log_path: Path | None, evidence_id: str | None
) -> dict[str, Any] | None:
    """Load this launch's evidence file. Other run ids are left untouched."""

    if log_path is None or evidence_id is None:
        return None
    path = _run_evidence_path(log_path, evidence_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"unreadable": True}
    if not isinstance(payload, dict) or payload.get("evidenceId") != evidence_id:
        return {"unreadable": True}
    return payload


def reap_owned_ledger(
    payload: dict[str, Any] | None,
    lookup: Callable[[int], ProcessIdentity | None],
    stop: Callable[[int], None],
) -> str:
    """Stop recorded children whose pid, creation time, and executable match.

    A missing or incomplete ledger is uncertain. The host exiting is not
    proof that no child is running. Baseline and controller pids are never
    signaled, and a process that already exited is not a failure.
    """

    if payload is None or payload.get("unreadable") is True:
        return "uncertain"
    expected = payload.get("expectedExe")
    if not isinstance(expected, str) or not expected:
        return "uncertain"
    controller = payload.get("controllerPid")
    if isinstance(controller, bool) or not isinstance(controller, int):
        return "uncertain"
    baseline = _as_int_list(payload.get("baseline"))
    owned = _as_records(payload.get("owned"))
    if baseline is None or owned is None:
        return "uncertain"
    protected = set(baseline)
    protected.add(controller)
    uncertain = False
    for record in owned:
        if not isinstance(record, dict):
            uncertain = True
            continue
        pid = record.get("pid")
        ticks = record.get("creationTicks")
        executable = record.get("executable")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            uncertain = True
            continue
        if not isinstance(ticks, str) or not ticks.isdigit():
            uncertain = True
            continue
        if not isinstance(executable, str) or not executable:
            uncertain = True
            continue
        if pid in protected:
            continue
        try:
            current = lookup(pid)
        except Exception:
            uncertain = True
            continue
        if current is None:
            continue
        if current.creation_ticks != ticks or not _same_exe(
            current.executable, executable
        ):
            uncertain = True
            continue
        if not _same_exe(executable, expected):
            uncertain = True
            continue
        try:
            stop(pid)
        except Exception:
            uncertain = True
            continue
        try:
            after = lookup(pid)
        except Exception:
            uncertain = True
            continue
        if (
            after is not None
            and after.creation_ticks == ticks
            and _same_exe(after.executable, executable)
        ):
            uncertain = True
    return "uncertain" if uncertain else "closed"


def cleanup_from_evidence(
    payload: dict[str, Any] | None,
    lookup: Callable[[int], ProcessIdentity | None],
    stop: Callable[[int], None],
) -> str:
    """Decide cleanup from this run's published phase.

    ``prelaunch`` is the only record that says no child was started. A later
    phase without a complete ownership list stays uncertain, and unknown
    processes are not stopped.
    """

    if payload is None or payload.get("unreadable") is True:
        return "uncertain"
    phase = payload.get("phase")
    if phase == "prelaunch":
        return "closed"
    if phase == "owned":
        owned = _as_records(payload.get("owned"))
        if not owned:
            return "uncertain"
        return reap_owned_ledger(payload, lookup, stop)
    return "uncertain"


def windows_process_identity(pid: int) -> ProcessIdentity | None:
    """Read one Windows process. A name or command line is not an identity."""

    if pid <= 0:
        raise OSError("pid")
    script = (
        "$proc = Get-CimInstance Win32_Process -Filter "
        f"('ProcessId={pid}') -ErrorAction SilentlyContinue; "
        "if (-not $proc -or -not $proc.ExecutablePath -or -not $proc.CreationDate) "
        "{ exit 0 }; "
        "[Console]::Out.WriteLine([string]$proc.CreationDate.ToUniversalTime().Ticks); "
        "[Console]::Out.WriteLine([string]$proc.ExecutablePath)"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise OSError("identity lookup failed")
    lines = [line for line in completed.stdout.splitlines() if line]
    if not lines:
        return None
    if len(lines) != 2 or not lines[0].isdigit() or not lines[1]:
        raise OSError("identity lookup failed")
    return ProcessIdentity(pid, lines[0], lines[1])


def windows_stop_process(pid: int) -> None:
    """Stop one pid that the caller has already identified. No name match."""

    if pid <= 0:
        raise OSError("pid")
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"Stop-Process -Id {pid} -Force -ErrorAction Stop",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        raise OSError("stop failed")


def host_command(
    *,
    profile_dir: Path,
    exe_path: Path,
    live_settings: Path,
    log_path: Path,
    evidence_id: str | None = None,
) -> list[str]:
    script = ASSETS / "pipe-host.ps1"
    evidence = evidence_id or new_evidence_id()
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-ConfirmLaunch",
        "-ProfileDir",
        str(profile_dir),
        "-ExePath",
        str(exe_path),
        "-LiveSettings",
        str(live_settings),
        "-LogPath",
        str(log_path),
        "-EvidenceId",
        evidence,
    ]


class PowershellCdpSession:
    """One isolated Grok process. Commands are single JSON lines.

    ``GROK_ISO_CONFIRM=1`` is set only for this process. The host refuses to
    start without ``-ConfirmLaunch`` as well.
    """

    def __init__(
        self,
        command: list[str],
        *,
        timeout: float = HOST_READY_TIMEOUT_SECONDS,
        popen: Any = subprocess.Popen,
        cleanup_timeout: float = 2.0,
        shutdown_timeout: float = HOST_SHUTDOWN_TIMEOUT_SECONDS,
        identity_lookup: Callable[[int], ProcessIdentity | None] | None = None,
        process_stop: Callable[[int], None] | None = None,
    ) -> None:
        self._command = list(command)
        self._evidence_id = _evidence_id_from_command(self._command)
        self._timeout = timeout
        self._cleanup_timeout = cleanup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._identity_lookup = identity_lookup or windows_process_identity
        self._process_stop = process_stop or windows_stop_process
        self._next = 1
        self._closed = False
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._proc: Any = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._proc = popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=windows_host_env(),
        )
        ready: Any = None
        try:
            self._stdout_thread = threading.Thread(
                target=self._read_stdout, daemon=True
            )
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, daemon=True
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
            ready = self._read()
        except TimeoutError as exc:
            self._fail_startup(TransportError("startup timeout"), cause=exc)
        except TransportError as exc:
            self._fail_startup(exc)
        except Exception as exc:
            self._fail_startup(TransportError("host_startup_failed"), cause=exc)
        if not isinstance(ready, dict) or ready.get("ready") is not True:
            detail = ready.get("error") if isinstance(ready, dict) else "host not ready"
            self._fail_startup(TransportError(str(detail)))

    def evaluate(self, expression: str) -> Any:
        payload = self._request({"op": "eval", "expression": expression})
        if payload.get("ok") is not True:
            if payload.get("error") == "timeout":
                raise TimeoutError("cdp evaluate")
            raise TransportError(str(payload.get("error") or "evaluate failed"))
        value = payload.get("value")
        return value if isinstance(value, dict) else {}

    def insert_text(self, text: str) -> None:
        payload = self._request({"op": "insert", "text": text})
        if payload.get("ok") is not True:
            if payload.get("error") == "timeout":
                raise TimeoutError("cdp insert")
            raise TransportError(str(payload.get("error") or "insert failed"))

    def _fail_startup(
        self, exc: TransportError, *, cause: BaseException | None = None
    ) -> NoReturn:
        cleanup = self._abort_owned_host()
        message = str(exc)
        if cleanup == "uncertain":
            message = f"{message}; host_cleanup_uncertain"
        message = self._with_baseline(message)
        raise TransportError(message, cleanup=cleanup) from (cause or exc)

    def _abort_owned_host(self) -> str:
        """Stop this host, then only the children it recorded.

        A stop file lets the host run its own finally block. Terminate is
        reserved for a host that is still alive after that wait. A forced
        stop without a ledger cannot prove the child is gone.
        """

        proc = self._proc
        self._closed = True
        if proc is None:
            self._close_pipes()
            return "closed"
        try:
            if proc.poll() is None:
                self._request_host_stop()
                try:
                    proc.wait(timeout=self._shutdown_timeout)
                except Exception:
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=self._cleanup_timeout)
                except Exception:
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=self._cleanup_timeout)
        except Exception:
            pass
        self._close_pipes()
        return self._cleanup_result()

    def _request_host_stop(self) -> None:
        log_path = _log_path_from_command(self._command)
        if log_path is not None and self._evidence_id is not None:
            stop_path = local_read_path(Path(str(log_path) + ".stop"))
            try:
                stop_path.parent.mkdir(parents=True, exist_ok=True)
                stop_path.write_text(self._evidence_id, encoding="utf-8")
            except OSError:
                pass
        stdin = getattr(self._proc, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except OSError:
                pass

    def _baseline_integrity(self) -> str:
        payload = load_run_evidence(
            _log_path_from_command(self._command),
            self._evidence_id,
        )
        if not isinstance(payload, dict) or payload.get("unreadable") is True:
            return "unknown"
        status = payload.get("baselineIntegrity")
        if status == "intact":
            return "intact"
        if status == "failed":
            return "failed"
        return "unknown"

    def _with_baseline(self, message: str) -> str:
        """Name a baseline miss without changing owned-process cleanup."""

        if self._baseline_integrity() != "failed":
            return message
        if "baseline_integrity_failed" in message:
            return message
        if not message:
            return "baseline_integrity_failed"
        return f"{message}; baseline_integrity_failed"

    def _cleanup_result(self) -> str:
        proc = self._proc
        host_alive = proc is not None and proc.poll() is None
        children = cleanup_from_evidence(
            load_run_evidence(
                _log_path_from_command(self._command),
                self._evidence_id,
            ),
            self._identity_lookup,
            self._process_stop,
        )
        if host_alive or children == "uncertain":
            return "uncertain"
        return "closed"

    def _close_pipes(self) -> None:
        proc = self._proc
        if proc is None:
            return
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is None:
                continue
            try:
                pipe.close()
            except Exception:
                continue

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.poll() is None and self._proc.stdin is not None:
                self._proc.stdin.write(json.dumps({"op": "close"}) + "\n")
                self._proc.stdin.flush()
            self._proc.wait(timeout=50)
        except Exception:
            if self._proc.poll() is None:
                self._proc.terminate()
        finally:
            self._close_pipes()
        cleanup = self._cleanup_result()
        if cleanup == "uncertain":
            raise TransportError(
                self._with_baseline("host_cleanup_uncertain"),
                cleanup="uncertain",
            )
        message = self._with_baseline("")
        if message:
            raise TransportError(message, cleanup="closed")

    def _request(self, body: dict[str, object]) -> dict[str, Any]:
        command_id = self._next
        self._next += 1
        message = dict(body)
        message["id"] = command_id
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()
        payload = self._read()
        if payload.get("id") != command_id:
            raise TransportError("host response id mismatch")
        return payload

    def _read_stdout(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _drain_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        self._proc.stderr.read()

    def _read(self) -> dict[str, Any]:
        while True:
            try:
                line = self._lines.get(timeout=self._timeout)
            except queue.Empty as exc:
                raise TimeoutError("cdp") from exc
            if not line:
                raise TransportError("host closed the pipe")
            payload = decode_protocol_line(line)
            if payload is None:
                continue
            return payload


def decode_protocol_line(line: str) -> dict[str, Any] | None:
    """Return one host object. Plain child logs are ignored.

    A line that starts like JSON and does not parse is a broken protocol
    message, not an application log.
    """

    stripped = line.strip()
    if not stripped.startswith("{"):
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise TransportError("host returned malformed json") from exc
    if not isinstance(payload, dict):
        raise TransportError("host returned no object")
    return payload


_KNOWN_HOST_ERRORS = {
    "launch not confirmed": "launch_not_confirmed",
    "normal profile": "normal_profile",
    "marker missing": "marker_missing",
    "marker mismatch": "marker_mismatch",
    "profile reparse": "profile_reparse",
    "permission preflight mismatch": "permission_mismatch",
    "isolated instance already open": "isolated_already_open",
    "host closed the pipe": "host_startup_failed",
    "startup timeout": "host_startup_failed",
    "host not ready": "host_not_ready",
    "host returned malformed json": "host_protocol_invalid",
    "host returned no object": "host_protocol_invalid",
}
_SAFE_HOST_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}\Z")


def classify_host_failure(exc: BaseException) -> str:
    """Return a stable code. Unknown host text is not copied into the result."""

    if isinstance(exc, FileNotFoundError):
        return "powershell_not_found"
    if isinstance(exc, TransportError):
        message = str(exc).strip()
        flags: list[str] = []
        suffixes = (
            ("; host_cleanup_uncertain", "host_cleanup_uncertain"),
            ("; baseline_integrity_failed", "baseline_integrity_failed"),
        )
        while True:
            for suffix, flag in suffixes:
                if message.endswith(suffix):
                    flags.append(flag)
                    message = message[: -len(suffix)].strip()
                    break
            else:
                break
        known = _KNOWN_HOST_ERRORS.get(message)
        if known is not None:
            code = known
        elif message == "":
            code = ""
        elif _SAFE_HOST_CODE.fullmatch(message):
            code = message
        else:
            code = "host_startup_failed"
        if not code:
            code = flags[0] if flags else "host_startup_failed"
            flags = flags[1:]
        for flag in flags:
            code = f"{code}:{flag}"
        return code
    return "host_startup_failed"


def windows_host_env(
    base: Mapping[str, str] | None = None,
    *,
    wsl: bool | None = None,
) -> dict[str, str]:
    """Environment for one host process. Does not change the parent process.

    From WSL, Windows interop only forwards variables named by ``WSLENV``.
    ``/w`` shares this confirmation toward Win32 and leaves every other entry
    in place. The host still requires ``-ConfirmLaunch`` as well.
    """

    env = dict(os.environ if base is None else base)
    env["GROK_ISO_CONFIRM"] = "1"
    if _use_wslenv(wsl):
        env["WSLENV"] = _wslenv_with_confirm(env.get("WSLENV", ""))
    return env


def _use_wslenv(wsl: bool | None) -> bool:
    if wsl is not None:
        return wsl
    if os.name == "nt":
        return False
    if Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
        return True
    try:
        version = Path("/proc/version").read_text(encoding="utf-8")
    except OSError:
        return False
    return "microsoft" in version.lower()


def _confirm_entry(flags: str) -> str:
    """One WSLENV entry that shares the literal value toward Windows.

    ``/p`` and ``/l`` translate paths, so they cannot carry ``1``. ``/u``
    alone never reaches Win32. ``/w`` does, without translating the value.
    """

    if flags == "":
        return "GROK_ISO_CONFIRM"
    keep_u = "u" in flags and "p" not in flags and "l" not in flags
    suffix = ("u" if keep_u else "") + "w"
    return f"GROK_ISO_CONFIRM/{suffix}"


def _wslenv_with_confirm(current: str) -> str:
    entries = [part for part in current.split(":") if part]
    updated: list[str] = []
    placed = False
    for entry in entries:
        name, slash, flags = entry.partition("/")
        if name != "GROK_ISO_CONFIRM":
            updated.append(entry)
            continue
        if placed:
            continue
        updated.append(_confirm_entry(flags if slash else ""))
        placed = True
    if not placed:
        updated.append("GROK_ISO_CONFIRM/w")
    return ":".join(updated)
