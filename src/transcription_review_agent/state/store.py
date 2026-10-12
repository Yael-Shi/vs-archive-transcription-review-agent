"""Durable checkpoint and append-only ledger.

The ledger is authoritative. On load, entries are replayed from the beginning.
A checkpoint that matches the replay is kept. A checkpoint behind the ledger
is rewritten from the replay. A checkpoint that disagrees with the replay, or
that is ahead of the ledger, is an integrity error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from transcription_review_agent.errors import StateIntegrityError, StateInvariantError
from transcription_review_agent.state.ledger import (
    LedgerEntry,
    dump_ledger_entry,
    parse_ledger_entry,
)
from transcription_review_agent.state.schema import ControllerState, initial_state
from transcription_review_agent.state.transitions import apply_ledger_entry, replay_ledger

CHECKPOINT_NAME = "controller_state.json"
LEDGER_NAME = "ledger.jsonl"


class StateStore:
    """Filesystem checkpoint plus JSONL ledger. No concurrency control."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.checkpoint_path = state_dir / CHECKPOINT_NAME
        self.ledger_path = state_dir / LEDGER_NAME

    def load(self) -> ControllerState:
        """Return a checkpoint consistent with the ledger."""

        entries = self.read_ledger()
        try:
            replayed = replay_ledger(entries)
        except StateInvariantError as exc:
            raise StateIntegrityError("ledger replay broke state invariants") from exc
        if not self.checkpoint_path.exists():
            if entries:
                self.write_checkpoint(replayed)
            return replayed

        checkpoint = _read_checkpoint(self.checkpoint_path)
        if checkpoint.applied_ledger_sequence > len(entries):
            raise StateIntegrityError(
                "checkpoint is ahead of the ledger; refusing to guess missing history"
            )
        try:
            prefix = replay_ledger(entries[: checkpoint.applied_ledger_sequence])
        except StateInvariantError as exc:
            raise StateIntegrityError("ledger replay broke state invariants") from exc
        if checkpoint != prefix:
            raise StateIntegrityError("checkpoint does not match ledger replay")
        if checkpoint.applied_ledger_sequence < len(entries):
            self.write_checkpoint(replayed)
            return replayed
        return checkpoint

    def commit(self, entry: LedgerEntry) -> ControllerState:
        """Append one ledger line, then atomically replace the checkpoint.

        If the process stops after the append and before the checkpoint
        write, the next ``load`` replays the new line.
        """

        current = self.load()
        updated = apply_ledger_entry(current, entry)
        self.append_ledger_line(entry)
        self.write_checkpoint(updated)
        return updated

    def read_ledger(self) -> list[LedgerEntry]:
        if not self.ledger_path.exists():
            return []
        entries: list[LedgerEntry] = []
        text = self.ledger_path.read_text(encoding="utf-8")
        if text == "":
            return []
        if not text.endswith("\n"):
            raise StateIntegrityError("ledger must end with a newline")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise StateIntegrityError(f"blank ledger line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StateIntegrityError(
                    f"ledger line {line_number} is not JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise StateIntegrityError(f"ledger line {line_number} is not an object")
            try:
                entry = parse_ledger_entry(payload)
            except ValueError as exc:
                raise StateIntegrityError(
                    f"ledger line {line_number} does not match the ledger schema"
                ) from exc
            if entry.sequence != line_number:
                raise StateIntegrityError(
                    f"ledger sequence {entry.sequence} is not contiguous at line {line_number}"
                )
            entries.append(entry)
        return entries

    def append_ledger_line(self, entry: LedgerEntry) -> None:
        """Append ``entry`` if it is the next sequence number. Never rewrites."""

        existing = self.read_ledger()
        expected = len(existing) + 1
        if entry.sequence != expected:
            raise StateIntegrityError(
                f"refusing to append sequence {entry.sequence}; next sequence is {expected}"
            )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dump_ledger_entry(entry), separators=(",", ":"), sort_keys=True)
        payload = (line + "\n").encode("utf-8")
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        fd = os.open(self.ledger_path, flags, 0o644)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)

    def write_checkpoint(self, state: ControllerState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json(indent=2) + "\n"
        _atomic_write(self.checkpoint_path, payload)


def _read_checkpoint(path: Path) -> ControllerState:
    try:
        return ControllerState.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StateIntegrityError("checkpoint JSON does not match the state schema") from exc


def _atomic_write(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    data = payload.encode("utf-8")
    flags = os.O_CREAT | os.O_TRUNC | os.O_WRONLY
    fd = os.open(temporary, flags, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)
