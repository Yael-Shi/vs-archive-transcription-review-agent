from __future__ import annotations

from pathlib import Path

import pytest

from transcription_review_agent.config import load_config
from transcription_review_agent.errors import ConfigError


def test_config_requires_origin_and_uses_local_defaults(tmp_path: Path) -> None:
    config = load_config(
        {"TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000/"},
        repo_root=tmp_path,
    )
    assert config.api_base_url == "http://127.0.0.1:8000"
    assert config.state_dir == (tmp_path / "state").resolve()
    assert config.batches_dir == (tmp_path / "outputs" / "batches").resolve()
    assert config.scan_page_limit == 50
    assert config.batch_size == 5


def test_batch_size_is_configured_without_code_changes(tmp_path: Path) -> None:
    config = load_config(
        {
            "TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000",
            "TRANSCRIPTION_REVIEW_BATCH_SIZE": "2",
        },
        repo_root=tmp_path,
    )
    assert config.batch_size == 2


def test_config_rejects_credentials_path_and_vs_archive_checkout(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config({}, repo_root=tmp_path)
    with pytest.raises(ConfigError):
        load_config(
            {"TRANSCRIPTION_REVIEW_API_BASE_URL": "http://user:secret@127.0.0.1:8000"},
            repo_root=tmp_path,
        )
    with pytest.raises(ConfigError):
        load_config(
            {"TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000/api"},
            repo_root=tmp_path,
        )
    with pytest.raises(ConfigError):
        load_config(
            {
                "TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000",
                "TRANSCRIPTION_REVIEW_STATE_DIR": "/home/yshif/vs-archive/vs-archive/state",
            },
            repo_root=tmp_path,
        )
    with pytest.raises(ConfigError):
        load_config(
            {
                "TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000",
                "TRANSCRIPTION_REVIEW_SCAN_PAGE_LIMIT": "201",
            },
            repo_root=tmp_path,
        )
    with pytest.raises(ConfigError):
        load_config(
            {
                "TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000",
                "TRANSCRIPTION_REVIEW_BATCH_SIZE": "0",
            },
            repo_root=tmp_path,
        )
    with pytest.raises(ConfigError):
        load_config(
            {
                "TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000",
                "TRANSCRIPTION_REVIEW_BATCH_SIZE": "201",
            },
            repo_root=tmp_path,
        )
    with pytest.raises(ConfigError):
        load_config(
            {
                "TRANSCRIPTION_REVIEW_API_BASE_URL": "http://127.0.0.1:8000",
                "TRANSCRIPTION_REVIEW_BATCH_SIZE": "many",
            },
            repo_root=tmp_path,
        )
