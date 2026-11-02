"""Local-image first-pass tasks stay outside the sealed review ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewOutputRejected,
)
from transcription_review_agent.review.local_image_pilot import (
    PILOT_02_SHA256,
    PILOT_BATCH_ID,
    PILOT_DOCUMENT_ID,
    PILOT_ORIGIN,
    check_local_image_first_pass_output,
    prepare_local_image_first_pass,
)
from transcription_review_agent.review.models import FirstPassOutput, WorkRequest
from transcription_review_agent.review.pngutil import encode_png
from test_review_workflow import _page

REAL_IMAGE = Path(
    "/mnt/c/Users/User/vs-archive-transcription-review-agent/"
    "desktop_pilot/pilot_02/input/page_001_original.png"
)
OCR = "ARCHIVE OCR TRANSCRIPTION THAT MUST NOT LEAK INTO PROVIDER FILES"


def _image() -> bytes:
    return encode_png(2, 2, [bytearray(b"\x00\x00"), bytearray(b"\x01\x02")], 1)


def _prepare(
    tmp_path: Path, payload: bytes | None = None, *, expected: str | None = None
):
    data = _image() if payload is None else payload
    image = tmp_path / "source" / "page.png"
    image.parent.mkdir()
    image.write_bytes(data)
    (image.parent / "transcription.txt").write_text(OCR, encoding="utf-8")
    digest = sha256_bytes(data) if expected is None else expected
    return prepare_local_image_first_pass(
        image,
        tmp_path / "pilot",
        expected_sha256=digest,
        language="fr",
    )


def test_hash_mismatch_writes_nothing(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(_image())
    with pytest.raises(ReviewIntegrityError, match="does not match"):
        prepare_local_image_first_pass(
            image,
            tmp_path / "pilot",
            expected_sha256="0" * 64,
            language="fr",
        )
    assert not (tmp_path / "pilot").exists()


def test_prepare_publishes_independent_tasks_without_a_seal(tmp_path: Path) -> None:
    paths = _prepare(tmp_path)
    prepare_local_image_first_pass(
        tmp_path / "source" / "page.png",
        paths.destination,
        expected_sha256=sha256_bytes(_image()),
        language="fr",
    )
    assert paths.gpt_instructions.is_file()
    assert paths.grok_instructions.is_file()
    assert paths.shared_output.is_dir()
    provenance = json.loads(paths.provenance.read_text(encoding="utf-8"))
    assert provenance["origin"] == PILOT_ORIGIN
    assert provenance["sealed_archive_bundle"] is False
    assert provenance["sha256"] == sha256_bytes(_image())
    assert provenance["width"] == 2 and provenance["height"] == 2
    assert provenance["color_type"] == "gray"
    forbidden_names = {
        "transcription.txt",
        "review_bundle.json",
        "batch_manifest.json",
        "document_manifest.json",
        "ledger.jsonl",
        "state.json",
    }
    found = {path.name for path in paths.destination.rglob("*") if path.is_file()}
    assert forbidden_names.isdisjoint(found)
    assert not (paths.destination / "controller").exists()
    for provider, interface_name in (("gpt", "GPT_WORK"), ("grok", "GROK")):
        task = paths.destination / "first_pass" / "tasks" / provider
        names = {
            path.relative_to(task).as_posix()
            for path in task.rglob("*")
            if path.is_file()
        }
        assert names == {
            f"instructions_{provider}.txt",
            "work_request.json",
            "output_schema.json",
            "README.txt",
            "input/source_manifest.json",
            "input/page_001_original.png",
            "input/local_image_provenance.json",
        }
        request = WorkRequest.model_validate_json(
            (task / "work_request.json").read_text(encoding="utf-8")
        )
        assert request.provider == provider
        assert request.provider_interface == interface_name
        assert request.batch_id == PILOT_BATCH_ID
        assert request.document_id == PILOT_DOCUMENT_ID
        assert request.source_bundle_sha256 == provenance["sha256"]
        assert (
            request.output_relative_path
            == f"first_pass/output/transcription_{provider}.json"
        )
        assert "original_ocr_transcription" in request.forbidden_inputs
        schema = json.loads((task / "output_schema.json").read_text(encoding="utf-8"))
        assert schema["schema_version"] == "review-provider-output-v1"
        assert schema["stage"] == "first_pass"
        assert schema["page_fields"]["regions"][0]["order"].startswith("integer")
        assert "examined_region_ids" in schema["page_fields"]["coverage"]
        assert "region.reading_order" in schema["rejected_examples"]
        text = "\n".join(
            path.read_text(encoding="utf-8") for path in task.rglob("*.json")
        )
        text += "\n" + "\n".join(
            path.read_text(encoding="utf-8") for path in task.rglob("*.txt")
        )
        assert OCR not in text
        assert "LOCAL_IMAGE_PILOT" in (task / f"instructions_{provider}.txt").read_text(
            encoding="utf-8"
        )
    copied = (paths.destination / "input" / "page_001_original.png").read_bytes()
    assert copied == _image()
    assert sha256_bytes(copied) == provenance["sha256"]


def test_real_pilot_02_image_prepares_when_present(tmp_path: Path) -> None:
    if not REAL_IMAGE.is_file():
        pytest.skip("Pilot 02 image is not mounted")
    assert sha256_bytes(REAL_IMAGE.read_bytes()) == PILOT_02_SHA256
    paths = prepare_local_image_first_pass(
        REAL_IMAGE,
        tmp_path / "pilot",
        expected_sha256=PILOT_02_SHA256,
        language="fr",
    )
    provenance = json.loads(paths.provenance.read_text(encoding="utf-8"))
    assert provenance["sha256"] == PILOT_02_SHA256
    assert provenance["width"] == 1536
    assert provenance["height"] == 2048
    assert provenance["color_type"] == "rgba"
    assert provenance["local_source_path"] == str(REAL_IMAGE.resolve())
    copied = paths.destination / "input" / "page_001_original.png"
    assert sha256_bytes(copied.read_bytes()) == PILOT_02_SHA256


def test_check_validates_provider_output_without_a_review_ledger(
    tmp_path: Path,
) -> None:
    paths = _prepare(tmp_path)
    request_path = (
        paths.destination / "first_pass" / "tasks" / "gpt" / "work_request.json"
    )
    request = WorkRequest.model_validate_json(request_path.read_text(encoding="utf-8"))
    page = _page("page_001", 1, "ligne visible")
    output = FirstPassOutput(
        provider=request.provider,
        provider_interface=request.provider_interface,
        task_id=request.task_id,
        job_id=request.job_id,
        batch_id=request.batch_id,
        document_id=request.document_id,
        attempt=request.attempt,
        source_bundle_sha256=request.source_bundle_sha256,
        pages=[page],
    )
    target = paths.shared_output / "transcription_gpt.json"
    target.write_bytes(output.model_dump_json().encode("utf-8"))
    checked = check_local_image_first_pass_output(paths.destination, "gpt")
    assert checked.pages[0].diplomatic_transcription == "ligne visible"
    assert not (paths.destination / "controller").exists()
    target.write_text(
        target.read_text(encoding="utf-8").replace("{", '{"confidence":0.9,', 1),
        encoding="utf-8",
    )
    with pytest.raises(ReviewOutputRejected):
        check_local_image_first_pass_output(paths.destination, "gpt")
