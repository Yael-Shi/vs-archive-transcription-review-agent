"""Reproducible localization: reference isolation, one-shot runs, and overlap."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from transcription_review_agent.bundling.hashing import sha256_bytes
from transcription_review_agent.bundling.publish import canonical_json
from transcription_review_agent.review.errors import (
    ReviewIntegrityError,
    ReviewPathError,
    ReviewUsageError,
)
from transcription_review_agent.review.learning_export import LearningCandidate
from transcription_review_agent.review.models import BBox
from transcription_review_agent.review.pngutil import decode_png, encode_png
from transcription_review_agent.review.reproducible_localization import (
    ConfirmedLocation,
    ConfirmedLocationReference,
    LocalizationRunError,
    LocationConfirmation,
    MAX_MODEL_RESPONSE_BYTES,
    ModelCall,
    ModelResponse,
    PilotCase,
    assert_reference_isolated,
    build_confirmed_reference,
    case_from_candidate,
    _OUTPUT_SCHEMA,
    codex_exec_argv,
    decode_model_payload,
    evaluate_locations,
    extract_model_json,
    interpret_model_output,
    prepare_tool_paths,
    model_input_document,
    render_prompt,
    run_localization,
    select_cases,
    write_reference_files,
)
from transcription_review_agent.review.visual_localization import (
    ProposalProvenance,
    VisualProposal,
    VisualProposalSidecar,
    normalize_measured_pixels,
)

_SHA = "ab" * 32


class Scripted:
    def __init__(
        self, responses: list[str] | None = None, error: Exception | None = None
    ):
        self.responses = list(responses or [])
        self.error = error
        self.calls: list[ModelCall] = []

    def run(self, call: ModelCall) -> ModelResponse:
        self.calls.append(call)
        if self.error is not None:
            raise self.error
        index = len(self.calls) - 1
        if index >= len(self.responses):
            raise AssertionError("runner made an extra model call")
        return ModelResponse(text=self.responses[index], model_id="mock", tool="mock")


def _png(path: Path, width: int = 40, height: int = 30) -> str:
    rows = [bytearray(b"\xff\xff\xff" * width) for _ in range(height)]
    data = encode_png(width, height, rows, 3)
    path.write_bytes(data)
    return sha256_bytes(data)


def _case(path: Path, ident: str = "a" * 20, **updates: object) -> PilotCase:
    digest = _png(path)
    payload: dict[str, object] = {
        "candidate_id": "lc_" + ident,
        "dispute_id": "dsp_" + "b" * 16,
        "document_id": 105,
        "page_id": "page_001",
        "page_image_sha256": digest,
        "page_width": 40,
        "page_height": 30,
        "dispute_category": "MISSING_CONTENT",
        "gpt_text": "alpha reading",
        "grok_text": "beta reading",
        "image_path": str(path),
    }
    payload.update(updates)
    return PilotCase.model_validate(payload)


def _response(case: PilotCase, status: str, boxes: list[dict[str, int]]) -> str:
    return json.dumps(
        {
            "candidate_id": case.candidate_id,
            "page_image_sha256": case.page_image_sha256,
            "status": status,
            "boxes": boxes,
        }
    )


def _with_boxes(case: PilotCase, boxes: list[dict[str, int]]) -> ConfirmedLocation:
    data = _reference(case).locations[0].model_dump()
    data["boxes"] = boxes
    data["measured_pixels"] = []
    return ConfirmedLocation.model_validate(data)


def _reference(case: PilotCase) -> ConfirmedLocationReference:
    location = ConfirmedLocation(
        candidate_id=case.candidate_id,
        dispute_id=case.dispute_id,
        document_id=case.document_id,
        page_id=case.page_id,
        page_image_sha256=case.page_image_sha256,
        page_width=case.page_width,
        page_height=case.page_height,
        dispute_category=case.dispute_category,
        boxes=[
            {
                "x": 222222,
                "y": 333333,
                "width": 444444,
                "height": 121212,
                "scale": 1_000_000,
            }
        ],
        measured_pixels=[
            {
                "x": 9,
                "y": 9,
                "width": 18,
                "height": 4,
                "page_width": case.page_width,
                "page_height": case.page_height,
            }
        ],
        source_method_version="visual-proposal-pilot-1",
    )
    return ConfirmedLocationReference(
        source_proposal_sha256=_SHA,
        locations=[location],
    )


def _box(x: int, y: int, width: int, height: int) -> dict[str, int]:
    return {"x": x, "y": y, "width": width, "height": height, "scale": 1_000_000}


def _proposal(
    ident: str,
    *,
    document_id: int,
    page_id: str,
    category: str,
    pixels: tuple[int, int, int, int, int, int],
) -> VisualProposal:
    x, y, width, height, page_width, page_height = pixels
    digest = hashlib.sha256(ident.encode()).hexdigest()
    box = normalize_measured_pixels(
        x=x,
        y=y,
        width=width,
        height=height,
        page_width=page_width,
        page_height=page_height,
    )
    return VisualProposal(
        candidate_id="lc_" + ident,
        dispute_id="dsp_" + ident[:16],
        document_id=document_id,
        page_id=page_id,
        page_image_sha256=digest,
        page_width=page_width,
        page_height=page_height,
        dispute_category=category,
        status="PROPOSED",
        method="MODEL_PROPOSAL",
        provenance=ProposalProvenance(
            image_sha256=digest,
            method_version="visual-proposal-pilot-1",
            model_id="example",
            tool="mock",
            request_sha256=_SHA,
            executed=True,
        ),
        boxes=[box],
        explanation=(
            "Not verified. "
            f"Measured pixels x={x} y={y} width={width} height={height} "
            f"on {page_width}x{page_height}."
        ),
    )


def _pilot_sidecar() -> VisualProposalSidecar:
    return VisualProposalSidecar(
        results=[
            _proposal(
                "a" * 20,
                document_id=105,
                page_id="page_001",
                category="MISSING_CONTENT",
                pixels=(10, 4, 12, 6, 80, 90),
            ),
            _proposal(
                "c" * 20,
                document_id=104,
                page_id="page_002",
                category="NAME_OR_ENTITY",
                pixels=(8, 20, 30, 10, 100, 120),
            ),
            _proposal(
                "d" * 20,
                document_id=108,
                page_id="page_005",
                category="DATE_OR_NUMBER",
                pixels=(6, 15, 40, 12, 110, 140),
            ),
        ]
    )


def _candidate_with_bbox() -> LearningCandidate:
    page_sha = _SHA
    return LearningCandidate(
        candidate_id="lc_" + "a" * 20,
        batch_id="batch-000001",
        document_id=105,
        source_bundle_sha256=_SHA,
        page_id="page_001",
        page_image_sha256=page_sha,
        page_image_filename="page_001.jpg",
        page_mime_type="image/jpeg",
        gpt_region_ids=[],
        grok_region_ids=[],
        dispute_id="dsp_" + "b" * 16,
        dispute_category="MISSING_CONTENT",
        learning_category="UNRESOLVED_DISAGREEMENT",
        model_supported=False,
        reading_role="FULL_DISPUTE",
        gpt_text="full alpha",
        grok_text="full beta",
        dispute_gpt_text="alpha reading",
        dispute_grok_text="beta reading",
        image_span_status="RECORDED_REGION_BBOX",
        image_span_limitation="recorded limitation",
        image_bbox=BBox(x=222222, y=333333, width=444444, height=121212),
        reconciled_status="UNRESOLVED",
        closure_status="SELECTED_UNRESOLVED",
        evidence=[],
        review_report_sha256=_SHA,
        comparison_sha256=_SHA,
        gpt_output_sha256=_SHA,
        grok_output_sha256=_SHA,
    )


def _none() -> None:
    return None


def _confirmations(sidecar: VisualProposalSidecar) -> list[LocationConfirmation]:
    return [
        LocationConfirmation(
            candidate_id=item.candidate_id,
            dispute_id=item.dispute_id,
            document_id=item.document_id,
            page_id=item.page_id,
            page_image_sha256=item.page_image_sha256,
            action="USER_VISUAL_CONFIRMATION",
            boxes=list(item.boxes),
        )
        for item in sidecar.results
    ]


def _call(case: PilotCase) -> ModelCall:
    return ModelCall(
        candidate_id=case.candidate_id,
        image_path=case.image_path,
        prompt="locate only",
        request_sha256=_SHA,
        prompt_sha256=_SHA,
    )


def test_reference_boxes_stay_out_of_model_input(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    case = case_from_candidate(
        _candidate_with_bbox(),
        page_width=40,
        page_height=30,
        image_path=image,
    )
    reference = _reference(case)
    document = model_input_document(case)
    assert set(document) == {
        "prompt_version",
        "candidate_id",
        "dispute_id",
        "document_id",
        "page_id",
        "page_image_sha256",
        "page_width",
        "page_height",
        "dispute_category",
        "gpt_text",
        "grok_text",
    }
    assert "boxes" not in document
    assert "image_bbox" not in document
    payload = render_prompt(case) + json.dumps(document)
    assert_reference_isolated(payload, reference)
    assert_reference_isolated(
        payload + "\ndispute_id: dsp_3ef17ff205fa9ee53\n", reference
    )
    with pytest.raises(ReviewIntegrityError, match="confirmed reference coordinate"):
        assert_reference_isolated(payload + "\nmeasured y=9\n", reference)
    for number in (222222, 333333, 444444, 121212, 9, 18):
        assert str(number) not in payload
    assert "reference" not in inspect.signature(run_localization).parameters
    assert case.gpt_text == "alpha reading"
    assert case.grok_text == "beta reading"


def test_confirmed_reference_copies_boxes_without_touching_the_source(
    tmp_path: Path,
) -> None:
    sidecar = _pilot_sidecar()
    source = tmp_path / "proposals.json"
    source.write_text(canonical_json(sidecar), encoding="utf-8")
    before = source.read_bytes()
    confirmations = _confirmations(sidecar)
    with pytest.raises(ReviewIntegrityError, match="recorded human confirmation"):
        build_confirmed_reference(before, sidecar, [])
    reference = build_confirmed_reference(before, sidecar, confirmations)
    assert source.read_bytes() == before
    assert reference.human_verified_transcription_count == 0
    assert reference.decision_ledger_written is False
    assert [item.candidate_id for item in reference.locations] == [
        "lc_" + "a" * 20,
        "lc_" + "c" * 20,
        "lc_" + "d" * 20,
    ]
    for copied, original in zip(reference.locations, sidecar.results, strict=True):
        assert copied.boxes == original.boxes
        assert copied.page_image_sha256 == original.page_image_sha256
        assert copied.location_authority == "HUMAN_CONFIRMED_LOCATION"
        assert copied.human_verification_status == "NOT_VERIFIED"
        assert copied.transcription_verified is False
        assert copied.measured_pixels
    original.boxes[0] = original.boxes[0].model_copy(
        update={"width": original.boxes[0].width - 1}
    )
    with pytest.raises(ReviewIntegrityError):
        build_confirmed_reference(
            before,
            VisualProposalSidecar(results=sidecar.results),
            _confirmations(VisualProposalSidecar(results=sidecar.results)),
        )
    written = write_reference_files(
        source, tmp_path / "reference", confirmations=confirmations
    )
    assert source.read_bytes() == before
    assert (tmp_path / "reference" / "reference.json").is_file()
    assert written.locations[0].boxes == sidecar.results[0].boxes


def test_invalid_model_output_is_one_error(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    executor = Scripted(["{not json"])
    record = run_localization(
        [case],
        tmp_path / "out",
        execute=True,
        executor=executor,
        discover_executable=_none,
    )
    assert len(executor.calls) == 1
    assert record.results.results[0].status == "ERROR"
    assert record.results.results[0].boxes == []
    assert record.retries == 0
    assert record.cases[0].calls == 1


def test_wrong_candidate_and_wrong_image_are_rejected(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    other = _response(case, "PROPOSED", [{"x": 2, "y": 2, "width": 8, "height": 6}])
    wrong_candidate = json.loads(other)
    wrong_candidate["candidate_id"] = "lc_" + "e" * 20
    wrong_image = json.loads(other)
    wrong_image["page_image_sha256"] = "cd" * 32
    response = ModelResponse(text="", model_id="mock", tool="mock")
    for payload, needle in (
        (json.dumps(wrong_candidate), "different candidate"),
        (json.dumps(wrong_image), "different page image"),
    ):
        proposal = interpret_model_output(case, payload, response)
        assert proposal.status == "ERROR"
        assert proposal.boxes == []
        assert needle in (proposal.explanation or "")
    mismatched = _proposal(
        "a" * 20,
        document_id=105,
        page_id="page_001",
        category="MISSING_CONTENT",
        pixels=(10, 4, 12, 6, 80, 90),
    )
    report = evaluate_locations(
        _reference(case),
        [mismatched],
        origin="mock",
    )
    assert report.cases[0].outcome == "ERROR"
    assert report.cases[0].best_iou is None
    assert report.live_model_executed is False


def test_coordinates_and_multiple_boxes(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    boxes = [
        {"x": 1, "y": 2, "width": 10, "height": 8},
        {"x": 12, "y": 3, "width": 9, "height": 7},
    ]
    executor = Scripted([_response(case, "PROPOSED", boxes)])
    record = run_localization(
        [case],
        tmp_path / "out",
        execute=True,
        executor=executor,
        discover_executable=_none,
    )
    proposal = record.results.results[0]
    assert proposal.status == "PROPOSED"
    assert len(proposal.boxes) == 2
    outside = interpret_model_output(
        case,
        _response(case, "PROPOSED", [{"x": 30, "y": 2, "width": 20, "height": 4}]),
        ModelResponse(text="", model_id="mock", tool="mock"),
    )
    full_page = interpret_model_output(
        case,
        _response(case, "PROPOSED", [{"x": 0, "y": 0, "width": 40, "height": 30}]),
        ModelResponse(text="", model_id="mock", tool="mock"),
    )
    assert outside.status == "ERROR"
    assert outside.boxes == []
    assert full_page.status == "ERROR"
    assert full_page.boxes == []


def test_ambiguous_and_not_found_stay_unscored_as_locations(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    reference = ConfirmedLocationReference(
        source_proposal_sha256=_SHA,
        locations=[_with_boxes(case, [_box(0, 0, 100, 100)])],
    )
    ambiguous = interpret_model_output(
        case,
        _response(
            case,
            "AMBIGUOUS",
            [
                {"x": 0, "y": 0, "width": 10, "height": 10},
                {"x": 20, "y": 10, "width": 8, "height": 6},
            ],
        ),
        ModelResponse(text="", model_id="mock", tool="mock"),
    )
    missing = interpret_model_output(
        case,
        _response(case, "NOT_FOUND", []),
        ModelResponse(text="", model_id="mock", tool="mock"),
    )
    thin = interpret_model_output(
        case,
        _response(case, "AMBIGUOUS", [{"x": 1, "y": 1, "width": 4, "height": 4}]),
        ModelResponse(text="", model_id="mock", tool="mock"),
    )
    boxed_absence = interpret_model_output(
        case,
        _response(case, "NOT_FOUND", [{"x": 1, "y": 1, "width": 4, "height": 4}]),
        ModelResponse(text="", model_id="mock", tool="mock"),
    )
    assert ambiguous.status == "AMBIGUOUS"
    assert len(ambiguous.boxes) == 2
    assert missing.status == "NOT_FOUND"
    assert missing.boxes == []
    assert thin.status == "ERROR"
    assert boxed_absence.status == "ERROR"
    stored = VisualProposal(
        candidate_id=case.candidate_id,
        dispute_id=case.dispute_id,
        document_id=case.document_id,
        page_id=case.page_id,
        page_image_sha256=case.page_image_sha256,
        page_width=case.page_width,
        page_height=case.page_height,
        dispute_category=case.dispute_category,
        status="PROPOSED",
        method="MODEL_PROPOSAL",
        provenance=ambiguous.provenance,
        boxes=[_box(0, 0, 100, 100), _box(50, 0, 100, 100)],
        explanation="unverified",
    )
    aligned = ambiguous.model_validate(
        ambiguous.model_dump()
        | {"boxes": [_box(0, 0, 100, 100), _box(50, 0, 100, 100)]}
    )
    report = evaluate_locations(reference, [aligned], origin="mock")
    assert report.cases[0].outcome == "AMBIGUOUS"
    assert report.cases[0].best_iou == "1.000000"
    overlap = evaluate_locations(reference, [stored], origin="mock")
    assert overlap.cases[0].outcome == "SCORED"
    assert overlap.cases[0].best_iou == "1.000000"
    assert overlap.cases[0].pairs[1].iou == "0.333333"
    absent = evaluate_locations(reference, [missing], origin="mock")
    assert absent.cases[0].outcome == "NOT_FOUND"
    assert absent.cases[0].best_iou is None
    assert absent.cases[0].pairs == []


def test_no_retry_no_overwrite_and_no_source_mutation(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    case = _case(image)
    before = image.read_bytes()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    failing = Scripted(error=LocalizationRunError("vision tool failed"))
    output = tmp_path / "out"
    record = run_localization(
        [case],
        output,
        execute=True,
        executor=failing,
        discover_executable=_none,
    )
    assert len(failing.calls) == 1
    assert record.results.results[0].status == "ERROR"
    assert "vision tool failed" in (record.results.results[0].explanation or "")
    saved = (output / "localization_run.json").read_bytes()
    again = Scripted(["{}"])
    with pytest.raises(ReviewIntegrityError):
        run_localization(
            [case],
            output,
            execute=True,
            executor=again,
            discover_executable=_none,
        )
    assert again.calls == []
    assert (output / "localization_run.json").read_bytes() == saved
    assert image.read_bytes() == before
    assert sentinel.read_text(encoding="utf-8") == "keep"
    dry = tmp_path / "dry"
    noisy = Scripted(error=AssertionError("dry run called a model"))
    plan = run_localization(
        [case],
        dry,
        execute=False,
        executor=noisy,
        discover_executable=_none,
    )
    assert noisy.calls == []
    assert plan.model_called is False
    assert plan.reference_boxes_included is False
    assert plan.max_requests_per_case == 1
    assert not (dry / "localization_run.json").exists()
    protected = tmp_path / "reviews"
    protected.mkdir()
    kept = protected / "review.json"
    kept.write_text("sealed", encoding="utf-8")
    with pytest.raises(ReviewPathError):
        run_localization(
            [case],
            protected / "nested",
            execute=False,
            discover_executable=_none,
            protected_roots=(protected,),
        )
    assert kept.read_text(encoding="utf-8") == "sealed"
    bad = case.model_copy(update={"page_image_sha256": "cd" * 32})
    called = Scripted(["{}"])
    with pytest.raises(ReviewIntegrityError):
        run_localization(
            [bad],
            tmp_path / "bad",
            execute=True,
            executor=called,
            discover_executable=_none,
        )
    assert called.calls == []
    assert not (tmp_path / "bad").exists()


def test_evaluation_is_deterministic_and_inspection_is_separate(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    reference = ConfirmedLocationReference(
        source_proposal_sha256=_SHA,
        locations=[_with_boxes(case, [_box(0, 0, 200, 100)])],
    )
    proposal = VisualProposal(
        candidate_id=case.candidate_id,
        dispute_id=case.dispute_id,
        document_id=case.document_id,
        page_id=case.page_id,
        page_image_sha256=case.page_image_sha256,
        page_width=case.page_width,
        page_height=case.page_height,
        dispute_category=case.dispute_category,
        status="PROPOSED",
        method="MODEL_PROPOSAL",
        provenance=ProposalProvenance(
            image_sha256=case.page_image_sha256,
            method_version="reproducible-localization-pilot-1",
            model_id="mock",
            tool="mock",
            request_sha256=_SHA,
            executed=True,
        ),
        boxes=[_box(100, 0, 200, 100)],
        explanation="unverified",
    )
    first = evaluate_locations(reference, [proposal], origin="mock")
    second = evaluate_locations(reference, [proposal], origin="mock")
    assert canonical_json(first) == canonical_json(second)
    assert first.cases[0].best_iou == "0.333333"
    assert first.reading_note.startswith("Geometric overlap does not")
    assert first.live_model_executed is False
    inspected = evaluate_locations(
        reference,
        [proposal],
        origin="prior_proposal",
        images={case.page_image_sha256: Path(case.image_path)},
        inspection_dir=tmp_path / "inspection",
    )
    assert inspected.live_model_executed is False
    crop = tmp_path / "inspection" / f"{case.candidate_id}_reference_crop_0.png"
    assert crop.is_file()
    width, height, _rows, _bpp = decode_png(crop.read_bytes())
    assert (width, height) != (case.page_width, case.page_height)
    assert (tmp_path / "inspection" / f"{case.candidate_id}_page_quarter.png").is_file()


def test_codex_approval_flag_precedes_exec(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    schema = tmp_path / "schema.json"
    last = tmp_path / "last.txt"
    argv = codex_exec_argv(
        executable=Path("/opt/codex"),
        image=image,
        schema_path=schema,
        last_message=last,
        prompt="locate only",
        path_for_tool=str,
    )
    assert argv == [
        "/opt/codex",
        "-a",
        "never",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--json",
        "--color",
        "never",
        "-s",
        "read-only",
        "-i",
        str(image),
        "--output-schema",
        str(schema),
        "-o",
        str(last),
        "locate only",
    ]


def test_codex_command_is_one_ephemeral_image_request(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    schema = tmp_path / "schema.json"
    last = tmp_path / "last.txt"
    reference = tmp_path / "reference.json"
    argv = codex_exec_argv(
        executable=Path("/opt/codex"),
        image=image,
        schema_path=schema,
        last_message=last,
        prompt="locate only",
        path_for_tool=str,
    )
    assert argv[:4] == ["/opt/codex", "-a", "never", "exec"]
    assert "--ephemeral" in argv
    assert "--ignore-user-config" in argv
    assert "--resume" not in argv
    assert "resume" not in argv
    assert argv.count("-i") == 1
    assert str(image) in argv
    assert str(reference) not in argv
    assert "-m" not in argv
    assert "--output-schema" in argv
    script = tmp_path / "codex"
    counter = tmp_path / "count.txt"
    payload = json.dumps(
        {
            "candidate_id": "lc_" + "a" * 20,
            "page_image_sha256": "pending",
            "status": "NOT_FOUND",
            "boxes": [],
        }
    )
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"counter = Path({str(counter)!r})\n"
        "counter.write_text((counter.read_text() if counter.exists() else '') + 'x')\n"
        "out = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        f"out.write_text({payload!r})\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    case = _case(image)
    fixed = json.dumps(
        {
            "candidate_id": case.candidate_id,
            "page_image_sha256": case.page_image_sha256,
            "status": "NOT_FOUND",
            "boxes": [],
        }
    )
    script.write_text(
        script.read_text(encoding="utf-8").replace(payload, fixed), encoding="utf-8"
    )
    from transcription_review_agent.review.reproducible_localization import (
        CodexCliRunner,
    )

    runner = CodexCliRunner(
        executable=script,
        output_dir=tmp_path / "live",
        timeout_seconds=5,
    )
    record = run_localization(
        [case],
        tmp_path / "live",
        execute=True,
        executor=runner,
        discover_executable=_none,
    )
    assert counter.read_text(encoding="utf-8") == "x"
    assert record.results.results[0].status == "NOT_FOUND"
    prompt = render_prompt(case)
    assert_reference_isolated(prompt, _reference(case))


def test_candidate_filter_keeps_pilot_order(tmp_path: Path) -> None:
    first = _case(tmp_path / "a.png", ident="a" * 20)
    second = _case(
        tmp_path / "b.png", ident="c" * 20, document_id=104, page_id="page_002"
    )
    chosen = select_cases([first, second], [second.candidate_id])
    assert [item.candidate_id for item in chosen] == [second.candidate_id]
    assert select_cases([first, second], None) == [first, second]
    with pytest.raises(ReviewUsageError):
        select_cases([first], [second.candidate_id])
    with pytest.raises(ReviewUsageError):
        select_cases([first], [first.candidate_id, first.candidate_id])


def test_codex_schema_omits_draft_keywords_and_numeric_bounds() -> None:
    schema = json.loads(_OUTPUT_SCHEMA)
    assert "$schema" not in schema
    assert "minimum" not in _OUTPUT_SCHEMA
    assert "maximum" not in _OUTPUT_SCHEMA
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    box = schema["properties"]["boxes"]["items"]
    assert set(box["required"]) == set(box["properties"])


def test_fenced_json_is_validated_and_utf16_decodes(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    payload = _response(case, "PROPOSED", [{"x": 1, "y": 1, "width": 4, "height": 4}])
    response = ModelResponse(text="", model_id="mock", tool="mock")
    fenced = interpret_model_output(
        case, "prefix\n```json\n" + payload + "\n```\n", response
    )
    assert fenced.status == "PROPOSED"
    assert len(fenced.boxes) == 1
    assert extract_model_json(payload) == payload.strip()
    decoded = decode_model_payload(("\ufeff" + payload).encode("utf-8"))
    again = interpret_model_output(case, decoded, response)
    assert again.status == "PROPOSED"
    utf16 = decode_model_payload(payload.encode("utf-16"))
    assert interpret_model_output(case, utf16, response).status == "PROPOSED"


def test_windows_executable_copies_the_page_onto_a_fresh_stage(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    schema = tmp_path / "model_output_schema.json"
    schema.write_text(_OUTPUT_SCHEMA, encoding="utf-8")
    last = tmp_path / "calls" / f"{case.candidate_id}.last.txt"
    last.parent.mkdir()
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"")
    image_path, schema_path, message_path, workdir = prepare_tool_paths(
        executable=executable,
        image=Path(case.image_path),
        schema_path=schema,
        last_message=last,
        stage_root=tmp_path / "stage",
    )
    assert image_path.read_bytes() == Path(case.image_path).read_bytes()
    assert image_path != Path(case.image_path)
    assert schema_path.read_text(encoding="utf-8") == _OUTPUT_SCHEMA
    assert message_path.parent == workdir
    assert workdir == image_path.parent
    with pytest.raises(LocalizationRunError, match="refusing to reuse"):
        prepare_tool_paths(
            executable=executable,
            image=Path(case.image_path),
            schema_path=schema,
            last_message=last,
            stage_root=tmp_path / "stage",
        )


def test_failed_codex_process_keeps_one_stderr_log(tmp_path: Path) -> None:
    script = tmp_path / "codex"
    script.write_text(
        "#!/usr/bin/env python3\nimport sys\n"
        "sys.stderr.write('sandbox failed\\n')\nraise SystemExit(2)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    case = _case(tmp_path / "page.png")
    from transcription_review_agent.review.reproducible_localization import (
        CodexCliRunner,
    )

    runner = CodexCliRunner(
        executable=script,
        output_dir=tmp_path / "live",
        timeout_seconds=5,
    )
    record = run_localization(
        [case],
        tmp_path / "live",
        execute=True,
        executor=runner,
        discover_executable=_none,
    )
    assert record.results.results[0].status == "ERROR"
    assert "sandbox failed" in (record.results.results[0].explanation or "")
    stderr = tmp_path / "live" / "calls" / f"{case.candidate_id}.stderr.txt"
    assert "sandbox failed" in stderr.read_text(encoding="utf-8")
    assert (tmp_path / "live" / "calls" / f"{case.candidate_id}.argv.txt").is_file()


def test_contradictory_confirmed_pixels_fail_closed(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    location = _reference(case).locations[0]
    consistent = location.model_dump()
    contradictory = json.loads(json.dumps(consistent))
    contradictory["measured_pixels"][0]["x"] = 0
    contradictory["measured_pixels"][0]["y"] = 0
    contradictory["measured_pixels"][0]["width"] = 1
    contradictory["measured_pixels"][0]["height"] = 1
    with pytest.raises(PydanticValidationError, match="not consistent"):
        ConfirmedLocation.model_validate(contradictory)
    payload = json.dumps(
        {
            "source_proposal_sha256": _SHA,
            "locations": [contradictory],
        }
    )
    with pytest.raises(PydanticValidationError, match="not consistent"):
        ConfirmedLocationReference.model_validate_json(payload)


def test_one_pixel_edge_shifts_do_not_accumulate(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    exact = normalize_measured_pixels(
        x=9, y=10, width=18, height=4, page_width=40, page_height=30
    )

    def location(pixels: tuple[int, int, int, int]) -> dict[str, object]:
        data = _reference(case).locations[0].model_dump()
        data["boxes"] = [exact.model_dump()]
        measured = data["measured_pixels"][0]
        measured["x"], measured["y"], measured["width"], measured["height"] = pixels
        return data

    ConfirmedLocation.model_validate(location((9, 10, 18, 4)))
    ConfirmedLocation.model_validate(location((9, 10, 19, 4)))
    ConfirmedLocation.model_validate(location((9, 10, 18, 5)))
    with pytest.raises(PydanticValidationError, match="not consistent"):
        ConfirmedLocation.model_validate(location((10, 10, 19, 4)))
    with pytest.raises(PydanticValidationError, match="not consistent"):
        ConfirmedLocation.model_validate(location((9, 11, 18, 5)))


def test_windows_stage_is_removed_when_setup_fails(tmp_path: Path) -> None:
    from transcription_review_agent.review.reproducible_localization import (
        CodexCliRunner,
    )

    case = _case(tmp_path / "page.png")
    stage_root = tmp_path / "stage"
    unrelated = stage_root / "unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    output = tmp_path / "live"
    output.mkdir()
    (output / "model_output_schema.json").write_text(_OUTPUT_SCHEMA, encoding="utf-8")
    executable = tmp_path / "codex.exe"
    executable.write_bytes(b"")

    def fail_conversion(path: Path) -> str:
        raise LocalizationRunError(f"path conversion failed for {path.name}")

    runner = CodexCliRunner(
        executable=executable,
        output_dir=output,
        timeout_seconds=5,
        path_for_tool=fail_conversion,
        stage_root=stage_root,
    )
    with pytest.raises(LocalizationRunError, match="path conversion failed"):
        runner.run(_call(case))
    assert not (stage_root / f"{case.candidate_id}.last").exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert Path(case.image_path).read_bytes()

    (output / "calls").mkdir(exist_ok=True)
    argv_path = output / "calls" / f"{case.candidate_id}.argv.txt"
    argv_path.write_text("already recorded\n", encoding="utf-8")
    runner = CodexCliRunner(
        executable=executable,
        output_dir=output,
        timeout_seconds=5,
        path_for_tool=str,
        stage_root=stage_root,
    )
    with pytest.raises(ReviewIntegrityError, match="refusing to overwrite"):
        runner.run(_call(case))
    assert argv_path.read_text(encoding="utf-8") == "already recorded\n"
    assert not (stage_root / f"{case.candidate_id}.last").exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_confirmation_must_match_the_copied_boxes(tmp_path: Path) -> None:
    sidecar = _pilot_sidecar()
    before = canonical_json(sidecar).encode("utf-8")
    confirmations = _confirmations(sidecar)
    changed = confirmations[0].model_copy(
        update={
            "boxes": [
                confirmations[0]
                .boxes[0]
                .model_copy(update={"x": confirmations[0].boxes[0].x + 50})
            ]
        }
    )
    with pytest.raises(
        ReviewIntegrityError, match="does not match the copied proposal"
    ):
        build_confirmed_reference(before, sidecar, [changed, *confirmations[1:]])
    repeated = [confirmations[0], *confirmations]
    with pytest.raises(ReviewIntegrityError, match="repeats a candidate"):
        build_confirmed_reference(before, sidecar, repeated)


def test_saved_pilot_reference_is_not_rewritten(tmp_path: Path) -> None:
    path = Path("/tmp/transcription_confirmed_location_reference_v1/reference.json")
    if not path.is_file():
        pytest.skip("saved pilot reference is not on this machine")
    before = path.read_bytes()
    loaded = ConfirmedLocationReference.model_validate_json(before)
    assert len(loaded.locations) == 3
    assert loaded.human_verified_transcription_count == 0
    assert {item.transcription_verified for item in loaded.locations} == {False}
    assert {item.confirmation for item in loaded.locations} == {
        "USER_VISUAL_CONFIRMATION"
    }
    proposals = tmp_path / "proposals.json"
    proposals.write_text(canonical_json(_pilot_sidecar()), encoding="utf-8")
    kept = tmp_path / "reference"
    kept.mkdir()
    (kept / "reference.json").write_text("prior pilot fixture", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError, match="overwrite"):
        write_reference_files(
            proposals,
            kept,
            confirmations=_confirmations(_pilot_sidecar()),
        )
    assert path.read_bytes() == before
    assert (kept / "reference.json").read_text(
        encoding="utf-8"
    ) == "prior pilot fixture"


def test_windows_stage_is_removed_on_success_failure_and_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transcription_review_agent.review import (
        reproducible_localization as runner_module,
    )
    from transcription_review_agent.review.reproducible_localization import (
        CodexCliRunner,
    )

    case = _case(tmp_path / "page.png")
    image_before = Path(case.image_path).read_bytes()
    stage_root = tmp_path / "stage"
    unrelated = stage_root / "unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    output = tmp_path / "live"
    output.mkdir()
    (output / "model_output_schema.json").write_text(_OUTPUT_SCHEMA, encoding="utf-8")
    preexisting = stage_root / f"{case.candidate_id}.last"
    preexisting.mkdir()
    (preexisting / "keep.txt").write_text("preexisting", encoding="utf-8")
    runner = CodexCliRunner(
        executable=tmp_path / "codex.exe",
        output_dir=output,
        timeout_seconds=1,
        path_for_tool=str,
        stage_root=stage_root,
    )
    with pytest.raises(LocalizationRunError, match="refusing to reuse"):
        runner.run(_call(case))
    assert (preexisting / "keep.txt").read_text(encoding="utf-8") == "preexisting"
    for child in preexisting.iterdir():
        child.unlink()
    preexisting.rmdir()

    def exe(body: str) -> Path:
        path = tmp_path / "codex.exe"
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
        return path

    success = exe(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "out.write_text('{\"status\":\"NOT_FOUND\"}\\n', encoding='utf-8')\n"
        'sys.stdout.write(\'{"model":"mock-model"}\\n\')\n'
    )
    runner = CodexCliRunner(
        executable=success,
        output_dir=output,
        timeout_seconds=5,
        path_for_tool=str,
        stage_root=stage_root,
    )
    response = runner.run(_call(case))
    assert response.model_id == "mock-model"
    assert not (stage_root / f"{case.candidate_id}.last").exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert Path(case.image_path).read_bytes() == image_before
    saved = output / "calls" / f"{case.candidate_id}.last.txt"
    assert "NOT_FOUND" in saved.read_text(encoding="utf-8")

    output_fail = tmp_path / "fail"
    output_fail.mkdir()
    (output_fail / "model_output_schema.json").write_text(
        _OUTPUT_SCHEMA, encoding="utf-8"
    )
    failing = exe(
        "#!/usr/bin/env python3\nimport sys\n"
        "sys.stderr.write('sandbox failed\\n')\nraise SystemExit(2)\n"
    )
    failed = CodexCliRunner(
        executable=failing,
        output_dir=output_fail,
        timeout_seconds=5,
        path_for_tool=str,
        stage_root=stage_root,
    )
    with pytest.raises(LocalizationRunError, match="sandbox failed"):
        failed.run(_call(case))
    assert not (stage_root / f"{case.candidate_id}.last").exists()
    assert "sandbox failed" in (
        output_fail / "calls" / f"{case.candidate_id}.stderr.txt"
    ).read_text(encoding="utf-8")

    output_timeout = tmp_path / "timeout"
    output_timeout.mkdir()
    (output_timeout / "model_output_schema.json").write_text(
        _OUTPUT_SCHEMA, encoding="utf-8"
    )
    sleeping = exe("#!/usr/bin/env python3\nimport time\ntime.sleep(30)\n")
    timed = CodexCliRunner(
        executable=sleeping,
        output_dir=output_timeout,
        timeout_seconds=1,
        path_for_tool=str,
        stage_root=stage_root,
    )
    with pytest.raises(LocalizationRunError, match="timed out"):
        timed.run(_call(case))
    assert not (stage_root / f"{case.candidate_id}.last").exists()
    assert (output_timeout / "calls" / f"{case.candidate_id}.stdout.txt").is_file()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"

    monkeypatch.setattr(runner_module, "MAX_PROCESS_OUTPUT_BYTES", 1024)
    output_bound = tmp_path / "bound"
    counter = tmp_path / "count.txt"
    flooding = exe(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"Path({str(counter)!r}).write_text('x', encoding='utf-8')\n"
        "sys.stdout.write('x' * 200000)\n"
    )
    bounded = CodexCliRunner(
        executable=flooding,
        output_dir=output_bound,
        timeout_seconds=5,
        path_for_tool=str,
        stage_root=stage_root,
    )
    record = run_localization(
        [case],
        output_bound,
        execute=True,
        executor=bounded,
        discover_executable=_none,
    )
    assert counter.read_text(encoding="utf-8") == "x"
    assert record.results.results[0].status == "ERROR"
    assert "capture limit" in (record.results.results[0].explanation or "")
    stdout = (output_bound / "calls" / f"{case.candidate_id}.stdout.txt").read_bytes()
    assert len(stdout) <= 1024
    assert not (stage_root / f"{case.candidate_id}.last").exists()
    assert counter.read_text(encoding="utf-8") == "x"


def test_oversized_model_text_is_one_error_without_a_retry(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    huge = "x" * (MAX_MODEL_RESPONSE_BYTES + 1)
    response = ModelResponse(text="", model_id="mock", tool="mock")
    proposal = interpret_model_output(case, huge, response)
    assert proposal.status == "ERROR"
    assert proposal.boxes == []
    assert "capture limit" in (proposal.explanation or "")
    with pytest.raises(LocalizationRunError, match="capture limit"):
        decode_model_payload(b"y" * (MAX_MODEL_RESPONSE_BYTES + 1))
    with pytest.raises(LocalizationRunError, match="capture limit"):
        extract_model_json(huge)


def test_failed_stage_copy_removes_only_the_directory_it_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transcription_review_agent.review import (
        reproducible_localization as runner_module,
    )

    case = _case(tmp_path / "page.png")
    schema = tmp_path / "model_output_schema.json"
    schema.write_text(_OUTPUT_SCHEMA, encoding="utf-8")
    last = tmp_path / "calls" / f"{case.candidate_id}.last.txt"
    last.parent.mkdir()
    stage_root = tmp_path / "stage"
    unrelated = stage_root / "unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")

    def fail_copy(source: Path, destination: Path) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(runner_module.shutil, "copyfile", fail_copy)
    with pytest.raises(OSError, match="copy failed"):
        prepare_tool_paths(
            executable=tmp_path / "codex.exe",
            image=Path(case.image_path),
            schema_path=schema,
            last_message=last,
            stage_root=stage_root,
        )
    assert not (stage_root / f"{case.candidate_id}.last").exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert Path(case.image_path).read_bytes()


def test_missing_codex_writes_nothing(tmp_path: Path) -> None:
    case = _case(tmp_path / "page.png")
    with pytest.raises(ReviewUsageError):
        run_localization(
            [case],
            tmp_path / "out",
            execute=True,
            discover_executable=_none,
        )
    assert not (tmp_path / "out").exists()
