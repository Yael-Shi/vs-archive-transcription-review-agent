"""Local ``review-next`` operations for an already sealed batch.

These commands prepare review work, inspect status, validate a dropped
provider file without writing state, accept that file, recover one invalid
challenge attempt, explicitly close the challenge stage, and finalize a
document. They do not scan VS Archive, open a socket, or write a sealed bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transcription_review_agent.errors import ConfigError
from transcription_review_agent.review.config import load_review_config
from transcription_review_agent.review.controller import CLOSURE_MODES, ReviewController
from transcription_review_agent.review.errors import (
    ReviewOutputRejected,
    ReviewPolicyError,
    ReviewUsageError,
)


def review_next(config: object | None = None, argv: list[str] | None = None) -> None:
    """Run one explicit local review operation.

    ``config`` is ignored. Discovery configuration is not loaded, and an empty
    argument list does not scan or download anything.
    """

    del config
    if not argv:
        raise ReviewUsageError(
            "review-next requires an operation. No VS Archive request is made."
        )
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise ReviewUsageError(
            "review-next usage was not recognized. No VS Archive request is made."
        ) from exc
    try:
        review_config = load_review_config()
    except ConfigError as exc:
        raise ReviewUsageError(f"{exc}. No VS Archive request is made.") from exc
    controller = ReviewController(
        batches_dir=Path(args.batches_dir),
        reviews_dir=Path(args.reviews_dir),
        config=review_config,
    )
    if args.operation == "prepare":
        _prepare(controller, args)
        return
    if args.operation == "status":
        status = controller.status(args.batch_id, args.document_id)
        print(
            json.dumps(
                {
                    "document_id": status.document_id,
                    "execution_state": status.execution_state,
                    "quality": status.quality,
                    "providers": status.providers,
                    "finalized": status.finalized,
                    "challenge_rounds": status.challenge_rounds,
                    "challenge_closure": status.challenge_closure,
                },
                sort_keys=True,
            )
        )
        return
    if args.operation == "accept":
        if args.provider is None:
            raise ReviewUsageError(
                "accept requires --provider. No VS Archive request is made."
            )
        controller.accept(
            args.batch_id,
            args.document_id,
            args.provider,
            stage=args.stage,
            round_index=args.round_index,
        )
        return
    if args.operation == "validate":
        if args.provider is None:
            raise ReviewUsageError(
                "validate requires --provider. No VS Archive request is made."
            )
        report = controller.validate_output(
            args.batch_id,
            args.document_id,
            args.provider,
            stage=args.stage,
            round_index=args.round_index,
        )
        print(json.dumps(report, sort_keys=True))
        if not report["output_valid"]:
            failure = report["failures"]
            assert isinstance(failure, list) and failure
            first = failure[0]
            assert isinstance(first, dict)
            raise ReviewOutputRejected(str(first["reason"]), str(first["message"]))
        return
    if args.operation == "recover-invalid-output":
        if args.provider is None:
            raise ReviewUsageError(
                "recover-invalid-output requires --provider. "
                "No VS Archive request is made."
            )
        if args.stage != "challenge":
            raise ReviewUsageError(
                "recover-invalid-output requires --stage challenge. "
                "No VS Archive request is made."
            )
        path = controller.recover_invalid_output(
            args.batch_id,
            args.document_id,
            args.provider,
            stage=args.stage,
            round_index=args.round_index,
        )
        print(path)
        return
    if args.operation == "close-challenge":
        if args.closure_mode is None or args.reason is None:
            raise ReviewUsageError(
                "close-challenge requires --closure-mode and --reason. "
                "No VS Archive request is made."
            )
        controller.close_challenge(
            args.batch_id,
            args.document_id,
            mode=args.closure_mode,
            reason=args.reason,
        )
        return
    controller.finalize(
        args.batch_id,
        args.document_id,
        provisional=args.provisional,
    )


def _prepare(controller: ReviewController, args: argparse.Namespace) -> None:
    controller.open_sealed_document(args.batch_id, args.document_id)
    providers = (args.provider,) if args.provider else ("gpt", "grok")
    evidence_path = Path(args.evidence_path) if args.evidence_path else None
    selected = list(args.dispute_id) if args.dispute_id else None
    for provider in providers:
        if args.stage == "first_pass":
            status = controller.status(args.batch_id, args.document_id)
            if status.providers[provider] not in {"NOT_STARTED", "RETRYABLE_FAILURE"}:
                continue
        try:
            controller.publish(
                args.batch_id,
                args.document_id,
                provider,
                stage=args.stage,
                evidence=args.evidence,
                evidence_path=evidence_path,
                selected_dispute_ids=selected,
            )
        except ReviewPolicyError as exc:
            message = str(exc)
            if (
                "already succeeded" in message
                or "quota-blocked" in message
                or "cannot be dispatched" in message
            ):
                continue
            raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="review-next", add_help=False)
    parser.add_argument(
        "operation",
        choices=(
            "prepare",
            "status",
            "validate",
            "accept",
            "recover-invalid-output",
            "close-challenge",
            "finalize",
        ),
    )
    parser.add_argument("--batches-dir", required=True)
    parser.add_argument("--reviews-dir", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--document-id", required=True, type=int)
    parser.add_argument("--provider", choices=("gpt", "grok"))
    parser.add_argument(
        "--stage", default="first_pass", choices=("first_pass", "challenge", "research")
    )
    parser.add_argument("--round-index", type=int)
    parser.add_argument("--evidence")
    parser.add_argument("--evidence-path")
    parser.add_argument(
        "--dispute-id",
        action="append",
        help="challenge stage only: select this substantive dispute ID (repeatable)",
    )
    parser.add_argument(
        "--closure-mode",
        choices=CLOSURE_MODES,
        help="close-challenge only: why the challenge stage ends",
    )
    parser.add_argument(
        "--reason",
        help="close-challenge only: the operator's explicit, recorded reason",
    )
    parser.add_argument("--provisional", action="store_true")
    return parser
