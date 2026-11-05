"""Conservative diplomatic transcription and the four final artifacts.

Historical identifications are copied into the report only. They are not
written into the diplomatic text.
"""

from __future__ import annotations

from transcription_review_agent.review.formatting import (
    content_is_unresolved,
    decision_selects_dispute,
    explicit_illegible_side,
    format_unresolved_alternatives,
    recorded_alternative_readings,
    render_aligned_span,
)
from transcription_review_agent.review.models import (
    ChallengeClosureRecord,
    ChallengeSelectionRecord,
    ComparisonFile,
    Dispute,
    FirstPassOutput,
    Identification,
    IndependenceRecord,
    PageTranscription,
    QualityLabel,
    ReconciledDecision,
    RegionAlignment,
    ReviewReport,
    SourceEvidence,
)
from transcription_review_agent.review.spans import same_span

_LABELS_HE = {
    QualityLabel.NO_MATERIAL_DISPUTES: "אין מחלוקות מהותיות",
    QualityLabel.RESOLVED_WITH_WARNINGS: "נפתר עם אזהרות",
    QualityLabel.UNRESOLVED_CONTENT: "תוכן שלא הוכרע",
    QualityLabel.INCOMPLETE_COVERAGE: "כיסוי חסר",
    QualityLabel.SINGLE_PROVIDER_PROVISIONAL: "דוח זמני מספק יחיד",
}


def render_transcription(
    *,
    page_ids: list[str],
    gpt: FirstPassOutput | None,
    grok: FirstPassOutput | None,
    comparison: ComparisonFile | None,
    decisions: list[ReconciledDecision],
) -> str:
    """Build diplomatic text without choosing an unresolved material reading."""

    if gpt is None and grok is None:
        raise ValueError("a transcription requires a provider result")
    by_id = {item.dispute_id: item for item in decisions}
    blocks: list[str] = []
    for page_id in page_ids:
        left = _find_page(gpt, page_id)
        right = _find_page(grok, page_id)
        if left is None or right is None:
            page = left or right
            if page is None:
                raise ValueError(f"missing page {page_id}")
            blocks.append(f"@@ {page_id}\n{_single_page(page)}")
            continue
        disputes = (
            []
            if comparison is None
            else [item for item in comparison.disputes if item.page_id == page_id]
        )
        alignments = (
            []
            if comparison is None
            else [item for item in comparison.alignments if item.page_id == page_id]
        )
        blocks.append(
            f"@@ {page_id}\n{_pair_page(left, right, disputes, by_id, alignments)}"
        )
    return "\n\n".join(blocks) + "\n"


def classify_quality(
    *,
    single_provider: bool,
    gpt: FirstPassOutput | None,
    grok: FirstPassOutput | None,
    comparison: ComparisonFile | None,
    decisions: list[ReconciledDecision] | None = None,
    challenge_closure: ChallengeClosureRecord | None = None,
    interim: bool = False,
) -> QualityLabel:
    """Quality is separate from execution. A finished run may stay unresolved.

    ``UNRESOLVED_CONTENT`` comes from comparison rows, decisions, closure, and
    recorded unresolved pages. Rendered marker strings are not consulted.
    """

    if single_provider:
        return QualityLabel.SINGLE_PROVIDER_PROVISIONAL
    if _coverage_gaps(gpt, grok):
        return QualityLabel.INCOMPLETE_COVERAGE
    if interim:
        return QualityLabel.UNRESOLVED_CONTENT
    if content_is_unresolved(comparison, decisions, challenge_closure):
        return QualityLabel.UNRESOLVED_CONTENT
    if comparison is not None and any(item.substantive for item in comparison.disputes):
        return QualityLabel.RESOLVED_WITH_WARNINGS
    return QualityLabel.NO_MATERIAL_DISPUTES


def coverage_gap_ids(
    gpt: FirstPassOutput | None,
    grok: FirstPassOutput | None,
) -> list[str]:
    return _coverage_gaps(gpt, grok)


def independence_records(
    identifications: list[Identification],
) -> list[IndependenceRecord]:
    records: list[IndependenceRecord] = []
    for item in identifications:
        rejected = item.uses_target_document_transcription
        records.append(
            IndependenceRecord(
                identification=item,
                independent=not rejected,
                corroboration_rejected=rejected,
            )
        )
    return records


def build_report(
    *,
    batch_id: str,
    document_id: int,
    quality: QualityLabel,
    page_ids: list[str],
    single_provider: bool,
    succeeded_provider: str | None,
    comparison: ComparisonFile | None,
    decisions: list[ReconciledDecision],
    identifications: list[Identification],
    review_completion: str,
    provider_failure_count: int,
    estimated_work_units: int,
    gpt: FirstPassOutput | None,
    grok: FirstPassOutput | None,
    source_evidence: list[SourceEvidence] | None = None,
    challenge_selections: list[ChallengeSelectionRecord] | None = None,
    challenge_closure: ChallengeClosureRecord | None = None,
) -> ReviewReport:
    unresolved = [] if comparison is None else list(comparison.unresolved_text_page_ids)
    return ReviewReport(
        batch_id=batch_id,
        document_id=document_id,
        review_completion=review_completion,  # type: ignore[arg-type]
        quality=quality,
        single_provider=single_provider,
        succeeded_provider=succeeded_provider,  # type: ignore[arg-type]
        coverage_gaps=coverage_gap_ids(gpt, grok),
        unresolved_text_page_ids=unresolved,
        disputes=[] if comparison is None else list(comparison.disputes),
        decisions=list(decisions),
        identifications=independence_records(identifications),
        provider_failure_count=provider_failure_count,
        estimated_work_units=estimated_work_units,
        page_ids=list(page_ids),
        source_evidence=[] if source_evidence is None else list(source_evidence),
        challenge_selections=(
            None if not challenge_selections else list(challenge_selections)
        ),
        challenge_closure=challenge_closure,
        alternative_readings=recorded_alternative_readings(gpt, grok) or None,
    )


def hebrew_summary(
    *,
    batch_id: str,
    document_id: int,
    language: str,
    page_count: int,
    quality: QualityLabel,
    substantive_count: int,
    coverage_gaps: list[str],
    single_provider: bool,
) -> str:
    """Concise Hebrew summary for a human reviewer."""

    gaps = "כן" if coverage_gaps else "לא"
    single = "כן" if single_provider else "לא"
    gap_pages = ", ".join(coverage_gaps) if coverage_gaps else "אין"
    return (
        "# סיכום ביקורת תמלול\n\n"
        f"- מסמך: {document_id}\n"
        f"- אצווה: {batch_id}\n"
        f"- שפה: {language}\n"
        f"- עמודים: {page_count}\n"
        f"- סיווג איכות: {_LABELS_HE[quality]}\n"
        f"- מחלוקות מהותיות: {substantive_count}\n"
        f"- כיסוי חסר: {gaps} ({gap_pages})\n"
        f"- ספק יחיד: {single}\n\n"
        "התמלול המוצע שומר על הכתיב הדיפלומטי. "
        "זיהוי היסטורי, אם נרשם, אינו משנה את הטקסט המקורי.\n"
    )


def _single_page(page: PageTranscription) -> str:
    body = page.diplomatic_transcription
    if not page.coverage.coverage_complete:
        body = "[coverage-incomplete]\n" + body
    return body


def _pair_page(
    left: PageTranscription,
    right: PageTranscription,
    disputes: list[Dispute],
    decisions: dict[str, ReconciledDecision],
    alignments: list[RegionAlignment],
) -> str:
    substantive = [item for item in disputes if item.substantive]
    order = next(
        (item for item in substantive if item.category.value == "READING_ORDER"),
        None,
    )
    if order is not None:
        decision = decisions.get(order.dispute_id)
        if not _reading_order_selected(decision, order):
            return format_unresolved_alternatives(
                left.diplomatic_transcription,
                right.diplomatic_transcription,
                left_illegible=explicit_illegible_side(
                    left,
                    [region.region_id for region in left.regions],
                    left.diplomatic_transcription,
                ),
                right_illegible=explicit_illegible_side(
                    right,
                    [region.region_id for region in right.regions],
                    right.diplomatic_transcription,
                ),
            )
    parts: list[str] = []
    if not left.coverage.coverage_complete or not right.coverage.coverage_complete:
        parts.append("[coverage-incomplete]")
    ordered = sorted(alignments, key=lambda item: item.sequence)
    for alignment in ordered:
        related = [
            item
            for item in substantive
            if item.category.value != "READING_ORDER" and same_span(item, alignment)
        ]
        parts.append(
            render_aligned_span(
                alignment.gpt_text,
                alignment.grok_text,
                related,
                decisions,
                left_illegible=explicit_illegible_side(
                    left, alignment.gpt_region_ids, alignment.gpt_text
                ),
                right_illegible=explicit_illegible_side(
                    right, alignment.grok_region_ids, alignment.grok_text
                ),
            )
        )
    # unresolved_text_present means a reading is still uncertain. Comparison
    # and quality already record that. It is not an unreadable region, so
    # the proposal does not add [unreadable]. A marker written in the
    # diplomatic text stays in that text.
    return "\n".join(part for part in parts if part != "")


def _reading_order_selected(
    decision: ReconciledDecision | None,
    dispute: Dispute,
) -> bool:
    """A resolved reading-order label keeps assembly on the aligned spans.

    The selected string for this category is the region-id sequence recorded
    on the dispute. It is not pasted into the diplomatic text. An unresolved
    or third reading still shows both provider transcriptions.
    """

    if decision is None or decision.decision == "THIRD_READING":
        return False
    return decision_selects_dispute(decision, dispute)


def _coverage_gaps(
    gpt: FirstPassOutput | None,
    grok: FirstPassOutput | None,
) -> list[str]:
    gaps: list[str] = []
    seen: set[str] = set()
    for output in (gpt, grok):
        if output is None:
            continue
        for page in output.pages:
            if not page.coverage.coverage_complete and page.page_id not in seen:
                gaps.append(page.page_id)
                seen.add(page.page_id)
    return gaps


def _find_page(
    output: FirstPassOutput | None, page_id: str
) -> PageTranscription | None:
    if output is None:
        return None
    for page in output.pages:
        if page.page_id == page_id:
            return page
    return None


def _region_text(page: PageTranscription, region_id: str) -> str:
    for region in page.regions:
        if region.region_id == region_id:
            return "\n".join(line.text for line in region.lines)
    return ""
