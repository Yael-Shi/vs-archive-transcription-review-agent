"""Slice A0: authorized 104/105 archival regression (opt-in, outside Git).

* Variable not set      -> SKIPPED ("not authorized/unavailable"); this is not a pass.
* Authorized run asked  -> a missing, changed or invalid pack FAILS; it never skips.

See tests/fixtures/regression/FIXTURE_POLICY.md.
"""

from __future__ import annotations

import pytest
from regression_fixtures import (
    ArchivalFixtureError,
    ArchivalFixtureUnavailable,
    ArchivalPack,
    check_document_expectations,
    load_archival_pack,
)

REQUIRED_DOCUMENTS = ("104", "105")


def _pack() -> ArchivalPack:
    try:
        return load_archival_pack()
    except ArchivalFixtureUnavailable as exc:
        pytest.skip(str(exc))
    except ArchivalFixtureError as exc:
        pytest.fail(
            f"authorized archival fixture run requested but unusable: {exc}",
            pytrace=False,
        )


@pytest.mark.parametrize("document_id", REQUIRED_DOCUMENTS)
def test_authorized_archival_comparison_regression(document_id: str) -> None:
    pack = _pack()
    document = pack.documents.get(document_id)
    if document is None:
        pytest.fail(
            f"authorized pack does not contain document {document_id}", pytrace=False
        )
    check_document_expectations(
        document.pair, document.expected, document.stale_dispute_ids
    )
