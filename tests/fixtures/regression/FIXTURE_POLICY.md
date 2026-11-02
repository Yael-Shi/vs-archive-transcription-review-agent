# Regression fixture policy (Track A, slice A0)

Machine-readable twin: `fixture_policy.json` (checked by the test suite).

## Exposure review (2026-11-02)

| Question | Finding |
| --- | --- |
| Repository visibility | **Unverified.** Remote is `github.com/Yael-Shi/vs-archive-transcription-review-agent`; the GitHub API was unreachable from the implementation environment. Treat as potentially public. |
| Forks / mirrors | Unknown. |
| Licensing, privacy or publication limits on archival transcriptions and names | Not reviewed. Unknown. |
| Git history retention acceptable | Not approved. Committed bytes are effectively permanent. |

## Decision

No archival content (text, excerpts, names, images, provider outputs, or
hashes/IDs derived from archival text) for documents 104 or 105 is committed
to Git. No approval has been recorded. `approved_by` in `fixture_policy.json`
is `null`; changing `archival_content_in_git` to `true` requires a named
approver, a reference, and a repository-visibility confirmation.

## Two tiers

1. **Committed (always runs).** `synthetic_pair/` holds fictional GPT/Grok
   first-pass files and `expected_comparison.json`. They cover exact string
   preservation, alignment kinds, every dispute category, substantive flags
   and stable IDs. Run by `tests/test_review_regression_fixtures.py`.
2. **Authorized archival (opt-in, outside Git).** Run by
   `tests/test_review_archival_regression.py` against a directory you provide.

### Authorized archival run

```bash
# Normal runs: variable unset -> tests are SKIPPED with reason
# "archival fixtures not authorized/unavailable" (not a pass, not a failure).

# Authorized run: point at a pack kept outside the repository.
VS_ARCHIVAL_FIXTURE_DIR=/secure/path/vs_archival_pack \
  .venv/bin/pytest tests/test_review_archival_regression.py

# Require the pack (fail instead of skip if it is not configured):
VS_ARCHIVAL_FIXTURES_REQUIRED=1 VS_ARCHIVAL_FIXTURE_DIR=... .venv/bin/pytest ...
```

Once a run is requested (`VS_ARCHIVAL_FIXTURE_DIR` set, or
`VS_ARCHIVAL_FIXTURES_REQUIRED=1`), a missing directory, missing manifest,
missing file, unreadable file or SHA-256 mismatch **fails** the test. It never
skips.

The pack directory must be outside the repository (the loader refuses a
directory inside it) and must contain `archival_fixture_manifest.json`:

```json
{
  "schema_version": "a0-archival-pack-v1",
  "authorized_by": "<name>",
  "authorization_reference": "<ticket, message or date>",
  "stale_dispute_ids": ["dsp_..."],
  "documents": {
    "104": {
      "gpt":  {"path": "104/transcription_gpt.json",  "sha256": "<64 hex>"},
      "grok": {"path": "104/transcription_grok.json", "sha256": "<64 hex>"},
      "expected": {
        "disputes": 138,
        "substantive": 86,
        "category_counts": {"WORDING": 60},
        "dispute_ids": ["dsp_..."],
        "required_rows": [
          {"category": "DATE_OR_NUMBER", "gpt_contains": "1946",
           "grok_contains": "1944", "substantive": true}
        ]
      }
    },
    "105": {"...": "same shape"}
  }
}
```

* `path` is relative to the pack directory and may not escape it.
* `expected.dispute_ids` and `stale_dispute_ids` are optional. Provide
  `dispute_ids` only if generated from the current regeneration; `stale_dispute_ids`
  lists old v001 IDs that must not reappear as expectations.
* Current regenerated reference counts (as of this record): document 104 =
  138 disputes / 86 substantive; document 105 = 105 / 82. Old local 104 v001
  comparison (156 / 141) is stale.
* Copy files into the pack by hand (read-only from `outputs/`); the test suite
  never writes to `outputs/`, `state/`, ledgers or bundles.

## Stale IDs

Dispute IDs are `dsp_` + first 16 hex of SHA-256 of
`category|page|region|gpt_text|grok_text`. An ID is valid only if it equals
that formula applied to its row. Old document 104 v001 IDs are never copied into
fixtures; where text is unchanged, the regenerated ID legitimately equals the old one,
and where text changed the old ID no longer exists.
