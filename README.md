# Transcription review agent

Read-only Pilot 01 controller for VS Archive transcription review. It will scan public English and French OCR documents, keep a durable scan cursor, and write immutable local bundles for up to the configured batch size (default five) reviewable documents per run.

This repository is separate from the VS Archive Django application. It does not import that project, write into that checkout, or send anything other than HTTP GET.

## Strict read-only boundary

Pilot 01 uses anonymous HTTP GET against:

- `/api/agent/v1/transcription-review/documents/scan`
- `/api/agent/v1/transcription-review/documents/<document_id>/review-bundle`
- presigned source URLs returned inside a `REVIEWABLE` bundle

The agent does not use AWS credentials, Django admin credentials, database credentials, GitHub or SSH credentials, or the VS Archive `.env`. Configuration is the public API origin plus local paths. Local SHA-256 digests of the exact UTF-8 transcription text and of each downloaded source file are authoritative. A server-supplied transcription hash is stored as provenance only.

No AI provider is part of this milestone.

## Pilot 01 scope

One run targets the configured batch size of `REVIEWABLE` documents (default five). Documents are atomic. Scan order is ascending `Document.id`.

| Bundle result | Scan cursor |
|---|---|
| HTTP 404 | Advance to that document id |
| `NON_ELIGIBLE` | Advance to that document id |
| `REVIEWABLE` | Advance when the document is selected. Bundling is a separate per-document phase |
| `BLOCKING_INTEGRITY_ERROR` | Stop the run. Do not move the cursor to or past that document |
| Any other HTTP status or outcome | Stop. Do not advance |

Discovery exhaustion with 1–4 reviewable documents is a valid short batch. Exhaustion with zero reviewable documents is a no-work result: the cursor is kept, no batch id is allocated, and no batch manifest is written. PDFs stay the original PDF bytes from the API. Rasterization is later.

`discover_next_batch` scans and selects documents. `bundle_selected_batch` downloads sources and publishes document bundles. `seal_selected_batch` writes the batch manifest. `review-next` does not do that work. It prepares, inspects, accepts, and finalizes a local review of a bundle that is already sealed. An invocation without an operation exits with status 2 and does not read discovery configuration or open a socket. The read-only HTTP client can scan and fetch a review bundle; it does not download presigned source URLs.

## Controller state

Hot state is `state/controller_state.json`. It is a restart checkpoint, not the history.

| Field | Role |
|---|---|
| `last_scanned_document_id` | Exclusive scan cursor. `null` means start at the beginning. Passed to scan as `after_id` (`null` becomes `0`) |
| `last_sealed_batch_id` | Last batch that was actually sealed, or `null` if none. No-work does not change it |
| `current_batch_id` | Open or most recently sealed batch, `batch-000001` style. `null` during scanning, no-work, and a block that happened before any reviewable document |
| `current_batch_document_ids` | Reviewable documents accepted into that batch, ascending |
| `batch_status` | `idle`, `scanning`, `accumulating`, `bundling`, `blocked`, `sealed`, or `no_work` |
| `blocked` | Document id, integrity code, and message when the run is stopped |
| `documents` | Per-document phase for the current batch: `selected`, `bundling`, `bundled` |
| `discovery_exhausted` | The open or sealed batch ran out of candidates, or a no-work result is terminal |
| `applied_ledger_sequence` | Last ledger entry reflected in this checkpoint |

`idle` exists only before the first ledger entry. `scanning` means the cursor has moved for a 404 or non-eligible document and no batch is open. `no_work` means discovery is exhausted and this run selected nothing.

A sealed checkpoint keeps that batch until the next reviewable document opens a batch or a later run records no-work. `last_sealed_batch_id` is what the next real batch increments, so repeated no-work cannot mint `batch-000002`, `batch-000003`, and so on.

`selected` means the document was `REVIEWABLE` and the cursor moved. It does not mean the bytes are durable. `bundled` means the sealed document manifest has been recorded. A blocked checkpoint keeps already selected documents when a batch exists, and keeps no batch id when the block happened first. It does not start bundling.

There is no concurrency control. One process owns `state/` and `outputs/`. Sealing does not add a separate lock: a second writer can still interleave the ledger and the checkpoint. Its temporary manifest name is unique so one attempt does not truncate another's unfinished file, and that is crash hygiene rather than multi-writer safety.

## No-work lifecycle

Do not open a batch before the first `REVIEWABLE` document.

1. Scan from `last_scanned_document_id`.
2. `document_not_found` and `document_non_eligible` advance the cursor and do not take a batch id. From `idle` or `no_work` this enters `scanning`.
3. If a `REVIEWABLE` document appears, the same `document_selected` entry allocates the next real id (`batch-000001` when nothing has sealed) and records that document. Selection then continues as a normal batch.
4. If discovery exhausts with zero selected documents, append one `no_work` ledger entry. Status becomes `no_work`. `current_batch_id` is null. `documents` is empty. `discovery_exhausted` is true. The cursor and `last_sealed_batch_id` stay as they were. No `batch_manifest.json` is published.
5. A later discovery run that is still exhausted at that same cursor does not append another `no_work` event and does not allocate a batch id. The checkpoint is already terminal.
6. When a newer document appears, scan again after the saved cursor. A 404 or non-eligible visit leaves `no_work` for `scanning` without a batch id. Another exhaustion can record `no_work` again, still without a batch id. The first new reviewable document opens the next real batch.

## Ledger

`state/ledger.jsonl` is append-only. One JSON object per line, sequences `1..n` with no gaps. The ledger is authoritative. On load the store replays it:

- a missing checkpoint is rebuilt from the ledger
- a checkpoint that matches the replay prefix is kept, and advanced if the ledger is ahead
- a checkpoint that disagrees with the replay, or is ahead of the ledger, is an integrity error

Historical skip, selection, block, seal, and no-work decisions remain in the ledger after the hot batch fields move on. `no_work` has no `batch_id`. A 404 or non-eligible visit carries a `batch_id` only while a batch is accumulating, including when that visit resolves a blocked checkpoint that already has selected documents. `run_blocked` carries a `batch_id` only when selected documents are already in an open batch. Re-fetching a blocked document is not a ledger event.

## Batch lifecycle

1. Visit candidates in id order. `document_not_found` and `document_non_eligible` advance the cursor and do not join a batch. The first `document_selected` allocates the batch id, records that document, and records the configured batch size in one entry. The cursor advances in that same entry. Later selections append phase `selected` to the open batch and do not change its size. At most that recorded size. An accumulating checkpoint always contains its first document. Changing `TRANSCRIPTION_REVIEW_BATCH_SIZE` does not resize an open or sealed batch. A missing size on an older ledger line means 5.
2. `run_blocked` — status `blocked`, cursor unchanged, blocked id strictly after the cursor and not in the batch. If documents are already selected, that batch id and those documents stay. If none are selected, `batch_id` is null, `current_batch_id` stays null, and no batch id is consumed. The run stops. Nothing is sealed.
3. While status is `blocked`, the controller re-fetches that same document and appends nothing for the re-fetch itself. The cursor stays before the document. The same document id, code, message, and details leave the checkpoint and the ledger unchanged, including `updated_at`. A different code, message, or details is one replacement `run_blocked` and still holds the cursor. If the document is reviewable, one `document_selected` resolves the block, advances the cursor to that id, and either joins the open batch or allocates the next real batch id. If it is non-eligible or missing, one `document_non_eligible` or `document_not_found` resolves the block and advances the cursor. An existing selected batch stays. No batch is allocated when none existed. There is no durable cleared-but-unclassified checkpoint. A crash before that outcome is recorded leaves the previous blocked checkpoint in place.
4. `discovery_exhausted` — no more candidates while at least one document is selected. A batch shorter than its recorded target is allowed.
5. `no_work` — no more candidates and zero selected documents. Not a batch.
6. `bundling_started` — the recorded target number of selected documents, or fewer after exhaustion. Zero documents cannot bundle.
7. `document_bundling` then `document_bundled` — one document at a time, ascending id. Completion records the local transcription SHA-256, each source SHA-256, and the manifest path. The cursor does not move.
8. `batch_sealed` — every selected document is `bundled`. Fewer documents than the recorded target requires exhaustion (`short_batch`). An empty seal is rejected. The batch manifest references each document manifest and its SHA-256. A manifest already on disk is reused only when it still verifies; it is never overwritten.

## Local bundle

```text
outputs/batches/<batch_id>/
  batch_manifest.json
  documents/<document_id>/
    document_manifest.json
    api/review_bundle.json
    transcription/transcription.txt
    sources/<order_index>_<source_kind>.<ext>
```

`document_manifest.json` records document id, archive item id, language, the exact API payload file, the exact transcription text file, the local transcription SHA-256, source metadata, original source files, the local SHA-256 of each source, API provenance (resolver, source set, revisions, optional server text hash), and a file list whose hashes match those artifacts. `sealed` is true. Source `bytes_role` is `original`. `pdf_rasterization` is `not_in_pilot_01`.

A manifest is published with exclusive create. A second publish of the same path fails. Exclusive create stops a second writer; it does not prove the other files are intact. `verify_document_bundle` and `verify_batch_bundle` re-read every listed file, recompute SHA-256, and fail closed on a missing, modified, or unexpected artifact. They do not rewrite anything. Server transcription hashes stay provenance. Local hashes are authoritative. `state/` and `outputs/` are gitignored.

## Fail-closed behavior

- Blocking integrity stops the run and leaves the cursor before the blocked document, whether or not a batch is open.
- An unchanged blocking result is re-fetched without a ledger entry. A reviewable, non-eligible, or 404 outcome resolves that block in the same entry that records the outcome.
- Unexpected HTTP statuses and outcomes do not advance the cursor.
- A checkpoint/ledger mismatch is not repaired by dropping history.
- A sealed manifest is not overwritten.
- A sealed bundle that fails hash verification is not repaired.
- Paths inside the VS Archive application checkout are rejected.
- API URLs with userinfo, queries, or a path are rejected.
- `VERIFIED` transcriptions are `NON_ELIGIBLE` on the server and cannot be sealed locally as reviewable.

## Configuration

Set these in the process environment. No dotenv file is read.

| Variable | Required | Default |
|---|---|---|
| `TRANSCRIPTION_REVIEW_API_BASE_URL` | yes | http(s) origin, no credentials and no path |
| `TRANSCRIPTION_REVIEW_STATE_DIR` | no | `./state` |
| `TRANSCRIPTION_REVIEW_BATCHES_DIR` | no | `./outputs/batches` |
| `TRANSCRIPTION_REVIEW_SCAN_PAGE_LIMIT` | no | `50` (1–200) |
| `TRANSCRIPTION_REVIEW_BATCH_SIZE` | no | `5` (1–200). Applies when a batch is allocated. An open batch keeps the size recorded on its first `document_selected` entry |
| `TRANSCRIPTION_REVIEW_REQUEST_TIMEOUT_SECONDS` | no | `30` |

## Read-only HTTP client

`TranscriptionReviewApiClient` is constructed from the API origin and a request timeout. It keeps one `httpx` session. Every request is a GET built from that origin plus a fixed path: scan uses `/api/agent/v1/transcription-review/documents/scan`, and a review bundle uses `bundle_path(document_id)`. Callers cannot pass a request URL. Redirects are not followed. One method call sends one request. There is no retry policy yet.

`scan(after_id, limit)` sends `after_id=0` when the local cursor is `None`. Any other cursor and `limit` are sent unchanged.

| Call | HTTP | Result |
|---|---|---|
| scan | 200 | `ScanResponse` when the echoed `after_id` and `limit` match the request. A mismatch raises `ApiProtocolError` |
| scan | 400 with `invalid_after_id` or `invalid_limit` | `ScanQueryError` carrying that body |
| scan | any other status | `UnexpectedApiResponseError` (not an empty or exhausted page) |
| review bundle | 200 | `ReviewBundleResponse` when `document.document_id` matches the requested id. A mismatch raises `ApiProtocolError`. `REVIEWABLE`, `NON_ELIGIBLE`, and `BLOCKING_INTEGRITY_ERROR` stay on the payload |
| review bundle | 404 | `ReviewBundleNotFound` (`NOT_FOUND`). The body is ignored |
| review bundle | any other status | `UnexpectedApiResponseError` |

Timeouts raise `ApiTimeoutError`. Connection and other network failures raise `ApiTransportError`. Invalid JSON raises `ApiInvalidJsonError`. A schema-invalid HTTP 200, or a schema-invalid HTTP 400 scan body, raises `ApiSchemaError`. A schema-valid HTTP 200 whose echoed scan cursor, scan limit, or bundle document id does not match the request raises `ApiProtocolError`. Exception text includes the method, path, and status when a response exists. It does not include response bodies, secrets, or presigned URLs.

The client does not download source files and does not write `state/` or `outputs/`.

## Invalid challenge output

`review-next validate` reads a provider file with the same parsing, identity, and evidence rules as `accept`. It does not append a ledger entry, rewrite the checkpoint, change provider status, or write the output file. `output_valid` and `eligible_for_acceptance` are separate. A valid file is not eligible once its attempt is `INVALID_OUTPUT`, and an attempt that is still `AWAITING_MANUAL_DISPATCH` can still have an invalid file. The command prints a JSON report. It exits 0 when the file is valid and 1 when it is not. Eligibility does not change that exit code.

`INVALID_OUTPUT` stays terminal for that attempt. `review-next recover-invalid-output` publishes one new attempt in the same challenge round: a new task id, job id, and output path. The limit is one recovery per provider per round. It does not open another round, change the comparison, the selected dispute ids, the selection hash, or the `EVIDENCE_BOUND_V1` contract, and it does not move or edit the rejected file. The rejected attempt's task bundle is copied under `attempts/a01/task/` before the live task directory is updated. A second call, after the new attempt is already awaiting dispatch, prints the same work request and appends nothing.

Recovery is refused when the provider already succeeded, the round is reconciled, the challenge stage is closed, the document is finalized, or the recovery limit is spent. `accept` still requires the new attempt's identity and, for `EVIDENCE_BOUND_V1`, an evidence reference and assessment on every selected dispute. Validate the new file before accept. A failed accept records `INVALID_OUTPUT` for the new attempt, and that second rejection cannot be recovered.

For the isolated document 105 pilot, after this code is reviewed, the sequence is:

```bash
review-next recover-invalid-output \
  --batches-dir outputs/batches \
  --reviews-dir outputs/reviews_track_a_pilot_105 \
  --batch-id batch-000001 \
  --document-id 105 \
  --provider gpt \
  --stage challenge \
  --round-index 1
```

Leave `challenge_round_01/output/challenge_gpt.json` unchanged. Read the printed `work_request.json`. Write a new JSON file only at its `output_relative_path` (`challenge_round_01/attempts/a02/output/challenge_gpt.json`). Copy `provider`, `provider_interface`, `task_id`, `job_id`, `batch_id`, `document_id`, `attempt`, `source_bundle_sha256`, `stage`, and `round_index` from that work request. Keep the five dispute decisions and their evidence references. `output_example.json` in the task directory shows the required shape; it is not a reading.

```bash
review-next validate \
  --batches-dir outputs/batches \
  --reviews-dir outputs/reviews_track_a_pilot_105 \
  --batch-id batch-000001 \
  --document-id 105 \
  --provider gpt \
  --stage challenge \
  --round-index 1

review-next accept \
  --batches-dir outputs/batches \
  --reviews-dir outputs/reviews_track_a_pilot_105 \
  --batch-id batch-000001 \
  --document-id 105 \
  --provider gpt \
  --stage challenge \
  --round-index 1
```

Accept only when validate reports `output_valid` true and `eligible_for_acceptance` true. Grok's attempt is a separate dispatch and is not changed by GPT's recovery.

## Quality checks

Install the development tools with the project:

```bash
python -m pip install -e '.[dev]'
```

`mypy` is 2.4.1 and `ruff` is 0.14.14. Those versions are pinned. Pytest stays `pytest>=8`.

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy
pytest
```

`ruff check` uses Ruff's default rules and covers every Python file under `src/` and `tests/`. `ruff format --check` uses those same directories, and the unchanged modules listed in `[tool.ruff.format] exclude` are left out of formatting. Pass a directory, not an excluded file path: a file named on the command line is formatted even when it is excluded. `mypy` checks `transcription_review_agent.review` only. It does not ignore errors in that package.

## This milestone

Schemas, cursor rules, state transitions, ledger replay, bundle manifests, discovery, source acquisition, and batch sealing are in place. `review-next` runs the local file-based review on a sealed bundle: prepare work, inspect status, accept a provider file, and finalize. An interim report is a versioned snapshot and does not close the document. A completed review is a later version and does not rewrite earlier ones. Dispatch to GPT Work and Grok Bot stays manual. `review-next` does not scan, call a model API, automate a desktop, or write the archive. An invocation without an operation exits with status 2 before opening a socket.
