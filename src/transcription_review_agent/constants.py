"""Pilot 01 constants shared by schemas, state, and bundle layout."""

PILOT_TARGET_REVIEWABLE_COUNT = 5
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 200

SCAN_SCHEMA_VERSION = "transcription-review-scan-v1"
BUNDLE_SCHEMA_VERSION = "transcription-review-bundle-v1"
STATE_SCHEMA_VERSION = 1
LOCAL_DOCUMENT_BUNDLE_SCHEMA_VERSION = "local-document-bundle-v1"
LOCAL_BATCH_SCHEMA_VERSION = "local-batch-v1"

SCAN_PATH = "/api/agent/v1/transcription-review/documents/scan"
BUNDLE_PATH_TEMPLATE = (
    "/api/agent/v1/transcription-review/documents/{document_id}/review-bundle"
)

DEFAULT_SCAN_LIMIT = 50
MIN_SCAN_LIMIT = 1
MAX_SCAN_LIMIT = 200

BATCH_ID_PATTERN = r"^batch-[0-9]{6}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
