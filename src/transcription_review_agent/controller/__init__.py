"""Controller package.

``discover_next_batch`` commits discovery decisions. ``bundle_selected_batch``
publishes verified document bundles for an already selected batch.
``seal_selected_batch`` publishes the batch manifest and records the seal.
``review_next`` runs a local review operation and does not scan or download.
"""

from transcription_review_agent.bundling.acquire import (
    BundleResult,
    bundle_selected_batch,
)
from transcription_review_agent.bundling.seal import SealResult, seal_selected_batch
from transcription_review_agent.controller.discovery import (
    DiscoveryResult,
    discover_next_batch,
)
from transcription_review_agent.controller.run import review_next

__all__ = [
    "BundleResult",
    "DiscoveryResult",
    "SealResult",
    "bundle_selected_batch",
    "discover_next_batch",
    "review_next",
    "seal_selected_batch",
]
