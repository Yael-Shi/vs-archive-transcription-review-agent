"""Controller package.

``discover_next_batch`` commits discovery decisions. ``bundle_selected_batch``
publishes verified document bundles for an already selected batch.
``review_next`` is still reserved and does not scan, download, or write state.
"""

from transcription_review_agent.bundling.acquire import (
    BundleResult,
    bundle_selected_batch,
)
from transcription_review_agent.controller.discovery import (
    DiscoveryResult,
    discover_next_batch,
)
from transcription_review_agent.controller.run import review_next

__all__ = [
    "BundleResult",
    "DiscoveryResult",
    "bundle_selected_batch",
    "discover_next_batch",
    "review_next",
]
