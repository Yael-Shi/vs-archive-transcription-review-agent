"""Controller package.

``discover_next_batch`` commits discovery decisions. ``review_next`` is still
reserved and does not scan, download, or write state.
"""

from transcription_review_agent.controller.discovery import (
    DiscoveryResult,
    discover_next_batch,
)
from transcription_review_agent.controller.run import review_next

__all__ = ["DiscoveryResult", "discover_next_batch", "review_next"]
