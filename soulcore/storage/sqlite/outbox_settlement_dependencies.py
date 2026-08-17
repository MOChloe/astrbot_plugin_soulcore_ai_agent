"""Transaction helpers shared by atomic Outbox settlement.

This composition module is the only place where one SQLite feature is allowed
to bind another feature's concrete transaction implementation.
"""

from __future__ import annotations

from ...features.media.sqlite.asset_projection_records import _link_media_to_message_sql
from ...features.stickers.sqlite.retrieval import record_sticker_usage_in_transaction
from ...features.timeline.sqlite.contact_clock import finalize_contact_attempt_sql

__all__ = [
    "_link_media_to_message_sql",
    "finalize_contact_attempt_sql",
    "record_sticker_usage_in_transaction",
]
