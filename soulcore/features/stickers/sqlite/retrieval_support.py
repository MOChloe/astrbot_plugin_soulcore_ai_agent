from __future__ import annotations

from .support import sqlite3

LIVE_STICKER_RUN_REF_CONDITION = """(
  ref.expires_at > ?
  OR EXISTS (
    SELECT 1 FROM instance_outbox delivery
    WHERE delivery.profile_id = ref.profile_id
      AND delivery.instance_id = ref.instance_id
      AND CAST(delivery.origin_run_id AS TEXT) = ref.run_id
      AND delivery.status IN ('PENDING', 'SENDING')
  )
)"""


def has_live_sticker_run_ref(
    conn: sqlite3.Connection,
    *,
    profile_id: str,
    item_id: str,
    now: str,
) -> bool:
    return (
        conn.execute(
            f"""SELECT 1 FROM sticker_run_candidates ref
            WHERE ref.profile_id = ? AND ref.item_id = ?
              AND {LIVE_STICKER_RUN_REF_CONDITION}
            LIMIT 1""",
            (profile_id, item_id, now),
        ).fetchone()
        is not None
    )


__all__ = ["LIVE_STICKER_RUN_REF_CONDITION", "has_live_sticker_run_ref"]
