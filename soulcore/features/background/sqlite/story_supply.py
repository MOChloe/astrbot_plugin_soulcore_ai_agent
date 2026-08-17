"""One exact supply ranking for story modules across all background consumers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from fractions import Fraction

from ..domain import BackgroundStorySource
from .rows import story_source_from_row


@dataclass(frozen=True, slots=True)
class StorySupplyCandidate:
    """A module paired with its latest character-association distance."""

    story: BackgroundStorySource
    narrative_distance: int | None

    @property
    def association(self) -> Fraction:
        if self.narrative_distance is None:
            return Fraction(0, 1)
        return Fraction(1, self.narrative_distance + 1)

    @property
    def supply_score(self) -> Fraction:
        return (Fraction(1, 1) + self.association) / (self.story.shown_count + 1)


def story_supply_sort_key(
    candidate: StorySupplyCandidate,
) -> tuple[Fraction, Fraction, int, int]:
    """Return the canonical descending key without floating-point rounding."""

    story = candidate.story
    return (
        candidate.supply_score,
        candidate.association,
        -story.shown_count,
        story.story_source_id,
    )


def rank_story_supply(
    candidates: tuple[StorySupplyCandidate, ...],
) -> tuple[StorySupplyCandidate, ...]:
    """Rank highest supply first using the single product-defined ordering."""

    return tuple(sorted(candidates, key=story_supply_sort_key, reverse=True))


def story_supply_eviction_order(
    candidates: tuple[StorySupplyCandidate, ...],
) -> tuple[StorySupplyCandidate, ...]:
    """Order capacity victims: oldest concluded, then worst ordinary supply."""

    concluded = sorted(
        (candidate for candidate in candidates if candidate.story.engagement_state == "CONCLUDED"),
        key=lambda candidate: candidate.story.story_source_id,
    )
    ordinary = sorted(
        (candidate for candidate in candidates if candidate.story.engagement_state != "CONCLUDED"),
        key=story_supply_sort_key,
    )
    return (*concluded, *ordinary)


def read_story_supply_candidates(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
) -> tuple[StorySupplyCandidate, ...]:
    """Read modules and derive association from the canonical narrative timeline."""

    rows = conn.execute(
        """WITH event_distances AS (
               SELECT event_id,
                      ROW_NUMBER() OVER (
                          ORDER BY frame_end_at DESC, event_id DESC
                      ) - 1 AS narrative_distance
               FROM background_role_timeline_events
               WHERE profile_id = ? AND instance_id = ?
           ),
           latest_associations AS (
               SELECT link.story_source_id,
                      MIN(event_distances.narrative_distance) AS narrative_distance
               FROM background_timeline_event_story_sources AS link
               JOIN event_distances ON event_distances.event_id = link.event_id
               GROUP BY link.story_source_id
           )
           SELECT story.*, latest_associations.narrative_distance
           FROM background_story_sources AS story
           LEFT JOIN latest_associations
             ON latest_associations.story_source_id = story.story_source_id
           WHERE story.profile_id = ? AND story.instance_id = ?""",
        (profile_id, instance_id, profile_id, instance_id),
    ).fetchall()
    return tuple(
        StorySupplyCandidate(
            story=story_source_from_row(row),
            narrative_distance=(
                int(row["narrative_distance"]) if row["narrative_distance"] is not None else None
            ),
        )
        for row in rows
    )


def ranked_story_supply(
    conn: sqlite3.Connection,
    profile_id: str,
    instance_id: str,
    *,
    include_concluded: bool = False,
) -> tuple[StorySupplyCandidate, ...]:
    """Read and rank the eligible module pool."""

    candidates = read_story_supply_candidates(conn, profile_id, instance_id)
    if not include_concluded:
        candidates = tuple(
            candidate for candidate in candidates if candidate.story.engagement_state != "CONCLUDED"
        )
    return rank_story_supply(candidates)


__all__ = [
    "StorySupplyCandidate",
    "rank_story_supply",
    "ranked_story_supply",
    "read_story_supply_candidates",
    "story_supply_sort_key",
    "story_supply_eviction_order",
]
