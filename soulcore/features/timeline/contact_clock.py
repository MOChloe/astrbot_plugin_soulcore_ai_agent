"""Evidence-gated autonomous contact clock."""

from __future__ import annotations

import random
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .contact_models import (
    ContactClaim,
    ContactClockRepository,
    ContactEvaluation,
    ContactEvidenceKind,
    ContactOpportunity,
    ContactOutcome,
    ContactPolicy,
    ContactProfileRepository,
    TimelineEvidence,
    contact_day_bucket_transition,
)
from .contact_parsing import _contact_claim, _field, _parse_datetime, _quiet_until


def _evidence_key(item: Mapping[str, Any]) -> tuple[str, str]:
    kind = str(item.get("evidence_kind") or ContactEvidenceKind.ROLE_TIMELINE_EVENT.value).upper()
    return kind, str(item.get("evidence_id") or "")


def _action_evidence_projection(action: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    action_id = str(action.get("evidence_ref") or "")
    occurred = action.get("occurred_at") or now
    if not isinstance(occurred, datetime):
        occurred = _parse_datetime(occurred) or now
    importance = float(action.get("importance") or 0.0)
    return {
        "evidence_id": action_id,
        "evidence_kind": ContactEvidenceKind.ACTION_RESULT.value,
        "summary": str(action.get("summary") or ""),
        "occurred_at": occurred,
        "important": importance >= 0.80,
        "importance": importance,
        "reason": str(action.get("reason") or ""),
    }


def _timeline_event_projection(event: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    occurred = event.get("frame_end_at") or now
    if not isinstance(occurred, datetime):
        occurred = _parse_datetime(occurred) or now
    source = str(event.get("source") or "")
    content = str(event.get("content") or "").strip()
    leftover = str(event.get("leftover_text") or "").strip()
    summary = content
    if leftover:
        summary = f"{content}\n可取用的残余素材：{leftover}" if content else leftover
    return {
        "evidence_id": str(event.get("event_id") or ""),
        "evidence_kind": ContactEvidenceKind.ROLE_TIMELINE_EVENT.value,
        "summary": summary,
        "occurred_at": occurred,
        "important": False,
        "importance": 0.6,
        "reason": source,
    }


def _append_unique_evidence(
    evidence: list[Any],
    keys: set[tuple[str, str]],
    projected: dict[str, Any],
) -> None:
    key = _evidence_key(projected)
    if not key[1] or key in keys:
        return
    evidence.append(projected)
    keys.add(key)


class ContactClock:
    """Open a contact opportunity whenever the character is free and policy allows it."""

    def __init__(
        self,
        repository: ContactClockRepository,
        *,
        profiles: ContactProfileRepository,
        random_source: random.Random | Any | None = None,
    ) -> None:
        self.repository = repository
        self.profiles = profiles
        self.random = random_source or random

    def next_regular_check(self, now: datetime, policy: ContactPolicy) -> datetime:
        minutes = self.random.randint(policy.check_min_minutes, policy.check_max_minutes)
        return now + timedelta(minutes=minutes)

    def evaluate(
        self,
        claim: ContactClaim,
        policy: ContactPolicy,
        *,
        now: datetime,
        local_now: datetime | None = None,
    ) -> ContactEvaluation:
        next_check = self.next_regular_check(now, policy)
        latest = (
            str(claim.timeline_event_through)
            if claim.timeline_event_through > claim.timeline_event_watermark
            else claim.latest_timeline_event_id
        )
        if not policy.enabled:
            # Advancing the watermark while disabled is the no-backfill rule.
            return ContactEvaluation(
                ContactOutcome.DISABLED_CONSUMED,
                next_check,
                latest,
                evidence_snapshot=claim.evidence,
            )
        quiet_until = _quiet_until(local_now or now, policy) if policy.quiet_enabled else None
        if quiet_until is not None:
            # Two aware datetimes carrying the same ZoneInfo are subtracted in
            # wall-clock space by Python.  Convert both ends to UTC first so a
            # DST transition inside the quiet interval cannot shift the real
            # retry instant by an hour.
            delta = max(
                timedelta(0),
                quiet_until.astimezone(UTC) - (local_now or now).astimezone(UTC),
            )
            retry_at = now + delta
            return ContactEvaluation(
                ContactOutcome.QUIET_HOURS,
                max(next_check, retry_at),
                None,
                retry_not_before=retry_at,
            )

        if claim.last_contact_at is not None:
            retry_at = claim.last_contact_at + timedelta(
                minutes=policy.min_contact_interval_minutes
            )
            if retry_at > now:
                return ContactEvaluation(
                    ContactOutcome.MIN_INTERVAL,
                    max(next_check, retry_at),
                    None,
                    retry_not_before=retry_at,
                )
        if policy.daily_limit is not None and claim.contacts_today >= policy.daily_limit:
            return ContactEvaluation(
                ContactOutcome.DAILY_LIMIT,
                next_check,
                latest,
                evidence_snapshot=claim.evidence,
            )
        if (
            policy.unanswered_limit is not None
            and claim.unanswered_count >= policy.unanswered_limit
        ):
            return ContactEvaluation(
                ContactOutcome.UNANSWERED_LIMIT,
                next_check,
                latest,
                evidence_snapshot=claim.evidence,
            )
        return ContactEvaluation(
            ContactOutcome.READY,
            next_check,
            # Claiming an opportunity only freezes evidence. Watermarks advance
            # after a real adapter dispatch.
            None,
            evidence_snapshot=claim.evidence,
        )

    async def run_once(
        self,
        profile_id: str,
        instance_id: str,
        policy: ContactPolicy,
        *,
        now: datetime,
        local_now: datetime | None = None,
    ) -> ContactOpportunity | None:
        rows = await self.repository.claim_contact_clock(
            now=now, limit=1, profile_id=profile_id, instance_id=instance_id
        )
        if not rows:
            return None
        return await self._process_claim(rows[0], policy, now=now, local_now=local_now)

    async def run_due(
        self,
        *,
        now: datetime,
        limit: int = 10,
        local_now: datetime | None = None,
    ) -> tuple[ContactOpportunity, ...]:
        rows = await self.repository.claim_contact_clock(now=now, limit=max(1, int(limit)))
        opportunities: list[ContactOpportunity] = []
        for raw in rows:
            profile_id = str(_field(raw, "profile_id", ""))
            instance_id = str(_field(raw, "instance_id", ""))
            if not profile_id or not instance_id:
                continue
            policy = ContactPolicy()
            policy = ContactPolicy.from_mapping(
                await self.repository.resolve_contact_policy(profile_id, instance_id)
            )
            opportunity = await self._process_claim(
                raw,
                policy,
                now=now,
                local_now=(
                    local_now
                    or (
                        now.astimezone(ZoneInfo(policy.timezone_name))
                        if policy.timezone_name
                        else now.astimezone()
                    )
                ),
            )
            if opportunity is not None:
                opportunities.append(opportunity)
        return tuple(opportunities)

    async def _process_claim(
        self,
        raw: Mapping[str, Any],
        policy: ContactPolicy,
        *,
        now: datetime,
        local_now: datetime | None,
    ) -> ContactOpportunity | None:
        profile_id = str(raw.get("profile_id") or "")
        instance_id = str(raw.get("instance_id") or "")
        enriched = await self._enrich_claim(raw, profile_id, instance_id, now=now)
        effective_local_now = local_now or now.astimezone()
        _, carry_daily_count = contact_day_bucket_transition(
            enriched.get("daily_bucket"),
            effective_local_now.date().isoformat(),
        )
        if not carry_daily_count:
            enriched["daily_success_count"] = 0
        claim = _contact_claim(enriched, profile_id=profile_id, instance_id=instance_id)
        evaluation = self.evaluate(claim, policy, now=now, local_now=local_now)
        current_state = await self.profiles.get_instance_state(profile_id, instance_id)
        activity_changed = int(current_state.activity_epoch) != claim.activity_epoch
        state_changed = int(getattr(current_state, "state_epoch", claim.state_epoch)) != (
            claim.state_epoch
        )
        if activity_changed or state_changed:
            await self._release_claim(
                claim,
                next_check_at=now + timedelta(minutes=1),
                reason=(
                    ContactOutcome.SUPERSEDED_BY_FOREGROUND.value
                    if activity_changed
                    else ContactOutcome.SUPERSEDED_BY_CORE_STATE.value
                ),
            )
            return None
        return await self._commit_evaluated_claim(
            enriched,
            claim,
            evaluation,
            policy=policy,
            now=now,
        )

    async def _commit_evaluated_claim(
        self,
        enriched: Mapping[str, Any],
        claim: ContactClaim,
        evaluation: ContactEvaluation,
        *,
        policy: ContactPolicy,
        now: datetime,
    ) -> ContactOpportunity | None:
        profile_id = claim.profile_id
        instance_id = claim.instance_id
        planned_at = _parse_datetime(enriched.get("next_check_at")) or now
        if planned_at.tzinfo is None or planned_at.utcoffset() is None:
            raise ValueError("contact planned time must be timezone-aware")
        planned_at = planned_at.astimezone(UTC)
        proactive_source_ref = f"contact-check:{profile_id}:{instance_id}:{planned_at.isoformat()}"
        committed = await self._commit_claim(
            claim,
            evaluation,
            policy=policy,
            now=now,
            proactive_frame_planned_at=planned_at,
            proactive_frame_source_ref=proactive_source_ref,
        )
        # A newer foreground activity changes the activity epoch and makes the
        # repository CAS fail.  No opportunity escapes that fence.
        if not committed:
            return None
        if evaluation.outcome is not ContactOutcome.READY:
            return None
        attempt_ref = self._attempt_ref(claim)
        if claim.evidence and not await self._reserve_evidence(claim, attempt_ref=attempt_ref):
            with suppress(KeyError):
                await self.repository.finalize_contact_attempt(
                    claim.profile_id,
                    claim.instance_id,
                    attempt_ref,
                    generation=claim.generation,
                    attempted=False,
                    success=False,
                    answered=False,
                )
            return None
        return ContactOpportunity(
            profile_id=profile_id,
            instance_id=instance_id,
            generation=claim.generation,
            activity_epoch=claim.activity_epoch,
            route_umo=claim.route_umo,
            evidence=claim.evidence,
            state_epoch=claim.state_epoch,
            attempt_ref=str(_field(committed, "attempt_ref", "") or attempt_ref),
            failure_mode=policy.failure_mode,
            reroll_count=claim.reroll_count,
            proactive_frame_planned_at=planned_at,
            proactive_frame_source_ref=proactive_source_ref,
        )

    async def _reserve_evidence(self, claim: ContactClaim, *, attempt_ref: str) -> bool:
        evidence = [
            {
                "evidence_kind": item.evidence_kind.value,
                "evidence_ref": item.evidence_id,
            }
            for item in claim.evidence
            if item.evidence_id
        ]
        if not evidence:
            return False
        try:
            rows = await self.repository.reserve_contact_evidence(
                claim.profile_id,
                claim.instance_id,
                attempt_ref=attempt_ref,
                generation=claim.generation,
                evidence=evidence,
            )
        except ValueError:
            return False
        return len(rows) == len(evidence)

    async def _enrich_claim(
        self,
        raw: Mapping[str, Any],
        profile_id: str,
        instance_id: str,
        *,
        now: datetime,
    ) -> dict[str, Any]:
        value = dict(raw)
        evidence = list(value.get("evidence") or ())
        keys = {_evidence_key(item) for item in evidence}
        await self._append_timeline_events(
            value,
            evidence,
            keys,
            profile_id,
            instance_id,
            now,
        )
        await self._append_action_evidence(value, evidence, keys, profile_id, instance_id, now)
        value["evidence"] = evidence
        return value

    async def _append_timeline_events(
        self,
        value: Mapping[str, Any],
        evidence: list[Any],
        existing_keys: set[tuple[str, str]],
        profile_id: str,
        instance_id: str,
        now: datetime,
    ) -> None:
        frozen = value.get("evidence_snapshot") or {}
        after = int(value.get("timeline_event_watermark") or 0)
        through = int(
            frozen.get("timeline_event_through", after) if isinstance(frozen, Mapping) else after
        )
        rows = await self.repository.list_role_timeline_events(
            profile_id,
            instance_id,
            after_event_id=after,
            through_event_id=through,
            limit=6,
        )
        for event in rows:
            projected = _timeline_event_projection(event, now)
            _append_unique_evidence(evidence, existing_keys, projected)

    async def _append_action_evidence(
        self,
        value: Mapping[str, Any],
        evidence: list[Any],
        existing_keys: set[tuple[str, str]],
        profile_id: str,
        instance_id: str,
        now: datetime,
    ) -> None:
        action_rows = list(value.get("action_result_evidence") or ())
        if not action_rows:
            action_rows = list(
                await self.repository.list_contact_action_results(profile_id, instance_id, limit=5)
            )
        for action in action_rows[:5]:
            projected = _action_evidence_projection(action, now)
            _append_unique_evidence(evidence, existing_keys, projected)

    async def _commit_claim(
        self,
        claim: ContactClaim,
        evaluation: ContactEvaluation,
        *,
        policy: ContactPolicy,
        now: datetime,
        proactive_frame_planned_at: datetime,
        proactive_frame_source_ref: str,
    ) -> Any | None:
        snapshot = {
            "outcome": evaluation.outcome.value,
            "attempt_ref": (
                self._attempt_ref(claim) if evaluation.outcome is ContactOutcome.READY else ""
            ),
            "generation": claim.generation,
            "reroll_count": claim.reroll_count,
            "activity_epoch": claim.activity_epoch,
            "state_epoch": claim.state_epoch,
            "route_umo": claim.route_umo,
            "failure_mode": policy.failure_mode,
            "proactive_frame_planned_at": proactive_frame_planned_at.isoformat(),
            "proactive_frame_source_ref": proactive_frame_source_ref,
            "items": [
                {
                    "evidence_id": item.evidence_id,
                    "evidence_kind": item.evidence_kind.value,
                    "summary": item.summary,
                    "occurred_at": item.occurred_at.isoformat(),
                    "important": item.important,
                    "importance": item.importance,
                    "reason": item.reason,
                }
                for item in evaluation.evidence_snapshot
            ],
        }
        latest = evaluation.consume_through_evidence_id
        watermark = int(latest) if str(latest or "").isdigit() else None
        attempt_ref = (
            self._attempt_ref(claim) if evaluation.outcome is ContactOutcome.READY else None
        )
        return await self.repository.commit_contact_clock(
            claim.profile_id,
            claim.instance_id,
            expected_version=claim.version,
            lease_token=claim.lease_token,
            expected_generation=claim.generation,
            expected_state_epoch=claim.state_epoch,
            expected_activity_epoch=claim.activity_epoch,
            next_check_at=evaluation.next_check_at,
            result=evaluation.outcome.value,
            reason=evaluation.outcome.value.lower(),
            timeline_event_watermark=watermark,
            deferred_evidence=(snapshot if evaluation.outcome is ContactOutcome.READY else {}),
            attempt_ref=attempt_ref,
            now=now,
        )

    @staticmethod
    def _attempt_ref(claim: ContactClaim) -> str:
        return f"contact:{claim.profile_id}:{claim.instance_id}:{claim.generation}"

    async def _release_claim(
        self,
        claim: ContactClaim,
        *,
        next_check_at: datetime,
        reason: str,
    ) -> bool:
        return bool(
            await self.repository.release_contact_clock(
                claim.profile_id,
                claim.instance_id,
                expected_version=claim.version,
                lease_token=claim.lease_token,
                next_check_at=next_check_at,
                reason=reason,
            )
        )


__all__ = [
    "ContactClaim",
    "ContactClock",
    "ContactClockRepository",
    "ContactEvidenceKind",
    "ContactEvaluation",
    "ContactOpportunity",
    "ContactOutcome",
    "ContactPolicy",
    "TimelineEvidence",
]
