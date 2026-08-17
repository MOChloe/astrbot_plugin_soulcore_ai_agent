"""Outbox state transitions, contact settlement and retry backoff."""

from __future__ import annotations

from typing import Any

from ...contracts.models import OutboxStatus
from ...shared.event_log import record_event


class OutboxSettlementMixin:
    async def _record_outbox_wait_once(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        key = (profile_id, instance_id, outbox_id, reason)
        if key in self._logged_outbox_waits:
            return
        self._logged_outbox_waits.add(key)
        await record_event(
            self.event_log,
            profile_id=profile_id,
            instance_id=instance_id,
            level="WARN",
            category="delivery",
            message="Outbox 暂未投递，等待路由恢复",
            details={"outbox_id": outbox_id, "reason": reason, **(details or {})},
        )

    async def _transition_scoped_outbox(
        self,
        profile_id: str,
        instance_id: str,
        outbox_id: int,
        status: OutboxStatus,
        *,
        error: str | None = None,
    ) -> bool:
        return await self.outbox.transition_instance_outbox(
            profile_id, instance_id, outbox_id, status, error=error
        )

    async def _settle_contact_outbox(
        self,
        profile_id: str,
        instance_id: str,
        payload: dict[str, Any],
        *,
        delivered: bool,
        reason: str,
        superseded: bool = False,
        attempted_unknown: bool = False,
    ) -> bool:
        """Settle autonomous contact only after the adapter outcome is known.

        Main Core producing text is intentionally insufficient: the attempt,
        daily quota and one-shot evidence remain untouched until this delivery
        boundary is reached.
        """

        attempt_ref = str(payload.get("contact_attempt_ref") or "").strip()
        if not attempt_ref:
            return False
        generation = int(payload.get("contact_generation") or 0)
        task_id = int(payload.get("ai_task_id") or 0) or None
        failure_mode = str(payload.get("contact_failure_mode") or "SKIP").upper()
        outcome = self._contact_outcome(
            delivered=delivered,
            attempted_unknown=attempted_unknown,
            superseded=superseded,
            failure_mode=failure_mode,
        )
        settled = await self._settle_contact_evidence(
            profile_id,
            instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            outcome=outcome,
        )
        finalized = await self._finalize_contact_attempt(
            profile_id,
            instance_id,
            attempt_ref=attempt_ref,
            generation=generation,
            task_id=task_id,
            delivered=delivered,
            attempted_unknown=attempted_unknown,
        )
        return settled or finalized

    @staticmethod
    def _contact_outcome(
        *,
        delivered: bool,
        attempted_unknown: bool,
        superseded: bool,
        failure_mode: str,
    ) -> str:
        if delivered:
            return "DELIVERED"
        if attempted_unknown:
            return "ATTEMPTED_UNKNOWN"
        # Foreground activity releases the reservation. The latest foreground
        # run may consume the projection without counting it as delivery.
        if superseded:
            return "SUPPRESSED"
        return "SUPERSEDED" if failure_mode == "SKIP" else "FAILED"

    async def _settle_contact_evidence(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        outcome: str,
    ) -> bool:
        return bool(
            await self.timeline.settle_contact_evidence(
                profile_id,
                instance_id,
                attempt_ref=attempt_ref,
                generation=generation,
                outcome=outcome,
            )
        )

    async def _finalize_contact_attempt(
        self,
        profile_id: str,
        instance_id: str,
        *,
        attempt_ref: str,
        generation: int,
        task_id: int | None,
        delivered: bool,
        attempted_unknown: bool,
    ) -> bool:
        try:
            return bool(
                await self.timeline.finalize_contact_attempt(
                    profile_id,
                    instance_id,
                    attempt_ref,
                    generation=generation,
                    attempted=bool(delivered or attempted_unknown),
                    success=bool(delivered),
                    answered=False,
                    task_id=task_id,
                )
            )
        except KeyError:
            return False

    async def _outbox_belongs_to_bound_instance(
        self, profile_id: str, instance_id: str, item: Any
    ) -> bool:
        """Reject an outbox row that drifted away from its owning instance."""

        if str(item.instance_id or "") != instance_id:
            return False
        instance = await self.profiles.get_character_instance(profile_id, instance_id)
        if instance is None:
            return False
        return str(instance.route_umo) == str(item.umo)


__all__ = ["OutboxSettlementMixin"]
