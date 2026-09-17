"""Player-facing read actions assembled from the existing durable sources."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ...features.timers.domain import TimerScope
from ...features.timers.main_core_views import schedule_summary
from ...features.timers.rules import next_occurrence
from .console_view_records import _outbox_view
from .delivery_attention import (
    delivery_failure_preference_key,
    parse_delivery_failure_acknowledgements,
)
from .player_views import (
    player_character_view,
    player_contact_ref,
    player_contact_view,
    player_current_life_view,
    player_intents_view,
    player_life_events_view,
    player_memories_view,
    player_people_view,
    player_portrait_view,
    player_release_notes_view,
    player_role_ref,
    player_role_view,
    player_world_view,
)
from .presentation import jsonable

_PLAYER_GUIDE_VERSION = 1
_PLAYER_GUIDE_PREFERENCE_KEY = "player.guide.seen_version"
_ADVANCED_GUIDE_VERSION = 1
_ADVANCED_GUIDE_PREFERENCE_KEY = "advanced.guide.seen_version"


def _delivery_acknowledgements_by_instance(
    preference_keys: Mapping[str, str],
    preference_values: Mapping[str, str],
) -> dict[str, frozenset[str]]:
    return {
        instance_id: frozenset(
            parse_delivery_failure_acknowledgements(preference_values.get(preference_key, ""))
        )
        for instance_id, preference_key in preference_keys.items()
    }


def _relationship_portrait_payload(payload: Mapping[str, Any], person_ref: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "page_size": 20,
        "person_ref": person_ref,
        "entry_page": 1,
        "entry_page_size": 20,
    }
    if "people_page" in payload:
        result["page"] = max(1, int(payload.get("people_page") or 1))
    return result


def _relationship_contact_projection(
    contact: Mapping[str, Any],
    scope: str,
    portrait: Mapping[str, Any],
) -> dict[str, Any]:
    projection = dict(contact)
    if scope == "group":
        return projection
    observed_name = str(portrait.get("selected_display_name") or "").strip()
    if observed_name:
        projection["display_name"] = observed_name
    return projection


def _relationship_contact_preferences(policies: Mapping[str, Any]) -> dict[str, Any]:
    effective = dict(policies.get("effective") or {})
    chat_policy = dict(policies.get("chat_policy") or {})
    return {
        "can_reply": bool(chat_policy.get("soulcore_enabled", True)),
        "can_send_images": bool(chat_policy.get("image_send_enabled", True)),
        "proactive_enabled": bool(effective.get("proactive_enabled", True)),
        "quiet_enabled": bool(effective.get("quiet_enabled", True)),
        "quiet_start": str(effective.get("quiet_start") or "23:00"),
        "quiet_end": str(effective.get("quiet_end") or "08:00"),
        "daily_limit_mode": str(effective.get("daily_limit_mode") or "LIMITED"),
        "daily_limit": effective.get("daily_success_limit"),
    }


class PlayerPageActionsMixin:
    async def _player_bootstrap(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows, selected = await self._player_roles(payload)
        guide = await self._player_guide_state()
        roles = await asyncio.gather(*(self._player_role_view(row, selected) for row in rows))
        if not selected:
            return {
                "version": self._plugin_version(),
                "roles": roles,
                "selected_role_ref": "",
                "selected_contact_ref": "",
                "contacts": [],
                "readiness": self._player_readiness(None),
                "problem_count": 0,
                "guide": guide,
            }
        contacts = await self._player_contact_list(selected)
        requested = str(payload.get("contact_ref") or "").strip()
        selected_contact = next(
            (item["contact_ref"] for item in contacts if item["contact_ref"] == requested),
            contacts[0]["contact_ref"] if contacts else "",
        )
        settings = await self._settings_snapshot(selected, "private")
        readiness = self._player_readiness(settings.get("readiness"))
        return {
            "version": self._plugin_version(),
            "roles": roles,
            "selected_role_ref": player_role_ref(selected),
            "selected_contact_ref": selected_contact,
            "contacts": contacts[:8],
            "readiness": readiness,
            "problem_count": int(not readiness["ready"])
            + sum(int(item["problem_count"]) for item in contacts),
            "guide": guide,
        }

    async def _player_role_view(self, row: Mapping[str, Any], selected: str) -> dict[str, Any]:
        profile_id = self._row_profile_id(row)
        snapshot = await self.character_models.snapshot(profile_id)
        model = (snapshot.get("character_model") or {}).get("model") or {}
        character_name = str((model.get("identity") or {}).get("name") or "").strip()
        # AstrBot's profile label identifies configuration, not the character.
        presentation = {**row, "display_name": character_name} if character_name else row
        return {
            **player_role_view(presentation, selected=profile_id == selected),
            "profile_name": str(row.get("name") or row.get("display_name") or "").strip(),
        }

    async def _player_contacts(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        page = max(1, int(payload.get("page") or 1))
        page_size = max(5, min(int(payload.get("page_size") or 20), 50))
        contacts = await self._player_contact_list(profile_id)
        start = (page - 1) * page_size
        items = contacts[start : start + page_size]
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": len(contacts),
            "has_more": start + page_size < len(contacts),
            "counts": {
                "private": sum(item["kind"] == "private" for item in contacts),
                "group": sum(item["kind"] == "group" for item in contacts),
            },
        }

    async def _player_history(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        page = max(1, int(payload.get("page") or 1))
        page_size = max(1, min(int(payload.get("page_size") or 20), 50))
        contact_page_size = max(5, min(int(payload.get("contact_page_size") or 20), 50))
        requested_contact_ref = str(payload.get("contact_ref") or "").strip()
        contacts, public_contacts = await self._player_history_contacts(
            profile_id, requested_contact_ref
        )
        result = await self.player_history.history(
            profile_id,
            contacts,
            page=page,
            page_size=page_size,
        )
        if requested_contact_ref and "contact_page" not in payload:
            selected_index = next(
                (
                    index
                    for index, item in enumerate(public_contacts)
                    if str(item.get("contact_ref") or "") == requested_contact_ref
                ),
                0,
            )
            contact_page = selected_index // contact_page_size + 1
        else:
            contact_page = max(1, int(payload.get("contact_page") or 1))
        contact_page_count = max(
            1, (len(public_contacts) + contact_page_size - 1) // contact_page_size
        )
        contact_page = min(contact_page, contact_page_count)
        contact_start = (contact_page - 1) * contact_page_size
        result["contacts"] = public_contacts[contact_start : contact_start + contact_page_size]
        result["contact_pagination"] = {
            "page": contact_page,
            "page_size": contact_page_size,
            "page_count": contact_page_count,
            "total": len(public_contacts),
            "has_more": contact_page < contact_page_count,
        }
        result["selected_contact_ref"] = requested_contact_ref
        return result

    async def _player_history_contacts(
        self, profile_id: str, requested_contact_ref: str
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
        snapshot, public_contacts = await asyncio.gather(
            self.profiles.role_instances_snapshot(profile_id),
            self._player_contact_list(profile_id),
        )
        internal_by_ref = {
            player_contact_ref(profile_id, str(item.get("instance_id") or "")): dict(item)
            for item in snapshot.get("instances") or ()
        }
        public_by_ref = {str(item.get("contact_ref") or ""): item for item in public_contacts}
        if requested_contact_ref and requested_contact_ref not in internal_by_ref:
            raise ValueError("联系人已经变化，请重新选择")
        selected_refs = (
            [requested_contact_ref]
            if requested_contact_ref
            else [
                str(item.get("contact_ref") or "")
                for item in public_contacts
                if str(item.get("contact_ref") or "") in internal_by_ref
            ]
        )
        contacts = [
            (internal_by_ref[contact_ref], public_by_ref[contact_ref])
            for contact_ref in selected_refs
            if contact_ref in public_by_ref
        ]
        return contacts, public_contacts

    async def _player_history_record(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        snapshot, public_contacts = await asyncio.gather(
            self.profiles.role_instances_snapshot(profile_id),
            self._player_contact_list(profile_id),
        )
        public_by_ref = {str(item.get("contact_ref") or ""): item for item in public_contacts}
        contacts = [
            (
                dict(item),
                public_by_ref.get(
                    player_contact_ref(profile_id, str(item.get("instance_id") or "")),
                    player_contact_view(profile_id, item),
                ),
            )
            for item in snapshot.get("instances") or ()
        ]
        return await self.player_history.record_from_contacts(
            profile_id,
            contacts,
            str(payload.get("record_ref") or ""),
        )

    async def _player_now(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        contact = await self._player_contact(profile_id, payload, required=False)
        if contact is None:
            return {
                "contact": None,
                "current": player_current_life_view({}),
                "events": [],
                "plans": [],
                "arrangements": [],
                "arrangement_pagination": _page_view(1, 10, 0),
                "problems": [],
            }
        instance_id = str(contact["instance_id"])
        background, detail, arrangements = await asyncio.gather(
            self.background.workspace(profile_id, instance_id),
            self._instance_detail(profile_id, instance_id, {"message_page_size": 5}),
            self._player_arrangements(
                profile_id,
                instance_id,
                page=max(1, int(payload.get("arrangement_page") or 1)),
                page_size=max(5, min(int(payload.get("arrangement_page_size") or 10), 20)),
            ),
        )
        problems = await self._player_instance_problems(profile_id, instance_id)
        if int(background.get("problem_count") or 0):
            problems.append(
                {
                    "code": "life_update_problem",
                    "title": "角色最近的生活没有继续更新",
                    "summary": "已经发生的经历仍然保留，可以在高级设置中查看原因。",
                    "action": "developer-background",
                }
            )
        return {
            "contact": player_contact_view(profile_id, contact),
            "current": player_current_life_view(background.get("current_role") or {}),
            "events": player_life_events_view(background.get("timeline")),
            "plans": player_intents_view(detail.get("character_intents")),
            "arrangements": arrangements["items"],
            "arrangement_pagination": arrangements["pagination"],
            "problems": problems,
        }

    async def _player_relationship(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.player_profiles is None:
            raise RuntimeError("player profile administration is unavailable")
        profile_id = await self._player_profile_id(payload)
        contact = await self._player_contact(profile_id, payload)
        instance_id = str(contact["instance_id"])
        scope = str(contact["scope"])
        person_ref = str(payload.get("person_ref") or "").strip()
        portrait, knowledge, policies, detail, arrangements = await asyncio.gather(
            self.player_profiles.snapshot(
                profile_id,
                instance_id,
                scope,
                _relationship_portrait_payload(payload, person_ref),
            ),
            self.knowledge.knowledge_snapshot(profile_id, instance_id),
            self.profile_settings.instance_contact_override_snapshot(profile_id, instance_id),
            self._instance_detail(profile_id, instance_id, {"message_page_size": 5}),
            self._player_arrangements(
                profile_id,
                instance_id,
                page=max(1, int(payload.get("arrangement_page") or 1)),
                page_size=max(5, min(int(payload.get("arrangement_page_size") or 10), 20)),
            ),
        )
        problems = await self._player_instance_problems(profile_id, instance_id)
        contact_projection = _relationship_contact_projection(contact, scope, portrait)
        return {
            "contact": player_contact_view(
                profile_id,
                contact_projection,
                latest_at=(detail.get("message_stats") or {}).get("latest_at"),
                problem_count=len(problems),
            ),
            "people": player_people_view(portrait.get("people")),
            "people_pagination": dict(portrait.get("pagination") or _page_view(1, 20, 0)),
            "selected_person_ref": str(portrait.get("selected_person_ref") or ""),
            "selected_display_name": str(portrait.get("selected_display_name") or ""),
            "portrait": player_portrait_view(portrait.get("entries")),
            "memories": player_memories_view(knowledge.get("memories")),
            "arrangements": arrangements["items"],
            "arrangement_pagination": arrangements["pagination"],
            "contact_preferences": _relationship_contact_preferences(policies),
            "problems": problems,
        }

    async def _player_about(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = await self._player_profile_id(payload)
        character, world = await asyncio.gather(
            self.character_models.snapshot(profile_id),
            self.background.world_snapshot(
                profile_id,
                lore_page=1,
                lore_page_size=1,
                boundary_page=max(1, int(payload.get("boundary_page") or 1)),
                boundary_page_size=max(1, min(int(payload.get("boundary_page_size") or 10), 50)),
                boundary_enabled_only=True,
            ),
        )
        return {
            "character": player_character_view(character),
            "world": player_world_view(world),
        }

    async def _release_notes(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return player_release_notes_view()

    async def _player_guide_acknowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        await self.profiles_repository.set_console_preference(
            _PLAYER_GUIDE_PREFERENCE_KEY,
            str(_PLAYER_GUIDE_VERSION),
        )
        return {"ok": True, "version": _PLAYER_GUIDE_VERSION, "seen": True}

    async def _player_guide_state(self) -> dict[str, Any]:
        seen_version = await self.profiles_repository.get_console_preference(
            _PLAYER_GUIDE_PREFERENCE_KEY
        )
        return {
            "version": _PLAYER_GUIDE_VERSION,
            "seen": seen_version == str(_PLAYER_GUIDE_VERSION),
        }

    async def _advanced_guide_acknowledge(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        await self.profiles_repository.set_console_preference(
            _ADVANCED_GUIDE_PREFERENCE_KEY,
            str(_ADVANCED_GUIDE_VERSION),
        )
        return {"ok": True, "version": _ADVANCED_GUIDE_VERSION, "seen": True}

    async def _advanced_guide_state(self) -> dict[str, Any]:
        seen_version = await self.profiles_repository.get_console_preference(
            _ADVANCED_GUIDE_PREFERENCE_KEY
        )
        return {
            "version": _ADVANCED_GUIDE_VERSION,
            "seen": seen_version == str(_ADVANCED_GUIDE_VERSION),
        }

    async def _player_roles(self, payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
        values = [
            value
            for value in (jsonable(item) for item in await self.profiles.sync_profiles())
            if isinstance(value, dict)
        ]
        requested_ref = str(payload.get("role_ref") or "").strip()
        for row in values:
            profile_id = self._row_profile_id(row)
            if requested_ref and player_role_ref(profile_id) == requested_ref:
                return values, profile_id
        preferred = str(
            await self.profiles_repository.get_console_preference(
                "role_settings.selected_profile_id"
            )
            or ""
        )
        known = {self._row_profile_id(row) for row in values}
        selected = (
            preferred if preferred in known else (self._row_profile_id(values[0]) if values else "")
        )
        return values, selected

    async def _player_profile_id(self, payload: Mapping[str, Any]) -> str:
        rows, selected = await self._player_roles(payload)
        if not selected or not rows:
            raise ValueError("还没有可使用的角色")
        return selected

    async def _player_contact(
        self,
        profile_id: str,
        payload: Mapping[str, Any],
        *,
        required: bool = True,
    ) -> dict[str, Any] | None:
        snapshot = await self.profiles.role_instances_snapshot(profile_id)
        instances = list(snapshot.get("instances") or [])
        requested = str(payload.get("contact_ref") or "").strip()
        if requested:
            for item in instances:
                if player_contact_ref(profile_id, str(item.get("instance_id") or "")) == requested:
                    return dict(item)
            raise ValueError("联系人已经变化，请返回联系人页面重新选择")
        if instances:
            return dict(instances[0])
        if required:
            raise ValueError("还没有可以查看的联系人")
        return None

    async def _player_contact_list(self, profile_id: str) -> list[dict[str, Any]]:
        snapshot = await self.profiles.role_instances_snapshot(profile_id)
        instances = list(snapshot.get("instances") or [])
        instance_ids = tuple(str(item.get("instance_id") or "") for item in instances)
        activity = await self.timeline.conversation_repository.list_instance_message_activity(
            profile_id, instance_ids
        )
        chat_policies = await asyncio.gather(
            *(
                self.profiles_repository.get_instance_chat_policy(profile_id, instance_id)
                for instance_id in instance_ids
            )
        )
        chat_policy_by_instance = {str(policy.instance_id): policy for policy in chat_policies}
        preference_keys = {
            instance_id: delivery_failure_preference_key(profile_id, instance_id)
            for instance_id in instance_ids
        }
        preference_values = await self.profiles_repository.get_console_preferences(
            tuple(preference_keys.values())
        )
        acknowledged_by_instance = _delivery_acknowledgements_by_instance(
            preference_keys,
            preference_values,
        )
        problem_counts = await asyncio.gather(
            *(
                self.timeline.delivery_problem_count(
                    profile_id,
                    instance_id,
                    tuple(acknowledged_by_instance.get(instance_id, ())),
                )
                for instance_id in instance_ids
            )
        )
        problem_count_by_instance = dict(zip(instance_ids, problem_counts, strict=True))
        rows = [
            self._player_contact_summary(
                profile_id,
                item,
                activity.get(str(item.get("instance_id") or ""), {}),
                chat_policy_by_instance.get(str(item.get("instance_id") or "")),
                problem_count_by_instance.get(str(item.get("instance_id") or ""), 0),
            )
            for item in instances
        ]
        rows.sort(
            key=lambda item: (
                str(item.get("latest_at") or ""),
                str(item.get("display_name") or ""),
            ),
            reverse=True,
        )
        rows.sort(key=lambda item: item["kind"] == "group")
        return rows

    @staticmethod
    def _player_contact_summary(
        profile_id: str,
        instance: Mapping[str, Any],
        activity: Mapping[str, Any],
        chat_policy: Any,
        delivery_problem_count: int,
    ) -> dict[str, Any]:
        projected_instance = dict(instance)
        if str(instance.get("scope") or "").lower() != "group":
            configured_name = str(
                getattr(chat_policy, "private_fallback_player_name", "") or ""
            ).strip()
            if (
                bool(getattr(chat_policy, "private_name_override_enabled", False))
                and configured_name
            ):
                projected_instance["display_name"] = configured_name
            else:
                observed_name = str(activity.get("latest_sender_name") or "").strip()
                if observed_name:
                    projected_instance["display_name"] = observed_name
        return player_contact_view(
            profile_id,
            projected_instance,
            latest_at=activity.get("latest_at"),
            problem_count=max(0, int(delivery_problem_count)),
        )

    async def _player_instance_problems(
        self, profile_id: str, instance_id: str
    ) -> list[dict[str, Any]]:
        acknowledged = await self._player_acknowledged(profile_id, instance_id)
        summary = await self.timeline.delivery_attention_summary(
            profile_id,
            instance_id,
            acknowledged,
        )
        count = int(summary.get("count") or 0)
        latest = summary.get("latest")
        if count <= 0 or not isinstance(latest, Mapping):
            return []
        view = _outbox_view(latest, acknowledged_failures=acknowledged)
        return [
            {
                "code": "qq_delivery_failed",
                "title": f"有 {count} 条回复没有送到 QQ" if count > 1 else "有一条回复没有送到 QQ",
                "summary": view["last_error"] or "回复已经保留，可以在高级设置中查看。",
                "occurred_at": view["not_before_at"],
                "occurrence_id": view["occurrence_id"],
                "action": "developer-contact",
            }
        ]

    async def _player_acknowledged(self, profile_id: str, instance_id: str) -> frozenset[str]:
        return parse_delivery_failure_acknowledgements(
            await self.profiles_repository.get_console_preference(
                delivery_failure_preference_key(profile_id, instance_id)
            )
        )

    async def _player_arrangements(
        self,
        profile_id: str,
        instance_id: str,
        *,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        if self.timer_repository is None:
            return {"items": [], "pagination": _page_view(1, page_size, 0)}
        scope = TimerScope(profile_id, instance_id)
        rules = []
        cursor = 0
        while True:
            batch = await self.timer_repository.list_rules(
                scope,
                limit=64,
                after_created_sequence=cursor,
            )
            rules.extend(batch.items)
            if batch.next_created_sequence is None:
                break
            cursor = int(batch.next_created_sequence)
        now = datetime.now(UTC)
        result = []
        for rule in rules:
            if str(rule.status.value) != "ACTIVE":
                continue
            due = next_occurrence(rule.schedule, after=now)
            timezone = str(rule.timezone or "UTC")
            result.append(
                {
                    "summary": str(rule.prompt),
                    "when": schedule_summary(rule.schedule, due, timezone, timezone),
                    "due_at": due.isoformat() if due is not None else None,
                }
            )
        size = max(5, min(int(page_size), 20))
        page_count = max(1, (len(result) + size - 1) // size)
        page = max(1, min(int(page), page_count))
        start = (page - 1) * size
        return {
            "items": result[start : start + size],
            "pagination": _page_view(page, size, len(result)),
        }

    @staticmethod
    def _player_readiness(value: Any) -> dict[str, Any]:
        readiness = dict(value or {})
        if bool(readiness.get("ready")):
            return {"ready": True, "problem": None}
        issues = list(readiness.get("issues") or [])
        model_issue = next(
            (item for item in issues if str(item.get("code") or "").startswith("main_model")),
            None,
        )
        return {
            "ready": False,
            "problem": {
                "code": "thinking_unavailable" if model_issue else "role_paused",
                "title": "当前角色还不能思考" if model_issue else "当前角色没有接收新消息",
                "summary": (
                    "还没有可用的思考服务。连接完成后，角色才能在 QQ 回答。"
                    if model_issue
                    else "角色资料会继续保留，可以在高级设置中重新启用。"
                ),
                "action": "developer-models" if model_issue else "developer-settings",
            },
        }

    @staticmethod
    def _row_profile_id(row: Mapping[str, Any]) -> str:
        return str(row.get("profile_id") or row.get("id") or "")

    @staticmethod
    def _plugin_version() -> str:
        from ...version import VERSION

        return f"v{VERSION}"


def _page_view(page: int, page_size: int, total: int) -> dict[str, int | bool]:
    size = max(1, int(page_size))
    count = max(0, int(total))
    page_count = max(1, (count + size - 1) // size)
    bounded_page = max(1, min(int(page), page_count))
    return {
        "page": bounded_page,
        "page_size": size,
        "page_count": page_count,
        "total": count,
        "has_more": bounded_page < page_count,
    }


__all__ = ["PlayerPageActionsMixin"]
