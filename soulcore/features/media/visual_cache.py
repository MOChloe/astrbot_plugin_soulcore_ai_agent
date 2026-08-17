from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from ...contracts.ai_models import AIVisionDescription
from ...contracts.vision import VisionTextState

VISUAL_OBSERVATION_CONTRACT_VERSION = 3
VISUAL_OBSERVATION_CACHE_HIGH_WATER = 500
VISUAL_OBSERVATION_CACHE_LOW_WATER = 450

_VISIBLE_TEXT_STATES = frozenset(item.value for item in VisionTextState)
_FIELD_LIMITS = {
    "visible_facts": 5000,
    "ocr_text": 2000,
    "subject_identity": 120,
    "scene_description": 5000,
    "visual_style": 120,
    "sticker_type": 120,
    "social_impression": 80,
    "visible_text_state": 40,
    "backend_id": 200,
    "model_id": 200,
}


class VisualCachePolicy(StrEnum):
    USE = "USE"
    REFRESH = "REFRESH"
    BYPASS = "BYPASS"


def _bounded_text(value: object, field: str) -> str:
    return str(value or "").strip()[: _FIELD_LIMITS[field]]


@dataclass(frozen=True, slots=True)
class CachedVisualObservation:
    visible_facts: str
    ocr_text: str
    subject_identity: str
    sequence_observation: str
    visual_style: str
    sticker_type: str
    visible_text_state: str
    safe: bool
    backend_id: str
    model_id: str
    safety_reason: str = ""
    social_impression: str = ""

    @classmethod
    def from_vision(
        cls,
        output: AIVisionDescription,
        *,
        backend_id: str,
    ) -> CachedVisualObservation | None:
        raw = dict(output.raw) if isinstance(output.raw, Mapping) else {}
        safe = raw.get("safe", output.safe)
        visible_facts = _bounded_text(output.visible_facts, "visible_facts")
        visible_text_state = _bounded_text(output.visible_text_state, "visible_text_state").upper()
        if (
            not visible_facts
            or visible_text_state not in _VISIBLE_TEXT_STATES
            or not isinstance(safe, bool)
        ):
            return None
        return cls(
            visible_facts=visible_facts,
            ocr_text=_bounded_text(output.ocr_text, "ocr_text"),
            subject_identity=_bounded_text(output.subject_identity, "subject_identity"),
            sequence_observation=_bounded_text(output.sequence_observation, "scene_description"),
            visual_style=_bounded_text(output.visual_style, "visual_style"),
            sticker_type=_bounded_text(output.sticker_type, "sticker_type"),
            visible_text_state=visible_text_state,
            safe=safe,
            backend_id=_bounded_text(backend_id, "backend_id"),
            model_id=_bounded_text(output.model, "model_id"),
            safety_reason=_bounded_text(output.safety_reason, "scene_description"),
            social_impression=_bounded_text(output.social_impression, "social_impression"),
        )

    @classmethod
    def from_record(cls, value: object) -> CachedVisualObservation | None:
        if not isinstance(value, Mapping):
            return None
        safe = value.get("safe")
        visible_facts = _bounded_text(value.get("visible_facts"), "visible_facts")
        aux = _cache_aux(value.get("scene_description"))
        visible_text_state = _cached_text_state(
            value.get("visible_text_state"),
            aux.get("text_state"),
            value.get("ocr_text"),
        )
        if (
            not visible_facts
            or visible_text_state not in _VISIBLE_TEXT_STATES
            or not isinstance(safe, bool)
        ):
            return None
        return cls(
            visible_facts=visible_facts,
            ocr_text=_bounded_text(value.get("ocr_text"), "ocr_text"),
            subject_identity=_bounded_text(value.get("subject_identity"), "subject_identity"),
            sequence_observation=_bounded_text(
                aux.get("sequence_observation"),
                "scene_description",
            ),
            visual_style=_bounded_text(value.get("visual_style"), "visual_style"),
            sticker_type=_bounded_text(value.get("sticker_type"), "sticker_type"),
            visible_text_state=visible_text_state,
            safe=safe,
            backend_id=_bounded_text(value.get("backend_id"), "backend_id"),
            model_id=_bounded_text(value.get("model_id"), "model_id"),
            safety_reason=_bounded_text(aux.get("safety_reason"), "scene_description"),
            social_impression=_bounded_text(
                aux.get("social_impression"),
                "social_impression",
            ),
        )

    def description(self) -> AIVisionDescription:
        return AIVisionDescription(
            visible_facts=self.visible_facts,
            ocr_text=self.ocr_text,
            subject_identity=self.subject_identity,
            sequence_observation=self.sequence_observation,
            visual_style=self.visual_style,
            sticker_type=self.sticker_type,
            social_impression=self.social_impression,
            visible_text_state=self.visible_text_state,
            safe=self.safe,
            safety_reason=self.safety_reason,
            model=self.model_id,
            raw={"safe": self.safe},
        )

    def physical_text_state(self) -> str:
        return (
            "NO_VISIBLE_TEXT"
            if self.visible_text_state == VisionTextState.NO_TEXT.value
            else "HAS_VISIBLE_TEXT"
        )

    def cache_scene_payload(self) -> str:
        return json.dumps(
            {
                "sequence_observation": self.sequence_observation,
                "text_state": self.visible_text_state,
                "safety_reason": self.safety_reason,
                "social_impression": self.social_impression,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _cache_aux(value: object) -> Mapping[str, object]:
    try:
        parsed = json.loads(str(value or ""))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _cached_text_state(
    physical_state: object,
    objective_state: object,
    ocr_text: object,
) -> str:
    state = str(objective_state or "").strip().upper()
    if state in _VISIBLE_TEXT_STATES:
        return state
    physical = str(physical_state or "").strip().upper()
    if physical == "NO_VISIBLE_TEXT":
        return VisionTextState.NO_TEXT.value
    if physical == "HAS_VISIBLE_TEXT":
        return (
            VisionTextState.TRANSCRIBED.value
            if str(ocr_text or "").strip()
            else VisionTextState.UNCLEAR_TEXT.value
        )
    return ""


__all__ = [
    "CachedVisualObservation",
    "VISUAL_OBSERVATION_CACHE_HIGH_WATER",
    "VISUAL_OBSERVATION_CACHE_LOW_WATER",
    "VISUAL_OBSERVATION_CONTRACT_VERSION",
    "VisualCachePolicy",
]
