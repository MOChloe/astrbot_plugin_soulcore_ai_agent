"""Application service for saves, revision freezing, and closed projections."""

from __future__ import annotations

from .compiler import CharacterProjectionCompiler, budget_rendered_character_projection
from .domain import (
    MAX_IDEMPOTENCY_KEY_CHARS,
    MAX_PROFILE_ID_CHARS,
    CharacterModel,
    CharacterModelError,
    CharacterModelIdempotencyConflict,
    CharacterModelNotFound,
    CharacterModelRevisionConflict,
    CharacterModelSave,
    CharacterModelSnapshot,
    CharacterProjection,
    CharacterTriggerEvaluation,
    FrozenCharacterModel,
    ProjectionPurpose,
    model_completion,
    model_content_fingerprint,
    normalize_character_model,
    normalize_trigger_match_text,
    save_request_fingerprint,
)
from .ports import CharacterModelRepositoryPort


class CharacterModelService:
    def __init__(
        self,
        repository: CharacterModelRepositoryPort,
        compiler: CharacterProjectionCompiler | None = None,
    ) -> None:
        self.repository = repository
        self.compiler = compiler or CharacterProjectionCompiler()

    async def get_current(self, profile_id: str) -> CharacterModelSnapshot | None:
        return await self.repository.load(self._profile_id(profile_id))

    async def freeze(self, profile_id: str) -> FrozenCharacterModel:
        normalized = self._profile_id(profile_id)
        snapshot = await self.repository.load(normalized)
        if snapshot is None:
            try:
                snapshot = await self.save_model(
                    normalized,
                    CharacterModel(),
                    expected_revision=0,
                    idempotency_key="system-empty-character-model-v5",
                )
            except (CharacterModelRevisionConflict, CharacterModelIdempotencyConflict):
                snapshot = await self.repository.load(normalized)
            if snapshot is None:
                raise CharacterModelNotFound(f"character model not found: {normalized}")
        return FrozenCharacterModel(
            profile_id=snapshot.profile_id,
            revision=snapshot.revision,
            content_fingerprint=snapshot.content_fingerprint,
        )

    async def _load_revision(self, profile_id: str, revision: int) -> CharacterModelSnapshot:
        normalized = self._profile_id(profile_id)
        snapshot = await self.repository.load(normalized, int(revision))
        if snapshot is None:
            raise CharacterModelNotFound(
                f"character model revision not found: {normalized}@{int(revision)}"
            )
        return snapshot

    async def save_model(
        self,
        profile_id: str,
        model: CharacterModel,
        *,
        expected_revision: int,
        idempotency_key: str,
    ) -> CharacterModelSnapshot:
        normalized_profile = self._profile_id(profile_id)
        expected = int(expected_revision)
        if expected < 0:
            raise CharacterModelError("expected_revision must be non-negative")
        key = str(idempotency_key or "").strip()
        if not key or len(key) > MAX_IDEMPOTENCY_KEY_CHARS:
            raise CharacterModelError(
                f"idempotency_key must contain 1-{MAX_IDEMPOTENCY_KEY_CHARS} characters"
            )
        normalized_model = normalize_character_model(model)
        content_fingerprint = model_content_fingerprint(normalized_model)
        command = CharacterModelSave(
            profile_id=normalized_profile,
            expected_revision=expected,
            idempotency_key=key,
            request_fingerprint=save_request_fingerprint(expected, content_fingerprint),
            content_fingerprint=content_fingerprint,
            model=normalized_model,
            completion=model_completion(normalized_model),
        )
        return await self.repository.save(command)

    async def project(
        self,
        frozen: FrozenCharacterModel,
        purpose: ProjectionPurpose,
        *,
        relevance_text: str = "",
    ) -> CharacterProjection:
        snapshot = await self._load_revision(frozen.profile_id, frozen.revision)
        if snapshot.content_fingerprint != frozen.content_fingerprint:
            raise CharacterModelError("frozen character model fingerprint mismatch")
        return self.compiler.compile(snapshot, purpose, relevance_text=relevance_text)

    async def evaluate_triggers(
        self,
        frozen: FrozenCharacterModel,
        inbound_turns: tuple[str, ...],
    ) -> CharacterTriggerEvaluation:
        snapshot = await self._load_revision(frozen.profile_id, frozen.revision)
        if snapshot.content_fingerprint != frozen.content_fingerprint:
            raise CharacterModelError("frozen character model fingerprint mismatch")
        turns = tuple(normalize_trigger_match_text(value) for value in inbound_turns)
        contents: list[str] = []
        seen_contents: set[str] = set()
        matched_rule_count = 0
        for rule in snapshot.model.trigger_rules:
            window = turns[: rule.lookback_turns]
            if not any(
                normalize_trigger_match_text(key) in turn for key in rule.keys for turn in window
            ):
                continue
            matched_rule_count += 1
            if rule.content in seen_contents:
                continue
            seen_contents.add(rule.content)
            contents.append(rule.content)
        searched = min(
            len(turns),
            max((rule.lookback_turns for rule in snapshot.model.trigger_rules), default=0),
        )
        return CharacterTriggerEvaluation(tuple(contents), matched_rule_count, searched)

    @staticmethod
    def _profile_id(value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized or len(normalized) > MAX_PROFILE_ID_CHARS:
            raise CharacterModelError(
                f"profile_id must contain 1-{MAX_PROFILE_ID_CHARS} characters"
            )
        return normalized


__all__ = ["CharacterModelService", "budget_rendered_character_projection"]
