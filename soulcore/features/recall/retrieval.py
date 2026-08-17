"""Cached candidate routes, weighted RRF fusion, rerank, and MMR selection."""

from __future__ import annotations

import math
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ...contracts.text_fingerprint import content_fingerprint
from .domain import RecallRequest
from .policy import (
    DATE_HINT as _DATE_HINT,
)
from .policy import (
    MatrixCacheEntry as _MatrixCacheEntry,
)
from .policy import (
    RecallCandidate as _Candidate,
)
from .policy import RecallPolicyV1
from .ports import RecallRepository
from .providers import AstrBotRecallProviderRegistry, RecallProviderConfigurationError
from .tokenization import (
    fts_query,
    has_distinctive_token_overlap,
    lexical_tokens,
    normalize_text,
    token_containment,
    token_overlap,
)

_LEXICAL_ANCHOR_LIMIT = 5


class RecallRetrievalMixin:
    if TYPE_CHECKING:
        repository: RecallRepository
        providers: AstrBotRecallProviderRegistry
        policy: RecallPolicyV1
        _matrix_cache_size: int
        _document_cache: OrderedDict[tuple[str, str], tuple[dict[str, Any], ...]]
        _graph_cache: OrderedDict[
            tuple[str, str], tuple[list[dict[str, Any]], list[dict[str, Any]]]
        ]
        _scene_cache: OrderedDict[tuple[str, str], list[dict[str, Any]]]
        _entity_index_cache: OrderedDict[tuple[str, str], dict[str, tuple[str, ...]]]
        _adjacency_cache: OrderedDict[
            tuple[str, str, tuple[str, ...]],
            tuple[
                dict[str, dict[str, Any]],
                dict[str, list[tuple[str, float]]],
                dict[str, list[tuple[str, float]]],
            ],
        ]
        _matrix_cache: OrderedDict[tuple[str, str, int], _MatrixCacheEntry]
        _confidence: Callable[[RecallRequest, _Candidate], float]
        _entity_route: Callable[
            [str, str, str, Sequence[Mapping[str, Any]]],
            tuple[list[tuple[str, float]], frozenset[str]],
        ]
        _predicate_score: Callable[[str, Mapping[str, Any]], float]
        _exact_score: Callable[..., float]
        _expansion_seed_quality: Callable[[_Candidate], float]
        _allowed_edges: Callable[[str], set[str]]
        _personalized_page_rank: Callable[
            [Mapping[str, float], Mapping[str, Sequence[tuple[str, float]]]], dict[str, float]
        ]
        _validated_vector: Callable[[Any, int], np.ndarray]
        _ensure_change_pairs: Callable[[list[_Candidate], Sequence[_Candidate], int], None]
        _prefetch_safe: Callable[[RecallRequest, _Candidate], bool]
        _time_sort_key: Callable[[Mapping[str, Any]], tuple[int, float, str]]

    async def _documents(self, profile_id: str, instance_id: str) -> tuple[dict[str, Any], ...]:
        scope = (profile_id, instance_id)
        cached = self._document_cache.get(scope)
        if cached is not None:
            self._document_cache.move_to_end(scope)
            return cached
        rows = await self.repository.list_documents(profile_id, instance_id)
        prepared = tuple(self._prepare_document(item) for item in rows)
        self._remember_scope(self._document_cache, scope, prepared)
        entity_index: dict[str, list[str]] = defaultdict(list)
        for document in prepared:
            for name in document.get("_recall_normalized_names", ()):
                if name:
                    entity_index[str(name)].append(str(document["document_key"]))
        self._remember_scope(
            self._entity_index_cache,
            scope,
            {name: tuple(dict.fromkeys(keys)) for name, keys in entity_index.items()},
        )
        return prepared

    async def _graph(
        self, profile_id: str, instance_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scope = (profile_id, instance_id)
        cached = self._graph_cache.get(scope)
        if cached is not None:
            self._graph_cache.move_to_end(scope)
            return cached
        graph = await self.repository.list_graph(profile_id, instance_id)
        self._remember_scope(self._graph_cache, scope, graph)
        return graph

    async def _scenes(self, profile_id: str, instance_id: str) -> list[dict[str, Any]]:
        scope = (profile_id, instance_id)
        cached = self._scene_cache.get(scope)
        if cached is not None:
            self._scene_cache.move_to_end(scope)
            return cached
        scenes = await self.repository.list_scenes(profile_id, instance_id)
        for scene in scenes:
            scene["_recall_tokens"] = lexical_tokens(scene.get("search_text", ""))
        self._remember_scope(self._scene_cache, scope, scenes)
        return scenes

    async def _adjacency(
        self,
        profile_id: str,
        instance_id: str,
        allowed: set[str],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, list[tuple[str, float]]],
        dict[str, list[tuple[str, float]]],
    ]:
        key = (profile_id, instance_id, tuple(sorted(allowed)))
        cached = self._adjacency_cache.get(key)
        if cached is not None:
            self._adjacency_cache.move_to_end(key)
            return cached
        nodes, graph_edges = await self._graph(profile_id, instance_id)
        nodes_by_key = {str(item["node_key"]): item for item in nodes}
        graph_adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in graph_edges:
            if str(edge.get("edge_type")) in allowed:
                graph_adjacency[str(edge["source_node_key"])].append(
                    (str(edge["target_node_key"]), float(edge.get("weight") or 1.0))
                )
        direct_adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in await self.repository.list_edges(profile_id, instance_id):
            if str(edge.get("edge_type")) in allowed:
                direct_adjacency[str(edge["source_document_key"])].append(
                    (str(edge["target_document_key"]), float(edge.get("weight") or 1.0))
                )
        value = (nodes_by_key, dict(graph_adjacency), dict(direct_adjacency))
        self._remember_scope(self._adjacency_cache, key, value)
        return value

    def _remember_scope(self, cache: OrderedDict[Any, Any], key: Any, value: Any) -> None:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self._matrix_cache_size:
            cache.popitem(last=False)

    @staticmethod
    def _prepare_document(document: dict[str, Any]) -> dict[str, Any]:
        document["_recall_normalized_names"] = tuple(
            normalize_text(item) for item in document.get("entity_names", ())
        )
        document["_recall_normalized_content"] = normalize_text(document.get("content", ""))
        document["_recall_content_tokens"] = lexical_tokens(
            document.get("content", ""), aliases=document.get("entity_names", ())
        )
        document["_recall_content_fingerprint"] = content_fingerprint(document.get("content", ""))
        return document

    async def _candidate_generation(
        self,
        request: RecallRequest,
        documents: Sequence[dict[str, Any]],
        diagnostics: dict[str, Any],
    ) -> list[_Candidate]:
        by_key = {str(item["document_key"]): item for item in documents}
        query_tokens = lexical_tokens(request.need)
        normalized_query = normalize_text(request.need)
        hidden_keys = self._hidden_document_keys(request, documents)
        routable_documents = [
            item for item in documents if str(item["document_key"]) not in hidden_keys
        ]
        routes, exact_entity_keys = await self._base_routes(
            request,
            routable_documents,
            by_key,
            hidden_keys=hidden_keys,
            query_tokens=query_tokens,
            normalized_query=normalized_query,
            diagnostics=diagnostics,
        )
        preliminary = self._fuse(routes, by_key)
        await self._add_expansion_routes(request, preliminary, by_key, routes, hidden_keys)
        candidates = self._fuse(routes, by_key)
        candidates = self._apply_exact_entity_scope(candidates, exact_entity_keys)
        self._apply_lexical_anchor_signals(candidates, query_tokens)
        await self._rerank(request, candidates, diagnostics)
        candidates.sort(key=lambda item: (-item.rrf, item.document["document_key"]))
        for candidate in candidates:
            candidate.confidence = self._confidence(request, candidate)
        diagnostics["routes"] = {name: len(rows) for name, rows in routes.items()}
        diagnostics["candidate_count"] = len(candidates)
        return candidates

    async def _add_expansion_routes(
        self,
        request: RecallRequest,
        preliminary: Sequence[_Candidate],
        by_key: Mapping[str, dict[str, Any]],
        routes: dict[str, list[tuple[str, float]]],
        hidden_keys: Collection[str],
    ) -> None:
        graph, scenes = await self._expansion_routes(request, preliminary, by_key)
        for name, rows in (("graph", graph), ("scene", scenes)):
            visible_rows = [pair for pair in rows if pair[0] not in hidden_keys]
            if visible_rows:
                routes[name] = visible_rows[: self.policy.graph_limit]

    @staticmethod
    def _apply_exact_entity_scope(
        candidates: list[_Candidate], exact_entity_keys: Collection[str]
    ) -> list[_Candidate]:
        # Fuzzy aliases widen recall, but must never become a hard scope. A
        # short unrelated entity can score as a typo candidate; only a literal
        # entity mention may constrain direct routes. Graph and scene evidence
        # remains eligible because it is linked back to the scoped entity.
        if not exact_entity_keys:
            return candidates
        return [
            item
            for item in candidates
            if str(item.document["document_key"]) in exact_entity_keys
            or "graph" in item.route_ranks
            or "scene" in item.route_ranks
        ]

    async def _base_routes(
        self,
        request: RecallRequest,
        documents: Sequence[dict[str, Any]],
        by_key: Mapping[str, dict[str, Any]],
        *,
        hidden_keys: Collection[str],
        query_tokens: Sequence[str],
        normalized_query: str,
        diagnostics: dict[str, Any],
    ) -> tuple[dict[str, list[tuple[str, float]]], frozenset[str]]:
        lexical_rows = [
            row
            for row in await self._lexical_rows(request)
            if str(row["document_key"]) not in hidden_keys
        ]
        routes = {
            "lexical": [
                (
                    str(row["document_key"]),
                    self._lexical_relevance(
                        query_tokens,
                        lexical_tokens(row.get("search_text", "")),
                    ),
                )
                for row in lexical_rows
            ]
        }
        entity, entity_keys = self._entity_route(
            request.profile_id,
            request.instance_id,
            normalized_query,
            documents,
        )
        if entity:
            routes["entity"] = entity[: self.policy.exact_limit]
            routes["predicate"] = self._predicate_route(normalized_query, entity_keys, by_key)
        routes["exact"] = self._exact_route(
            request,
            documents,
            lexical_rows,
            by_key,
            normalized_query=normalized_query,
            query_tokens=query_tokens,
        )
        dense = await self._dense_route(request, diagnostics)
        if dense:
            routes["dense"] = [pair for pair in dense if pair[0] not in hidden_keys][
                : self.policy.dense_limit
            ]
        # Candidate generation limits each route for bounded RRF work, but the
        # exact-entity scope must retain every matching document.  Reusing the
        # truncated top-40 route here makes a frequent participant's older
        # lexical hit disappear solely because its document key sorts later.
        exact_entity_keys = frozenset(key for key, score in entity if score >= 0.95)
        return routes, exact_entity_keys

    @staticmethod
    def _hidden_document_keys(
        request: RecallRequest, documents: Sequence[Mapping[str, Any]]
    ) -> frozenset[str]:
        visible = set(request.visible_source_fingerprints)
        excluded = set(request.excluded_document_keys)
        if not visible and not excluded:
            return frozenset()
        fingerprints = {
            str(document["document_key"]): str(
                document.get("_recall_content_fingerprint")
                or content_fingerprint(document.get("content", ""))
            )
            for document in documents
        }
        excluded_content = {
            fingerprints[str(document["document_key"])]
            for document in documents
            if str(document["document_key"]) in excluded
        }
        return frozenset(
            str(document["document_key"])
            for document in documents
            if str(document["document_key"]) in excluded
            or str(document.get("source_fingerprint") or "") in visible
            or fingerprints[str(document["document_key"])] in visible
            or fingerprints[str(document["document_key"])] in excluded_content
        )

    @staticmethod
    def _lexical_relevance(query_tokens: Sequence[str], document_tokens: Sequence[str]) -> float:
        return max(
            token_overlap(query_tokens, document_tokens),
            token_containment(query_tokens, document_tokens),
        )

    @staticmethod
    def _apply_lexical_anchor_signals(
        candidates: Sequence[_Candidate], query_tokens: Sequence[str]
    ) -> None:
        """Calibrate concise top-BM25 evidence without changing route order.

        Participant labels and provenance text legitimately belong in the FTS
        projection, but they dilute shorter-side overlap for a one-sentence
        correction.  Keep that calibration separate from lexical relevance so
        it cannot alter RRF, graph seeds, or the high-precision prefetch gate.
        """

        for candidate in candidates:
            if candidate.route_ranks.get("lexical", 10**9) > _LEXICAL_ANCHOR_LIMIT:
                continue
            document_tokens = lexical_tokens(candidate.document.get("search_text", ""))
            containment = token_containment(query_tokens, document_tokens)
            if containment >= 0.20 and has_distinctive_token_overlap(query_tokens, document_tokens):
                candidate.signals["lexical_anchor"] = containment

    async def _lexical_rows(self, request: RecallRequest) -> list[dict[str, Any]]:
        query = fts_query(request.need)
        if not query:
            return []
        return await self.repository.fts_search(
            request.profile_id,
            request.instance_id,
            query,
            limit=self.policy.bm25_limit,
        )

    def _predicate_route(
        self,
        normalized_query: str,
        entity_keys: Collection[str],
        by_key: Mapping[str, dict[str, Any]],
    ) -> list[tuple[str, float]]:
        predicate = sorted(
            (
                (key, self._predicate_score(normalized_query, by_key[key]))
                for key in entity_keys
                if key in by_key
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [pair for pair in predicate if pair[1] > 0][: self.policy.exact_limit]

    def _exact_route(
        self,
        request: RecallRequest,
        documents: Sequence[dict[str, Any]],
        lexical_rows: Sequence[dict[str, Any]],
        by_key: Mapping[str, dict[str, Any]],
        *,
        normalized_query: str,
        query_tokens: Sequence[str],
    ) -> list[tuple[str, float]]:
        exact_documents = {
            str(row["document_key"]): by_key[str(row["document_key"])]
            for row in lexical_rows
            if str(row["document_key"]) in by_key
        }
        exact_documents.update(self._dated_exact_documents(documents, normalized_query))
        exact = sorted(
            (
                (
                    str(item["document_key"]),
                    self._exact_score(
                        request.need,
                        item,
                        normalized_query=normalized_query,
                        query_tokens=query_tokens,
                    ),
                )
                for item in exact_documents.values()
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [pair for pair in exact if pair[1] > 0][: self.policy.exact_limit]

    @staticmethod
    def _dated_exact_documents(
        documents: Sequence[dict[str, Any]], normalized_query: str
    ) -> dict[str, dict[str, Any]]:
        hints = _DATE_HINT.findall(normalized_query)
        if not hints:
            return {}
        return {
            str(item["document_key"]): item
            for item in documents
            if any(
                hint
                in " ".join(
                    str(item.get(key) or "") for key in ("occurred_at", "valid_from", "valid_until")
                )
                for hint in hints
            )
        }

    def _fuse(
        self,
        routes: Mapping[str, Sequence[tuple[str, float]]],
        documents: Mapping[str, dict[str, Any]],
    ) -> list[_Candidate]:
        candidates: dict[str, _Candidate] = {}
        weights = {
            "lexical": self.policy.lexical_weight,
            "dense": self.policy.dense_weight,
            "exact": self.policy.exact_weight,
            "entity": self.policy.exact_weight,
            "predicate": self.policy.lexical_weight,
            "graph": self.policy.graph_weight,
            "scene": self.policy.scene_weight,
            "rerank": self.policy.rerank_weight,
        }
        for route, rows in routes.items():
            for rank, (key, signal) in enumerate(rows, start=1):
                document = documents.get(key)
                if document is None:
                    continue
                item = candidates.setdefault(key, _Candidate(document, {}, {}))
                item.route_ranks[route] = rank
                item.signals[route] = max(item.signals.get(route, 0.0), float(signal))
                item.rrf += float(weights.get(route, 1.0)) / (self.policy.rrf_k + rank)
        return list(candidates.values())

    async def _dense_route(
        self, request: RecallRequest, diagnostics: dict[str, Any]
    ) -> list[tuple[str, float]]:
        try:
            return await self._dense_route_ready(request, diagnostics)
        except RecallProviderConfigurationError as exc:
            diagnostics["degradations"].append(
                {"stage": "dense", "code": exc.code, "detail": str(exc)}
            )
            return []
        except Exception as exc:
            diagnostics["degradations"].append(
                {"stage": "dense", "code": type(exc).__name__, "detail": str(exc)[:200]}
            )
            return []

    async def _dense_route_ready(
        self, request: RecallRequest, diagnostics: dict[str, Any]
    ) -> list[tuple[str, float]]:
        selection = await self.providers.selection(self.repository, request.profile_id)
        if not selection.embedding_provider_id:
            return []
        provider = self.providers.embedding(selection.embedding_provider_id)
        assert provider is not None
        fingerprint, dimension = self.providers.embedding_fingerprint(provider)
        active = await self.repository.active_generation(request.profile_id, request.instance_id)
        if (
            active is None
            or str(active.get("provider_fingerprint") or "") != fingerprint
            or int(active.get("vector_dimension") or 0) != dimension
        ):
            diagnostics["degradations"].append({"stage": "dense", "code": "generation_not_ready"})
            return []
        vector = self._validated_vector(await provider.get_embedding(request.need), dimension)
        cache_key = (request.profile_id, request.instance_id, int(active["generation_id"]))
        entry = await self._matrix(cache_key, dimension)
        if entry.matrix.size == 0:
            return []
        scores = entry.matrix @ vector
        indexes = np.argsort(-scores, kind="stable")[: self.policy.dense_limit]
        return [
            (entry.document_keys[int(index)], float(scores[int(index)]))
            for index in indexes
            if math.isfinite(float(scores[int(index)]))
        ]

    async def _matrix(self, cache_key: tuple[str, str, int], dimension: int) -> _MatrixCacheEntry:
        cached = self._matrix_cache.get(cache_key)
        if cached is not None:
            self._matrix_cache.move_to_end(cache_key)
            return cached
        keys, vectors = self._embedding_matrix_rows(
            await self.repository.embedding_rows(cache_key[2]), dimension
        )
        matrix = (
            np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
            if vectors
            else np.empty((0, dimension), dtype=np.float32)
        )
        entry = _MatrixCacheEntry(tuple(keys), matrix)
        self._matrix_cache[cache_key] = entry
        self._matrix_cache.move_to_end(cache_key)
        while len(self._matrix_cache) > self._matrix_cache_size:
            self._matrix_cache.popitem(last=False)
        return entry

    @staticmethod
    def _embedding_matrix_rows(
        rows: Sequence[dict[str, Any]], dimension: int
    ) -> tuple[list[str], list[np.ndarray]]:
        keys: list[str] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            if int(row.get("vector_dimension") or 0) != dimension:
                continue
            vector = np.frombuffer(bytes(row["vector_blob"]), dtype=np.float32)
            if vector.size != dimension or not np.isfinite(vector).all():
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            keys.append(str(row["document_key"]))
            vectors.append(np.ascontiguousarray(vector / norm, dtype=np.float32))
        return keys, vectors

    async def _expansion_routes(
        self,
        request: RecallRequest,
        preliminary: Sequence[_Candidate],
        documents: Mapping[str, dict[str, Any]],
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        visible = set(request.visible_source_fingerprints)
        excluded = set(request.excluded_document_keys)
        seed = {}
        for item in preliminary[:30]:
            document = item.document
            document_key = str(document["document_key"])
            if document_key in excluded:
                continue
            if str(document.get("source_fingerprint") or "") in visible:
                continue
            if content_fingerprint(document.get("content", "")) in visible:
                continue
            quality = self._expansion_seed_quality(item)
            if quality >= 0.32:
                seed[document_key] = item.rrf * quality
        if not seed:
            return [], []
        graph = await self._graph_expansion(request, seed, documents)
        scenes = await self._scene_expansion(request, seed, documents)
        return graph, scenes

    async def _graph_expansion(
        self,
        request: RecallRequest,
        seed: Mapping[str, float],
        documents: Mapping[str, dict[str, Any]],
    ) -> list[tuple[str, float]]:
        nodes, graph_adjacency, direct_adjacency = await self._adjacency(
            request.profile_id,
            request.instance_id,
            self._allowed_edges(request.need),
        )
        graph_seed = {
            str(node["node_key"]): seed[document_key]
            for node in nodes.values()
            if (document_key := str(node.get("document_key") or "")) in seed
        }
        document_mass: dict[str, float] = defaultdict(float)
        for node_key, score in self._personalized_page_rank(graph_seed, graph_adjacency).items():
            node = nodes.get(node_key)
            document_key = str(node.get("document_key") or "") if node else ""
            if document_key and document_key not in seed:
                document_mass[document_key] = max(document_mass[document_key], score)
        for key, score in self._personalized_page_rank(seed, direct_adjacency).items():
            if key not in seed:
                document_mass[key] = max(document_mass[key], score)
        return sorted(
            ((key, score) for key, score in document_mass.items() if key in documents),
            key=lambda pair: (-pair[1], pair[0]),
        )

    async def _scene_expansion(
        self,
        request: RecallRequest,
        seed: Mapping[str, float],
        documents: Mapping[str, dict[str, Any]],
    ) -> list[tuple[str, float]]:
        query_tokens = lexical_tokens(request.need)
        scenes = await self._scenes(request.profile_id, request.instance_id)
        scored = sorted(
            (
                (
                    scene,
                    token_overlap(
                        query_tokens,
                        scene.get("_recall_tokens") or lexical_tokens(scene.get("search_text", "")),
                    ),
                )
                for scene in scenes
            ),
            key=lambda pair: (-pair[1], str(pair[0].get("scene_key") or "")),
        )
        members: dict[str, float] = {}
        for scene, score in scored[:10]:
            if score < 0.22:
                continue
            for key in scene.get("member_keys", ()):
                if key in documents and key not in seed:
                    members[key] = max(members.get(key, 0.0), float(score))
        return sorted(members.items(), key=lambda pair: (-pair[1], pair[0]))

    async def _rerank(
        self,
        request: RecallRequest,
        candidates: list[_Candidate],
        diagnostics: dict[str, Any],
    ) -> None:
        candidates.sort(key=lambda item: (-item.rrf, item.document["document_key"]))
        top = candidates[: self.policy.rerank_limit]
        if not top:
            return
        try:
            await self._apply_rerank(request, top)
        except RecallProviderConfigurationError as exc:
            diagnostics["degradations"].append(
                {"stage": "rerank", "code": exc.code, "detail": str(exc)}
            )
        except Exception as exc:
            diagnostics["degradations"].append(
                {"stage": "rerank", "code": type(exc).__name__, "detail": str(exc)[:200]}
            )

    async def _apply_rerank(self, request: RecallRequest, candidates: Sequence[_Candidate]) -> None:
        selection = await self.providers.selection(self.repository, request.profile_id)
        if not selection.rerank_provider_id:
            return
        provider = self.providers.rerank(selection.rerank_provider_id)
        assert provider is not None
        results = await provider.rerank(
            request.need,
            [str(item.document.get("content") or "") for item in candidates],
            top_n=len(candidates),
        )
        for rank, result in enumerate(results, start=1):
            index = int(getattr(result, "index", -1))
            score = float(getattr(result, "relevance_score", 0.0))
            if not 0 <= index < len(candidates) or not math.isfinite(score):
                continue
            item = candidates[index]
            item.route_ranks["rerank"] = rank
            item.signals["rerank"] = score
            item.rrf += self.policy.rerank_weight / (self.policy.rrf_k + rank)


__all__ = ["RecallRetrievalMixin"]
