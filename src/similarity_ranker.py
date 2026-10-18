"""
Embedding-based candidate ranking for UDS service search.

Uses spaCy's en_core_web_lg model (pretrained word vectors) to rank services
against a natural-language query by cosine similarity. This only produces a
SHORTLIST of candidates — it does not decide the final answer. That decision
belongs to TieredSemanticMatcher, which verifies the shortlist with an LLM.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import spacy

from src.repository import DiagnosticService, ISimilarityRanker


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SpacyCosineRanker(ISimilarityRanker):
    """Ranks services by cosine similarity between averaged word vectors."""

    def __init__(self, model_name: str = "en_core_web_lg") -> None:
        self._nlp = spacy.load(model_name)
        self._vector_cache: dict[str, np.ndarray] = {}

    def _vector_for(self, service: DiagnosticService) -> np.ndarray:
        cached = self._vector_cache.get(service.name)
        if cached is not None:
            return cached
        # Name + description together, so the embedding reflects the full
        # meaning of the service, not just its often-terse catalogue name.
        text = f"{service.name}. {service.description}"
        vector = self._nlp(text).vector
        self._vector_cache[service.name] = vector
        return vector

    def rank(
        self, query: str, services: Iterable[DiagnosticService], top_k: int = 5
    ) -> list[tuple[DiagnosticService, float]]:
        query_vector = self._nlp(query).vector
        scored = [
            (service, _cosine_similarity(query_vector, self._vector_for(service)))
            for service in services
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]