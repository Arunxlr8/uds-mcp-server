"""
Tiered semantic matcher for UDS service search.

Stage 1: SpacyCosineRanker produces a shortlist by embedding similarity.
Stage 2: An LLM is shown ONLY that shortlist and asked to confirm whether any
         of them genuinely answer the query.
Stage 3 (escalation): If the LLM rejects the entire shortlist, it is shown the
         FULL catalogue and asked again — covers cases where the embedding
         ranker's top-k missed the actual correct answer.

Disabled by default — enable with UDS_ENABLE_LLM_MATCH=true in .env, which
requires valid Azure OpenAI credentials and the en_core_web_lg spaCy model.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from src.repository import DiagnosticService, ISemanticMatcher, ISimilarityRanker

load_dotenv()

_SHORTLIST_PROMPT = (
    "You are verifying candidate matches for a UDS (ISO 14229) diagnostic service "
    "lookup. Below is a SHORTLIST of candidates, already pre-ranked by embedding "
    "similarity to the user's query. Decide whether any of them genuinely and "
    "correctly answer the query. Respond with ONLY a JSON array of the exact "
    "service names from the shortlist that correctly answer it, most relevant "
    "first. Do not invent names. If none of them are actually correct, respond "
    "with exactly: []"
)

_FULL_CATALOGUE_PROMPT = (
    "The initial shortlist did not contain a correct answer. Below is the "
    "COMPLETE catalogue of UDS (ISO 14229) diagnostic services. Carefully "
    "determine whether any service in it genuinely answers the user's query. "
    "Respond with ONLY a JSON array of exact service names from the catalogue "
    "that correctly answer it, most relevant first. Do not invent names that "
    "are not in the catalogue. If truly nothing matches, respond with exactly: []"
)


class TieredSemanticMatcher(ISemanticMatcher):
    """
    Combines an embedding ranker (candidate generation) with an LLM
    (verification + full-catalogue escalation). Neither stage decides alone:
    the ranker only proposes, the LLM only confirms or searches wider.
    """

    def __init__(self, ranker: ISimilarityRanker, candidate_k: int = 5) -> None:
        self._ranker = ranker
        self._candidate_k = candidate_k
        self._llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            temperature=0.2,
        )

    def find_best_matches(
        self, query: str, services: Iterable[DiagnosticService], limit: int = 3
    ) -> list[DiagnosticService]:
        services = list(services)

        ranked = self._ranker.rank(query, services, top_k=self._candidate_k)
        shortlist = [service for service, _score in ranked]

        verified = self._ask_llm(query, shortlist, _SHORTLIST_PROMPT)
        if verified:
            return verified[:limit]

        print(
            "[TieredSemanticMatcher] Shortlist rejected; escalating to full catalogue.",
            file=sys.stderr,
        )
        return self._ask_llm(query, services, _FULL_CATALOGUE_PROMPT)[:limit]

    def _ask_llm(
        self, query: str, pool: list[DiagnosticService], system_prompt: str
    ) -> list[DiagnosticService]:
        by_name = {s.name.lower(): s for s in pool}
        catalogue = "\n".join(
            f"{i + 1}. {s.name} (SID {s.request_sid}): {s.description}"
            for i, s in enumerate(pool)
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Candidates:\n{catalogue}\n\nUser query: {query}"),
        ]

        try:
            response = self._llm.invoke(messages)
            names = self._parse_names(response.content)
        except Exception as exc:  # network/auth/parsing failure — fail soft, never crash the tool
            print(f"[TieredSemanticMatcher] LLM call failed: {exc}", file=sys.stderr)
            return []

        return [by_name[n.lower()] for n in names if n.lower() in by_name]

    @staticmethod
    def _parse_names(raw_text: str) -> list[str]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return []