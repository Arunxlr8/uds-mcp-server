"""
Optional LLM-backed semantic matcher for UDS service search.

Used by UdsServiceRepository as a fallback when plain keyword matching
(search_by_description) finds nothing. Disabled by default — enable with
UDS_ENABLE_LLM_MATCH=true in .env, which requires valid Azure OpenAI
credentials.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from src.repository import DiagnosticService, ISemanticMatcher

load_dotenv()

_SYSTEM_PROMPT = (
    "You are matching a user's question to UDS (ISO 14229) diagnostic services. "
    "You will be given a numbered catalogue of services and a user query. "
    "Respond with ONLY a JSON array of the exact service names from the catalogue "
    "that best answer the query, most relevant first. Do not invent service names "
    "that are not in the catalogue. If nothing matches, respond with []."
)


class AzureLLMMatcher(ISemanticMatcher):
    """Ranks services using Azure OpenAI, grounded strictly in the given catalogue."""

    def __init__(self) -> None:
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
        catalogue = "\n".join(
            f"{i + 1}. {s.name} (SID {s.request_sid}): {s.description}"
            for i, s in enumerate(services)
        )
        by_name = {s.name.lower(): s for s in services}

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Catalogue:\n{catalogue}\n\nUser query: {query}"),
        ]

        try:
            response = self._llm.invoke(messages)
            names = self._parse_names(response.content)
        except Exception as exc:  # network/auth/parsing failure — fail soft, don't crash the tool
            print(f"[AzureLLMMatcher] LLM call failed, returning no matches: {exc}")
            return []

        matched = [by_name[n.lower()] for n in names if n.lower() in by_name]
        return matched[:limit]

    @staticmethod
    def _parse_names(raw_text: str) -> list[str]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Model sometimes wraps the array in prose or a code fence — extract it.
            match = re.search(r"\[.*\]", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return []