"""
Domain layer for UDS (ISO 14229) diagnostic services.

Defines the abstract interfaces (so new domains like HIL or CAN signals can
implement the same contracts later without touching this file) and the
concrete UDS implementation backed by the scraped JSON file.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosticService:
    """One UDS service, as produced by scraper.py."""

    name: str
    request_sid: str
    response_sid: Optional[str]
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Interfaces — pure abstractions, no spaCy/LangChain/Azure dependency here
# ---------------------------------------------------------------------------

class IServiceRepository(ABC):
    """Read-only access to a catalogue of diagnostic services."""

    @abstractmethod
    def get_by_sid(self, sid: str) -> Optional[DiagnosticService]:
        """Exact lookup by request SID, e.g. '0x31'."""

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[DiagnosticService]:
        """Exact, case-insensitive lookup by service name, e.g. 'RoutineControl'."""

    @abstractmethod
    def search_by_description(self, query: str, limit: int = 3) -> list[DiagnosticService]:
        """Natural-language search, delegated entirely to the injected matcher."""


class IDataSource(ABC):
    """Supplies raw service records. The repository doesn't care if it's JSON, a DB, etc."""

    @abstractmethod
    def load(self) -> Iterable[DiagnosticService]:
        ...


class ISimilarityRanker(ABC):
    """
    Optional collaborator: ranks services against a query by embedding similarity.
    Returns (service, score) pairs — a shortlist, not a final answer.
    """

    @abstractmethod
    def rank(
        self, query: str, services: Iterable[DiagnosticService], top_k: int = 5
    ) -> list[tuple[DiagnosticService, float]]:
        ...


class ISemanticMatcher(ABC):
    """
    Optional collaborator: decides the final natural-language match, typically
    by combining a similarity ranker with an LLM verification step.
    """

    @abstractmethod
    def find_best_matches(
        self, query: str, services: Iterable[DiagnosticService], limit: int = 3
    ) -> list[DiagnosticService]:
        ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class JsonFileDataSource(IDataSource):
    """Loads services from the JSON file produced by scraper.py."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> Iterable[DiagnosticService]:
        with self._path.open(encoding="utf-8") as fh:
            records = json.load(fh)
        return [DiagnosticService(**r) for r in records]


_SID_PATTERN = re.compile(r"^0x[0-9a-fA-F]{1,2}$", re.IGNORECASE)


def normalize_sid(sid: str) -> str:
    """Turn '0X31', ' 31 ', '0x031' etc. into a canonical '0x31'."""
    cleaned = sid.strip().lower()
    if not cleaned.startswith("0x"):
        cleaned = "0x" + cleaned
    value = int(cleaned, 16)
    return f"0x{value:02x}"


def looks_like_sid(text: str) -> bool:
    return bool(_SID_PATTERN.match(text.strip()))


class UdsServiceRepository(IServiceRepository):
    """UDS implementation of IServiceRepository."""

    def __init__(self, source: IDataSource, matcher: Optional[ISemanticMatcher] = None) -> None:
        services = list(source.load())
        self._by_sid = {normalize_sid(s.request_sid): s for s in services}
        self._by_name = {s.name.lower(): s for s in services}
        self._services = services
        self._matcher = matcher

    def get_by_sid(self, sid: str) -> Optional[DiagnosticService]:
        try:
            return self._by_sid.get(normalize_sid(sid))
        except ValueError:
            return None

    def get_by_name(self, name: str) -> Optional[DiagnosticService]:
        return self._by_name.get(name.strip().lower())

    def search_by_description(self, query: str, limit: int = 3) -> list[DiagnosticService]:
        """
        UDS service names/descriptions are a fixed, standardized vocabulary
        (ISO 14229) — we do not tokenize, split, or rewrite them. All
        free-text matching is delegated to the injected semantic matcher.
        With no matcher configured, this correctly returns no matches rather
        than guessing with weak keyword heuristics.
        """
        if self._matcher is None:
            return []
        return self._matcher.find_best_matches(query, self._services, limit)