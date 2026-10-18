"""
UDS MCP server entry point.

Exposes one tool, `diagnostic_lookup`, that accepts:
  * a request SID            -> "0x31"          -> service details
  * a service name           -> "RoutineControl" -> service details
  * a natural-language query -> "reset the ECU"  -> best-matching services
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from src.repository import (
    DiagnosticService,
    ISemanticMatcher,
    IServiceRepository,
    JsonFileDataSource,
    UdsServiceRepository,
    looks_like_sid,
)

load_dotenv()

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "uds_services.json"

mcp = FastMCP("uds-diagnostics")


def _build_matcher() -> Optional[ISemanticMatcher]:
    if os.getenv("UDS_ENABLE_LLM_MATCH", "false").strip().lower() != "true":
        return None
    from src.llm_matcher import AzureLLMMatcher  # imported lazily: no hard Azure dependency

    return AzureLLMMatcher()


def _build_repository() -> IServiceRepository:
    return UdsServiceRepository(JsonFileDataSource(DATA_PATH), matcher=_build_matcher())


repository: IServiceRepository = _build_repository()


def _format(service: DiagnosticService) -> dict:
    return {
        "name": service.name,
        "request_sid": service.request_sid,
        "response_sid": service.response_sid,
        "description": service.description,
    }


@mcp.tool()
def diagnostic_lookup(query: str) -> str:
    """Look up a UDS diagnostic service.

    Accepts a request SID (e.g. "0x31"), a service name (e.g. "RoutineControl"),
    or a natural-language description (e.g. "reset the ECU"). Returns the service
    name, request SID, response SID, and description as JSON.
    """
    text = query.strip()
    if not text:
        return json.dumps({"error": "Query must not be empty."})

    if looks_like_sid(text):
        service = repository.get_by_sid(text)
        if service is None:
            return json.dumps({"error": f"No UDS service with SID {text}."})
        return json.dumps(_format(service), indent=2)

    service = repository.get_by_name(text)
    if service is not None:
        return json.dumps(_format(service), indent=2)

    matches = repository.search_by_description(text)
    if not matches:
        return json.dumps({"error": f"No UDS service matches '{text}'."})
    return json.dumps({"matches": [_format(m) for m in matches]}, indent=2)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()