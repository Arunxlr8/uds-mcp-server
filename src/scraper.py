"""
One-off ingestion script: scrape the UDS services table from Wikipedia using
LangChain's WebBaseLoader, and write it to data/uds_services.json.

Run once (or whenever the source changes):
    uv run python -m src.scraper
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

SOURCE_URL = os.getenv(
    "UDS_SOURCE_URL",
    "https://en.wikipedia.org/wiki/Unified_Diagnostic_Services",
)
OUTPUT_PATH = Path(
    os.getenv("UDS_OUTPUT_PATH", "data/uds_services.json")
)
USER_AGENT = os.getenv("UDS_USER_AGENT", "uds-mcp-server/0.1 (educational project)")

_HEX_PATTERN = re.compile(r"0x[0-9a-fA-F]{2}", re.IGNORECASE)
_FOOTNOTE_PATTERN = re.compile(r"\[\d+\]")


def _clean(text: str) -> str:
    """Strip footnote markers like '[2]' and collapse whitespace."""
    text = _FOOTNOTE_PATTERN.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_soup():
    """Use LangChain's WebBaseLoader to fetch the page and return a BeautifulSoup object."""
    loader = WebBaseLoader(
        web_path=SOURCE_URL,
        requests_kwargs={"headers": {"User-Agent": USER_AGENT}, "timeout": 30},
    )
    return loader.scrape()  # returns a bs4.BeautifulSoup object


def _find_services_table(soup):
    """
    Locate the specific UDS services table.

    The page header is "Services", the sub-header is "SID (Service Identifier)",
    and the table columns are:
        Request Service Identifier (SID) | Response Service Identifier (RSID)
        | Service name | Description

    We match on the request/response SID header phrases rather than on
    "description" alone, because other tables further down the page
    (e.g. Negative Response Codes) also contain a description-like column.
    """
    for table in soup.select("table.wikitable"):        # table.wikitable -> Wikipedia's default table style
        header_cells = table.find_all("th")     # th -> Table Header..
        header_blob = " ".join(_clean(th.get_text()).lower() for th in header_cells)
        if "request service identifier" in header_blob and "response service identifier" in header_blob:
            return table
    raise RuntimeError(
        "Could not locate the UDS Services table. "
        "The page structure may have changed — inspect the fetched HTML."
    )


def _column_index(headers: list[str], *keywords: str) -> int | None:
    for idx, header in enumerate(headers):
        if any(keyword in header for keyword in keywords):
            return idx
    return None


def parse_services(soup) -> list[dict]:
    table = _find_services_table(soup)

    header_row = table.find("tr")
    headers = [_clean(cell.get_text()).lower() for cell in header_row.find_all(["th", "td"])]

    request_col = _column_index(headers, "request service identifier")
    response_col = _column_index(headers, "response service identifier")
    name_col = _column_index(headers, "service name", "service")
    desc_col = _column_index(headers, "description")

    if None in (request_col, response_col, name_col, desc_col):
        raise RuntimeError(f"Unexpected table headers: {headers}")

    services: list[dict] = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) <= max(request_col, response_col, name_col, desc_col):
            continue  # malformed or spacer row

        request_text = _clean(cells[request_col].get_text())
        response_text = _clean(cells[response_col].get_text())

        request_match = _HEX_PATTERN.search(request_text)
        response_match = _HEX_PATTERN.search(response_text)
        if not request_match:
            continue  # skip rows that aren't actual service entries

        services.append(
            {
                "name": _clean(cells[name_col].get_text()),
                "request_sid": request_match.group(0).lower(),
                "response_sid": (response_match.group(0).lower() if response_match else None),
                "description": _clean(cells[desc_col].get_text()),
            }
        )
    return services


def main() -> int:
    print(f"Fetching {SOURCE_URL} via LangChain WebBaseLoader ...")
    soup = _fetch_soup()

    services = parse_services(soup)
    if not services:
        print("No services parsed; aborting without overwriting the data file.")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(services, fh, indent=2)

    print(f"Wrote {len(services)} services to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())