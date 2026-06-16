"""Best-effort Transfermarkt squad-value scraper (isolated + optional).

Kept separate from ``download.py`` so its fragility never affects the core
pipeline. Parses the WC2026 participants table into ``team, squad_value_eur``.
Returns None on any structural surprise rather than raising.
"""
from __future__ import annotations

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .team_names import normalize_team

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}


def _parse_value(text: str) -> float | None:
    """'€1.53bn' / '€947.00m' / '€500k' -> euros as float."""
    t = text.strip().replace("€", "").lower()
    m = re.match(r"([\d.,]+)\s*(bn|m|k)?", t)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    mult = {"bn": 1e9, "m": 1e6, "k": 1e3, None: 1.0}[m.group(2)]
    return num * mult


def scrape_wc2026_squad_values(url: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception:  # noqa: BLE001
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    rows = []
    for row in soup.select("table.items tbody tr"):
        link = row.select_one("td.hauptlink a")
        val_cell = row.select_one("td.rechts")
        if not link or not val_cell:
            continue
        team = normalize_team(link.get_text(strip=True))
        value = _parse_value(val_cell.get_text(strip=True))
        if team and value:
            rows.append({"team": team, "squad_value_eur": value})
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates("team")
