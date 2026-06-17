"""Capture real TheStatsAPI response shapes (run on the VPS — local sandbox is firewall-blocked).

Read-only. Pulls one ``/matches`` page for the 2026 World Cup, then one match's ``/stats``,
``/odds`` and ``/lineups``, prints a concise summary (what id/team-name fields actually exist, and
whether ``fixture_map`` already resolves them), and saves the raw JSON under
``data/raw/thestatsapi_samples/`` so the shapes can be locked into tests offline afterwards.

Why: every per-game feature (xG, odds incl. BTTS, confirmed XI) is keyed by TheStatsAPI's ``mt_``
match id. The matcher in ``src/data/fixture_map.py`` is shape-tolerant by necessity (the listing
shape was never captured to a fixture). This script removes the guessing.

Usage on the VPS:
    THESTATSAPI_KEY=... python scripts/capture_thestatsapi_shapes.py
(optionally ``SEASON_ID=...`` to pin the 2026 season).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import thestatsapi as ts          # noqa: E402
from src.data import fixture_map as fm           # noqa: E402


def _p(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)[:2000]


def main() -> int:
    if not ts.is_available():
        print("NO KEY — set THESTATSAPI_KEY in the environment first.")
        return 2
    print(f"connectivity: {ts.connectivity_check()}")

    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "thestatsapi_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    season = os.environ.get("SEASON_ID") or None
    ms = ts.matches(season_id=season, ttl=0.0)        # fresh
    print(f"\n/matches returned {len(ms)} items (season_id={season!r}).")
    if not ms:
        print("No matches — check competition/season id.")
        return 1
    (out_dir / "matches_sample.json").write_text(_p_full(ms[:5]), encoding="utf-8")

    first = ms[0]
    print("\nfirst match item KEYS:", sorted(first.keys()))
    h, a = fm._team_names(first)
    mid = fm.match_id_of(first)
    print(f"fixture_map sees -> id={mid!r}  home={h!r}  away={a!r}  date={fm._date_of(first)!r}")
    if not (h and a and mid):
        print("\n*** fixture_map could NOT read this shape — paste the item below into a test ***")
        print(_p(first))
        return 1
    print("\nfull first item:\n" + _p(first))

    # one finished match for stats/odds, one upcoming-ish for lineups
    def _status(m):
        return str(m.get("status") or m.get("state") or "").lower()
    finished = next((m for m in ms if "fin" in _status(m) or "ft" in _status(m)), first)
    fmid = fm.match_id_of(finished)
    print(f"\nprobing match_id={fmid} ({fm._team_names(finished)})")
    for name, fn in (("stats", ts.match_xg), ("odds", ts.match_odds),
                     ("lineups", ts.match_lineups)):
        try:
            r = fn(fmid)
        except Exception as e:  # noqa: BLE001
            r = f"ERROR {e}"
        print(f"  /{name}: {('present' if r else 'none')} -> {str(r)[:120]}")
    # save the raw odds payload (richest shape — BTTS/Pinnacle/openings) for fixture-locking
    odds = ts.match_odds(fmid)
    if odds:
        (out_dir / "odds_sample.json").write_text(_p_full(odds), encoding="utf-8")
        books = [b.get("bookmaker") for b in (odds.get("bookmakers") or [])]
        mkts = sorted({k for b in (odds.get("bookmakers") or [])
                       for k in (b.get("markets") or {})})
        print(f"\nodds books: {books}\nodds markets: {mkts}")
    print(f"\nsaved samples to {out_dir}")
    return 0


def _p_full(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
