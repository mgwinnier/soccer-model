"""Harvest every 2026 World Cup squad's player market values -> a committed static file.

Market values change only at transfer windows, so a snapshot is plenty — and it lets the live
app attach values to ESPN lineups with ZERO per-request TheStatsAPI calls (no rate limits, works
free on the cloud). Re-run occasionally to refresh.

Collects all team ids from the current WC season's matches, pulls each squad
(/teams/{id}/players), and writes data/feeds/wc_squad_values.json:
  {"generated_at", "teams": {<normalized team name>: {"team_id", "players":
     [{"name", "position", "market_value"}], "total_value"}}}

Run:  python scripts/harvest_squad_values.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import thestatsapi as ts          # noqa: E402
from src.data import fixture_map as fm           # noqa: E402
from src.data.team_names import normalize_team   # noqa: E402


def main() -> int:
    if not ts.is_available():
        print("NO KEY"); return 2
    sid = ts.current_season_id()
    cands = ts.matches(competition_id=ts.WC_COMP, season_id=sid, ttl=0.0)
    # unique team ids + names from the schedule
    teams: dict[str, str] = {}
    for c in cands:
        for side in ("home_team", "away_team"):
            t = c.get(side) or {}
            if t.get("id") and t.get("name"):
                teams[t["id"]] = t["name"]
    print(f"{len(teams)} WC teams to harvest…")
    import time
    out: dict = {}
    for i, (tid, name) in enumerate(sorted(teams.items(), key=lambda x: x[1]), 1):
        squad = ts.team_squad(tid, ttl=0.0)
        if not squad:                         # empty is usually a transient 429 — retry once
            time.sleep(8)
            squad = ts.team_squad(tid, ttl=0.0)
        players = [{"name": p["name"], "position": p.get("position"),
                    "market_value": p.get("market_value")} for p in squad]
        total = sum(p["market_value"] or 0 for p in players)
        key = normalize_team(name) or name
        out[key] = {"team_id": tid, "players": players, "total_value": total}
        n_mv = sum(1 for p in players if p["market_value"])
        print(f"  [{i}/{len(teams)}] {name}: {len(players)} players, {n_mv} valued, "
              f"total €{total/1e6:.0f}M")
        time.sleep(1.0)                        # be gentle — avoid the sustained-429 wipeout
    feed = {"generated_at": None, "teams": out}
    p = Path(__file__).resolve().parents[1] / "data" / "feeds" / "wc_squad_values.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(feed, indent=1, ensure_ascii=False), encoding="utf-8")
    valued = sum(1 for t in out.values() if t["total_value"] > 0)
    print(f"\nwrote {p} — {len(out)} teams ({valued} with values)")
    return 0 if valued else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(main())
