"""Probe whether the football API publishes pre-match BTTS in the final window before KO.

Open question (validated only ~2.5h out so far): TheStatsAPI's football `/matches/{id}/odds`
404s pre-match for upcoming WC games, but the finished games DO carry BTTS — suggesting the line
is captured in ONE snapshot at/around kickoff. This script, run ~30 min before KO, settles it:
it hits the football /odds directly for today's still-scheduled WC matches and reports whether
BTTS (and any markets) are populated yet.

Run:  python scripts/check_football_btts_at_ko.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import thestatsapi as ts          # noqa: E402
from src.data import fixture_map as fm           # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if not ts.is_available():
        print("NO KEY"); return 2
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    print(f"now (UTC): {now.isoformat()[:19]}")
    sched = ts.matches(competition_id="comp_6107", season_id="sn_118868",
                       status="scheduled", ttl=0.0)
    todays = sorted([m for m in sched if str(m.get("utc_date", ""))[:10] == today],
                    key=lambda m: m.get("utc_date", ""))
    if not todays:
        print("no still-scheduled WC matches today (they may have kicked off / finished).")
        return 0
    for m in todays:
        mid = fm.match_id_of(m); h, a = fm._team_names(m)
        ko = str(m.get("utc_date", ""))[11:16]
        od = ts.match_odds(mid, ttl=0.0)
        if not od:
            print(f"  {h} v {a} (KO {ko}) -> still NO odds (404)")
            continue
        bks = od.get("bookmakers", []) or []
        mkts = sorted({k for b in bks for k in (b.get("markets") or {})})
        bt = ts.btts_prices(od)
        print(f"  {h} v {a} (KO {ko}) -> ODDS! books={[b.get('bookmaker') for b in bks]} "
              f"markets={mkts} | BTTS={bt}")
    print("\nIf BTTS shows above, the football API DOES publish it pre-match near KO "
          "(single snapshot, openings likely null -> no CLV). If all say NO odds, it's settled-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
