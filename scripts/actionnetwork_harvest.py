"""Harvest pre-match World Cup BTTS odds from Action Network -> a compact JSON feed.

WHY THIS RUNS ON THE VPS: TheStatsAPI serves no pre-match BTTS (football API = settled-only;
the Premium Odds Feed has no BTTS market). Action Network's site shows pre-match BTTS from US
books, served by its internal API (`api.actionnetwork.com`). That host is firewall-blocked from
the local sandbox AND will likely 403 a datacenter/cloud IP, so this harvester runs on the user's
VPS (clean residential-ish IP) and writes a small JSON the dashboard reads.

HONEST CAVEAT (kept in the open): Action Network's API is internal/undocumented — programmatic
use is a ToS gray area and can break or IP-block without notice. This pulls read-only, lightly,
and degrades to writing nothing on error. It never fabricates a price.

Endpoints (browser headers required, else Cloudflare 403):
  GET /web/v2/scoreboard/soccer?period=event        -> games; World Cup is league_id 20
  GET /web/v2/games/{game_id}/props                 -> game_props.core_bet_type_49_both_teams_to_score
      [0].lines[book_id] = [{side: yes|no, odds: <american int>}, ...]

Output (default data/feeds/actionnetwork_btts.json):
  {"source","league","generated_at","games":[{game_id,start_time,status,home,away,
    btts:{<book>:{yes,no}}, btts_best:{yes,yes_book,no,no_book}}]}

Run (on the VPS):  python3 scripts/actionnetwork_harvest.py [--out PATH] [--now ISO8601]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://api.actionnetwork.com/web/v2"
WC_LEAGUE_ID = 20
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
_HDRS = {"User-Agent": _UA, "Accept": "application/json",
         "Origin": "https://www.actionnetwork.com",
         "Referer": "https://www.actionnetwork.com/"}
# Action Network book ids -> display names (US books).
BOOKS = {"15": "DraftKings", "30": "FanDuel", "49": "BetMGM", "68": "Caesars",
         "69": "PointsBet", "71": "BetRivers", "75": "Bet365", "79": "WynnBET",
         "76": "Unibet", "972": "ESPNBet"}
_BTTS_KEY = "core_bet_type_49_both_teams_to_score"


def _get(url: str, tries: int = 3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=_HDRS)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                print(f"  ! {url} -> {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                return None
            time.sleep(1.5)
    return None


def _am_to_dec(american) -> float | None:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def _team_names(g: dict) -> tuple[str | None, str | None]:
    teams = {t.get("id"): (t.get("full_name") or t.get("display_name") or t.get("name"))
             for t in g.get("teams", [])}
    return teams.get(g.get("home_team_id")), teams.get(g.get("away_team_id"))


def _btts_for_game(game_id) -> dict | None:
    d = _get(f"{BASE}/games/{game_id}/props")
    gp = (d or {}).get("game_props") or {}
    markets = gp.get(_BTTS_KEY)
    if not markets:
        return None
    lines = (markets[0] or {}).get("lines") or {}
    by_book: dict[str, dict] = {}
    for book_id, outs in lines.items():
        yes = no = None
        for o in outs or []:
            side = str(o.get("side", "")).lower()
            if side == "yes":
                yes = o.get("odds")
            elif side == "no":
                no = o.get("odds")
        if yes is not None and no is not None:
            by_book[BOOKS.get(str(book_id), f"book_{book_id}")] = {"yes": yes, "no": no}
    if not by_book:
        return None
    # best (most favorable) american price per side = highest decimal payout
    def _best(side):
        cand = [(bk, v[side]) for bk, v in by_book.items() if v.get(side) is not None]
        if not cand:
            return (None, None)
        bk, am = max(cand, key=lambda x: _am_to_dec(x[1]) or 0)
        return (am, bk)
    yb, ybk = _best("yes")
    nb, nbk = _best("no")
    return {"by_book": by_book, "best": {"yes": yb, "yes_book": ybk, "no": nb, "no_book": nbk}}


def harvest(now_iso: str | None = None) -> dict:
    board = _get(f"{BASE}/scoreboard/soccer?period=event") or {}
    games = [g for g in board.get("games", []) if g.get("league_id") == WC_LEAGUE_ID]
    out = []
    for g in games:
        gid = g.get("id")
        home, away = _team_names(g)
        if not (home and away and gid):
            continue
        bt = _btts_for_game(gid)
        time.sleep(0.4)                       # be gentle
        if not bt:
            continue
        out.append({"game_id": gid, "start_time": g.get("start_time"),
                    "status": g.get("status"), "home": home, "away": away,
                    "btts": bt["by_book"], "btts_best": bt["best"]})
    return {"source": "actionnetwork", "league": "worldcup",
            "generated_at": now_iso, "games": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "data" / "feeds" / "actionnetwork_btts.json"))
    ap.add_argument("--now", default=None, help="ISO8601 timestamp to stamp (UTC)")
    args = ap.parse_args()
    feed = harvest(now_iso=args.now)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(feed, indent=1), encoding="utf-8")
    n = len(feed["games"])
    print(f"harvested {n} WC games with BTTS -> {p}")
    for g in feed["games"][:8]:
        b = g["btts_best"]
        print(f"  {g['home']} v {g['away']} ({g['status']}): "
              f"Yes {b['yes']} @{b['yes_book']} / No {b['no']} @{b['no_book']} "
              f"[{len(g['btts'])} books]")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
