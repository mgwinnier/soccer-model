"""Harvest historical international closing odds (Bet365) from ESPN.

Sweeps ESPN's international competition feeds since 2019, collecting every finished
match's teams, score, and retained closing odds (via the summary endpoint's `odds`
array — see ``fetch_summary_odds``). The result is a tidy
``data/processed/odds_history.parquet`` we can grade the model against at scale.

This makes thousands of HTTP calls — it's a one-time, **resumable**, throttled,
backgroundable harvest. Everything is cached (per-game summaries + the parquet);
re-running only fetches what's missing.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from .odds import fetch_espn_range, fetch_summary_odds

# ESPN international competition slugs with betting coverage (~2019+).
LEAGUES = [
    "fifa.world",                    # World Cup
    "uefa.euro", "uefa.euroq",       # Euros + qualifiers
    "uefa.nations",                  # Nations League
    "conmebol.america",              # Copa América
    "concacaf.nations", "concacaf.gold",
    "fifa.worldq.uefa", "fifa.worldq.conmebol", "fifa.worldq.concacaf",
    "fifa.worldq.afc", "fifa.worldq.caf", "fifa.worldq.ofc",
    "fifa.friendly",                 # international friendlies
    "afc.asian", "caf.nations",      # Asian Cup, AFCON
]

OUT_COLS = ["date", "league", "game_id", "home_team", "away_team",
            "home_score", "away_score", "ml_home", "ml_away", "ml_draw",
            "total_line", "ou_over_odds", "ou_under_odds",
            "spread_home_line", "spread_home_odds", "spread_away_odds"]


def _log(msg: str) -> None:
    print(f"[odds_history] {msg}", flush=True)


def harvest(start_year: int = 2019, leagues: list[str] | None = None,
            window_days: int = 45, throttle: float = 0.25,
            cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    ensure_dirs(cfg)
    out_path = path_for("data_processed", cfg) / "odds_history.parquet"
    leagues = leagues or LEAGUES

    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=OUT_COLS)
    seen = set(existing["game_id"].astype(str)) if len(existing) else set()
    rows: list[dict] = []

    today = date.today()
    for league in leagues:
        n_league = 0
        cur = date(start_year, 1, 1)
        while cur <= today:
            end = min(cur + timedelta(days=window_days - 1), today)
            try:
                events = fetch_espn_range(cur.strftime("%Y-%m-%d"),
                                          end.strftime("%Y-%m-%d"), league=league, cfg=cfg)
            except Exception:  # noqa: BLE001
                events = []
            for ev in events:
                gid = str(ev.get("game_id") or "")
                if not gid or gid in seen:
                    continue
                if ev["status"] != "post" or ev["home_score"] is None:
                    continue
                seen.add(gid)
                try:
                    od = fetch_summary_odds(gid, league=league, cfg=cfg)
                except Exception:  # noqa: BLE001
                    od = None
                time.sleep(throttle)
                if not od or od.get("ml_home") is None:
                    continue
                rows.append({
                    "date": pd.to_datetime(ev["date"]).tz_localize(None),
                    "league": league, "game_id": gid,
                    "home_team": ev["home_team"], "away_team": ev["away_team"],
                    "home_score": ev["home_score"], "away_score": ev["away_score"],
                    **{k: od.get(k) for k in OUT_COLS[7:]},
                })
                n_league += 1
            cur = end + timedelta(days=1)
        _log(f"{league}: +{n_league} matches with odds")
        # checkpoint after each league so a long run is resumable
        if rows:
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            combined = combined.dropna(subset=["home_team", "away_team"]).drop_duplicates("game_id")
            combined.to_parquet(out_path, index=False)
            existing, rows = combined, []

    final = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=OUT_COLS)
    _log(f"done — {len(final)} international matches with Bet365 odds -> {out_path}")
    return final


def load_odds_history(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    p = path_for("data_processed", cfg) / "odds_history.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=OUT_COLS)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else 2019
    harvest(start_year=yr)
