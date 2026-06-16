"""xG from StatsBomb open data (GitHub-hosted JSON — no key, no CAPTCHA).

FBref is IP/CAPTCHA-blocked in many environments, so StatsBomb is our reliable
primary xG source. It carries full event data (with per-shot xG) for modern men's
international tournaments — World Cups and Euros from the xG era. We sum each
match's shot xG by team into the same tidy table FBref would have produced:
``date, home_team, away_team, home_xg, away_xg``. Cached to
``data/raw/statsbomb/xg.parquet``; per-match failures are skipped, never fatal.
"""
from __future__ import annotations

import pandas as pd

from ..config import load_config, path_for
from .team_names import normalize_team

_MIN_SEASON_YEAR = 2016  # xG models only reliable for recent tournaments


def _log(msg: str) -> None:
    print(f"[statsbomb] {msg}", flush=True)


def _modern_mens_tournaments(sb) -> list[tuple[int, int, str]]:
    comps = sb.competitions()
    keep = comps[
        (comps["competition_gender"] == "male")
        & (comps["competition_name"].str.contains("World Cup|European Championship",
                                                  regex=True, na=False))
    ].copy()

    def _year(s):
        try:
            return int(str(s)[:4])
        except ValueError:
            return 0

    keep = keep[keep["season_name"].map(_year) >= _MIN_SEASON_YEAR]
    # exclude youth competitions
    keep = keep[~keep["competition_name"].str.contains("U20|U-20|U17", na=False)]
    return list(keep[["competition_id", "season_id", "competition_name"]]
                .itertuples(index=False, name=None))


def fetch_statsbomb_xg(cfg: dict | None = None, refresh: bool = False) -> pd.DataFrame:
    cfg = cfg or load_config()
    out_dir = path_for("data_raw", cfg) / "statsbomb"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "xg.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    cols = ["date", "home_team", "away_team", "home_xg", "away_xg"]
    try:
        from statsbombpy import sb
    except ImportError:
        _log("statsbombpy not installed — skipping")
        return pd.DataFrame(columns=cols)

    try:
        tournaments = _modern_mens_tournaments(sb)
    except Exception as exc:  # noqa: BLE001
        _log(f"could not list competitions ({exc}); skipping")
        return pd.DataFrame(columns=cols)

    rows = []
    for comp_id, season_id, cname in tournaments:
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception:  # noqa: BLE001
            continue
        n_ok = 0
        for mt in matches.itertuples(index=False):
            try:
                ev = sb.events(match_id=mt.match_id)
                shots = ev[ev["type"] == "Shot"]
                if "shot_statsbomb_xg" not in shots.columns:
                    continue
                by_team = shots.groupby("team")["shot_statsbomb_xg"].sum()
                hx = float(by_team.get(mt.home_team, 0.0))
                ax = float(by_team.get(mt.away_team, 0.0))
                rows.append({
                    "date": mt.match_date, "home_team": mt.home_team,
                    "away_team": mt.away_team, "home_xg": hx, "away_xg": ax,
                })
                n_ok += 1
            except Exception:  # noqa: BLE001
                continue
        _log(f"{cname} {season_id}: {n_ok} matches with xG")

    if not rows:
        empty = pd.DataFrame(columns=cols)
        empty.to_parquet(cache, index=False)
        return empty

    xg = pd.DataFrame(rows)
    xg["date"] = pd.to_datetime(xg["date"])
    xg["home_team"] = xg["home_team"].map(normalize_team)
    xg["away_team"] = xg["away_team"].map(normalize_team)
    xg = xg.dropna(subset=["home_team", "away_team"])
    xg = xg.drop_duplicates(["date", "home_team", "away_team"]).reset_index(drop=True)
    xg.to_parquet(cache, index=False)
    _log(f"cached {len(xg)} xG matches -> {cache}")
    return xg


if __name__ == "__main__":
    df = fetch_statsbomb_xg(refresh=True)
    print(df.head(10).to_string(index=False))
    print(f"\ntotal: {len(df)} matches with xG")
