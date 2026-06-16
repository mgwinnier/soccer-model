"""Expected-goals (xG) data for internationals, via FBref (`soccerdata`).

Honest scope: FBref only carries xG for the **World Cup and the Euros** (from
~2017 on), so this covers a few hundred elite-tournament matches — not the full
results history. Those are exactly the games where style/quality matter most, and
the xG signal is wired in NaN-tolerantly elsewhere; the ablation backtest decides
whether it earns its place.

One schedule request per competition-season already carries per-match xG, so this
is light. Everything is cached to ``data/raw/fbref/xg.parquet`` and failures for
any single competition are skipped, never fatal.
"""
from __future__ import annotations

import pandas as pd

from ..config import load_config, path_for
from .team_names import normalize_team

# (FBref competition, season) pairs that carry xG. Seasons use soccerdata codes.
_COMPETITIONS = [
    ("INT-World Cup", "2018"),
    ("INT-World Cup", "2022"),
    ("INT-World Cup", "2026"),
    ("INT-European Championship", "2021"),
    ("INT-European Championship", "2024"),
]


def _log(msg: str) -> None:
    print(f"[fbref] {msg}", flush=True)


def fetch_xg(cfg: dict | None = None, refresh: bool = False) -> pd.DataFrame:
    """Return a tidy xG table: date, home_team, away_team, home_xg, away_xg.

    Primary source is StatsBomb (reliable, key-less); FBref is a fallback only
    used if StatsBomb yields nothing, since FBref is frequently IP/CAPTCHA-blocked.
    """
    cfg = cfg or load_config()
    # Primary: StatsBomb open data
    from .statsbomb import fetch_statsbomb_xg
    sb_xg = fetch_statsbomb_xg(cfg, refresh=refresh)
    if not sb_xg.empty:
        return sb_xg

    out_dir = path_for("data_raw", cfg) / "fbref"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "xg.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    try:
        import soccerdata as sd
    except ImportError:
        _log("soccerdata not installed — skipping xG (features will be NaN)")
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_xg", "away_xg"])

    frames = []
    for comp, season in _COMPETITIONS:
        try:
            fb = sd.FBref(comp, season)
            sched = fb.read_schedule().reset_index()
            if "home_xg" not in sched.columns:
                _log(f"{comp} {season}: no xG columns, skipping")
                continue
            df = sched[["date", "home_team", "away_team", "home_xg", "away_xg"]].copy()
            df = df.dropna(subset=["home_xg", "away_xg"])
            frames.append(df)
            _log(f"{comp} {season}: {len(df)} matches with xG")
        except Exception as exc:  # noqa: BLE001
            _log(f"{comp} {season}: fetch failed ({type(exc).__name__}), skipping")
            continue

    if not frames:
        empty = pd.DataFrame(columns=["date", "home_team", "away_team", "home_xg", "away_xg"])
        empty.to_parquet(cache, index=False)
        return empty

    xg = pd.concat(frames, ignore_index=True)
    xg["date"] = pd.to_datetime(xg["date"])
    xg["home_team"] = xg["home_team"].map(normalize_team)
    xg["away_team"] = xg["away_team"].map(normalize_team)
    xg["home_xg"] = pd.to_numeric(xg["home_xg"], errors="coerce")
    xg["away_xg"] = pd.to_numeric(xg["away_xg"], errors="coerce")
    xg = xg.dropna(subset=["home_team", "away_team", "home_xg", "away_xg"])
    xg = xg.drop_duplicates(["date", "home_team", "away_team"]).reset_index(drop=True)
    xg.to_parquet(cache, index=False)
    _log(f"cached {len(xg)} xG matches -> {cache}")
    return xg


def attach_xg_to_matches(matches: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Left-join xG onto the canonical match table by (date, teams).

    Tolerant join: matches on the same calendar day + both team names. Returns
    ``matches`` with added ``home_xg`` / ``away_xg`` (NaN where unavailable).
    """
    cfg = cfg or load_config()
    xg = fetch_xg(cfg)
    if xg.empty:
        matches = matches.copy()
        matches["home_xg"] = float("nan")
        matches["away_xg"] = float("nan")
        return matches
    xg = xg.copy()
    xg["day"] = xg["date"].dt.normalize()
    m = matches.copy()
    m["day"] = pd.to_datetime(m["date"]).dt.normalize()
    merged = m.merge(
        xg[["day", "home_team", "away_team", "home_xg", "away_xg"]],
        on=["day", "home_team", "away_team"], how="left",
    ).drop(columns="day")
    return merged


if __name__ == "__main__":
    df = fetch_xg(refresh=True)
    print(df.head(10).to_string(index=False))
    print(f"\ntotal xG matches: {len(df)}")
