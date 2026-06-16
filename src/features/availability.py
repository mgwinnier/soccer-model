"""Turn squad availability into a team-strength adjustment.

This is where the two API keys pay off together:

  * **Kaggle FIFA ratings** give every player an overall rating, so we know each
    nation's key players and how much of its quality each one represents.
  * **API-Football injuries** tell us which of those players are unavailable for
    an upcoming match.

`availability_multiplier()` combines them into a factor (≤ 1.0) applied to a
team's expected goals: lose 10% of your key-XI rating points to injury and your
attack is scaled down accordingly. With no Kaggle data or no injuries it returns
1.0 (no effect) — so it only ever *sharpens* imminent-match predictions, never
fabricates.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..config import load_config, path_for
from ..data.team_names import normalize_team

# How strongly missing quality scales expected goals. 1.0 means a team missing
# 10% of its key-XI rating loses ~10% of its attacking output (capped).
_SENSITIVITY = 1.0
_MAX_PENALTY = 0.25          # never knock a team down more than 25%
_KEY_XI = 16                 # "key players" = top-16 by rating (XI + key subs)


def _find_fifa_csv(cfg: dict):
    raw = path_for("data_raw", cfg)
    for sub in ["fifa_players", "."]:
        d = raw / sub
        if not d.exists():
            continue
        # Prefer the comprehensive EA Sports FC export, then any single export.
        for preferred in ("male_players.csv", "players.csv"):
            if (d / preferred).exists():
                return d / preferred
        csvs = [p for p in sorted(d.glob("*.csv"))
                if "female" not in p.name.lower() and "team" not in p.name.lower()
                and "coach" not in p.name.lower()]
        if csvs:
            return csvs[-1]   # latest-named (e.g. FIFA23 over FIFA17)
    return None


@lru_cache(maxsize=1)
def load_fifa_ratings() -> pd.DataFrame:
    """Per-player [player, nationality, overall] from the EA FC / FIFA dataset.

    Robust to column-name variants and to the multi-version `male_players.csv`
    export (filters to the latest `fifa_version`). Empty frame if undownloaded."""
    cfg = load_config()
    path = _find_fifa_csv(cfg)
    cols = ["player", "nationality", "overall"]
    if path is None:
        return pd.DataFrame(columns=cols)
    header = pd.read_csv(path, nrows=0)
    lower = {c.lower(): c for c in header.columns}

    def pick(*opts):
        for o in opts:
            if o in lower:
                return lower[o]
        return None

    name_c = pick("short_name", "name", "long_name", "player", "known_as")
    nat_c = pick("nationality_name", "nationality", "nation", "country")
    ovr_c = pick("overall", "overall_rating", "ovr", "rating")
    ver_c = pick("fifa_version")
    if not (name_c and nat_c and ovr_c):
        return pd.DataFrame(columns=cols)
    usecols = [c for c in (name_c, nat_c, ovr_c, ver_c) if c]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if ver_c:                                   # keep only the newest game version
        df = df[df[ver_c] == df[ver_c].max()]
    out = df[[name_c, nat_c, ovr_c]].copy()
    out.columns = cols
    out["player"] = out["player"].astype(str).str.strip()
    out["nationality"] = out["nationality"].map(normalize_team)
    out["overall"] = pd.to_numeric(out["overall"], errors="coerce")
    return out.dropna(subset=["nationality", "overall"])


def team_key_players(team: str) -> dict[str, float]:
    """{player_name_lower: overall} for a nation's top-rated players."""
    team = normalize_team(team)
    fifa = load_fifa_ratings()
    if fifa.empty:
        return {}
    sub = fifa[fifa["nationality"] == team].nlargest(_KEY_XI, "overall")
    return {str(n).lower(): float(o) for n, o in zip(sub["player"], sub["overall"])}


def availability_multiplier(team: str, injured: list[str]) -> float:
    """Expected-goals multiplier (≤1.0) for a team missing `injured` players."""
    key = team_key_players(team)
    if not key or not injured:
        return 1.0
    injured_set = {n.lower() for n in injured}
    total = sum(key.values())
    missing = sum(r for n, r in key.items() if n in injured_set)
    if total <= 0:
        return 1.0
    penalty = min(_SENSITIVITY * (missing / total), _MAX_PENALTY)
    return 1.0 - penalty


def availability_report(team: str, injured: list[str]) -> dict:
    """Human-readable summary for the dashboard."""
    key = team_key_players(team)
    out_key = [n for n in injured if n.lower() in key]
    return {
        "team": team,
        "multiplier": availability_multiplier(team, injured),
        "key_players_out": out_key,
        "n_injured": len(injured),
        "have_ratings": bool(key),
    }
