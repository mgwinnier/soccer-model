"""2026 World Cup structure: groups, hosts, played results, and bracket seeding.

The **group composition is exact** (verified against the official draw and
cross-checked against matches already played). Each group's four teams are
listed below.

The **knockout bracket** in 2026 uses 12 group winners, 12 runners-up, and the
8 best third-placed teams, slotted into a 32-team bracket via a 495-row official
allocation table. That table is fiddly and we could not extract it reliably, so
this module builds a *structurally faithful approximation*: all 32 qualifiers
are ranked (winners > runners-up > thirds, then by points/GD/GF) and snake-seeded
into a balanced single-elimination bracket. This keeps the dominant drivers —
group advancement and relative strength — exactly right; only the precise
cross-bracket pairings differ from FIFA's table. ``KNOCKOUT_SLOTS`` is left as a
hook to drop in the exact tree if desired.
"""
from __future__ import annotations

import pandas as pd

from ..data.team_names import normalize_team

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Sweden", "Japan", "Netherlands", "Tunisia"],
    "G": ["New Zealand", "Iran", "Belgium", "Egypt"],
    "H": ["Uruguay", "Saudi Arabia", "Spain", "Cape Verde"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# Co-hosts get home advantage in their group matches.
HOST_TEAMS = {"United States", "Mexico", "Canada"}

# Hook for the exact FIFA R32 slot tree (left None -> seeded bracket is used).
KNOCKOUT_SLOTS = None


def all_teams() -> list[str]:
    return [t for g in GROUPS.values() for t in g]


def team_to_group() -> dict[str, str]:
    return {normalize_team(t): g for g, teams in GROUPS.items() for t in teams}


def load_played_results(cfg=None) -> dict[frozenset, tuple[str, int, int]]:
    """Map {home, away} -> (home_team, home_goals, away_goals) for 2026 WC games."""
    from ..config import load_config, path_for
    cfg = cfg or load_config()
    path = path_for("data_processed", cfg) / "matches.parquet"
    m = pd.read_parquet(path)
    wc = m[(m["tournament"] == "FIFA World Cup") & (m["date"] >= pd.Timestamp("2026-06-01"))]
    out: dict[frozenset, tuple[str, int, int]] = {}
    for r in wc.itertuples(index=False):
        key = frozenset((r.home_team, r.away_team))
        out[key] = (r.home_team, int(r.home_score), int(r.away_score))
    return out


def validate_groups(cfg=None) -> None:
    """Assert every played 2026 match is between two same-group teams."""
    t2g = team_to_group()
    played = load_played_results(cfg)
    for key in played:
        a, b = tuple(key)
        ga, gb = t2g.get(a), t2g.get(b)
        assert ga is not None and ga == gb, (
            f"Played match {a} vs {b} crosses groups ({ga} vs {gb}) — "
            f"group data inconsistent with results!"
        )
