"""Hand-coded group memberships for past tournament group stages.

The match spine (``data/clean.py``) carries only a ``tournament`` string and a
date — no group / matchday / stage column — so the group each team belonged to
must be supplied externally to reconstruct standings. Team names are written in
readable form and normalized through ``team_names.normalize_team`` at load, so
they line up with the normalized names in the match data.

**Scope is deliberately limited to top-2-advance group stages** (every team in a
group competes only for the two automatic qualification spots). The 6-group Euros
(2016/2020/2024) advance the four best third-placed teams *across* groups, which
makes a single group's "needs" depend on other groups' simultaneous results — an
ill-defined within-group quantity. Rather than emit a fabricated flag there, those
formats are omitted until proper cross-group best-third logic is added. This keeps
every motivation feature correct (per the project's no-silent-fabrication rule),
at the cost of a smaller validation sample (~72 final-round matches).
"""
from __future__ import annotations

from .team_names import normalize_team

# Each entry: (tournament_string, year) -> {group_letter: [4 teams]}.
# tournament_string must match data/clean.py's `tournament` values exactly.
_RAW: dict[tuple[str, int], dict[str, list[str]]] = {
    ("FIFA World Cup", 2010): {
        "A": ["South Africa", "Mexico", "Uruguay", "France"],
        "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
        "C": ["England", "United States", "Algeria", "Slovenia"],
        "D": ["Germany", "Australia", "Serbia", "Ghana"],
        "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
        "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
        "G": ["Brazil", "North Korea", "Ivory Coast", "Portugal"],
        "H": ["Spain", "Switzerland", "Honduras", "Chile"],
    },
    ("FIFA World Cup", 2014): {
        "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "B": ["Spain", "Netherlands", "Chile", "Australia"],
        "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
        "D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        "G": ["Germany", "Portugal", "Ghana", "United States"],
        "H": ["Belgium", "Algeria", "Russia", "South Korea"],
    },
    ("FIFA World Cup", 2018): {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
    ("FIFA World Cup", 2022): {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
    ("UEFA Euro", 2012): {
        "A": ["Poland", "Greece", "Russia", "Czech Republic"],
        "B": ["Netherlands", "Denmark", "Germany", "Portugal"],
        "C": ["Spain", "Italy", "Republic of Ireland", "Croatia"],
        "D": ["Ukraine", "Sweden", "France", "England"],
    },
}


def _normalize_groups(raw: dict[str, list[str]]) -> dict[str, list[str]]:
    return {g: [normalize_team(t) for t in teams] for g, teams in raw.items()}


# Public: {(tournament, year): {group: [normalized teams]}}, top-2-advance only.
COVERED: dict[tuple[str, int], dict[str, list[str]]] = {
    key: _normalize_groups(groups) for key, groups in _RAW.items()
}


def team_to_group(tournament: str, year: int) -> dict[str, str] | None:
    """Normalized team -> group letter for a covered tournament-year, else None."""
    groups = COVERED.get((tournament, year))
    if groups is None:
        return None
    return {t: g for g, teams in groups.items() for t in teams}
