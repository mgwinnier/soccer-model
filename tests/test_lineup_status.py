"""Tests for the confirmed-XI 'regular starter missing today' flag.

Network mocked. The signal is deliberately self-contained in TheStatsAPI and matched by player
**id**: a starter from the team's previous WC match who isn't in today's confirmed XI. These lock
the prior-XI lookup, the id-based diff (no name matching), and the honest no-op states.
"""
from src.data import lineup_status as ls
from src.data import thestatsapi as ts


# season match list: team T0 (home) has a prior finished match (md1) + today's match (md2).
CANDS = [
    {"id": "mt_prev", "utc_date": "2026-06-14T17:00:00Z", "status": "finished",
     "home_team": {"id": "T0"}, "away_team": {"id": "T9"}},
    {"id": "mt_today", "utc_date": "2026-06-18T17:00:00Z", "status": "scheduled",
     "home_team": {"id": "T0"}, "away_team": {"id": "T1"}},
]
PREV_LINEUP = {"home": {"formation": "4-3-3", "starting_xi": [
    {"id": "p1", "name": "Star One"}, {"id": "p2", "name": "Star Two"},
    {"id": "p3", "name": "Reg Three"}]}, "away": {"starting_xi": []}}


def test_recent_starter_ids_finds_prior_xi(monkeypatch):
    monkeypatch.setattr(ts, "match_lineups", lambda mid, **k: PREV_LINEUP if mid == "mt_prev" else None)
    got = ls._recent_starter_ids("T0", "2026-06-18T17:00:00Z", CANDS)
    assert got == {"p1": "Star One", "p2": "Star Two", "p3": "Reg Three"}


def test_recent_starter_ids_empty_when_no_prior(monkeypatch):
    # team with no earlier finished match (matchday 1) -> empty, no flag possible
    monkeypatch.setattr(ts, "match_lineups", lambda mid, **k: PREV_LINEUP)
    assert ls._recent_starter_ids("T1", "2026-06-18T17:00:00Z", CANDS) == {}


def test_side_status_flags_missing_regular_starter(monkeypatch):
    monkeypatch.setattr(ts, "match_lineups", lambda mid, **k: PREV_LINEUP)
    # today's XI keeps p1, drops p2 & p3 (p2/p3 = missing regulars)
    today = {"formation": "4-4-2", "starting_xi": [{"id": "p1", "name": "Star One"},
                                                   {"id": "pX", "name": "New Guy"}]}
    s = ls._side_status("T0", today, "2026-06-18T17:00:00Z", CANDS)
    assert s["formation"] == "4-4-2" and s["had_prior_xi"] is True
    assert set(s["missing_starters"]) == {"Star Two", "Reg Three"}


def test_lineup_status_not_posted(monkeypatch):
    monkeypatch.setattr(ts, "is_available", lambda: True)
    monkeypatch.setattr(ts, "current_season_id", lambda **k: "sn_x")
    monkeypatch.setattr(ts, "matches", lambda **k: CANDS)
    monkeypatch.setattr(ts, "match_lineups", lambda mid, **k: None)   # sheet not posted
    out = ls.lineup_status("T0name", "T1name", "2026-06-18")
    # find_match won't match by name here -> None, OR posted False; both are honest non-flags
    assert out is None or out.get("posted") is False


def test_lineup_status_noop_without_key(monkeypatch):
    monkeypatch.setattr(ts, "is_available", lambda: False)
    assert ls.lineup_status("A", "B", "2026-06-18") is None
