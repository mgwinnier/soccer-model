"""Tests for confirmed-XI status + player ratings + 'regular starter missing today' flag.

Network mocked. Signals are self-contained in TheStatsAPI, matched by player **id**: recent-match
ratings from ``/player-stats`` and the prior XI from that endpoint's ``started`` flag.
"""
from src.data import lineup_status as ls
from src.data import thestatsapi as ts


CANDS = [
    {"id": "mt_prev", "utc_date": "2026-06-14T17:00:00Z", "status": "finished",
     "home_team": {"id": "T0"}, "away_team": {"id": "T9"}},
    {"id": "mt_today", "utc_date": "2026-06-18T17:00:00Z", "status": "scheduled",
     "home_team": {"id": "T0"}, "away_team": {"id": "T1"}},
]
PREV_RATINGS = {
    "p1": {"name": "Star One", "rating": 7.5, "started": True, "team_id": "T0", "position": "F"},
    "p2": {"name": "Star Two", "rating": 8.0, "started": True, "team_id": "T0", "position": "M"},
    "p3": {"name": "Reg Three", "rating": 6.5, "started": True, "team_id": "T0", "position": "D"},
    "z9": {"name": "Opp Guy", "rating": 6.0, "started": True, "team_id": "T9", "position": "M"},
}


def test_team_recent_builds_history_and_prior_xi(monkeypatch):
    monkeypatch.setattr(ts, "match_player_ratings", lambda mid, **k: PREV_RATINGS)
    hist, names, prior, used = ls._team_recent("T0", "2026-06-18T17:00:00Z", CANDS)
    assert prior == {"p1": "Star One", "p2": "Star Two", "p3": "Reg Three"}   # only T0 starters
    assert hist["p2"] == [8.0] and names["p1"] == "Star One"
    assert "z9" not in hist                                                   # opponent excluded


def test_side_status_lists_xi_with_ratings_and_flags_missing(monkeypatch):
    monkeypatch.setattr(ts, "match_player_ratings", lambda mid, **k: PREV_RATINGS)
    today = {"formation": "4-4-2", "starting_xi": [{"id": "p1", "name": "Star One"},
                                                   {"id": "pX", "name": "New Guy"}]}
    s = ls._side_status("T0", today, "2026-06-18T17:00:00Z", CANDS)
    names = {r["name"]: r for r in s["xi"]}
    assert names["Star One"]["avg"] == 7.5 and names["Star One"]["recent"] == [7.5]
    assert names["New Guy"]["avg"] is None                       # no prior rating
    miss = {x["name"]: x["avg"] for x in s["missing_starters"]}
    assert miss == {"Star Two": 8.0, "Reg Three": 6.5}           # dropped regulars + their form


def test_lineup_status_not_posted(monkeypatch):
    monkeypatch.setattr(ts, "is_available", lambda: True)
    monkeypatch.setattr(ts, "current_season_id", lambda **k: "sn_x")
    monkeypatch.setattr(ts, "matches", lambda **k: CANDS)
    monkeypatch.setattr(ts, "match_lineups", lambda mid, **k: None)   # sheet not posted
    out = ls.lineup_status("T0name", "T1name", "2026-06-18")
    assert out is None or out.get("posted") is False


def test_availability_from_status_bounded():
    assert ls.availability_from_status({"missing_starters": []}) == 1.0
    # two ~7.5 regulars out -> saturates near the 10% cap, never below 0.90
    big = {"missing_starters": [{"name": "A", "avg": 7.5}, {"name": "B", "avg": 7.5}]}
    mult = ls.availability_from_status(big)
    assert 0.90 <= mult < 1.0
    assert ls.availability_from_status({"missing_starters": [{"name": "C", "avg": 6.1}]}) > 0.99


def test_lineup_availability_noop_when_not_posted():
    assert ls.lineup_availability("A", "B", "2026-06-18",
                                  status={"posted": False}) == (1.0, 1.0)


def test_lineup_status_noop_without_key(monkeypatch):
    monkeypatch.setattr(ts, "is_available", lambda: False)
    assert ls.lineup_status("A", "B", "2026-06-18") is None
