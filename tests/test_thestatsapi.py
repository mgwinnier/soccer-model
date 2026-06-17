"""Tests for the TheStatsAPI client parsing (network mocked with real captured shapes).

The live network is firewall-blocked locally + only exercised on the cloud/VPS, so these
lock the PARSING to the actual response shapes validated on the VPS: /stats wraps in
``data`` with ``overview.expected_goals.all``; /odds has ``bookmakers[].markets`` incl.
``btts``; /lineups has ``home``/``away`` with ``starting_xi``."""
from src.data import thestatsapi as ts


def test_unwrap_handles_data_envelope():
    assert ts._unwrap({"data": {"x": 1}}) == {"x": 1}
    assert ts._unwrap({"x": 1}) == {"x": 1}           # already unwrapped
    assert ts._unwrap(None) == {}


def test_match_xg_parses_real_shape(monkeypatch):
    # exact shape captured from the 2022 final (Argentina 3.23 / France 2.27)
    payload = {"data": {"match_id": "mt_1", "overview": {
        "ball_possession": {"all": {"home": 60, "away": 40}},
        "expected_goals": {"all": {"home": 3.23, "away": 2.27},
                           "first_half": {"home": 1.41, "away": 0}}}}}
    monkeypatch.setattr(ts, "_get", lambda *a, **k: payload)
    assert ts.match_xg("mt_1") == (3.23, 2.27)


def test_match_xg_none_when_absent(monkeypatch):
    monkeypatch.setattr(ts, "_get", lambda *a, **k: {"data": {"overview": {}}})
    assert ts.match_xg("mt_1") is None
    monkeypatch.setattr(ts, "_get", lambda *a, **k: None)   # network no-op
    assert ts.match_xg("mt_1") is None


def test_match_odds_exposes_btts(monkeypatch):
    payload = {"data": {"match_id": "mt_1", "bookmakers": [
        {"bookmaker": "Bet365", "markets": {
            "match_odds": {"home": {"opening": None, "last_seen": "2.700"},
                           "draw": {"last_seen": "3.100"}, "away": {"last_seen": "2.880"}},
            "btts": {"yes": {"last_seen": "1.910"}, "no": {"last_seen": "1.910"}},
            "total_goals": {"2.5": {"over": {"last_seen": "2.200"},
                                    "under": {"last_seen": "1.670"}}}}}]}}
    monkeypatch.setattr(ts, "_get", lambda *a, **k: payload)
    o = ts.match_odds("mt_1")
    assert o and o["bookmakers"][0]["bookmaker"] == "Bet365"
    assert "btts" in o["bookmakers"][0]["markets"]
    assert o["bookmakers"][0]["markets"]["btts"]["yes"]["last_seen"] == "1.910"


def test_match_lineups_requires_home(monkeypatch):
    payload = {"data": {"confirmed": True, "home": {"formation": "4-3-3",
               "starting_xi": [{"name": "X", "position": "G"}]}, "away": {}}}
    monkeypatch.setattr(ts, "_get", lambda *a, **k: payload)
    assert ts.match_lineups("mt_1")["home"]["formation"] == "4-3-3"
    monkeypatch.setattr(ts, "_get", lambda *a, **k: {"data": {"confirmed": False}})
    assert ts.match_lineups("mt_1") is None      # not announced yet


def test_no_key_is_graceful_noop(monkeypatch):
    monkeypatch.setattr(ts, "api_key", lambda: None)
    assert ts.is_available() is False
    assert ts.connectivity_check() == "no_key"
    assert ts._get("/anything") is None
