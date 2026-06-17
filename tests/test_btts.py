"""Tests for the real BTTS market wiring (TheStatsAPI line -> model EV/Kelly).

Network is mocked with the real captured odds shape (Bet365 ``btts`` yes/no with
``opening``/``last_seen``). Locks: price extraction + book preference, two-way de-vig, the
decimal-priced evaluator, and that ``value._attach_btts`` appends a Yes+No bet from the model's
P(BTTS) — and never fabricates a price when the market is absent.
"""
import pandas as pd

from src.data import thestatsapi as ts
from src.data.odds import decimal_to_american, american_to_decimal
from src.predict.betting import evaluate_bet_decimal, expected_value
from src.predict import value as value_mod


def test_decimal_to_american_roundtrips():
    for dec in (1.91, 2.10, 1.67, 3.50, 1.40):
        am = decimal_to_american(dec)
        assert abs(american_to_decimal(am) - dec) < 0.01
    assert decimal_to_american(1.0) is None


def _payload(yes="2.100", no="1.670", book="Bet365"):
    return {"bookmakers": [{"bookmaker": book, "markets": {
        "btts": {"yes": {"opening": None, "last_seen": yes},
                 "no": {"opening": None, "last_seen": no}}}}]}


def test_btts_prices_parses_real_shape():
    pr = ts.btts_prices(_payload())
    assert pr == {"book": "Bet365", "yes": 2.10, "no": 1.67}


def test_btts_prices_prefers_sharp_book():
    p = {"bookmakers": [
        {"bookmaker": "Bet365", "markets": {"btts": {"yes": {"last_seen": "2.00"},
                                                     "no": {"last_seen": "1.80"}}}},
        {"bookmaker": "Pinnacle", "markets": {"btts": {"yes": {"last_seen": "2.05"},
                                                       "no": {"last_seen": "1.85"}}}}]}
    assert ts.btts_prices(p)["book"] == "Pinnacle"


def test_btts_prices_none_when_one_side_missing_or_absent():
    half = {"bookmakers": [{"bookmaker": "Bet365",
                            "markets": {"btts": {"yes": {"last_seen": "2.0"}}}}]}
    assert ts.btts_prices(half) is None
    assert ts.btts_prices({"bookmakers": []}) is None
    assert ts.btts_prices(None) is None


def test_evaluate_bet_decimal_ev_sign():
    # model 60% on a 2.10 line -> clearly +EV; 40% -> -EV
    assert evaluate_bet_decimal("BTTS", "Both Teams Score: Yes", 2.10, 0.60, 0.55).ev > 0
    assert evaluate_bet_decimal("BTTS", "Both Teams Score: Yes", 2.10, 0.40, 0.55).ev < 0
    # decimal preserved exactly (no american round-trip loss)
    assert evaluate_bet_decimal("BTTS", "x", 2.10, 0.5, None).decimal == 2.10


def test_attach_btts_builds_yes_and_no(monkeypatch):
    monkeypatch.setattr(ts, "is_available", lambda: True)
    monkeypatch.setattr(ts, "matches", lambda **k: [
        {"id": "mt_1", "home_team": {"name": "Brazil"}, "away_team": {"name": "Spain"},
         "utc_date": "2026-06-20T18:00:00Z", "odds_available": True}])
    monkeypatch.setattr(ts, "match_odds", lambda mid, **k: _payload())
    matches = [{"home": "Brazil", "away": "Spain", "date": "2026-06-20",
                "analysis": {"btts": 0.62}, "bets": []}]
    value_mod._attach_btts(matches, bankroll=1000, kelly_fraction=0.25, cfg={})
    btts = [b for b in matches[0]["bets"] if b.market == "BTTS"]
    assert len(btts) == 2
    yes = next(b for b in btts if "Yes" in b.selection)
    no = next(b for b in btts if "No" in b.selection)
    assert abs(yes.model_p - 0.62) < 1e-9 and abs(no.model_p - 0.38) < 1e-9
    assert yes.decimal == 2.10 and no.decimal == 1.67
    # de-vig: fair_yes + fair_no == 1
    assert abs(yes.fair_p + no.fair_p - 1.0) < 1e-9
    assert matches[0]["btts_book"] == "Bet365"


def test_attach_btts_skips_when_no_odds_posted(monkeypatch):
    # the listing says odds_available=False -> skip the throttled /odds call entirely
    monkeypatch.setattr(ts, "is_available", lambda: True)
    monkeypatch.setattr(ts, "matches", lambda **k: [
        {"id": "mt_1", "home_team": {"name": "Brazil"}, "away_team": {"name": "Spain"},
         "utc_date": "2026-06-20T18:00:00Z", "odds_available": False}])
    called = {"odds": False}

    def _boom(*a, **k):
        called["odds"] = True
        return _payload()
    monkeypatch.setattr(ts, "match_odds", _boom)
    matches = [{"home": "Brazil", "away": "Spain", "date": "2026-06-20",
                "analysis": {"btts": 0.62}, "bets": []}]
    value_mod._attach_btts(matches, 1000, 0.25, {})
    assert matches[0]["bets"] == [] and called["odds"] is False   # no wasted call


def test_attach_btts_noop_without_key(monkeypatch):
    monkeypatch.setattr(ts, "is_available", lambda: False)
    matches = [{"home": "Brazil", "away": "Spain", "date": "2026-06-20",
                "analysis": {"btts": 0.62}, "bets": []}]
    value_mod._attach_btts(matches, 1000, 0.25, {})
    assert matches[0]["bets"] == []      # graceful no-op, no fabricated line
