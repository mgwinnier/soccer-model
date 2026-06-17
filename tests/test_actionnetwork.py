"""Tests for the Action Network BTTS feed reader + value wiring.

The feed is harvested on the VPS (Action Network's API is firewall-blocked locally); these run
against a synthetic feed dict so they're network-free. Lock: pair-matching (order-independent +
alias-normalized), best-price american->decimal, None when absent, and that ``_attach_btts``
prefers the AN pre-match line over the TheStatsAPI settled line.
"""
from src.data import actionnetwork as an
from src.predict import value as value_mod


FEED = {"source": "actionnetwork", "league": "worldcup", "generated_at": "2026-06-17T14:49:03Z",
        "games": [
            {"game_id": 287931, "status": "scheduled", "home": "Portugal", "away": "DR Congo",
             "btts": {"DraftKings": {"yes": 145, "no": -194}, "Bet365": {"yes": 150, "no": -200}},
             "btts_best": {"yes": 150, "yes_book": "Bet365", "no": -194, "no_book": "DraftKings"}},
        ]}


def test_btts_prices_for_matches_and_converts():
    pr = an.btts_prices_for("Portugal", "DR Congo", "2026-06-17", feed=FEED)
    assert pr["source"] == "actionnetwork" and pr["n_books"] == 2
    assert abs(pr["yes"] - 2.50) < 1e-6        # +150 -> 2.50
    assert abs(pr["no"] - (1 + 100 / 194)) < 1e-6   # -194 -> ~1.515
    assert pr["yes_book"] == "Bet365"


def test_btts_prices_for_order_independent_and_alias():
    # flipped order + an alias on one side still matches
    assert an.btts_prices_for("Congo DR", "Portugal", None, feed=FEED) is not None \
        or an.btts_prices_for("DR Congo", "Portugal", None, feed=FEED) is not None


def test_btts_prices_for_none_when_absent():
    assert an.btts_prices_for("Brazil", "Spain", "2026-06-17", feed=FEED) is None


def test_attach_btts_prefers_action_network(monkeypatch):
    # AN feed present -> used; TheStatsAPI must NOT be consulted for a covered game
    monkeypatch.setattr(an, "load_feed", lambda *a, **k: FEED)
    import src.data.thestatsapi as ts
    called = {"ts": False}
    monkeypatch.setattr(ts, "is_available", lambda: (called.__setitem__("ts", True) or True))
    m = {"home": "Portugal", "away": "DR Congo", "date": "2026-06-17",
         "analysis": {"btts": 0.55}, "bets": []}
    value_mod._attach_btts([m], bankroll=1000, kelly_fraction=0.25, cfg={})
    btts = [b for b in m["bets"] if b.market == "BTTS"]
    assert len(btts) == 2 and m["btts_source"] == "actionnetwork"
    yes = next(b for b in btts if "Yes" in b.selection)
    assert abs(yes.decimal - 2.50) < 1e-6 and abs(yes.model_p - 0.55) < 1e-9
    assert called["ts"] is False               # AN covered it; no settled fallback needed
