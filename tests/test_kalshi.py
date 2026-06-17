"""Tests for the Kalshi exchange-odds client (network mocked with the real captured shape)."""
from src.data import kalshi as k


# Real KXWCGAME shape: 3 markets per event, prices in the *_dollars fields, team in yes_sub_title.
GHAPAN = [
    {"event_ticker": "KXWCGAME-26JUN17GHAPAN", "ticker": "KXWCGAME-26JUN17GHAPAN-GHA",
     "yes_sub_title": "Ghana", "yes_bid_dollars": 0.42, "yes_ask_dollars": 0.43,
     "last_price_dollars": 0.43, "previous_yes_ask_dollars": 0.44, "status": "active"},
    {"event_ticker": "KXWCGAME-26JUN17GHAPAN", "ticker": "KXWCGAME-26JUN17GHAPAN-TIE",
     "yes_sub_title": "Tie", "yes_bid_dollars": 0.29, "yes_ask_dollars": 0.30,
     "last_price_dollars": 0.30, "previous_yes_ask_dollars": 0.29, "status": "active"},
    {"event_ticker": "KXWCGAME-26JUN17GHAPAN", "ticker": "KXWCGAME-26JUN17GHAPAN-PAN",
     "yes_sub_title": "Panama", "yes_bid_dollars": 0.29, "yes_ask_dollars": 0.30,
     "last_price_dollars": 0.30, "previous_yes_ask_dollars": 0.29, "status": "active"},
]
# A fixture using Kalshi's name variants (must alias to our canonical names).
CODUZB = [
    {"event_ticker": "KXWCGAME-26JUN27CODUZB", "ticker": "...-COD", "yes_sub_title": "Congo DR",
     "yes_bid_dollars": 0.40, "yes_ask_dollars": 0.42, "last_price_dollars": 0.41,
     "previous_yes_ask_dollars": 0.40},
    {"event_ticker": "KXWCGAME-26JUN27CODUZB", "ticker": "...-TIE", "yes_sub_title": "Tie",
     "yes_bid_dollars": 0.27, "yes_ask_dollars": 0.29, "last_price_dollars": 0.28,
     "previous_yes_ask_dollars": 0.28},
    {"event_ticker": "KXWCGAME-26JUN27CODUZB", "ticker": "...-UZB", "yes_sub_title": "Uzbekistan",
     "yes_bid_dollars": 0.30, "yes_ask_dollars": 0.32, "last_price_dollars": 0.31,
     "previous_yes_ask_dollars": 0.31},
]
ALL = GHAPAN + CODUZB


def test_match_winner_maps_outcomes():
    w = k.match_winner("Ghana", "Panama", markets=ALL)
    assert w["H"]["ask"] == 0.43 and w["A"]["ask"] == 0.30 and w["D"]["ask"] == 0.30
    assert w["H"]["bid"] == 0.42 and w["H"]["prev_ask"] == 0.44       # movement field present
    assert w["event"] == "KXWCGAME-26JUN17GHAPAN"


def test_match_winner_orientation_swaps_by_name():
    a = k.match_winner("Ghana", "Panama", markets=ALL)
    b = k.match_winner("Panama", "Ghana", markets=ALL)               # reversed
    assert b["H"]["ask"] == a["A"]["ask"] and b["A"]["ask"] == a["H"]["ask"]
    assert a["D"]["ask"] == b["D"]["ask"]                            # tie symmetric


def test_match_winner_aliases_kalshi_names():
    # "Congo DR" must resolve to our "DR Congo"
    w = k.match_winner("DR Congo", "Uzbekistan", markets=ALL)
    assert w is not None and w["H"]["ask"] == 0.42 and w["A"]["ask"] == 0.32


def test_match_winner_none_when_absent():
    assert k.match_winner("Brazil", "Spain", markets=ALL) is None
    assert k.match_winner("Ghana", "Brazil", markets=ALL) is None    # only one team present


def test_ask_decimal_and_norm():
    assert abs(k.ask_decimal(0.40) - 2.5) < 1e-9
    assert k.ask_decimal(0) is None and k.ask_decimal(None) is None
    assert k._norm("Turkiye") == k._norm("Turkey")                   # alias collapses
    assert k._norm("IR Iran") == k._norm("Iran")


def test_signal_buy_sell_hold():
    # model 0.40 vs ask 0.30 -> BUY (underpriced), ev_buy > 0
    s = k.signal(0.40, 0.29, 0.30, buy_edge=0.05, sell_edge=0.05)
    assert s["action"] == "BUY" and s["ev_buy"] > 0
    # model 0.20 vs bid 0.29 -> SELL (market richer than model); No side has +EV
    s = k.signal(0.20, 0.29, 0.30)
    assert s["action"] == "SELL" and s["ev_sell"] > 0
    # inside the spread -> HOLD
    assert k.signal(0.30, 0.29, 0.30)["action"] == "HOLD"
    # missing prices never crash
    assert k.signal(None, None, None)["action"] == "HOLD"


def test_winner_futures_parses(monkeypatch):
    monkeypatch.setattr(k, "_markets", lambda *a, **kw: [
        {"yes_sub_title": "France", "yes_bid_dollars": 0.184, "yes_ask_dollars": 0.186,
         "last_price_dollars": 0.186, "previous_yes_ask_dollars": 0.185, "ticker": "KXMENWORLDCUP-26-FR"},
        {"yes_sub_title": "Turkiye", "yes_bid_dollars": 0.004, "yes_ask_dollars": 0.005,
         "last_price_dollars": 0.004, "previous_yes_ask_dollars": 0.004, "ticker": "KXMENWORLDCUP-26-TR"}])
    fut = k.winner_futures()
    assert fut["France"]["ask"] == 0.186
    assert "Turkey" in fut                                            # aliased from "Turkiye"
