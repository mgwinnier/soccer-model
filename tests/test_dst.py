"""Tests for the DST open book-odds client (network mocked with the real captured shape).

DST serves the user's PayPerHead book line on an open host (no auth). These lock the market
parse (3-way moneyline / totals ladder / BTTS) and the home/away orientation by team pair.
"""
from src.data import dst


GROUPED = [
    {"statistic": "to win", "markets": [
        {"condition": [{"type": "1"}], "odds": 1.30},
        {"condition": [{"type": "x"}], "odds": 5.10},
        {"condition": [{"type": "2"}], "odds": 10.0}]},
    {"statistic": "both teams to score", "markets": [
        {"condition": [{"type": "yes"}], "odds": 2.15},
        {"condition": [{"type": "no"}], "odds": 1.63}]},
    {"statistic": "total goals", "markets": [
        {"condition": [{"type": "over", "value": 2.5}], "odds": 1.74},
        {"condition": [{"type": "under", "value": 2.5}], "odds": 1.97},
        {"condition": [{"type": "over", "value": 3.5}], "odds": 2.90}]},  # no under 3.5 -> dropped
]
GAMES = [{"providers": [{"id": 276472}],
          "team1": [{"title": "portugal"}], "team2": [{"title": "dr congo"}]}]


def test_parse_markets():
    p = dst.parse_markets(GROUPED)
    assert p["moneyline"] == {"H": 1.30, "D": 5.10, "A": 10.0}
    assert p["btts"] == {"yes": 2.15, "no": 1.63}
    assert p["totals"] == {2.5: {"over": 1.74, "under": 1.97}}   # 3.5 has no under -> excluded


def test_book_odds_orientation(monkeypatch):
    monkeypatch.setattr(dst, "game_markets", lambda g, **k: GROUPED)
    a = dst.book_odds("Portugal", "DR Congo", gms=GAMES)
    assert a["moneyline"]["H"] == 1.30 and a["moneyline"]["A"] == 10.0 and a["flipped"] is False
    b = dst.book_odds("DR Congo", "Portugal", gms=GAMES)   # reversed -> H must be DRC
    assert b["moneyline"]["H"] == 10.0 and b["moneyline"]["A"] == 1.30 and b["flipped"] is True
    assert a["btts"] == b["btts"]                            # symmetric


def test_book_odds_none_when_absent(monkeypatch):
    monkeypatch.setattr(dst, "game_markets", lambda g, **k: GROUPED)
    assert dst.book_odds("Brazil", "Spain", gms=GAMES) is None
