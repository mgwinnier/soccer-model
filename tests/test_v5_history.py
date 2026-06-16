"""Tests for historical-odds parsing, market-bias de-bias, and backtest stats."""
import numpy as np
import pytest

from src.data.odds import _pick_odds_entry
from src.models.market_bias import fit_market_bias, MarketBias
from src.backtest.odds_history_backtest import _bootstrap_roi


# --------------------------------------------------- Bet365 odds-array parse
def test_pick_odds_entry_prefers_bet365():
    data = {
        "pickcenter": [],
        "odds": [
            {"provider": {"name": "Bet 365"}, "homeTeamOdds": {}},          # empty
            {"provider": {"name": "Unibet"}, "homeTeamOdds": {"moneyLine": 160}},
            {"provider": {"name": "Bet365"}, "homeTeamOdds": {"moneyLine": 170}},
        ],
    }
    o = _pick_odds_entry(data)
    assert o["provider"]["name"] == "Bet365"
    assert o["homeTeamOdds"]["moneyLine"] == 170


def test_pick_odds_entry_falls_back_to_first_usable():
    data = {"pickcenter": [], "odds": [
        {"provider": {"name": "Unibet"}, "homeTeamOdds": {"moneyLine": 150}}]}
    assert _pick_odds_entry(data)["provider"]["name"] == "Unibet"


def test_pick_odds_entry_none_when_no_moneyline():
    assert _pick_odds_entry({"pickcenter": [], "odds": [{"homeTeamOdds": {}}]}) is None


# --------------------------------------------------------------- market bias
def test_fit_market_bias_per_role_mean():
    tk = ["TG:over", "TG:over", "MR:H"]
    mb = fit_market_bias(tk, [0.40, 0.50, 0.60], [0.50, 0.50, 0.50])
    assert mb.bias["TG:over"] == pytest.approx(-0.05)
    assert mb.bias["MR:H"] == pytest.approx(0.10)


def test_market_bias_recenter_applies_shrunk_offset():
    mb = MarketBias({"TG:over": -0.05}, shrink=0.8)
    # 0.40 - 0.8 * (-0.05) = 0.44
    assert mb.recenter("TG:over", 0.40) == pytest.approx(0.44)
    assert mb.recenter("UNKNOWN", 0.40) == 0.40       # no bias -> unchanged


def test_market_bias_roundtrip(tmp_path):
    mb = MarketBias({"MR:A": 0.03}, shrink=0.7)
    p = tmp_path / "bias.json"
    mb.save(p)
    loaded = MarketBias.load(p)
    assert loaded.bias == {"MR:A": 0.03} and loaded.shrink == 0.7


# ------------------------------------------------------------- bootstrap CI
def test_bootstrap_roi_brackets_mean():
    pnl = np.concatenate([np.full(60, 0.9), np.full(40, -1.0)])  # +EV-ish sample
    mean, lo, hi = _bootstrap_roi(pnl, n=500)
    assert lo <= mean <= hi
    assert mean == pytest.approx(pnl.mean(), abs=1e-9)


def test_bootstrap_roi_empty():
    mean, lo, hi = _bootstrap_roi(np.array([]))
    assert np.isnan(mean) and np.isnan(lo)
