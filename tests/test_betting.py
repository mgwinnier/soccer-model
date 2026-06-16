"""Tests for EV, Kelly, over/under + spread identities, and exposure capping."""
import numpy as np
import pandas as pd
import pytest

from src.predict.betting import expected_value, kelly_fraction, kelly_stake, evaluate_bet
from src.predict.predict_match import _over_prob, _cover_prob
from src.predict.value import best_bets, cap_exposure


# ----------------------------------------------------------------- EV
def test_ev_zero_at_fair_odds():
    # fair decimal odds for p=0.5 is 2.0 -> EV exactly 0
    assert expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert expected_value(0.25, 4.0) == pytest.approx(0.0)


def test_ev_sign():
    assert expected_value(0.6, 2.0) > 0      # underpriced -> +EV
    assert expected_value(0.4, 2.0) < 0      # overpriced -> -EV


# ----------------------------------------------------------------- Kelly
def test_kelly_known_value():
    # p=0.6, decimal=2.0 -> (0.6*2 - 1)/(2-1) = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.20)


def test_kelly_zero_without_edge():
    assert kelly_fraction(0.5, 2.0) == 0.0   # break-even -> no bet
    assert kelly_fraction(0.3, 2.0) == 0.0   # negative -> clipped to 0


def test_kelly_capped_and_stake():
    assert 0.0 <= kelly_fraction(0.99, 10.0) <= 1.0
    # half-Kelly stake of the 0.20 fraction on a $1000 bankroll = $100
    assert kelly_stake(0.6, 2.0, bankroll=1000, fraction=0.5) == pytest.approx(100.0)


def test_evaluate_bet_bundles_fields():
    b = evaluate_bet("Match Result", "France", american=-200, model_p=0.7,
                     fair_p=0.66, bankroll=1000, fraction=0.5)
    assert b.decimal == pytest.approx(1.5)
    assert b.edge == pytest.approx(0.04)
    assert b.ev == pytest.approx(0.7 * 0.5 - 0.3)


# --------------------------------------------------- over/under + spread
def _toy_matrix():
    # small Poisson-ish 4x4 matrix, normalized
    lam, mu = 1.3, 1.0
    k = np.arange(4)
    fact = np.array([1, 1, 2, 6])
    ph = np.exp(-lam) * lam ** k / fact
    pa = np.exp(-mu) * mu ** k / fact
    mat = np.outer(ph, pa)
    return mat / mat.sum()


def test_over_under_complementary():
    mat = _toy_matrix()
    assert _over_prob(mat, 2.5) + (1 - _over_prob(mat, 2.5)) == pytest.approx(1.0)
    # ladder is monotonically decreasing
    ladder = [_over_prob(mat, ln) for ln in (0.5, 1.5, 2.5, 3.5)]
    assert all(ladder[i] >= ladder[i + 1] for i in range(len(ladder) - 1))


def test_spread_halfline_sums_to_one_no_push():
    mat = _toy_matrix()
    ph, push, pa = _cover_prob(mat, -1.5)
    assert push == pytest.approx(0.0)
    assert ph + pa == pytest.approx(1.0)


def test_spread_integer_line_has_push():
    mat = _toy_matrix()
    ph, push, pa = _cover_prob(mat, -1.0)   # home -1: push when home wins by exactly 1
    assert push > 0
    assert ph + push + pa == pytest.approx(1.0)


# ----------------------------------------------------------- value board
def test_best_bets_filters_and_sorts():
    df = pd.DataFrame({
        "ev": [0.10, -0.05, 0.30, 0.02], "stake": [10, 0, 30, 5],
        "match": ["a", "b", "c", "d"], "market": ["x"] * 4,
        "selection": ["s"] * 4, "american": [100] * 4, "model_p": [0.5] * 4,
        "fair_p": [0.5] * 4, "edge": [0.0] * 4, "kelly_used": [0.0] * 4,
        "date": [pd.Timestamp("2026-06-16")] * 4,
    })
    bb = best_bets(df, min_ev=0.05)
    assert list(bb["ev"]) == [0.30, 0.10]   # filtered + sorted desc


def test_cap_exposure_scales_down():
    df = pd.DataFrame({"stake": [600.0, 800.0], "ev": [0.1, 0.1]})
    capped = cap_exposure(df, bankroll=1000, max_fraction=1.0)
    assert capped["stake"].sum() == pytest.approx(1000.0)
    # already under cap -> unchanged
    df2 = pd.DataFrame({"stake": [100.0, 200.0], "ev": [0.1, 0.1]})
    assert cap_exposure(df2, 1000, 1.0)["stake"].sum() == pytest.approx(300.0)
