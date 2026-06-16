"""Tests for the matchup-strength-aware World-Cup goals correction.

The model under-projects WC goals vs ACTUAL results, and not uniformly — the favorite
(higher expected goals) is under-projected more than the underdog. ``models/wc_goals.py``
scales the favored/underdog sides by separate factors. These tests pin the mechanism, not
exact fitted numbers.
"""
import numpy as np

from src.models import wc_goals
from src.predict.predict_match import MatchPredictor


def test_scales_are_meaningful_and_favorite_bigger():
    fav, dog = wc_goals.load_scales()
    assert 1.0 < dog < fav < 1.40          # both uplift; favorite gets the larger one


def test_correct_applies_favorite_scale_to_higher_side():
    fav, dog = wc_goals.WC_FAV_SCALE, wc_goals.WC_DOG_SCALE
    # lam is the favorite
    lam, mu = wc_goals.correct(2.0, 1.0)
    assert np.isclose(lam, 2.0 * fav) and np.isclose(mu, 1.0 * dog)
    # mu is the favorite (lower lam) -> scales swap
    lam2, mu2 = wc_goals.correct(0.8, 1.6)
    assert np.isclose(lam2, 0.8 * dog) and np.isclose(mu2, 1.6 * fav)


def test_favorite_gets_a_larger_uplift_than_its_underdog():
    # a heavy-favorite matchup: the favored side's goals rise by a bigger fraction
    lam, mu = 2.4, 0.6
    clam, cmu = wc_goals.correct(lam, mu)
    assert (clam / lam) > (cmu / mu)


def test_predictor_applies_correction_no_market_line_needed():
    mp = MatchPredictor()
    raw = mp.dc.expected_goals("Spain", "Cape Verde", True)     # un-corrected DC
    a = mp.analyze("Spain", "Cape Verde", neutral=True)         # no line passed
    eg = a["expected_goals"]
    # corrected total exceeds raw (under-projection fixed) and favorite side scaled most
    assert sum(eg) > sum(raw)
    fav_raw = max(raw); fav_corr = max(eg)
    assert (fav_corr / fav_raw) > 1.10


def test_outcome_probs_normalize():
    mp = MatchPredictor()
    p = mp.analyze("Argentina", "Mexico", neutral=True)["probs"]
    assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-6
