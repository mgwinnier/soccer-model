"""Tests for the Elo rating engine math."""
import pytest

from src.features.elo import EloEngine, expected_score, goal_diff_multiplier


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1900, 1500) > 0.9
    a = expected_score(1700, 1500)
    b = expected_score(1500, 1700)
    assert a + b == pytest.approx(1.0)


def test_home_advantage_shifts_expectation():
    assert expected_score(1500, 1500, home_adv=100) > 0.5


def test_goal_diff_multiplier():
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(3) == pytest.approx((11 + 3) / 8)
    assert goal_diff_multiplier(4) == pytest.approx((11 + 4) / 8)


def test_update_conserves_and_directs_points():
    e = EloEngine(base_rating=1500, k_factor=40, home_advantage=0)
    rh, ra = e.update_one("A", "B", 1, 0, neutral=True, importance=1.0)
    assert rh == 1500 and ra == 1500  # pre-match ratings returned
    # equal teams, home wins by 1 -> +/- 20, and total rating conserved
    assert e.rating("A") == pytest.approx(1520)
    assert e.rating("B") == pytest.approx(1480)
    assert e.rating("A") + e.rating("B") == pytest.approx(3000)


def test_bigger_win_moves_more():
    e1 = EloEngine(base_rating=1500, k_factor=40, home_advantage=0)
    e1.update_one("A", "B", 1, 0, neutral=True)
    e2 = EloEngine(base_rating=1500, k_factor=40, home_advantage=0)
    e2.update_one("A", "B", 4, 0, neutral=True)
    assert e2.rating("A") > e1.rating("A")
