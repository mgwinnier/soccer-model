"""Tests for tournament structure and group-standings logic."""
import numpy as np
import pandas as pd
import pytest

from src.simulate.bracket_2026 import GROUPS, all_teams
from src.config import path_for


def test_groups_are_complete_and_disjoint():
    teams = all_teams()
    assert len(teams) == 48
    assert len(set(teams)) == 48
    assert len(GROUPS) == 12
    assert all(len(v) == 4 for v in GROUPS.values())


def test_bracket_seed_order_is_permutation():
    from src.simulate.tournament import _BRACKET32
    assert sorted(_BRACKET32) == list(range(1, 33))


def test_group_standings_composite_orders_correctly():
    """Points dominate GD, GD dominates GF (the composite-key contract)."""
    # team scores: (points, gd, gf)
    pts = np.array([[9, 6, 3, 0]], dtype=float)
    gd = np.array([[5, 5, 1, -11]], dtype=float)
    gf = np.array([[8, 3, 4, 1]], dtype=float)
    elo = np.array([1500, 1500, 1500, 1500.0])
    comp = pts * 1e6 + gd * 1e3 + gf * 10 + elo[None, :] * 1e-3
    order = np.argsort(-comp, axis=1)[0]
    assert list(order) == [0, 1, 2, 3]  # by points


def test_goal_difference_breaks_point_ties():
    pts = np.array([[6, 6, 6, 6]], dtype=float)
    gd = np.array([[3, 1, -1, -3]], dtype=float)
    gf = np.array([[5, 5, 5, 5]], dtype=float)
    elo = np.array([1500.0] * 4)
    comp = pts * 1e6 + gd * 1e3 + gf * 10 + elo[None, :] * 1e-3
    order = np.argsort(-comp, axis=1)[0]
    assert list(order) == [0, 1, 2, 3]  # by GD


def test_simulation_smoke():
    """Integration: a small sim yields valid champion probabilities."""
    if not (path_for("data_processed") / "matches.parquet").exists():
        pytest.skip("processed data not built")
    from src.simulate.tournament import run_simulation
    df = run_simulation(n_iter=500, write=False)
    assert len(df) == 48
    assert df["champion"].sum() == pytest.approx(1.0, abs=1e-6)
    assert (df["advance"] <= 1.0).all() and (df["advance"] >= 0.0).all()
    # the strongest sides should carry meaningful title odds
    top = set(df.head(8)["team"])
    assert len({"Argentina", "Spain", "England", "France", "Brazil"} & top) >= 3
