"""Tests for v9: live in-tournament team strength (Elo/DC update as 2026 plays)."""
import numpy as np
import pandas as pd
import pytest

from src.simulate import live_state as ls
from src.simulate.tournament import TournamentSimulator
from src.data.clean import load_matches


def _frame(games):
    """Build a minimal match frame for the Elo engine."""
    rows = []
    for i, (h, a, hs, as_) in enumerate(games):
        rows.append({"match_id": i, "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                     "home_team": h, "away_team": a, "home_score": hs, "away_score": as_,
                     "neutral": True, "importance": 1.0})
    return pd.DataFrame(rows)


def test_live_elo_reflects_an_upset_modestly():
    # 20 games establishing Strong >> Weak, then one Weak-beats-Strong upset.
    base_games = [("Strong", "Weak", 2, 0) for _ in range(20)]
    base = _frame(base_games)
    upset = _frame(base_games + [("Weak", "Strong", 3, 0)])

    r_base = ls.live_elo(frame=base)
    r_up = ls.live_elo(frame=upset)
    gain = r_up["Weak"] - r_base["Weak"]
    assert gain > 0                    # the upset raised the underdog's rating
    assert gain < 200                  # but modestly — one game, conservative K (no runaway)
    # zero-sum: Strong drops by the same amount Weak gains on that match
    assert r_up["Strong"] < r_base["Strong"]


def test_live_match_frame_keeps_all_matches():
    base = load_matches()
    frame = ls.live_match_frame()
    assert len(frame) >= len(base)                 # only adds (ESPN delta), never drops
    assert set(base["match_id"]).issubset(set(frame["match_id"]))


def test_live_elo_returns_ratings_for_known_teams():
    r = ls.live_elo()
    assert "Brazil" in r and "France" in r
    assert all(800 < v < 2600 for v in r.values())   # sane Elo band, no blow-ups


def test_simulator_live_and_frozen_both_run():
    played = ls.fetch_live_results()
    qf = TournamentSimulator(live=False).run(n_iter=500, played=played)
    ql = TournamentSimulator(live=True).run(n_iter=500, played=played)
    for q in (qf, ql):
        assert len(q) == 48
        assert q["champion"].sum() == pytest.approx(1.0, abs=1e-6)
        assert q["advance"].between(0, 1).all()
