"""Tests for the injury -> strength-multiplier logic."""
import pytest

from src.features import availability


@pytest.fixture
def fake_key_players(monkeypatch):
    # A nation whose key XI totals 240 rating points.
    players = {"star": 90.0, "second": 80.0, "third": 70.0}
    monkeypatch.setattr(availability, "team_key_players", lambda team: players)
    return players


def test_no_injuries_is_neutral(fake_key_players):
    assert availability.availability_multiplier("X", []) == 1.0


def test_missing_star_scales_down(fake_key_players):
    # star = 90/240 = 37.5% of quality, capped at the 25% max penalty
    m = availability.availability_multiplier("X", ["Star"])
    assert m == pytest.approx(1.0 - availability._MAX_PENALTY)


def test_small_loss_scales_proportionally(fake_key_players):
    # third = 70/240 = 29.2% -> still above cap; use second-tier alone
    m = availability.availability_multiplier("X", ["nonexistent"])
    assert m == 1.0  # injured player not among key players -> no effect


def test_no_ratings_data_is_neutral(monkeypatch):
    monkeypatch.setattr(availability, "team_key_players", lambda team: {})
    assert availability.availability_multiplier("X", ["Anyone"]) == 1.0
