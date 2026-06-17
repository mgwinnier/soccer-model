"""Tests for the squad market-value reader (accent-tolerant name match + value-based key-out)."""
from src.data import squad_values as sv

TEAMS = {"Portugal": {"team_id": "tm_09479", "total_value": 255_000_000, "players": [
    {"name": "João Neves", "position": "M", "market_value": 148_000_000},
    {"name": "Vitinha", "position": "M", "market_value": 102_000_000},
    {"name": "Cristiano Ronaldo", "position": "F", "market_value": 5_000_000}]}}


def _patch(monkeypatch):
    monkeypatch.setattr(sv, "_load", lambda *a, **k: TEAMS)


def test_player_value_accent_and_lastname(monkeypatch):
    _patch(monkeypatch)
    assert sv.player_value("Portugal", "Joao Neves") == 148_000_000      # accent-insensitive
    assert sv.player_value("Portugal", "Neves") == 148_000_000           # last-name fallback
    assert sv.player_value("Portugal", "Nobody") is None
    assert sv.total_value("Portugal") == 255_000_000


def test_key_absentees(monkeypatch):
    _patch(monkeypatch)
    # XI has only Ronaldo -> the two most valuable (Neves, Vitinha) are flagged out
    out = sv.key_absentees("Portugal", ["Cristiano Ronaldo"], top_n=2)
    assert {x["name"] for x in out} == {"João Neves", "Vitinha"}
    # Neves in the XI -> not flagged
    out2 = sv.key_absentees("Portugal", ["Joao Neves", "Vitinha"], top_n=2)
    assert out2 == []


def test_xi_value(monkeypatch):
    _patch(monkeypatch)
    s, share = sv.xi_value("Portugal", ["Joao Neves", "Vitinha"])   # 250M of 255M total
    assert s == 250_000_000 and abs(share - 250_000_000 / 255_000_000) < 1e-9
    assert sv.xi_value("Atlantis", ["x"]) == (None, None)


def test_unknown_team(monkeypatch):
    _patch(monkeypatch)
    assert sv.player_value("Atlantis", "X") is None
    assert sv.key_absentees("Atlantis", [], top_n=3) == []
