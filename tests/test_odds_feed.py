"""Tests for the Premium Odds Feed client (network mocked with the real captured shape).

Locks the de-vig of the sharp 1X2/totals snapshot and the honest None paths.
"""
from src.data import odds_feed as of


def test_sharp_1x2_devig(monkeypatch):
    payload = {"data": {"match_id": "pmt_1", "is_settled": False, "odds": [
        {"market": "moneyline", "period": 0, "price_1": 1.30, "price_x": 5.65,
         "price_2": 13.25, "cutoff_at": "2026-06-17T17:00:00", "recorded_at": "2026-06-17T14:00:00"}]}}
    monkeypatch.setattr(of, "_get", lambda *a, **k: payload)
    r = of.sharp_1x2("pmt_1")
    assert r["dec"]["H"] == 1.30 and r["cutoff_at"] == "2026-06-17T17:00:00"
    f = r["fair"]
    assert abs(f["H"] + f["D"] + f["A"] - 1.0) < 1e-9 and f["H"] > 0.7   # heavy favorite


def test_sharp_total_picks_line(monkeypatch):
    payload = {"data": {"odds": [
        {"market": "totals", "period": 0, "line": 2.5, "price_1": 1.90, "price_2": 1.95},
        {"market": "totals", "period": 0, "line": 3.5, "price_1": 3.1, "price_2": 1.38}]}}
    monkeypatch.setattr(of, "_get", lambda *a, **k: payload)
    r = of.sharp_total("pmt_1", line=2.5)
    assert r["line"] == 2.5 and r["dec"]["over"] == 1.90
    assert abs(r["fair"]["over"] + r["fair"]["under"] - 1.0) < 1e-9


def test_none_paths(monkeypatch):
    monkeypatch.setattr(of, "_get", lambda *a, **k: None)
    assert of.sharp_1x2("pmt_x") is None
    assert of.sharp_total("pmt_x") is None
    monkeypatch.setattr(of, "_get", lambda *a, **k: {"data": {"odds": []}})
    assert of.sharp_1x2("pmt_x") is None
