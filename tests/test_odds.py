"""Tests for odds conversion and de-vigging."""
import pytest

from src.data.odds import american_to_decimal, decimal_to_prob, devig


def test_american_to_decimal():
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert american_to_decimal(+150) == pytest.approx(2.5)
    assert american_to_decimal(+100) == pytest.approx(2.0)
    assert american_to_decimal(None) is None


def test_decimal_to_prob():
    assert decimal_to_prob(2.0) == pytest.approx(0.5)
    assert decimal_to_prob(4.0) == pytest.approx(0.25)


def test_devig_sums_to_one_and_removes_margin():
    # raw book probs sum to >1 (the overround)
    raw = [0.60, 0.30, 0.20]  # sums to 1.10
    fair = devig(raw, "proportional")
    assert fair is not None
    assert sum(fair) == pytest.approx(1.0)
    # ordering preserved, each shrunk
    assert fair[0] > fair[1] > fair[2]
    assert fair[0] < 0.60


def test_devig_handles_missing():
    assert devig([0.5, None, 0.3]) is None


def test_devig_shin_sums_to_one():
    fair = devig([0.55, 0.28, 0.25], "shin")
    assert fair is not None
    assert sum(fair) == pytest.approx(1.0, abs=1e-6)
