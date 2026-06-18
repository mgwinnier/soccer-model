"""Tests for disciplined (longshot-tapered) stake sizing."""
from src.predict import staking
from src.predict.betting import kelly_fraction


def test_reliability_ratio_taper():
    assert staking.reliability_ratio(1.8) == 1.0          # favourite: untouched
    assert staking.reliability_ratio(2.6) == 1.0          # pickem: untouched
    assert staking.reliability_ratio(4.0) == 1.0          # boundary: untouched
    assert abs(staking.reliability_ratio(5.5) - 0.54) < 1e-9
    assert abs(staking.reliability_ratio(9.0) - 0.54) < 1e-9
    mid = staking.reliability_ratio(4.75)                  # halfway through the taper
    assert 0.54 < mid < 1.0 and abs(mid - 0.77) < 0.02


def test_reliability_p_only_hits_the_tail():
    assert staking.reliability_p(0.40, 2.5) == 0.40        # pickem unchanged
    assert staking.reliability_p(0.12, 6.0) < 0.12 * 0.6   # longshot haircut (~0.54x)


def test_units_unchanged_on_the_bulk():
    # odds <= 4.0 -> identical to today's capped quarter-Kelly
    for p, d in [(0.58, 1.8), (0.45, 2.6), (0.30, 3.5)]:
        expected = min(kelly_fraction(p, d) * 0.25 * 100, 2.0)
        assert abs(staking.recommended_units(p, None, d, 0.25, 2.0) - expected) < 1e-9


def test_units_taper_longshot_toward_zero():
    # 22% at 5.5 is above break-even (1/5.5=18.2%), so raw Kelly stakes something...
    raw = min(kelly_fraction(0.22, 5.5) * 0.25 * 100, 2.0)
    assert raw > 0
    rec = staking.recommended_units(0.22, None, 5.5, 0.25, 2.0)
    assert rec < raw                                          # ...the haircut sizes it down
    # 0.22 * 0.54 = 0.119 < 0.182 break-even -> Kelly collapses to 0
    assert rec == 0.0


def test_units_respects_cap_and_bad_input():
    assert staking.recommended_units(0.95, None, 1.6, 1.0, 2.0) == 2.0   # cap honoured
    assert staking.recommended_units(None, None, 2.0) == 0.0
    assert staking.recommended_units(0.5, None, 1.0) == 0.0              # no edge at evens


def test_recommended_ev_lower_on_tail():
    from src.predict.betting import expected_value
    # pickem: same as raw EV; longshot: lower than raw EV
    assert abs(staking.recommended_ev(0.45, 2.5) - expected_value(0.45, 2.5)) < 1e-9
    assert staking.recommended_ev(0.16, 6.0) < expected_value(0.16, 6.0)
