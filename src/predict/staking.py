"""Disciplined stake sizing — a display/recommendation layer on top of Kelly.

The deployed probabilities stay pure (market-independent); this module only decides *how much to
recommend staking*. The one adjustment: **down-weight longshots**, because the model overrates the tail.

Grounded + validated: on ~14.5k out-of-sample international bets the 5.0+ odds bucket wins ~7.8% vs the
~11.8% the model expects (reliability ratio ≈ 0.54). On the 2022 WC that tail went 0/8 (−5.81u); on the
2026 WC it ran hot — sizing it down kept ~all the upside (+4.89 of +5.06u) on a tenth of the stake. The
well-calibrated buckets (odds ≤ ~4) were ~1.0 and recalibrating them only added noise, so we leave them
**exactly as today's capped quarter-Kelly**. Only the tail (≳4.5) tapers.
"""
from __future__ import annotations

from .betting import expected_value, kelly_fraction

# Measured out-of-sample reliability of the 5.0+ longshot bucket (actual win rate ÷ mean model prob).
_LONGSHOT_RELIABILITY = 0.54
_TAPER_LO = 4.0      # ≤ this: untouched (well-calibrated)
_TAPER_HI = 5.5      # ≥ this: full longshot haircut


def reliability_ratio(decimal: float | None) -> float:
    """Stake-sizing reliability multiplier on the win prob: 1.0 for odds ≤ 4.0 (no change to the
    calibrated bulk), tapering linearly to ~0.54 by 5.5, flat beyond. Never touches favourites/pickems."""
    if decimal is None or decimal <= _TAPER_LO:
        return 1.0
    if decimal >= _TAPER_HI:
        return _LONGSHOT_RELIABILITY
    frac = (decimal - _TAPER_LO) / (_TAPER_HI - _TAPER_LO)
    return 1.0 + (_LONGSHOT_RELIABILITY - 1.0) * frac


def reliability_p(model_p: float | None, decimal: float | None) -> float | None:
    """The win probability used for *sizing only* (never for display): model_p haircut on the tail."""
    if model_p is None:
        return None
    return float(min(max(model_p * reliability_ratio(decimal), 1e-4), 0.99))


def recommended_units(model_p: float | None, fair_p: float | None, decimal: float | None,
                      frac: float = 0.25, cap_u: float = 2.0) -> float:
    """Recommended stake in units (1u = 1% of bankroll): fractional Kelly on the reliability-adjusted
    win prob, capped. For odds ≤ 4.0 this equals today's capped quarter-Kelly exactly; longshots taper
    toward ~0 as Kelly collapses on the haircut probability."""
    if model_p is None or decimal is None or not (decimal > 1):
        return 0.0
    p_adj = reliability_p(model_p, decimal)
    kf = kelly_fraction(p_adj, decimal)
    return float(min(kf * frac * 100.0, cap_u))


def recommended_ev(model_p: float | None, decimal: float | None) -> float:
    """EV at the reliability-adjusted prob — a 'true-EV' for the tail (longshots read far lower)."""
    if model_p is None or decimal is None:
        return 0.0
    return expected_value(reliability_p(model_p, decimal), decimal)
