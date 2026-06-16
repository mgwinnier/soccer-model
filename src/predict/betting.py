"""Bet evaluation: Expected Value and Kelly-criterion staking.

Pure functions over a model probability `p` and the bookmaker's **offered decimal
odds** `d` (the vigged price you actually get paid at — *not* the de-vigged "fair"
probability). All staking is intentionally conservative-friendly: Kelly is clipped
to [0, 1] and the UI defaults to a fractional multiplier.

Honest framing: EV and Kelly are only as trustworthy as `p`. A positive EV means
"the model thinks this price is too long", not "free money".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from ..data.odds import american_to_decimal, decimal_to_prob


def expected_value(p: float, decimal_odds: float) -> float:
    """EV per $1 staked = p·(d−1) − (1−p).  Zero at fair odds (d = 1/p)."""
    if decimal_odds is None or decimal_odds <= 1 or p is None:
        return 0.0
    return p * (decimal_odds - 1.0) - (1.0 - p)


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Full-Kelly bankroll fraction = (p·d − 1)/(d − 1), clipped to [0, 1].

    Returns 0 when there is no edge (the bet should not be made)."""
    if decimal_odds is None or decimal_odds <= 1 or p is None:
        return 0.0
    b = decimal_odds - 1.0
    f = (p * decimal_odds - 1.0) / b
    return float(min(max(f, 0.0), 1.0))


def kelly_stake(p: float, decimal_odds: float, bankroll: float,
                fraction: float = 0.5) -> float:
    """Recommended $ stake = bankroll · fraction · full-Kelly."""
    return round(bankroll * fraction * kelly_fraction(p, decimal_odds), 2)


@dataclass
class BetEval:
    market: str          # "Match Result" | "Total Goals" | "Spread"
    selection: str       # e.g. "France", "Over 2.5", "France -1.5"
    american: float | None
    decimal: float | None
    model_p: float       # model probability of this selection
    fair_p: float | None  # de-vigged market probability (for transparency)
    edge: float | None    # model_p - fair_p
    ev: float            # per $1 staked, at the offered price
    kelly_full: float
    kelly_used: float    # kelly_full * fraction
    stake: float         # $ at the given bankroll/fraction

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_bet(market: str, selection: str, american: float | None,
                 model_p: float, fair_p: float | None,
                 bankroll: float = 100.0, fraction: float = 0.5) -> BetEval:
    """Bundle EV + Kelly for one selection at its offered American price."""
    dec = american_to_decimal(american)
    ev = expected_value(model_p, dec) if dec else 0.0
    kf = kelly_fraction(model_p, dec) if dec else 0.0
    edge = (model_p - fair_p) if (fair_p is not None) else None
    return BetEval(
        market=market, selection=selection, american=american, decimal=dec,
        model_p=model_p, fair_p=fair_p, edge=edge, ev=ev,
        kelly_full=kf, kelly_used=kf * fraction,
        stake=round(bankroll * fraction * kf, 2),
    )
