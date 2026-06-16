"""Bivariate Poisson goals model (Karlis–Ntzoufras) with a shared component.

Home = X + Z, Away = Y + Z with X~Pois(λ1), Y~Pois(λ2), Z~Pois(λ3); the shared
Z induces **goal correlation** (Cov = λ3). We reuse the Dixon-Coles two-stage fit
for the marginal attack/defense rates, then fit a single global λ3 by MLE and
build the scoreline matrix from the bivariate PMF instead of independent Poisson.

International football goals are only weakly correlated, so λ3 is expected to be
small — the model is offered as an ablation-gated alternative, kept only if it
improves market calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from .dixon_coles import DixonColesModel, _fact

_MIN = 1e-6


def _bp_matrix(lam1: float, lam2: float, lam3: float, max_goals: int) -> np.ndarray:
    """Bivariate-Poisson scoreline probability matrix (home rows, away cols)."""
    k = np.arange(max_goals + 1)
    fact = _fact(k)
    base = np.exp(-(lam1 + lam2 + lam3))
    px = lam1 ** k / fact          # X marginal-ish term
    py = lam2 ** k / fact
    mat = np.zeros((max_goals + 1, max_goals + 1))
    ratio = lam3 / (lam1 * lam2) if lam1 > 0 and lam2 > 0 else 0.0
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            s = 0.0
            for kk in range(min(x, y) + 1):
                s += (_choose(x, kk) * _choose(y, kk) * fact[kk] * ratio ** kk)
            mat[x, y] = base * px[x] * py[y] * s
    return mat / mat.sum()


def _choose(n: int, r: int) -> float:
    if r < 0 or r > n:
        return 0.0
    return _fact(np.array(n))[()] / (_fact(np.array(r))[()] * _fact(np.array(n - r))[()])


class BivariatePoissonModel(DixonColesModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lambda3_ = 0.0

    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None):
        super().fit(matches, as_of)        # attack/defense/home/intercept (+ DC rho)
        self._fit_lambda3(matches)
        return self

    def _fit_lambda3(self, matches: pd.DataFrame) -> None:
        neutral = matches["neutral"].astype(bool).to_numpy()
        m_h, m_a = self._lambdas(matches["home_team"], matches["away_team"], neutral)
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        # only low-score pairs contribute meaningfully and keep it fast
        cap = 6
        mask = (hs <= cap) & (as_ <= cap)
        m_h, m_a, hs, as_ = m_h[mask], m_a[mask], hs[mask], as_[mask]

        def nll(l3: float) -> float:
            l1 = np.clip(m_h - l3, _MIN, None)
            l2 = np.clip(m_a - l3, _MIN, None)
            base = -(l1 + l2 + l3)
            logp = base + hs * np.log(l1) - _logfact(hs) + as_ * np.log(l2) - _logfact(as_)
            # shared-component correction term (k summation), vectorised lightly
            ratio = l3 / (l1 * l2)
            corr = np.ones_like(l1)
            both = np.minimum(hs, as_)
            for i in range(len(l1)):
                s = 0.0
                for kk in range(int(both[i]) + 1):
                    s += (_choose(int(hs[i]), kk) * _choose(int(as_[i]), kk)
                          * _fact(np.array(kk))[()] * ratio[i] ** kk)
                corr[i] = s
            return -np.sum(logp + np.log(np.clip(corr, _MIN, None)))

        upper = max(0.01, float(min(m_h.min(), m_a.min()) * 0.5))
        res = minimize_scalar(nll, bounds=(0.0, min(upper, 0.4)), method="bounded")
        self.lambda3_ = float(res.x)

    def scoreline_matrix(self, lam: float, mu: float) -> np.ndarray:
        l3 = min(self.lambda3_, lam - _MIN, mu - _MIN)
        l3 = max(l3, 0.0)
        return _bp_matrix(max(lam - l3, _MIN), max(mu - l3, _MIN), l3, self.max_goals)


def _logfact(k: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln
    return gammaln(k + 1)
