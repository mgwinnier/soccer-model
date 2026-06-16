"""Dixon-Coles bivariate Poisson goals model with time decay.

Fitting uses the standard fast two-stage approach:

  1. **Attack / defense / home advantage** via weighted Poisson regression.
     Each match becomes two "goals scored" observations (home-attacking and
     away-attacking) with one-hot attack/defense dummies and a home indicator.
     This sub-problem is convex and solves quickly with ridge regularization.
     Sample weights = exp(-xi·age) · match_importance (recent, important games
     count more).
  2. **Low-score correlation rho** (the Dixon-Coles correction) via a 1-D MLE
     holding the attack/defense rates fixed.

The model then yields a full scoreline probability matrix per fixture, from
which W/D/L, exact scores, and over/under all follow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import PoissonRegressor

from ..config import load_config
from .base import scoreline_to_outcome_probs


def _dc_tau(h: np.ndarray, a: np.ndarray, lam: float, mu: float, rho: float) -> np.ndarray:
    """Dixon-Coles low-score correction for score grids h, a."""
    tau = np.ones(np.broadcast(h, a).shape, dtype=float)
    tau = np.where((h == 0) & (a == 0), 1 - lam * mu * rho, tau)
    tau = np.where((h == 0) & (a == 1), 1 + lam * rho, tau)
    tau = np.where((h == 1) & (a == 0), 1 + mu * rho, tau)
    tau = np.where((h == 1) & (a == 1), 1 - rho, tau)
    return tau


@dataclass
class DixonColesModel:
    xi: float = 0.0018
    max_goals: int = 10
    l2_penalty: float = 1e-4

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "DixonColesModel":
        cfg = cfg or load_config()
        dc = cfg["dixon_coles"]
        return cls(xi=dc["xi"], max_goals=dc["max_goals"], l2_penalty=dc["l2_penalty"])

    # --------------------------------------------------------------- targets
    def _targets(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Regression targets (home, away). Base model = actual goals scored."""
        return (matches["home_score"].to_numpy().astype(float),
                matches["away_score"].to_numpy().astype(float))

    # ------------------------------------------------------------------ fit
    def fit(self, matches: pd.DataFrame, as_of: pd.Timestamp | None = None) -> "DixonColesModel":
        as_of = as_of or matches["date"].max()
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams_ = teams
        self._tidx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        age_days = (as_of - matches["date"]).dt.days.to_numpy()
        w = np.exp(-self.xi * age_days) * matches["importance"].to_numpy()

        hi = matches["home_team"].map(self._tidx).to_numpy()
        ai = matches["away_team"].map(self._tidx).to_numpy()
        not_neutral = (~matches["neutral"].astype(bool)).to_numpy().astype(float)
        m = len(matches)
        rows = np.arange(m)

        def onehot(idx):
            return csr_matrix((np.ones(m), (rows, idx)), shape=(m, n))

        # Observation block 1: home team attacking (defender = away), is_home active
        atk_h, def_h = onehot(hi), onehot(ai)
        X_home = hstack([atk_h, def_h, csr_matrix(not_neutral.reshape(-1, 1))]).tocsr()
        # Observation block 2: away team attacking (defender = home), is_home = 0
        atk_a, def_a = onehot(ai), onehot(hi)
        X_away = hstack([atk_a, def_a, csr_matrix(np.zeros((m, 1)))]).tocsr()
        # Targets are pluggable: the goals model uses scores; the xG model blends
        # in expected goals where available (see XgDixonColesModel).
        y_home, y_away = self._targets(matches)

        X = _vstack_sparse(X_home, X_away)
        y = np.concatenate([y_home, y_away])
        sw = np.concatenate([w, w])

        reg = PoissonRegressor(alpha=self.l2_penalty, max_iter=400, fit_intercept=True)
        reg.fit(X, y, sample_weight=sw)
        coef = reg.coef_
        self.intercept_ = float(reg.intercept_)
        self.attack_ = coef[:n]
        self.defense_ = coef[n:2 * n]
        self.home_adv_ = float(coef[2 * n])

        self._fit_rho(matches, w)
        self._mean_attack = float(self.attack_.mean())
        return self

    def _fit_rho(self, matches: pd.DataFrame, w: np.ndarray) -> None:
        lam, mu = self._lambdas(matches["home_team"], matches["away_team"],
                                matches["neutral"].astype(bool).to_numpy())
        hs = matches["home_score"].to_numpy()
        as_ = matches["away_score"].to_numpy()
        low = (hs <= 1) & (as_ <= 1)
        h, a = hs[low].astype(float), as_[low].astype(float)
        lam_l, mu_l, w_l = lam[low], mu[low], w[low]

        def nll(rho: float) -> float:
            tau = _dc_tau(h, a, lam_l, mu_l, rho)
            tau = np.clip(tau, 1e-6, None)
            return -np.sum(w_l * np.log(tau))

        res = minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded")
        self.rho_ = float(res.x)

    # -------------------------------------------------------------- predict
    def _lambdas(self, home: pd.Series, away: pd.Series, neutral: np.ndarray):
        hi = home.map(self._tidx)
        ai = away.map(self._tidx)
        # unseen teams -> league-average (attack/defense 0 contribution)
        def vec(idx, table):
            arr = np.full(len(idx), np.nan)
            known = idx.notna().to_numpy()
            arr[known] = table[idx[idx.notna()].astype(int).to_numpy()]
            arr[~known] = 0.0
            return arr
        atk_h, def_a = vec(hi, self.attack_), vec(ai, self.defense_)
        atk_a, def_h = vec(ai, self.attack_), vec(hi, self.defense_)
        ha = np.where(neutral, 0.0, self.home_adv_)
        lam = np.exp(self.intercept_ + atk_h + def_a + ha)
        mu = np.exp(self.intercept_ + atk_a + def_h)
        return lam, mu

    def scoreline_matrix(self, lam: float, mu: float) -> np.ndarray:
        k = np.arange(self.max_goals + 1)
        ph = np.exp(-lam) * lam ** k / _fact(k)
        pa = np.exp(-mu) * mu ** k / _fact(k)
        mat = np.outer(ph, pa)
        H, A = np.meshgrid(k, k, indexing="ij")
        mat = mat * _dc_tau(H, A, lam, mu, self.rho_)
        return mat / mat.sum()

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        neutral = fixtures["neutral"].astype(bool).to_numpy() if "neutral" in fixtures \
            else np.zeros(len(fixtures), dtype=bool)
        lam, mu = self._lambdas(fixtures["home_team"], fixtures["away_team"], neutral)
        out = np.empty((len(fixtures), 3))
        for i in range(len(fixtures)):
            mat = self.scoreline_matrix(lam[i], mu[i])
            out[i] = scoreline_to_outcome_probs(mat)
        return out

    def expected_goals(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        lam, mu = self._lambdas(pd.Series([home]), pd.Series([away]),
                                np.array([neutral]))
        return float(lam[0]), float(mu[0])


# ----- small numeric helpers -------------------------------------------------
_FACT = np.array([1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
                  39916800, 479001600], dtype=float)


def _fact(k: np.ndarray) -> np.ndarray:
    return _FACT[k]


def _vstack_sparse(a: csr_matrix, b: csr_matrix) -> csr_matrix:
    from scipy.sparse import vstack as _vs
    return _vs([a, b]).tocsr()
