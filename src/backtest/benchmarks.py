"""Reference baselines a serious model must beat.

* **Climatology** — predict the historical base rate of H/D/A for every match.
  This is the information-free floor; any model worth keeping must beat it.
* **Home/neutral prior** — base rates split by whether the game is neutral,
  capturing pure home advantage with zero team knowledge.

The Elo-only model (an ensemble member) serves as the strong, transparent
benchmark. Bookmaker odds would be the gold standard, but de-vigged *international*
closing odds are not freely/publicly redistributable, so we benchmark against
Elo and climatology and say so plainly rather than fake a market line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.base import OUTCOMES, results_to_labels


class ClimatologyBaseline:
    """Constant H/D/A probabilities from training-set frequencies."""

    def fit(self, train: pd.DataFrame) -> "ClimatologyBaseline":
        counts = train["result"].value_counts(normalize=True)
        self.p_ = np.array([counts.get(o, 0.0) for o in OUTCOMES])
        self.p_ = self.p_ / self.p_.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p_, (len(df), 1))


class HomePriorBaseline:
    """Base rates conditioned only on neutral vs non-neutral venue."""

    def fit(self, train: pd.DataFrame) -> "HomePriorBaseline":
        self.tbl_ = {}
        for neutral, grp in train.groupby(train["neutral"].astype(bool)):
            counts = grp["result"].value_counts(normalize=True)
            p = np.array([counts.get(o, 0.0) for o in OUTCOMES])
            self.tbl_[bool(neutral)] = p / p.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(df), 3))
        neutral = df["neutral"].astype(bool).to_numpy()
        for i, nv in enumerate(neutral):
            out[i] = self.tbl_.get(bool(nv), self.tbl_[False])
        return out
