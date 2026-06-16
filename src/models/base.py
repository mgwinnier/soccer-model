"""Shared model interface and outcome utilities.

All match models expose ``predict_proba(fixtures) -> ndarray[n, 3]`` with columns
ordered ``[P(home win), P(draw), P(away win)]`` == ``OUTCOMES``. Keeping a single
canonical ordering everywhere (models, metrics, ensemble, simulator) avoids a
whole class of silent index-swap bugs.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]
OUTCOME_INDEX = {o: i for i, o in enumerate(OUTCOMES)}


def results_to_labels(results: pd.Series) -> np.ndarray:
    """Map 'H'/'D'/'A' strings to integer class indices 0/1/2."""
    return results.map(OUTCOME_INDEX).to_numpy()


def scoreline_to_outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """Collapse a (max_goals+1, max_goals+1) home/away score matrix to H/D/A."""
    home = np.tril(matrix, -1).sum()   # home_goals > away_goals
    draw = np.trace(matrix)
    away = np.triu(matrix, 1).sum()    # away_goals > home_goals
    total = home + draw + away
    return home / total, draw / total, away / total


class MatchModel(Protocol):
    """Structural type implemented by every base model."""

    def predict_proba(self, fixtures: pd.DataFrame) -> np.ndarray:
        """Return an (n, 3) array of [P(H), P(D), P(A)] for each fixture row."""
        ...
