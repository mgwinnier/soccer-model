"""Tests for the proper scoring rules."""
import numpy as np

from src.backtest.metrics import (
    ranked_probability_score, log_loss_score, brier_score, accuracy,
)


def test_rps_perfect_is_zero():
    probs = np.array([[1.0, 0.0, 0.0]])
    labels = np.array([0])  # home win
    assert ranked_probability_score(probs, labels) == 0.0


def test_rps_worst_case_is_one():
    # predict certain away win, home actually wins -> maximal ordinal error
    probs = np.array([[0.0, 0.0, 1.0]])
    labels = np.array([0])
    assert ranked_probability_score(probs, labels) == 1.0


def test_rps_orders_by_distance():
    # a draw prediction is "closer" to a home win than an away-win prediction
    home = np.array([0])
    near = ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), home)
    far = ranked_probability_score(np.array([[0.0, 0.0, 1.0]]), home)
    assert near < far


def test_log_loss_and_brier_perfect():
    probs = np.array([[1.0, 0.0, 0.0]])
    labels = np.array([0])
    assert log_loss_score(probs, labels) < 1e-9
    assert brier_score(probs, labels) == 0.0


def test_accuracy():
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    labels = np.array([0, 2])
    assert accuracy(probs, labels) == 1.0
