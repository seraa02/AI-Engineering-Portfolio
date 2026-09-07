"""Tests for calibration statistics."""
import pytest
from src.calibration import (
    weighted_cohens_kappa,
    spearman_correlation,
    compute_agreement,
    top_disagreements,
)


def test_kappa_perfect_agreement():
    ratings = [1, 2, 3, 4, 5]
    kappa = weighted_cohens_kappa(ratings, ratings, scale=5)
    assert abs(kappa - 1.0) < 0.01


def test_kappa_maximum_disagreement():
    # Perfect systematic disagreement: kappa = 0 with linear weighting
    # because Po=0 and Pe=0 (marginals are both at extremes → no overlap in E)
    a = [1, 1, 1, 1, 1]
    b = [5, 5, 5, 5, 5]
    kappa = weighted_cohens_kappa(a, b, scale=5)
    assert kappa <= 0  # No agreement (0 or negative)


def test_kappa_moderate_agreement():
    a = [3, 4, 3, 4, 5, 2, 3, 4, 5, 3]
    b = [3, 4, 3, 3, 5, 2, 3, 4, 4, 3]  # mostly agree, 2 differ by 1
    kappa = weighted_cohens_kappa(a, b, scale=5)
    assert kappa > 0.5


def test_spearman_perfect():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [1.0, 2.0, 3.0, 4.0, 5.0]
    rho = spearman_correlation(xs, ys)
    assert abs(rho - 1.0) < 0.01


def test_spearman_negative():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 4.0, 3.0, 2.0, 1.0]
    rho = spearman_correlation(xs, ys)
    assert abs(rho - (-1.0)) < 0.01


def test_spearman_no_correlation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 1.0, 5.0, 2.0, 4.0]
    rho = spearman_correlation(xs, ys)
    assert abs(rho) < 0.5


def test_compute_agreement():
    a = [4, 4, 5, 3, 4, 4, 5, 3, 4, 4]
    b = [4, 3, 5, 3, 4, 4, 4, 3, 4, 5]
    result = compute_agreement(a, b, scale=5)
    assert result.n == 10
    assert result.weighted_kappa > 0
    assert result.spearman_rho > 0
    assert result.n_disagreements >= 0


def test_top_disagreements():
    ids = ["ex_001", "ex_002", "ex_003"]
    a = [5, 1, 3]
    b = [1, 5, 3]  # ex_001 and ex_002 have large disagreements
    top = top_disagreements(ids, a, b, top_n=2)
    assert len(top) == 2
    assert top[0]["diff"] == 4
    assert top[0]["id"] in ["ex_001", "ex_002"]
