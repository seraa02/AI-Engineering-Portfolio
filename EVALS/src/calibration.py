"""
Human self-agreement calibration.

Computes weighted Cohen's kappa and Spearman correlation between two sets
of ratings from the same labeler on the same examples (or between two labelers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AgreementResult:
    n: int
    weighted_kappa: float
    spearman_rho: float
    n_disagreements: int          # |score_A - score_B| >= 1
    n_large_disagreements: int    # |score_A - score_B| >= 2
    mean_abs_diff: float


def weighted_cohens_kappa(ratings_a: list[int], ratings_b: list[int], scale: int) -> float:
    """
    Linear-weighted Cohen's kappa.
    weight[i][j] = 1 - |i - j| / (scale - 1)
    """
    assert len(ratings_a) == len(ratings_b), "Rating lists must be same length"
    n = len(ratings_a)
    if n == 0:
        return 0.0

    categories = list(range(1, scale + 1))
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Weight matrix
    W = [[1 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]

    # Observed agreement matrix
    O = [[0.0] * k for _ in range(k)]
    for a, b in zip(ratings_a, ratings_b):
        if a in cat_to_idx and b in cat_to_idx:
            O[cat_to_idx[a]][cat_to_idx[b]] += 1 / n

    # Marginal distributions
    row_marginals = [sum(O[i]) for i in range(k)]
    col_marginals = [sum(O[i][j] for i in range(k)) for j in range(k)]

    # Expected agreement matrix
    E = [[row_marginals[i] * col_marginals[j] for j in range(k)] for i in range(k)]

    # Weighted observed and expected
    Po = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    Pe = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k))

    if Pe == 1.0:
        return 1.0
    return (Po - Pe) / (1 - Pe)


def spearman_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation."""
    n = len(xs)
    if n < 2:
        return 0.0

    def rank(lst: list[float]) -> list[float]:
        indexed = sorted(enumerate(lst), key=lambda x: x[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rank_x = rank(xs)
    rank_y = rank(ys)

    # Pearson correlation on ranks
    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n
    num = sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y))
    denom = math.sqrt(
        sum((rx - mean_rx) ** 2 for rx in rank_x)
        * sum((ry - mean_ry) ** 2 for ry in rank_y)
    )
    return num / denom if denom != 0 else 0.0


def compute_agreement(
    ratings_a: list[int],
    ratings_b: list[int],
    scale: int,
) -> AgreementResult:
    """Compute all agreement statistics between two rating lists."""
    n = len(ratings_a)
    diffs = [abs(a - b) for a, b in zip(ratings_a, ratings_b)]
    return AgreementResult(
        n=n,
        weighted_kappa=weighted_cohens_kappa(ratings_a, ratings_b, scale),
        spearman_rho=spearman_correlation([float(r) for r in ratings_a], [float(r) for r in ratings_b]),
        n_disagreements=sum(1 for d in diffs if d >= 1),
        n_large_disagreements=sum(1 for d in diffs if d >= 2),
        mean_abs_diff=sum(diffs) / n if n > 0 else 0.0,
    )


def top_disagreements(
    example_ids: list[str],
    ratings_a: list[int],
    ratings_b: list[int],
    top_n: int = 20,
) -> list[dict]:
    """Return top_n examples with the largest score disagreements."""
    pairs = [
        {"id": eid, "rating_a": a, "rating_b": b, "diff": abs(a - b)}
        for eid, a, b in zip(example_ids, ratings_a, ratings_b)
    ]
    return sorted(pairs, key=lambda x: -x["diff"])[:top_n]
