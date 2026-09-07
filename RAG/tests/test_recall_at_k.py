"""
Tests for eval/recall_at_k.py's recall@k aggregation.

What these tests prove:
    - recall@k counts a hit whenever the gold chunk's rank is <= k, so a
      chunk ranked 2nd counts toward recall@3/@5/@10 but not recall@1.
    - A label whose gold chunk was never retrieved (rank=None) counts as
      a miss at every k, not an error.
    - An empty label set returns 0.0 rather than raising ZeroDivisionError.
"""

from eval.recall_at_k import _recall_at_k


def test_recall_counts_hit_at_and_above_its_rank():
    # One label found at rank 2: misses recall@1, hits recall@3/@5.
    result = _recall_at_k([2], k_values=[1, 3, 5])
    assert result == {"recall@1": 0.0, "recall@3": 1.0, "recall@5": 1.0}


def test_recall_treats_none_rank_as_a_miss_at_every_k():
    result = _recall_at_k([None], k_values=[1, 5, 10])
    assert result == {"recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0}


def test_recall_averages_across_multiple_labels():
    # rank 1 (hit at k=1), rank 4 (miss at k=1, hit at k=5), None (always miss).
    result = _recall_at_k([1, 4, None], k_values=[1, 5])
    assert result["recall@1"] == 1 / 3
    assert result["recall@5"] == 2 / 3


def test_recall_returns_zero_for_empty_label_set():
    assert _recall_at_k([], k_values=[1, 5]) == {"recall@1": 0.0, "recall@5": 0.0}
