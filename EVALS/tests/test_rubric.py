"""Tests for rubric scoring logic."""
import pytest
from src.rubric import CRITERIA, get_criterion, weighted_avg, criterion_names


def test_criteria_count():
    assert len(CRITERIA) == 5


def test_all_criteria_have_anchors():
    for c in CRITERIA:
        assert len(c.anchors) >= 2
        for score, anchor in c.anchors.items():
            assert 1 <= score <= c.scale


def test_get_criterion_by_name():
    c = get_criterion("factual_accuracy")
    assert c is not None
    assert c.scale == 5


def test_get_criterion_missing():
    assert get_criterion("nonexistent") is None


def test_criterion_names():
    names = criterion_names()
    assert "factual_accuracy" in names
    assert "hallucination_avoidance" in names
    assert len(names) == 5


def test_weighted_avg_perfect_scores():
    scores = {"factual_accuracy": 5, "citation_quality": 5, "completeness": 5,
              "coherence": 3, "hallucination_avoidance": 3}
    avg = weighted_avg(scores)
    assert abs(avg - 1.0) < 0.001


def test_weighted_avg_minimum_scores():
    scores = {"factual_accuracy": 1, "citation_quality": 1, "completeness": 1,
              "coherence": 1, "hallucination_avoidance": 1}
    avg = weighted_avg(scores)
    assert abs(avg - 0.0) < 0.001


def test_weighted_avg_partial():
    scores = {"factual_accuracy": 3}  # midpoint on 5-pt scale
    avg = weighted_avg(scores)
    assert 0.4 < avg < 0.6


def test_weighted_avg_empty():
    assert weighted_avg({}) == 0.0
