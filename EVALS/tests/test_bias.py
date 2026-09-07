"""Tests for bias measurements."""
import pytest
from src.bias import (
    measure_position_bias,
    measure_length_bias,
    measure_self_preference_bias,
)


def test_position_bias_no_flip():
    """When judge is perfectly consistent, flip_rate=0."""
    def consistent_judge(question, a, b):
        return "A"  # Always picks first

    pairs = [
        {"question": "Q", "answer_a": "Good answer with citation.", "answer_b": "Bad answer."},
    ] * 10

    result = measure_position_bias(consistent_judge, pairs)
    assert result.n_pairs == 10
    # When judge always says "A", BA order maps "A" to original B,
    # so AB winner=a vs BA winner=b → 100% flip rate for position-biased judge
    assert result.flip_rate == 1.0
    assert result.concern is True


def test_position_bias_consistent_by_content():
    """A judge that always picks the longer answer flips consistently."""
    def length_judge(question, a, b):
        return "A" if len(a) >= len(b) else "B"

    pairs = [
        {"question": "Q", "answer_a": "Short.", "answer_b": "This is a much longer and more detailed answer."},
    ] * 10

    result = measure_position_bias(length_judge, pairs)
    assert result.flip_rate == 0.0  # Same relative order regardless of position
    assert result.concern is False


def test_position_bias_empty():
    result = measure_position_bias(lambda q, a, b: "A", [])
    assert result.flip_rate == 0.0
    assert result.n_pairs == 0


def test_length_bias_positive_correlation():
    """Judge that rewards length shows positive length bias."""
    def length_biased_score(question, answer):
        return min(5.0, len(answer.split()) * 0.5)

    examples = [
        {"question": "Q", "answer": "Short answer."},
        {"question": "Q", "answer": "A slightly longer answer with more words."},
        {"question": "Q", "answer": "An even longer and more detailed answer with many more words to fill space."},
        {"question": "Q", "answer": "Extremely verbose answer " * 5},
    ]

    result = measure_length_bias(length_biased_score, examples)
    assert result.spearman_rho > 0.5
    assert result.concern is True


def test_length_bias_no_correlation():
    """Judge that gives constant scores shows no length bias."""
    def constant_score(question, answer):
        return 3.0

    examples = [
        {"question": "Q", "answer": "Short."},
        {"question": "Q", "answer": "Medium length answer here."},
        {"question": "Q", "answer": "Much longer answer that goes on for quite a while."},
    ]

    result = measure_length_bias(constant_score, examples)
    assert abs(result.spearman_rho) < 0.01
    assert result.concern is False


def test_self_preference_bias_detected():
    """Judge that scores native outputs higher shows self-preference bias."""
    def biased_score(question, answer):
        return 4.5 if "[claude]" in answer else 3.0

    native = [{"question": "Q", "answer": "Answer [claude] style"}] * 10
    other = [{"question": "Q", "answer": "Answer [gpt] style"}] * 10

    result = measure_self_preference_bias(biased_score, native, other)
    assert result.delta > 0.3
    assert result.concern is True


def test_self_preference_bias_not_detected():
    """Unbiased judge shows no self-preference."""
    def fair_score(question, answer):
        return 3.5  # same for all

    native = [{"question": "Q", "answer": "Claude answer"}] * 5
    other = [{"question": "Q", "answer": "GPT answer"}] * 5

    result = measure_self_preference_bias(fair_score, native, other)
    assert abs(result.delta) < 0.001
    assert result.concern is False
