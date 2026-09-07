"""Tests for evaluation metrics."""
import pytest
from src.teacher.schema import Entity, ExtractionExample
from src.evaluation.metrics import (
    EvalMetrics,
    evaluate,
    evaluate_by_type,
    score_example,
)


def make_entity(text: str, etype: str = "COMPANY", start: int = 0) -> Entity:
    return Entity(text=text, entity_type=etype, start=start, end=start + len(text))


def make_example(id: str, entities: list) -> ExtractionExample:
    return ExtractionExample(id=id, text="text", entities=entities)


class TestEvalMetrics:
    def test_perfect_score(self):
        m = EvalMetrics.from_counts(tp=10, fp=0, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_zero_score(self):
        m = EvalMetrics.from_counts(tp=0, fp=5, fn=5)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_precision_only(self):
        m = EvalMetrics.from_counts(tp=5, fp=0, fn=5)
        assert m.precision == 1.0
        assert m.recall == 0.5
        assert m.f1 == pytest.approx(2 * 1.0 * 0.5 / 1.5, rel=1e-4)

    def test_no_predictions(self):
        m = EvalMetrics.from_counts(tp=0, fp=0, fn=5)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_support(self):
        m = EvalMetrics.from_counts(tp=3, fp=2, fn=7)
        assert m.support == 10  # tp + fn


class TestScoreExample:
    def test_perfect_match(self):
        gold = [make_entity("NVIDIA", "COMPANY"), make_entity("AMD", "COMPANY", 10)]
        pred = [make_entity("NVIDIA", "COMPANY"), make_entity("AMD", "COMPANY", 10)]
        tp, fp, fn = score_example(gold, pred)
        assert tp == 2
        assert fp == 0
        assert fn == 0

    def test_all_missed(self):
        gold = [make_entity("NVIDIA"), make_entity("AMD", start=10)]
        pred = []
        tp, fp, fn = score_example(gold, pred)
        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_all_spurious(self):
        gold = []
        pred = [make_entity("NVIDIA"), make_entity("AMD", start=10)]
        tp, fp, fn = score_example(gold, pred)
        assert tp == 0
        assert fp == 2
        assert fn == 0

    def test_partial_match(self):
        gold = [make_entity("NVIDIA"), make_entity("AMD", start=10), make_entity("Intel", start=20)]
        pred = [make_entity("NVIDIA"), make_entity("Microsoft", start=30)]
        tp, fp, fn = score_example(gold, pred)
        assert tp == 1
        assert fp == 1
        assert fn == 2

    def test_case_insensitive(self):
        gold = [make_entity("nvidia", "COMPANY")]
        pred = [make_entity("NVIDIA", "COMPANY")]
        tp, fp, fn = score_example(gold, pred)
        assert tp == 1

    def test_type_must_match(self):
        gold = [make_entity("NVIDIA", "COMPANY")]
        pred = [make_entity("NVIDIA", "PRODUCT")]  # Wrong type
        tp, fp, fn = score_example(gold, pred)
        assert tp == 0
        assert fp == 1
        assert fn == 1


class TestEvaluate:
    def test_perfect_evaluation(self):
        gold = [make_example("e1", [make_entity("NVIDIA")])]
        pred = [make_example("e1", [make_entity("NVIDIA")])]
        metrics = evaluate(gold, pred)
        assert metrics.f1 == 1.0

    def test_missing_prediction(self):
        gold = [make_example("e1", [make_entity("NVIDIA")])]
        pred = []  # No predictions
        metrics = evaluate(gold, pred)
        assert metrics.recall == 0.0

    def test_aggregate_across_examples(self):
        gold = [
            make_example("e1", [make_entity("NVIDIA")]),
            make_example("e2", [make_entity("AMD", start=0)]),
        ]
        pred = [
            make_example("e1", [make_entity("NVIDIA")]),  # TP
            make_example("e2", []),                         # FN
        ]
        metrics = evaluate(gold, pred)
        assert metrics.true_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.recall == 0.5

    def test_support_count(self):
        gold = [
            make_example("e1", [make_entity("NVIDIA"), make_entity("AMD", start=10)]),
        ]
        pred = [make_example("e1", [])]
        metrics = evaluate(gold, pred)
        assert metrics.support == 2


class TestEvaluateByType:
    def test_per_type_breakdown(self):
        gold = [
            make_example("e1", [
                make_entity("NVIDIA", "COMPANY"),
                make_entity("$5B", "METRIC", start=10),
            ])
        ]
        pred = [
            make_example("e1", [
                make_entity("NVIDIA", "COMPANY"),  # TP for COMPANY
                # METRIC missed
            ])
        ]
        by_type = evaluate_by_type(gold, pred)
        assert "COMPANY" in by_type
        assert "METRIC" in by_type
        assert by_type["COMPANY"].f1 == 1.0
        assert by_type["METRIC"].recall == 0.0
