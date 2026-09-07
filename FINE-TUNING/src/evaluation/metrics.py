"""
Evaluation metrics for NER distillation.

Computes span-level precision, recall, and F1 for entity extraction.
An entity prediction is correct if text and entity_type both match
(span-exact match, case-insensitive on text).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.teacher.schema import Entity, ExtractionExample


@dataclass
class EvalMetrics:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int          # Total gold entities

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> "EvalMetrics":
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return cls(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            support=tp + fn,
        )


def _entity_key(entity: Entity) -> tuple[str, str]:
    """Canonical key for entity matching: (normalized_text, entity_type)."""
    return (entity.text.lower().strip(), entity.entity_type.upper())


def score_example(
    gold_entities: list[Entity],
    pred_entities: list[Entity],
) -> tuple[int, int, int]:
    """
    Compute (TP, FP, FN) for a single example using exact-match on (text, type).
    """
    gold_keys = [_entity_key(e) for e in gold_entities]
    pred_keys = [_entity_key(e) for e in pred_entities]

    # Allow duplicate entities by using multisets
    from collections import Counter
    gold_counts = Counter(gold_keys)
    pred_counts = Counter(pred_keys)

    tp = sum((gold_counts & pred_counts).values())
    fp = sum(pred_counts.values()) - tp
    fn = sum(gold_counts.values()) - tp

    return tp, fp, fn


def evaluate(
    gold_examples: list[ExtractionExample],
    pred_examples: list[ExtractionExample],
) -> EvalMetrics:
    """
    Evaluate predicted examples against gold examples.

    gold_examples and pred_examples must be aligned (same IDs in same order).
    """
    total_tp = total_fp = total_fn = 0

    gold_by_id = {ex.id: ex for ex in gold_examples}
    pred_by_id = {ex.id: ex for ex in pred_examples}

    for ex_id, gold_ex in gold_by_id.items():
        pred_ex = pred_by_id.get(ex_id)
        pred_entities = pred_ex.entities if pred_ex else []
        tp, fp, fn = score_example(gold_ex.entities, pred_entities)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return EvalMetrics.from_counts(total_tp, total_fp, total_fn)


def evaluate_by_type(
    gold_examples: list[ExtractionExample],
    pred_examples: list[ExtractionExample],
) -> dict[str, EvalMetrics]:
    """Return per-entity-type metrics."""
    from collections import defaultdict

    type_tp: dict[str, int] = defaultdict(int)
    type_fp: dict[str, int] = defaultdict(int)
    type_fn: dict[str, int] = defaultdict(int)

    gold_by_id = {ex.id: ex for ex in gold_examples}
    pred_by_id = {ex.id: ex for ex in pred_examples}

    for ex_id, gold_ex in gold_by_id.items():
        pred_ex = pred_by_id.get(ex_id)
        pred_entities = pred_ex.entities if pred_ex else []

        from collections import Counter
        gold_by_type = Counter(_entity_key(e) for e in gold_ex.entities)
        pred_by_type = Counter(_entity_key(e) for e in pred_entities)

        # Build per-type gold/pred
        all_types = {k[1] for k in set(gold_by_type) | set(pred_by_type)}
        for etype in all_types:
            g = Counter({k: v for k, v in gold_by_type.items() if k[1] == etype})
            p = Counter({k: v for k, v in pred_by_type.items() if k[1] == etype})
            tp = sum((g & p).values())
            fp = sum(p.values()) - tp
            fn = sum(g.values()) - tp
            type_tp[etype] += tp
            type_fp[etype] += fp
            type_fn[etype] += fn

    return {
        etype: EvalMetrics.from_counts(type_tp[etype], type_fp[etype], type_fn[etype])
        for etype in set(type_tp) | set(type_fp) | set(type_fn)
    }
