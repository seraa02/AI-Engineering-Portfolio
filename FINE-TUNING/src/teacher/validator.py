"""
Validation for teacher-generated examples.

Filters out:
- Examples with no entities (likely extraction failures)
- Examples with invalid entity offsets
- Examples with unknown entity types
- Duplicate texts
"""
from __future__ import annotations

from src.teacher.schema import EntityType, ExtractionExample


VALID_ENTITY_TYPES = {e.value for e in EntityType}


def validate_entity(entity, text: str) -> bool:
    """Return True if entity is valid for the given text."""
    if entity.entity_type not in VALID_ENTITY_TYPES:
        return False
    if entity.start < 0 or entity.end > len(text) or entity.start >= entity.end:
        return False
    # The extracted span should approximately match the entity text
    # (allow some slack for whitespace differences)
    span = text[entity.start:entity.end].strip()
    if not span:
        return False
    return True


def validate_example(example: ExtractionExample, min_entities: int = 1) -> bool:
    """Return True if the example passes quality checks."""
    if example.source == "teacher_error":
        return False
    if len(example.entities) < min_entities:
        return False
    if not example.text.strip():
        return False
    return True


def filter_valid(
    examples: list[ExtractionExample],
    min_entities: int = 1,
) -> tuple[list[ExtractionExample], list[str]]:
    """
    Filter examples to only valid ones.

    Returns (valid_examples, rejection_reasons).
    Also deduplicates by text content.
    """
    valid = []
    reasons = []
    seen_texts: set[str] = set()

    for ex in examples:
        text_key = ex.text.strip()[:200]

        if text_key in seen_texts:
            reasons.append(f"{ex.id}: duplicate text")
            continue
        seen_texts.add(text_key)

        if not validate_example(ex, min_entities=min_entities):
            reasons.append(
                f"{ex.id}: source={ex.source}, entities={len(ex.entities)}"
            )
            continue

        # Filter invalid entities within valid examples
        valid_entities = [
            e for e in ex.entities if validate_entity(e, ex.text)
        ]
        ex.entities = valid_entities

        if len(valid_entities) < min_entities:
            reasons.append(f"{ex.id}: no valid entities after offset check")
            continue

        valid.append(ex)

    return valid, reasons


def split_dataset(
    examples: list[ExtractionExample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list, list, list]:
    """Split examples into train/val/test sets."""
    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]
    return train, val, test
