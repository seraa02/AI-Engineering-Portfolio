"""Tests for the teacher output validator."""
import pytest
from src.teacher.schema import Entity, ExtractionExample
from src.teacher.validator import (
    filter_valid,
    split_dataset,
    validate_entity,
    validate_example,
)


def make_entity(text="NVIDIA", etype="COMPANY", start=0, end=None) -> Entity:
    if end is None:
        end = start + len(text)
    return Entity(text=text, entity_type=etype, start=start, end=end)


def make_example(id="ex_001", text="NVIDIA Corp reported revenue.", entities=None, source="teacher") -> ExtractionExample:
    if entities is None:
        entities = [make_entity()]
    return ExtractionExample(id=id, text=text, entities=entities, source=source)


class TestValidateEntity:
    def test_valid_entity(self):
        text = "NVIDIA reported revenue."
        e = make_entity("NVIDIA", "COMPANY", 0, 6)
        assert validate_entity(e, text)

    def test_unknown_type_invalid(self):
        text = "NVIDIA"
        e = make_entity("NVIDIA", "UNKNOWN_TYPE", 0, 6)
        assert not validate_entity(e, text)

    def test_negative_start_invalid(self):
        text = "NVIDIA"
        e = make_entity("NVIDIA", "COMPANY", -1, 5)
        assert not validate_entity(e, text)

    def test_end_beyond_text_invalid(self):
        text = "NVIDIA"
        e = make_entity("NVIDIA", "COMPANY", 0, 100)
        assert not validate_entity(e, text)

    def test_start_ge_end_invalid(self):
        text = "NVIDIA"
        e = make_entity("NVIDIA", "COMPANY", 5, 5)
        assert not validate_entity(e, text)


class TestValidateExample:
    def test_valid_example(self):
        ex = make_example()
        assert validate_example(ex)

    def test_teacher_error_invalid(self):
        ex = make_example(source="teacher_error")
        assert not validate_example(ex)

    def test_no_entities_invalid(self):
        ex = make_example(entities=[])
        assert not validate_example(ex, min_entities=1)

    def test_empty_text_invalid(self):
        ex = ExtractionExample(id="e", text="   ", entities=[make_entity()])
        assert not validate_example(ex)


class TestFilterValid:
    def test_filters_teacher_errors(self):
        examples = [
            make_example("e1", source="teacher"),
            make_example("e2", source="teacher_error"),
        ]
        valid, reasons = filter_valid(examples)
        assert len(valid) == 1
        assert valid[0].id == "e1"
        assert len(reasons) == 1

    def test_deduplicates_same_text(self):
        text = "NVIDIA Corp reported revenue."
        examples = [
            make_example("e1", text=text),
            make_example("e2", text=text),
        ]
        valid, reasons = filter_valid(examples)
        assert len(valid) == 1
        assert len(reasons) == 1
        assert "duplicate" in reasons[0]

    def test_keeps_different_texts(self):
        examples = [
            make_example("e1", text="NVIDIA Corp reported revenue."),
            make_example("e2", text="AMD GPU accelerator."),
        ]
        valid, _ = filter_valid(examples)
        assert len(valid) == 2

    def test_filters_invalid_offsets(self):
        text = "NVIDIA Corp"
        bad_entity = Entity(text="NVIDIA", entity_type="COMPANY", start=0, end=200)  # beyond text
        ex = ExtractionExample(id="e1", text=text, entities=[bad_entity], source="teacher")
        valid, reasons = filter_valid([ex])
        assert len(valid) == 0  # all entities filtered, below min_entities=1


class TestSplitDataset:
    def test_split_proportions(self):
        examples = [make_example(f"e{i}") for i in range(100)]
        train, val, test = split_dataset(examples, train_ratio=0.8, val_ratio=0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        examples = [make_example(f"e{i}") for i in range(100)]
        train, val, test = split_dataset(examples)
        all_ids = [ex.id for ex in train + val + test]
        assert len(all_ids) == len(set(all_ids))  # no duplicates across splits

    def test_total_preserved(self):
        examples = [make_example(f"e{i}") for i in range(50)]
        train, val, test = split_dataset(examples)
        assert len(train) + len(val) + len(test) == 50
