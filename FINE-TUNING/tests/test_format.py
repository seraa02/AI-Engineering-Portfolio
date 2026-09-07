"""Tests for the student prompt formatter."""
import json
import pytest
from src.teacher.schema import Entity, ExtractionExample
from src.student.format import (
    INSTRUCTION,
    build_dataset_records,
    format_as_prompt,
    format_example,
    format_for_training,
)


def make_example() -> ExtractionExample:
    return ExtractionExample(
        id="ex_001",
        text="NVIDIA reported $5B revenue.",
        entities=[
            Entity(text="NVIDIA", entity_type="COMPANY", start=0, end=6),
            Entity(text="$5B", entity_type="METRIC", start=16, end=19),
        ],
    )


class TestFormatExample:
    def test_has_required_keys(self):
        ex = make_example()
        d = format_example(ex)
        assert "instruction" in d
        assert "input" in d
        assert "output" in d
        assert "id" in d

    def test_instruction_consistent(self):
        ex = make_example()
        d = format_example(ex)
        assert d["instruction"] == INSTRUCTION

    def test_input_is_text(self):
        ex = make_example()
        d = format_example(ex)
        assert d["input"] == ex.text

    def test_output_is_valid_json(self):
        ex = make_example()
        d = format_example(ex)
        parsed = json.loads(d["output"])
        assert "entities" in parsed
        assert len(parsed["entities"]) == 2

    def test_output_contains_entities(self):
        ex = make_example()
        d = format_example(ex)
        parsed = json.loads(d["output"])
        texts = {e["text"] for e in parsed["entities"]}
        assert "NVIDIA" in texts
        assert "$5B" in texts

    def test_id_preserved(self):
        ex = make_example()
        d = format_example(ex)
        assert d["id"] == "ex_001"


class TestFormatAsPrompt:
    def test_contains_instruction_section(self):
        prompt = format_as_prompt("NVIDIA reported revenue.")
        assert "### Instruction:" in prompt

    def test_contains_input_section(self):
        text = "NVIDIA reported revenue."
        prompt = format_as_prompt(text)
        assert "### Input:" in prompt
        assert text in prompt

    def test_ends_with_response_header(self):
        prompt = format_as_prompt("text")
        assert prompt.endswith("### Response:\n")


class TestFormatForTraining:
    def test_contains_response(self):
        ex = make_example()
        full = format_for_training(ex)
        assert "### Response:" in full
        # Should contain the JSON output
        assert '"entities"' in full

    def test_contains_input_text(self):
        ex = make_example()
        full = format_for_training(ex)
        assert ex.text in full


class TestBuildDatasetRecords:
    def test_produces_correct_count(self):
        examples = [make_example() for _ in range(5)]
        # Give each a unique id
        for i, ex in enumerate(examples):
            ex.id = f"ex_{i:03d}"
        records = build_dataset_records(examples)
        assert len(records) == 5

    def test_all_records_have_required_fields(self):
        examples = [make_example()]
        records = build_dataset_records(examples)
        assert all("instruction" in r and "input" in r and "output" in r for r in records)
