"""
Format teacher-labeled examples into the instruction-tuning format
expected by the student model.

We use the Alpaca prompt format which works well for small LMs:

### Instruction:
Extract named entities from the following SEC filing excerpt.
Return a JSON object with an "entities" list.

### Input:
<text>

### Response:
{"entities": [...]}
"""
from __future__ import annotations

import json

from src.teacher.schema import ExtractionExample


INSTRUCTION = (
    "Extract named entities from the following SEC 10-K filing excerpt. "
    "Identify COMPANY, PRODUCT, PERSON, REGULATION, METRIC, and LOCATION entities. "
    "Return a JSON object with an 'entities' list, each entity having: "
    "text, entity_type, start (char offset), end (char offset)."
)


def format_example(example: ExtractionExample) -> dict:
    """
    Convert an ExtractionExample into an instruction-tuning dict.

    Returns:
        {
            "instruction": str,
            "input": str,
            "output": str,    # JSON string of entities
            "id": str,
        }
    """
    output = json.dumps(
        {"entities": [e.to_dict() for e in example.entities]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "instruction": INSTRUCTION,
        "input": example.text,
        "output": output,
        "id": example.id,
        "difficulty": example.difficulty,
    }


def format_as_prompt(text: str) -> str:
    """Format a raw text as a prompt for inference (no output)."""
    return (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{text}\n\n"
        f"### Response:\n"
    )


def format_for_training(example: ExtractionExample) -> str:
    """Full prompt + response string for training (teacher-forcing)."""
    d = format_example(example)
    return (
        f"### Instruction:\n{d['instruction']}\n\n"
        f"### Input:\n{d['input']}\n\n"
        f"### Response:\n{d['output']}"
    )


def build_dataset_records(examples: list[ExtractionExample]) -> list[dict]:
    """Convert a list of examples to JSON-serializable training records."""
    return [format_example(ex) for ex in examples]
