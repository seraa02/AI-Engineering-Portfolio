"""
Generate synthetic teacher-labeled dataset for testing without a live Claude API.

Produces:
  datasets/train.json  — 800 examples
  datasets/val.json    — 100 examples
  datasets/test.json   — 100 examples

Entity offsets are computed deterministically from the text.
"""
from __future__ import annotations

import json
import random
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.teacher.schema import Entity, EntityType, ExtractionExample
from src.student.format import build_dataset_records


# SEC filing excerpts (synthetic, representative of 10-K language)
TEMPLATES = [
    (
        "NVIDIA Corporation reported revenue of $26.0 billion for fiscal year 2024, "
        "driven by strong demand for its H100 GPU in data center applications. "
        "CEO Jensen Huang highlighted the role of CUDA in enabling AI workloads. "
        "The company competes with Advanced Micro Devices and Intel Corporation in the GPU market.",
        [
            ("NVIDIA Corporation", EntityType.COMPANY),
            ("$26.0 billion", EntityType.METRIC),
            ("H100 GPU", EntityType.PRODUCT),
            ("Jensen Huang", EntityType.PERSON),
            ("CUDA", EntityType.PRODUCT),
            ("Advanced Micro Devices", EntityType.COMPANY),
            ("Intel Corporation", EntityType.COMPANY),
        ],
    ),
    (
        "Amazon Web Services generated operating income of $9.4 billion in Q3 2024. "
        "AWS competes with Microsoft Azure and Google Cloud Platform. "
        "The company is subject to oversight by the Federal Trade Commission "
        "and compliance with GDPR in European markets.",
        [
            ("Amazon Web Services", EntityType.COMPANY),
            ("$9.4 billion", EntityType.METRIC),
            ("Microsoft Azure", EntityType.PRODUCT),
            ("Google Cloud Platform", EntityType.PRODUCT),
            ("Federal Trade Commission", EntityType.REGULATION),
            ("GDPR", EntityType.REGULATION),
        ],
    ),
    (
        "Microsoft Corporation's Azure AI services contributed to a 29% increase in cloud revenue. "
        "CFO Amy Hood stated that capital expenditures would increase to support AI infrastructure. "
        "The company is headquartered in Redmond, Washington and operates globally.",
        [
            ("Microsoft Corporation", EntityType.COMPANY),
            ("Azure AI", EntityType.PRODUCT),
            ("29% increase", EntityType.METRIC),
            ("Amy Hood", EntityType.PERSON),
            ("Redmond, Washington", EntityType.LOCATION),
        ],
    ),
    (
        "Alphabet Inc. disclosed risks related to the Digital Markets Act in the European Union. "
        "Google Search and YouTube remain the primary revenue drivers. "
        "The company's DeepMind division continues research into large language models.",
        [
            ("Alphabet Inc.", EntityType.COMPANY),
            ("Digital Markets Act", EntityType.REGULATION),
            ("European Union", EntityType.LOCATION),
            ("Google Search", EntityType.PRODUCT),
            ("YouTube", EntityType.PRODUCT),
            ("DeepMind", EntityType.COMPANY),
        ],
    ),
    (
        "Advanced Micro Devices reported gross margin of 49% for Q4 2024. "
        "The Instinct MI300X accelerator gained significant traction in AI training workloads. "
        "CEO Lisa Su presented the roadmap at the company's financial analyst day in San Jose, California.",
        [
            ("Advanced Micro Devices", EntityType.COMPANY),
            ("49%", EntityType.METRIC),
            ("Instinct MI300X", EntityType.PRODUCT),
            ("Lisa Su", EntityType.PERSON),
            ("San Jose, California", EntityType.LOCATION),
        ],
    ),
    (
        "Intel Corporation is restructuring its foundry business to comply with "
        "the CHIPS and Science Act requirements. The company operates fabrication "
        "facilities in Oregon and Arizona. Pat Gelsinger serves as CEO.",
        [
            ("Intel Corporation", EntityType.COMPANY),
            ("CHIPS and Science Act", EntityType.REGULATION),
            ("Oregon", EntityType.LOCATION),
            ("Arizona", EntityType.LOCATION),
            ("Pat Gelsinger", EntityType.PERSON),
        ],
    ),
    (
        "The company's compliance with Sarbanes-Oxley Section 404 was reviewed by "
        "its auditor PricewaterhouseCoopers. Material weaknesses in internal controls "
        "were identified and remediated during fiscal 2024.",
        [
            ("Sarbanes-Oxley Section 404", EntityType.REGULATION),
            ("PricewaterhouseCoopers", EntityType.COMPANY),
        ],
    ),
    (
        "GAAP net income decreased by $1.2 billion year-over-year due to "
        "increased research and development expenses. Non-GAAP EPS grew 18% "
        "to $4.32 per diluted share.",
        [
            ("GAAP", EntityType.REGULATION),
            ("$1.2 billion", EntityType.METRIC),
            ("Non-GAAP EPS", EntityType.METRIC),
            ("18%", EntityType.METRIC),
            ("$4.32", EntityType.METRIC),
        ],
    ),
]


def _find_offset(text: str, span: str) -> tuple[int, int]:
    """Find the first occurrence of span in text."""
    idx = text.find(span)
    if idx == -1:
        return 0, len(span)
    return idx, idx + len(span)


def make_example(template_idx: int, variant: int = 0) -> ExtractionExample:
    text, entity_specs = TEMPLATES[template_idx % len(TEMPLATES)]

    # Add slight variation by appending variant info
    if variant > 0:
        text = text + f" (Exhibit {variant})"

    entities = []
    for span, etype in entity_specs:
        start, end = _find_offset(text, span)
        entities.append(Entity(
            text=span,
            entity_type=etype.value,
            start=start,
            end=end,
            confidence=1.0,
        ))

    n_entities = len(entities)
    difficulty = "easy" if n_entities <= 3 else ("hard" if n_entities >= 6 else "medium")

    return ExtractionExample(
        id=f"syn_{template_idx:03d}_{variant:04d}",
        text=text,
        entities=entities,
        source="synthetic",
        difficulty=difficulty,
        token_count=len(text) // 4,
        teacher_cost_usd=0.0,
    )


def generate_all(n_train=800, n_val=100, n_test=100) -> tuple[list, list, list]:
    rng = random.Random(42)
    total = n_train + n_val + n_test

    all_examples = []
    for i in range(total):
        tmpl_idx = i % len(TEMPLATES)
        variant = i // len(TEMPLATES)
        all_examples.append(make_example(tmpl_idx, variant))

    rng.shuffle(all_examples)
    train = all_examples[:n_train]
    val = all_examples[n_train:n_train + n_val]
    test = all_examples[n_train + n_val:]
    return train, val, test


if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "datasets"
    out_dir.mkdir(exist_ok=True)

    train, val, test = generate_all()

    for split_name, split in [("train", train), ("val", val), ("test", test)]:
        records = build_dataset_records(split)
        path = out_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"Wrote {len(records)} {split_name} examples to {path}")

    # Also write raw gold for evaluation
    gold_path = out_dir / "test_gold.json"
    with open(gold_path, "w") as f:
        json.dump([ex.to_dict() for ex in test], f, indent=2)
    print(f"Wrote {len(test)} gold test examples to {gold_path}")
    print("Done.")
