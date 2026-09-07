"""
Dataset loading and generation helpers.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "examples.json"
_LABELS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "human_labels.json"


def load_examples() -> list[dict]:
    return json.loads(_DATASET_PATH.read_text())


def load_human_labels() -> list[dict]:
    return json.loads(_LABELS_PATH.read_text())


def generate_dataset(n: int = 200, seed: int = 42) -> list[dict]:
    """
    Generate a realistic synthetic evaluation dataset.
    Categories: easy (25%), medium (40%), hard (20%), adversarial (10%), broken (5%)
    """
    random.seed(seed)

    companies = ["NVIDIA", "AMD", "Intel", "Microsoft", "Amazon", "Alphabet/Google"]
    topics = ["competitors", "acquisitions", "subsidiaries", "products", "segments", "regulations"]

    question_templates = {
        "easy": [
            "Who does {company} compete with in the {segment} segment?",
            "What products does {company} offer in its {segment} business?",
            "What regulatory bodies oversee {company}?",
        ],
        "medium": [
            "How does {company}'s {segment} segment compare to its competitors?",
            "What are the main risk factors for {company} in {year}?",
            "Describe {company}'s supply chain dependencies.",
        ],
        "hard": [
            "Trace the competitive relationship between {company} and {company2} across all segments.",
            "Compare {company}'s acquisition strategy to {company2}'s over the past 5 years.",
            "What would happen to {company} if {company2} exited the market?",
        ],
        "adversarial": [
            "Confirm that {company} is a wholly-owned subsidiary of {company2}.",
            "Why did {company} acquire {company2} last year?",
        ],
        "deliberately_broken": [
            "Tell me about {company}'s revenue.",
        ],
    }

    answer_templates = {
        "easy": {
            "good": "{company} competes primarily with {company2} and {company3} in the {segment} segment, as disclosed in Item 1 of the 10-K filing. [chunk_{ticker}_001]",
            "medium_quality": "{company} faces competition in {segment}. There are several competitors. [chunk_{ticker}_001]",
        },
        "medium": {
            "good": "The {segment} segment generated substantial revenue, with key competitors including {company2}. The filing notes risk factors including supply chain concentration. [chunk_{ticker}_045]",
            "poor": "I don't have that information.",
        },
        "hard": {
            "good": "{company} and {company2} compete across multiple segments including data center and client computing. The filing details their competitive positioning across three business areas. [chunk_{ticker}_021] [chunk_{ticker2}_033]",
            "hallucinated": "{company} recently acquired {company2} for $50 billion in a landmark deal that reshaped the semiconductor industry.",
        },
        "adversarial": {
            "correct_refusal": "The retrieved evidence does not support this claim. There is no statement in the filings that {company} is a subsidiary of {company2}.",
            "wrong_answer": "Yes, {company} is indeed a subsidiary of {company2} as mentioned in the filing.",
        },
        "deliberately_broken": {
            "no_citation": "{company} had revenue of $X billion last year with strong growth in all segments.",
            "wrong_citation": "{company} revenue grew 25% [chunk_WRONG_999]",
        },
    }

    examples = []
    id_counter = 1

    category_counts = {
        "easy": 50,
        "medium": 80,
        "hard": 40,
        "adversarial": 20,
        "deliberately_broken": 10,
    }

    segment_choices = ["Data Center", "Client", "Gaming", "Cloud", "AI", "Networking"]
    ticker_map = {
        "NVIDIA": "NVDA", "AMD": "AMD", "Intel": "INTC",
        "Microsoft": "MSFT", "Amazon": "AMZN", "Alphabet/Google": "GOOGL"
    }

    for category, count in category_counts.items():
        for i in range(count):
            company = random.choice(companies)
            company2 = random.choice([c for c in companies if c != company])
            company3 = random.choice([c for c in companies if c != company and c != company2])
            segment = random.choice(segment_choices)
            ticker = ticker_map[company]
            ticker2 = ticker_map[company2]
            year = random.choice(["2024", "2025", "2026"])

            q_templates = question_templates.get(category, question_templates["easy"])
            question = random.choice(q_templates).format(
                company=company, company2=company2, segment=segment, year=year
            )

            # Pick answer quality based on category
            if category == "easy":
                if random.random() < 0.7:
                    answer_key = "good"
                    answer = answer_templates["easy"]["good"].format(
                        company=company, company2=company2, company3=company3,
                        segment=segment, ticker=ticker
                    )
                    expected_scores = {"factual_accuracy": 5, "citation_quality": 5, "completeness": 5, "coherence": 3, "hallucination_avoidance": 3}
                else:
                    answer = answer_templates["easy"]["medium_quality"].format(
                        company=company, segment=segment, ticker=ticker
                    )
                    expected_scores = {"factual_accuracy": 3, "citation_quality": 2, "completeness": 3, "coherence": 2, "hallucination_avoidance": 3}
                has_citations = "[chunk_" in answer

            elif category == "medium":
                if random.random() < 0.6:
                    answer = answer_templates["medium"]["good"].format(
                        company=company, company2=company2, segment=segment, ticker=ticker
                    )
                    expected_scores = {"factual_accuracy": 4, "citation_quality": 4, "completeness": 4, "coherence": 3, "hallucination_avoidance": 3}
                else:
                    answer = answer_templates["medium"]["poor"].format(company=company)
                    expected_scores = {"factual_accuracy": 2, "citation_quality": 1, "completeness": 2, "coherence": 2, "hallucination_avoidance": 3}
                has_citations = "[chunk_" in answer

            elif category == "hard":
                if random.random() < 0.5:
                    answer = answer_templates["hard"]["good"].format(
                        company=company, company2=company2, ticker=ticker, ticker2=ticker2
                    )
                    expected_scores = {"factual_accuracy": 4, "citation_quality": 4, "completeness": 3, "coherence": 3, "hallucination_avoidance": 3}
                else:
                    answer = answer_templates["hard"]["hallucinated"].format(
                        company=company, company2=company2
                    )
                    expected_scores = {"factual_accuracy": 1, "citation_quality": 1, "completeness": 2, "coherence": 3, "hallucination_avoidance": 1}
                has_citations = "[chunk_" in answer

            elif category == "adversarial":
                if random.random() < 0.6:
                    answer = answer_templates["adversarial"]["correct_refusal"].format(
                        company=company, company2=company2
                    )
                    expected_scores = {"factual_accuracy": 5, "citation_quality": 5, "completeness": 5, "coherence": 3, "hallucination_avoidance": 3}
                else:
                    answer = answer_templates["adversarial"]["wrong_answer"].format(
                        company=company, company2=company2
                    )
                    expected_scores = {"factual_accuracy": 1, "citation_quality": 2, "completeness": 3, "coherence": 3, "hallucination_avoidance": 1}
                has_citations = False

            else:  # deliberately_broken
                if random.random() < 0.5:
                    answer = answer_templates["deliberately_broken"]["no_citation"].format(company=company)
                    expected_scores = {"factual_accuracy": 2, "citation_quality": 1, "completeness": 2, "coherence": 3, "hallucination_avoidance": 2}
                else:
                    answer = answer_templates["deliberately_broken"]["wrong_citation"].format(company=company)
                    expected_scores = {"factual_accuracy": 2, "citation_quality": 1, "completeness": 2, "coherence": 2, "hallucination_avoidance": 2}
                has_citations = False

            examples.append({
                "id": f"ex_{id_counter:03d}",
                "input": question,
                "model_output": answer,
                "category": category,
                "company": company,
                "expected_scores": expected_scores,
                "has_citations": has_citations,
                "output_length_tokens": int(len(answer.split()) / 0.75),
            })
            id_counter += 1

    random.shuffle(examples)
    # Reassign IDs after shuffle
    for i, ex in enumerate(examples, 1):
        ex["id"] = f"ex_{i:03d}"

    return examples


def generate_human_labels(examples: list[dict], seed: int = 42) -> list[dict]:
    """
    Generate simulated human labels for 200 examples,
    plus 20 double-labels for self-agreement measurement.

    Labels are realistic (not all 5s):
    - Based on expected_scores with ±1 noise
    - Inter-session variance built in (double-labels differ by ±0-2)
    """
    random.seed(seed)
    labels = []
    double_label_ids = {ex["id"] for ex in examples[:20]}  # first 20 get double-labeled

    criterion_scales = {
        "factual_accuracy": 5,
        "citation_quality": 5,
        "completeness": 5,
        "coherence": 3,
        "hallucination_avoidance": 3,
    }

    def noisy_score(base_score: int, scale: int, noise_level: int = 1) -> int:
        delta = random.randint(-noise_level, noise_level)
        return max(1, min(scale, base_score + delta))

    for ex in examples:
        expected = ex.get("expected_scores", {})
        scores = {
            criterion: noisy_score(expected.get(criterion, 3), scale, noise_level=1)
            for criterion, scale in criterion_scales.items()
        }
        labels.append({
            "example_id": ex["id"],
            "labeler_id": "labeler_001",
            "session": "session_1",
            "scores": scores,
            "notes": "",
            "timestamp": "2026-08-01T10:00:00Z",
        })

        # Add double-label with more noise (different session = more variance)
        if ex["id"] in double_label_ids:
            scores_2 = {
                criterion: noisy_score(expected.get(criterion, 3), scale, noise_level=2)
                for criterion, scale in criterion_scales.items()
            }
            labels.append({
                "example_id": ex["id"],
                "labeler_id": "labeler_001",
                "session": "session_2",
                "scores": scores_2,
                "notes": "Second labeling session",
                "timestamp": "2026-08-08T10:00:00Z",
            })

    return labels
