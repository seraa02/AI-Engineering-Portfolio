"""
Teacher pipeline: Claude generates labeled entity extraction examples
from SEC 10-K filing excerpts.

The teacher uses claude-sonnet-4-6 to produce high-quality JSON-structured
entity labels. These become the training signal for the student model.

Cost tracking: claude-sonnet-4-6 pricing at $3/$15 per MTok in/out.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

import anthropic

from src.teacher.schema import Entity, EntityType, ExtractionExample


# claude-sonnet-4-6 pricing (per token)
COST_PER_INPUT_TOKEN = 3.0 / 1_000_000
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000

TEACHER_MODEL = os.getenv("TEACHER_MODEL", "claude-sonnet-4-6")

_SYSTEM_PROMPT = """\
You are an expert at extracting named entities from SEC 10-K financial filings.

Given a text excerpt, identify all named entities of these types:
- COMPANY: Organizations, corporations, subsidiaries, joint ventures
- PRODUCT: Products, services, platforms, technologies
- PERSON: Named individuals (executives, board members, etc.)
- REGULATION: Laws, regulations, standards (e.g., GAAP, SOX, Dodd-Frank)
- METRIC: Financial metrics with values (e.g., "$5.2 billion revenue", "15% growth")
- LOCATION: Geographic locations (countries, cities, regions)

For each entity, provide:
- text: the exact surface form as it appears in the text
- entity_type: one of the types above
- start: character offset of the entity start
- end: character offset of the entity end (exclusive)

Return ONLY a JSON object with this structure:
{
  "entities": [
    {"text": "NVIDIA", "entity_type": "COMPANY", "start": 0, "end": 6},
    ...
  ]
}
"""


def _count_tokens_approx(text: str) -> int:
    """Approximate token count: ~4 chars per token."""
    return len(text) // 4


def generate_example(
    text: str,
    client: Optional[anthropic.Anthropic] = None,
    example_id: Optional[str] = None,
    difficulty: str = "medium",
) -> ExtractionExample:
    """
    Generate one labeled example using the teacher model.
    Returns an ExtractionExample with entities labeled by Claude.
    """
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    if example_id is None:
        example_id = str(uuid.uuid4())[:8]

    try:
        message = client.messages.create(
            model=TEACHER_MODEL,
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Extract entities from:\n\n{text}"}],
        )

        input_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)

        raw = message.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        entities = []
        for e in data.get("entities", []):
            try:
                entities.append(Entity(
                    text=e["text"],
                    entity_type=e["entity_type"],
                    start=e.get("start", 0),
                    end=e.get("end", len(e["text"])),
                    confidence=e.get("confidence", 1.0),
                ))
            except (KeyError, TypeError):
                continue

        return ExtractionExample(
            id=example_id,
            text=text,
            entities=entities,
            source="teacher",
            difficulty=difficulty,
            token_count=input_tokens + output_tokens,
            teacher_cost_usd=cost,
        )

    except Exception as exc:
        # Return empty example on error — will be filtered during validation
        return ExtractionExample(
            id=example_id,
            text=text,
            entities=[],
            source="teacher_error",
            difficulty=difficulty,
        )


def generate_batch(
    texts: list[str],
    client: Optional[anthropic.Anthropic] = None,
    delay_between_calls: float = 0.5,
) -> list[ExtractionExample]:
    """Generate labeled examples for a batch of texts."""
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

    examples = []
    for i, text in enumerate(texts):
        difficulty = "easy" if len(text) < 200 else ("hard" if len(text) > 600 else "medium")
        example = generate_example(text, client, example_id=f"ex_{i:04d}", difficulty=difficulty)
        examples.append(example)
        if i < len(texts) - 1:
            time.sleep(delay_between_calls)
    return examples
