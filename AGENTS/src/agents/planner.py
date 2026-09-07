"""
Planner Agent — receives a research question and creates structured sub-questions.

Uses claude-haiku-4-5 (cheap, fast) via the Anthropic SDK directly.
Returns a PlannerOutput with typed SubQuestion list.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

import anthropic

from src.schemas import (
    PlannerInput,
    PlannerOutput,
    SubQuestion,
    TraceEntry,
    utc_now,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a research planning assistant. Your job is to break down a complex research question into specific, answerable sub-questions.

Rules:
1. Each sub-question must be specific and searchable
2. Sub-questions should cover different aspects of the main question
3. Do NOT repeat similar sub-questions
4. Output ONLY valid JSON, no prose

Output format (JSON):
{
    "reasoning": "brief explanation of your decomposition strategy",
    "subquestions": [
        {"id": "sq_1", "text": "specific sub-question text"},
        {"id": "sq_2", "text": "specific sub-question text"}
    ]
}
"""


def run_planner(
    planner_input: PlannerInput,
    anthropic_client: anthropic.Anthropic,
    model: str = "claude-haiku-4-5",
) -> tuple[PlannerOutput, TraceEntry]:
    """
    Run the Planner agent.
    
    Args:
        planner_input: Typed input with question and max_subquestions
        anthropic_client: Initialized Anthropic client
        model: Model to use (default: claude-haiku-4-5)
    
    Returns:
        Tuple of (PlannerOutput, TraceEntry)
    """
    start_time = time.time()
    timestamp = utc_now()

    user_message = (
        f"Research question: {planner_input.question}\n\n"
        f"Create exactly {planner_input.max_subquestions} sub-questions (or fewer if the topic is narrow). "
        f"Output only the JSON."
    )

    input_dict = planner_input.model_dump()

    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            system=PLANNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        content = response.content[0].text.strip()

        # Parse JSON response
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from response if wrapped in markdown
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse planner response as JSON: {content[:200]}")

        reasoning = parsed.get("reasoning", "")
        raw_subquestions = parsed.get("subquestions", [])

        # Build typed SubQuestion list, enforce max
        subquestions = []
        for i, sq in enumerate(raw_subquestions[: planner_input.max_subquestions]):
            sq_id = sq.get("id", f"sq_{i+1}") or f"sq_{i+1}"
            sq_text = sq.get("text", "").strip()
            if not sq_text:
                continue
            subquestions.append(
                SubQuestion(id=sq_id, text=sq_text, status="pending")
            )

        if not subquestions:
            # Fallback: create one subquestion from the original question
            subquestions = [
                SubQuestion(
                    id="sq_1",
                    text=planner_input.question,
                    status="pending",
                )
            ]

        output = PlannerOutput(subquestions=subquestions, reasoning=reasoning)

        latency_ms = (time.time() - start_time) * 1000
        token_usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        trace_entry = TraceEntry(
            agent="planner",
            input=input_dict,
            output=output.model_dump(),
            tool_calls=[],
            token_usage=token_usage,
            latency_ms=latency_ms,
            status="success",
            timestamp=timestamp,
        )

        logger.info(
            "Planner created %d sub-questions in %.1fms",
            len(subquestions),
            latency_ms,
        )
        return output, trace_entry

    except Exception as exc:
        latency_ms = (time.time() - start_time) * 1000
        logger.error("Planner failed: %s", exc)

        trace_entry = TraceEntry(
            agent="planner",
            input=input_dict,
            output={"error": str(exc)},
            tool_calls=[],
            token_usage={"input_tokens": 0, "output_tokens": 0},
            latency_ms=latency_ms,
            status="error",
            timestamp=timestamp,
        )
        raise
