#!/usr/bin/env python3
"""
Example script demonstrating the Multi-Agent Research Assistant.

Usage:
    python scripts/run_example.py

Requires:
    - ANTHROPIC_API_KEY and TAVILY_API_KEY set in environment or .env
    - Redis running at REDIS_URL (default: redis://localhost:6379)
"""

import os
import sys
import json
import time
import logging

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

def main():
    from dotenv import load_dotenv
    load_dotenv()

    import anthropic
    import redis as redis_lib
    from tavily import TavilyClient

    from src.config import get_settings
    from src.schemas import BudgetConfig
    from src.state import make_initial_state, load_state
    from src.graph import ResearchGraph

    settings = get_settings()

    # Initialize clients
    anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    redis_client = redis_lib.Redis.from_url(settings.redis_url, decode_responses=True)

    # Define research question
    question = "What are the main causes and consequences of inflation in 2024?"
    run_id = f"example-{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"Multi-Agent Research Assistant")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}\n")

    # Create budget config
    budget_config = BudgetConfig(
        max_subquestions=3,
        max_searches_per_subquestion=2,
        max_total_tokens=50_000,
        wall_clock_timeout_seconds=120,
    )

    # Create initial state
    initial_state = make_initial_state(
        run_id=run_id,
        question=question,
        budget_config=budget_config,
    )

    # Create and run graph
    graph = ResearchGraph(
        anthropic_client=anthropic_client,
        tavily_client=tavily_client,
        redis_client=redis_client,
        planner_model=settings.planner_model,
        supervisor_model=settings.supervisor_model,
        writer_model=settings.writer_model,
        researcher_model=settings.researcher_model,
    )

    print("Running research pipeline...\n")
    start_time = time.time()

    final_state = graph.run(initial_state)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Research Complete!")
    print(f"Status: {final_state['status']}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    # Display sub-questions
    print("Sub-questions researched:")
    for sq in final_state["subquestions"]:
        print(f"  [{sq['status']}] {sq['text']}")

    print(f"\nFindings summary:")
    total_findings = sum(len(v) for v in final_state["findings"].values())
    print(f"  Total findings: {total_findings}")

    print(f"\nToken usage:")
    usage = final_state["budget_usage"]
    print(f"  Total tokens: {usage['total_tokens_used']}")

    # Display trace summary
    print(f"\nExecution trace ({len(final_state['trace'])} steps):")
    for entry in final_state["trace"]:
        tokens = entry["token_usage"].get("input_tokens", 0) + entry["token_usage"].get("output_tokens", 0)
        print(
            f"  [{entry['status']}] {entry['agent']}: "
            f"{entry['latency_ms']:.0f}ms, {tokens} tokens"
        )

    # Display report
    if final_state.get("report"):
        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}")
        print(final_state["report"])
    else:
        print(f"\nNo report generated. Error: {final_state.get('error', 'Unknown')}")

    # Demonstrate restart/resume
    print(f"\n{'='*60}")
    print("Demonstrating Redis persistence (restart/resume):")
    reloaded = load_state(redis_client, run_id)
    if reloaded:
        print(f"  Successfully loaded state from Redis for run_id={run_id}")
        print(f"  Status: {reloaded['status']}")
        print(f"  Sub-questions: {len(reloaded['subquestions'])}")
    else:
        print("  State not found in Redis (Redis may not be running)")


if __name__ == "__main__":
    main()
