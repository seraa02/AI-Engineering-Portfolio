"""
Benchmark: hybrid (graph + vector) vs. vector-only baseline.

Purpose:
    Run every item in eval/golden_dataset.json through both systems and
    report accuracy broken out by hop count/difficulty, plus latency and
    cost per query and one-time ingestion cost -- the portfolio artifact
    this whole project builds toward (per the build guide: "This
    benchmark table is the portfolio artifact. Put it at the top of the
    README, above the architecture diagram.").

Grading methodology (documented, not hidden):
    A mechanical, not an LLM-judge, check: for non-out-of-scope items,
    correct = every gold_entity appears (case-insensitive substring) in
    the answer text. For out-of-scope items, correct = the system
    produced zero grounded claims (i.e. it declined rather than
    fabricating an answer). This is a defensible floor for a
    development-tier benchmark, not a replacement for the LLM-as-judge
    methodology Project 4 builds -- see README's Known Limitations.

    Aggregation items whose gold_count is still null (pending manual
    verification per annotations.json) are graded on entity coverage
    only, same as other items -- the count itself isn't checked until a
    human fills in gold_count.

The hybrid system is the real router -> retrieval -> merge pipeline
(src/api/main.py's logic, called directly rather than over HTTP for
speed). The vector-only baseline is the same merge/citation-validation
code path, but retrieval is forced to vector search only -- this isolates
what the graph adds, rather than comparing two different answering
strategies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from src.config import get_settings
from src.ingest.embed import get_connection as get_pg_connection
from src.ingest.extract import compute_total_extraction_cost
from src.ingest.graph_writer import get_driver
from src.retrieval.graph_query import GraphQueryPlan, execute_graph_query
from src.retrieval.merge import build_answer, graph_rows_to_statements
from src.retrieval.router import classify_route
from src.retrieval.vector_query import vector_search

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = _PROJECT_ROOT / "eval" / "golden_dataset.json"
RESULTS_PATH = _PROJECT_ROOT / "eval" / "benchmark_results.json"


@dataclass
class QueryOutcome:
    question_id: str
    correct: bool
    route: str | None
    latency_ms: float
    cost_usd: float
    answer_text: str
    graph_evidence_count: int
    vector_evidence_count: int


def _estimate_cost_usd(route: str, is_vector_only: bool = False) -> float:
    """
    Estimate per-query API cost in USD based on route.
    Uses Claude Haiku pricing ($0.80/MTok input, $4.00/MTok output).

    Router call:   ~200 input + ~50 output tokens
    GRAPH answer:  ~1500 input + ~200 output tokens
    VECTOR answer: ~2000 input + ~250 output tokens
    BOTH answer:   ~2500 input + ~300 output tokens
    """
    INPUT_RATE = 0.80 / 1_000_000
    OUTPUT_RATE = 4.00 / 1_000_000

    if is_vector_only:
        # No router call; answer only
        return round((1800 * INPUT_RATE) + (220 * OUTPUT_RATE), 6)

    router = (200 * INPUT_RATE) + (50 * OUTPUT_RATE)
    if route == "GRAPH":
        answer = (1500 * INPUT_RATE) + (200 * OUTPUT_RATE)
    elif route == "VECTOR":
        answer = (2000 * INPUT_RATE) + (250 * OUTPUT_RATE)
    else:  # BOTH
        answer = (2500 * INPUT_RATE) + (300 * OUTPUT_RATE)
    return round(router + answer, 6)


@dataclass
class SystemResults:
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def accuracy(self, difficulty: str | None = None) -> float | None:
        items = self.outcomes if difficulty is None else [o for o in self.outcomes if _difficulty_of(o.question_id) == difficulty]
        if not items:
            return None
        return sum(o.correct for o in items) / len(items)

    def avg_latency_ms(self) -> float:
        return sum(o.latency_ms for o in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0

    def p50_latency_ms(self) -> float:
        if not self.outcomes:
            return 0.0
        lats = sorted(o.latency_ms for o in self.outcomes)
        mid = len(lats) // 2
        return (lats[mid - 1] + lats[mid]) / 2 if len(lats) % 2 == 0 else lats[mid]

    def p95_latency_ms(self) -> float:
        if not self.outcomes:
            return 0.0
        lats = sorted(o.latency_ms for o in self.outcomes)
        idx = int(len(lats) * 0.95)
        return lats[min(idx, len(lats) - 1)]

    def avg_cost_usd(self) -> float:
        return sum(o.cost_usd for o in self.outcomes) / len(self.outcomes) if self.outcomes else 0.0


_golden_by_id: dict[str, dict] = {}


def _difficulty_of(question_id: str) -> str:
    return _golden_by_id[question_id]["difficulty"]


def _grade(item: dict, claims: list, answer_text: str) -> bool:
    """
    gold_entities is a list of SYNONYM GROUPS (e.g. [["AMD", "Advanced
    Micro Devices"], ["Intel"]]) -- correct requires at least one string
    from EVERY group to appear in the answer, not every string in a flat
    list. A flat exact-legal-name check (the original version of this
    function) silently failed correct answers: "NVIDIA CORP" is not a
    substring of a fluent answer that says "NVIDIA's Compute & Networking
    segment...". See golden_dataset.json's top-level _note.
    """
    if item["difficulty"] == "out_of_scope":
        return len(claims) == 0
    gold_entities = item.get("gold_entities", [])
    if not gold_entities:
        return len(claims) > 0  # no specific entities to check -- just require a grounded answer
    lowered = answer_text.lower()
    return all(any(synonym.lower() in lowered for synonym in group) for group in gold_entities)


def run_hybrid(client, driver, pg_conn, question: str) -> tuple[str, str, list, int, int]:
    """Returns (route, answer_text, claims, graph_evidence_count, vector_evidence_count)."""
    decision = classify_route(client, question)

    graph_statements = []
    if decision.route in ("GRAPH", "BOTH") and decision.query_type:
        plan = GraphQueryPlan(
            query_type=decision.query_type,
            entity_names=decision.entity_names,
            relationship_type=decision.relationship_type,
            second_relationship_type=decision.second_relationship_type,
            max_hops=decision.max_hops or 3,
        )
        rows = execute_graph_query(driver, plan)
        graph_statements = graph_rows_to_statements(plan, rows)

    vector_results = []
    if decision.route in ("VECTOR", "BOTH"):
        vector_results = vector_search(pg_conn, question, k=5)

    merged = build_answer(client, question, graph_statements, vector_results)
    return decision.route, merged.answer_text, merged.claims, merged.graph_evidence_count, merged.vector_evidence_count


def run_vector_only(client, pg_conn, question: str) -> tuple[str, list, int]:
    """Baseline: vector search only, no router, no graph."""
    vector_results = vector_search(pg_conn, question, k=5)
    merged = build_answer(client, question, [], vector_results)
    return merged.answer_text, merged.claims, merged.vector_evidence_count


def run_benchmark() -> None:
    global _golden_by_id

    golden = json.loads(GOLDEN_PATH.read_text())
    items = [item for item in golden["items"]]
    _golden_by_id = {item["id"]: item for item in items}

    questions_by_id = {
        q["id"]: q["question"]
        for q in json.loads((_PROJECT_ROOT / "eval" / "questions.json").read_text())["questions"]
    }

    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    driver = get_driver(settings)
    pg_conn = get_pg_connection(settings)

    hybrid = SystemResults()
    vector_only = SystemResults()

    try:
        for item in items:
            qid, question = item["id"], questions_by_id[item["id"]]

            start = time.monotonic()
            route, answer, claims, gcount, vcount = run_hybrid(client, driver, pg_conn, question)
            hybrid_latency = (time.monotonic() - start) * 1000
            hybrid.outcomes.append(
                QueryOutcome(
                    question_id=qid,
                    correct=_grade(item, claims, answer),
                    route=route,
                    latency_ms=hybrid_latency,
                    cost_usd=_estimate_cost_usd(route),
                    answer_text=answer,
                    graph_evidence_count=gcount,
                    vector_evidence_count=vcount,
                )
            )

            start = time.monotonic()
            v_answer, v_claims, v_vcount = run_vector_only(client, pg_conn, question)
            vector_latency = (time.monotonic() - start) * 1000
            vector_only.outcomes.append(
                QueryOutcome(
                    question_id=qid,
                    correct=_grade(item, v_claims, v_answer),
                    route="VECTOR",
                    latency_ms=vector_latency,
                    cost_usd=_estimate_cost_usd("VECTOR", is_vector_only=True),
                    answer_text=v_answer,
                    graph_evidence_count=0,
                    vector_evidence_count=v_vcount,
                )
            )
            print(
                f"[{qid}] hybrid={'OK' if hybrid.outcomes[-1].correct else 'MISS'} "
                f"({route}, {hybrid_latency:.0f}ms) | "
                f"vector_only={'OK' if vector_only.outcomes[-1].correct else 'MISS'} "
                f"({vector_latency:.0f}ms)"
            )
    finally:
        driver.close()
        pg_conn.close()

    difficulties = sorted({item["difficulty"] for item in items})
    report = {
        "overall": {"hybrid_accuracy": hybrid.accuracy(), "vector_only_accuracy": vector_only.accuracy()},
        "by_difficulty": {
            d: {"hybrid_accuracy": hybrid.accuracy(d), "vector_only_accuracy": vector_only.accuracy(d)}
            for d in difficulties
        },
        "latency_ms": {
            "hybrid_avg": hybrid.avg_latency_ms(),
            "hybrid_p50": hybrid.p50_latency_ms(),
            "hybrid_p95": hybrid.p95_latency_ms(),
            "vector_only_avg": vector_only.avg_latency_ms(),
            "vector_only_p50": vector_only.p50_latency_ms(),
            "vector_only_p95": vector_only.p95_latency_ms(),
        },
        "cost_usd": {
            "hybrid_avg_per_query": hybrid.avg_cost_usd(),
            "vector_only_avg_per_query": vector_only.avg_cost_usd(),
        },
        "ingestion_cost": compute_total_extraction_cost(),
        "per_question": [
            {
                "question_id": h.question_id,
                "difficulty": _difficulty_of(h.question_id),
                "hybrid_correct": h.correct,
                "hybrid_route": h.route,
                "hybrid_latency_ms": round(h.latency_ms, 1),
                "hybrid_cost_usd": h.cost_usd,
                "vector_only_correct": v.correct,
                "vector_only_latency_ms": round(v.latency_ms, 1),
                "vector_only_cost_usd": v.cost_usd,
            }
            for h, v in zip(hybrid.outcomes, vector_only.outcomes)
        ],
    }

    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    print()
    print("=== Benchmark complete ===")
    print(f"Overall accuracy: hybrid={report['overall']['hybrid_accuracy']:.0%}  vector_only={report['overall']['vector_only_accuracy']:.0%}")
    for d in difficulties:
        r = report["by_difficulty"][d]
        print(f"  {d:15s} hybrid={r['hybrid_accuracy']:.0%}  vector_only={r['vector_only_accuracy']:.0%}")
    print(f"Latency (avg/p50/p95): hybrid={report['latency_ms']['hybrid_avg']:.0f}/{report['latency_ms']['hybrid_p50']:.0f}/{report['latency_ms']['hybrid_p95']:.0f}ms  "
          f"vector_only={report['latency_ms']['vector_only_avg']:.0f}/{report['latency_ms']['vector_only_p50']:.0f}/{report['latency_ms']['vector_only_p95']:.0f}ms")
    print(f"Avg cost/query: hybrid=${report['cost_usd']['hybrid_avg_per_query']:.6f}  vector_only=${report['cost_usd']['vector_only_avg_per_query']:.6f}")
    print(f"Ingestion cost (one-time): ${report['ingestion_cost']['total_cost_usd']}")
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    run_benchmark()
