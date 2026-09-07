"""
Recall@k measurement and ef_search tuning for the vector retrieval path.

Purpose:
    The build guide's Phase 2 "Done when": "Index with HNSW, tune
    ef_search, and measure recall at k on a small labeled set before
    moving on. Do not build the router on top of retrieval you have not
    measured." This script is that measurement -- run once against
    eval/recall_labels.json (see build_recall_labels.py for how that set
    was built) to pick src/retrieval/vector_query.py's DEFAULT_EF_SEARCH
    with evidence instead of a guess.

Method:
    For each candidate ef_search value, run every labeled question through
    vector_search() and check whether the gold_chunk_id appears in the
    top-k results, for several k. recall@k = fraction of labeled questions
    where the gold chunk was retrieved within the top k.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.ingest.embed import get_connection
from src.retrieval.vector_query import vector_search

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = _PROJECT_ROOT / "eval" / "recall_labels.json"
RESULTS_PATH = _PROJECT_ROOT / "eval" / "recall_results.json"

EF_SEARCH_CANDIDATES = [10, 40, 100, 200]
K_VALUES = [1, 3, 5, 10]


def _recall_at_k(ranks: list[int | None], k_values: list[int]) -> dict[str, float]:
    """
    ranks[i] is the 1-indexed position of label i's gold chunk in its
    retrieved list, or None if it wasn't retrieved at all. recall@k is
    the fraction of labels whose gold chunk ranked at or above k.
    Pulled out as a pure function so the aggregation math is testable
    without a live database (see tests/test_recall_at_k.py).
    """
    if not ranks:
        return {f"recall@{k}": 0.0 for k in k_values}
    return {
        f"recall@{k}": sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)
        for k in k_values
    }


def run_recall_sweep() -> dict:
    labels = json.loads(LABELS_PATH.read_text())["labels"]
    if not labels:
        raise RuntimeError(f"No labels in {LABELS_PATH} -- run build_recall_labels.py first.")

    conn = get_connection(get_settings())
    max_k = max(K_VALUES)

    results: dict[str, dict[str, float]] = {}
    try:
        for ef_search in EF_SEARCH_CANDIDATES:
            ranks: list[int | None] = []
            for label in labels:
                retrieved = vector_search(conn, label["question"], k=max_k, ef_search=ef_search)
                retrieved_ids = [r.chunk_id for r in retrieved]
                ranks.append(
                    retrieved_ids.index(label["gold_chunk_id"]) + 1
                    if label["gold_chunk_id"] in retrieved_ids
                    else None
                )
            results[str(ef_search)] = _recall_at_k(ranks, K_VALUES)
    finally:
        conn.close()

    return {"n_labels": len(labels), "by_ef_search": results}


def main() -> None:
    report = run_recall_sweep()
    RESULTS_PATH.write_text(json.dumps(report, indent=2))

    print(f"=== Recall@k sweep ({report['n_labels']} labeled questions) ===")
    header = "ef_search".ljust(10) + "".join(f"recall@{k}".rjust(12) for k in K_VALUES)
    print(header)
    for ef_search, metrics in report["by_ef_search"].items():
        row = str(ef_search).ljust(10) + "".join(f"{metrics[f'recall@{k}']:.0%}".rjust(12) for k in K_VALUES)
        print(row)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
