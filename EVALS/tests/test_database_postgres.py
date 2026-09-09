"""
Real Postgres integration tests (not SQLite).

Requires a live Postgres reachable at POSTGRES_DSN (see EVALS/.env). Skipped automatically
if the DB isn't reachable, so the default test run (no DB running) still passes cleanly.
Uses a dedicated labeler_id/example_id prefix ("pytest_") and cleans up after itself so
it never pollutes real human-labeling data.
"""
from __future__ import annotations

import os
import uuid

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from src.database import (
    count_labeled_examples,
    get_human_labels,
    get_latest_run,
    get_postgres_connection,
    get_run_history,
    insert_example_eval,
    insert_human_label,
    insert_run,
)


def _postgres_available() -> bool:
    try:
        conn = get_postgres_connection()
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="requires a live Postgres at POSTGRES_DSN (docker container 'evals_postgres')",
)


@pytest.fixture
def pg_conn():
    conn = get_postgres_connection()
    yield conn
    conn.close()


@pytest.fixture
def run_id():
    return f"pytest_run_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def example_id():
    return f"pytest_ex_{uuid.uuid4().hex[:8]}"


def test_schema_has_three_tables(pg_conn):
    rows = pg_conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    ).fetchall()
    names = {dict(r)["table_name"] for r in rows}
    assert {"evaluation_runs", "example_evaluations", "human_labels"} <= names


def test_insert_and_retrieve_run_postgres(pg_conn, run_id):
    insert_run(pg_conn, run_id, n_examples=5, overall_score=0.82,
               scores_by_criterion={"factual_accuracy": 4.1})
    history = get_run_history(pg_conn, limit=5)
    assert any(r["run_id"] == run_id for r in history)
    pg_conn.execute("DELETE FROM evaluation_runs WHERE run_id = ?", (run_id,))
    pg_conn.commit()


def test_get_latest_run_postgres(pg_conn, run_id):
    insert_run(pg_conn, run_id, n_examples=3, overall_score=0.7,
               scores_by_criterion={"hallucination_avoidance": 0.9})
    latest = get_latest_run(pg_conn)
    assert latest is not None
    assert isinstance(latest["scores_by_criterion"], dict)
    pg_conn.execute("DELETE FROM evaluation_runs WHERE run_id = ?", (run_id,))
    pg_conn.commit()


def test_insert_example_eval_postgres(pg_conn, run_id, example_id):
    insert_run(pg_conn, run_id, n_examples=1, overall_score=0.8, scores_by_criterion={})
    insert_example_eval(
        pg_conn, run_id, example_id, "easy",
        scores={"factual_accuracy": 4, "citation_quality": 5},
        reasoning={"factual_accuracy": "Accurate.", "citation_quality": "Well cited."},
    )
    rows = pg_conn.execute(
        "SELECT * FROM example_evaluations WHERE run_id = ?", (run_id,)
    ).fetchall()
    assert len(rows) == 2
    pg_conn.execute("DELETE FROM example_evaluations WHERE run_id = ?", (run_id,))
    pg_conn.execute("DELETE FROM evaluation_runs WHERE run_id = ?", (run_id,))
    pg_conn.commit()


def test_human_label_round_trip_jsonb(pg_conn, example_id):
    """Labeler id, timestamp, per-criterion scores, and free-text note all persist correctly."""
    insert_human_label(
        pg_conn, example_id, labeler_id="pytest_labeler", session="session_1",
        scores={"factual_accuracy": 3, "citation_quality": 2, "completeness": 4,
                "coherence": 2, "hallucination_avoidance": 3},
        notes="Rubric ambiguous on partial citations.",
    )
    labels = get_human_labels(pg_conn, example_id=example_id)
    assert len(labels) == 1
    row = labels[0]
    assert row["labeler_id"] == "pytest_labeler"
    assert row["session"] == "session_1"
    assert row["notes"] == "Rubric ambiguous on partial citations."
    assert row["scores"]["factual_accuracy"] == 3  # JSONB round-trips as a real dict, not a string
    assert row["labeled_at"] is not None
    pg_conn.execute("DELETE FROM human_labels WHERE example_id = ?", (example_id,))
    pg_conn.commit()


def test_double_label_same_example_different_session(pg_conn, example_id):
    """Simulates the 'label 20 twice on different days' workflow (Phase 2 of the PDF spec)."""
    insert_human_label(pg_conn, example_id, "pytest_labeler", "session_1",
                        {"factual_accuracy": 4, "citation_quality": 4, "completeness": 4,
                         "coherence": 3, "hallucination_avoidance": 3}, notes="")
    insert_human_label(pg_conn, example_id, "pytest_labeler", "session_2",
                        {"factual_accuracy": 3, "citation_quality": 4, "completeness": 4,
                         "coherence": 3, "hallucination_avoidance": 2}, notes="second pass")
    labels = get_human_labels(pg_conn, example_id=example_id)
    assert len(labels) == 2
    sessions = {l["session"] for l in labels}
    assert sessions == {"session_1", "session_2"}
    counts = count_labeled_examples(pg_conn)
    assert counts["n_double_labeled"] >= 1
    pg_conn.execute("DELETE FROM human_labels WHERE example_id = ?", (example_id,))
    pg_conn.commit()
