"""Tests for database operations using SQLite in-memory."""
import pytest
from src.database import get_sqlite_connection, insert_run, insert_example_eval, get_run_history, get_criterion_trend


@pytest.fixture
def conn():
    c = get_sqlite_connection(":memory:")
    yield c
    c.close()


def test_insert_and_retrieve_run(conn):
    insert_run(conn, "run-001", n_examples=10, overall_score=0.75,
                scores_by_criterion={"factual_accuracy": 4.2})
    history = get_run_history(conn)
    assert len(history) == 1
    assert history[0]["run_id"] == "run-001"
    assert history[0]["overall_score"] == 0.75


def test_insert_multiple_runs(conn):
    for i in range(5):
        insert_run(conn, f"run-{i:03d}", n_examples=10, overall_score=0.6 + i * 0.05,
                    scores_by_criterion={})
    history = get_run_history(conn, limit=3)
    assert len(history) == 3


def test_insert_example_eval(conn):
    insert_run(conn, "run-001", n_examples=1, overall_score=0.8, scores_by_criterion={})
    insert_example_eval(
        conn, "run-001", "ex_001", "easy",
        scores={"factual_accuracy": 4, "citation_quality": 5},
        reasoning={"factual_accuracy": "Accurate.", "citation_quality": "Well cited."},
    )
    rows = conn.execute("SELECT * FROM example_evaluations WHERE run_id='run-001'").fetchall()
    assert len(rows) == 2  # one per criterion


def test_criterion_trend(conn):
    for i in range(3):
        run_id = f"run-{i:03d}"
        insert_run(conn, run_id, n_examples=5, overall_score=0.7, scores_by_criterion={})
        insert_example_eval(conn, run_id, "ex_001", "easy",
                             scores={"factual_accuracy": 3 + i},
                             reasoning={})
    trend = get_criterion_trend(conn, "factual_accuracy")
    assert len(trend) == 3


def test_run_id_unique_constraint(conn):
    insert_run(conn, "run-dup", n_examples=5, overall_score=0.5, scores_by_criterion={})
    with pytest.raises(Exception):  # UNIQUE constraint violation
        insert_run(conn, "run-dup", n_examples=5, overall_score=0.5, scores_by_criterion={})
