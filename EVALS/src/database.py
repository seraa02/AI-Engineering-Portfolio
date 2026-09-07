"""
Postgres storage for evaluation runs and per-example scores.

In tests, uses SQLite in-memory (via the same SQL schema — Postgres-compatible subset).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional


def _create_schema(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            run_at TEXT DEFAULT (datetime('now')),
            app_version TEXT,
            model_version TEXT,
            n_examples INTEGER,
            overall_score REAL,
            scores_by_criterion TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS example_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            example_id TEXT NOT NULL,
            category TEXT,
            criterion TEXT,
            score INTEGER,
            judge_reasoning TEXT,
            evaluated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def get_sqlite_connection(db_path: str = ":memory:"):
    """Create a SQLite connection with the evaluation schema (for testing / local dev)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    return conn


def insert_run(
    conn,
    run_id: str,
    n_examples: int,
    overall_score: float,
    scores_by_criterion: dict,
    app_version: str = "dev",
    model_version: str = "claude-haiku-4-5",
) -> None:
    conn.execute(
        """
        INSERT INTO evaluation_runs
            (run_id, app_version, model_version, n_examples, overall_score, scores_by_criterion)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, app_version, model_version, n_examples, overall_score, json.dumps(scores_by_criterion)),
    )
    conn.commit()


def insert_example_eval(
    conn,
    run_id: str,
    example_id: str,
    category: str,
    scores: dict[str, int],
    reasoning: dict[str, str],
) -> None:
    for criterion, score in scores.items():
        conn.execute(
            """
            INSERT INTO example_evaluations
                (run_id, example_id, category, criterion, score, judge_reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, example_id, category, criterion, score, reasoning.get(criterion, "")),
        )
    conn.commit()


def get_run_history(conn, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM evaluation_runs ORDER BY run_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


def get_criterion_trend(conn, criterion: str, limit: int = 20) -> list[dict]:
    """Average score for a criterion across the most recent runs."""
    rows = conn.execute(
        """
        SELECT r.run_at, AVG(e.score) as avg_score
        FROM example_evaluations e
        JOIN evaluation_runs r ON e.run_id = r.run_id
        WHERE e.criterion = ?
        GROUP BY r.run_id, r.run_at
        ORDER BY r.run_at DESC
        LIMIT ?
        """,
        (criterion, limit),
    ).fetchall()
    return [dict(r) for r in rows]
