"""
Storage for evaluation runs, per-example judge scores, and human calibration labels.

Two backends, same logical schema:
  - SQLite (in-memory or file) — used by the test suite and for offline/local dev.
  - Postgres — the real production store. Labeler id + timestamp + per-criterion
    scores + free-text notes live in `human_labels`; judge run history lives in
    `evaluation_runs` / `example_evaluations`.

`get_postgres_connection()` returns a thin wrapper around a psycopg2 connection that
exposes the same `.execute(sql, params).fetchall()` / `.commit()` / `.close()` surface
as `sqlite3.Connection`, so `insert_run`, `insert_example_eval`, `get_run_history`,
`get_criterion_trend`, `insert_human_label`, and `get_human_labels` all work unchanged
against either backend — SQL uses `?` placeholders throughout and the Postgres wrapper
rewrites them to `%s` before executing.
"""
from __future__ import annotations

import json
import os
from typing import Optional


# ---------------------------------------------------------------------------
# SQLite backend (tests / local dev)
# ---------------------------------------------------------------------------

def _create_schema_sqlite(conn) -> None:
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS human_labels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            example_id TEXT NOT NULL,
            labeler_id TEXT NOT NULL,
            session TEXT NOT NULL,
            scores TEXT NOT NULL,
            notes TEXT,
            labeled_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()


def get_sqlite_connection(db_path: str = ":memory:"):
    """Create a SQLite connection with the evaluation schema (for testing / local dev)."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _create_schema_sqlite(conn)
    return conn


# ---------------------------------------------------------------------------
# Postgres backend (production — real labeler data lives here)
# ---------------------------------------------------------------------------

class _PGConnWrapper:
    """
    Minimal sqlite3.Connection-compatible wrapper around a psycopg2 connection.

    Lets `insert_run`, `insert_example_eval`, `get_run_history`, `get_criterion_trend`,
    `insert_human_label`, and `get_human_labels` run unmodified against Postgres: SQL is
    written once with `?` placeholders and `%s`-ified here.
    """

    def __init__(self, pg_conn):
        self._conn = pg_conn

    def execute(self, sql: str, params=()):
        from psycopg2.extras import RealDictCursor

        cur = self._conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql.replace("?", "%s"), tuple(params))
        return cur

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __getattr__(self, name):
        # Fall through to the underlying psycopg2 connection for anything else.
        return getattr(self._conn, name)


def _create_schema_postgres(conn: "_PGConnWrapper") -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id SERIAL PRIMARY KEY,
            run_id TEXT UNIQUE NOT NULL,
            run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            app_version TEXT,
            model_version TEXT,
            n_examples INTEGER,
            overall_score REAL,
            scores_by_criterion TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS example_evaluations (
            id SERIAL PRIMARY KEY,
            run_id TEXT NOT NULL,
            example_id TEXT NOT NULL,
            category TEXT,
            criterion TEXT,
            score INTEGER,
            judge_reasoning TEXT,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS human_labels (
            id SERIAL PRIMARY KEY,
            example_id TEXT NOT NULL,
            labeler_id TEXT NOT NULL,
            session TEXT NOT NULL,
            scores JSONB NOT NULL,
            notes TEXT,
            labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_human_labels_example ON human_labels(example_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_human_labels_labeler ON human_labels(labeler_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_example_evals_run ON example_evaluations(run_id)")
    conn.commit()


def get_postgres_connection(dsn: Optional[str] = None) -> _PGConnWrapper:
    """
    Connect to the real Postgres instance and ensure the schema exists.

    dsn defaults to the POSTGRES_DSN environment variable (see .env / .env.example),
    e.g. postgresql://evals_user:changeme125@localhost:5434/evals_db
    """
    import psycopg2

    dsn = dsn or os.environ.get("POSTGRES_DSN")
    if not dsn:
        raise ValueError(
            "No Postgres DSN provided and POSTGRES_DSN is not set in the environment. "
            "Set POSTGRES_DSN in EVALS/.env, e.g. "
            "postgresql://evals_user:changeme125@localhost:5434/evals_db"
        )
    raw_conn = psycopg2.connect(dsn)
    conn = _PGConnWrapper(raw_conn)
    _create_schema_postgres(conn)
    return conn


# ---------------------------------------------------------------------------
# Judge run history (works against either backend)
# ---------------------------------------------------------------------------

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


def get_latest_run(conn) -> Optional[dict]:
    """Most recent stored run, or None if no runs exist yet. Used for CI regression checks."""
    history = get_run_history(conn, limit=1)
    if not history:
        return None
    run = history[0]
    scores = run.get("scores_by_criterion")
    if isinstance(scores, str):
        run["scores_by_criterion"] = json.loads(scores) if scores else {}
    return run


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


# ---------------------------------------------------------------------------
# Human labels — the real calibration ground truth (Phase 2)
# ---------------------------------------------------------------------------

def insert_human_label(
    conn,
    example_id: str,
    labeler_id: str,
    session: str,
    scores: dict[str, int],
    notes: str = "",
) -> None:
    """
    Store one human label: a labeler's per-criterion scores + free-text note for one
    example, tagged with a session id so the same example can be labeled twice on
    different days for self-agreement measurement.
    """
    conn.execute(
        """
        INSERT INTO human_labels (example_id, labeler_id, session, scores, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (example_id, labeler_id, session, json.dumps(scores), notes),
    )
    conn.commit()


def get_human_labels(conn, example_id: Optional[str] = None, labeler_id: Optional[str] = None) -> list[dict]:
    """Fetch human labels, optionally filtered by example or labeler."""
    sql = "SELECT * FROM human_labels"
    clauses, params = [], []
    if example_id:
        clauses.append("example_id = ?")
        params.append(example_id)
    if labeler_id:
        clauses.append("labeler_id = ?")
        params.append(labeler_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY labeled_at"
    rows = conn.execute(sql, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if isinstance(d.get("scores"), str):
            d["scores"] = json.loads(d["scores"])
        result.append(d)
    return result


def count_labeled_examples(conn) -> dict:
    """Summary stats used by the Streamlit labeling app's progress bar."""
    rows = conn.execute("SELECT DISTINCT example_id FROM human_labels").fetchall()
    n_unique = len(rows)
    dup_rows = conn.execute(
        """
        SELECT example_id, COUNT(*) as n FROM human_labels
        GROUP BY example_id HAVING COUNT(*) >= 2
        """
    ).fetchall()
    n_double_labeled = len(dup_rows)
    total_rows = conn.execute("SELECT COUNT(*) as n FROM human_labels").fetchall()
    n_total = dict(total_rows[0])["n"] if total_rows else 0
    return {
        "n_unique_examples_labeled": n_unique,
        "n_double_labeled": n_double_labeled,
        "n_total_label_rows": n_total,
    }
