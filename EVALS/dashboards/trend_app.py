"""
Streamlit trend dashboard: judge-run score history over time (Phase 5 of the PDF spec —
"chart scores over time in Streamlit so quality has a trend line, not an anecdote").

Reads `evaluation_runs` / `example_evaluations` from the real Postgres instance, written by
`python -m src.ci.eval_gate --live --store` (see src/ci/eval_gate.py).

Run:
    streamlit run dashboards/trend_app.py --server.headless true --server.port 8502
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.ci.eval_gate import THRESHOLDS
from src.database import get_criterion_trend, get_postgres_connection, get_run_history
from src.rubric import criterion_names

st.set_page_config(page_title="LLM-Judge Score Trends", layout="wide")


@st.cache_resource
def get_conn():
    return get_postgres_connection()


def main():
    st.title("LLM-Judge — Score Trends Over Time")
    st.caption(
        "Every CI eval-gate run is stored here. This is what turns 'did quality regress?' "
        "into a chart instead of a one-off anecdote."
    )

    conn = get_conn()
    runs = get_run_history(conn, limit=200)

    if not runs:
        st.info(
            "No runs stored yet. Run `python -m src.ci.eval_gate --live --store` "
            "(or `--degraded --live --store` to see a regression) to populate this dashboard."
        )
        return

    df = pd.DataFrame(runs)
    df["run_at"] = pd.to_datetime(df["run_at"])
    df = df.sort_values("run_at")

    st.subheader("Overall weighted score per run")
    st.line_chart(df.set_index("run_at")["overall_score"])

    st.subheader("Per-criterion trend")
    criterion = st.selectbox("Criterion", criterion_names())
    trend = get_criterion_trend(conn, criterion, limit=200)
    if trend:
        tdf = pd.DataFrame(trend)
        tdf["run_at"] = pd.to_datetime(tdf["run_at"])
        tdf = tdf.sort_values("run_at")
        st.line_chart(tdf.set_index("run_at")["avg_score"])
        threshold = THRESHOLDS.get(criterion)
        if threshold is not None:
            st.caption(f"CI floor for {criterion}: raw avg score, gate threshold on normalized scale = {threshold}")
    else:
        st.info(f"No per-example data yet for '{criterion}'.")

    st.subheader("Raw run history")
    st.dataframe(df[["run_id", "run_at", "n_examples", "overall_score", "app_version", "model_version"]])


if __name__ == "__main__":
    main()
