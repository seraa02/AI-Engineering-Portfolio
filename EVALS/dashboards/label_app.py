"""
Streamlit human-labeling interface for the LLM-judge calibration dataset (PDF Phase 2).

Shows one (question, answer) example at a time with the full rubric (5 criteria, every
scale point worded), takes a per-criterion score + a free-text note, and writes the row
to the real Postgres `human_labels` table (labeler id + timestamp + scores + note).

Run:
    streamlit run dashboards/label_app.py --server.headless true

The sidebar lets a labeler pick their labeler_id and a session tag ("session_1" the first
time through the 200 examples, "session_2" on a later day for the 20 double-labeled items),
and tracks progress against the live Postgres data so you always know how many of the 200
you've done and how many of the 20 double-labels remain.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.database import count_labeled_examples, get_human_labels, get_postgres_connection, insert_human_label
from src.dataset import load_examples
from src.rubric import CRITERIA

st.set_page_config(page_title="LLM-Judge Human Labeling", layout="wide")


@st.cache_resource
def get_conn():
    return get_postgres_connection()


@st.cache_data(ttl=5)
def get_examples():
    return load_examples()


def already_labeled_sessions(conn, example_id: str) -> set[str]:
    return {row["session"] for row in get_human_labels(conn, example_id=example_id)}


def main():
    st.title("LLM-Judge Calibration — Human Labeling")
    st.caption(
        "Phase 2 of the LLM-as-Judge project: build the real human calibration ground truth. "
        "Score each answer against the rubric below, add a note when the rubric feels ambiguous "
        "(those notes are exactly what Phase 3 uses to sharpen the wording)."
    )

    conn = get_conn()
    examples = get_examples()

    # --- sidebar: labeler identity + progress ---
    st.sidebar.header("Labeler")
    labeler_id = st.sidebar.text_input("Labeler ID", value="sahera")
    session = st.sidebar.selectbox(
        "Session",
        ["session_1", "session_2"],
        help="Use session_1 for your first pass through the 200 examples. "
             "Come back on a DIFFERENT DAY and use session_2 to re-label the "
             "same 20 examples for the self-agreement check (Phase 2/3).",
    )

    counts = count_labeled_examples(conn)
    st.sidebar.metric("Unique examples labeled", f"{counts['n_unique_examples_labeled']} / {len(examples)}")
    st.sidebar.metric("Double-labeled (target 20)", counts["n_double_labeled"])
    st.sidebar.metric("Total label rows", counts["n_total_label_rows"])
    st.sidebar.progress(min(counts["n_unique_examples_labeled"] / max(len(examples), 1), 1.0))

    # --- pick which example to show ---
    labeled_ids = {
        row["example_id"] for row in get_human_labels(conn) if row["labeler_id"] == labeler_id
    }
    if session == "session_1":
        candidates = [ex for ex in examples if ex["id"] not in labeled_ids] or examples
    else:
        # session_2: only offer examples already labeled in session_1 (the double-label set)
        s1_ids = {
            row["example_id"] for row in get_human_labels(conn)
            if row["labeler_id"] == labeler_id and row["session"] == "session_1"
        }
        s2_ids = {
            row["example_id"] for row in get_human_labels(conn)
            if row["labeler_id"] == labeler_id and row["session"] == "session_2"
        }
        candidates = [ex for ex in examples if ex["id"] in (s1_ids - s2_ids)]
        if not candidates:
            st.info(
                "No session_1 labels available yet to double-label, or you've already "
                "double-labeled everything you've labeled so far. Label some in session_1 first, "
                "then come back on a different day and switch to session_2."
            )
            return

    ex_ids = [ex["id"] for ex in candidates]
    chosen_id = st.selectbox("Example", ex_ids, index=0)
    example = next(ex for ex in candidates if ex["id"] == chosen_id)

    prior_sessions = already_labeled_sessions(conn, example["id"])
    if session in prior_sessions:
        st.warning(f"You already have a `{session}` label for `{example['id']}`. Submitting again adds another row.")

    # --- show the example ---
    st.subheader("Question")
    st.write(example["input"])
    st.subheader("Answer to evaluate")
    st.code(example["model_output"], language=None)
    st.caption(f"category: {example.get('category', '?')} · id: {example['id']}")

    # --- rubric + score inputs ---
    st.subheader("Rubric")
    scores: dict[str, int] = {}
    for criterion in CRITERIA:
        with st.expander(f"{criterion.name}  (weight {criterion.weight}×, scale 1–{criterion.scale})", expanded=True):
            st.write(criterion.description)
            for point in sorted(criterion.anchors):
                anchor = criterion.anchors[point]
                st.markdown(f"- **{point} — {anchor.label}**: _{anchor.example}_")
            scores[criterion.name] = st.radio(
                f"Score for {criterion.name}",
                options=list(range(1, criterion.scale + 1)),
                horizontal=True,
                key=f"score_{criterion.name}_{example['id']}_{session}",
            )

    notes = st.text_area(
        "Free-text note (optional, but write one whenever the rubric felt ambiguous — "
        "this is the input to Phase 3's rubric-sharpening pass)",
        key=f"notes_{example['id']}_{session}",
    )

    if st.button("Submit label", type="primary"):
        insert_human_label(conn, example["id"], labeler_id, session, scores, notes)
        st.success(f"Saved label for {example['id']} ({session}, labeler={labeler_id}).")
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
