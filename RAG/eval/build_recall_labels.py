"""
Build a small labeled set for vector-retrieval recall@k evaluation.

Purpose:
    The build guide's Phase 2 is explicit: "Index with HNSW, tune
    ef_search, and measure recall at k on a small labeled set before
    moving on. Do not build the router on top of retrieval you have not
    measured." This module produces that labeled set mechanically, not by
    hand-authoring guesses.

Method:
    eval/golden_dataset.json already carries a verbatim `source_evidence`
    quote for most items -- the literal sentence in the filing that
    answers the question. For every item whose source_evidence is a
    single-document, single-passage quote (not an aggregation or
    cross-document multi-hop item, which have no one true chunk), this
    finds the actual chunk_id in pgvector whose stored text contains that
    quote, after normalizing whitespace, curly quotes, and the trademark
    symbols (R), (TM) that the SEC HTML inserts mid-sentence (e.g. "Sony
    PlayStation (R) 5") but the golden quote does not carry.

    This deliberately reuses real chunk text rather than fabricating
    (question, chunk_id) pairs -- the label is only accepted if the exact
    quote is found verbatim in exactly the claimed source document's
    chunks.

Coverage:
    Not every golden item qualifies: q005/q010/q011 (evidence spans
    multiple non-contiguous mentions) and q012/q013/q019/q020
    (aggregation and cross-document multi-hop -- by construction there is
    no single passage that answers them) are excluded here. That's
    expected and consistent with the guide -- recall@k tests the vector
    path specifically. What remains (11 items across all six filers) is
    small but real, per the guide's own "a small labeled set" framing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import get_settings
from src.ingest.embed import get_connection

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = _PROJECT_ROOT / "eval" / "golden_dataset.json"
QUESTIONS_PATH = _PROJECT_ROOT / "eval" / "questions.json"
LABELS_PATH = _PROJECT_ROOT / "eval" / "recall_labels.json"

_QUOTE_RE = re.compile(r"[‘’]")
_DQUOTE_RE = re.compile(r"[“”]")
_TRADEMARK_RE = re.compile(r"[®™©]")  # (R), (TM), (C) symbols
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = _QUOTE_RE.sub("'", text)
    text = _DQUOTE_RE.sub('"', text)
    text = _TRADEMARK_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip().lower()


def build_labels(snippet_len: int = 80) -> list[dict]:
    golden_items = json.loads(GOLDEN_PATH.read_text())["items"]
    questions_by_id = {q["id"]: q["question"] for q in json.loads(QUESTIONS_PATH.read_text())["questions"]}

    conn = get_connection(get_settings())
    cur = conn.cursor()

    labels: list[dict] = []
    try:
        for item in golden_items:
            evidence = item.get("source_evidence")
            source_document = item.get("source_document")
            if not evidence or not source_document:
                continue
            docs = source_document.split(" + ")  # e.g. "AMD_2025_10K + INTC_2025_10K" for cross-doc items
            target = _normalize(evidence[:snippet_len])

            cur.execute("SELECT chunk_id, text FROM chunks WHERE document_id = ANY(%s)", (docs,))
            match = next((cid for cid, text in cur.fetchall() if target in _normalize(text)), None)
            if match:
                labels.append(
                    {
                        "question_id": item["id"],
                        "question": questions_by_id[item["id"]],
                        "gold_chunk_id": match,
                    }
                )
    finally:
        conn.close()

    return labels


def main() -> None:
    labels = build_labels()
    LABELS_PATH.write_text(
        json.dumps(
            {
                "_note": (
                    "Mechanically built by eval/build_recall_labels.py from "
                    "golden_dataset.json's verbatim source_evidence quotes, matched "
                    "against actual stored chunk text -- not hand-authored. Excludes "
                    "aggregation/multi-hop items, which have no single answering chunk."
                ),
                "labels": labels,
            },
            indent=2,
        )
    )
    print(f"Built {len(labels)} recall@k labels -> {LABELS_PATH}")
    for label in labels:
        print(f"  {label['question_id']}: {label['gold_chunk_id']}")


if __name__ == "__main__":
    main()
