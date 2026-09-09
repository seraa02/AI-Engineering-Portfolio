"""
Real, live-API bias measurement — Phase 4 of the PDF spec.

Runs the three bias diagnostics (position, length, self-preference) in src/bias.py against the
REAL Claude judge (src.judge.judge / judge_pairwise), not the mock. Uses a small sample (kept
small deliberately to bound API spend — the whole script costs well under $0.50 at Haiku
pricing) built from real questions + real source evidence in Project 1's golden dataset
(RAG/eval/questions.json, RAG/eval/golden_dataset.json), plus deliberately-constructed
Claude-style vs GPT-style answer variants for the self-preference check.

This directly replaces the prior state of this project, where bias.py's math was only ever
exercised in tests with hand-written stub judge_fn/score_fn lambdas — never against the live
judge. The numbers this script prints are real (n is small, so treat them as a live smoke test
of the bias-detection mechanism, not a publication-grade estimate — a proper measurement needs
a larger live sample, which costs proportionally more).

Usage:
    python scripts/run_live_bias_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import os
import anthropic

from src.bias import measure_length_bias, measure_position_bias, measure_self_preference_bias
from src.judge import judge, judge_pairwise

MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")


# ---------------------------------------------------------------------------
# Real questions + real source evidence, from Project 1's golden dataset
# ---------------------------------------------------------------------------
REAL_QA = [
    {
        "question": "Who does AMD compete with in the Data Center segment?",
        "evidence": "In the Data Center segment, we compete primarily against Intel Corporation "
                    "(Intel) and Nvidia Corporation (Nvidia) with our CPU, GPU DPU and AI NIC "
                    "server products.",
        "good_short": "AMD competes with Intel and NVIDIA in Data Center. [chunk_AMD_012]",
        "good_long": (
            "AMD's Data Center segment faces significant competitive pressure from two "
            "primary rivals: Intel Corporation, the long-established incumbent in server "
            "CPUs, and NVIDIA Corporation, the dominant force in data-center-scale GPU "
            "compute. AMD competes across its full product stack in this segment — CPUs, "
            "GPUs, DPUs (data processing units), and AI-focused network interface cards — "
            "positioning itself as a full-stack alternative to both companies rather than "
            "competing in a single product category alone. [chunk_AMD_012]"
        ),
    },
    {
        "question": "What company did NVIDIA acquire in 2020?",
        "evidence": "Our acquisition of Mellanox in 2020 expanded our offerings to include "
                    "networking, enabled our platforms to be data center scale, and led to "
                    "the introduction of a new processor class.",
        "good_short": "NVIDIA acquired Mellanox in 2020. [chunk_NVDA_007]",
        "good_long": (
            "NVIDIA completed its acquisition of Mellanox Technologies in 2020, a deal that "
            "significantly broadened the company's product portfolio beyond GPUs into "
            "high-performance networking. The acquisition enabled NVIDIA to offer "
            "data-center-scale platforms and directly led to the introduction of an "
            "entirely new processor class combining compute and networking capabilities "
            "on a single roadmap. [chunk_NVDA_007]"
        ),
    },
    {
        "question": "Is Sony a customer of AMD?",
        "evidence": "AMD semi-custom SoC products power the Sony PlayStation 5, the Microsoft "
                    "Xbox Series S and X game consoles, as well as the recently revealed "
                    "Valve Steam Machine PC.",
        "good_short": "Yes, Sony uses AMD semi-custom SoCs in the PS5. [chunk_AMD_045]",
        "good_long": (
            "Yes, Sony is a customer of AMD. AMD's semi-custom system-on-chip (SoC) "
            "products are used to power the Sony PlayStation 5 console, placing Sony "
            "alongside Microsoft (Xbox Series S/X) and Valve (Steam Machine PC) as major "
            "gaming-console customers relying on AMD's semi-custom silicon business. "
            "[chunk_AMD_045]"
        ),
    },
]


def run_length_bias():
    print("\n=== LENGTH BIAS (quality held constant, only length varies) ===")
    examples = []
    for qa in REAL_QA:
        examples.append({"question": qa["question"], "answer": qa["good_short"]})
        examples.append({"question": qa["question"], "answer": qa["good_long"]})

    def score_fn(question, answer):
        # find matching evidence for this question
        evidence = next(qa["evidence"] for qa in REAL_QA if qa["question"] == question)
        return judge(anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]),
                     question, answer, model=MODEL, context=evidence).weighted_score

    result = measure_length_bias(score_fn, examples)
    print(f"n_examples={result.n_examples}  spearman_rho={result.spearman_rho:.3f}  "
          f"concern={result.concern} (threshold |rho|>0.2)")
    return result


def run_position_bias():
    print("\n=== POSITION BIAS (same pair, both orders) ===")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    pairs = []
    for qa in REAL_QA:
        pairs.append({
            "question": qa["question"],
            "answer_a": qa["good_short"],
            "answer_b": "I don't know.",
        })
    # add a genuinely close pair (both good/plausible) where a flip is more likely
    pairs.append({
        "question": REAL_QA[0]["question"],
        "answer_a": REAL_QA[0]["good_short"],
        "answer_b": REAL_QA[0]["good_long"],
    })

    def judge_fn(question, answer_a, answer_b):
        return judge_pairwise(client, question, answer_a, answer_b, model=MODEL)

    result = measure_position_bias(judge_fn, pairs)
    print(f"n_pairs={result.n_pairs}  flip_rate={result.flip_rate:.3f}  "
          f"concern={result.concern} (threshold >0.15)")
    return result


def run_self_preference_bias():
    print("\n=== SELF-PREFERENCE BIAS (Claude judging Claude-style vs GPT-style phrasing) ===")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Same facts, same citation, deliberately different register/phrasing style.
    claude_style = [
        {"question": qa["question"],
         "answer": qa["good_short"]}
        for qa in REAL_QA
    ]
    gpt_style = [
        {"question": REAL_QA[0]["question"],
         "answer": "Great question! AMD's main competitors in the Data Center space are Intel "
                    "and NVIDIA — they go head-to-head across CPUs and GPUs. [chunk_AMD_012]"},
        {"question": REAL_QA[1]["question"],
         "answer": "Sure thing — back in 2020, NVIDIA picked up Mellanox, which was a big move "
                    "into networking for them! [chunk_NVDA_007]"},
        {"question": REAL_QA[2]["question"],
         "answer": "Yep! Sony's PS5 actually runs on a custom AMD chip, so yes, they're "
                    "definitely a customer. [chunk_AMD_045]"},
    ]

    def score_fn(question, answer):
        evidence = next((qa["evidence"] for qa in REAL_QA if qa["question"] == question), None)
        return judge(client, question, answer, model=MODEL, context=evidence).weighted_score

    result = measure_self_preference_bias(score_fn, claude_style, gpt_style)
    print(f"n_pairs={result.n_pairs}  avg_native(Claude-style)={result.avg_score_native:.3f}  "
          f"avg_other(GPT-style)={result.avg_score_other:.3f}  delta={result.delta:.3f}  "
          f"concern={result.concern} (threshold |delta|>0.3)")
    return result


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. This script makes real API calls.", file=sys.stderr)
        sys.exit(2)

    print(f"Running live bias checks with model={MODEL}. This costs a small amount of real API spend.")

    length_result = run_length_bias()
    position_result = run_position_bias()
    self_pref_result = run_self_preference_bias()

    summary = {
        "note": "Small live sample (n given per check) against the REAL Claude judge — a live "
                "smoke test of the bias-detection mechanism, not a publication-grade estimate.",
        "model": MODEL,
        "length_bias": {"n_examples": length_result.n_examples, "spearman_rho": round(length_result.spearman_rho, 4), "concern": length_result.concern},
        "position_bias": {"n_pairs": position_result.n_pairs, "flip_rate": round(position_result.flip_rate, 4), "concern": position_result.concern},
        "self_preference_bias": {"n_pairs": self_pref_result.n_pairs, "delta": round(self_pref_result.delta, 4), "concern": self_pref_result.concern},
    }
    out_path = Path(__file__).resolve().parent.parent / "datasets" / "live_bias_check_results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
