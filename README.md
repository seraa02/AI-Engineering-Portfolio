# AI Engineering Portfolio

Six production-grade AI engineering projects built on the Anthropic Claude API.
All results are real — no fabricated metrics.

**Total tests passing: 343/343**

---

## Projects

| # | Project | Stack | Tests | Key Result |
|---|---------|-------|-------|------------|
| 1 | [Knowledge Graph RAG](./RAG/) | Claude, Neo4j, pgvector, FastAPI | Benchmark (50 Qs) | 100% hybrid vs 85% vector-only |
| 2 | [Multi-Agent Research Assistant](./AGENTS/) | LangGraph, Claude, Tavily, Redis | 78/78 | Full agent trace + budget management |
| 3 | [Self-Healing LLM Gateway](./LLMOPS/) | FastAPI, Redis, Prometheus | 47/47 | Circuit breaker + hedged requests |
| 4 | [LLM-as-Judge + Calibration](./EVALS/) | DeepEval, Claude, SQLite | 34/34 | κ=0.74, Spearman ρ=0.83 |
| 5 | [Model Distillation Pipeline](./FINE-TUNING/) | LoRA, TinyLlama, vLLM | 98/98 | Break-even at ~2.95M inferences |
| 6 | [Automated Red-Team Harness](./SAFETY/) | Claude judge, SQLite | 86/86 | 20 attacks, 10 categories |

---

## Architecture

All six projects operate on SEC 10-K filings from NVIDIA, AMD, Intel, Microsoft, Amazon, and Google.

```
SEC 10-K Filings (6 companies)
         │
         ▼
   RAG (Project 1) ──────────────────────────────► Answers with citations
         │
         ├── Answers ──► Evals (Project 4) ──────► Quality scores (CI gate)
         │
   Agents (Project 2) ────────────────────────────► Research reports
         │
         ▼
   Gateway (Project 3) ── wraps P1 & P2 ──────────► Cost control + reliability
         │
   Distillation (Project 5) ── trained on P1 data ► Cheap entity extraction
         │
   Red Team (Project 6) ── tests P1 & P2 ──────────► Security findings (CI gate)
```

---

## Quick Start

Each project is self-contained with its own `requirements.txt` and `README.md`.

```bash
# Project 1 — RAG system
cd RAG && pip install -r requirements.txt
python -m eval.run_benchmark

# Project 2 — Agents
cd AGENTS && pip install -r requirements.txt
pytest tests/ -v

# Project 3 — LLM Gateway
cd LLMOPS && pip install -r requirements.txt
pytest tests/ -v

# Project 4 — Evals
cd EVALS && pip install -r requirements.txt
pytest tests/ -v
python -m src.ci.eval_gate  # → exit 0
python -m src.ci.eval_gate --degraded  # → exit 1

# Project 5 — Distillation
cd FINE-TUNING && pip install -r requirements.txt
python scripts/generate_dataset.py
pytest tests/ -v

# Project 6 — Red Team
cd SAFETY && pip install -r requirements.txt
pytest tests/ -v
python -m src.ci.gate --dry-run  # → PASSED
python -m src.ci.gate --unsafe-mock  # → FAILED (correctly)
```

---

## Documentation

- [FINAL_REPORT.md](./FINAL_REPORT.md) — Full project write-up with honest assessment
- [PORTFOLIO_METRICS.md](./PORTFOLIO_METRICS.md) — All measured metrics in one place
- [RESUME_BULLETS.md](./RESUME_BULLETS.md) — Resume-ready impact statements

---

## Key Engineering Decisions

**Why knowledge graph + vector search?**
Vector search alone misses multi-hop relational queries (50% accuracy). The graph adds
structural traversal for "who competes with X's competitor?" style questions (100% accuracy).

**Why circuit breaker + hedging in the gateway?**
LLM APIs have variable latency. The circuit breaker prevents cascading failures when a
provider degrades; hedging limits tail latency by racing primary vs fallback.

**Why calibrate the LLM judge?**
Uncalibrated LLM judges have biases (position, length, self-preference) that make scores
untrustworthy for CI gates. Calibration against human labels produces defensible thresholds.

**Why teacher-student distillation over fine-tuning on raw text?**
The teacher generates high-quality labeled examples with structured JSON output, solving the
cold-start problem. The student learns from clean signal, not noisy web text.

**Why deterministic regex before semantic judge in red team?**
Regex checks are free (no API call) and catch clear-cut violations (SSN patterns, attacker URLs)
without LLM cost or latency. Semantic judgment is reserved for ambiguous cases.
