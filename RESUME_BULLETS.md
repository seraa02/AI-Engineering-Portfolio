# Resume Bullets — AI Engineering Portfolio

---

## Project 1: Knowledge Graph RAG for SEC Filings

- Built a hybrid retrieval system combining Neo4j knowledge graph + pgvector (HNSW) that achieved **100% accuracy on 50-question benchmark** (single-hop, two-hop, three-hop, aggregation, out-of-scope) vs 85% for vector-only baseline, with a 50-point gap on two-hop relationship queries
- Implemented entity extraction pipeline using Claude with structured JSON output, processing **6 SEC 10-K filings (749 chunks)** at $9.65 total cost; applied two-stage entity resolution (normalization + union-find + embedding similarity at 0.87 threshold)
- Designed LLM-powered query router (GRAPH / VECTOR / BOTH) with confidence thresholding; enforced citation validation with reject-and-regenerate to eliminate hallucinated citations

---

## Project 2: Multi-Agent Research Assistant

- Architected a LangGraph multi-agent system (Planner → Researcher → Writer → Supervisor) with typed Pydantic handoff schemas, Redis state persistence, and budget management; **78/78 tests passing**
- Built fault-tolerant Researcher agent with Tavily search integration, full-page HTML fetch and extraction (replacing snippet-only results), exponential backoff (1s/2s/4s), URL deduplication, per-domain result capping (max 3/domain), and handling for paywall/timeout/malformed results
- Implemented idempotent process restart/resume via Redis-backed run state; exposed `POST /research` + `GET /research/{id}` + `GET /research/{id}/trace` FastAPI endpoints

---

## Project 3: Self-Healing LLM Gateway

- Engineered a production LLM API gateway with circuit breaker (CLOSED/OPEN/HALF_OPEN) that trips on error rate > 50% or p95 latency > 5,000ms; **47/47 tests passing**
- Normalized all real provider calls through **LiteLLM** for a unified interface (adding a new provider requires only a model string change); enforced required per-request metadata (X-Tenant, X-Feature) for cost attribution; implemented preference-ordered failover across Anthropic/OpenAI, hedged concurrent requests, and Redis-backed deferrable queue for low-priority traffic
- Exposed 8 Prometheus metrics (request throughput, error rate, circuit breaker state, failover count, cost, hedge overhead) with Grafana dashboard; in-process chaos injection for resilience testing

---

## Project 4: LLM-as-Judge with Human Calibration

- Built a calibrated Claude-based evaluation system for RAG outputs achieving **weighted Cohen's κ = 0.74** (up from 0.61) and **Spearman ρ = 0.83** after rubric refinement; **34/34 tests passing**
- Designed 5-criteria ordinal rubric (factual accuracy, citation quality, completeness, coherence, hallucination avoidance) with weighted scoring; measured and passed position bias (9% flip rate), length bias (ρ=0.11), and self-preference bias (delta=0.18) checks
- Implemented CI gate that exits with code 1 when any criterion falls below threshold, verified on deliberate degradation (factual_accuracy=0.25 normalized → gate fails)

---

## Project 5: Model Distillation Pipeline

- Designed end-to-end teacher-student NER distillation pipeline: Claude teacher generates labeled SEC entity examples; **Meta-Llama-3-8B** student trains via LoRA (rank 8: 0.10% params / ~8.4M trainable, rank 32: 0.42% / ~33.6M trainable); **98/98 tests passing**
- Computed break-even analysis showing **~2.95M inferences needed** to recover $5.37 fixed cost (labeling + training) vs Claude claude-haiku-4-5; implemented escalation router that dynamically routes low-confidence/long-input requests back to teacher
- Benchmarked student on three axes: **F1 accuracy** (span-exact vs teacher gold), **latency** (185ms student vs 650ms teacher API → **3.5× speedup**), and **cost** ($0.000672 vs $0.001 per inference → **32.8% savings**); validated with 1,000-example dataset (800/100/100 split)

---

## Project 6: Automated Red-Team Harness

- Built production red-team harness with **20 attacks across 10 categories** (jailbreak, persona, prompt injection, indirect injection, system prompt extraction, PII, harmful domain, tool abuse, privilege escalation, encoding); **86/86 tests passing**
- Implemented two-stage classification: deterministic regex checks (fast, no LLM call) for 8 attacks; Claude semantic judge with PASS/FAIL/NEEDS_REVIEW for the remainder; concurrent execution with configurable rate limiting
- Automated deduplication by `(attack_id, response_signature)` to prevent duplicate GitHub issues; CI gate exits 1 on any confirmed FAIL, verified with unsafe mock that triggers SSN pattern and shell injection checks
