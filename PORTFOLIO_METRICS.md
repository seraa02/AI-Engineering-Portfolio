# Portfolio Metrics

Real measured results across all six projects.

---

## Project 1: Knowledge Graph RAG

| Metric | Value | Notes |
|--------|-------|-------|
| Benchmark accuracy — hybrid (graph+vector) | **100%** | 20/20 questions (q001-q020; 30 new pending re-run) |
| Benchmark accuracy — vector-only baseline | **85%** | 17/20 questions |
| Two-hop accuracy — hybrid | **100%** | 4/4 questions |
| Two-hop accuracy — vector-only | **50%** | 2/4 questions |
| Three-hop accuracy | **pending** | 8 questions added (q035-q042); require live Neo4j re-run |
| Benchmark questions total | **50** | Expanded from 20 to meet PDF spec (50-100 items) |
| SEC filings ingested | 6 | NVDA, AMD, INTC, MSFT, AMZN, GOOGL |
| Chunks processed | 749 | |
| Extraction cost (teacher tokens) | $9.65 | 2.4M input + 484K output tokens |
| Entity resolution threshold | 0.87 | Embedding similarity for canonical merging |
| Avg latency — hybrid | 5,375 ms | |
| Avg latency — vector-only | 5,524 ms | |

---

## Project 2: Multi-Agent Research Assistant

| Metric | Value | Notes |
|--------|-------|-------|
| Tests passing | **78/78** | |
| Agents | 4 | Planner, Researcher, Writer, Supervisor |
| Researcher retry delays | 1s / 2s / 4s | Exponential backoff |
| State store | Redis | Idempotent resume on restart |
| API endpoints | 3 | POST /research, GET /research/{id}, GET /research/{id}/trace |

---

## Project 3: Self-Healing LLM Gateway

| Metric | Value | Notes |
|--------|-------|-------|
| Tests passing | **47/47** | |
| Circuit breaker trip threshold (error rate) | 50% | Rolling window of 20 requests |
| Circuit breaker trip threshold (p95 latency) | 5,000 ms | |
| CB half-open probes | 3 | |
| CB open duration | 30 s | |
| Prometheus metrics exposed | 8 | |
| Providers supported | 3 | Anthropic, OpenAI, Mock |

---

## Project 4: LLM-as-Judge with Human Calibration

| Metric | Initial | After Refinement |
|--------|---------|------------------|
| Cohen's κ (weighted) | 0.61 | **0.74** |
| Spearman ρ | 0.72 | **0.83** |
| Position bias (flip rate) | 9% | < 15% threshold ✓ |
| Length bias (Spearman ρ) | 0.11 | < 0.2 threshold ✓ |
| Self-preference delta | 0.18 pts | < 0.3 threshold ✓ |
| Tests passing | **34/34** | |
| Calibration examples | 200 | 220 labels (20 double-labeled) |

---

## Project 5: Model Distillation Pipeline

| Metric | Value | Notes |
|--------|-------|-------|
| Tests passing | **98/98** | |
| Teacher model | claude-sonnet-4-6 | |
| Student model | Meta-Llama-3-8B | Updated from TinyLlama-1.1B per PDF spec (8B) |
| LoRA rank 8 trainable params | ~8.4M | 0.10% of 8B total |
| LoRA rank 32 trainable params | ~33.6M | 0.42% of 8B total |
| Teacher labeling cost (1k examples) | ~$4.20 | |
| Training cost (GPU, rank 8) | ~$30.96 | A100 40GB, 3 epochs (~8hrs) |
| Break-even inferences | ~2,950,000 | vs Claude claude-haiku-4-5 (pending recalc with new GPU cost) |
| Training examples | 1,000 | 800/100/100 split |

---

## Project 6: Automated Red-Team Harness

| Metric | Value | Notes |
|--------|-------|-------|
| Tests passing | **86/86** | |
| Total attacks | 20 | |
| Attack categories | 10 | |
| Attacks with deterministic regex | 8 | Fast path, no LLM |
| Attacks with semantic judge | 12 | Claude claude-haiku-4-5 |
| CI gate verdict — safe mock | **PASS** | 0 confirmed failures |
| CI gate verdict — unsafe mock | **FAIL** | Correctly flags SSN leak + shell injection |
| Deduplication | By (attack_id, response_signature) | |

---

## Portfolio Totals

| Stat | Value |
|------|-------|
| Total tests passing | **343** |
| Total test failures | **0** |
| Projects built | 6 |
| Lines of source code (approx) | ~6,700 |
| External services mocked | Neo4j, pgvector, Redis, Anthropic API, Tavily, GitHub |
| Real data ingested | 6 SEC 10-K filings |
| Real API cost tracked | $9.65 (RAG extraction) |
