# Project 1 Acceptance Checklist

**Project:** Knowledge Graph RAG for Enterprise Data
**Corpus:** 6 SEC 10-K filings (NVDA, AMD, INTC, MSFT, AMZN, GOOGL) — FY2025/2026
**Audited:** 2026-09-07
**Auditor:** Autonomous audit against PDF spec "BASWE — Build These Six Projects"

---

## Legend

| Status | Meaning |
|--------|---------|
| PASS | Requirement implemented and verified against the actual codebase/data |
| PARTIAL | Requirement implemented but not fully to spec — see explanation |
| FAIL | Requirement not implemented |
| N/A | Not applicable to this corpus or configuration |

---

## Phase 1 — Knowledge Graph

| # | Requirement | Status | Evidence / Notes |
|---|-------------|--------|-----------------|
| 1 | Uses a corpus with meaningful relationships | PASS | 6 tech company 10-Ks with cross-document competition, acquisition, supply, customer, regulatory relationships. See `data/raw/manifest.json`. |
| 2 | Uses an explicitly constrained ontology | PASS | `config/ontology.yaml` — status=DEFINED, fully documented with evidence quotes from the corpus |
| 3 | ~5–10 entity types | PASS | **5 entity types:** Company, Product, BusinessSegment, RegulatoryBody, Person |
| 4 | ~8–15 relationship types | PASS | **12 relationship types:** COMPETES_WITH, ACQUIRED, SUBSIDIARY_OF, SUPPLIES, CUSTOMER_OF, PARTNERS_WITH, DEPENDS_ON, REGULATED_BY, OFFERS_PRODUCT, PART_OF_SEGMENT, HAS_SEGMENT, OFFICER_OF |
| 5 | Chunks documents | PASS | `src/ingest/chunk.py` — 749 chunks across 6 documents. Shared chunk_id between graph and vector pipelines. |
| 6 | Uses Claude for structured entity/relationship extraction | PASS | `src/ingest/extract.py` — uses `client.messages.create` with `output_config.format.type=json_schema` |
| 7 | Strict extraction schema: entity_type, canonical_name, relationship_type, source_chunk_id, confidence | PASS | Schema enforced via JSON Schema in `Ontology.json_schema()`. `source_chunk_id` set by pipeline (not trusted from model). `canonical_name` set by `resolve.py` after extraction. All 5 fields present in `ExtractedEntity` / `ExtractedRelationship` Pydantic models. |
| 8 | Validates extraction output | PASS | Pydantic validation after JSON parse in `_parse_and_validate()`. Dangling relationship endpoints dropped. |
| 9 | Retries invalid extraction responses | PASS | `MAX_RETRIES = 2` in `extract.py`. Double max_tokens on truncation. |
| 10 | Performs entity resolution | PASS | `src/ingest/resolve.py` — two-stage (normalization + embedding similarity) union-find clustering |
| 11 | Handles aliases (Acme Corp / Acme Corporation / ACME) | PASS | `normalize_name()` strips corporate suffixes; seed_groups force-merge known abbreviations (AMD ↔ Advanced Micro Devices). Aliases stored on graph nodes. |
| 12 | Normalization + embedding similarity with tuned threshold | PARTIAL | Threshold=0.87 chosen empirically (verified 0.75 causes over-merging). Not calibrated against a labeled alias set — documented as known limitation in README. |
| 13 | Aliases stored on nodes | PASS | `_MERGE_NODE_QUERY` uses `ON CREATE SET n.aliases / ON MATCH SET ... acc + a` — aliases accumulate idempotently. |
| 14 | MERGE not CREATE for idempotent ingestion | PASS | `graph_writer.py` uses `MERGE` for both nodes and relationships. Uniqueness constraints created per label. |
| 15 | Source chunk IDs on graph relationships | PASS | `r.source_chunk_ids` list property on every relationship. New chunk_id appended, never duplicated (coalesce check). |
| 16 | Graph relationships traceable to source text | PASS | `r.source_chunk_ids` → pgvector `chunks.chunk_id` → `chunks.text`. Shared key verified by design in embed.py. |
| 17 | Budgets extraction cost | PASS | Token counts tracked per-chunk, per-document, and corpus-wide. Total: 2,400,196 input / 484,962 output = **$9.65** one-time cost at Sonnet 5 pricing. |
| 18 | Caches extraction results using document hashes | PASS | Cache key = `sha256(chunk_id | model | prompt_version | schema_version | text)`. Changing the prompt or ontology invalidates exactly the right entries. 750 entries in `data/cache/extraction/`. |

**Phase 1 overall: 17 PASS, 1 PARTIAL**

---

## Phase 2 — Vector Index

| # | Requirement | Status | Evidence / Notes |
|---|-------------|--------|-----------------|
| 1 | Same document chunks used for graph are embedded | PASS | Both pipelines consume `chunk_document(doc)` from `src/ingest/chunk.py`. Same `chunk_id` used as primary key in pgvector and as `source_chunk_ids` on graph relationships. |
| 2 | pgvector used | PASS | `src/ingest/embed.py` — psycopg + pgvector, `chunks` table with `vector(1024)` column |
| 3 | HNSW indexing | PASS | `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` in `embed.py` |
| 4 | Metadata: document_id, section_path, date, entity_ids | PARTIAL | document_id, section_path, company, ticker, document_type, filing_date all stored. **entity_ids not stored** — the entity link is via chunk_id (retrievable from graph), not a direct column in pgvector. |
| 5 | Graph and vector stores share same chunk IDs | PASS | `chunk_id` is the explicit shared key documented in schema.py's docstring. |
| 6 | Retrieval recall@k measured on labeled dataset | PASS | `eval/recall_at_k.py` — 11 labeled (question, gold_chunk_id) pairs. Results in `eval/recall_results.json`. |
| 7 | ef_search tuned/measured | PASS | Sweep over ef_search ∈ {10, 40, 100, 200} — no difference at 749 chunks. DEFAULT_EF_SEARCH=40 documented with rationale. |

**Phase 2 overall: 6 PASS, 1 PARTIAL**

---

## Phase 3 — Retrieval Router

| # | Requirement | Status | Evidence / Notes |
|---|-------------|--------|-----------------|
| 1 | A routing mechanism exists | PASS | `src/retrieval/router.py` — `classify_route()` function |
| 2 | Router output is an explicit enum | PASS | `Route = Literal["GRAPH", "VECTOR", "BOTH"]` + JSON Schema `"enum": ["GRAPH", "VECTOR", "BOTH"]` |
| 3 | Low-confidence routing falls back to BOTH | PASS | `CONFIDENCE_THRESHOLD = 0.6`; below threshold, route forced to BOTH and `forced_both_low_confidence=True` logged |
| 4 | Graph retrieval: connection, multi-hop, comparisons, aggregations | PASS | `query_type` enum: one_hop, two_hop_chain, path_between, aggregate_count. Examples in router prompt. |
| 5 | Vector retrieval: definitions, policy lookups, single-fact | PASS | VECTOR route + few-shot examples in system prompt |
| 6 | Entities from questions resolved to graph node IDs | PASS | `resolve_entity_name()` in `graph_query.py` — exact canonical_name match, alias match, normalized-form fallback |
| 7 | Cypher is parameterized | PASS | All templates use `$name`, `$name_a`, `$name_b` bound parameters. Only relationship_type (validated enum) is string-interpolated. |
| 8 | LLM never generates raw Cypher | PASS | LLM outputs only `query_type` enum + `entity_names` + `relationship_type` + `max_hops`. Templates are fixed. Docstring in `graph_query.py` makes this a documented security boundary. |
| 9 | Query templates used | PASS | `_ONE_HOP_ALL`, `_ONE_HOP_TYPED`, `_TWO_HOP_CHAIN`, `_PATH_BETWEEN`, `_AGGREGATE_COUNT` |
| 10 | Routing decisions logged | PASS | `ROUTING_LOG_PATH` = `data/cache/routing_log.jsonl` — every decision written with question, route, entity_names, query_plan, confidence, latency_ms, timestamp |

**Phase 3 overall: 10 PASS**

---

## Phase 4 — Grounded Answer

| # | Requirement | Status | Evidence / Notes |
|---|-------------|--------|-----------------|
| 1 | Graph traversal produces readable statements | PASS | `graph_rows_to_statements()` in `merge.py` — converts raw Cypher rows to prose sentences using `_RELATIONSHIP_PHRASING` dictionary |
| 2 | Vector retrieval produces passages | PASS | `vector_search()` returns `VectorResult` with text field; passed as passages to build_answer |
| 3 | Results are deduplicated | PASS | Graph statements and vector passages assembled into distinct context sections; duplicate chunk_ids in `retrieved_chunk_ids` set are naturally deduplicated |
| 4 | Graph-derived facts and retrieved passages explicitly distinguished | PASS | Context built with `=== GRAPH FACTS ===` and `=== VECTOR PASSAGES ===` section headers |
| 5 | Every answer claim has a citation | PASS | Structured output schema forces every claim to include `chunk_id` field |
| 6 | Every citation maps to actually retrieved chunk_id | PASS | `retrieved_chunk_ids` set validated mechanically; claims citing unretrieved IDs → `rejected` list |
| 7 | Invalid citations cause regeneration/rejection | PASS | `for attempt in range(2)` — on first rejection, second attempt with explicit correction listing invalid IDs |
| 8 | Hallucinated/nonexistent sources cannot be returned | PASS | Claim validation is mechanical (set membership), not a trust-the-model check. Rejected claims counted in `rejected_claim_count` in API response. |

**Phase 4 overall: 8 PASS**

---

## Phase 5 — Benchmark

| # | Requirement | Status | Evidence / Notes |
|---|-------------|--------|-----------------|
| 1 | ~50–100 benchmark questions | PARTIAL | **20 questions** (PDF recommends 50-100). Development-tier golden set covers all required difficulty categories. Acknowledged in README as known limitation. |
| 2 | Covers single-hop questions | PASS | 13 single-hop questions (q001–q011, q017, q018) |
| 3 | Covers two-hop questions | PASS | 2 two-hop questions (q019, q020) |
| 4 | Covers three-hop questions | FAIL | No three-hop questions in current dataset. The corpus (6 documents, 749 chunks) has limited three-hop graph paths. |
| 5 | Covers aggregation questions | PASS | 2 aggregation questions (q012, q013) |
| 6 | Covers out-of-scope/refusal | PASS | 3 out-of-scope questions (q014, q015, q016) |
| 7 | Hybrid vs. plain vector baseline | PASS | `run_benchmark.py` runs both systems on identical questions and corpus |
| 8 | Overall accuracy measured | PASS | Hybrid: 100% (20/20), Vector-only: 85% (17/20) |
| 9 | Accuracy by hop count | PASS | Broken down by single_hop, two_hop, aggregation, out_of_scope |
| 10 | Latency measured | PASS | Per-query latency tracked; averages: hybrid=5,375ms, vector-only=5,524ms |
| 11 | p50 latency | FAIL | Only average latency reported. p50/p95 not computed. |
| 12 | p95 latency | FAIL | Only average latency reported. p95 not computed. |
| 13 | Cost/query | PARTIAL | One-time ingestion cost reported ($9.65 total). Per-query cost not computed separately. |
| 14 | Graph ingestion cost | PASS | $9.65 (2,400,196 input + 484,962 output tokens at Sonnet 5 pricing) |
| 15 | Retrieval recall@k | PASS | Measured: recall@5 = 45% at ef_search=40 on 11-item labeled set |
| 16 | Accuracy delta by hop count is the key result | PASS | Single-hop: hybrid=100% vs vector=85%; Two-hop: hybrid=100% vs vector=50%. Gap grows with hop complexity as expected. |

**Phase 5 overall: 9 PASS, 2 PARTIAL, 4 FAIL**

---

## Final Acceptance Criterion

| Requirement | Status | Evidence |
|-------------|--------|---------|
| FastAPI endpoint answers questions with validated citations | PASS | `src/api/main.py` — `POST /ask` returns `AskResponse` with `claims[]` (each with `chunk_id`), `rejected_claim_count`, route info |
| README opens with benchmark table comparing system to vector-only baseline by hop count | PASS | `README.md` lines 8-24 — benchmark table is the first content section after the intro paragraph |

---

## Summary

| Phase | PASS | PARTIAL | FAIL |
|-------|------|---------|------|
| Phase 1 — Knowledge Graph | 17 | 1 | 0 |
| Phase 2 — Vector Index | 6 | 1 | 0 |
| Phase 3 — Retrieval Router | 10 | 0 | 0 |
| Phase 4 — Grounded Answer | 8 | 0 | 0 |
| Phase 5 — Benchmark | 9 | 2 | 4 |
| **Total** | **50** | **4** | **4** |

---

## Outstanding Items

### PARTIAL items

**P2-4: entity_ids not stored in pgvector metadata**
The `chunks` table stores company/ticker/document_type/filing_date but not a list of entity_ids extracted from each chunk. The entity→chunk link is traversable via the graph (relationship.source_chunk_ids), but a direct entity_ids column in pgvector is missing. Low-priority: the shared chunk_id already enables bidirectional lookup.

**P1-12: Embedding similarity threshold not calibrated against labeled set**
Threshold=0.87 was chosen empirically (0.75 causes over-merging on this corpus, documented with measurement). A proper labeled alias dataset would allow a precision-recall curve. Left as a stretch goal.

**P5-1: 20 questions instead of 50-100**
The golden set has 20 questions covering all required difficulty categories. Expanding to 50-100 would require additional human annotation of multi-hop questions, which requires the graph to be built and queryable. All 20 items were manually verified against real filing text.

**P5-13: Per-query cost not separately computed**
The one-time ingestion cost ($9.65) is measured precisely. Per-query cost requires logging Anthropic token usage per API call at query time — the infrastructure (usage tracking in extraction) exists but was not wired into the benchmark runner.

### FAIL items

**P5-4: No three-hop questions**
The current corpus of 6 documents does not reliably produce three-hop graph paths (A→B→C→D) that are verifiable against filing text. Three-hop questions require at minimum three independently documented relationships in the filing corpus, which the current 6-company graph has limited examples of. This would require either expanding the corpus or crafting artificial three-hop paths (rejected — the guide says not to fabricate).

**P5-11/12: p50/p95 latency not computed**
The benchmark records per-question latency but aggregates to average only. Adding percentile computation to `run_benchmark.py` is straightforward but requires a re-run of the benchmark (which incurs API cost).

---

## Recommendation

Project 1 satisfies the PDF's functional requirements completely. The gaps are measurement depth (p50/p95 latency, per-query cost, three-hop questions), not missing architectural components. All five phases are implemented and working against the real SEC corpus.

**Project 1 is COMPLETE at the functional level. Measurement depth is PARTIAL.**
