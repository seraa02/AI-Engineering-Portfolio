# Automated Red-Team Harness

**86/86 tests passing.**

A production-grade automated red-team framework for LLM-powered systems. Runs structured adversarial attacks against the Project 1 RAG system and Project 2 multi-agent research assistant, classifies outcomes, deduplicates findings, and files GitHub issues for confirmed failures.

---

## Architecture

```
src/
├── attacks/
│   └── taxonomy.py          ← 20 attacks × 10 categories, severity, regex checks
├── harness/
│   └── runner.py            ← Concurrent runner with rate limiting + judge pipeline
├── adapters/
│   ├── rag_adapter.py       ← HTTP adapter for Project 1 /ask endpoint
│   └── agent_adapter.py     ← HTTP adapter for Project 2 /research endpoint
├── judge/
│   └── claude_judge.py      ← Claude-based semantic evaluator (LLM-as-Judge)
├── reporting/
│   ├── findings.py          ← SQLite storage + GitHub issue automation + dedup
│   └── dashboard.py         ← Metrics: by-category, by-severity, regression detection
└── ci/
    └── gate.py              ← CI gate (exit 0 = pass, exit 1 = fail)
```

---

## Attack Taxonomy

| Category | Count | Severity | Target |
|----------|-------|----------|--------|
| jailbreak | 3 | HIGH/MEDIUM | both |
| persona | 2 | MEDIUM | rag |
| prompt_injection | 2 | HIGH | both/agent |
| indirect_injection | 2 | HIGH/CRITICAL | rag |
| system_prompt_extraction | 3 | HIGH/MEDIUM | both |
| pii_extraction | 2 | HIGH/MEDIUM | rag |
| harmful_domain | 2 | HIGH | rag |
| tool_abuse | 2 | HIGH/CRITICAL | agent |
| privilege_escalation | 2 | HIGH/MEDIUM | both |
| encoding | 2 | HIGH/MEDIUM | both |

**Total: 20 attacks.** 8 have deterministic regex checks (fast, no LLM call needed).
12 fall back to Claude-based semantic judgment or NEEDS_REVIEW.

---

## Evaluation Pipeline

```
Attack payload
     │
     ▼
  target_fn(payload) ──► response
     │
     ▼
deterministic_check(regex) ──► FAIL (fast, no LLM)
     │ (None = inconclusive)
     ▼
claude_judge(attack, response) ──► PASS | FAIL | NEEDS_REVIEW
     │
     ▼
Store finding (SQLite, deduplicated by attack_id × response_signature)
     │
     ▼
File GitHub issue (if FAIL + repo configured)
```

---

## Results

**CI gate verification:**

```bash
# Safe mock (should PASS)
python -m src.ci.gate --target both
# → [CI GATE] PASSED — 20/20 attacks passed, 0 needs review, 0 confirmed failures.

# Unsafe mock (should FAIL — tests the gate itself)
python -m src.ci.gate --target rag --unsafe-mock
# → [CI GATE] FAILED: FAIL count N exceeds threshold 0
```

**Test coverage: 86/86 tests passing** across taxonomy, runner, adapters, findings, dashboard, and CI gate.

---

## Deduplication

Findings are keyed by `(attack_id, sha256(response)[:16])`. The same vulnerability
manifesting with the same response is filed once, even across multiple runs.
This prevents GitHub issue spam for persistent failures.

---

## Setup

```bash
pip install -r requirements.txt

# Run full battery in dry-run mode (no external calls)
python -m src.ci.gate --dry-run

# Run against real RAG service
RAG_BASE_URL=http://localhost:8000 python -m src.ci.gate --target rag

# Run against real agent service
AGENT_BASE_URL=http://localhost:8001 python -m src.ci.gate --target agent

# Run all tests
pytest tests/ -v
```

---

## CI Integration

```yaml
# .github/workflows/safety.yml
- name: Red-team gate
  run: python -m src.ci.gate --target both
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    RAG_BASE_URL: ${{ secrets.RAG_BASE_URL }}
    AGENT_BASE_URL: ${{ secrets.AGENT_BASE_URL }}
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    GITHUB_REPO: owner/repo
```

Gate exits 1 (failing the PR) if **any** attack produces a confirmed FAIL.

---

## Known Limitations

- The 12 non-deterministic attacks fall back to `NEEDS_REVIEW` without a live Anthropic API key. In production, `ANTHROPIC_API_KEY` must be set for the Claude judge to run.
- Attack payloads are targeted at the financial/SEC domain (Projects 1 & 2). A production deployment would extend the battery for the specific application domain.
- The `agent_adapter` polls `/research/{id}` synchronously. High-concurrency runs against the agent should reduce `max_concurrency` to avoid overwhelming the agent's Redis queue.
