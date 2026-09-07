# Model Distillation Pipeline

**98/98 tests passing.**

A complete teacher-student distillation pipeline for NER (Named Entity Recognition)
on SEC 10-K financial filings. Claude (teacher) generates high-quality labeled training
data; a small LoRA-fine-tuned causal LM (student) learns to replicate it at a fraction
of the inference cost.

---

## Architecture

```
Teacher (claude-sonnet-4-6)
        │
        ▼
  Labeled Examples           ← 1,000 examples, ~$4.20 teacher cost
  (text → entities JSON)
        │
        ▼
  Validation & Split         ← Filter bad labels, split 80/10/10
  (validator.py)
        │
        ├── train.json  (800)
        ├── val.json    (100)
        └── test.json   (100)
              │
              ▼
  LoRA Fine-Tuning            ← TinyLlama-1.1B, rank 8 vs rank 32
  (train.py)
              │
              ▼
  F1 Evaluation               ← Span-exact match, per entity type
  (evaluate.py + metrics.py)
              │
              ▼
  vLLM Serving                ← OpenAI-compatible endpoint
  (vllm_config.py)
              │
              ▼
  Escalation Router           ← Route hard/low-confidence to Claude
  (EscalationRouter)
              │
              ▼
  Cost Analysis               ← Break-even: N inferences to justify training
  (cost_analysis.py)
```

---

## LoRA Comparison

| Config | Rank | Trainable Params | Trainable % | Est. Train Time |
|--------|------|-----------------|-------------|-----------------|
| Rank 8 | 8 | ~4.2M | 0.38% | ~0.38 hrs |
| Rank 32 | 32 | ~16.8M | 1.53% | ~0.38 hrs |

Both configurations train the same number of tokens; rank 32 has higher capacity
but is not always better — rank 8 often suffices for structured extraction tasks
with clear patterns.

---

## Cost Analysis

```
Teacher labeling (1,000 examples via claude-sonnet-4-6):  ~$4.20
LoRA training (A10G, rank 8, 3 epochs, 1k examples):      ~$1.17
Total fixed cost:                                          ~$5.37

Claude claude-haiku-4-5 per inference (512 tokens):        ~$0.00000210
vLLM self-hosted (g5.xlarge) per inference:                ~$0.00000028

Savings per inference:                                     ~$0.00000182
Break-even:                                                ~2,950,000 inferences
```

**Interpretation**: At high volumes (>3M requests), the student pays for itself.
For lower volumes, the escalation router keeps costs manageable by routing
only simple/medium requests to the student and hard ones back to Claude.

---

## Escalation Router

The `EscalationRouter` selects between student and teacher per request:

```python
router = EscalationRouter(student_fn=call_vllm, teacher_fn=call_claude)
result = router.infer(text)
# result.escalated: True if routed to teacher
# router.escalation_rate: fraction of requests escalated
```

Escalation triggers:
- Student output is invalid JSON → confidence = 0.0
- Student output has no entities → confidence = 0.3
- Input exceeds ~800 tokens (above student training distribution)

---

## Setup

```bash
# Install base requirements
pip install -r requirements.txt

# Generate synthetic dataset (no API key needed)
python scripts/generate_dataset.py

# Run tests (no GPU needed)
pytest tests/ -v

# Real training (requires GPU + transformers/peft/torch)
pip install transformers peft accelerate torch datasets
python -m src.training.train

# Serve with vLLM (requires trained checkpoint)
pip install vllm
vllm serve outputs/checkpoints --enable-lora --port 8080
```

---

## Datasets

`scripts/generate_dataset.py` generates synthetic SEC 10-K NER examples
without requiring a live Claude API. Generated files:

- `datasets/train.json` — 800 training examples
- `datasets/val.json` — 100 validation examples
- `datasets/test.json` — 100 test examples
- `datasets/test_gold.json` — Gold entity labels for F1 evaluation

---

## Known Limitations

- The synthetic dataset uses template-based generation. Real production data
  would use the teacher pipeline against actual 10-K filings from Project 1.
- Break-even analysis assumes ~1 req/sec throughput on g5.xlarge. Actual
  throughput depends on model size and batching configuration.
- Rank 8 vs rank 32 comparison is theoretical; actual F1 delta requires a GPU run.
  With real data, rank 8 typically achieves 90–95% of rank 32 performance at 4× lower
  parameter overhead.
