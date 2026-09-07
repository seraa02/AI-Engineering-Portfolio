"""
Cost analysis for the distillation pipeline.

Computes:
- Teacher labeling cost (Claude API calls to generate training data)
- Student training cost (GPU hours for LoRA fine-tuning)
- Serving cost comparison (Claude API vs vLLM self-hosted)
- Break-even analysis: how many inferences justify fine-tuning?
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Pricing constants (as of 2025 — adjust as needed)
# ---------------------------------------------------------------------------

# Claude teacher (claude-sonnet-4-6)
TEACHER_COST_PER_INPUT_TOKEN = 3.0 / 1_000_000   # $3 / MTok
TEACHER_COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000  # $15 / MTok

# Claude claude-haiku-4-5 (cheaper production option)
HAIKU_COST_PER_INPUT_TOKEN = 0.80 / 1_000_000
HAIKU_COST_PER_OUTPUT_TOKEN = 4.0 / 1_000_000

# GPU costs (AWS p3.2xlarge ~$3.06/hr, A100 ~$2.21/hr spot)
GPU_COST_PER_HOUR = 3.06

# vLLM serving (self-hosted on g5.xlarge ~$1.006/hr)
VLLM_INSTANCE_COST_PER_HOUR = 1.006
VLLM_REQUESTS_PER_HOUR = 3600  # ~1 req/sec throughput for small model


@dataclass
class TeacherCostReport:
    n_examples: int
    total_input_tokens: int
    total_output_tokens: int
    cost_usd: float
    avg_cost_per_example: float


@dataclass
class TrainingCostReport:
    model_name: str
    lora_rank: int
    training_hours: float
    gpu_cost_usd: float
    trainable_params: int
    total_params: int
    trainable_pct: float


@dataclass
class BreakEvenReport:
    teacher_label_cost: float
    training_cost: float
    total_fixed_cost: float
    claude_cost_per_inference: float
    student_cost_per_inference: float
    savings_per_inference: float
    break_even_n_inferences: int
    note: str


def compute_teacher_cost(examples: list) -> TeacherCostReport:
    """Aggregate teacher labeling costs from example list."""
    total_input = sum(getattr(ex, "token_count", 0) // 2 for ex in examples)
    total_output = sum(getattr(ex, "token_count", 0) // 2 for ex in examples)
    total_cost = sum(getattr(ex, "teacher_cost_usd", 0.0) for ex in examples)

    return TeacherCostReport(
        n_examples=len(examples),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        cost_usd=total_cost,
        avg_cost_per_example=total_cost / len(examples) if examples else 0.0,
    )


def estimate_training_cost(
    model_name: str = "TinyLlama-1.1B",
    lora_rank: int = 8,
    n_examples: int = 1000,
    epochs: int = 3,
    gpu_cost_per_hour: float = GPU_COST_PER_HOUR,
) -> TrainingCostReport:
    """
    Estimate LoRA fine-tuning cost.

    Rough throughput: ~500 tokens/sec on A10G for a 1B model.
    Average example: ~256 tokens.
    """
    avg_tokens_per_example = 256
    tokens_per_second = 500

    total_tokens = n_examples * epochs * avg_tokens_per_example
    training_seconds = total_tokens / tokens_per_second
    training_hours = training_seconds / 3600

    gpu_cost = training_hours * gpu_cost_per_hour

    # LoRA param estimates for a 1.1B model
    if lora_rank == 8:
        trainable_params = 4_194_304   # ~4M params
        total_params = 1_100_000_000
    elif lora_rank == 32:
        trainable_params = 16_777_216  # ~16M params
        total_params = 1_100_000_000
    else:
        trainable_params = lora_rank * 524_288
        total_params = 1_100_000_000

    return TrainingCostReport(
        model_name=model_name,
        lora_rank=lora_rank,
        training_hours=training_hours,
        gpu_cost_usd=gpu_cost,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=100.0 * trainable_params / total_params,
    )


def compute_break_even(
    teacher_label_cost: float,
    training_cost: float,
    avg_tokens_per_inference: int = 512,
    student_instance_cost_per_hour: float = VLLM_INSTANCE_COST_PER_HOUR,
    student_requests_per_hour: int = VLLM_REQUESTS_PER_HOUR,
    teacher_model: str = "haiku",
) -> BreakEvenReport:
    """
    Compute break-even point: how many inferences to recover fixed costs.

    Fixed costs = labeling + training.
    Variable saving per inference = (Claude cost) - (vLLM self-hosted cost).
    """
    # Claude inference cost per request
    if teacher_model == "haiku":
        claude_cost = (avg_tokens_per_inference // 2) * HAIKU_COST_PER_INPUT_TOKEN + \
                      (avg_tokens_per_inference // 2) * HAIKU_COST_PER_OUTPUT_TOKEN
    else:
        claude_cost = (avg_tokens_per_inference // 2) * TEACHER_COST_PER_INPUT_TOKEN + \
                      (avg_tokens_per_inference // 2) * TEACHER_COST_PER_OUTPUT_TOKEN

    # Student (vLLM) cost per request
    student_cost = student_instance_cost_per_hour / student_requests_per_hour

    savings_per_inference = claude_cost - student_cost
    total_fixed = teacher_label_cost + training_cost

    if savings_per_inference <= 0:
        break_even = -1
        note = "Student costs more per inference than teacher at this volume — not worth fine-tuning."
    else:
        break_even = int(total_fixed / savings_per_inference) + 1
        note = f"After {break_even:,} inferences, self-hosted student pays for itself."

    return BreakEvenReport(
        teacher_label_cost=teacher_label_cost,
        training_cost=training_cost,
        total_fixed_cost=total_fixed,
        claude_cost_per_inference=claude_cost,
        student_cost_per_inference=student_cost,
        savings_per_inference=savings_per_inference,
        break_even_n_inferences=break_even,
        note=note,
    )


def print_cost_report(
    teacher: TeacherCostReport,
    training: TrainingCostReport,
    break_even: BreakEvenReport,
) -> None:
    print("\n" + "=" * 60)
    print("DISTILLATION COST ANALYSIS")
    print("=" * 60)
    print(f"\nTeacher Labeling (Claude):")
    print(f"  Examples labeled:     {teacher.n_examples:,}")
    print(f"  Total cost:           ${teacher.cost_usd:.4f}")
    print(f"  Avg cost/example:     ${teacher.avg_cost_per_example:.6f}")
    print(f"\nStudent Training ({training.model_name}, LoRA r={training.lora_rank}):")
    print(f"  Trainable params:     {training.trainable_params:,} ({training.trainable_pct:.2f}%)")
    print(f"  Training time:        {training.training_hours:.2f} hours")
    print(f"  GPU cost:             ${training.gpu_cost_usd:.4f}")
    print(f"\nBreak-Even Analysis:")
    print(f"  Fixed cost total:     ${break_even.total_fixed_cost:.4f}")
    print(f"  Claude (Haiku)/req:   ${break_even.claude_cost_per_inference:.8f}")
    print(f"  Student (vLLM)/req:   ${break_even.student_cost_per_inference:.8f}")
    print(f"  Savings/inference:    ${break_even.savings_per_inference:.8f}")
    print(f"  Break-even at:        {break_even.break_even_n_inferences:,} inferences")
    print(f"\n  {break_even.note}")
    print("=" * 60 + "\n")
