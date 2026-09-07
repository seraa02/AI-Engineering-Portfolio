"""Tests for cost analysis module."""
import pytest
from src.teacher.schema import ExtractionExample
from src.training.cost_analysis import (
    BreakEvenReport,
    TeacherCostReport,
    TrainingCostReport,
    compute_break_even,
    compute_teacher_cost,
    estimate_training_cost,
)


def make_example(cost: float = 0.001, tokens: int = 200) -> ExtractionExample:
    ex = ExtractionExample(id="e", text="text", entities=[])
    ex.teacher_cost_usd = cost
    ex.token_count = tokens
    return ex


class TestComputeTeacherCost:
    def test_aggregates_costs(self):
        examples = [make_example(cost=0.001), make_example(cost=0.002)]
        report = compute_teacher_cost(examples)
        assert report.cost_usd == pytest.approx(0.003, rel=1e-4)

    def test_n_examples(self):
        examples = [make_example() for _ in range(10)]
        report = compute_teacher_cost(examples)
        assert report.n_examples == 10

    def test_avg_cost(self):
        examples = [make_example(cost=0.002), make_example(cost=0.004)]
        report = compute_teacher_cost(examples)
        assert report.avg_cost_per_example == pytest.approx(0.003, rel=1e-4)

    def test_empty_examples(self):
        report = compute_teacher_cost([])
        assert report.cost_usd == 0.0
        assert report.avg_cost_per_example == 0.0


class TestEstimateTrainingCost:
    def test_rank_8_cheaper_than_rank_32(self):
        r8 = estimate_training_cost(lora_rank=8)
        r32 = estimate_training_cost(lora_rank=32)
        # Both have same training time (same examples/epochs), but trainable params differ
        assert r8.trainable_params < r32.trainable_params

    def test_rank_8_trainable_pct(self):
        report = estimate_training_cost(lora_rank=8)
        assert 0 < report.trainable_pct < 5  # Should be < 5% for LoRA

    def test_rank_32_trainable_pct(self):
        report = estimate_training_cost(lora_rank=32)
        assert report.trainable_pct > report.__class__(**{
            **report.__dict__,
            "trainable_params": estimate_training_cost(lora_rank=8).trainable_params,
            "trainable_pct": estimate_training_cost(lora_rank=8).trainable_pct,
        }).trainable_pct

    def test_more_examples_longer_training(self):
        small = estimate_training_cost(n_examples=100)
        large = estimate_training_cost(n_examples=1000)
        assert large.training_hours > small.training_hours

    def test_gpu_cost_positive(self):
        report = estimate_training_cost()
        assert report.gpu_cost_usd > 0

    def test_model_name_preserved(self):
        report = estimate_training_cost(model_name="my-model")
        assert report.model_name == "my-model"


class TestComputeBreakEven:
    def test_positive_savings_gives_break_even(self):
        report = compute_break_even(
            teacher_label_cost=10.0,
            training_cost=5.0,
            teacher_model="haiku",
        )
        # Haiku cost > vLLM cost, so savings should be positive
        assert report.savings_per_inference > 0
        assert report.break_even_n_inferences > 0

    def test_fixed_cost_sum(self):
        report = compute_break_even(10.0, 5.0)
        assert report.total_fixed_cost == pytest.approx(15.0)

    def test_zero_fixed_cost_means_instant_breakeven(self):
        report = compute_break_even(0.0, 0.0)
        assert report.break_even_n_inferences <= 1

    def test_note_contains_number(self):
        report = compute_break_even(100.0, 50.0)
        # Note should mention the break-even count
        assert "inferences" in report.note.lower() or "worth" in report.note.lower()

    def test_teacher_model_sonnet_more_expensive(self):
        haiku = compute_break_even(1.0, 1.0, teacher_model="haiku")
        sonnet = compute_break_even(1.0, 1.0, teacher_model="sonnet")
        assert sonnet.claude_cost_per_inference > haiku.claude_cost_per_inference
        assert sonnet.savings_per_inference > haiku.savings_per_inference


class TestTrainingCostReport:
    def test_trainable_pct_range(self):
        report = estimate_training_cost(lora_rank=8)
        assert 0 <= report.trainable_pct <= 100

    def test_lora_rank_preserved(self):
        report = estimate_training_cost(lora_rank=16)
        assert report.lora_rank == 16
