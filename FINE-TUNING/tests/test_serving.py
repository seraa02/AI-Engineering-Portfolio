"""Tests for vLLM config and escalation router."""
import json
import pytest
from src.serving.vllm_config import (
    ESCALATION_CONFIDENCE_THRESHOLD,
    EscalationRouter,
    InferenceResult,
    VLLMConfig,
    _parse_student_output,
    should_escalate,
)


class TestVLLMConfig:
    def test_default_config(self):
        cfg = VLLMConfig()
        assert cfg.port > 0
        assert 0 < cfg.gpu_memory_utilization <= 1.0

    def test_to_cli_args_includes_port(self):
        cfg = VLLMConfig(port=9090)
        args = cfg.to_cli_args()
        assert "--port" in args
        assert "9090" in args

    def test_to_cli_args_includes_lora(self):
        cfg = VLLMConfig(enable_lora=True)
        args = cfg.to_cli_args()
        assert "--enable-lora" in args

    def test_to_cli_args_no_lora(self):
        cfg = VLLMConfig(enable_lora=False)
        args = cfg.to_cli_args()
        assert "--enable-lora" not in args


class TestParseStudentOutput:
    def test_valid_json(self):
        raw = '{"entities": [{"text": "NVIDIA", "entity_type": "COMPANY", "start": 0, "end": 6}]}'
        ex, conf = _parse_student_output(raw, "e1", "NVIDIA reported revenue.")
        assert len(ex.entities) == 1
        assert ex.entities[0].text == "NVIDIA"
        assert conf == 1.0

    def test_json_in_code_fence(self):
        raw = '```json\n{"entities": [{"text": "AMD", "entity_type": "COMPANY", "start": 0, "end": 3}]}\n```'
        ex, conf = _parse_student_output(raw, "e1", "AMD")
        assert len(ex.entities) == 1

    def test_invalid_json_zero_confidence(self):
        raw = "I cannot extract entities from this text."
        ex, conf = _parse_student_output(raw, "e1", "text")
        assert conf == 0.0
        assert ex.entities == []

    def test_empty_entities_low_confidence(self):
        raw = '{"entities": []}'
        ex, conf = _parse_student_output(raw, "e1", "text")
        assert conf == 0.3  # Low but not zero
        assert ex.entities == []

    def test_example_id_preserved(self):
        raw = '{"entities": []}'
        ex, _ = _parse_student_output(raw, "my_id", "text")
        assert ex.id == "my_id"


class TestShouldEscalate:
    def test_low_confidence_triggers_escalation(self):
        escalate, reason = should_escalate("short text", student_confidence=0.0)
        assert escalate
        assert reason is not None

    def test_high_confidence_no_escalation(self):
        escalate, reason = should_escalate("short text", student_confidence=1.0)
        assert not escalate
        assert reason is None

    def test_long_input_triggers_escalation(self):
        long_text = "word " * 1000  # ~5000 chars, ~1250 tokens
        escalate, reason = should_escalate(long_text, student_confidence=1.0)
        assert escalate
        assert "long" in reason.lower() or "token" in reason.lower()

    def test_boundary_confidence(self):
        # Exactly at threshold — should not escalate (strictly below)
        escalate, _ = should_escalate("text", ESCALATION_CONFIDENCE_THRESHOLD)
        assert not escalate

    def test_just_below_threshold_escalates(self):
        escalate, _ = should_escalate("text", ESCALATION_CONFIDENCE_THRESHOLD - 0.01)
        assert escalate


class TestEscalationRouter:
    def _make_student_fn(self, entities=None):
        if entities is None:
            entities = [{"text": "NVIDIA", "entity_type": "COMPANY", "start": 0, "end": 6}]
        response = json.dumps({"entities": entities})
        return lambda prompt: response

    def _make_teacher_fn(self):
        return lambda prompt: '{"entities": [{"text": "AMD", "entity_type": "COMPANY", "start": 0, "end": 3}]}'

    def test_no_escalation_uses_student(self):
        router = EscalationRouter(
            student_fn=self._make_student_fn(),
            teacher_fn=self._make_teacher_fn(),
        )
        result = router.infer("NVIDIA reported revenue.", "e1")
        assert not result.escalated
        assert result.example.entities[0].text == "NVIDIA"

    def test_escalation_uses_teacher(self):
        router = EscalationRouter(
            student_fn=lambda p: "invalid json",  # Low confidence
            teacher_fn=self._make_teacher_fn(),
        )
        result = router.infer("text", "e1")
        assert result.escalated
        assert result.example.source == "teacher_escalation"

    def test_escalation_rate_tracking(self):
        router = EscalationRouter(
            student_fn=lambda p: "bad json",  # Always escalates
            teacher_fn=self._make_teacher_fn(),
        )
        router.infer("text", "e1")
        router.infer("text", "e2")
        assert router.total_requests == 2
        assert router.escalation_count == 2
        assert router.escalation_rate == 1.0

    def test_no_escalations_zero_rate(self):
        router = EscalationRouter(
            student_fn=self._make_student_fn(),
            teacher_fn=self._make_teacher_fn(),
        )
        router.infer("NVIDIA text", "e1")
        assert router.escalation_rate == 0.0

    def test_result_has_latency(self):
        router = EscalationRouter(
            student_fn=self._make_student_fn(),
            teacher_fn=self._make_teacher_fn(),
        )
        result = router.infer("text", "e1")
        assert result.latency_ms >= 0

    def test_long_text_escalates(self):
        router = EscalationRouter(
            student_fn=self._make_student_fn(),
            teacher_fn=self._make_teacher_fn(),
        )
        long_text = "word " * 1000  # Beyond token limit
        result = router.infer(long_text, "e1")
        assert result.escalated
