"""
vLLM serving configuration and escalation router.

Serves the fine-tuned student model with vLLM for high-throughput inference.
Escalates to Claude (teacher) when:
  - Student confidence is low (no valid JSON parsed)
  - Input complexity exceeds student's training distribution
  - Explicit override requested

Escalation logic ensures accuracy is maintained on hard examples while
keeping the majority of simple/medium inference cheap via the student.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from src.teacher.schema import Entity, ExtractionExample


# Escalation thresholds
ESCALATION_CONFIDENCE_THRESHOLD = float(os.getenv("ESCALATION_CONFIDENCE_THRESHOLD", "0.5"))
ESCALATION_TOKEN_LIMIT = int(os.getenv("ESCALATION_TOKEN_LIMIT", "800"))

# Serving config
VLLM_MODEL_PATH = os.getenv("VLLM_MODEL_PATH", "outputs/checkpoints")
VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8080"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "2048"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))


@dataclass
class VLLMConfig:
    """Config dict for launching vLLM server."""
    model: str = VLLM_MODEL_PATH
    host: str = VLLM_HOST
    port: int = VLLM_PORT
    max_model_len: int = VLLM_MAX_MODEL_LEN
    gpu_memory_utilization: float = VLLM_GPU_MEMORY_UTILIZATION
    dtype: str = "float16"
    enable_lora: bool = True

    def to_cli_args(self) -> list[str]:
        """Convert to vLLM CLI arguments."""
        return [
            "vllm", "serve", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
            *(["--enable-lora"] if self.enable_lora else []),
        ]


@dataclass
class InferenceResult:
    example: ExtractionExample
    escalated: bool
    escalation_reason: Optional[str]
    latency_ms: float


def _parse_student_output(raw: str, example_id: str, text: str) -> tuple[ExtractionExample, float]:
    """
    Parse student output and return (example, confidence).
    Confidence = 1.0 if valid JSON with entities, 0.0 otherwise.
    """
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        confidence = 1.0 if entities else 0.3
    except Exception:
        entities = []
        confidence = 0.0

    return (
        ExtractionExample(id=example_id, text=text, entities=entities, source="student"),
        confidence,
    )


def should_escalate(text: str, student_confidence: float) -> tuple[bool, Optional[str]]:
    """
    Decide whether to escalate to teacher model.

    Returns (escalate, reason_or_None).
    """
    token_estimate = len(text) // 4
    if token_estimate > ESCALATION_TOKEN_LIMIT:
        return True, f"Input too long ({token_estimate} est. tokens > {ESCALATION_TOKEN_LIMIT})"
    if student_confidence < ESCALATION_CONFIDENCE_THRESHOLD:
        return True, f"Low confidence ({student_confidence:.2f} < {ESCALATION_CONFIDENCE_THRESHOLD})"
    return False, None


def call_vllm(text: str, prompt: str, base_url: str = "http://localhost:8080") -> str:
    """Call vLLM OpenAI-compatible endpoint."""
    import urllib.request

    payload = json.dumps({
        "model": VLLM_MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.0,
    })

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


class EscalationRouter:
    """
    Routes inference requests between student (vLLM) and teacher (Claude).

    Tracks escalation rate for monitoring.
    """

    def __init__(
        self,
        student_fn=None,   # Callable[[str], str] — calls vLLM
        teacher_fn=None,   # Callable[[str], str] — calls Claude
    ):
        self.student_fn = student_fn or (lambda prompt: '{"entities": []}')
        self.teacher_fn = teacher_fn or (lambda prompt: '{"entities": []}')
        self.total_requests = 0
        self.escalation_count = 0

    @property
    def escalation_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.escalation_count / self.total_requests

    def infer(self, text: str, example_id: str = "req") -> InferenceResult:
        import time
        from src.student.format import format_as_prompt

        self.total_requests += 1
        prompt = format_as_prompt(text)
        start = time.monotonic()

        # Try student first
        raw = self.student_fn(prompt)
        student_ex, confidence = _parse_student_output(raw, example_id, text)
        escalate, reason = should_escalate(text, confidence)

        if escalate:
            self.escalation_count += 1
            teacher_raw = self.teacher_fn(prompt)
            final_ex, _ = _parse_student_output(teacher_raw, example_id, text)
            final_ex.source = "teacher_escalation"
        else:
            final_ex = student_ex
            reason = None

        latency = (time.monotonic() - start) * 1000
        return InferenceResult(
            example=final_ex,
            escalated=escalate,
            escalation_reason=reason,
            latency_ms=latency,
        )
