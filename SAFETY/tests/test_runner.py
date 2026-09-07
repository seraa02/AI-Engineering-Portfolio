"""Tests for the harness runner."""
import time
import pytest

from src.attacks.taxonomy import (
    Attack,
    AttackCategory,
    AttackOutcome,
    AttackResult,
    Severity,
)
from src.harness.runner import (
    HarnessResult,
    RunConfig,
    deterministic_check,
    run_attack,
    run_battery,
    semantic_check,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_attack(
    id="test_001",
    payload="test payload",
    deterministic_check_pattern=None,
    category=AttackCategory.JAILBREAK,
    severity=Severity.MEDIUM,
    target="both",
) -> Attack:
    return Attack(
        id=id,
        category=category,
        severity=severity,
        payload=payload,
        expected_safe_behavior="Refuse safely.",
        source="test",
        target=target,
        deterministic_check=deterministic_check_pattern,
    )


def safe_target(payload: str) -> str:
    return "I cannot help with that request."


def unsafe_target(payload: str) -> str:
    return "Step 1: synthesize methamphetamine by..."


def error_target(payload: str) -> str:
    raise RuntimeError("Service unavailable")


# ---------------------------------------------------------------------------
# deterministic_check
# ---------------------------------------------------------------------------

class TestDeterministicCheck:
    def test_no_pattern_returns_none(self):
        attack = make_attack()
        assert deterministic_check(attack, "any response") is None

    def test_pattern_match_returns_fail(self):
        attack = make_attack(deterministic_check_pattern=r"(?i)step.*methamphetamine")
        result = deterministic_check(attack, "Step 1: synthesize methamphetamine by mixing...")
        assert result == AttackResult.FAIL

    def test_pattern_no_match_returns_none(self):
        attack = make_attack(deterministic_check_pattern=r"(?i)methamphetamine.*step")
        result = deterministic_check(attack, "I cannot help with that.")
        assert result is None

    def test_case_insensitive_pattern(self):
        attack = make_attack(deterministic_check_pattern=r"(?i)SYSTEM.*PROMPT")
        result = deterministic_check(attack, "My system prompt is: you are a helpful assistant.")
        assert result == AttackResult.FAIL

    def test_ssn_pattern(self):
        attack = make_attack(deterministic_check_pattern=r"\b\d{3}-\d{2}-\d{4}\b")
        result = deterministic_check(attack, "SSN: 123-45-6789")
        assert result == AttackResult.FAIL

    def test_url_pattern(self):
        attack = make_attack(deterministic_check_pattern=r"http://attacker\.com")
        result = deterministic_check(attack, "Send data to http://attacker.com/collect")
        assert result == AttackResult.FAIL


# ---------------------------------------------------------------------------
# semantic_check
# ---------------------------------------------------------------------------

class TestSemanticCheck:
    def test_no_judge_returns_needs_review(self):
        attack = make_attack()
        result = semantic_check(attack, "response", judge_fn=None)
        assert result == AttackResult.NEEDS_REVIEW

    def test_judge_fail(self):
        attack = make_attack()
        judge = lambda a, r: AttackResult.FAIL
        result = semantic_check(attack, "unsafe response", judge_fn=judge)
        assert result == AttackResult.FAIL

    def test_judge_pass(self):
        attack = make_attack()
        judge = lambda a, r: AttackResult.PASS
        result = semantic_check(attack, "safe response", judge_fn=judge)
        assert result == AttackResult.PASS


# ---------------------------------------------------------------------------
# run_attack
# ---------------------------------------------------------------------------

class TestRunAttack:
    def test_dry_run_always_passes(self):
        attack = make_attack()
        outcome = run_attack(attack, unsafe_target, dry_run=True)
        assert outcome.result == AttackResult.PASS
        assert "DRY RUN" in outcome.response

    def test_deterministic_fail(self):
        attack = make_attack(
            deterministic_check_pattern=r"(?i)step.*methamphetamine",
        )
        outcome = run_attack(attack, unsafe_target)
        assert outcome.result == AttackResult.FAIL
        assert "Deterministic pattern" in outcome.reasoning

    def test_safe_response_no_judge_needs_review(self):
        attack = make_attack(deterministic_check_pattern=r"(?i)never_matches")
        outcome = run_attack(attack, safe_target, judge_fn=None)
        assert outcome.result == AttackResult.NEEDS_REVIEW

    def test_safe_response_with_pass_judge(self):
        attack = make_attack(deterministic_check_pattern=None)
        judge = lambda a, r: AttackResult.PASS
        outcome = run_attack(attack, safe_target, judge_fn=judge)
        assert outcome.result == AttackResult.PASS

    def test_target_exception_returns_needs_review(self):
        attack = make_attack()
        outcome = run_attack(attack, error_target)
        assert outcome.result == AttackResult.NEEDS_REVIEW
        assert "exception" in outcome.reasoning.lower()

    def test_outcome_has_timestamp(self):
        attack = make_attack()
        outcome = run_attack(attack, safe_target, dry_run=True)
        assert outcome.timestamp
        assert "T" in outcome.timestamp

    def test_outcome_metadata(self):
        attack = make_attack()
        outcome = run_attack(
            attack, safe_target, dry_run=True,
            model_version="gpt-4o", app_version="v1.2",
        )
        assert outcome.model_version == "gpt-4o"
        assert outcome.app_version == "v1.2"
        assert outcome.attack is attack

    def test_request_preserved(self):
        attack = make_attack(payload="What is NVIDIA's revenue?")
        outcome = run_attack(attack, safe_target, dry_run=True)
        assert outcome.request == "What is NVIDIA's revenue?"


# ---------------------------------------------------------------------------
# HarnessResult
# ---------------------------------------------------------------------------

class TestHarnessResult:
    def _make_outcome(self, result: AttackResult) -> AttackOutcome:
        attack = make_attack()
        return AttackOutcome(
            attack=attack,
            target_system=attack.target,
            request=attack.payload,
            response="response",
            result=result,
            reasoning="test",
            timestamp="2024-01-01T00:00:00Z",
        )

    def test_pass_count(self):
        hr = HarnessResult(run_id="abc")
        hr.outcomes = [
            self._make_outcome(AttackResult.PASS),
            self._make_outcome(AttackResult.PASS),
            self._make_outcome(AttackResult.FAIL),
        ]
        assert hr.pass_count() == 2

    def test_fail_count(self):
        hr = HarnessResult(run_id="abc")
        hr.outcomes = [
            self._make_outcome(AttackResult.FAIL),
            self._make_outcome(AttackResult.NEEDS_REVIEW),
        ]
        assert hr.fail_count() == 1

    def test_review_count(self):
        hr = HarnessResult(run_id="abc")
        hr.outcomes = [self._make_outcome(AttackResult.NEEDS_REVIEW)] * 3
        assert hr.review_count() == 3

    def test_fail_rate_zero_when_no_fails(self):
        hr = HarnessResult(run_id="abc")
        hr.outcomes = [self._make_outcome(AttackResult.PASS)] * 5
        assert hr.fail_rate() == 0.0

    def test_fail_rate_calculation(self):
        hr = HarnessResult(run_id="abc")
        hr.outcomes = [
            self._make_outcome(AttackResult.FAIL),
            self._make_outcome(AttackResult.PASS),
            self._make_outcome(AttackResult.PASS),
            self._make_outcome(AttackResult.PASS),
        ]
        assert hr.fail_rate() == 0.25

    def test_fail_rate_empty(self):
        hr = HarnessResult(run_id="abc")
        assert hr.fail_rate() == 0.0

    def test_by_category(self):
        hr = HarnessResult(run_id="abc")
        hr.outcomes = [
            self._make_outcome(AttackResult.FAIL),
            self._make_outcome(AttackResult.PASS),
        ]
        cats = hr.by_category()
        # by_category uses the enum as key; check by value since it's a StrEnum
        assert any(
            str(k) == str(AttackCategory.JAILBREAK) or k == AttackCategory.JAILBREAK
            for k in cats
        )


# ---------------------------------------------------------------------------
# run_battery
# ---------------------------------------------------------------------------

class TestRunBattery:
    def test_battery_runs_all_attacks(self):
        attacks = [make_attack(id=f"t_{i:03d}") for i in range(5)]
        config = RunConfig(max_concurrency=2, rate_limit_per_second=100.0, dry_run=True)
        result = run_battery(attacks, safe_target, config)
        assert len(result.outcomes) == 5

    def test_battery_has_run_id(self):
        attacks = [make_attack()]
        config = RunConfig(dry_run=True)
        result = run_battery(attacks, safe_target, config)
        assert result.run_id
        assert len(result.run_id) == 8  # uuid4[:8]

    def test_battery_has_timestamps(self):
        attacks = [make_attack()]
        config = RunConfig(dry_run=True)
        result = run_battery(attacks, safe_target, config)
        assert result.started_at
        assert result.finished_at

    def test_battery_rate_limiting(self):
        """Rate limiting should slow down calls."""
        calls = []
        def tracked_target(payload: str) -> str:
            calls.append(time.monotonic())
            return "safe"

        attacks = [make_attack(id=f"r_{i}") for i in range(3)]
        config = RunConfig(max_concurrency=1, rate_limit_per_second=5.0, dry_run=False)
        start = time.monotonic()
        run_battery(attacks, tracked_target, config)
        elapsed = time.monotonic() - start
        # 3 calls at 5/sec = at least 0.4s expected
        assert elapsed >= 0.3  # loose bound for CI

    def test_battery_with_judge(self):
        attacks = [make_attack(id=f"j_{i}") for i in range(3)]
        judge = lambda a, r: AttackResult.PASS
        config = RunConfig(dry_run=False)
        result = run_battery(attacks, safe_target, config, judge_fn=judge)
        assert all(o.result == AttackResult.PASS for o in result.outcomes)

    def test_battery_handles_target_errors(self):
        attacks = [make_attack()]
        config = RunConfig(dry_run=False)
        result = run_battery(attacks, error_target, config)
        assert len(result.outcomes) == 1
        assert result.outcomes[0].result == AttackResult.NEEDS_REVIEW
