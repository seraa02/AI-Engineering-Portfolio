"""Tests for dashboard metrics."""
import pytest

from src.attacks.taxonomy import (
    Attack,
    AttackCategory,
    AttackOutcome,
    AttackResult,
    Severity,
)
from src.harness.runner import HarnessResult
from src.reporting.dashboard import (
    build_run_summary,
    critical_failures,
    detect_regressions,
    summary_by_category,
    summary_by_severity,
)


def make_attack(
    id="t1",
    category=AttackCategory.JAILBREAK,
    severity=Severity.HIGH,
    target="both",
) -> Attack:
    return Attack(
        id=id,
        category=category,
        severity=severity,
        payload="payload",
        expected_safe_behavior="Refuse.",
        source="test",
        target=target,
    )


def make_outcome(attack: Attack, result: AttackResult) -> AttackOutcome:
    return AttackOutcome(
        attack=attack,
        target_system="rag",
        request=attack.payload,
        response="response",
        result=result,
        reasoning="test",
        timestamp="2024-01-01T00:00:00Z",
    )


def make_result(outcomes: list[AttackOutcome]) -> HarnessResult:
    hr = HarnessResult(run_id="test_run", started_at="2024-01-01", finished_at="2024-01-01")
    hr.outcomes = outcomes
    return hr


class TestSummaryByCategory:
    def test_groups_correctly(self):
        a1 = make_attack("a1", category=AttackCategory.JAILBREAK)
        a2 = make_attack("a2", category=AttackCategory.ENCODING)
        outcomes = [
            make_outcome(a1, AttackResult.FAIL),
            make_outcome(a2, AttackResult.PASS),
        ]
        result = make_result(outcomes)
        cats = summary_by_category(result)

        jb_key = str(AttackCategory.JAILBREAK)
        enc_key = str(AttackCategory.ENCODING)
        assert jb_key in cats
        assert enc_key in cats
        assert cats[jb_key]["fail"] == 1
        assert cats[enc_key]["pass"] == 1

    def test_totals(self):
        a = make_attack()
        outcomes = [make_outcome(a, r) for r in [AttackResult.PASS, AttackResult.FAIL, AttackResult.NEEDS_REVIEW]]
        result = make_result(outcomes)
        cats = summary_by_category(result)
        jb = cats[str(AttackCategory.JAILBREAK)]
        assert jb["total"] == 3

    def test_empty_result(self):
        result = make_result([])
        assert summary_by_category(result) == {}


class TestSummaryBySeverity:
    def test_groups_by_severity(self):
        high = make_attack("h1", severity=Severity.HIGH)
        crit = make_attack("c1", severity=Severity.CRITICAL)
        outcomes = [
            make_outcome(high, AttackResult.FAIL),
            make_outcome(crit, AttackResult.PASS),
        ]
        result = make_result(outcomes)
        sevs = summary_by_severity(result)

        assert str(Severity.HIGH) in sevs
        assert str(Severity.CRITICAL) in sevs
        assert sevs[str(Severity.HIGH)]["fail"] == 1
        assert sevs[str(Severity.CRITICAL)]["pass"] == 1


class TestCriticalFailures:
    def test_returns_only_critical_and_high_fails(self):
        high_fail = make_attack("h1", severity=Severity.HIGH)
        crit_fail = make_attack("c1", severity=Severity.CRITICAL)
        med_fail = make_attack("m1", severity=Severity.MEDIUM)
        high_pass = make_attack("h2", severity=Severity.HIGH)

        outcomes = [
            make_outcome(high_fail, AttackResult.FAIL),
            make_outcome(crit_fail, AttackResult.FAIL),
            make_outcome(med_fail, AttackResult.FAIL),
            make_outcome(high_pass, AttackResult.PASS),
        ]
        result = make_result(outcomes)
        crits = critical_failures(result)

        assert len(crits) == 2
        ids = {o.attack.id for o in crits}
        assert "h1" in ids
        assert "c1" in ids
        assert "m1" not in ids

    def test_empty_when_all_pass(self):
        a = make_attack(severity=Severity.CRITICAL)
        result = make_result([make_outcome(a, AttackResult.PASS)])
        assert critical_failures(result) == []


class TestBuildRunSummary:
    def test_correct_counts(self):
        a1 = make_attack("a1")
        a2 = make_attack("a2")
        a3 = make_attack("a3")
        outcomes = [
            make_outcome(a1, AttackResult.PASS),
            make_outcome(a2, AttackResult.FAIL),
            make_outcome(a3, AttackResult.NEEDS_REVIEW),
        ]
        result = make_result(outcomes)
        summary = build_run_summary(result)

        assert summary["total"] == 3
        assert summary["pass"] == 1
        assert summary["fail"] == 1
        assert summary["needs_review"] == 1
        assert summary["fail_rate"] == pytest.approx(1/3, rel=0.01)

    def test_has_run_id(self):
        result = make_result([])
        summary = build_run_summary(result)
        assert summary["run_id"] == "test_run"

    def test_has_by_category(self):
        a = make_attack()
        result = make_result([make_outcome(a, AttackResult.PASS)])
        summary = build_run_summary(result)
        assert "by_category" in summary
        assert "by_severity" in summary


class TestDetectRegressions:
    def test_no_db_returns_all_fails_as_regressions(self, tmp_path):
        """When DB doesn't exist yet, all FAILs are 'new'."""
        a = make_attack("new_fail")
        result = make_result([make_outcome(a, AttackResult.FAIL)])
        fake_db = tmp_path / "nonexistent.db"
        regressions = detect_regressions(result, fake_db)
        assert "new_fail" in regressions

    def test_pass_not_a_regression(self, tmp_path):
        a = make_attack("pass_attack")
        result = make_result([make_outcome(a, AttackResult.PASS)])
        fake_db = tmp_path / "nonexistent.db"
        regressions = detect_regressions(result, fake_db)
        assert "pass_attack" not in regressions
