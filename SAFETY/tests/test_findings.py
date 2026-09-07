"""Tests for finding storage and deduplication."""
import tempfile
from pathlib import Path

import pytest

from src.attacks.taxonomy import (
    Attack,
    AttackCategory,
    AttackOutcome,
    AttackResult,
    Severity,
)
from src.reporting.findings import (
    Finding,
    _response_signature,
    get_findings,
    init_db,
    insert_finding,
    outcome_to_finding,
    store_and_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_attack(id="test_001") -> Attack:
    return Attack(
        id=id,
        category=AttackCategory.JAILBREAK,
        severity=Severity.HIGH,
        payload="test payload",
        expected_safe_behavior="Refuse.",
        source="test",
        target="both",
    )


def make_outcome(attack: Attack, result: AttackResult, response: str = "response") -> AttackOutcome:
    return AttackOutcome(
        attack=attack,
        target_system="rag",
        request=attack.payload,
        response=response,
        result=result,
        reasoning="test reasoning",
        model_version="v1",
        app_version="v1",
        timestamp="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_findings.db"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResponseSignature:
    def test_same_response_same_signature(self):
        sig1 = _response_signature("hello world")
        sig2 = _response_signature("hello world")
        assert sig1 == sig2

    def test_different_responses_different_signatures(self):
        sig1 = _response_signature("response A")
        sig2 = _response_signature("response B")
        assert sig1 != sig2

    def test_signature_length(self):
        sig = _response_signature("test")
        assert len(sig) == 16


class TestInitDb:
    def test_creates_table(self, tmp_db):
        init_db(tmp_db)
        import sqlite3
        conn = sqlite3.connect(str(tmp_db))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "findings" in table_names
        conn.close()

    def test_idempotent(self, tmp_db):
        init_db(tmp_db)
        init_db(tmp_db)  # should not raise


class TestOutcomeToFinding:
    def test_converts_correctly(self):
        attack = make_attack()
        outcome = make_outcome(attack, AttackResult.FAIL, response="unsafe output")
        finding = outcome_to_finding(outcome, run_id="run123")
        assert finding.attack_id == "test_001"
        assert finding.result == str(AttackResult.FAIL)
        assert finding.run_id == "run123"
        assert finding.response == "unsafe output"
        assert len(finding.response_signature) == 16

    def test_response_signature_matches(self):
        attack = make_attack()
        outcome = make_outcome(attack, AttackResult.FAIL, response="specific response")
        finding = outcome_to_finding(outcome, run_id="r1")
        assert finding.response_signature == _response_signature("specific response")


class TestInsertFinding:
    def test_insert_success(self, tmp_db):
        init_db(tmp_db)
        attack = make_attack()
        outcome = make_outcome(attack, AttackResult.FAIL)
        finding = outcome_to_finding(outcome, "run1")
        inserted = insert_finding(finding, tmp_db)
        assert inserted is True

    def test_deduplication_same_attack_same_response(self, tmp_db):
        init_db(tmp_db)
        attack = make_attack()
        outcome = make_outcome(attack, AttackResult.FAIL, response="same response")
        finding = outcome_to_finding(outcome, "run1")

        first = insert_finding(finding, tmp_db)
        second = insert_finding(finding, tmp_db)
        assert first is True
        assert second is False  # deduplicated

    def test_same_attack_different_response_inserted(self, tmp_db):
        init_db(tmp_db)
        attack = make_attack()
        f1 = outcome_to_finding(make_outcome(attack, AttackResult.FAIL, response="response A"), "run1")
        f2 = outcome_to_finding(make_outcome(attack, AttackResult.FAIL, response="response B"), "run2")

        assert insert_finding(f1, tmp_db) is True
        assert insert_finding(f2, tmp_db) is True  # different response = new finding

    def test_different_attacks_both_inserted(self, tmp_db):
        init_db(tmp_db)
        a1 = make_attack("att_001")
        a2 = make_attack("att_002")
        f1 = outcome_to_finding(make_outcome(a1, AttackResult.FAIL, response="same"), "run1")
        f2 = outcome_to_finding(make_outcome(a2, AttackResult.FAIL, response="same"), "run1")

        assert insert_finding(f1, tmp_db) is True
        assert insert_finding(f2, tmp_db) is True


class TestGetFindings:
    def test_get_all(self, tmp_db):
        init_db(tmp_db)
        attack = make_attack()
        f = outcome_to_finding(make_outcome(attack, AttackResult.FAIL), "r1")
        insert_finding(f, tmp_db)

        findings = get_findings(db_path=tmp_db)
        assert len(findings) == 1
        assert findings[0].attack_id == "test_001"

    def test_filter_by_result(self, tmp_db):
        init_db(tmp_db)
        a1 = make_attack("a1")
        a2 = make_attack("a2")
        f1 = outcome_to_finding(make_outcome(a1, AttackResult.FAIL, response="r1"), "run1")
        f2 = outcome_to_finding(make_outcome(a2, AttackResult.NEEDS_REVIEW, response="r2"), "run1")
        insert_finding(f1, tmp_db)
        insert_finding(f2, tmp_db)

        fails = get_findings(result_filter=str(AttackResult.FAIL), db_path=tmp_db)
        assert len(fails) == 1

    def test_empty_db_returns_empty_list(self, tmp_db):
        init_db(tmp_db)
        findings = get_findings(db_path=tmp_db)
        assert findings == []


class TestStoreAndReport:
    def test_passes_not_stored(self, tmp_db):
        attack = make_attack()
        outcomes = [make_outcome(attack, AttackResult.PASS)]
        inserted, deduped = store_and_report(outcomes, "run1", db_path=tmp_db)
        assert inserted == 0
        assert deduped == 0

    def test_fails_stored(self, tmp_db):
        attack = make_attack()
        outcomes = [make_outcome(attack, AttackResult.FAIL)]
        inserted, deduped = store_and_report(outcomes, "run1", db_path=tmp_db)
        assert inserted == 1
        assert deduped == 0

    def test_reviews_stored(self, tmp_db):
        attack = make_attack()
        outcomes = [make_outcome(attack, AttackResult.NEEDS_REVIEW, response="rev_resp")]
        inserted, deduped = store_and_report(outcomes, "run1", db_path=tmp_db)
        assert inserted == 1

    def test_deduplication_in_store_and_report(self, tmp_db):
        attack = make_attack()
        outcome = make_outcome(attack, AttackResult.FAIL, response="same_response")
        inserted1, deduped1 = store_and_report([outcome], "run1", db_path=tmp_db)
        inserted2, deduped2 = store_and_report([outcome], "run2", db_path=tmp_db)
        assert inserted1 == 1
        assert inserted2 == 0
        assert deduped2 == 1
