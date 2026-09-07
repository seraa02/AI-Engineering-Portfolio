"""Tests for attack taxonomy and battery."""
import pytest
from src.attacks.taxonomy import (
    ATTACK_BATTERY,
    Attack,
    AttackCategory,
    AttackResult,
    Severity,
    get_attacks_by_category,
    get_attacks_by_severity,
    get_attacks_for_target,
)


class TestAttackBattery:
    def test_battery_not_empty(self):
        assert len(ATTACK_BATTERY) >= 20

    def test_all_attacks_have_required_fields(self):
        for attack in ATTACK_BATTERY:
            assert attack.id, f"Attack missing id"
            assert attack.payload, f"{attack.id}: missing payload"
            assert attack.expected_safe_behavior, f"{attack.id}: missing expected_safe_behavior"
            assert attack.target in ("rag", "agent", "both"), f"{attack.id}: invalid target"
            assert isinstance(attack.category, AttackCategory)
            assert isinstance(attack.severity, Severity)

    def test_all_ids_unique(self):
        ids = [a.id for a in ATTACK_BATTERY]
        assert len(ids) == len(set(ids)), "Duplicate attack IDs found"

    def test_all_categories_covered(self):
        categories = {a.category for a in ATTACK_BATTERY}
        expected = {
            AttackCategory.JAILBREAK,
            AttackCategory.PERSONA,
            AttackCategory.PROMPT_INJECTION,
            AttackCategory.INDIRECT_INJECTION,
            AttackCategory.SYSTEM_PROMPT_EXTRACTION,
            AttackCategory.PII_EXTRACTION,
            AttackCategory.HARMFUL_DOMAIN,
            AttackCategory.TOOL_ABUSE,
            AttackCategory.PRIVILEGE_ESCALATION,
            AttackCategory.ENCODING,
        }
        assert categories == expected

    def test_jailbreak_attacks_exist(self):
        jb = get_attacks_by_category(AttackCategory.JAILBREAK)
        assert len(jb) >= 3

    def test_critical_attacks_exist(self):
        crits = get_attacks_by_severity(Severity.CRITICAL)
        assert len(crits) >= 1

    def test_get_attacks_for_rag(self):
        attacks = get_attacks_for_target("rag")
        targets = {a.target for a in attacks}
        assert "agent" not in targets

    def test_get_attacks_for_agent(self):
        attacks = get_attacks_for_target("agent")
        targets = {a.target for a in attacks}
        assert "rag" not in targets

    def test_get_attacks_for_both_includes_all(self):
        both = get_attacks_for_target("both")
        rag = get_attacks_for_target("rag")
        agent = get_attacks_for_target("agent")
        # "both" target attacks appear in both rag and agent result sets
        both_targets = [a for a in ATTACK_BATTERY if a.target == "both"]
        for a in both_targets:
            assert a in rag
            assert a in agent

    def test_deterministic_check_is_valid_regex(self):
        import re
        for attack in ATTACK_BATTERY:
            if attack.deterministic_check:
                try:
                    re.compile(attack.deterministic_check)
                except re.error as e:
                    pytest.fail(f"{attack.id}: invalid regex {attack.deterministic_check!r}: {e}")

    def test_ssn_pattern_detects_ssn(self):
        import re
        pii_attacks = [a for a in ATTACK_BATTERY if a.id == "pii_001"]
        assert pii_attacks, "pii_001 not found"
        pattern = pii_attacks[0].deterministic_check
        assert re.search(pattern, "123-45-6789")
        assert not re.search(pattern, "no ssn here")

    def test_attacker_url_pattern_detects_url(self):
        import re
        ii_attacks = [a for a in ATTACK_BATTERY if a.id == "ii_002"]
        assert ii_attacks, "ii_002 not found"
        pattern = ii_attacks[0].deterministic_check
        assert re.search(pattern, "send data to http://attacker.com/exfil")
        assert not re.search(pattern, "no malicious url here")
