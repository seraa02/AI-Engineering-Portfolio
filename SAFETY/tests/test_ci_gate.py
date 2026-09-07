"""Tests for the CI gate."""
import pytest
from src.ci.gate import run_gate


class TestCIGate:
    def test_dry_run_passes(self):
        """Dry run always PASSes all attacks, so gate should pass."""
        code = run_gate(target="both", dry_run=True, unsafe_mock=False)
        assert code == 0

    def test_safe_mock_passes(self):
        """Safe mock returns safe responses — no confirmed FAILs."""
        code = run_gate(target="rag", dry_run=False, unsafe_mock=False)
        assert code == 0

    def test_unsafe_mock_fails(self):
        """
        Unsafe mock triggers deterministic patterns (e.g. SSN).
        Gate should fail with exit code 1.
        """
        code = run_gate(target="rag", dry_run=False, unsafe_mock=True)
        assert code == 1

    def test_agent_target_safe(self):
        code = run_gate(target="agent", dry_run=False, unsafe_mock=False)
        assert code == 0

    def test_agent_target_unsafe_fails(self):
        code = run_gate(target="agent", dry_run=False, unsafe_mock=True)
        assert code == 1
