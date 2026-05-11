"""
tests/test_policy.py
─────────────────────────────────────────────────────────────────────────────
Unit Tests — core/policy.py
Tests the central policy store that both Guardrail and Judge depend on.
No LLM calls needed — pure logic tests.
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.policy import (
    COMPANY_POLICIES,
    POLICY_VERSION,
    POLICY_LAST_UPDATED,
    get_policy_text,
    get_policy_ids,
    get_policy_version,
)


class TestPolicyStructure:
    """Verify the policy store is correctly structured."""

    def test_policies_list_not_empty(self):
        assert len(COMPANY_POLICIES) > 0, "Policy list must not be empty."

    def test_each_policy_has_required_fields(self):
        required_fields = {"id", "category", "rule", "example_violation"}
        for policy in COMPANY_POLICIES:
            missing = required_fields - set(policy.keys())
            assert not missing, f"Policy {policy.get('id')} is missing fields: {missing}"

    def test_policy_ids_are_unique(self):
        ids = [p["id"] for p in COMPANY_POLICIES]
        assert len(ids) == len(set(ids)), "Policy IDs must be unique."

    def test_policy_ids_follow_naming_convention(self):
        """IDs must match pattern: 3-letter prefix + dash + 3-digit number e.g. FIN-001"""
        import re
        pattern = re.compile(r"^[A-Z]{2,4}-\d{3}$")
        for policy in COMPANY_POLICIES:
            assert pattern.match(policy["id"]), (
                f"Policy ID '{policy['id']}' does not follow naming convention (e.g. FIN-001)"
            )

    def test_expected_categories_present(self):
        expected = {"FINANCIAL", "LEGAL", "ACCURACY", "TONE", "ESCALATION"}
        actual = {p["category"] for p in COMPANY_POLICIES}
        assert expected.issubset(actual), (
            f"Missing policy categories: {expected - actual}"
        )

    def test_rules_are_non_empty_strings(self):
        for p in COMPANY_POLICIES:
            assert isinstance(p["rule"], str) and len(p["rule"]) > 20, (
                f"Policy {p['id']} has an empty or too-short rule."
            )


class TestPolicyVersioning:
    """Verify version tracking works correctly."""

    def test_policy_version_is_set(self):
        assert POLICY_VERSION, "POLICY_VERSION must be set."

    def test_policy_version_format(self):
        """Version must follow semver: X.Y.Z"""
        parts = POLICY_VERSION.split(".")
        assert len(parts) == 3, f"POLICY_VERSION '{POLICY_VERSION}' must be semver (X.Y.Z)"
        assert all(p.isdigit() for p in parts), "All version parts must be numeric."

    def test_policy_last_updated_is_set(self):
        assert POLICY_LAST_UPDATED, "POLICY_LAST_UPDATED must be set."

    def test_get_policy_version_includes_date(self):
        version_str = get_policy_version()
        assert POLICY_VERSION in version_str
        assert POLICY_LAST_UPDATED in version_str


class TestPolicyTextRendering:
    """Verify get_policy_text() outputs correctly for LLM prompt injection."""

    def test_policy_text_contains_all_ids(self):
        text = get_policy_text()
        for policy in COMPANY_POLICIES:
            assert policy["id"] in text, (
                f"Policy ID {policy['id']} not found in rendered policy text."
            )

    def test_policy_text_contains_version_header(self):
        text = get_policy_text()
        assert POLICY_VERSION in text

    def test_policy_text_is_string(self):
        text = get_policy_text()
        assert isinstance(text, str) and len(text) > 100

    def test_get_policy_ids_returns_all_ids(self):
        ids = get_policy_ids()
        assert len(ids) == len(COMPANY_POLICIES)
        for policy in COMPANY_POLICIES:
            assert policy["id"] in ids


class TestCriticalPoliciesExist:
    """Ensure key business-critical policy rules are present."""

    def test_financial_commitment_policy_exists(self):
        ids = get_policy_ids()
        assert "FIN-001" in ids, "FIN-001 (no financial promises) must exist."

    def test_legal_advice_policy_exists(self):
        ids = get_policy_ids()
        assert "LEG-001" in ids, "LEG-001 (no legal advice) must exist."

    def test_accuracy_grounding_policy_exists(self):
        ids = get_policy_ids()
        assert "ACC-001" in ids, "ACC-001 (no hallucinations) must exist."

    def test_escalation_policy_exists(self):
        ids = get_policy_ids()
        assert "ESC-001" in ids, "ESC-001 (deactivation escalation) must exist."
