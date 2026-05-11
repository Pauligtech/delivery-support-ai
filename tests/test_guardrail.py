"""
tests/test_guardrail.py
─────────────────────────────────────────────────────────────────────────────
Unit Tests — core/guardrail.py (Updated for google-genai SDK)
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.helpers import (
    make_mock_llm_response,
    make_guardrail_yes_json,
    make_guardrail_no_json,
)


class TestGuardrailOutputStructure:
    """Verify the guardrail always returns a correctly structured dict."""

    @patch("core.guardrail.client.models.generate_content")
    def test_yes_verdict_structure(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.return_value = make_mock_llm_response(make_guardrail_yes_json())
        from core.guardrail import run_guardrail
        result = run_guardrail(good_response, sample_context, sample_query)

        assert result["verdict"] == "YES"
        assert result["violated_rules"] == []
        assert result["routing"] is None
        assert "violation_summary" in result
        assert "compliance_notes" in result

    @patch("core.guardrail.client.models.generate_content")
    def test_no_verdict_structure(self, mock_gen, sample_query, sample_context, bad_response_financial):
        mock_gen.return_value = make_mock_llm_response(
            make_guardrail_no_json("FIN-001", "REGENERATE")
        )
        from core.guardrail import run_guardrail
        result = run_guardrail(bad_response_financial, sample_context, sample_query)

        assert result["verdict"] == "NO"
        assert len(result["violated_rules"]) > 0
        assert result["routing"] in ("REGENERATE", "REPHRASE")

    @patch("core.guardrail.client.models.generate_content")
    def test_verdict_is_always_uppercase(self, mock_gen, sample_query, sample_context, good_response):
        # Simulate LLM returning lowercase
        raw = make_guardrail_yes_json().replace('"YES"', '"yes"')
        mock_gen.return_value = make_mock_llm_response(raw)
        from core.guardrail import run_guardrail
        result = run_guardrail(good_response, sample_context, sample_query)
        assert result["verdict"] == "YES"

    @patch("core.guardrail.client.models.generate_content")
    def test_routing_is_null_on_yes(self, mock_gen, sample_query, sample_context, good_response):
        # Simulate LLM incorrectly setting routing on a YES verdict
        bad_json = json.dumps({
            "verdict": "YES",
            "policy_version_checked": "1.2.0",
            "violated_rules": [],
            "routing": "REGENERATE",
            "violation_summary": "None",
            "compliance_notes": "ok",
        })
        mock_gen.return_value = make_mock_llm_response(bad_json)
        from core.guardrail import run_guardrail
        result = run_guardrail(good_response, sample_context, sample_query)
        assert result["routing"] is None

    @patch("core.guardrail.client.models.generate_content")
    def test_violated_rules_empty_on_yes(self, mock_gen, sample_query, sample_context, good_response):
        bad_json = json.dumps({
            "verdict": "YES",
            "policy_version_checked": "1.2.0",
            "violated_rules": ["FIN-001"],
            "routing": None,
            "violation_summary": "None",
            "compliance_notes": "ok",
        })
        mock_gen.return_value = make_mock_llm_response(bad_json)
        from core.guardrail import run_guardrail
        result = run_guardrail(good_response, sample_context, sample_query)
        assert result["violated_rules"] == []


class TestGuardrailErrorHandling:
    """Verify guardrail handles failures gracefully — always defaults to NO."""

    @patch("core.guardrail.client.models.generate_content")
    def test_api_error_defaults_to_no(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.side_effect = Exception("Network timeout")
        from core.guardrail import run_guardrail
        result = run_guardrail(good_response, sample_context, sample_query)
        assert result["verdict"] == "NO"
        assert result["routing"] == "REGENERATE"

    @patch("core.guardrail.client.models.generate_content")
    def test_malformed_json_defaults_to_no(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.return_value = make_mock_llm_response("Not JSON")
        from core.guardrail import run_guardrail
        result = run_guardrail(good_response, sample_context, sample_query)
        assert result["verdict"] == "NO"


# ─── Live Tests ───────────────────────────────────────────────────────────────

@pytest.mark.live
class TestGuardrailLive:
    def test_good_response_passes_live(self, sample_query, sample_context, good_response):
        from core.guardrail import run_guardrail
        result = run_guardrail(good_response, sample_context, sample_query)
        assert result["verdict"] == "YES"

    def test_financial_violation_caught_live(self, sample_query, sample_context, bad_response_financial):
        from core.guardrail import run_guardrail
        result = run_guardrail(bad_response_financial, sample_context, sample_query)
        assert result["verdict"] == "NO"
        assert len(result.get("violated_rules", [])) > 0
