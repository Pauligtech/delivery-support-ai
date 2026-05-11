"""
tests/helpers.py
─────────────────────────────────────────────────────────────────────────────
Shared test helper functions (NOT pytest fixtures).
Import these directly in test files:
    from tests.helpers import make_mock_llm_response, make_guardrail_yes_json

Fixtures (sample_conversation, good_response, etc.) live in conftest.py
and are injected automatically by pytest — do NOT import them manually.
─────────────────────────────────────────────────────────────────────────────
"""

import json
from unittest.mock import MagicMock


def make_mock_llm_response(text: str) -> MagicMock:
    """Create a mock Gemini GenerativeModel response object with .text attribute."""
    mock_response = MagicMock()
    mock_response.text = text
    return mock_response


def make_guardrail_yes_json() -> str:
    """Return a JSON string representing a passing guardrail result."""
    return json.dumps({
        "verdict": "YES",
        "policy_version_checked": "1.2.0 (2026-05-05)",
        "violated_rules": [],
        "routing": None,
        "violation_summary": "None",
        "compliance_notes": "All policy rules satisfied.",
    })


def make_guardrail_no_json(rule: str = "FIN-001", routing: str = "REGENERATE") -> str:
    """Return a JSON string representing a failing guardrail result."""
    return json.dumps({
        "verdict": "NO",
        "policy_version_checked": "1.2.0 (2026-05-05)",
        "violated_rules": [rule],
        "routing": routing,
        "violation_summary": f"Response violated {rule}.",
        "compliance_notes": "See violated_rules for details.",
    })


def make_judge_yes_json(scores: dict | None = None) -> str:
    """Return a JSON string representing a passing judge result."""
    scores = scores or {
        "retrieval_correctness": 4,
        "response_accuracy": 5,
        "grammar_language": 4,
        "coherence_to_context": 5,
        "relevance_to_request": 5,
    }
    return json.dumps({
        "verdict": "YES",
        "policy_version_evaluated": "1.2.0 (2026-05-05)",
        "policy_sync_issue": False,
        "policy_sync_note": None,
        "newly_violated_rules": [],
        "scores": scores,
        "overall_score": round(sum(scores.values()) / len(scores), 1),
        "failed_thresholds": [],
        "routing": None,
        "judge_summary": "Response is accurate, policy-compliant, and directly addresses the Dasher's issue.",
    })


def make_judge_no_json(failed: list[str], routing: str = "REGENERATE") -> str:
    """Return a JSON string representing a failing judge result."""
    scores = {
        "retrieval_correctness": 4,
        "response_accuracy": 2,
        "grammar_language": 4,
        "coherence_to_context": 3,
        "relevance_to_request": 3,
    }
    return json.dumps({
        "verdict": "NO",
        "policy_version_evaluated": "1.2.0 (2026-05-05)",
        "policy_sync_issue": False,
        "policy_sync_note": None,
        "newly_violated_rules": [],
        "scores": scores,
        "overall_score": round(sum(scores.values()) / len(scores), 1),
        "failed_thresholds": failed,
        "routing": routing,
        "judge_summary": "Response failed quality thresholds on response_accuracy.",
    })
