"""
tests/test_judge.py — Unit Tests for core/judge.py (Updated for google-genai SDK)
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.judge import QUALITY_THRESHOLDS
from tests.helpers import (
    make_mock_llm_response,
    make_guardrail_yes_json,
    make_judge_yes_json,
    make_judge_no_json,
)


def _guardrail_yes():
    return json.loads(make_guardrail_yes_json())


class TestJudgeOutputStructure:

    @patch("core.judge.client.models.generate_content")
    def test_yes_verdict_has_required_keys(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.return_value = make_mock_llm_response(make_judge_yes_json())
        from core.judge import run_judge
        result = run_judge(sample_query, sample_context, good_response, _guardrail_yes())
        required = {"verdict", "scores", "routing", "judge_summary"}
        assert required.issubset(set(result.keys()))

    @patch("core.judge.client.models.generate_content")
    def test_all_five_metric_scores_present(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.return_value = make_mock_llm_response(make_judge_yes_json())
        from core.judge import run_judge
        result = run_judge(sample_query, sample_context, good_response, _guardrail_yes())
        expected = {"retrieval_correctness", "response_accuracy", "grammar_language",
                    "coherence_to_context", "relevance_to_request"}
        assert expected == set(result["scores"].keys())

    @patch("core.judge.client.models.generate_content")
    def test_routing_is_null_on_yes(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.return_value = make_mock_llm_response(make_judge_yes_json())
        from core.judge import run_judge
        result = run_judge(sample_query, sample_context, good_response, _guardrail_yes())
        assert result["routing"] is None


class TestJudgeQualityThresholds:

    @patch("core.judge.client.models.generate_content")
    def test_high_scores_yield_yes(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.return_value = make_mock_llm_response(
            make_judge_yes_json({k: 5 for k in QUALITY_THRESHOLDS})
        )
        from core.judge import run_judge
        result = run_judge(sample_query, sample_context, good_response, _guardrail_yes())
        assert result["verdict"] == "YES"

    @patch("core.judge.client.models.generate_content")
    def test_low_accuracy_yields_no(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.return_value = make_mock_llm_response(
            make_judge_no_json(["response_accuracy"], "REGENERATE")
        )
        from core.judge import run_judge
        result = run_judge(sample_query, sample_context, good_response, _guardrail_yes())
        assert result["verdict"] == "NO"


class TestJudgeErrorHandling:

    @patch("core.judge.client.models.generate_content")
    def test_api_error_defaults_to_no(self, mock_gen, sample_query, sample_context, good_response):
        mock_gen.side_effect = Exception("timeout")
        from core.judge import run_judge
        result = run_judge(sample_query, sample_context, good_response, _guardrail_yes())
        assert result["verdict"] == "NO"
        assert result["routing"] == "REGENERATE"
        assert result["overall_score"] == 0.0


@pytest.mark.live
class TestJudgeLive:
    def test_good_response_passes(self, sample_query, sample_context, good_response):
        from core.judge import run_judge
        result = run_judge(sample_query, sample_context, good_response, _guardrail_yes())
        assert result["verdict"] == "YES"
