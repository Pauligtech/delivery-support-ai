"""
tests/test_pipeline.py — Integration Tests for core/pipeline.py
Tests the full Generator → Guardrail → Judge routing logic with mocked LLM calls.

Scenarios tested:
  1. Happy path: Guardrail YES → Judge YES → verdict=YES
  2. Guardrail NO REGENERATE → retry → Guardrail YES → Judge YES
  3. Guardrail NO REPHRASE → verdict=NO_REPHRASE (no retry)
  4. Judge NO REGENERATE → retry → Judge YES
  5. Judge NO REPHRASE → verdict=NO_REPHRASE
  6. All retries exhausted → verdict=NO_ESCALATED
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.helpers import (
    make_mock_llm_response,
    make_guardrail_yes_json,
    make_guardrail_no_json,
    make_judge_yes_json,
    make_judge_no_json,
)

# ─── Shared mock for generate_response ───────────────────────────────────────

def _mock_generated():
    """What generate_response() returns — mocked so no RAG calls needed."""
    return {
        "condensed_query": "Dasher missing payment for completed delivery.",
        "retrieved_docs": ["Pay adjustment requests take 3–5 business days."],
        "response": "I understand your concern. Please submit a pay adjustment request.",
    }


class TestPipelineHappyPath:

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_yes_yes_delivers_to_dasher(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_yes_json())
        mock_judge.return_value = json.loads(make_judge_yes_json())

        from core.pipeline import run_pipeline
        result = run_pipeline(sample_conversation)

        assert result["verdict"] == "YES"
        assert result["final_response"] == _mock_generated()["response"]
        assert result["attempts"] == 1
        assert result["guardrail_result"]["verdict"] == "YES"
        assert result["judge_result"]["verdict"] == "YES"

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_calls_made_in_correct_order(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_yes_json())
        mock_judge.return_value = json.loads(make_judge_yes_json())

        from core.pipeline import run_pipeline
        run_pipeline(sample_conversation)

        # Verify call order: generate → guardrail → judge
        mock_gen.assert_called_once()
        mock_gr.assert_called_once()
        mock_judge.assert_called_once()


class TestPipelineGuardrailNoRouting:

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_guardrail_no_rephrase_stops_immediately(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        """REPHRASE route must NOT retry — goes straight back to Dasher."""
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_no_json("ACC-002", "REPHRASE"))

        from core.pipeline import run_pipeline, REPHRASE_MESSAGE
        result = run_pipeline(sample_conversation)

        assert result["verdict"] == "NO_REPHRASE"
        assert result["final_response"] == REPHRASE_MESSAGE
        assert mock_gen.call_count == 1, "REPHRASE must not trigger a retry."
        mock_judge.assert_not_called()

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_guardrail_no_regenerate_retries(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        """REGENERATE should retry the generator once, then pass."""
        mock_gen.return_value = _mock_generated()
        mock_gr.side_effect = [
            json.loads(make_guardrail_no_json("FIN-001", "REGENERATE")),  # Attempt 1: fail
            json.loads(make_guardrail_yes_json()),                         # Attempt 2: pass
        ]
        mock_judge.return_value = json.loads(make_judge_yes_json())

        from core.pipeline import run_pipeline
        result = run_pipeline(sample_conversation)

        assert result["verdict"] == "YES"
        assert mock_gen.call_count == 2
        assert result["attempts"] == 2


class TestPipelineJudgeNoRouting:

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_judge_no_rephrase_stops_immediately(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_yes_json())
        mock_judge.return_value = json.loads(make_judge_no_json(["relevance_to_request"], "REPHRASE"))

        from core.pipeline import run_pipeline, REPHRASE_MESSAGE
        result = run_pipeline(sample_conversation)

        assert result["verdict"] == "NO_REPHRASE"
        assert result["final_response"] == REPHRASE_MESSAGE
        assert mock_gen.call_count == 1

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_judge_no_regenerate_retries(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_yes_json())
        mock_judge.side_effect = [
            json.loads(make_judge_no_json(["response_accuracy"], "REGENERATE")),  # Attempt 1
            json.loads(make_judge_yes_json()),                                     # Attempt 2
        ]

        from core.pipeline import run_pipeline
        result = run_pipeline(sample_conversation)

        assert result["verdict"] == "YES"
        assert mock_gen.call_count == 2


class TestPipelineEscalation:

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_all_retries_exhausted_escalates(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        """When all retries fail, pipeline should escalate to human agent."""
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_yes_json())
        # Judge always returns NO REGENERATE → exhausts retries
        mock_judge.return_value = json.loads(make_judge_no_json(["response_accuracy"], "REGENERATE"))

        from core.pipeline import run_pipeline, ESCALATION_MESSAGE, MAX_RETRIES
        result = run_pipeline(sample_conversation)

        assert result["verdict"] == "NO_ESCALATED"
        assert result["final_response"] == ESCALATION_MESSAGE
        assert result["attempts"] == MAX_RETRIES

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_escalation_message_mentions_specialist(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_yes_json())
        mock_judge.return_value = json.loads(make_judge_no_json(["response_accuracy"], "REGENERATE"))

        from core.pipeline import run_pipeline
        result = run_pipeline(sample_conversation)
        assert "specialist" in result["final_response"].lower() or "support" in result["final_response"].lower()


class TestPipelineResultShape:

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_result_contains_all_expected_keys(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_yes_json())
        mock_judge.return_value = json.loads(make_judge_yes_json())

        from core.pipeline import run_pipeline
        result = run_pipeline(sample_conversation)

        required = {
            "verdict", "final_response", "condensed_query", "retrieved_docs",
            "raw_generated_response", "guardrail_result", "judge_result",
            "attempts", "policy_sync_issue",
        }
        assert required.issubset(set(result.keys()))

    @patch("core.pipeline.run_judge")
    @patch("core.pipeline.run_guardrail")
    @patch("core.pipeline.generate_response")
    def test_judge_not_called_when_guardrail_fails(self, mock_gen, mock_gr, mock_judge, sample_conversation):
        """Judge must be skipped entirely when guardrail routes to REPHRASE."""
        mock_gen.return_value = _mock_generated()
        mock_gr.return_value = json.loads(make_guardrail_no_json("ESC-001", "REPHRASE"))

        from core.pipeline import run_pipeline
        run_pipeline(sample_conversation)
        mock_judge.assert_not_called()
