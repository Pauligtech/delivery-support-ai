"""
tests/test_condenser.py
─────────────────────────────────────────────────────────────────────────────
Unit Tests — core/condenser.py (Updated for google-genai SDK)
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.helpers import make_mock_llm_response


class TestCondenseConversation:
    """Test the condense_conversation() function."""

    @patch("core.condenser.client.models.generate_content")
    def test_returns_string(self, mock_gen, sample_conversation):
        mock_gen.return_value = make_mock_llm_response(
            "Dasher did not receive $8.50 payment for a completed delivery."
        )
        from core.condenser import condense_conversation
        result = condense_conversation(sample_conversation)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("core.condenser.client.models.generate_content")
    def test_strips_quotes_from_output(self, mock_gen, sample_conversation):
        mock_gen.return_value = make_mock_llm_response(
            '"Dasher missing payment issue."'
        )
        from core.condenser import condense_conversation
        result = condense_conversation(sample_conversation)
        assert not result.startswith('"'), "Quotes should be stripped from output."

    @patch("core.condenser.client.models.generate_content")
    def test_empty_conversation_returns_fallback(self, mock_gen):
        from core.condenser import condense_conversation
        result = condense_conversation([])
        # Should return fallback string without calling model
        assert isinstance(result, str)
        assert len(result) > 0
        mock_gen.assert_not_called()

    @patch("core.condenser.client.models.generate_content")
    def test_fallback_on_api_error(self, mock_gen, sample_conversation):
        mock_gen.side_effect = Exception("API timeout")
        from core.condenser import condense_conversation
        result = condense_conversation(sample_conversation)
        # Should fall back to last Dasher message instead of crashing
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("core.condenser.client.models.generate_content")
    def test_uses_last_dasher_message_as_fallback(self, mock_gen):
        mock_gen.side_effect = Exception("API timeout")
        from core.condenser import condense_conversation
        conversation = [
            {"role": "dasher", "content": "I have a payment problem."},
            {"role": "agent",  "content": "Can you describe it?"},
            {"role": "dasher", "content": "My last delivery pay is missing."},
        ]
        result = condense_conversation(conversation)
        assert "missing" in result.lower() or "payment" in result.lower()

    @patch("core.condenser.client.models.generate_content")
    def test_model_called_with_correct_args(self, mock_gen, sample_conversation):
        mock_gen.return_value = make_mock_llm_response("Test query.")

        from core.condenser import condense_conversation
        condense_conversation(sample_conversation)

        # Verify model was called with flash
        args, kwargs = mock_gen.call_args
        assert kwargs["model"] == "gemini-1.5-flash"
