"""
tests/conftest.py
─────────────────────────────────────────────────────────────────────────────
Pytest fixtures only — auto-discovered and injected by pytest.
Do NOT import this file directly. Use pytest fixture injection instead.

Plain helper functions (make_mock_llm_response, make_guardrail_yes_json, etc.)
live in tests/helpers.py and can be imported normally.
─────────────────────────────────────────────────────────────────────────────
"""

import pytest


# ─── Reusable sample data fixtures ────────────────────────────────────────────

@pytest.fixture
def sample_conversation():
    """A realistic multi-turn Dasher conversation about a missing payment."""
    return [
        {"role": "dasher", "content": "Hi, I have an issue with my pay."},
        {"role": "agent",  "content": "I'm sorry to hear that. Can you tell me more?"},
        {"role": "dasher", "content": "I completed a delivery last night but the $8.50 payment never showed up in my earnings."},
    ]


@pytest.fixture
def sample_query():
    return "Dasher did not receive $8.50 payment for a completed delivery from last night."


@pytest.fixture
def sample_context():
    return (
        "## Problem: Dasher did not receive payment for a completed delivery\n"
        "Resolution: Verify the delivery appears in Order History as Completed. "
        "Submit a Pay Adjustment Request via the Support Portal. "
        "Processing takes 3–5 business days."
    )


@pytest.fixture
def good_response():
    """A policy-compliant, high-quality response."""
    return (
        "I understand your concern regarding the missing payment. "
        "Please check the 'Earnings' section in your Dasher app and select 'Order History' "
        "to verify if the delivery is marked as completed. "
        "If it is listed but the payment is incorrect, you can submit a Pay Adjustment Request "
        "through our Support Portal. These requests are typically processed within 3 to 5 business days."
    )


@pytest.fixture
def bad_response_financial():
    """A response that violates FIN-001 (promises a specific dollar amount)."""
    return (
        "No worries! We will credit $8.50 directly to your account by tomorrow morning. "
        "You should see it reflected in your earnings within 24 hours."
    )


@pytest.fixture
def bad_response_hallucinated():
    """A response that violates ACC-001 (hallucinated facts not in context)."""
    return (
        "Your payment is likely delayed because of the new DoorDash Weekly Holdback Policy "
        "introduced in March 2024, which temporarily holds payments for new Dashers. "
        "You will receive it after completing 50 deliveries."
    )


@pytest.fixture
def bad_response_timeline():
    """A response that violates ESC-002 (invents a specific timeline)."""
    return (
        "Your payment will definitely arrive within 24 hours. "
        "Our system automatically processes missed payments overnight."
    )


@pytest.fixture
def vague_conversation():
    """A conversation where the Dasher's query is too vague to answer safely."""
    return [
        {"role": "dasher", "content": "Something is wrong."},
    ]

