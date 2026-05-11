"""
core/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Master Orchestration Pipeline
Connects: Generator → Guardrail → Judge → Routing Decision

YES/NO Flow:
  Generator produces a response
      ↓
  Guardrail checks policy compliance
      ├── NO (REPHRASE)   → Ask Dasher to rephrase query
      ├── NO (REGENERATE) → Retry generator (up to MAX_RETRIES)
      └── YES             → Pass to Judge
                                ↓
                            Judge evaluates quality + policy sync
                                ├── NO (REPHRASE)   → Ask Dasher to rephrase
                                ├── NO (REGENERATE) → Retry generator
                                └── YES             → ✅ Deliver to Dasher
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from core.generator import generate_response
from core.guardrail import run_guardrail
from core.judge import run_judge

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────
MAX_RETRIES = 2   # Max regeneration attempts before escalating to human

# ─── Rephrase messages sent back to the Dasher ───────────────────────────────
REPHRASE_MESSAGE = (
    "I want to make sure I give you the most accurate help possible. "
    "Could you please provide a bit more detail about your issue? "
    "For example: What type of delivery was it? What happened specifically? "
    "This will help me find the right solution for you."
)

# ─── Escalation message when all retries are exhausted ───────────────────────
ESCALATION_MESSAGE = (
    "I sincerely apologize for the inconvenience. I wasn't able to generate "
    "a response that fully meets our quality standards for your specific issue. "
    "I'm connecting you with a DoorDash Support Specialist who can assist you "
    "directly. Please hold — someone will be with you shortly."
)


def run_pipeline(conversation_history: list[dict]) -> dict:
    """
    Execute the full end-to-end pipeline for a Dasher support interaction.

    Args:
        conversation_history: List of message dicts — [{"role": "dasher"/"agent", "content": "..."}]

    Returns:
        dict with keys:
          verdict              → "YES" | "NO_REPHRASE" | "NO_ESCALATED"
          final_response       → str  (message to display to the Dasher)
          condensed_query      → str
          retrieved_docs       → list[str]
          guardrail_result     → dict
          judge_result         → dict | None
          attempts             → int (how many generation attempts were made)
          policy_sync_issue    → bool (whether the Judge caught a policy drift)
    """
    attempt = 0
    guardrail_result = None
    judge_result = None
    last_generated = None

    while attempt < MAX_RETRIES:
        attempt += 1
        logger.info("Pipeline attempt %d/%d", attempt, MAX_RETRIES)

        # ── Step 1: Generate response via RAG ─────────────────────────────
        generated = generate_response(conversation_history)
        last_generated = generated
        query   = generated["condensed_query"]
        context = "\n\n".join(generated["retrieved_docs"])
        response_text = generated["response"]

        logger.info("Generated response (attempt %d): %s...", attempt, response_text[:80])

        # ── Step 2: Guardrail — Policy Compliance Check ───────────────────
        guardrail_result = run_guardrail(
            response=response_text,
            context=context,
            query=query,
        )
        logger.info("Guardrail verdict: %s", guardrail_result["verdict"])

        if guardrail_result["verdict"] == "NO":
            routing = guardrail_result.get("routing")

            if routing == "REPHRASE":
                # Dasher query is the root cause — no point retrying
                logger.info("Guardrail → REPHRASE. Stopping pipeline.")
                return _build_result(
                    verdict="NO_REPHRASE",
                    final_response=REPHRASE_MESSAGE,
                    generated=last_generated,
                    guardrail_result=guardrail_result,
                    judge_result=None,
                    attempts=attempt,
                )

            # REGENERATE — retry the loop
            logger.info("Guardrail → REGENERATE. Retrying (attempt %d).", attempt)
            continue

        # ── Step 3: Judge — Quality + Policy Sync Check ───────────────────
        judge_result = run_judge(
            query=query,
            context=context,
            response=response_text,
            guardrail_result=guardrail_result,
        )
        logger.info(
            "Judge verdict: %s | Score: %s | Policy sync issue: %s",
            judge_result["verdict"],
            judge_result.get("overall_score"),
            judge_result.get("policy_sync_issue"),
        )

        if judge_result["verdict"] == "YES":
            # ✅ Both Guardrail and Judge approved — safe to deliver
            return _build_result(
                verdict="YES",
                final_response=response_text,
                generated=last_generated,
                guardrail_result=guardrail_result,
                judge_result=judge_result,
                attempts=attempt,
            )

        # Judge said NO
        routing = judge_result.get("routing")
        if routing == "REPHRASE":
            logger.info("Judge → REPHRASE. Stopping pipeline.")
            return _build_result(
                verdict="NO_REPHRASE",
                final_response=REPHRASE_MESSAGE,
                generated=last_generated,
                guardrail_result=guardrail_result,
                judge_result=judge_result,
                attempts=attempt,
            )

        # REGENERATE — retry the loop
        logger.info("Judge → REGENERATE. Retrying (attempt %d).", attempt)

    # ── All retries exhausted → Escalate to human ─────────────────────────
    logger.warning("All %d attempts exhausted. Escalating to human agent.", MAX_RETRIES)
    return _build_result(
        verdict="NO_ESCALATED",
        final_response=ESCALATION_MESSAGE,
        generated=last_generated,
        guardrail_result=guardrail_result,
        judge_result=judge_result,
        attempts=attempt,
    )


def _build_result(
    verdict: str,
    final_response: str,
    generated: dict,
    guardrail_result: dict,
    judge_result: dict | None,
    attempts: int,
) -> dict:
    """Assemble the standardised pipeline result dict."""
    return {
        "verdict": verdict,
        "final_response": final_response,
        "condensed_query": generated.get("condensed_query", ""),
        "retrieved_docs": generated.get("retrieved_docs", []),
        "raw_generated_response": generated.get("response", ""),
        "guardrail_result": guardrail_result,
        "judge_result": judge_result,
        "attempts": attempts,
        "policy_sync_issue": (
            judge_result.get("policy_sync_issue", False) if judge_result else False
        ),
    }
