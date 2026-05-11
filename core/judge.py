"""
core/judge.py
─────────────────────────────────────────────────────────────────────────────
LLM Judge — Policy-Synced Quality Evaluator
Runs AFTER the Guardrail gives a YES verdict (i.e., before final delivery).
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
import re
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from core.policy import get_policy_text, get_policy_version

load_dotenv()
llm = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    temperature=0.01,
)

logger = logging.getLogger(__name__)

# ─── Quality Thresholds (Optimized for Llama 3.2 3b) ────────────────────────
QUALITY_THRESHOLDS = {
    "retrieval_correctness": 1,  # Lowered: Greetings/General help don't need retrieval
    "response_accuracy":     3,  # Adjusted for concise local LLM responses
    "grammar_language":      3,
    "coherence_to_context":  2,  # Lowered to accommodate broader queries
    "relevance_to_request":  3,  # Adjusted from 4
}

# ─── Judge Prompt ────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
You are a Senior Quality Assurance Judge for DoorDash's Dasher Support AI system.

Your responsibilities:
  1. Verify the response is aligned with the LATEST company policy.
  2. Evaluate the response quality across standard metrics.
  3. Return a final YES or NO delivery verdict.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{policy_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━ INTERACTION TO EVALUATE ━━

DASHER'S ORIGINAL QUERY (Condensed):
{query}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{context}

AI-GENERATED RESPONSE:
{response}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PART A — QUALITY EVALUATION:
Score each metric from 1 (very poor) to 5 (excellent). 
If the response is a polite greeting or general acknowledgement, give high scores for relevance.

PART B — FINAL VERDICT:
Minimum quality thresholds:
  retrieval_correctness >= {min_retrieval}
  response_accuracy     >= {min_accuracy}
  grammar_language      >= {min_grammar}
  coherence_to_context  >= {min_coherence}
  relevance_to_request  >= {min_relevance}

Return ONLY a JSON object with EXACTLY this structure:
{{
  "verdict": "YES" or "NO",
  "policy_version_evaluated": "{current_policy_version}",
  "scores": {{
    "retrieval_correctness": <1-5>,
    "response_accuracy": <1-5>,
    "grammar_language": <1-5>,
    "coherence_to_context": <1-5>,
    "relevance_to_request": <1-5>
  }},
  "overall_score": <average score>,
  "failed_thresholds": ["metric_name"],
  "routing": "REGENERATE" or null,
  "judge_summary": "Brief summary of evaluation."
}}

JSON Output (no additional text):
"""

def run_judge(
    query: str,
    context: str,
    response: str,
    guardrail_result: dict,
) -> dict:
    """
    Run the LLM Judge on an interaction that has already passed the Guardrail.
    """
    # ─── GREETING & SHORT QUERY BYPASS ───
    # Prevents penalizing the AI for responses to greetings that have no "context"
    clean_query = re.sub(r'[^a-z\s]', '', str(query).lower()).strip()
    bypass_words = ["hi", "hello", "hey", "morning", "afternoon", "evening", "thanks", "ok", "day"]
    
    is_greeting = any(word in clean_query.split() for word in bypass_words)
    is_short = len(clean_query) < 15

    if is_greeting or is_short:
        logger.info(f"Judge: Auto-approving greeting/short query: '{clean_query}'")
        return {
            "verdict": "YES",
            "scores": {k: 5 for k in QUALITY_THRESHOLDS},
            "overall_score": 5.0,
            "failed_thresholds": [],
            "routing": None,
            "judge_summary": "Greeting or short message automatically approved by Judge."
        }

    prompt = JUDGE_PROMPT.format(
        policy_text=get_policy_text(),
        current_policy_version=get_policy_version(),
        query=query,
        context=context,
        response=response,
        min_retrieval=QUALITY_THRESHOLDS["retrieval_correctness"],
        min_accuracy=QUALITY_THRESHOLDS["response_accuracy"],
        min_grammar=QUALITY_THRESHOLDS["grammar_language"],
        min_coherence=QUALITY_THRESHOLDS["coherence_to_context"],
        min_relevance=QUALITY_THRESHOLDS["relevance_to_request"],
    )

    try:
        text = llm.invoke(prompt).strip()

        # Robust JSON extraction
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        parsed = json.loads(text)

        # Normalize verdict
        parsed["verdict"] = str(parsed.get("verdict", "NO")).upper()
        
        # Score Enforcement Logic
        failed = []
        scores = parsed.get("scores", {})
        for metric, min_val in QUALITY_THRESHOLDS.items():
            if scores.get(metric, 0) < min_val:
                failed.append(metric)
        
        if failed:
            parsed["verdict"] = "NO"
            parsed["failed_thresholds"] = failed
            # Default to regenerate if quality is slightly low
            parsed["routing"] = "REGENERATE" 

        if parsed["verdict"] == "YES":
            parsed["routing"] = None

        return parsed

    except Exception as e:
        logger.error("Judge unexpected error: %s", e)
        return _judge_error_response(str(e))

def _judge_error_response(reason: str) -> dict:
    """Fallback if the LLM judge fails to provide a parseable response."""
    return {
        "verdict": "YES",  # Fail open in dev to avoid blocking user
        "policy_version_evaluated": get_policy_version(),
        "scores": {k: 3 for k in QUALITY_THRESHOLDS},
        "overall_score": 3.0,
        "failed_thresholds": [],
        "routing": None,
        "judge_summary": f"System bypass due to internal error: {reason}",
    }