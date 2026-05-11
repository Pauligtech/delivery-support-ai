"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI Backend — Delivery Support Chatbot
Uses the master pipeline: Generator → Guardrail → Judge → YES/NO routing
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.pipeline import run_pipeline
from core.judge import run_judge, QUALITY_THRESHOLDS
from core.policy import get_policy_version, POLICY_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="DoorDash Delivery Support Chatbot",
    description="RAG-based Dasher support with Guardrail + Judge pipeline.",
    version=POLICY_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_FILE = "logs/interaction_log.jsonl"


# ─── Request / Response Models ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_history: List[Dict]
    # e.g. [{"role": "dasher", "content": "I didn't get paid for my last delivery"}]


class ChatResponse(BaseModel):
    verdict: str                   # "YES" | "NO_REPHRASE" | "NO_ESCALATED"
    message: str                   # Final message to show the Dasher
    guardrail_passed: bool
    judge_passed: Optional[bool]
    overall_quality_score: Optional[float]
    policy_sync_issue: bool
    attempts: int
    policy_version: str


class EvaluateRequest(BaseModel):
    query: str
    context: str
    response: str
    guardrail_result: Dict


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "policy_version": get_policy_version()}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Runs: Generator → Guardrail → Judge → Returns YES/NO verdict + message.
    """
    if not request.conversation_history:
        raise HTTPException(status_code=400, detail="conversation_history cannot be empty.")

    result = run_pipeline(request.conversation_history)

    # ── Log interaction ──────────────────────────────────────────────────
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": result["condensed_query"],
        "context_snippet": result["retrieved_docs"][0][:200] if result["retrieved_docs"] else "",
        "raw_response": result["raw_generated_response"],
        "verdict": result["verdict"],
        "guardrail": result["guardrail_result"],
        "judge": result["judge_result"],
        "attempts": result["attempts"],
        "policy_version": get_policy_version(),
    }
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # ── Build response ───────────────────────────────────────────────────
    guardrail_passed = result["guardrail_result"]["verdict"] == "YES" if result["guardrail_result"] else False
    judge_passed = (
        result["judge_result"]["verdict"] == "YES"
        if result["judge_result"] else None
    )
    overall_score = (
        result["judge_result"].get("overall_score")
        if result["judge_result"] else None
    )

    return ChatResponse(
        verdict=result["verdict"],
        message=result["final_response"],
        guardrail_passed=guardrail_passed,
        judge_passed=judge_passed,
        overall_quality_score=overall_score,
        policy_sync_issue=result["policy_sync_issue"],
        attempts=result["attempts"],
        policy_version=get_policy_version(),
    )


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """
    Standalone Judge endpoint for batch/offline evaluation of a specific interaction.
    Does NOT run the full pipeline — useful for evaluating logged interactions.
    """
    judge_result = run_judge(
        query=request.query,
        context=request.context,
        response=request.response,
        guardrail_result=request.guardrail_result,
    )
    return judge_result


@app.get("/policy")
async def get_policy():
    """Return the current active policy version and thresholds."""
    from core.policy import COMPANY_POLICIES
    return {
        "policy_version": get_policy_version(),
        "quality_thresholds": QUALITY_THRESHOLDS,
        "rules": COMPANY_POLICIES,
    }
