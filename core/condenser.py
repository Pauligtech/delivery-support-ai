"""
core/condenser.py
─────────────────────────────────────────────────────────────────────────────
Conversation Condenser
Transforms a multi-turn Dasher conversation into a single, precise problem
statement that the RAG retriever can search against accurately.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import logging
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()
llm = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    temperature=0.1, # Lowered for more consistency
)

logger = logging.getLogger(__name__)

# ─── Improved Prompt ─────────────────────────────────────────────────────────
CONDENSER_PROMPT = """\
You are a support triage assistant.
Read the chat and output ONLY one search query sentence describing the core issue.

Rules:
1. No intro text ("Here is the query...").
2. No conversational filler.
3. If the user asks about a policy, mention "policy" and the topic.
4. Output ONLY the query.

━━ CONVERSATION ━━
{conversation}
━━ END ━━

Search Query:"""


def condense_conversation(conversation_history: list[dict]) -> str:
    """
    Summarise conversation into a searchable statement.
    """
    if not conversation_history:
        return "unspecified support issue"

    # Format conversation for the prompt
    formatted = "\n".join(
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_history
    )

    prompt = CONDENSER_PROMPT.format(conversation=formatted)

    try:
        response_text = llm.invoke(prompt).strip()
        
        # Clean up any weird Llama prefixes
        condensed = response_text.replace("Search Query:", "").strip().strip('"')
        
        # CRITICAL DEBUG PRINT - Watch your uvicorn terminal for this!
        print(f"\n[CONDENSER DEBUG] Raw LLM Output: '{response_text}'")
        print(f"[CONDENSER DEBUG] Cleaned Query: '{condensed}'\n")

        # Fallback if LLM returns nothing
        if not condensed:
            raise ValueError("LLM returned an empty string")

        logger.info("Condensed query: %s", condensed)
        return condensed

    except Exception as e:
        logger.error("Condenser error: %s", e)
        # Robust Fallback: Just return the last thing the dasher said
        for msg in reversed(conversation_history):
            if msg.get("role") == "dasher":
                return msg["content"]
        return "delivery issue"