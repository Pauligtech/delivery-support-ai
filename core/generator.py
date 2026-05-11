"""
core/generator.py
─────────────────────────────────────────────────────────────────────────────
LLM Response Generator (Updated for google-genai SDK)
Orchestrates the RAG pipeline:
  1. Condense the conversation → searchable query
  2. Retrieve relevant KB chunks from ChromaDB
  3. Inject context + query into Gemini → generate response
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import logging
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from core.condenser import condense_conversation
from core.retriever import retrieve_docs

load_dotenv()
llm = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    temperature=0.4,
)

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a DoorDash Dasher Support Specialist — a knowledgeable, empathetic, and \
professional AI assistant dedicated to helping Dashers resolve their issues quickly.

━━ YOUR PERSONA ━━
- Warm and professional: acknowledge the Dasher's concern before offering a solution
- Precise: provide step-by-step instructions when applicable
- Honest: if the answer is not in the context, admit it and offer escalation
- Concise: target 3–5 sentences unless a detailed step-by-step is required

━━ STRICT RULES ━━
1. Answer ONLY from the provided context — do not invent facts
2. Do NOT promise specific dollar amounts, timelines, or outcomes
3. Do NOT provide legal advice or employment classification statements
4. If the issue involves account deactivation, safety, or legal matters — \
   immediately tell the Dasher you are connecting them with a specialist
5. Always end with a helpful closing (e.g., "Is there anything else I can help you with?")

━━ CONTEXT (retrieved from DoorDash Knowledge Base) ━━
{context}

━━ DASHER'S ISSUE ━━
{query}

━━ YOUR RESPONSE ━━"""


def generate_response(conversation_history: list[dict]) -> dict:
    """
    Full RAG pipeline: condense conversation → retrieve context → generate response.
    """
    # ── Step 1: Condense conversation ────────────────────────────────────
    query = condense_conversation(conversation_history)

    # ── Step 2: Retrieve relevant knowledge ──────────────────────────────
    retrieved_docs = retrieve_docs(query, k=5)

    if not retrieved_docs:
        logger.warning("No documents retrieved for query: %s", query)
        context = "No relevant articles found in the knowledge base."
    else:
        context = "\n\n---\n\n".join(retrieved_docs)

    # ── Step 3: Generate response ─────────────────────────────────────────
    prompt = SYSTEM_PROMPT.format(context=context, query=query)

    try:
        response_text = llm.invoke(prompt).strip()
        logger.info("Generated response (%d chars)", len(response_text))

    except Exception as e:
        logger.error("Generator error: %s", e)
        response_text = (
            "I'm sorry, I'm having trouble processing your request right now. "
            "Please try again or contact DoorDash Support directly."
        )

    return {
        "condensed_query": query,
        "retrieved_docs": retrieved_docs,
        "response": response_text,
    }
