📦 Dasher Support AI: Policy-Guarded RAG System

An intelligent support chatbot designed for DoorDash Dashers, featuring a robust multi-stage RAG (Retrieval-Augmented Generation) pipeline with automated safety guardrails and quality judging.

🚀 Key Features

Intelligent Triage: Uses a Conversation Condenser to transform chat history into precise search queries.

Policy-Synced RAG: Retrieves context from markdown-based policy documents to ensure factual accuracy.

Multi-Layer Safety: * Guardrail: An immediate compliance check that prevents the delivery of unsafe or non-compliant content.

Judge: A Senior QA layer that evaluates responses against 5 quality metrics (Accuracy, Relevance, Grammar, Coherence, and Retrieval Correctness).

Auto-Escalation: Automatically detects when a human specialist is needed (e.g., accidents or safety threats).

Local LLM Integration: Powered entirely by llama3.2:3b running via Ollama for privacy and speed.

🏗️ The Pipeline Architecture

Condenser: Summarizes the user's intent.

Retriever: Finds relevant policy sections in the ChromaDB vector store.

Generator: Crafts a response using the retrieved context.

Guardrail: Validates the response against safety policies.

Judge: Scores the response. If the score is too low or a policy sync issue is found, it triggers a Regenerate or Escalate routing.

🛠️ Tech Stack

Language: Python 3.10+

LLM Framework: LangChain / Ollama

Model: llama3.2:3b

Backend: FastAPI / Uvicorn

Frontend: Streamlit

Vector Store: ChromaDB

🚦 Getting Started

Prerequisites

Ollama installed and running.

Model pulled: ollama pull llama3.2:3b

Python 3.10+

Installation

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/delivery-support-ai.git](https://github.com/YOUR_USERNAME/delivery-support-ai.git)
cd delivery-support-ai

Set up virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Configure Environment:
Create a .env file in the root directory:

OLLAMA_MODEL=llama3.2:3b
PYTHONPATH=.

Running the App

You need to run the Backend and the Frontend in two separate terminals.

Terminal 1: FastAPI Backend

uvicorn api.main:app --host 127.0.0.1 --port 8000

Terminal 2: Streamlit UI

cd ui
streamlit run app.py

📂 Project Structure

├── api/                # FastAPI endpoint logic
├── core/               # The "Brain" (Guardrail, Judge, Condenser, etc.)
├── data/               # Vector store and raw policy documents
├── ui/                 # Streamlit frontend
├── .env                # Environment variables (DO NOT PUSH)
├── .gitignore          # Git exclusion rules
└── README.md           # This file

🛡️ Safety & Quality Logic

The system is configured to be "helpful yet compliant."

Greetings: Automated bypasses for "Hi", "Hello", and "Good morning" to prevent unnecessary processing.

Thresholds: The Judge uses a minimum accuracy threshold of 3.0/5.0. Anything lower is automatically flagged for human escalation to ensure Dashers get only the best advice.
