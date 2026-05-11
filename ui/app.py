"""
ui/app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit Frontend — DoorDash Dasher Support Chatbot
Connects to the FastAPI backend at localhost:8000

Features:
  • Real-time chat interface with Dasher / Agent message bubbles
  • YES/NO verdict badge on every response
  • Quality score display (from LLM Judge)
  • Policy sync alert when the Judge detects a policy version drift
  • Sidebar: conversation debug info + pipeline diagnostics
─────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import httpx
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dasher Support Chat",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a192f; }

    /* Chat bubbles */
    .dasher-bubble {
        background: #112240;
        border-left: 4px solid #64ffda;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #ccd6f6;
    }
    .agent-bubble {
        background: #233554;
        border-left: 4px solid #4facfe;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e6f1ff;
    }

    /* Verdict badges */
    .verdict-yes {
        background: #1a472a;
        color: #52c41a;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .verdict-no {
        background: #4a1010;
        color: #ff4d4f;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }

    /* Score bar */
    .score-bar {
        background: #0a3d62;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    /* Input area */
    .stChatInput textarea {
        background-color: #1a1a2e !important;
        color: #f1f1f1 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # Conversation history
if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = []     # Per-turn pipeline diagnostics


# ─── Sidebar — Diagnostics ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛵 Dasher Support")
    st.markdown("**Powered by DoorDash AI**")
    st.divider()

    # Health check
    try:
        health = httpx.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"✅ API Online")
        st.caption(f"Policy: `{health.get('policy_version', 'unknown')}`")
    except Exception:
        st.error("❌ API Offline — start the FastAPI server first")

    st.divider()
    st.markdown("### 📊 Pipeline Diagnostics")

    if st.session_state.pipeline_logs:
        last = st.session_state.pipeline_logs[-1]
        verdict = last.get("verdict", "?")

        if verdict == "YES":
            st.markdown('<span class="verdict-yes">✅ YES — Delivered</span>', unsafe_allow_html=True)
        elif verdict == "NO_REPHRASE":
            st.markdown('<span class="verdict-no">↩ NO — Rephrase Requested</span>', unsafe_allow_html=True)
        elif verdict == "NO_ESCALATED":
            st.markdown('<span class="verdict-no">🚨 NO — Escalated to Human</span>', unsafe_allow_html=True)

        st.caption(f"Attempts: {last.get('attempts', '?')}")
        st.caption(f"Guardrail: {'✅ Passed' if last.get('guardrail_passed') else '❌ Failed'}")

        judge_score = last.get("overall_quality_score")
        if judge_score is not None:
            st.caption(f"Judge Score: {judge_score}/5")
            st.progress(judge_score / 5)

        if last.get("policy_sync_issue"):
            st.warning("⚠️ Policy Sync Issue Detected!")

        st.caption(f"Policy: `{last.get('policy_version', '?')}`")
    else:
        st.caption("No interactions yet.")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.pipeline_logs = []
        st.rerun()

    # Condensed query display
    if st.session_state.pipeline_logs:
        last = st.session_state.pipeline_logs[-1]
        with st.expander("🔍 Condensed Query"):
            st.code(last.get("condensed_query", "N/A"), language=None)


# ─── Main Chat Area ───────────────────────────────────────────────────────────
col_main, col_pad = st.columns([3, 1])

with col_main:
    st.markdown("## 🛵 Dasher Support Chat")
    st.caption("Describe your issue below and our AI will assist you right away.")
    st.divider()

    # Display message history
    for msg in st.session_state.messages:
        if msg["role"] == "dasher":
            st.markdown(
                f'<div class="dasher-bubble">🧑 <strong>You:</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            verdict_badge = ""
            if msg.get("verdict") == "YES":
                verdict_badge = '<span class="verdict-yes">✅ AI Verified</span>'
            elif msg.get("verdict") == "NO_REPHRASE":
                verdict_badge = '<span class="verdict-no">↩ Rephrase Requested</span>'
            elif msg.get("verdict") == "NO_ESCALATED":
                verdict_badge = '<span class="verdict-no">🚨 Escalated</span>'

            score_badge = ""
            if msg.get("score") is not None:
                score_badge = f'<span class="score-bar">Judge Score: {msg["score"]}/5</span>'

            st.markdown(
                f'<div class="agent-bubble">🤖 <strong>Support Agent</strong> '
                f'{verdict_badge} {score_badge}<br><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Describe your issue (e.g. 'I didn't get paid for my last delivery')..."):

        # Add Dasher message
        st.session_state.messages.append({"role": "dasher", "content": prompt})

        # Build conversation for API (only role + content)
        api_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with st.spinner("🔍 Searching knowledge base and generating response..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/chat",
                    json={"conversation_history": api_history},
                    timeout=60,
                )
                data = resp.json()

                # Store pipeline log for sidebar
                st.session_state.pipeline_logs.append({
                    "verdict": data.get("verdict"),
                    "attempts": data.get("attempts"),
                    "guardrail_passed": data.get("guardrail_passed"),
                    "overall_quality_score": data.get("overall_quality_score"),
                    "policy_sync_issue": data.get("policy_sync_issue"),
                    "policy_version": data.get("policy_version"),
                    "condensed_query": data.get("condensed_query", ""),
                    "timestamp": datetime.now().isoformat(),
                })

                # Add agent message with metadata
                st.session_state.messages.append({
                    "role": "agent",
                    "content": data["message"],
                    "verdict": data.get("verdict"),
                    "score": data.get("overall_quality_score"),
                })

            except httpx.ConnectError:
                st.session_state.messages.append({
                    "role": "agent",
                    "content": "❌ Cannot connect to the API server. Is the FastAPI backend running?"
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "agent",
                    "content": f"❌ Backend Error: {e}"
                })

        st.rerun()
