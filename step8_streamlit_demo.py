"""
RAG Live Project — Agentic Streamlit Chat Demo
=================================================
WHAT WE'RE DOING: A native ChatGPT-like interface seamlessly 
powered by our LangGraph Agent.

RUN: streamlit run step8_streamlit_demo.py
"""

import os
from dotenv import load_dotenv
import streamlit as st

# Import our autonomous pipeline
from rag_pipeline import create_rag_pipeline, create_agent

load_dotenv()

# ── Page config ──────────────────────────────────────────

st.set_page_config(page_title="NovaTech Agentic Assistant", page_icon="📋", layout="wide")
st.title("📋 NovaTech Agentic RAG Assistant")
st.caption("Ask questions about HR, IT, Finance & Compliance policies — powered by an autonomous LangGraph Search Agent!")

# ── Sidebar controls ────────────────────────────────────

with st.sidebar:
    st.header("App Mode")
    st.success("🤖 AGENT ONLINE — The model dynamically decides when to fetch facts from company policies!")

    st.divider()
    st.markdown("**Sample queries reflecting agent reasoning:**")
    st.markdown("""
- What is the difference between IT policy and HR policy?
- How many days of earned leave do I get?
- Can I use a corporate card to buy a laptop if I break mine?
- Explain the password sharing rule.
    """)
    st.divider()
    st.caption("Step 8 — Agentic Streamlit Live Project")

# ── Agent Loading ───────────────────────────────────────

@st.cache_resource
def load_application_agent():
    # Only load chroma db and LLM wrapper once
    _, vectorstore = create_rag_pipeline()
    return create_agent(vectorstore)

with st.spinner("Initializing LangGraph Agent..."):
    agent = load_application_agent()

# ── Chat history ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-render chat messages gracefully matching ChatGPT UI
for msg in st.session_state.messages:
    # We display standard roles natively ('user' or 'assistant')
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Interaction Runtime ─────────────────────────────────

if question := st.chat_input("Ask a question about NovaTech operations..."):
    # Render user command
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Process and Render Agent Output natively
    with st.chat_message("assistant"):
        with st.spinner("Thinking & Searching..."):
            
            # Map native Streamlit dictionary to LangGraph compliant tuple pairs
            langgraph_messages = [
                (m["role"], m["content"]) for m in st.session_state.messages
            ]

            try:
                # Hand conversation over to the Agent 
                # (Notice we no longer manually `retriever.invoke`, the agent does it!)
                result = agent.invoke({"messages": langgraph_messages})
                
                # Fetch final synthesized answer
                answer = result["messages"][-1].content
            except Exception as e:
                answer = f"⚠️ Agent crashed: {e}"

        st.markdown(answer)

    # Attach response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
