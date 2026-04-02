"""
RAG Live Project — Step 8: Streamlit Chat Demo
=================================================
WHAT WE'RE DOING: A live chat UI where you can talk to
the RAG system and toggle between RAG and No-RAG mode
to see the difference in real time.

RUN: streamlit run step8_streamlit_demo.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Page config ──────────────────────────────────────────

st.set_page_config(page_title="NovaTech Policy Assistant", page_icon="📋", layout="wide")
st.title("📋 NovaTech Policy Assistant")
st.caption("Ask questions about HR, IT, Finance & Compliance policies — powered by RAG")

# ── Sidebar controls ────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle("Use RAG (Retrieval)", value=True)
    if use_rag:
        st.success("RAG ON — answers grounded in real policy documents")
    else:
        st.warning("RAG OFF — LLM answers from its own knowledge (may hallucinate)")

    st.divider()
    st.markdown("**Sample questions:**")
    st.markdown("""
**HR:**
- How many days of earned leave do I get per year?
- What is the notice period for a Band 5 employee?
- What is the employee referral bonus for tech roles?

**IT:**
- What is the minimum password length?
- How do I report a cybersecurity incident?
- What is the laptop refresh cycle?

**Finance:**
- What is the daily hotel limit for Tier 1 cities?
- What is the max client entertainment spend per person?

**Try this (not in docs):**
- What is NovaTech's cryptocurrency reimbursement policy?
    """)
    st.divider()
    st.caption("Step 8 — RAG Live Project")

# ── Load models (cached so they load only once) ─────────

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

@st.cache_resource
def load_vectorstore():
    embedding_model = load_embedding_model()
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding_model,
        collection_name="technova_policies",
    )

@st.cache_resource
def load_llm():
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)


with st.spinner("Loading models (first time takes ~30 seconds)..."):
    embedding_model = load_embedding_model()
    vectorstore = load_vectorstore()
    llm = load_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Prompt templates ─────────────────────────────────────

rag_template = """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in our policy documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

no_rag_template = """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question about company policies.

QUESTION: {question}

ANSWER:"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
no_rag_prompt = ChatPromptTemplate.from_template(no_rag_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Chat history ─────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📄 Sources used"):
                st.markdown(msg["sources"])

# ── Chat input ───────────────────────────────────────────

if question := st.chat_input("Ask a question about NovaTech policies..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if use_rag:
                # Retrieve relevant chunks
                docs = retriever.invoke(question)
                context = format_docs(docs)

                chain = rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})

                # Build source info
                sources_list = []
                for i, doc in enumerate(docs, 1):
                    source_name = Path(doc.metadata["source"]).name
                    preview = doc.page_content[:150].replace("\n", " ")
                    sources_list.append(f"**{i}. {source_name}**\n> {preview}...")

                sources_text = "\n\n".join(sources_list)
            else:
                chain = no_rag_prompt | llm | StrOutputParser()
                answer = chain.invoke({"question": question})
                sources_text = None

        # Display answer
        mode_tag = "🟢 RAG" if use_rag else "🔴 No RAG"
        st.markdown(f"*{mode_tag}*\n\n{answer}")

        if sources_text:
            with st.expander("📄 Sources used"):
                st.markdown(sources_text)

    # Save to history
    msg_data = {"role": "assistant", "content": f"*{mode_tag}*\n\n{answer}"}
    if sources_text:
        msg_data["sources"] = sources_text
    st.session_state.messages.append(msg_data)
