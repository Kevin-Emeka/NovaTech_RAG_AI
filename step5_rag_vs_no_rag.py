"""
RAG Live Project — Step 5: RAG vs. No-RAG Comparison
======================================================
WHAT WE'RE DOING: Asking the SAME questions WITH and WITHOUT RAG
to demonstrate exactly why RAG matters. This is the most impactful
demo in the entire session.

KEY INSIGHT: Without RAG, the LLM hallucinates confident-sounding
but completely fabricated company policies.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("=" * 60)
print("STEP 5: RAG vs. No-RAG — The Hallucination Demo")
print("=" * 60)

# ── Setup ───────────────────────────────────────────────────

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="technova_policies",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# RAG chain
rag_template = """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt | llm | StrOutputParser()
)

# No-RAG chain (LLM answers from its own knowledge)
no_rag_template = """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question about company policies.

QUESTION: {question}

ANSWER:"""

no_rag_prompt = ChatPromptTemplate.from_template(no_rag_template)
no_rag_chain = (
    {"question": RunnablePassthrough()}
    | no_rag_prompt | llm | StrOutputParser()
)


# ── 5A: Side-by-side comparison ─────────────────────────────

questions = [
    "How many days of earned leave do employees get per year at NovaTech?",
    "What is the minimum password length and how often must it be changed?",
    "What is the SLA for containing a P1 critical cybersecurity incident?",
    "What is the daily hotel limit for business travel to North America?",
    "What is the employee referral bonus for a technical role at Band 3 or above?",
]

for i, question in enumerate(questions, 1):
    print(f"\n{'=' * 60}")
    print(f"QUESTION {i}: {question}")
    print(f"{'=' * 60}")

    # Without RAG
    no_rag_answer = no_rag_chain.invoke(question)
    print(f"\n[X] WITHOUT RAG (LLM guesses):")
    print(f"   {no_rag_answer}")

    # With RAG
    rag_answer = rag_chain.invoke(question)
    print(f"\n[OK] WITH RAG (LLM uses retrieved documents):")
    print(f"   {rag_answer}")

    # Show what was retrieved
    docs = retriever.invoke(question)
    sources = set(Path(d.metadata["source"]).name for d in docs)
    print(f"\n   [doc] Sources used: {', '.join(sources)}")


# ── 5B: The killer demo — question about something specific ─

print(f"\n{'=' * 60}")
print("KILLER DEMO: Highly specific factual question")
print(f"{'=' * 60}")

specific_q = "If I want to work remotely from another country, how many days am I allowed per year and whose approval do I need?"

print(f"\nQ: {specific_q}")

no_rag_ans = no_rag_chain.invoke(specific_q)
print(f"\n[X] WITHOUT RAG:")
print(f"   {no_rag_ans}")

rag_ans = rag_chain.invoke(specific_q)
print(f"\n[OK] WITH RAG:")
print(f"   {rag_ans}")

print(f"\n[doc] GROUND TRUTH (from 06_Remote_Work_Policy.docx):")
print(f"   45 days per year, requires CHRO + Legal + Finance approval")


# ── 5C: Question the documents DON'T answer ─────────────────

print(f"\n{'=' * 60}")
print("EDGE CASE: Question not in the documents")
print(f"{'=' * 60}")

unknown_q = "What is NovaTech's policy on cryptocurrency reimbursement for employee expenses?"

print(f"\nQ: {unknown_q}")

no_rag_ans = no_rag_chain.invoke(unknown_q)
print(f"\n[X] WITHOUT RAG:")
print(f"   {no_rag_ans}")

rag_ans = rag_chain.invoke(unknown_q)
print(f"\n[OK] WITH RAG:")
print(f"   {rag_ans}")

print(f"\n   -> The RAG system correctly says it doesn't have that information,")
print(f"     while the LLM without RAG invents a plausible-sounding policy!")


# ── Summary ─────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("KEY TAKEAWAYS")
print(f"{'=' * 60}")
print("""
1. WITHOUT RAG: The LLM confidently invents policies that sound real
   but are completely fabricated. This is hallucination.

2. WITH RAG: The LLM answers from actual documents and gets the
   specific numbers and details correct.

3. WHEN INFO IS MISSING: RAG systems correctly decline to answer,
   while bare LLMs fabricate an answer.

4. THE FIX IS IN THE INPUT: Same LLM, same temperature, same
   everything — the only difference is the retrieved context.
   RAG doesn't fix the model. It fixes the input.
""")

print(f"[OK] Step 5 complete. You've seen the power of RAG firsthand.\n")
