"""
RAG Live Project — Step 7: Advanced Patterns
==============================================
WHAT WE'RE DOING: Two advanced patterns that solve real production
problems. Students can experiment with these during free time.

PATTERN 1: Multi-Query RAG — rephrase the question multiple ways
PATTERN 2: Conversational RAG — handle follow-up questions
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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


# ═══════════════════════════════════════════════════════════
# 7A: MULTI-QUERY RAG
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("7A: MULTI-QUERY RAG")
print("=" * 60)
print("""
Problem: A vague question retrieves poor chunks.
Solution: Generate 3 rephrasings, retrieve for each, combine results.
This dramatically improves recall for ambiguous questions.
""")

def multi_query_rag(question, k=3):
    """Generate multiple query variations and retrieve from all of them."""

    # Step 1: Generate query variations using the LLM
    rephrase_prompt = ChatPromptTemplate.from_template(
        """Generate 3 different versions of the following question.
Each version should use different words but ask about the same thing.
Return ONLY the 3 questions, one per line, no numbering.

Original question: {question}"""
    )

    chain = rephrase_prompt | llm | StrOutputParser()
    variations_text = chain.invoke({"question": question})
    variations = [q.strip() for q in variations_text.strip().split("\n") if q.strip()]

    print(f"  Original:    \"{question}\"")
    print(f"  Variations:")
    for v in variations:
        print(f"    -> \"{v}\"")

    # Step 2: Retrieve for original + all variations
    all_queries = [question] + variations
    all_docs = []
    seen_content = set()

    for query in all_queries:
        docs = retriever.invoke(query)
        for doc in docs:
            # Deduplicate by content
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_docs.append(doc)

    print(f"\n  Retrieved {len(all_docs)} unique chunks (from {len(all_queries)} queries)")

    # Step 3: Generate answer from all retrieved context
    context = "\n\n".join(d.page_content for d in all_docs[:6])  # Cap at 6

    answer_prompt = ChatPromptTemplate.from_template(
        """Answer based ONLY on the context. Be specific.
If the context doesn't contain the answer, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    )

    answer_chain = answer_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke({"context": context, "question": question})

    sources = set(Path(d.metadata["source"]).name for d in all_docs[:6])
    return answer, sources


# Demo: vague question that benefits from multi-query
print("\n--- Demo: Vague question ---")
answer, sources = multi_query_rag("What should I know about data security and encryption?")
print(f"\n  Answer: {answer}")
print(f"  Sources: {', '.join(sources)}")

print("\n--- Demo: Specific question ---")
answer, sources = multi_query_rag("What are the travel expense limits and reimbursement rules?")
print(f"\n  Answer: {answer}")
print(f"  Sources: {', '.join(sources)}")


# ═══════════════════════════════════════════════════════════
# 7B: CONVERSATIONAL RAG (handling follow-up questions)
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("7B: CONVERSATIONAL RAG")
print(f"{'=' * 60}")
print("""
Problem: "What about the second point?" — the retriever doesn't
know what "the second point" refers to without conversation history.
Solution: Rewrite follow-up questions using chat history first.
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# The contextualizer: rewrites follow-up questions into standalone ones
contextualize_prompt = ChatPromptTemplate.from_template(
    """Given the chat history and the latest user question, rewrite the
question to be a standalone question that doesn't need the chat history
to understand. Do NOT answer the question, just rewrite it.
If the question is already standalone, return it as-is.

CHAT HISTORY:
{chat_history}

LATEST QUESTION: {question}

STANDALONE QUESTION:"""
)

answer_prompt = ChatPromptTemplate.from_template(
    """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer based ONLY on the provided context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)


def conversational_rag(question, chat_history=None):
    """RAG that handles follow-up questions using chat history."""
    if chat_history is None:
        chat_history = []

    # Step 1: If there's history, rewrite the question
    if chat_history:
        history_text = "\n".join(
            f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}"
            for i, msg in enumerate(chat_history)
        )

        rewrite_chain = contextualize_prompt | llm | StrOutputParser()
        standalone_question = rewrite_chain.invoke({
            "chat_history": history_text,
            "question": question,
        })
        print(f"  [Rewritten] \"{question}\" -> \"{standalone_question}\"")
    else:
        standalone_question = question
        print(f"  [Standalone] \"{question}\"")

    # Step 2: Retrieve using the standalone question
    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    # Step 3: Generate answer
    chain = answer_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": standalone_question})

    return answer


# Simulate a multi-turn conversation
print("\n--- Simulating a conversation ---\n")

history = []

# Turn 1
q1 = "What is the cybersecurity incident response process at NovaTech?"
print(f"User: {q1}")
a1 = conversational_rag(q1, history)
print(f"Assistant: {a1}\n")
history.extend([q1, a1])

# Turn 2 (follow-up — needs context from turn 1)
q2 = "What about the response time SLA for critical incidents?"
print(f"User: {q2}")
a2 = conversational_rag(q2, history)
print(f"Assistant: {a2}\n")
history.extend([q2, a2])

# Turn 3 (another follow-up)
q3 = "Who leads the CSIRT team and what is the security hotline number?"
print(f"User: {q3}")
a3 = conversational_rag(q3, history)
print(f"Assistant: {a3}\n")

print(f"\n{'=' * 60}")
print("KEY INSIGHT")
print(f"{'=' * 60}")
print("""
The contextualize step is crucial:
  "What about the internet speed?" alone would retrieve random chunks.
  After rewriting: "What are the internet speed requirements for
  remote work at TechNova?" retrieves the right chunks.

This is how ChatGPT-style RAG applications handle multi-turn
conversations — they rewrite each question before retrieval.
""")

print(f"[OK] Step 7 complete. You've seen advanced RAG patterns.\n")
print(f"{'=' * 60}")
print("EXPERIMENT TIME!")
print(f"{'=' * 60}")
print("""
Try these on your own:

1. Add your own documents to sample_docs/ and re-index
2. Try different chunk sizes (200, 1000) and compare answers
3. Ask questions that span multiple documents
4. Try to break it — find questions it gets wrong, then debug
5. Modify the system prompt and see how answers change
""")
