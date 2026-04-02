"""
RAG Live Project — Step 4: Complete RAG Chain
===============================================
WHAT WE'RE DOING: Connecting the retriever to an LLM to build
the complete RAG pipeline. This is where everything comes together.

KEY CONCEPT: We first build the prompt MANUALLY so you see exactly
what the LLM receives. Then we wire it up with LangChain.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "Set OPENAI_API_KEY in .env file!"

print("=" * 60)
print("STEP 4: Complete RAG Chain")
print("=" * 60)


# ── 4A: Rebuild our vector store (or load existing) ─────────

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

PERSIST_DIR = "./chroma_db"

# Try to load existing DB, or create fresh
if Path(PERSIST_DIR).exists():
    print("\nLoading existing vector store...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_model,
        collection_name="technova_policies",
    )
    print(f"  Loaded. Collection has {vectorstore._collection.count()} chunks.")
else:
    print("\nCreating vector store from scratch...")
    loader = DirectoryLoader(
        "sample_docs", glob="*.docx",
        loader_cls=Docx2txtLoader,
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(loader.load())
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embedding_model,
        persist_directory=PERSIST_DIR, collection_name="technova_policies",
    )
    print(f"  Created with {len(chunks)} chunks.")

# Create retriever (top 3 most similar chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ── 4B: THE MANUAL APPROACH (see exactly what the LLM gets) ─

print(f"\n{'=' * 60}")
print("4B: BUILDING THE PROMPT MANUALLY")
print("    (so you see exactly what the LLM receives)")
print(f"{'=' * 60}")

question = "What is the minimum password length and expiry policy for NovaTech accounts?"

# Step 1: Retrieve relevant chunks
retrieved_docs = retriever.invoke(question)

print(f"\nQuestion: \"{question}\"")
print(f"\nRetrieved {len(retrieved_docs)} chunks:")
for i, doc in enumerate(retrieved_docs):
    source = Path(doc.metadata["source"]).name
    print(f"\n  --- Chunk {i+1} (from {source}) ---")
    print(f"  {doc.page_content[:200]}...")

# Step 2: Build the augmented prompt manually
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

manual_prompt = f"""You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in the available documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

print(f"\n{'=' * 60}")
print("THE ACTUAL PROMPT SENT TO THE LLM")
print("(This is the 'augmented' in Retrieval-Augmented Generation)")
print(f"{'=' * 60}")
print(f"\n{manual_prompt[:1000]}")
if len(manual_prompt) > 1000:
    print(f"... [{len(manual_prompt) - 1000} more characters] ...")
print(f"\nTotal prompt length: {len(manual_prompt)} characters")

# Step 3: Send to LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

print(f"\n{'=' * 60}")
print("LLM RESPONSE (manual approach)")
print(f"{'=' * 60}")

from langchain_core.messages import HumanMessage
response = llm.invoke([HumanMessage(content=manual_prompt)])
print(f"\n{response.content}")


# ── 4C: THE LANGCHAIN APPROACH (same thing, cleaner code) ───

print(f"\n{'=' * 60}")
print("4C: LANGCHAIN RAG CHAIN (production-ready version)")
print(f"{'=' * 60}")

# Define the prompt template
template = """You are a helpful policy assistant for NovaTech Solutions Pvt. Ltd.
Answer the employee's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have that information in the available documents."
Keep your answer concise and specific.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

prompt = ChatPromptTemplate.from_template(template)

# Helper: format retrieved docs into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the chain: retriever → format → prompt → LLM → parse output
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Test it!
answer = rag_chain.invoke(question)
print(f"\nQ: {question}")
print(f"A: {answer}")


# ── 4D: Test with multiple questions ────────────────────────

print(f"\n{'=' * 60}")
print("4D: TESTING WITH MULTIPLE QUESTIONS")
print(f"{'=' * 60}")

test_questions = [
    "How many days of earned leave do employees get per year?",
    "What is the SLA for a P1 critical cybersecurity incident?",
    "What happens if I get a rating of 2 in my performance review?",
    "What is the daily hotel limit for international travel to North America?",
    "What is the laptop refresh cycle at NovaTech?",
    "How do I report a security incident and what is the hotline number?",
    "What are the core working hours for remote employees?",
]

for q in test_questions:
    answer = rag_chain.invoke(q)
    print(f"\nQ: {q}")
    print(f"A: {answer}")
    print("-" * 40)


# ── 4E: Add source tracking ────────────────────────────────

print(f"\n{'=' * 60}")
print("4E: RAG WITH SOURCE CITATIONS")
print(f"{'=' * 60}")

# Modified chain that also returns source documents
from langchain_core.runnables import RunnableParallel

# Retrieve docs and keep them alongside the answer
def rag_with_sources(question):
    """RAG that returns both the answer and the source chunks."""
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    sources = set()
    for doc in docs:
        sources.add(Path(doc.metadata["source"]).name)

    return {
        "question": question,
        "answer": answer,
        "sources": list(sources),
        "num_chunks_used": len(docs),
    }

# Demo with sources
result = rag_with_sources("What happens if I lose my company laptop and how soon must I report it?")
print(f"\nQ: {result['question']}")
print(f"A: {result['answer']}")
print(f"Sources: {', '.join(result['sources'])}")
print(f"Chunks used: {result['num_chunks_used']}")

print(f"\n[OK] Step 4 complete. You have a working RAG system!\n")
