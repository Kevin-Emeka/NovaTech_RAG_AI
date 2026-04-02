"""
RAG Live Project — Step 3: Embed & Store in Vector Database
=============================================================
WHAT WE'RE DOING: Taking our text chunks and converting them into
vectors using an embedding model, then storing them in ChromaDB
for fast similarity search.

KEY CONCEPT: The embedding model (all-MiniLM-L6-v2) and the LLM
(gpt-4o-mini) are TWO DIFFERENT MODELS doing TWO DIFFERENT JOBS.
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import time

# ── 3A: Recreate our chunks (from Step 2) ──────────────────

print("=" * 60)
print("STEP 3: Embedding & Vector Store")
print("=" * 60)

loader = DirectoryLoader(
    "sample_docs", glob="*.docx",
    loader_cls=Docx2txtLoader,
)
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)
print(f"\n{len(chunks)} chunks ready for embedding")


# ── 3B: Load the embedding model ────────────────────────────

print(f"\n{'=' * 60}")
print("LOADING EMBEDDING MODEL")
print(f"{'=' * 60}")

# This is the EMBEDDING model — small, free, runs on CPU
# It converts text → vector (384 dimensions)
# NOT the same as the LLM that generates answers!
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

print("Model: all-MiniLM-L6-v2")
print("Type:  Sentence embedding model (NOT an LLM)")
print("Size:  ~80MB, runs on CPU")
print("Output: 384-dimensional vectors")


# ── 3C: See what an embedding looks like ────────────────────

print(f"\n{'=' * 60}")
print("WHAT AN EMBEDDING LOOKS LIKE")
print(f"{'=' * 60}")

sample_text = "What is the SLA for containing a critical cybersecurity incident?"
sample_vector = embedding_model.embed_query(sample_text)

print(f"\nInput text: \"{sample_text}\"")
print(f"Output vector: {len(sample_vector)} dimensions")
print(f"First 10 values: {[round(v, 4) for v in sample_vector[:10]]}")
print(f"Last 10 values:  {[round(v, 4) for v in sample_vector[-10:]]}")

# Show that similar texts have similar vectors
text_a = "What is the laptop refresh cycle?"
text_b = "How often are company laptops replaced?"
text_c = "What is the maternity leave entitlement?"

vec_a = embedding_model.embed_query(text_a)
vec_b = embedding_model.embed_query(text_b)
vec_c = embedding_model.embed_query(text_c)

# Compute cosine similarity manually
import numpy as np

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"\n{'=' * 60}")
print("COSINE SIMILARITY DEMO (Day 2 callback!)")
print(f"{'=' * 60}")
print(f"\n  A: \"{text_a}\"")
print(f"  B: \"{text_b}\"")
print(f"  C: \"{text_c}\"")
print(f"\n  sim(A, B) = {cosine_sim(vec_a, vec_b):.4f}  <- Similar meaning!")
print(f"  sim(A, C) = {cosine_sim(vec_a, vec_c):.4f}  <- Different topics")
print(f"  sim(B, C) = {cosine_sim(vec_b, vec_c):.4f}  <- Different topics")
print(f"\n  -> The embedding model captures that A and B are about the same thing,")
print(f"    even though they use completely different words.")


# ── 3D: Create the vector store ─────────────────────────────

print(f"\n{'=' * 60}")
print("CREATING VECTOR STORE (ChromaDB)")
print(f"{'=' * 60}")

# Delete previous database if it exists (clean start)
PERSIST_DIR = "./chroma_db"

start_time = time.time()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=PERSIST_DIR,
    collection_name="technova_policies",
)

elapsed = time.time() - start_time
print(f"\n  Embedded and stored {len(chunks)} chunks in {elapsed:.1f} seconds")
print(f"  Persist directory: {PERSIST_DIR}")
print(f"  Collection name: 'technova_policies'")
print(f"  Each chunk is now a 384-dim vector in the database")


# ── 3E: Test retrieval (WITHOUT LLM — just the vector DB) ──

print(f"\n{'=' * 60}")
print("TESTING RETRIEVAL (no LLM yet — just vector search)")
print(f"{'=' * 60}")

test_questions = [
    "How many sick days do I get per year?",
    "What is the minimum password length for NovaTech accounts?",
    "What is the daily hotel limit for Tier 1 city business travel?",
    "How do I report a cybersecurity incident?",
    "What rating do I need for a promotion?",
]

for question in test_questions:
    print(f"\n  Q: \"{question}\"")
    results = vectorstore.similarity_search_with_score(question, k=2)
    for i, (doc, score) in enumerate(results):
        source = Path(doc.metadata["source"]).name
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"    [{i+1}] score={score:.4f} | {source}")
        print(f"        \"{preview}...\"")

print(f"\n{'=' * 60}")
print("KEY OBSERVATION")
print(f"{'=' * 60}")
print("The vector DB retrieves the RIGHT chunks for each question —")
print("leave questions get leave chunks, security questions get security chunks.")
print("This works even though the question wording differs from the document text!")
print("That's the power of semantic search via embeddings.")

print(f"\n[OK] Step 3 complete. Vector store ready. Now let's add the LLM.\n")
