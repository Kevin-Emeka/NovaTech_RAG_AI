"""
RAG Live Project — Step 6: Making RAG Production-Ready
========================================================
WHAT WE'RE DOING: The basic RAG works. Now we learn what breaks
in production and how to fix it.

TOPICS: Chunking experiments, retrieval debugging, failure modes,
metadata filtering, evaluation basics.
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

load_dotenv()

print("=" * 60)
print("STEP 6: Production-Ready RAG")
print("=" * 60)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


# ═══════════════════════════════════════════════════════════
# 6A: CHUNKING EXPERIMENTS — How chunk size affects answers
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("6A: HOW CHUNK SIZE AFFECTS ANSWER QUALITY")
print(f"{'=' * 60}")

loader = DirectoryLoader(
    "sample_docs", glob="*.docx",
    loader_cls=Docx2txtLoader,
)
raw_docs = loader.load()

question = "What are the increment percentages for each performance rating level?"

rag_template = """Answer based ONLY on the context. Be specific with numbers.
If the context doesn't contain the answer, say "Not found in context."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

prompt = ChatPromptTemplate.from_template(rag_template)

for chunk_size in [200, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=50
    )
    chunks = splitter.split_documents(raw_docs)

    # Create temporary vectorstore for this experiment
    temp_vs = Chroma.from_documents(
        documents=chunks, embedding=embedding_model,
        collection_name=f"experiment_{chunk_size}",
    )
    temp_retriever = temp_vs.as_retriever(search_kwargs={"k": 3})

    # Get retrieved chunks
    docs = temp_retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    # Get answer
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    print(f"\n  chunk_size={chunk_size}:")
    print(f"  Retrieved {len(docs)} chunks, total context: {len(context)} chars")
    print(f"  Answer: {answer[:200]}...")

    # Cleanup
    temp_vs.delete_collection()

print(f"\n  -> Notice: too-small chunks may miss the increment table entirely")
print(f"  -> Too-large chunks include irrelevant content that can confuse the LLM")


# ═══════════════════════════════════════════════════════════
# 6B: RETRIEVAL DEBUGGING — Is it a retrieval or generation problem?
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("6B: DEBUGGING — Retrieval vs. Generation Problems")
print(f"{'=' * 60}")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
    collection_name="technova_policies",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def debug_rag(question):
    """Debug function: shows what was retrieved and the final answer."""
    print(f"\n  Q: \"{question}\"")

    # Step 1: Check what was retrieved
    docs = retriever.invoke(question)
    print(f"\n  RETRIEVED CHUNKS:")
    for i, doc in enumerate(docs):
        source = Path(doc.metadata["source"]).name
        print(f"    [{i+1}] {source}: \"{doc.page_content[:100]}...\"")

    # Step 2: Check if the answer is IN the retrieved chunks
    context = "\n\n".join(d.page_content for d in docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    print(f"\n  ANSWER: {answer}")

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    sources = set(Path(d.metadata["source"]).name for d in docs)
    print(f"    Sources used: {', '.join(sources)}")
    print(f"    -> If wrong source: RETRIEVAL problem (fix embeddings/chunks)")
    print(f"    -> If right source but wrong answer: GENERATION problem (fix prompt)")

# Test with a tricky question
debug_rag("What is the notice period for a Band 5 Senior Manager?")
print()
debug_rag("What security tools does NovaTech use for endpoint protection?")


# ═══════════════════════════════════════════════════════════
# 6C: METADATA FILTERING — Narrow retrieval to specific docs
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("6C: METADATA FILTERING")
print(f"{'=' * 60}")

# Rebuild with richer metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

enriched_chunks = []
for doc in raw_docs:
    source_name = Path(doc.metadata["source"]).name
    # Add document category based on filename
    category_map = {
        "01_Employee_Handbook_Code_of_Conduct.docx": "Compliance",
        "02_Recruitment_Onboarding_Policy.docx": "HR",
        "03_Leave_Attendance_Policy.docx": "HR",
        "04_Performance_Management_Policy.docx": "HR",
        "05_Compensation_Benefits_Policy.docx": "Finance",
        "06_Remote_Work_Policy.docx": "HR",
        "07_Grievance_Disciplinary_Policy.docx": "HR",
        "08_POSH_Diversity_Inclusion_Policy.docx": "Compliance",
        "09_Separation_Offboarding_Policy.docx": "HR",
        "10_IT_Acceptable_Use_Policy.docx": "IT",
        "11_Data_Security_Privacy_Policy.docx": "IT",
        "12_IT_Asset_Management_Policy.docx": "IT",
        "13_Cybersecurity_Incident_Response_Policy.docx": "IT",
        "14_Learning_Development_Policy.docx": "HR",
        "15_Travel_Expense_Policy.docx": "Finance",
    }
    category = category_map.get(source_name, "General")

    chunks = splitter.split_documents([doc])
    for chunk in chunks:
        chunk.metadata["category"] = category
        chunk.metadata["document"] = source_name
    enriched_chunks.extend(chunks)

# Create new vectorstore with metadata
meta_vs = Chroma.from_documents(
    documents=enriched_chunks,
    embedding=embedding_model,
    collection_name="technova_with_metadata",
)

# Search with metadata filter
print("\nSearch WITHOUT filter (all documents):")
results = meta_vs.similarity_search("data encryption and password policy", k=3)
for r in results:
    print(f"  [{r.metadata['category']}] {r.metadata['document']}: {r.page_content[:80]}...")

print("\nSearch WITH filter (IT category only):")
results = meta_vs.similarity_search(
    "data encryption and password policy",
    k=3,
    filter={"category": "IT"},
)
for r in results:
    print(f"  [{r.metadata['category']}] {r.metadata['document']}: {r.page_content[:80]}...")

print("\n  -> Metadata filters let you narrow search to specific doc types")
print("  -> Useful for: 'search only HR docs' or 'search only docs from 2025'")

meta_vs.delete_collection()


# ═══════════════════════════════════════════════════════════
# 6D: COMMON FAILURE MODES
# ═══════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("6D: COMMON RAG FAILURE MODES")
print(f"{'=' * 60}")

failures = [
    ("Wrong chunks retrieved",
     "Embedding model can't distinguish domain-specific terms",
     "Use domain-specific embeddings or add metadata filters"),

    ("Right chunks but wrong answer",
     "Prompt doesn't constrain the LLM enough",
     "Improve system prompt, add 'answer ONLY from context'"),

    ("Answer spreads across multiple chunks",
     "Information was split at a chunk boundary",
     "Increase chunk size or overlap for that document type"),

    ("Stale answers after document update",
     "Vectors not re-computed after source docs changed",
     "Re-index changed documents (track doc hashes)"),

    ("Slow response time",
     "Too many chunks retrieved or context too large",
     "Reduce K, use metadata pre-filtering, compress context"),
]

for i, (mode, cause, fix) in enumerate(failures, 1):
    print(f"\n  {i}. {mode}")
    print(f"     Cause: {cause}")
    print(f"     Fix:   {fix}")

print(f"\n[OK] Step 6 complete. You know how to debug and improve RAG.\n")
