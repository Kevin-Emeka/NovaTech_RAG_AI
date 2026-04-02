"""
RAG Live Project — Step 2: Chunking
=====================================
WHAT WE'RE DOING: Splitting documents into smaller pieces (chunks)
that can be individually retrieved. This is the most underrated step
in RAG — get it wrong and nothing downstream works.

KEY CONCEPT: chunk_size and chunk_overlap control the trade-off
between context preservation and retrieval precision.
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 2A: Load documents using LangChain's loader ────────────

print("=" * 60)
print("STEP 2: Chunking Documents")
print("=" * 60)

# DirectoryLoader loads all matching files from a folder
loader = DirectoryLoader(
    "sample_docs",
    glob="*.docx",
    loader_cls=Docx2txtLoader,
)
raw_documents = loader.load()

print(f"\nLoaded {len(raw_documents)} documents via LangChain")
for doc in raw_documents:
    source = Path(doc.metadata["source"]).name
    print(f"  {source:30} — {len(doc.page_content):,} chars")


# ── 2B: Chunk with RecursiveCharacterTextSplitter ───────────

print(f"\n{'=' * 60}")
print("CHUNKING WITH RecursiveCharacterTextSplitter")
print(f"{'=' * 60}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Target size in characters
    chunk_overlap=50,      # Characters of overlap between adjacent chunks
    length_function=len,   # How to measure length (characters here)
    separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
)

chunks = splitter.split_documents(raw_documents)

print(f"\nOriginal documents: {len(raw_documents)}")
print(f"After chunking:     {len(chunks)} chunks")
print(f"Chunk size target:  500 characters")
print(f"Overlap:            50 characters")


# ── 2C: Inspect the chunks ──────────────────────────────────

print(f"\n{'=' * 60}")
print("INSPECTING CHUNKS")
print(f"{'=' * 60}")

# Show the first 5 chunks
for i, chunk in enumerate(chunks[:5]):
    source = Path(chunk.metadata["source"]).name
    print(f"\n--- Chunk {i} (from {source}) ---")
    print(f"Length: {len(chunk.page_content)} chars")
    print(f"Content preview: {chunk.page_content[:200]}...")

# Show chunk size distribution
sizes = [len(c.page_content) for c in chunks]
print(f"\n{'=' * 60}")
print("CHUNK SIZE DISTRIBUTION")
print(f"{'=' * 60}")
print(f"  Min:     {min(sizes)} chars")
print(f"  Max:     {max(sizes)} chars")
print(f"  Average: {sum(sizes) / len(sizes):.0f} chars")
print(f"  Median:  {sorted(sizes)[len(sizes)//2]} chars")


# ── 2D: Demonstrate overlap ─────────────────────────────────

print(f"\n{'=' * 60}")
print("DEMONSTRATING OVERLAP")
print(f"{'=' * 60}")

# Find two adjacent chunks from the same document
for i in range(len(chunks) - 1):
    if chunks[i].metadata["source"] == chunks[i + 1].metadata["source"]:
        end_of_chunk_i = chunks[i].page_content[-80:]
        start_of_chunk_j = chunks[i + 1].page_content[:80]
        print(f"\nEnd of chunk {i}:")
        print(f'  "...{end_of_chunk_i}"')
        print(f"\nStart of chunk {i + 1}:")
        print(f'  "{start_of_chunk_j}..."')

        # Find the overlap
        overlap_text = ""
        for length in range(min(80, len(end_of_chunk_i), len(start_of_chunk_j)), 0, -1):
            if end_of_chunk_i.endswith(start_of_chunk_j[:length]):
                overlap_text = start_of_chunk_j[:length]
                break

        if overlap_text:
            print(f"\n  OVERLAP ({len(overlap_text)} chars):")
            print(f'  "{overlap_text}"')
        else:
            print(f"\n  (Overlap exists but split at a separator boundary)")
        print(f"\n  ^ This overlap ensures no information is lost at chunk boundaries")
        break


# ── 2E: Compare different chunk sizes ───────────────────────

print(f"\n{'=' * 60}")
print("EXPERIMENT: Different Chunk Sizes")
print(f"{'=' * 60}")

for size in [200, 500, 1000, 2000]:
    test_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=50
    )
    test_chunks = test_splitter.split_documents(raw_documents)
    avg_size = sum(len(c.page_content) for c in test_chunks) / len(test_chunks)
    print(f"  chunk_size={size:5}  ->  {len(test_chunks):3} chunks  (avg {avg_size:.0f} chars each)")

print(f"\n  -> Smaller chunks = more pieces, finer retrieval, but may lose context")
print(f"  -> Larger chunks = fewer pieces, more context per chunk, but less precise")
print(f"  -> We'll use 500 as our working default")


# ── Save chunks for next step ───────────────────────────────

# We don't need to save to disk — next step will recreate them.
# But let's verify metadata is preserved.
print(f"\n{'=' * 60}")
print("METADATA CHECK")
print(f"{'=' * 60}")
sample = chunks[0]
print(f"  Each chunk carries metadata: {sample.metadata}")
print(f"  This lets us trace answers back to source documents!")

print(f"\n[OK] Step 2 complete. {len(chunks)} chunks ready for embedding.\n")
