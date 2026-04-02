"""
RAG Live Project -Step 1: Load & Inspect Documents
=====================================================
WHAT WE'RE DOING: Loading our company's policy documents and
understanding what we're working with before any RAG magic.

This is the "look at your data first" step -always do this.
"""

import os
from pathlib import Path
from docx import Document

# ── 1A: Load documents from our sample_docs folder ──────────

docs_folder = Path("sample_docs")

print("=" * 60)
print("STEP 1: Loading and Inspecting Documents")
print("=" * 60)

# List all files in our docs folder
print(f"\nDocuments found in '{docs_folder}':")
print("-" * 40)

documents = {}
for filepath in sorted(docs_folder.glob("*.docx")):
    doc = Document(filepath)
    content = "\n".join(para.text for para in doc.paragraphs)
    documents[filepath.name] = content
    print(f"  {filepath.name:30} -{len(content):,} characters, ~{len(content.split()):,} words")

print(f"\nTotal documents: {len(documents)}")
total_chars = sum(len(c) for c in documents.values())
total_words = sum(len(c.split()) for c in documents.values())
print(f"Total content: {total_chars:,} characters, ~{total_words:,} words")


# ── 1B: Look at one document to understand the content ──────

print("\n" + "=" * 60)
preview_file = "03_Leave_Attendance_Policy.docx"
print(f"PREVIEW: First 500 characters of '{preview_file}'")
print("=" * 60)
print(documents[preview_file][:500])
print("...")


# ── 1C: Estimate token count (for cost awareness) ───────────

try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    total_tokens = sum(len(enc.encode(c)) for c in documents.values())
    print(f"\n{'=' * 60}")
    print(f"TOKEN ESTIMATE (gpt-4o-mini tokenizer)")
    print(f"{'=' * 60}")
    print(f"Total tokens across all documents: {total_tokens:,}")
    print(f"Estimated embedding cost: negligible (we use a free local model)")
    print(f"Estimated LLM cost per query: ~$0.0001 (gpt-4o-mini)")
except ImportError:
    print("\n(tiktoken not installed -skipping token count)")


# ── 1D: Why we can't just dump everything into one prompt ───

print(f"\n{'=' * 60}")
print("WHY WE NEED RAG (not just a big prompt)")
print(f"{'=' * 60}")
print(f"Total content: ~{total_words:,} words ~ ~{int(total_words / 0.75):,} tokens")
print(f"GPT-4o-mini context window: 128,000 tokens")
print(f"Our documents fit in one prompt... but in production:")
print(f"  ->500 documents would be ~{total_words * 100:,} tokens -way too large")
print(f"  ->Most content is irrelevant to any specific question")
print(f"  ->Cost scales linearly with input tokens")
print(f"  ->RAG retrieves ONLY the relevant chunks -efficient & accurate")

print(f"\n[OK] Step 1 complete. We know our data. Now let's chunk it.\n")
