"""
ingestion.py
------------
Turns an uploaded file (PDF / DOCX / TXT) into clean text, then splits
that text into passage-sized chunks that are small enough to feed into
a transformer model (roughly 300-500 tokens each).

Everything downstream (summarizer, question generator, recommender)
consumes the output of `chunk_text()`, so this module has no ML
dependencies -- it's pure text handling and should be rock solid
before you touch the model code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------
# Step 1: raw text extraction per file type
# ---------------------------------------------------------------------

def extract_text(file_path: str) -> str:
    """Dispatch to the right extractor based on file extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext == ".docx":
        return _extract_docx(file_path)
    elif ext == ".txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_pdf(file_path: str) -> str:
    import fitz  # PyMuPDF

    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_docx(file_path: str) -> str:
    import docx

    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------
# Step 2: cleaning
# ---------------------------------------------------------------------

def clean_text(raw_text: str) -> str:
    """
    Strip the junk that PDFs/DOCX commonly leave behind:
    - repeated whitespace / stray newlines from PDF line-wrapping
    - page-number-only lines (e.g. "12", "Page 3 of 10")
    - multiple blank lines collapsed to one
    """
    lines = raw_text.splitlines()
    cleaned_lines = []

    page_number_pattern = re.compile(r"^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if page_number_pattern.match(stripped):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)
    # Collapse excessive whitespace introduced by PDF line-wrapping
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ---------------------------------------------------------------------
# Step 3: chunking into model-sized passages
# ---------------------------------------------------------------------

def chunk_text(text: str, max_words: int = 400, overlap_words: int = 40) -> List[str]:
    """
    Split cleaned text into overlapping word-count chunks.

    A small overlap (default 40 words) keeps context from getting cut
    mid-idea at chunk boundaries, which matters for both summarization
    and question generation quality.

    max_words=400 is a safe default for T5/BART-family models with a
    512-token input limit (words != tokens, so this leaves headroom).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap_words  # step forward, but overlap a bit

    return chunks


# ---------------------------------------------------------------------
# Convenience: do all three steps in one call
# ---------------------------------------------------------------------

def process_file(file_path: str, max_words: int = 400, overlap_words: int = 40) -> List[str]:
    raw = extract_text(file_path)
    cleaned = clean_text(raw)
    return chunk_text(cleaned, max_words=max_words, overlap_words=overlap_words)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ingestion.py <path-to-file>")
        sys.exit(1)

    passages = process_file(sys.argv[1])
    print(f"Extracted {len(passages)} passage(s):\n")
    for i, p in enumerate(passages, 1):
        print(f"--- Passage {i} ({len(p.split())} words) ---")
        print(p[:300] + ("..." if len(p) > 300 else ""))
        print()