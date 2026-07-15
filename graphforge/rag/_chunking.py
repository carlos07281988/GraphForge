"""Text chunking utilities for RAG.

Provides :func:`chunk_text` and :class:`ChunkingStrategy` for splitting
documents into manageable pieces for embedding.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional


class ChunkingStrategy(str, Enum):
    """Strategy for splitting text into chunks."""

    FIXED = "fixed"
    """Split into fixed-size chunks with optional overlap."""

    RECURSIVE = "recursive"
    """Recursively split on separators (paragraphs → sentences → words)."""

    SENTENCE = "sentence"
    """Split on sentence boundaries."""


def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Split text into chunks for embedding.

    Parameters
    ----------
    text:
        Text to split.
    strategy:
        Chunking strategy to use.
    chunk_size:
        Maximum characters per chunk (for fixed/recursive strategies).
    chunk_overlap:
        Overlap between chunks in characters.
    separators:
        Separators to use for recursive splitting (default:
        ``["\\n\\n", "\\n", ".", "?", "!", " ", ""]``).

    Returns
    -------
    List of text chunks.

    Examples
    --------
    .. code-block:: python

        from graphforge.rag import chunk_text, ChunkingStrategy

        chunks = chunk_text(long_text, strategy="recursive", chunk_size=500)
        store.add_texts(chunks)
    """
    if not text:
        return []

    if strategy == ChunkingStrategy.FIXED:
        return _chunk_fixed(text, chunk_size, chunk_overlap)
    elif strategy == ChunkingStrategy.SENTENCE:
        return _chunk_sentences(text, chunk_size, chunk_overlap)
    else:  # RECURSIVE
        seps = separators or ["\n\n", "\n", ".", "?", "!", " ", ""]
        return _chunk_recursive(text, chunk_size, chunk_overlap, seps)


def _chunk_fixed(text: str, size: int, overlap: int) -> List[str]:
    """Split into fixed-size chunks."""
    if size <= 0:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
        if start >= len(text):
            break
    return chunks


def _chunk_sentences(text: str, size: int, overlap: int) -> List[str]:
    """Split on sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > size and current:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s if current else s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _chunk_recursive(
    text: str, size: int, overlap: int, separators: List[str]
) -> List[str]:
    """Recursively split on separators."""
    if not text:
        return []
    if len(text) <= size:
        return [text]

    for sep in separators:
        if sep == " " and len(text) <= size:
            return [text]
        if sep == "":
            # Fall back to fixed-size
            return _chunk_fixed(text, size, overlap)
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) > size and current:
                    chunks.append(current.strip())
                    # Apply overlap: take last 'overlap' chars of current as start
                    current = part
                else:
                    current = candidate
            if current.strip():
                chunks.append(current.strip())
            # Check if all chunks are within size
            if all(len(c) <= size for c in chunks):
                return chunks
            # Otherwise, recurse on oversized chunks
            result = []
            for c in chunks:
                if len(c) <= size:
                    result.append(c)
                else:
                    result.extend(_chunk_recursive(c, size, overlap, separators[separators.index(sep) + 1:]))
            return result
    return _chunk_fixed(text, size, overlap)


__all__ = [
    "ChunkingStrategy",
    "chunk_text",
]
