from __future__ import annotations

from typing import List, Optional, Tuple

from .common import Chunk, stable_id, whitespace_tokenize


def _chunk_bounds_from_offsets(
    offsets: List[Tuple[int, int]],
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> List[Tuple[int, int]]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")

    n = len(offsets)
    if n == 0:
        return []

    bounds: List[Tuple[int, int]] = []
    start_tok = 0
    while start_tok < n:
        end_tok = min(n, start_tok + target_tokens)
        start_char = offsets[start_tok][0]
        end_char = offsets[end_tok - 1][1]
        bounds.append((start_char, end_char))
        if end_tok == n:
            break
        start_tok = max(0, end_tok - overlap_tokens)

    return bounds


def chunk_document(
    *,
    doc_id: str,
    source: str,
    text: str,
    target_tokens: int = 1000,
    overlap_tokens: int = 150,
    tokenizer_model: Optional[str] = None,
) -> List[Chunk]:
    """
    Chunks a document into overlapping spans, storing provenance boundaries.

    If tokenizer_model is provided, uses HF offsets mapping for more robust token counts.
    Otherwise falls back to whitespace tokenization.
    """
    offsets: List[Tuple[int, int]]
    if tokenizer_model:
        from .common import get_transformers_tokenizer

        tokenizer = get_transformers_tokenizer(tokenizer_model)
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"] if a != b]
    else:
        offsets = [(a, b) for _, a, b in whitespace_tokenize(text)]

    bounds = _chunk_bounds_from_offsets(offsets, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    chunks: List[Chunk] = []
    for start_char, end_char in bounds:
        chunk_id = stable_id("chunk", doc_id, str(start_char), str(end_char))
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
            )
        )
    return chunks

