from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .common import Document, Span, normalize_ro_text, read_jsonl, stable_id


def load_ro_stories_from_hf(
    dataset_name: str,
    split: str,
    *,
    revision: Optional[str] = None,
    text_field: str = "text",
    title_field: Optional[str] = None,
    author_field: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Document]:
    """
    Loads RO-Stories (or any HF text dataset) into the internal Document schema.
    """
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face 'datasets' is required to load RO-Stories from HF. "
            "Install it or provide ro_stories.path_jsonl instead."
        ) from e

    ds = load_dataset(dataset_name, split=split, revision=revision)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    docs: List[Document] = []
    for i, row in enumerate(ds):
        raw_text = row.get(text_field)
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        title = row.get(title_field) if title_field else None
        if title is not None and not isinstance(title, str):
            title = str(title)

        author = row.get(author_field) if author_field else None
        if author is not None and not isinstance(author, str):
            author = str(author)

        # Stable doc ids: dataset + split + index
        doc_id = stable_id("doc", "ro_stories", dataset_name, split, str(i))
        meta = {
            "hf_dataset": dataset_name,
            "hf_split": split,
            "hf_revision": revision,
            "row_index": i,
        }
        if author:
            meta["author"] = author

        docs.append(
            Document(
                doc_id=doc_id,
                source="ro_stories",
                title=title,
                text=normalize_ro_text(raw_text),
                spans=(),
                meta=meta,
            )
        )

    return docs


def load_documents_from_jsonl(path: str | Path, *, source: str) -> List[Document]:
    """
    Generic JSONL ingest for either corpus.

    Expected format per line:
      {
        "doc_id": "...",            (optional; auto-generated if missing)
        "title": "...",             (optional)
        "text": "...",              (required)
        "spans": [                  (optional; HistNERo typically)
          {"start_char": 0, "end_char": 10, "label": "PER", "surface": "..." }
        ],
        "meta": {...}               (optional)
      }
    """
    docs: List[Document] = []
    path = Path(path)
    for i, row in enumerate(read_jsonl(path)):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        raw_doc_id = row.get("doc_id")
        doc_id = raw_doc_id if isinstance(raw_doc_id, str) and raw_doc_id else stable_id(
            "doc", source, str(path), str(i)
        )

        title = row.get("title")
        if title is not None and not isinstance(title, str):
            title = str(title)

        spans_in = row.get("spans") or []
        spans: List[Span] = []
        if isinstance(spans_in, list):
            for s in spans_in:
                if not isinstance(s, dict):
                    continue
                try:
                    start = int(s.get("start_char"))
                    end = int(s.get("end_char"))
                    label = str(s.get("label"))
                except Exception:
                    continue
                surface = s.get("surface")
                if surface is not None and not isinstance(surface, str):
                    surface = str(surface)
                spans.append(Span(start_char=start, end_char=end, label=label, surface=surface))

        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        docs.append(
            Document(
                doc_id=doc_id,
                source=source,
                title=title,
                text=normalize_ro_text(text),
                spans=tuple(spans),
                meta=meta,
            )
        )

    return docs

