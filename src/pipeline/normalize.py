from __future__ import annotations

from typing import Iterable, List

from .common import Document, Span, clamp_span, normalize_ro_text, slice_text


def normalize_documents(docs: Iterable[Document]) -> List[Document]:
    """
    Ensures NFC + Romanian diacritics consistency and validates span bounds.
    """
    out: List[Document] = []
    for d in docs:
        text = normalize_ro_text(d.text)
        spans: List[Span] = []
        for s in d.spans:
            start, end = clamp_span(s.start_char, s.end_char, len(text))
            surface = s.surface
            if surface is None:
                surface = slice_text(text, start, end)
            else:
                surface = normalize_ro_text(surface)
            spans.append(Span(start_char=start, end_char=end, label=s.label, surface=surface))

        out.append(
            Document(
                doc_id=d.doc_id,
                source=d.source,
                title=d.title,
                text=text,
                spans=tuple(spans),
                meta=dict(d.meta),
            )
        )
    return out

