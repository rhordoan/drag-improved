from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


ROMANIAN_DIACRITICS_MAP = {
    # cedilla -> comma-below
    "\u015f": "\u0219",  # ş -> ș
    "\u0163": "\u021b",  # ţ -> ț
    "\u015e": "\u0218",  # Ş -> Ș
    "\u0162": "\u021a",  # Ţ -> Ț
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    """
    Deterministic, stable IDs for MERGE-friendly Neo4j loading.
    """
    joined = "\u241f".join(parts)  # unlikely separator
    return f"{prefix}_{sha1_hex(joined)[:16]}"


def normalize_ro_text(text: str) -> str:
    # NFC normalize + enforce Romanian comma-below forms for s/t.
    text = unicodedata.normalize("NFC", text)
    return text.translate(str.maketrans(ROMANIAN_DIACRITICS_MAP))


_PUNCT_RE = re.compile(r"[\s\.,;:\!\?\(\)\[\]\{\}\"'`“”„«»/\\]+", re.UNICODE)


def normalize_mention(surface: str) -> str:
    s = normalize_ro_text(surface).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def get_transformers_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for tokenizer-based chunking. "
            "Install it or set chunking.tokenizer_model=null to use whitespace tokenization."
        ) from e
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def whitespace_tokenize(text: str) -> List[Tuple[str, int, int]]:
    """
    Returns (token, start_char, end_char) tuples.
    """
    tokens: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"\S+", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def approximate_sentence_spans(text: str) -> List[Tuple[int, int]]:
    """
    Very lightweight sentence splitting; good enough for MVP heuristics.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.!?]\s+", text):
        end = m.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def clamp_span(start: int, end: int, n: int) -> Tuple[int, int]:
    start = max(0, min(start, n))
    end = max(0, min(end, n))
    if end < start:
        start, end = end, start
    return start, end


def slice_text(text: str, start: int, end: int) -> str:
    start, end = clamp_span(start, end, len(text))
    return text[start:end]


def compact_text(text: str, max_len: int = 240) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


@dataclasses.dataclass(frozen=True)
class Span:
    start_char: int
    end_char: int
    label: str
    surface: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Document:
    doc_id: str
    source: str  # "ro_stories" | "histnero"
    title: Optional[str]
    text: str
    spans: Tuple[Span, ...] = ()
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    start_char: int
    end_char: int
    text: str


@dataclasses.dataclass(frozen=True)
class Mention:
    mention_id: str
    entity_id: str
    surface: str
    start_char: int
    end_char: int
    doc_id: str
    chunk_id: str
    source: str
    entity_type: str  # Character/Person/Location/Event
    confidence: float


@dataclasses.dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str  # Character/Person/Location/Event
    canonical_name: str
    aliases: Tuple[str, ...] = ()
    is_fictional: Optional[bool] = None
    source: Optional[str] = None
    meta: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Relation:
    source_entity_id: str
    predicate: str
    target_entity_id: str
    doc_id: str
    chunk_id: str
    source: str
    confidence: float
    evidence_text: Optional[str] = None
    rel_type: Optional[str] = None


def asdict_dataclass(obj: Any) -> Dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        return d
    raise TypeError(f"Not a dataclass: {type(obj)}")

