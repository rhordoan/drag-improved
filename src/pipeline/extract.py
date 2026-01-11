from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .common import (
    Chunk,
    Document,
    Entity,
    Mention,
    Relation,
    approximate_sentence_spans,
    compact_text,
    normalize_mention,
    slice_text,
    stable_id,
)


HISTNERO_LABEL_TO_ENTITY_TYPE = {
    "PER": "Person",
    "PERSON": "Person",
    "LOC": "Location",
    "LOCATION": "Location",
    "GPE": "Location",
    "ORG": "Person",  # MVP: keep schema small; refine later
    "DATE": "Event",
    "TIME": "Event",
}


HF_NER_LABEL_TO_ENTITY_TYPE = {
    "PER": "Character",
    "PERSON": "Character",
    "LOC": "Location",
    "LOCATION": "Location",
    "GPE": "Location",
    "ORG": "Character",  # narrative organizations are often factions; refine later
    "MISC": "Character",
}


_CAP_SEQ_RE = re.compile(
    r"\b[A-ZĂÂÎȘȚ][A-Za-zĂÂÎȘȚăâîșț\-\']+(?:\s+[A-ZĂÂÎȘȚ][A-Za-zĂÂÎȘȚăâîșț\-\']+){0,3}\b",
    re.UNICODE,
)

# Romanian stopwords and common non-entity words to exclude from regex NER
_STOPWORDS_RO = {
    'într-o', 'într-un', 'de', 'la', 'pe', 'cu', 'din', 'în', 'pentru',
    'și', 'sau', 'dar', 'dacă', 'când', 'unde', 'cum', 'ce', 'care',
    'mai', 'mult', 'puțin', 'toate', 'toți', 'acestea', 'aceștia',
    'acolo', 'aici', 'atunci', 'acum', 'seara', 'dimineața', 'dimineata',
    'era', 'este', 'fost', 'sunt', 'fi', 'avea', 'face', 'fiind'
}

# Extra Titlecase sentence-starters that appear as false entities in RO-Stories.
_STOPWORDS_TITLECASE = {
    "Erau",
    "Nu",
    "Cât",
    "Apoi",
    "Ș-apoi",
    "Într-una",
    "Draga",
    "Povești",
}


def _entities_from_histnero_spans(doc: Document, chunk: Chunk) -> Tuple[List[Mention], List[Entity]]:
    mentions: List[Mention] = []
    entities: List[Entity] = []
    for s in doc.spans:
        if s.end_char <= chunk.start_char or s.start_char >= chunk.end_char:
            continue
        start = max(s.start_char, chunk.start_char)
        end = min(s.end_char, chunk.end_char)
        surface = slice_text(doc.text, start, end).strip()
        if not surface:
            continue
        raw_label = (s.label or "").upper()
        entity_type = HISTNERO_LABEL_TO_ENTITY_TYPE.get(raw_label, "Person")

        mention_id = stable_id("m", doc.doc_id, chunk.chunk_id, str(start), str(end), surface)
        entity_id = stable_id("ent", "mention", mention_id)

        mentions.append(
            Mention(
                mention_id=mention_id,
                entity_id=entity_id,
                surface=surface,
                start_char=start,
                end_char=end,
                doc_id=doc.doc_id,
                chunk_id=chunk.chunk_id,
                source=doc.source,
                entity_type=entity_type,
                confidence=0.99,
            )
        )

        entities.append(
            Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=surface,
                aliases=(),
                is_fictional=False,
                source=doc.source,
                meta={"label": raw_label},
            )
        )
    return mentions, entities


def _entities_from_regex_ner(doc: Document, chunk: Chunk) -> Tuple[List[Mention], List[Entity]]:
    mentions: List[Mention] = []
    entities: List[Entity] = []
    for m in _CAP_SEQ_RE.finditer(chunk.text):
        surface = m.group(0).strip()
        # Filter stopwords and very short names
        if len(surface) < 2 or surface.lower() in _STOPWORDS_RO or surface in _STOPWORDS_TITLECASE:
            continue
        start = chunk.start_char + m.start()
        end = chunk.start_char + m.end()
        mention_id = stable_id("m", doc.doc_id, chunk.chunk_id, str(start), str(end), surface)
        entity_id = stable_id("ent", "mention", mention_id)
        entity_type = "Character" if doc.source == "ro_stories" else "Person"
        is_fictional = True if entity_type == "Character" else False

        mentions.append(
            Mention(
                mention_id=mention_id,
                entity_id=entity_id,
                surface=surface,
                start_char=start,
                end_char=end,
                doc_id=doc.doc_id,
                chunk_id=chunk.chunk_id,
                source=doc.source,
                entity_type=entity_type,
                confidence=0.50,
            )
        )
        entities.append(
            Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=surface,
                aliases=(),
                is_fictional=is_fictional,
                source=doc.source,
                meta={"extractor": "regex_capitalized"},
            )
        )
    return mentions, entities


@lru_cache(maxsize=8)
def _get_transformers_ner_pipeline(model_name: str, aggregation_strategy: str):
    """
    Cached HF pipeline for NER.

    Note: we force `use_fast=False` to avoid Windows/tiktoken conversion issues for XLM-R.
    """
    from transformers import AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return pipeline(
        "token-classification",
        model=model_name,
        tokenizer=tok,
        aggregation_strategy=aggregation_strategy,
    )


def _entities_from_transformers_ner(
    doc: Document,
    chunk: Chunk,
    *,
    model: str,
    aggregation_strategy: str = "simple",
) -> Tuple[List[Mention], List[Entity]]:
    try:
        # Import is kept for dependency error message; actual pipeline is cached above.
        import transformers  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for transformer NER. "
            "Install it or set extraction.ner_engine=regex."
        ) from e

    nlp = _get_transformers_ner_pipeline(model, aggregation_strategy)
    preds = nlp(chunk.text)

    mentions: List[Mention] = []
    entities: List[Entity] = []
    search_cursor = 0
    for p in preds:
        surface = str(p.get("word", "")).strip()
        if not surface:
            continue
        p_start = p.get("start")
        p_end = p.get("end")
        if p_start is None or p_end is None:
            # Some slow-tokenizer pipelines (notably XLM-R on Windows) return None offsets.
            # Fallback: locate the surface in the chunk text.
            idx = chunk.text.find(surface, search_cursor)
            if idx == -1:
                # Try from beginning as a last resort
                idx = chunk.text.find(surface)
            if idx == -1:
                continue
            start = chunk.start_char + idx
            end = start + len(surface)
            search_cursor = idx + len(surface)
        else:
            start = chunk.start_char + int(p_start)
            end = chunk.start_char + int(p_end)
        label = str(p.get("entity_group") or p.get("entity") or "").upper()
        entity_type = HF_NER_LABEL_TO_ENTITY_TYPE.get(label, "Character" if doc.source == "ro_stories" else "Person")

        # For RO-Stories, treat PER as Character; for HistNERo, PER as Person.
        if doc.source != "ro_stories" and entity_type == "Character":
            entity_type = "Person"

        is_fictional = True if entity_type == "Character" else False if entity_type == "Person" else None

        mention_id = stable_id("m", doc.doc_id, chunk.chunk_id, str(start), str(end), surface)
        entity_id = stable_id("ent", "mention", mention_id)
        score = float(p.get("score", 0.5))

        mentions.append(
            Mention(
                mention_id=mention_id,
                entity_id=entity_id,
                surface=surface,
                start_char=start,
                end_char=end,
                doc_id=doc.doc_id,
                chunk_id=chunk.chunk_id,
                source=doc.source,
                entity_type=entity_type,
                confidence=score,
            )
        )
        entities.append(
            Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                canonical_name=surface,
                aliases=(),
                is_fictional=is_fictional,
                source=doc.source,
                meta={"extractor": "transformers_ner", "label": label, "score": score},
            )
        )
    return mentions, entities


def _merge_mentions_entities(
    mentions_a: List[Mention],
    entities_a: List[Entity],
    mentions_b: List[Mention],
    entities_b: List[Entity],
) -> Tuple[List[Mention], List[Entity]]:
    """
    Merge two mention/entity lists, deduplicating by span + normalized surface.
    Keeps higher-confidence mention if duplicates collide.
    """
    by_key: Dict[Tuple[str, int, int, str], Mention] = {}
    ent_by_id: Dict[str, Entity] = {}

    def add(mentions: List[Mention], entities: List[Entity]):
        for e in entities:
            ent_by_id[e.entity_id] = e
        for m in mentions:
            key = (m.doc_id, m.start_char, m.end_char, normalize_mention(m.surface))
            prev = by_key.get(key)
            if prev is None or m.confidence > prev.confidence:
                by_key[key] = m

    add(mentions_a, entities_a)
    add(mentions_b, entities_b)

    merged_mentions = list(by_key.values())
    merged_entities = [ent_by_id[m.entity_id] for m in merged_mentions if m.entity_id in ent_by_id]
    return merged_mentions, merged_entities


def extract_candidates_for_chunk(
    doc: Document,
    chunk: Chunk,
    *,
    ner_engine: str = "regex",  # "regex" | "transformers" | "ensemble"
    ner_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Mention], List[Entity]]:
    # Content-addressable cache for incremental rebuilds (covers regex/transformers/ensemble).
    # Note: HistNERo spans are deterministic and cheap; we skip caching that path.
    cache_path: Optional[Path] = None
    if cache_dir and doc.source != "histnero":
        try:
            base = Path(cache_dir) / "ner_candidates"
            base.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha1()
            h.update(ner_engine.encode("utf-8"))
            h.update((ner_model or "").encode("utf-8"))
            h.update(doc.doc_id.encode("utf-8"))
            h.update(chunk.chunk_id.encode("utf-8"))
            h.update(chunk.text.encode("utf-8"))
            cache_path = base / f"{h.hexdigest()}.json"
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                ms = [Mention(**m) for m in (payload.get("mentions") or [])]
                es = [Entity(**e) for e in (payload.get("entities") or [])]
                return ms, es
        except Exception:
            cache_path = None

    if doc.source == "histnero" and doc.spans:
        return _entities_from_histnero_spans(doc, chunk)

    if ner_engine == "ensemble":
        if not ner_model:
            raise ValueError("ner_model must be set when ner_engine='ensemble'")
        m1, e1 = _entities_from_regex_ner(doc, chunk)
        m2, e2 = _entities_from_transformers_ner(doc, chunk, model=ner_model)
        mentions, entities = _merge_mentions_entities(m1, e1, m2, e2)
    elif ner_engine == "transformers":
        if not ner_model:
            raise ValueError("ner_model must be set when ner_engine='transformers'")
        mentions, entities = _entities_from_transformers_ner(doc, chunk, model=ner_model)
    else:
        mentions, entities = _entities_from_regex_ner(doc, chunk)

    if cache_path:
        try:
            cache_path.write_text(
                json.dumps(
                    {"mentions": [asdict(m) for m in mentions], "entities": [e.__dict__ for e in entities]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    return mentions, entities


def extract_relations_heuristic(
    *,
    doc: Document,
    chunk: Chunk,
    mentions: Sequence[Mention],
    sentence_level: bool = True,
    max_pairs_per_sentence: int = 40,
) -> List[Relation]:
    """
    MVP relation extraction: sentence co-occurrence heuristics.
    - Character <-> Character => INTERACTS_WITH
    - (Character|Person|Event) -> Location => LOCATED_IN

    Evidence: compacted sentence text.
    """
    if not mentions:
        return []

    # Index mentions by char span for quick sentence assignment.
    rels: List[Relation] = []
    sent_spans = approximate_sentence_spans(chunk.text) if sentence_level else [(0, len(chunk.text))]

    for s_start, s_end in sent_spans:
        abs_start = chunk.start_char + s_start
        abs_end = chunk.start_char + s_end
        sent_mentions = [
            m
            for m in mentions
            if not (m.end_char <= abs_start or m.start_char >= abs_end)
        ]
        if len(sent_mentions) < 2:
            continue

        sent_text = compact_text(chunk.text[s_start:s_end])
        # Dedup by entity_id within sentence
        uniq: Dict[str, Mention] = {}
        for m in sent_mentions:
            uniq[m.entity_id] = m
        sent_mentions = list(uniq.values())

        pairs = 0
        for i in range(len(sent_mentions)):
            for j in range(i + 1, len(sent_mentions)):
                if pairs >= max_pairs_per_sentence:
                    break
                a = sent_mentions[i]
                b = sent_mentions[j]
                pairs += 1

                # Character interactions
                if a.entity_id == b.entity_id:
                    continue

                if a.entity_type == "Character" and b.entity_type == "Character":
                    rels.append(
                        Relation(
                            source_entity_id=a.entity_id,
                            predicate="INTERACTS_WITH",
                            target_entity_id=b.entity_id,
                            doc_id=doc.doc_id,
                            chunk_id=chunk.chunk_id,
                            source=doc.source,
                            confidence=min(a.confidence, b.confidence, 0.60),
                            evidence_text=sent_text,
                            rel_type="cooccur_sentence",
                        )
                    )
                    continue

                # Located-in heuristics
                if a.entity_type != "Location" and b.entity_type == "Location":
                    rels.append(
                        Relation(
                            source_entity_id=a.entity_id,
                            predicate="LOCATED_IN",
                            target_entity_id=b.entity_id,
                            doc_id=doc.doc_id,
                            chunk_id=chunk.chunk_id,
                            source=doc.source,
                            confidence=min(a.confidence, b.confidence, 0.55),
                            evidence_text=sent_text,
                            rel_type="cooccur_sentence",
                        )
                    )
                elif b.entity_type != "Location" and a.entity_type == "Location":
                    rels.append(
                        Relation(
                            source_entity_id=b.entity_id,
                            predicate="LOCATED_IN",
                            target_entity_id=a.entity_id,
                            doc_id=doc.doc_id,
                            chunk_id=chunk.chunk_id,
                            source=doc.source,
                            confidence=min(a.confidence, b.confidence, 0.55),
                            evidence_text=sent_text,
                            rel_type="cooccur_sentence",
                        )
                    )

    return rels


_ALLOWED_LLM_PREDICATES = {
    "INTERACTS_WITH",
    "LOCATED_IN",
    "LOVES",
    "HATES",
    "KILLS",
    "TRAVELS_TO",
    "DESCENDS_FROM",
    "OWNS",
}


def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Best-effort extraction of a JSON array/object from an LLM response.
    """
    t = text.strip()
    if not t:
        return None
    # Prefer array
    a0 = t.find("[")
    a1 = t.rfind("]")
    if a0 != -1 and a1 != -1 and a1 > a0:
        return t[a0 : a1 + 1]
    o0 = t.find("{")
    o1 = t.rfind("}")
    if o0 != -1 and o1 != -1 and o1 > o0:
        return t[o0 : o1 + 1]
    return None


def _extract_first_json_array(text: str) -> Optional[str]:
    """
    Extract the first JSON array substring using bracket matching.
    More robust than taking the last closing bracket when extra text exists.
    """
    if not text:
        return None
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_relations_ollama(
    *,
    doc: Document,
    chunk: Chunk,
    mentions: Sequence[Mention],
    base_url: str,
    model: str,
    timeout_s: int = 60,
    max_entities: int = 25,
    max_relations: int = 30,
    context_text: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Relation]:
    """
    LLM-based relation extraction using an Ollama-compatible API.

    Uses Ollama **JSON mode** by calling `/api/chat` with a JSON Schema in `format`.
    """
    if not mentions:
        return []

    # Unique entities for this chunk, deduped by normalized surface form.
    # This prevents the LLM from seeing many IDs for the same name (common with mention-based IDs).
    uniq_by_surface: Dict[str, Mention] = {}
    for m in mentions:
        k = normalize_mention(m.surface)
        if not k:
            continue
        prev = uniq_by_surface.get(k)
        if prev is None:
            uniq_by_surface[k] = m
            continue
        # Keep higher-confidence mention; tie-breaker: shorter surface (less noisy)
        if m.confidence > prev.confidence:
            uniq_by_surface[k] = m
        elif m.confidence == prev.confidence and len(m.surface) < len(prev.surface):
            uniq_by_surface[k] = m

    ents = list(uniq_by_surface.values())
    ents.sort(key=lambda m: (-m.confidence, len(m.surface), m.entity_id))
    ents = ents[:max_entities]

    entity_table = [
        {
            "entity_id": m.entity_id,
            "name": m.surface,
            "type": m.entity_type,
        }
        for m in ents
    ]
    allowed_ids = {m.entity_id for m in ents}

    system_prompt = (
        "You extract knowledge-graph relations from Romanian text.\n"
        "You MUST follow the output JSON schema exactly.\n"
        "Do not invent entities. Use ONLY provided entity_id values.\n"
        "No self-relations (source_entity_id != target_entity_id).\n"
        "Every relation MUST include evidence_text as an exact quote from the provided Text.\n"
        "If a relation is not explicitly supported by the Text, omit it.\n"
    )

    user_prompt = (
        f"Allowed predicates: {sorted(_ALLOWED_LLM_PREDICATES)}\n"
        f"Output at most {max_relations} relations.\n\n"
        "Entities (use these IDs exactly):\n"
        + json.dumps(entity_table, ensure_ascii=False)
        + ("\n\nContext (previous chunk, may help coreference):\n" + context_text if context_text else "")
        + "\n\nText:\n"
        + chunk.text
    )

    schema = {
        "type": "array",
        "maxItems": max_relations,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_entity_id": {"type": "string"},
                "predicate": {"type": "string", "enum": sorted(_ALLOWED_LLM_PREDICATES)},
                "target_entity_id": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "evidence_text": {"type": "string", "minLength": 1, "maxLength": 240},
            },
            "required": ["source_entity_id", "predicate", "target_entity_id", "confidence", "evidence_text"],
        },
    }

    try:
        import requests
    except Exception as e:  # pragma: no cover
        raise RuntimeError("requests is required for Ollama LLM relation extraction.") from e

    sess = requests.Session()
    payload = {
        "model": model,
        "stream": False,
        "format": schema,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        },
    }

    # Build type map for validation (prevents nonsensical edges)
    id_to_type = {m.entity_id: m.entity_type for m in ents}

    def _type_allows(pred: str, src_type: str, tgt_type: str) -> bool:
        if pred == "INTERACTS_WITH":
            return src_type != "Location" and tgt_type != "Location"
        if pred in ("LOCATED_IN", "TRAVELS_TO"):
            return src_type != "Location" and tgt_type == "Location"
        if pred in ("DESCENDS_FROM", "LOVES", "HATES", "KILLS"):
            return src_type != "Location" and tgt_type != "Location"
        if pred == "OWNS":
            return src_type != "Location"
        return True

    def _build_relations_from_parsed(parsed_list: list) -> List[Relation]:
        rels_out: List[Relation] = []
        seen = set()
        for r in parsed_list[:max_relations]:
            if not isinstance(r, dict):
                continue
            s = str(r.get("source_entity_id") or "").strip()
            t = str(r.get("target_entity_id") or "").strip()
            p = str(r.get("predicate") or "").strip().upper()
            if not s or not t or s == t:
                continue
            if s not in allowed_ids or t not in allowed_ids:
                continue
            if p not in _ALLOWED_LLM_PREDICATES:
                continue

            ev = r.get("evidence_text")
            if ev is None:
                continue
            if not isinstance(ev, str):
                ev = str(ev)
            ev = ev.strip()
            if not ev:
                continue
            if ev not in chunk.text:
                continue

            src_type = id_to_type.get(s, "")
            tgt_type = id_to_type.get(t, "")
            if src_type and tgt_type and not _type_allows(p, src_type, tgt_type):
                continue

            key = (s, p, t, doc.doc_id, chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)

            conf = r.get("confidence", 0.75)
            try:
                conf_f = float(conf)
            except Exception:
                conf_f = 0.75
            conf_f = max(0.0, min(1.0, conf_f))

            rels_out.append(
                Relation(
                    source_entity_id=s,
                    predicate=p,
                    target_entity_id=t,
                    doc_id=doc.doc_id,
                    chunk_id=chunk.chunk_id,
                    source=doc.source,
                    confidence=conf_f,
                    evidence_text=ev,
                    rel_type="llm_ollama_json",
                )
            )
        return rels_out

    # Cache key (content-addressable): model + schema + chunk text + entity table (+ context hash)
    cache_path: Optional[Path] = None
    if cache_dir:
        try:
            base = Path(cache_dir) / "ollama_relations_json"
            base.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha1()
            h.update(model.encode("utf-8"))
            h.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
            h.update(chunk.text.encode("utf-8"))
            h.update(json.dumps(entity_table, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            if context_text:
                h.update(context_text.encode("utf-8"))
            cache_path = base / f"{h.hexdigest()}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return _build_relations_from_parsed(cached)
        except Exception:
            cache_path = None

    resp = None
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = sess.post(
                f"{base_url.rstrip('/')}/api/chat",
                json=payload,
                timeout=timeout_s,
            )
            resp.raise_for_status()
            last_err = None
            break
        except Exception as e:
            last_err = e
            # Backoff: 1s, 2s, 4s
            try:
                import time as _time
                _time.sleep(2**attempt)
            except Exception:
                pass

    if resp is None or last_err is not None:
        # Fail open: return no relations instead of crashing the pipeline.
        return []
    data = resp.json()
    content = ""
    msg = data.get("message")
    if isinstance(msg, dict):
        content = msg.get("content") or ""
    if not content:
        return []

    # In JSON schema mode, content *should* be JSON, but some models still add extra text.
    parsed: object
    try:
        parsed = json.loads(content)
    except Exception:
        json_str = (
            _extract_first_json_array(content)
            or _extract_json_from_text(content)
            or ""
        )
        if not json_str:
            return []
        try:
            parsed = json.loads(json_str)
        except Exception:
            return []

    if not isinstance(parsed, list):
        return []

    rels = _build_relations_from_parsed(parsed)

    # Save cache (store parsed list)
    if cache_path:
        try:
            cache_path.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    return rels


def extract_for_documents(
    docs: Sequence[Document],
    doc_chunks: Dict[str, List[Chunk]],
    *,
    ner_engine: str = "regex",
    ner_model: Optional[str] = None,
    relations_engine: str = "heuristic",  # "heuristic" | "ollama"
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_timeout_s: int = 60,
    cache_dir: Optional[str] = None,
) -> Tuple[List[Mention], List[Entity], List[Relation]]:
    mentions_all: List[Mention] = []
    entities_all: List[Entity] = []
    relations_all: List[Relation] = []

    doc_by_id = {d.doc_id: d for d in docs}
    for doc_id, chunks in doc_chunks.items():
        doc = doc_by_id[doc_id]
        prev_chunk_text: Optional[str] = None
        prev_chunk_norms: Optional[set] = None
        for ch in chunks:
            mentions, entities = extract_candidates_for_chunk(
                doc,
                ch,
                ner_engine=ner_engine,
                ner_model=ner_model,
                cache_dir=cache_dir,
            )
            mentions_all.extend(mentions)
            entities_all.extend(entities)
            if relations_engine == "heuristic":
                relations_all.extend(extract_relations_heuristic(doc=doc, chunk=ch, mentions=mentions))
            elif relations_engine == "ollama":
                if not llm_base_url or not llm_model:
                    raise ValueError("llm_base_url and llm_model must be set when relations_engine='ollama'")
                cur_norms = {normalize_mention(m.surface) for m in mentions if m.surface}
                ctx = prev_chunk_text if (prev_chunk_text and prev_chunk_norms and (cur_norms & prev_chunk_norms)) else None
                relations_all.extend(
                    extract_relations_ollama(
                        doc=doc,
                        chunk=ch,
                        mentions=mentions,
                        base_url=llm_base_url,
                        model=llm_model,
                        timeout_s=llm_timeout_s,
                        context_text=ctx,
                        cache_dir=cache_dir,
                    )
                )
            prev_chunk_text = ch.text
            prev_chunk_norms = {normalize_mention(m.surface) for m in mentions if m.surface}

    return mentions_all, entities_all, relations_all

