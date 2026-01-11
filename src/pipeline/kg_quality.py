from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .common import Entity, Mention, Relation, normalize_mention


@dataclass
class QCReport:
    removed_entities: int = 0
    removed_mentions: int = 0
    removed_relations: int = 0
    removed_self_loops: int = 0
    removed_type_violations: int = 0
    removed_duplicates: int = 0
    junk_hubs_suppressed: int = 0
    notes: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


_JUNK_EXACT = {
    # Common sentence starters / stopwords that become entities in naive systems
    "nu",
    "erau",
    "cât",
    "apoi",
    "și-apoi",
    "ș-apoi",
    "într-una",
    "draga",
    "povești",
    "mai",
}

_SYMMETRIC_PREDS = {"INTERACTS_WITH", "LOVES", "HATES"}


def _type_allows(pred: str, src_type: str, tgt_type: str) -> bool:
    # Strict, graph-level constraints.
    if pred in {"LOCATED_IN", "TRAVELS_TO"}:
        return src_type != "Location" and tgt_type == "Location"
    if pred in {"INTERACTS_WITH", "LOVES", "HATES", "KILLS", "OWNS", "DESCENDS_FROM"}:
        return src_type != "Location" and tgt_type != "Location"
    # Default: allow
    return True


def _is_junk_entity(e: Entity, degree: int) -> bool:
    name = (e.canonical_name or "").strip()
    norm = normalize_mention(name)
    if not norm:
        return True
    if norm in _JUNK_EXACT:
        return True
    # Too short is usually junk, especially if it becomes a hub
    if len(norm) <= 2 and degree >= 20:
        return True
    # Single token and extremely high degree is suspicious (e.g., "Fata" in folk tales)
    if " " not in norm and degree >= 5000 and len(norm) <= 6:
        return True
    return False


def quality_control(
    *,
    entities: Sequence[Entity],
    mentions: Sequence[Mention],
    relations: Sequence[Relation],
) -> Tuple[List[Entity], List[Mention], List[Relation], QCReport]:
    """
    Graph-level QC:
    - Remove self-loops
    - Enforce type constraints for known predicates
    - Deduplicate symmetric edges (canonical direction)
    - Suppress junk hub entities and drop their mentions/relations
    """
    report = QCReport()

    ent_by_id: Dict[str, Entity] = {e.entity_id: e for e in entities}

    # Degree for hub detection (entity relations only)
    degree = Counter()
    for r in relations:
        degree[r.source_entity_id] += 1
        degree[r.target_entity_id] += 1

    junk_ids: Set[str] = set()
    for e in entities:
        if _is_junk_entity(e, degree[e.entity_id]):
            junk_ids.add(e.entity_id)

    if junk_ids:
        report.junk_hubs_suppressed = len(junk_ids)
        report.notes.append(f"Suppressed junk entities: {len(junk_ids)}")

    # Filter entities
    kept_entities = [e for e in entities if e.entity_id not in junk_ids]
    report.removed_entities = len(entities) - len(kept_entities)

    kept_entity_ids = {e.entity_id for e in kept_entities}

    # Filter mentions
    kept_mentions = [m for m in mentions if m.entity_id in kept_entity_ids]
    report.removed_mentions = len(mentions) - len(kept_mentions)

    # Filter + validate + dedupe relations
    out_rels: List[Relation] = []
    seen: Set[Tuple[str, str, str, str, str]] = set()  # (src, pred, tgt, doc, chunk)
    for r in relations:
        if r.source_entity_id == r.target_entity_id:
            report.removed_self_loops += 1
            continue

        if r.source_entity_id not in kept_entity_ids or r.target_entity_id not in kept_entity_ids:
            continue

        src_t = ent_by_id.get(r.source_entity_id).entity_type if r.source_entity_id in ent_by_id else ""
        tgt_t = ent_by_id.get(r.target_entity_id).entity_type if r.target_entity_id in ent_by_id else ""
        if src_t and tgt_t and not _type_allows(r.predicate, src_t, tgt_t):
            report.removed_type_violations += 1
            continue

        src = r.source_entity_id
        tgt = r.target_entity_id
        pred = r.predicate

        # Canonicalize symmetric predicates to one direction
        if pred in _SYMMETRIC_PREDS:
            if tgt < src:
                src, tgt = tgt, src

        key = (src, pred, tgt, r.doc_id, r.chunk_id)
        if key in seen:
            report.removed_duplicates += 1
            continue
        seen.add(key)

        # Rewrite if direction was canonicalized
        if src != r.source_entity_id or tgt != r.target_entity_id:
            out_rels.append(
                Relation(
                    source_entity_id=src,
                    predicate=pred,
                    target_entity_id=tgt,
                    doc_id=r.doc_id,
                    chunk_id=r.chunk_id,
                    source=r.source,
                    confidence=r.confidence,
                    evidence_text=r.evidence_text,
                    rel_type=r.rel_type,
                )
            )
        else:
            out_rels.append(r)

    report.removed_relations = len(relations) - len(out_rels)
    return kept_entities, kept_mentions, out_rels, report

