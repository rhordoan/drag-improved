from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .common import Entity, Relation, stable_id


_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

_EVENT_PREDS = {"TRAVELS_TO", "INTERACTS_WITH", "KILLS", "LOVES", "HATES"}


def derive_event_nodes(
    *,
    entities: Sequence[Entity],
    relations: Sequence[Relation],
) -> Tuple[List[Entity], List[Relation]]:
    """
    Derive explicit Event nodes from certain relation predicates.

    Returns:
      - event_entities (Entity objects with entity_type='Event')
      - event_relations (PARTICIPATES_IN / HAS_PARTICIPANT / HAS_DESTINATION)
    """
    event_entities: List[Entity] = []
    event_relations: List[Relation] = []

    for r in relations:
        if r.predicate not in _EVENT_PREDS:
            continue

        ev_text = r.evidence_text or ""
        year = None
        m = _YEAR_RE.search(ev_text)
        if m:
            year = int(m.group(1))

        event_id = stable_id(
            "evt",
            r.doc_id,
            r.chunk_id,
            r.predicate,
            r.source_entity_id,
            r.target_entity_id,
            ev_text,
        )

        event_entities.append(
            Entity(
                entity_id=event_id,
                entity_type="Event",
                canonical_name=r.predicate,
                aliases=(),
                is_fictional=None,
                source=r.source,
                meta={
                    "event_predicate": r.predicate,
                    "doc_id": r.doc_id,
                    "chunk_id": r.chunk_id,
                    "evidence_text": ev_text,
                    "year": year,
                },
            )
        )

        # Participants
        event_relations.append(
            Relation(
                source_entity_id=r.source_entity_id,
                predicate="PARTICIPATES_IN",
                target_entity_id=event_id,
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                source=r.source,
                confidence=r.confidence,
                evidence_text=ev_text,
                rel_type="derived_event",
            )
        )
        event_relations.append(
            Relation(
                source_entity_id=event_id,
                predicate="HAS_PARTICIPANT",
                target_entity_id=r.source_entity_id,
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                source=r.source,
                confidence=r.confidence,
                evidence_text=ev_text,
                rel_type="derived_event",
            )
        )

        if r.predicate == "INTERACTS_WITH":
            event_relations.append(
                Relation(
                    source_entity_id=r.target_entity_id,
                    predicate="PARTICIPATES_IN",
                    target_entity_id=event_id,
                    doc_id=r.doc_id,
                    chunk_id=r.chunk_id,
                    source=r.source,
                    confidence=r.confidence,
                    evidence_text=ev_text,
                    rel_type="derived_event",
                )
            )
            event_relations.append(
                Relation(
                    source_entity_id=event_id,
                    predicate="HAS_PARTICIPANT",
                    target_entity_id=r.target_entity_id,
                    doc_id=r.doc_id,
                    chunk_id=r.chunk_id,
                    source=r.source,
                    confidence=r.confidence,
                    evidence_text=ev_text,
                    rel_type="derived_event",
                )
            )

        if r.predicate == "TRAVELS_TO":
            # target is destination location
            event_relations.append(
                Relation(
                    source_entity_id=event_id,
                    predicate="HAS_DESTINATION",
                    target_entity_id=r.target_entity_id,
                    doc_id=r.doc_id,
                    chunk_id=r.chunk_id,
                    source=r.source,
                    confidence=r.confidence,
                    evidence_text=ev_text,
                    rel_type="derived_event",
                )
            )

    # Deduplicate event entities by id (stable_id should already, but just in case)
    uniq = {}
    for e in event_entities:
        uniq[e.entity_id] = e
    return list(uniq.values()), event_relations

