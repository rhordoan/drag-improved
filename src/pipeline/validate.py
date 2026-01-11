from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from .common import Entity, Mention, Relation


ALLOWED_ENTITY_TYPES = {"Character", "Person", "Location", "Event"}
ALLOWED_PREDICATES = {
    "INTERACTS_WITH",
    "LOCATED_IN",
    "TRAVELS_TO",
    "LOVES",
    "HATES",
    "KILLS",
    "DESCENDS_FROM",
    "OWNS",
    # Event schema
    "PARTICIPATES_IN",
    "HAS_PARTICIPANT",
    "HAS_DESTINATION",
    # Resolution/grounding
    "SAME_AS",
    "CANDIDATE_SAME_AS",
    "BASED_ON",
}


def validate_artifacts(
    *,
    mentions: Sequence[Mention],
    entities: Sequence[Entity],
    relations: Sequence[Relation],
) -> Dict[str, object]:
    """
    Returns a small validation report. Does not raise unless a critical error is found.
    """
    entity_ids = {e.entity_id for e in entities}
    mention_ids = {m.mention_id for m in mentions}

    errors: List[str] = []
    warnings: List[str] = []

    for m in mentions:
        if m.entity_type not in ALLOWED_ENTITY_TYPES:
            warnings.append(f"Unknown entity_type in mention {m.mention_id}: {m.entity_type}")
        if m.entity_id not in entity_ids:
            errors.append(f"Mention {m.mention_id} refers to missing entity_id={m.entity_id}")
        if m.end_char < m.start_char:
            errors.append(f"Mention {m.mention_id} has end_char < start_char")

    for e in entities:
        if e.entity_type not in ALLOWED_ENTITY_TYPES:
            warnings.append(f"Unknown entity_type in entity {e.entity_id}: {e.entity_type}")

    for r in relations:
        if r.predicate not in ALLOWED_PREDICATES:
            warnings.append(f"Unknown predicate {r.predicate} in relation {r.source_entity_id}->{r.target_entity_id}")
        if r.source_entity_id not in entity_ids:
            errors.append(f"Relation has missing source_entity_id={r.source_entity_id}")
        if r.target_entity_id not in entity_ids:
            errors.append(f"Relation has missing target_entity_id={r.target_entity_id}")

    if errors:
        # Keep it strict: validation failures should stop the run.
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors[:50]))

    ent_type_counts = Counter(e.entity_type for e in entities)
    pred_counts = Counter(r.predicate for r in relations)

    return {
        "mentions": len(mentions),
        "entities": len(entities),
        "relations": len(relations),
        "entity_type_counts": dict(ent_type_counts),
        "predicate_counts": dict(pred_counts),
        "warnings": warnings[:200],
    }

