from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .common import Entity, Mention, Relation


def _sources_for_entity(mentions: Sequence[Mention]) -> Dict[str, Set[str]]:
    sources: Dict[str, Set[str]] = defaultdict(set)
    for m in mentions:
        sources[m.entity_id].add(m.source)
    return sources


def ground_across_corpora_exact(
    *,
    resolved_entities: Sequence[Entity],
    mentions: Sequence[Mention],
) -> List[Relation]:
    """
    Exact grounding via resolution-normalized keys.

    - If a resolved entity has evidence in both corpora and same type => SAME_AS
    - If RO Character aligns to HistNERo Person with same norm => BASED_ON
    """
    sources_by_ent = _sources_for_entity(mentions)

    # Group by (type, norm) from entity.meta populated by resolve_entities_lexical
    by_type_norm: Dict[Tuple[str, str], List[Entity]] = defaultdict(list)
    for e in resolved_entities:
        norm = (e.meta or {}).get("norm")
        if not isinstance(norm, str):
            continue
        by_type_norm[(e.entity_type, norm)].append(e)

    rels: List[Relation] = []

    # SAME_AS for identical type when we have both corpora
    for (etype, norm), ents in by_type_norm.items():
        ro_ids = [e.entity_id for e in ents if "ro_stories" in sources_by_ent.get(e.entity_id, set())]
        hi_ids = [e.entity_id for e in ents if "histnero" in sources_by_ent.get(e.entity_id, set())]
        for ro_id in ro_ids:
            for hi_id in hi_ids:
                if ro_id == hi_id:
                    continue
                rels.append(
                    Relation(
                        source_entity_id=ro_id,
                        predicate="SAME_AS",
                        target_entity_id=hi_id,
                        doc_id="__grounding__",
                        chunk_id="__grounding__",
                        source="grounding",
                        confidence=0.95,
                        evidence_text=None,
                        rel_type="exact_norm_same_type",
                    )
                )

    # BASED_ON when a RO Character matches a HistNERo Person by norm
    char_norm_to_ent: Dict[str, List[str]] = defaultdict(list)
    person_norm_to_ent: Dict[str, List[str]] = defaultdict(list)
    for e in resolved_entities:
        norm = (e.meta or {}).get("norm")
        if not isinstance(norm, str):
            continue
        srcs = sources_by_ent.get(e.entity_id, set())
        if e.entity_type == "Character" and "ro_stories" in srcs:
            char_norm_to_ent[norm].append(e.entity_id)
        if e.entity_type == "Person" and "histnero" in srcs:
            person_norm_to_ent[norm].append(e.entity_id)

    for norm, char_ids in char_norm_to_ent.items():
        person_ids = person_norm_to_ent.get(norm) or []
        for char_id in char_ids:
            for person_id in person_ids:
                if char_id == person_id:
                    continue
                rels.append(
                    Relation(
                        source_entity_id=char_id,
                        predicate="BASED_ON",
                        target_entity_id=person_id,
                        doc_id="__grounding__",
                        chunk_id="__grounding__",
                        source="grounding",
                        confidence=0.85,
                        evidence_text=None,
                        rel_type="exact_norm_character_person",
                    )
                )

    return rels

