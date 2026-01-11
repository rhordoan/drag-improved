from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .common import Entity, Mention, normalize_mention, stable_id


def resolve_entities_lexical(
    *,
    entities: Sequence[Entity],
    mentions: Sequence[Mention],
    scope: str = "per_source",  # "global" | "per_doc" | "per_source"
    min_norm_len: int = 2,
) -> Tuple[List[Entity], List[Mention], Dict[str, str]]:
    """
    Conservative lexical clustering: group by normalized surface form + entity_type (+ optional doc_id).

    Returns:
      - resolved_entities
      - updated_mentions (with entity_id rewritten to resolved id)
      - mapping old_entity_id -> resolved_entity_id
    """
    # Build surface frequency per original entity_id from mentions.
    surface_by_entity: Dict[str, List[str]] = defaultdict(list)
    meta_by_entity: Dict[str, Dict[str, object]] = {}
    for e in entities:
        meta_by_entity[e.entity_id] = dict(e.meta or {})

    for m in mentions:
        surface_by_entity[m.entity_id].append(m.surface)

    # Keying: keep type compatibility; optionally isolate by doc.
    def key_for(m: Mention) -> Tuple[str, str, str]:
        norm = normalize_mention(m.surface)
        if len(norm) < min_norm_len:
            norm = f"__short__:{norm}"
        if scope == "per_doc":
            return (m.entity_type, norm, m.doc_id)
        if scope == "per_source":
            return (m.entity_type, norm, m.source)
        return (m.entity_type, norm, "__global__")

    groups: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    for m in mentions:
        groups[key_for(m)].append(m.entity_id)

    # Create resolved entities
    resolved_entities: List[Entity] = []
    old_to_new: Dict[str, str] = {}
    for (entity_type, norm, scope_id), old_ids in groups.items():
        # Deduplicate the old_ids list
        old_ids_uniq = sorted(set(old_ids))
        # Choose canonical surface: most common surface across mentions in cluster
        surfaces = []
        for oid in old_ids_uniq:
            surfaces.extend(surface_by_entity.get(oid, []))
        surface_counts = Counter(surfaces)
        canonical = surface_counts.most_common(1)[0][0] if surface_counts else norm
        aliases = tuple(sorted({s for s in surfaces if s and s != canonical}))

        resolved_id = stable_id("ent", "resolved", entity_type, norm, scope_id)
        for oid in old_ids_uniq:
            old_to_new[oid] = resolved_id

        # Pick a stable is_fictional: Character => True, Person => False, else None
        is_fictional = True if entity_type == "Character" else False if entity_type == "Person" else None

        resolved_entities.append(
            Entity(
                entity_id=resolved_id,
                entity_type=entity_type,
                canonical_name=canonical,
                aliases=aliases,
                is_fictional=is_fictional,
                source=None,  # multi-source after clustering
                meta={
                    "resolution": "lexical",
                    "norm": norm,
                    "scope": scope,
                    "member_entity_ids": old_ids_uniq,
                    "surface_counts_top": surface_counts.most_common(5),
                },
            )
        )

    updated_mentions: List[Mention] = []
    for m in mentions:
        new_id = old_to_new.get(m.entity_id, m.entity_id)
        updated_mentions.append(
            Mention(
                mention_id=m.mention_id,
                entity_id=new_id,
                surface=m.surface,
                start_char=m.start_char,
                end_char=m.end_char,
                doc_id=m.doc_id,
                chunk_id=m.chunk_id,
                source=m.source,
                entity_type=m.entity_type,
                confidence=m.confidence,
            )
        )

    return resolved_entities, updated_mentions, old_to_new

