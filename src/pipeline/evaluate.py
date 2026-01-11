from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .common import Entity, Mention, Relation, compact_text


def _entity_sources_from_mentions(mentions: Sequence[Mention]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = defaultdict(set)
    for m in mentions:
        out[m.entity_id].add(m.source)
    return out


def sample_relations_for_audit(
    relations: Sequence[Relation],
    *,
    n: int = 100,
    seed: int = 42,
    predicates: Optional[Set[str]] = None,
) -> List[dict]:
    rows = [r for r in relations if predicates is None or r.predicate in predicates]
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[: min(n, len(rows))]
    out = []
    for r in rows:
        out.append(
            {
                "subject": r.source_entity_id,
                "predicate": r.predicate,
                "object": r.target_entity_id,
                "confidence": r.confidence,
                "source": r.source,
                "doc_id": r.doc_id,
                "chunk_id": r.chunk_id,
                "evidence_text": r.evidence_text,
                "label": None,  # human: correct/incorrect/unclear
                "notes": None,
            }
        )
    return out


def compute_metrics(
    *,
    docs: Sequence[dict],
    chunks_count: int,
    mentions: Sequence[Mention],
    entities: Sequence[Entity],
    relations: Sequence[Relation],
) -> Dict[str, object]:
    doc_counts = Counter(d["source"] for d in docs)
    mention_counts = Counter(m.source for m in mentions)
    mention_type_counts = Counter((m.source, m.entity_type) for m in mentions)
    entity_type_counts = Counter(e.entity_type for e in entities)
    predicate_counts = Counter(r.predicate for r in relations)

    ent_sources = _entity_sources_from_mentions(mentions)
    ro_entities = {eid for eid, srcs in ent_sources.items() if "ro_stories" in srcs}
    grounded_ro = {
        r.source_entity_id
        for r in relations
        if r.predicate in {"SAME_AS", "BASED_ON"} and r.source_entity_id in ro_entities
    }
    grounding_rate = (len(grounded_ro) / len(ro_entities)) if ro_entities else 0.0

    # Degree / hub diagnostics (entity relations only)
    degree = Counter()
    for r in relations:
        if r.predicate in {"INTERACTS_WITH", "LOCATED_IN", "PARTICIPATES_IN", "SAME_AS", "BASED_ON"}:
            degree[r.source_entity_id] += 1
            degree[r.target_entity_id] += 1
    top_hubs = degree.most_common(15)

    return {
        "docs": dict(doc_counts),
        "chunks": chunks_count,
        "mentions": dict(mention_counts),
        "mentions_by_source_type": {f"{k[0]}::{k[1]}": v for k, v in mention_type_counts.items()},
        "entities_total": len(entities),
        "entities_by_type": dict(entity_type_counts),
        "relations_total": len(relations),
        "relations_by_predicate": dict(predicate_counts),
        "grounding_rate_ro_to_histnero": grounding_rate,
        "top_hubs": top_hubs,
    }


def render_markdown_report(metrics: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# RoLit-KG MVP Report\n")
    lines.append("## Counts\n")
    lines.append(f"- **Docs**: {metrics.get('docs')}")
    lines.append(f"- **Chunks**: {metrics.get('chunks')}")
    lines.append(f"- **Mentions**: {metrics.get('mentions')}")
    lines.append(f"- **Entities (total)**: {metrics.get('entities_total')}")
    lines.append(f"- **Entities by type**: {metrics.get('entities_by_type')}")
    lines.append(f"- **Relations (total)**: {metrics.get('relations_total')}")
    lines.append(f"- **Relations by predicate**: {metrics.get('relations_by_predicate')}")
    lines.append("\n## Grounding\n")
    lines.append(f"- **RO-Stories entities grounded to HistNERo (exact match)**: {metrics.get('grounding_rate_ro_to_histnero'):.3f}")
    lines.append("\n## Noise diagnostics\n")
    lines.append("- **Top hubs (entity_id, degree)**:")
    for eid, deg in (metrics.get("top_hubs") or []):
        lines.append(f"  - `{eid}`: {deg}")
    return "\n".join(lines) + "\n"

