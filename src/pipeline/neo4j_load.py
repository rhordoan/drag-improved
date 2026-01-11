from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .common import Entity, Mention, Relation, ensure_dir, stable_id, write_json


def build_constraints_cypher() -> str:
    return "\n".join(
        [
            "CREATE CONSTRAINT work_id_unique IF NOT EXISTS FOR (w:Work) REQUIRE w.work_id IS UNIQUE;",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;",
            "CREATE CONSTRAINT mention_id_unique IF NOT EXISTS FOR (m:Mention) REQUIRE m.mention_id IS UNIQUE;",
            "CREATE INDEX mention_doc_id IF NOT EXISTS FOR (m:Mention) ON (m.doc_id);",
        ]
    )


def _label_for_entity_type(entity_type: str) -> str:
    # Secondary label for convenience; all entities also carry :Entity.
    return entity_type


def _escape_str(s: str) -> str:
    # Minimal Cypher string escaping (single-quoted).
    return s.replace("\\", "\\\\").replace("'", "\\'")


def cypher_literal(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(cypher_literal(v) for v in value) + "]"
    return "'" + _escape_str(str(value)) + "'"


def build_graph_records(
    *,
    docs: Sequence[Dict[str, object]],
    mentions: Sequence[Mention],
    entities: Sequence[Entity],
    relations: Sequence[Relation],
) -> Dict[str, object]:
    """
    Produces dictionaries that can be used either for Cypher generation or Neo4j UNWIND loading.
    """
    works = []
    for d in docs:
        works.append(
            {
                "work_id": d["work_id"],
                "title": d.get("title"),
                "author": d.get("author"),
                "source": d.get("source"),
            }
        )

    mention_rows = [asdict(m) for m in mentions]
    entity_rows = [
        {
            "entity_id": e.entity_id,
            "entity_type": e.entity_type,
            "canonical_name": e.canonical_name,
            "aliases": list(e.aliases),
            "is_fictional": e.is_fictional,
            "source": e.source,
            "meta": e.meta,
        }
        for e in entities
    ]
    relation_rows = [asdict(r) for r in relations]

    # Mention-level coreference edges (derived): mentions that refer to the same entity within a doc.
    # We connect a representative mention to other mentions with different surfaces.
    by_doc_entity: Dict[Tuple[str, str], List[dict]] = {}
    for m in mention_rows:
        key = (str(m.get("doc_id")), str(m.get("entity_id")))
        by_doc_entity.setdefault(key, []).append(m)

    coref_edges: List[dict] = []
    for (doc_id, entity_id), ms in by_doc_entity.items():
        if len(ms) < 2:
            continue
        # pick representative as earliest mention
        ms_sorted = sorted(ms, key=lambda x: (int(x.get("start_char") or 0), str(x.get("mention_id"))))
        rep = ms_sorted[0]
        rep_surface = str(rep.get("surface") or "")
        for other in ms_sorted[1:]:
            if str(other.get("surface") or "") == rep_surface:
                continue
            coref_edges.append(
                {
                    "source_mention_id": rep["mention_id"],
                    "target_mention_id": other["mention_id"],
                    "doc_id": doc_id,
                    "entity_id": entity_id,
                }
            )

    return {
        "works": works,
        "mentions": mention_rows,
        "entities": entity_rows,
        "relations": relation_rows,
        "mention_coref": coref_edges,
    }


def write_cypher_files(
    *,
    out_dir: str | Path,
    graph_records: Dict[str, object],
) -> Dict[str, str]:
    out_dir = ensure_dir(out_dir)
    constraints_path = out_dir / "constraints.cypher"
    load_path = out_dir / "load.cypher"

    constraints_path.write_text(build_constraints_cypher() + "\n", encoding="utf-8")

    works = graph_records["works"]
    mentions = graph_records["mentions"]
    entities = graph_records["entities"]
    relations = graph_records["relations"]
    mention_coref = graph_records.get("mention_coref") or []

    lines: List[str] = []
    lines.append("// Auto-generated RoLit-KG MVP Cypher (idempotent MERGEs)\n")
    lines.append("// 1) Works\n")
    for w in works:
        lines.append(
            "MERGE (w:Work {work_id: %s})\n"
            "SET w.title=%s, w.author=%s, w.source=%s;"
            % (
                cypher_literal(w["work_id"]),
                cypher_literal(w.get("title")),
                cypher_literal(w.get("author")),
                cypher_literal(w.get("source")),
            )
        )
    lines.append("\n// 2) Entities\n")
    for e in entities:
        label = _label_for_entity_type(str(e.get("entity_type")))
        lines.append(
            "MERGE (e:Entity:%s {entity_id: %s})\n"
            "SET e.entity_type=%s, e.canonical_name=%s, e.aliases=%s, e.is_fictional=%s, e.source=%s, e.meta=%s;"
            % (
                label,
                cypher_literal(e["entity_id"]),
                cypher_literal(e.get("entity_type")),
                cypher_literal(e.get("canonical_name")),
                cypher_literal(e.get("aliases") or []),
                cypher_literal(e.get("is_fictional")),
                cypher_literal(e.get("source")),
                cypher_literal(e.get("meta") or {}),
            )
        )
    lines.append("\n// 3) Mentions\n")
    for m in mentions:
        lines.append(
            "MERGE (m:Mention {mention_id: %s})\n"
            "SET m.surface=%s, m.start_char=%s, m.end_char=%s, m.doc_id=%s, m.chunk_id=%s, m.source=%s, m.entity_type=%s, m.confidence=%s;"
            % (
                cypher_literal(m["mention_id"]),
                cypher_literal(m.get("surface")),
                cypher_literal(m.get("start_char")),
                cypher_literal(m.get("end_char")),
                cypher_literal(m.get("doc_id")),
                cypher_literal(m.get("chunk_id")),
                cypher_literal(m.get("source")),
                cypher_literal(m.get("entity_type")),
                cypher_literal(m.get("confidence")),
            )
        )
    lines.append("\n// 4) Work -> Mention and Mention -> Entity\n")
    for m in mentions:
        work_id = stable_id("work", str(m.get("source")), str(m.get("doc_id")))
        lines.append(
            "MATCH (w:Work {work_id: %s}), (m:Mention {mention_id: %s})\n"
            "MERGE (w)-[:HAS_MENTION]->(m);"
            % (cypher_literal(work_id), cypher_literal(m["mention_id"]))
        )
        lines.append(
            "MATCH (m:Mention {mention_id: %s}), (e:Entity {entity_id: %s})\n"
            "MERGE (m)-[:REFERS_TO]->(e);"
            % (cypher_literal(m["mention_id"]), cypher_literal(m.get("entity_id")))
        )
        if m.get("entity_type") == "Character":
            lines.append(
                "MATCH (w:Work {work_id: %s}), (e:Entity:Character {entity_id: %s})\n"
                "MERGE (w)-[:HAS_CHARACTER]->(e);"
                % (cypher_literal(work_id), cypher_literal(m.get("entity_id")))
            )
    lines.append("\n// 5) Entity relations\n")
    for r in relations:
        pred = str(r.get("predicate"))
        # Skip non-entity-to-entity predicates handled above.
        if pred in {"MENTIONS"}:
            continue
        lines.append(
            "MATCH (a:Entity {entity_id: %s}), (b:Entity {entity_id: %s})\n"
            "MERGE (a)-[r:%s {doc_id:%s, chunk_id:%s, source:%s, evidence_chunk_id:%s}]->(b)\n"
            "SET r.confidence=%s, r.evidence_text=%s, r.rel_type=%s;"
            % (
                cypher_literal(r.get("source_entity_id")),
                cypher_literal(r.get("target_entity_id")),
                pred,
                cypher_literal(r.get("doc_id")),
                cypher_literal(r.get("chunk_id")),
                cypher_literal(r.get("source")),
                cypher_literal(r.get("chunk_id")),
                cypher_literal(r.get("confidence")),
                cypher_literal(r.get("evidence_text")),
                cypher_literal(r.get("rel_type")),
            )
        )

    lines.append("\n// 6) Mention coreference (derived)\n")
    for ce in mention_coref:
        lines.append(
            "MATCH (a:Mention {mention_id: %s}), (b:Mention {mention_id: %s})\n"
            "MERGE (a)-[r:COREFERS_WITH {doc_id:%s, entity_id:%s}]->(b);"
            % (
                cypher_literal(ce.get("source_mention_id")),
                cypher_literal(ce.get("target_mention_id")),
                cypher_literal(ce.get("doc_id")),
                cypher_literal(ce.get("entity_id")),
            )
        )

    load_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"constraints": str(constraints_path), "load": str(load_path)}


def load_to_neo4j_unwind(
    *,
    uri: str,
    user: str,
    password: str,
    graph_records: Dict[str, object],
    database: Optional[str] = None,
    batch_size: int = 2000,
) -> None:
    """
    Optional direct loader. Requires `neo4j` Python driver.
    """
    try:
        from neo4j import GraphDatabase
    except Exception as e:  # pragma: no cover
        raise RuntimeError("neo4j driver not installed. Install 'neo4j' or set neo4j.enabled=false.") from e

    def batched(rows: List[dict]):
        for i in range(0, len(rows), batch_size):
            yield rows[i : i + batch_size]

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            # Constraints
            for stmt in build_constraints_cypher().splitlines():
                session.run(stmt)

            # Works
            q_work = """
            UNWIND $rows AS row
            MERGE (w:Work {work_id: row.work_id})
            SET w.title = row.title, w.author = row.author, w.source = row.source
            """
            for batch in batched(list(graph_records["works"])):  # type: ignore[arg-type]
                session.run(q_work, rows=batch)

            # Entities
            q_ent = """
            UNWIND $rows AS row
            MERGE (e:Entity {entity_id: row.entity_id})
            SET e.entity_type = row.entity_type,
                e.canonical_name = row.canonical_name,
                e.aliases = row.aliases,
                e.is_fictional = row.is_fictional,
                e.source = row.source,
                e.meta = row.meta
            WITH e, row
            CALL apoc.create.addLabels(e, [row.entity_type]) YIELD node
            RETURN count(*)
            """
            # APOC is optional; if missing, we still have :Entity. We'll best-effort.
            try_apoc = True
            for batch in batched(list(graph_records["entities"])):  # type: ignore[arg-type]
                if try_apoc:
                    try:
                        session.run(q_ent, rows=batch).consume()
                        continue
                    except Exception:
                        try_apoc = False
                q_ent_no_apoc = """
                UNWIND $rows AS row
                MERGE (e:Entity {entity_id: row.entity_id})
                SET e.entity_type = row.entity_type,
                    e.canonical_name = row.canonical_name,
                    e.aliases = row.aliases,
                    e.is_fictional = row.is_fictional,
                    e.source = row.source,
                    e.meta = row.meta
                """
                session.run(q_ent_no_apoc, rows=batch)

            # Mentions
            q_m = """
            UNWIND $rows AS row
            MERGE (m:Mention {mention_id: row.mention_id})
            SET m.surface=row.surface, m.start_char=row.start_char, m.end_char=row.end_char,
                m.doc_id=row.doc_id, m.chunk_id=row.chunk_id, m.source=row.source,
                m.entity_type=row.entity_type, m.confidence=row.confidence
            """
            for batch in batched(list(graph_records["mentions"])):  # type: ignore[arg-type]
                session.run(q_m, rows=batch)

            # Mention links + Work edges
            q_link = """
            UNWIND $rows AS row
            MATCH (m:Mention {mention_id: row.mention_id})
            MATCH (e:Entity {entity_id: row.entity_id})
            MERGE (m)-[:MENTIONS]->(e)
            WITH row, m
            MATCH (w:Work {work_id: row.work_id})
            MERGE (w)-[:MENTIONED_IN]->(m)
            """
            link_rows = []
            for m in graph_records["mentions"]:  # type: ignore[assignment]
                link_rows.append(
                    {
                        "mention_id": m["mention_id"],
                        "entity_id": m["entity_id"],
                        "work_id": stable_id("work", str(m.get("source")), str(m.get("doc_id"))),
                    }
                )
            for batch in batched(link_rows):
                session.run(q_link, rows=batch)

            # Entity relations (limited set via UNWIND + CASE)
            # Note: dynamic relationship types are not allowed in pure Cypher; we handle the small set explicitly.
            q_rel = """
            UNWIND $rows AS row
            MATCH (a:Entity {entity_id: row.source_entity_id})
            MATCH (b:Entity {entity_id: row.target_entity_id})
            CALL {
              WITH a, b, row
              WITH a, b, row WHERE row.predicate = 'INTERACTS_WITH'
              MERGE (a)-[r:INTERACTS_WITH {doc_id: row.doc_id, chunk_id: row.chunk_id, source: row.source, evidence_chunk_id: row.chunk_id}]->(b)
              SET r.confidence = row.confidence, r.evidence_text = row.evidence_text, r.rel_type = row.rel_type
              RETURN 1 AS _
              UNION
              WITH a, b, row WHERE row.predicate = 'LOCATED_IN'
              MERGE (a)-[r:LOCATED_IN {doc_id: row.doc_id, chunk_id: row.chunk_id, source: row.source, evidence_chunk_id: row.chunk_id}]->(b)
              SET r.confidence = row.confidence, r.evidence_text = row.evidence_text, r.rel_type = row.rel_type
              RETURN 1 AS _
              UNION
              WITH a, b, row WHERE row.predicate = 'SAME_AS'
              MERGE (a)-[r:SAME_AS {doc_id: row.doc_id, chunk_id: row.chunk_id, source: row.source, evidence_chunk_id: row.chunk_id}]->(b)
              SET r.confidence = row.confidence, r.evidence_text = row.evidence_text, r.rel_type = row.rel_type
              RETURN 1 AS _
              UNION
              WITH a, b, row WHERE row.predicate = 'BASED_ON'
              MERGE (a)-[r:BASED_ON {doc_id: row.doc_id, chunk_id: row.chunk_id, source: row.source, evidence_chunk_id: row.chunk_id}]->(b)
              SET r.confidence = row.confidence, r.evidence_text = row.evidence_text, r.rel_type = row.rel_type
              RETURN 1 AS _
            }
            RETURN count(*)
            """
            rel_rows = [
                r for r in graph_records["relations"]  # type: ignore[list-item]
                if r.get("predicate") in {"INTERACTS_WITH", "LOCATED_IN", "SAME_AS", "BASED_ON"}
            ]
            for batch in batched(rel_rows):
                session.run(q_rel, rows=batch)
    finally:
        driver.close()

