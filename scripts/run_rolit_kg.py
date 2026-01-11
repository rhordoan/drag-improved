#!/usr/bin/env python3
"""
RoLit-KG MVP runner (RO-Stories + HistNERo -> Neo4j/Cypher + reports).

This is intentionally independent from the D-RAG training codepaths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pipeline.chunking import chunk_document
from src.pipeline.common import (
    Chunk,
    Document,
    ensure_dir,
    utc_timestamp,
    write_json,
    write_jsonl,
)
from src.pipeline.evaluate import compute_metrics, render_markdown_report, sample_relations_for_audit
from src.pipeline.extract import extract_for_documents
from src.pipeline.ground import ground_across_corpora_exact
from src.pipeline.ingest import load_documents_from_jsonl, load_ro_stories_from_hf
from src.pipeline.neo4j_load import build_graph_records, load_to_neo4j_unwind, write_cypher_files
from src.pipeline.normalize import normalize_documents
from src.pipeline.resolve import resolve_entities_lexical
from src.pipeline.validate import validate_artifacts
from src.pipeline.common import stable_id


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Config is YAML; install PyYAML (pip install pyyaml) or use a .json config.") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def remap_relations_entity_ids(relations, old_to_new):
    from src.pipeline.common import Relation

    out = []
    for r in relations:
        out.append(
            Relation(
                source_entity_id=old_to_new.get(r.source_entity_id, r.source_entity_id),
                predicate=r.predicate,
                target_entity_id=old_to_new.get(r.target_entity_id, r.target_entity_id),
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                source=r.source,
                confidence=r.confidence,
                evidence_text=r.evidence_text,
                rel_type=r.rel_type,
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RoLit-KG MVP pipeline")
    parser.add_argument("--config", type=str, default="configs/rolit_kg.yaml", help="Path to YAML/JSON config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--limit_ro_stories", type=int, default=None, help="Limit number of RO-Stories docs")
    parser.add_argument("--limit_histnero", type=int, default=None, help="Limit number of HistNERo docs (JSONL only)")
    parser.add_argument("--no_neo4j", action="store_true", help="Skip Neo4j direct loading (still writes Cypher files)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_name = str(cfg.get("run_name") or "rolit_kg")
    base_out = args.output_dir or str(cfg.get("output_dir") or f"data/processed/rolit_kg/{run_name}_{utc_timestamp()}")
    out_dir = ensure_dir(base_out)
    ensure_dir(out_dir / "artifacts")
    ensure_dir(out_dir / "cypher")
    ensure_dir(out_dir / "reports")

    # --------------------
    # Ingest
    # --------------------
    ro_cfg = cfg.get("ro_stories") or {}
    hist_cfg = cfg.get("histnero") or {}

    ro_docs: List[Document] = []
    if ro_cfg.get("path_jsonl"):
        ro_docs = load_documents_from_jsonl(ro_cfg["path_jsonl"], source="ro_stories")
    else:
        if ro_cfg.get("hf_dataset"):
            ro_docs = load_ro_stories_from_hf(
                dataset_name=str(ro_cfg.get("hf_dataset")),
                split=str(ro_cfg.get("hf_split") or "train"),
                revision=ro_cfg.get("hf_revision"),
                text_field=str(ro_cfg.get("text_field") or "text"),
                title_field=ro_cfg.get("title_field"),
                author_field=ro_cfg.get("author_field"),
                limit=args.limit_ro_stories if args.limit_ro_stories is not None else ro_cfg.get("limit"),
            )
        else:
            # Fully offline default: use included sample JSONL
            ro_docs = load_documents_from_jsonl("data/rolit_kg_sample/ro_stories_sample.jsonl", source="ro_stories")
            if args.limit_ro_stories is not None:
                ro_docs = ro_docs[: args.limit_ro_stories]

    hist_docs: List[Document] = []
    if hist_cfg.get("path_jsonl"):
        hist_docs = load_documents_from_jsonl(hist_cfg["path_jsonl"], source="histnero")
        if args.limit_histnero is not None:
            hist_docs = hist_docs[: args.limit_histnero]
        elif hist_cfg.get("limit") is not None:
            hist_docs = hist_docs[: int(hist_cfg["limit"])]

    docs = normalize_documents([*ro_docs, *hist_docs])

    # --------------------
    # Chunk
    # --------------------
    ch_cfg = cfg.get("chunking") or {}
    target_tokens = int(ch_cfg.get("target_tokens") or 1000)
    overlap_tokens = int(ch_cfg.get("overlap_tokens") or 150)
    tokenizer_model = ch_cfg.get("tokenizer_model")
    if tokenizer_model is not None:
        tokenizer_model = str(tokenizer_model)

    doc_chunks: Dict[str, List[Chunk]] = {}
    for d in docs:
        doc_chunks[d.doc_id] = chunk_document(
            doc_id=d.doc_id,
            source=d.source,
            text=d.text,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            tokenizer_model=tokenizer_model,
        )

    # --------------------
    # Extract
    # --------------------
    ex_cfg = cfg.get("extraction") or {}
    ner_engine = str(ex_cfg.get("ner_engine") or "regex")
    ner_model = ex_cfg.get("ner_model")
    if ner_model is not None:
        ner_model = str(ner_model)
    relations_engine = str(ex_cfg.get("relations_engine") or "heuristic")

    mentions, entities, relations = extract_for_documents(
        docs,
        doc_chunks,
        ner_engine=ner_engine,
        ner_model=ner_model,
        relations_engine=relations_engine,
    )

    # --------------------
    # Resolve + Ground
    # --------------------
    res_cfg = cfg.get("resolution") or {}
    scope = str(res_cfg.get("scope") or "per_source")
    resolved_entities, mentions_resolved, old_to_new = resolve_entities_lexical(
        entities=entities,
        mentions=mentions,
        scope=scope,
        min_norm_len=int(res_cfg.get("min_norm_len") or 2),
    )
    relations_resolved = remap_relations_entity_ids(relations, old_to_new)

    grounding_relations = []
    if hist_docs:
        grounding_relations = ground_across_corpora_exact(
            resolved_entities=resolved_entities,
            mentions=mentions_resolved,
        )

    all_relations = [*relations_resolved, *grounding_relations]

    # --------------------
    # Validate
    # --------------------
    val_report = validate_artifacts(mentions=mentions_resolved, entities=resolved_entities, relations=all_relations)

    # --------------------
    # Persist artifacts
    # --------------------
    docs_rows = []
    for d in docs:
        docs_rows.append(
            {
                "doc_id": d.doc_id,
                "source": d.source,
                "title": d.title,
                "author": d.meta.get("author"),
                "text_len": len(d.text),
                "work_id": stable_id("work", d.source, d.doc_id),
                "meta": d.meta,
            }
        )
    chunks_rows = []
    for d in docs:
        for ch in doc_chunks[d.doc_id]:
            chunks_rows.append(
                {
                    "chunk_id": ch.chunk_id,
                    "doc_id": ch.doc_id,
                    "source": ch.source,
                    "start_char": ch.start_char,
                    "end_char": ch.end_char,
                    "text_len": len(ch.text),
                }
            )

    write_json(out_dir / "artifacts" / "config_resolved.json", cfg)
    write_json(out_dir / "artifacts" / "validation.json", val_report)
    write_jsonl(out_dir / "artifacts" / "docs.jsonl", docs_rows)
    write_jsonl(out_dir / "artifacts" / "chunks.jsonl", chunks_rows)
    write_jsonl(out_dir / "artifacts" / "mentions.jsonl", [m.__dict__ for m in mentions_resolved])
    write_jsonl(out_dir / "artifacts" / "entities.jsonl", [e.__dict__ for e in resolved_entities])
    write_jsonl(out_dir / "artifacts" / "relations.jsonl", [r.__dict__ for r in all_relations])

    # Audit sample
    audit_cfg = cfg.get("audit") or {}
    audit_n = int(audit_cfg.get("n") or 100)
    audit_seed = int(audit_cfg.get("seed") or 42)
    audit_rows = sample_relations_for_audit(
        all_relations, n=audit_n, seed=audit_seed, predicates={"INTERACTS_WITH", "LOCATED_IN", "BASED_ON", "SAME_AS"}
    )
    write_json(out_dir / "reports" / "audit_sample.json", audit_rows)

    # Metrics report
    metrics = compute_metrics(
        docs=docs_rows,
        chunks_count=len(chunks_rows),
        mentions=mentions_resolved,
        entities=resolved_entities,
        relations=all_relations,
    )
    write_json(out_dir / "reports" / "metrics.json", metrics)
    (out_dir / "reports" / "report.md").write_text(render_markdown_report(metrics), encoding="utf-8")

    # --------------------
    # Neo4j export / load
    # --------------------
    graph_records = build_graph_records(docs=docs_rows, mentions=mentions_resolved, entities=resolved_entities, relations=all_relations)
    write_json(out_dir / "artifacts" / "graph_records.json", graph_records)
    cypher_paths = write_cypher_files(out_dir=out_dir / "cypher", graph_records=graph_records)

    neo_cfg = cfg.get("neo4j") or {}
    neo_enabled = bool(neo_cfg.get("enabled", False)) and not args.no_neo4j
    if neo_enabled:
        load_to_neo4j_unwind(
            uri=str(neo_cfg.get("uri")),
            user=str(neo_cfg.get("user")),
            password=str(neo_cfg.get("password")),
            database=neo_cfg.get("database"),
            graph_records=graph_records,
            batch_size=int(neo_cfg.get("batch_size") or 2000),
        )

    print(f"RoLit-KG MVP complete. Output: {out_dir}")
    print(f"- Cypher constraints: {cypher_paths['constraints']}")
    print(f"- Cypher load:        {cypher_paths['load']}")
    print(f"- Report:             {out_dir / 'reports' / 'report.md'}")


if __name__ == "__main__":
    main()

