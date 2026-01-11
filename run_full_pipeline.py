"""
Run complete RoLit-KG pipeline with optimized resolution on full dataset.

This script:
1. Loads full RO-Stories and HistNERo datasets
2. Applies all pipeline stages with optimizations
3. Uses Ollama embeddings for entity resolution
4. Generates Neo4j Cypher scripts
5. Creates comprehensive reports
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import pipeline components
from src.pipeline.common import (
    ensure_dir,
    utc_timestamp,
    write_json,
    write_jsonl,
    stable_id,
    normalize_mention,
)
from src.pipeline.ingest import load_documents_from_jsonl, load_ro_stories_from_hf
from src.pipeline.normalize import normalize_documents
from src.pipeline.chunking import chunk_document
from src.pipeline.extract import extract_for_documents
from src.pipeline.validate import validate_artifacts
from src.pipeline.ground import ground_across_corpora_exact
from src.pipeline.neo4j_load import build_graph_records, write_cypher_files
from src.pipeline.evaluate import compute_metrics, render_markdown_report
from src.pipeline.analytics import compute_graph_analytics
from src.pipeline.kg_quality import quality_control
from src.pipeline.events import derive_event_nodes

# Import optimized resolution
from src.pipeline.resolve_ollama import OllamaEmbeddingResolver, OllamaConfig
from src.pipeline.resolve_optimized import (
    _cluster_entities_scipy,
    _create_candidate_relations,
    _merge_entities,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run RoLit-KG pipeline with optimized resolution")
    
    # Data sources
    parser.add_argument("--ro_stories_jsonl", type=str, default="data/rolit_kg_sample/ro_stories_sample.jsonl")
    parser.add_argument("--histnero_jsonl", type=str, default="data/rolit_kg_sample/histnero_sample.jsonl")
    parser.add_argument("--limit_ro_stories", type=int, default=None)
    parser.add_argument("--limit_histnero", type=int, default=None)
    
    # Ollama config
    parser.add_argument("--ollama_url", type=str, default="http://inference.ccrolabs.com")
    parser.add_argument("--ollama_model", type=str, default="nomic-embed-text")
    
    # Resolution config
    parser.add_argument(
        "--resolution_mode",
        type=str,
        default="ollama",
        choices=["ollama", "lexical"],
        help="Entity resolution mode. Use 'lexical' for large corpora to avoid O(n^2) similarity blow-ups.",
    )
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    parser.add_argument("--candidate_threshold", type=float, default=0.70)
    parser.add_argument("--resolution_scope", type=str, default="per_source", choices=["per_source", "global", "per_doc"])
    parser.add_argument(
        "--emit_candidate_relations",
        action="store_true",
        help="If set, emit CANDIDATE_SAME_AS relations (can be huge for large corpora).",
    )
    parser.add_argument(
        "--checkpoint_after_extract",
        action="store_true",
        help="If set, persist raw Stage-4 artifacts before resolution (recommended for large runs).",
    )
    parser.add_argument(
        "--emit_event_nodes",
        action="store_true",
        help="If set, derive explicit Event nodes with participant edges from extracted relations.",
    )
    
    # Chunking config
    parser.add_argument("--chunk_size", type=int, default=250)
    parser.add_argument("--chunk_overlap", type=int, default=40)
    
    # Extraction config (real NER + LLM relations)
    parser.add_argument(
        "--ner_engine",
        type=str,
        default="regex",
        choices=["regex", "transformers", "ensemble"],
        help="NER engine to use during extraction.",
    )
    parser.add_argument(
        "--ner_model",
        type=str,
        default="Davlan/xlm-roberta-base-ner-hrl",
        help="Transformers NER model name (used when --ner_engine transformers).",
    )
    parser.add_argument(
        "--relations_engine",
        type=str,
        default="heuristic",
        choices=["heuristic", "ollama"],
        help="Relation extraction engine. 'ollama' uses an LLM to produce JSON relations.",
    )
    parser.add_argument(
        "--ollama_rel_model",
        type=str,
        default=None,
        help="Ollama model for relation extraction (defaults to --ollama_model).",
    )
    parser.add_argument(
        "--ollama_rel_timeout_s",
        type=int,
        default=180,
        help="Timeout (seconds) for Ollama relation extraction requests.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for incremental rebuilds (LLM relation cache, etc.).",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="rolitkg_optimized")
    
    # Neo4j
    parser.add_argument("--no_neo4j", action="store_true")
    
    return parser.parse_args()

def _write_raw_checkpoint(
    out_dir: Path,
    docs,
    doc_chunks,
    mentions,
    entities,
    relations,
    val_report,
) -> None:
    """
    Persist Stage-4/5 artifacts before resolution so runs are resumable.
    """
    logger.info("Checkpoint: Writing raw artifacts (pre-resolution)...")

    # Prepare document rows
    docs_rows = []
    for d in docs:
        docs_rows.append({
            "doc_id": d.doc_id,
            "source": d.source,
            "title": d.title,
            "author": d.meta.get("author"),
            "text_len": len(d.text),
            "work_id": stable_id("work", d.source, d.doc_id),
            "meta": d.meta,
        })

    # Prepare chunk rows
    chunks_rows = []
    for d in docs:
        for ch in doc_chunks[d.doc_id]:
            chunks_rows.append({
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "source": ch.source,
                "start_char": ch.start_char,
                "end_char": ch.end_char,
                "text_len": len(ch.text),
            })

    write_json(out_dir / "artifacts" / "validation_raw.json", val_report)
    write_jsonl(out_dir / "artifacts" / "docs_raw.jsonl", docs_rows)
    write_jsonl(out_dir / "artifacts" / "chunks_raw.jsonl", chunks_rows)
    write_jsonl(out_dir / "artifacts" / "mentions_raw.jsonl", [m.__dict__ for m in mentions])
    write_jsonl(out_dir / "artifacts" / "entities_raw.jsonl", [e.__dict__ for e in entities])
    write_jsonl(out_dir / "artifacts" / "relations_raw.jsonl", [r.__dict__ for r in relations])

    logger.info("Checkpoint: Raw artifacts written.")


def resolve_entities_lexical(
    entities,
    mentions,
    scope: str,
):
    """
    Fast, scalable entity resolution by normalized surface form.

    This avoids O(n^2) similarity and is appropriate for full-corpus runs.
    """
    logger.info(f"Starting lexical resolution for {len(entities)} entities")

    # Group entities by scope
    entity_groups: Dict[str, List] = {}
    for e in entities:
        if scope == "per_source":
            key = e.source
        elif scope == "per_doc":
            doc_mentions = [m for m in mentions if m.entity_id == e.entity_id]
            key = doc_mentions[0].doc_id if doc_mentions else "unknown"
        else:  # global
            key = "global"

        entity_groups.setdefault(key, []).append(e)

    resolved_entities = []
    old_to_new = {}

    for group_key, group_entities in entity_groups.items():
        logger.info(f"Lexical resolution group '{group_key}' with {len(group_entities)} entities")

        buckets: Dict[str, List[int]] = {}
        for idx, e in enumerate(group_entities):
            k = normalize_mention(e.canonical_name or "")
            if not k:
                k = e.canonical_name or e.entity_id
            buckets.setdefault(k, []).append(idx)

        for _, idxs in buckets.items():
            if len(idxs) == 1:
                e = group_entities[idxs[0]]
                resolved_entities.append(e)
                old_to_new[e.entity_id] = e.entity_id
            else:
                cluster_entities = [group_entities[i] for i in idxs]
                merged = _merge_entities(cluster_entities, mentions)
                resolved_entities.append(merged)
                for e in cluster_entities:
                    old_to_new[e.entity_id] = merged.entity_id

    # Update mentions
    updated_mentions = []
    from src.pipeline.common import Mention
    for m in mentions:
        new_entity_id = old_to_new.get(m.entity_id, m.entity_id)
        updated_mentions.append(
            Mention(
                mention_id=m.mention_id,
                entity_id=new_entity_id,
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

    logger.info(f"Lexical resolution complete: {len(entities)} -> {len(resolved_entities)} entities")
    return resolved_entities, updated_mentions, old_to_new, []


def resolve_entities_with_ollama(
    entities,
    mentions,
    config: OllamaConfig,
    similarity_threshold: float,
    candidate_threshold: float,
    scope: str,
):
    """
    Resolve entities using Ollama embeddings.
    
    Returns:
        - resolved_entities
        - updated_mentions
        - old_to_new mapping
        - candidate_relations
    """
    logger.info(f"Starting Ollama-based resolution for {len(entities)} entities")
    
    # Initialize resolver
    resolver = OllamaEmbeddingResolver(config)
    
    # Group entities by scope
    entity_groups: Dict[str, List] = {}
    for e in entities:
        if scope == "per_source":
            key = e.source
        elif scope == "per_doc":
            doc_mentions = [m for m in mentions if m.entity_id == e.entity_id]
            key = doc_mentions[0].doc_id if doc_mentions else "unknown"
        else:  # global
            key = "global"
        
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(e)
    
    logger.info(f"Grouped into {len(entity_groups)} scopes")
    
    # Resolve within each group
    resolved_entities = []
    old_to_new = {}
    all_candidate_relations = []
    
    for group_key, group_entities in entity_groups.items():
        logger.info(f"Processing group '{group_key}' with {len(group_entities)} entities")
        
        # Find similar pairs
        start = time.time()
        similar_pairs = resolver.find_similar_pairs(group_entities, similarity_threshold)
        elapsed = time.time() - start
        logger.info(f"Found {len(similar_pairs)} similar pairs in {elapsed:.2f}s")
        
        # Find candidate pairs
        candidate_pairs = resolver.find_similar_pairs(group_entities, candidate_threshold)
        similar_set = {(i, j) for i, j, _ in similar_pairs}
        candidate_pairs = [(i, j, sim) for i, j, sim in candidate_pairs if (i, j) not in similar_set]
        logger.info(f"Found {len(candidate_pairs)} candidate pairs")
        
        # Cluster
        clusters = _cluster_entities_scipy(len(group_entities), similar_pairs)
        logger.info(f"Formed {len(clusters)} clusters")
        
        # Create candidate relations
        group_candidates = _create_candidate_relations(group_entities, candidate_pairs)
        all_candidate_relations.extend(group_candidates)
        
        # Merge entities
        for cluster in clusters:
            if len(cluster) == 1:
                e = group_entities[cluster[0]]
                resolved_entities.append(e)
                old_to_new[e.entity_id] = e.entity_id
            else:
                cluster_entities = [group_entities[i] for i in cluster]
                merged = _merge_entities(cluster_entities, mentions)
                resolved_entities.append(merged)
                for e in cluster_entities:
                    old_to_new[e.entity_id] = merged.entity_id
    
    # Update mentions
    updated_mentions = []
    for m in mentions:
        from src.pipeline.common import Mention
        new_entity_id = old_to_new.get(m.entity_id, m.entity_id)
        updated_mentions.append(
            Mention(
                mention_id=m.mention_id,
                entity_id=new_entity_id,
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
    
    logger.info(f"Resolution complete: {len(entities)} -> {len(resolved_entities)} entities")
    logger.info(f"Cache stats: {resolver.get_cache_stats()}")
    
    return resolved_entities, updated_mentions, old_to_new, all_candidate_relations


def remap_relations_entity_ids(relations, old_to_new):
    """Remap relation entity IDs after resolution."""
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


def main():
    """Run complete RoLit-KG pipeline."""
    args = parse_args()
    
    # Create output directory
    if args.output_dir:
        out_dir = ensure_dir(args.output_dir)
    else:
        timestamp = utc_timestamp()
        out_dir = ensure_dir(f"outputs/rolit_kg/{args.run_name}_{timestamp}")
    
    ensure_dir(out_dir / "artifacts")
    ensure_dir(out_dir / "cypher")
    ensure_dir(out_dir / "reports")
    
    logger.info(f"Output directory: {out_dir}")
    logger.info("="*80)
    logger.info("ROLIT-KG OPTIMIZED PIPELINE")
    logger.info("="*80)
    
    # ========================================
    # Stage 1: Ingest
    # ========================================
    logger.info("\nStage 1: Ingesting documents...")
    start_total = time.time()
    
    # Load RO-Stories
    logger.info(f"Loading RO-Stories from {args.ro_stories_jsonl}")
    ro_docs = load_documents_from_jsonl(args.ro_stories_jsonl, source="ro_stories")
    if args.limit_ro_stories is not None:
        ro_docs = ro_docs[:args.limit_ro_stories]
    logger.info(f"Loaded {len(ro_docs)} RO-Stories documents")
    
    # Load HistNERo
    logger.info(f"Loading HistNERo from {args.histnero_jsonl}")
    hist_docs = load_documents_from_jsonl(args.histnero_jsonl, source="histnero")
    if args.limit_histnero is not None:
        hist_docs = hist_docs[:args.limit_histnero]
    logger.info(f"Loaded {len(hist_docs)} HistNERo documents")
    
    all_docs = ro_docs + hist_docs
    logger.info(f"Total documents: {len(all_docs)}")
    
    # ========================================
    # Stage 2: Normalize
    # ========================================
    logger.info("\nStage 2: Normalizing documents...")
    docs = normalize_documents(all_docs)
    logger.info(f"Normalized {len(docs)} documents")
    
    # ========================================
    # Stage 3: Chunk
    # ========================================
    logger.info("\nStage 3: Chunking documents...")
    doc_chunks = {}
    total_chunks = 0
    for d in docs:
        chunks = chunk_document(
            doc_id=d.doc_id,
            source=d.source,
            text=d.text,
            target_tokens=args.chunk_size,
            overlap_tokens=args.chunk_overlap,
            tokenizer_model=None,  # Use whitespace
        )
        doc_chunks[d.doc_id] = chunks
        total_chunks += len(chunks)
    
    logger.info(f"Created {total_chunks} chunks")
    
    # ========================================
    # Stage 4: Extract
    # ========================================
    logger.info("\nStage 4: Extracting entities and relations...")
    start_extract = time.time()
    
    rel_llm_model = args.ollama_rel_model or args.ollama_model

    mentions, entities, relations = extract_for_documents(
        docs,
        doc_chunks,
        ner_engine=args.ner_engine,
        ner_model=args.ner_model if args.ner_engine in ("transformers", "ensemble") else None,
        relations_engine=args.relations_engine,
        llm_base_url=args.ollama_url if args.relations_engine == "ollama" else None,
        llm_model=rel_llm_model if args.relations_engine == "ollama" else None,
        llm_timeout_s=args.ollama_rel_timeout_s if args.relations_engine == "ollama" else 60,
        cache_dir=args.cache_dir,
    )
    
    elapsed_extract = time.time() - start_extract
    logger.info(f"Extracted {len(mentions)} mentions, {len(entities)} entities, {len(relations)} relations")
    logger.info(f"Extraction time: {elapsed_extract:.2f}s")
    
    # ========================================
    # Stage 5: Validate
    # ========================================
    logger.info("\nStage 5: Validating artifacts...")
    val_report = validate_artifacts(mentions=mentions, entities=entities, relations=relations)
    logger.info(f"Validation: {val_report.get('summary', {})}")

    if args.checkpoint_after_extract:
        _write_raw_checkpoint(
            out_dir=out_dir,
            docs=docs,
            doc_chunks=doc_chunks,
            mentions=mentions,
            entities=entities,
            relations=relations,
            val_report=val_report,
        )
    
    # ========================================
    # Stage 6: Resolve with Ollama
    # ========================================
    logger.info("\nStage 6: Resolving entities...")
    start_resolve = time.time()
    
    ollama_config = OllamaConfig(
        base_url=args.ollama_url,
        model=args.ollama_model,
    )
    
    if args.resolution_mode == "lexical":
        resolved_entities, mentions_resolved, old_to_new, candidate_relations = resolve_entities_lexical(
            entities=entities,
            mentions=mentions,
            scope=args.resolution_scope,
        )
    else:
        resolved_entities, mentions_resolved, old_to_new, candidate_relations = resolve_entities_with_ollama(
            entities,
            mentions,
            ollama_config,
            args.similarity_threshold,
            args.candidate_threshold,
            args.resolution_scope,
        )

        if not args.emit_candidate_relations:
            candidate_relations = []
    
    elapsed_resolve = time.time() - start_resolve
    logger.info(f"Resolution time: {elapsed_resolve:.2f}s")
    logger.info(f"Entities: {len(entities)} -> {len(resolved_entities)}")
    logger.info(f"Merged: {len(entities) - len(resolved_entities)} entities")
    logger.info(f"Candidate matches: {len(candidate_relations)}")
    
    # Remap relations
    relations_resolved = remap_relations_entity_ids(relations, old_to_new)

    # ========================================
    # QC: Graph-level quality control
    # ========================================
    logger.info("\nQC: Applying graph-level constraints/dedup/junk suppression...")
    resolved_entities, mentions_resolved, relations_resolved, qc_report = quality_control(
        entities=resolved_entities,
        mentions=mentions_resolved,
        relations=relations_resolved,
    )
    write_json(out_dir / "reports" / "qc_report.json", qc_report.__dict__)
    logger.info(
        f"QC complete: entities={len(resolved_entities)}, mentions={len(mentions_resolved)}, relations={len(relations_resolved)}"
    )

    if args.emit_event_nodes:
        logger.info("Deriving Event nodes from relations...")
        event_entities, event_relations = derive_event_nodes(
            entities=resolved_entities,
            relations=relations_resolved,
        )
        # Append events into the entity list; keep relations separate until combine step.
        resolved_entities = list(resolved_entities) + event_entities
        relations_resolved = list(relations_resolved) + event_relations
        logger.info(f"Added {len(event_entities)} Event nodes and {len(event_relations)} event edges")
    
    # ========================================
    # Stage 7: Ground across corpora
    # ========================================
    logger.info("\nStage 7: Grounding across corpora...")
    grounding_relations = []
    if hist_docs and ro_docs:
        grounding_relations = ground_across_corpora_exact(
            resolved_entities=resolved_entities,
            mentions=mentions_resolved,
        )
        logger.info(f"Found {len(grounding_relations)} grounding relations")
    
    # Combine all relations
    all_relations = relations_resolved + grounding_relations + candidate_relations
    # Final QC pass including grounding/candidates
    logger.info("QC: Final pass on combined relations...")
    resolved_entities, mentions_resolved, all_relations, qc_report2 = quality_control(
        entities=resolved_entities,
        mentions=mentions_resolved,
        relations=all_relations,
    )
    write_json(out_dir / "reports" / "qc_report_final.json", qc_report2.__dict__)
    logger.info(f"Total relations: {len(all_relations)}")
    
    # ========================================
    # Stage 8: Persist artifacts
    # ========================================
    logger.info("\nStage 8: Persisting artifacts...")
    
    # Prepare document rows
    docs_rows = []
    for d in docs:
        docs_rows.append({
            "doc_id": d.doc_id,
            "source": d.source,
            "title": d.title,
            "author": d.meta.get("author"),
            "text_len": len(d.text),
            "work_id": stable_id("work", d.source, d.doc_id),
            "meta": d.meta,
        })
    
    # Prepare chunk rows
    chunks_rows = []
    for d in docs:
        for ch in doc_chunks[d.doc_id]:
            chunks_rows.append({
                "chunk_id": ch.chunk_id,
                "doc_id": ch.doc_id,
                "source": ch.source,
                "start_char": ch.start_char,
                "end_char": ch.end_char,
                "text_len": len(ch.text),
            })
    
    # Write artifacts
    write_json(out_dir / "artifacts" / "validation.json", val_report)
    write_jsonl(out_dir / "artifacts" / "docs.jsonl", docs_rows)
    write_jsonl(out_dir / "artifacts" / "chunks.jsonl", chunks_rows)
    write_jsonl(out_dir / "artifacts" / "mentions.jsonl", [m.__dict__ for m in mentions_resolved])
    write_jsonl(out_dir / "artifacts" / "entities.jsonl", [e.__dict__ for e in resolved_entities])
    write_jsonl(out_dir / "artifacts" / "relations.jsonl", [r.__dict__ for r in all_relations])
    
    logger.info(f"Wrote artifacts to {out_dir / 'artifacts'}")
    
    # ========================================
    # Stage 9: Compute metrics and reports
    # ========================================
    logger.info("\nStage 9: Computing metrics...")
    
    metrics = compute_metrics(
        docs=docs_rows,
        chunks_count=len(chunks_rows),
        mentions=mentions_resolved,
        entities=resolved_entities,
        relations=all_relations,
    )
    
    write_json(out_dir / "reports" / "metrics.json", metrics)
    (out_dir / "reports" / "report.md").write_text(render_markdown_report(metrics), encoding="utf-8")
    
    logger.info("Metrics report generated")
    
    # ========================================
    # Stage 10: Graph analytics
    # ========================================
    logger.info("\nStage 10: Computing graph analytics...")
    
    try:
        graph_metrics = compute_graph_analytics(
            resolved_entities,
            all_relations,
            output_path=str(out_dir / "reports" / "analytics.json"),
        )
        logger.info(f"Graph analytics: {graph_metrics.node_count} nodes, {graph_metrics.edge_count} edges")
    except Exception as e:
        logger.error(f"Graph analytics failed: {e}")
    
    # ========================================
    # Stage 11: Generate Neo4j Cypher
    # ========================================
    logger.info("\nStage 11: Generating Neo4j Cypher scripts...")
    
    graph_records = build_graph_records(
        docs=docs_rows,
        mentions=mentions_resolved,
        entities=resolved_entities,
        relations=all_relations,
    )
    
    write_json(out_dir / "artifacts" / "graph_records.json", graph_records)
    cypher_paths = write_cypher_files(out_dir=out_dir / "cypher", graph_records=graph_records)
    
    logger.info(f"Cypher scripts generated:")
    logger.info(f"  - Constraints: {cypher_paths['constraints']}")
    logger.info(f"  - Load: {cypher_paths['load']}")
    
    # ========================================
    # Summary
    # ========================================
    elapsed_total = time.time() - start_total
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"\nTotal time: {elapsed_total:.2f}s ({elapsed_total/60:.2f} minutes)")
    logger.info(f"Extraction: {elapsed_extract:.2f}s")
    logger.info(f"Resolution: {elapsed_resolve:.2f}s")
    logger.info(f"\nStatistics:")
    logger.info(f"  Documents: {len(docs)}")
    logger.info(f"  Chunks: {total_chunks}")
    logger.info(f"  Mentions: {len(mentions_resolved)}")
    logger.info(f"  Entities (pre-resolution): {len(entities)}")
    logger.info(f"  Entities (post-resolution): {len(resolved_entities)}")
    logger.info(f"  Relations: {len(all_relations)}")
    logger.info(f"    - Extracted: {len(relations_resolved)}")
    logger.info(f"    - Grounding: {len(grounding_relations)}")
    logger.info(f"    - Candidates: {len(candidate_relations)}")
    logger.info(f"\nOutput: {out_dir}")
    logger.info(f"  - Artifacts: {out_dir / 'artifacts'}")
    logger.info(f"  - Reports: {out_dir / 'reports'}")
    logger.info(f"  - Cypher: {out_dir / 'cypher'}")
    
    # Write summary
    summary = {
        "run_name": args.run_name,
        "timestamp": utc_timestamp(),
        "elapsed_seconds": elapsed_total,
        "stats": {
            "documents": len(docs),
            "chunks": total_chunks,
            "mentions": len(mentions_resolved),
            "entities_pre_resolution": len(entities),
            "entities_post_resolution": len(resolved_entities),
            "entities_merged": len(entities) - len(resolved_entities),
            "relations_total": len(all_relations),
            "relations_extracted": len(relations_resolved),
            "relations_grounding": len(grounding_relations),
            "relations_candidates": len(candidate_relations),
        },
        "config": {
            "ollama_url": args.ollama_url,
            "ollama_model": args.ollama_model,
            "similarity_threshold": args.similarity_threshold,
            "candidate_threshold": args.candidate_threshold,
            "resolution_scope": args.resolution_scope,
            "chunk_size": args.chunk_size,
        },
    }
    
    write_json(out_dir / "summary.json", summary)
    logger.info(f"\nSummary written to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
