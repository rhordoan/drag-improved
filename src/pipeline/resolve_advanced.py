"""
Advanced entity resolution using embeddings and clustering.

Features:
- Embedding-based similarity
- Hierarchical clustering
- Cross-lingual entity matching
- Confidence scoring for CANDIDATE_SAME_AS edges
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .common import Entity, Mention, Relation, normalize_mention, stable_id


@dataclass
class ResolutionConfig:
    """Configuration for advanced entity resolution."""
    use_embeddings: bool = True
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.85
    candidate_threshold: float = 0.70  # For CANDIDATE_SAME_AS edges
    
    use_lexical: bool = True
    lexical_threshold: float = 0.9
    
    scope: str = "per_source"  # "per_source" | "global" | "per_doc"
    min_norm_len: int = 2
    
    cache_embeddings: bool = True
    batch_size: int = 32


class EmbeddingResolver:
    """Entity resolution using semantic embeddings."""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self._model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                print(f"Warning: Failed to load embedding model: {e}")
                print("Falling back to lexical matching only.")
                self.config.use_embeddings = False
        return self._model
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with caching."""
        if not self.config.use_embeddings:
            return None
        
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        model = self._get_model()
        if model is None:
            return None
        
        try:
            embedding = model.encode([text], convert_to_numpy=True)[0]
            if self.config.cache_embeddings:
                self._embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Warning: Failed to encode text: {e}")
            return None
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return similarity


def resolve_entities_advanced(
    entities: List[Entity],
    mentions: List[Mention],
    config: ResolutionConfig,
) -> Tuple[List[Entity], List[Mention], Dict[str, str], List[Relation]]:
    """
    Advanced entity resolution with embeddings and confidence scoring.
    
    Returns:
        - Resolved entities
        - Updated mentions with new entity_id
        - Mapping from old entity_id to new entity_id
        - CANDIDATE_SAME_AS relations for ambiguous matches
    """
    if not entities:
        return [], mentions, {}, []
    
    resolver = EmbeddingResolver(config)
    
    # Group entities by scope
    entity_groups: Dict[str, List[Entity]] = {}
    for e in entities:
        if config.scope == "per_source":
            key = e.source
        elif config.scope == "per_doc":
            # Extract doc_id from first mention
            doc_mentions = [m for m in mentions if m.entity_id == e.entity_id]
            key = doc_mentions[0].doc_id if doc_mentions else "unknown"
        else:  # global
            key = "global"
        
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(e)
    
    # Resolve within each group
    resolved_entities: List[Entity] = []
    old_to_new: Dict[str, str] = {}
    candidate_relations: List[Relation] = []
    
    for group_key, group_entities in entity_groups.items():
        group_resolved, group_mapping, group_candidates = _resolve_entity_group(
            group_entities,
            mentions,
            config,
            resolver,
        )
        resolved_entities.extend(group_resolved)
        old_to_new.update(group_mapping)
        candidate_relations.extend(group_candidates)
    
    # Update mentions with new entity IDs
    updated_mentions = []
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
    
    return resolved_entities, updated_mentions, old_to_new, candidate_relations


def _resolve_entity_group(
    entities: List[Entity],
    mentions: List[Mention],
    config: ResolutionConfig,
    resolver: EmbeddingResolver,
) -> Tuple[List[Entity], Dict[str, str], List[Relation]]:
    """Resolve entities within a single group using hierarchical clustering."""
    if len(entities) <= 1:
        return entities, {}, []
    
    # Build similarity matrix
    n = len(entities)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            e1, e2 = entities[i], entities[j]
            
            # Lexical similarity
            norm1 = normalize_mention(e1.canonical_name)
            norm2 = normalize_mention(e2.canonical_name)
            
            lexical_sim = 1.0 if norm1 == norm2 else 0.0
            
            # Embedding similarity
            embedding_sim = 0.0
            if config.use_embeddings:
                embedding_sim = resolver.compute_similarity(e1.canonical_name, e2.canonical_name)
            
            # Combined similarity
            if config.use_lexical and config.use_embeddings:
                similarity = 0.4 * lexical_sim + 0.6 * embedding_sim
            elif config.use_lexical:
                similarity = lexical_sim
            else:
                similarity = embedding_sim
            
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Hierarchical clustering using similarity threshold
    clusters = _cluster_entities(entities, similarity_matrix, config.similarity_threshold)
    
    # Find candidate matches (above candidate threshold but below similarity threshold)
    candidate_relations = _find_candidate_matches(
        entities,
        similarity_matrix,
        config.candidate_threshold,
        config.similarity_threshold,
    )
    
    # Create merged entities
    resolved_entities: List[Entity] = []
    old_to_new: Dict[str, str] = {}
    
    for cluster in clusters:
        if len(cluster) == 1:
            # No merge needed
            e = entities[cluster[0]]
            resolved_entities.append(e)
            old_to_new[e.entity_id] = e.entity_id
        else:
            # Merge cluster
            cluster_entities = [entities[i] for i in cluster]
            merged_entity = _merge_entities(cluster_entities, mentions)
            resolved_entities.append(merged_entity)
            
            for e in cluster_entities:
                old_to_new[e.entity_id] = merged_entity.entity_id
    
    return resolved_entities, old_to_new, candidate_relations


def _cluster_entities(
    entities: List[Entity],
    similarity_matrix: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    """Simple agglomerative clustering based on similarity threshold."""
    n = len(entities)
    clusters = [[i] for i in range(n)]
    
    while True:
        # Find most similar pair of clusters
        max_sim = -1.0
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Average linkage: average similarity between all pairs
                sims = []
                for ei in clusters[i]:
                    for ej in clusters[j]:
                        sims.append(similarity_matrix[ei, ej])
                
                if sims:
                    avg_sim = np.mean(sims)
                    if avg_sim > max_sim:
                        max_sim = avg_sim
                        merge_i, merge_j = i, j
        
        # Stop if no pair exceeds threshold
        if max_sim < threshold:
            break
        
        # Merge clusters
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]
    
    return clusters


def _find_candidate_matches(
    entities: List[Entity],
    similarity_matrix: np.ndarray,
    candidate_threshold: float,
    similarity_threshold: float,
) -> List[Relation]:
    """Find candidate matches that are ambiguous (similarity in intermediate range)."""
    candidate_relations: List[Relation] = []
    
    n = len(entities)
    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            
            # Candidate if between thresholds
            if candidate_threshold <= sim < similarity_threshold:
                e1, e2 = entities[i], entities[j]
                
                # Create bidirectional CANDIDATE_SAME_AS edges
                rel1 = Relation(
                    source_entity_id=e1.entity_id,
                    predicate="CANDIDATE_SAME_AS",
                    target_entity_id=e2.entity_id,
                    doc_id="",  # Cross-document relation
                    chunk_id="",
                    source="resolution",
                    confidence=float(sim),
                    evidence_text=f"Similarity: {sim:.2f}",
                    rel_type="candidate_match",
                )
                candidate_relations.append(rel1)
                
                rel2 = Relation(
                    source_entity_id=e2.entity_id,
                    predicate="CANDIDATE_SAME_AS",
                    target_entity_id=e1.entity_id,
                    doc_id="",
                    chunk_id="",
                    source="resolution",
                    confidence=float(sim),
                    evidence_text=f"Similarity: {sim:.2f}",
                    rel_type="candidate_match",
                )
                candidate_relations.append(rel2)
    
    return candidate_relations


def _merge_entities(
    cluster_entities: List[Entity],
    mentions: List[Mention],
) -> Entity:
    """Merge a cluster of entities into a single canonical entity."""
    # Choose entity with most mentions as canonical
    entity_mention_counts = {}
    for e in cluster_entities:
        count = sum(1 for m in mentions if m.entity_id == e.entity_id)
        entity_mention_counts[e.entity_id] = count
    
    canonical = max(cluster_entities, key=lambda e: entity_mention_counts[e.entity_id])
    
    # Collect all aliases
    all_aliases = set(canonical.aliases)
    for e in cluster_entities:
        all_aliases.add(e.canonical_name)
        all_aliases.update(e.aliases)
    
    # Remove canonical name from aliases
    all_aliases.discard(canonical.canonical_name)
    
    # Create new entity ID based on canonical name
    new_entity_id = stable_id("ent", "merged", canonical.source, canonical.canonical_name)
    
    # Merge metadata
    merged_meta = dict(canonical.meta)
    merged_meta["merged_from"] = [e.entity_id for e in cluster_entities]
    merged_meta["mention_count"] = sum(entity_mention_counts.values())
    
    return Entity(
        entity_id=new_entity_id,
        entity_type=canonical.entity_type,
        canonical_name=canonical.canonical_name,
        aliases=tuple(sorted(all_aliases)),
        is_fictional=canonical.is_fictional,
        source=canonical.source,
        meta=merged_meta,
    )
