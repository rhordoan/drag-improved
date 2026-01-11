"""
Optimized entity resolution using approximate nearest neighbors and efficient clustering.

Improvements over resolve_advanced.py:
- Uses FAISS for approximate nearest neighbor search (O(n log n) instead of O(n²))
- Batched embedding computation
- Efficient scipy clustering instead of naive implementation
- String operation caching
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .common import Entity, Mention, Relation, normalize_mention, stable_id

logger = logging.getLogger(__name__)


@dataclass
class ResolutionConfig:
    """Configuration for optimized entity resolution."""
    use_embeddings: bool = True
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.85
    candidate_threshold: float = 0.70
    
    use_lexical: bool = True
    lexical_threshold: float = 0.9
    
    scope: str = "per_source"
    min_norm_len: int = 2
    
    # Optimization parameters
    use_faiss: bool = True  # Use approximate nearest neighbors
    faiss_nprobe: int = 8  # FAISS search parameter
    batch_size: int = 32  # Batch size for embedding computation
    max_cache_size: int = 10000  # Max items in string cache


class EmbeddingResolver:
    """Optimized entity resolution using semantic embeddings with FAISS."""
    
    def __init__(self, config: ResolutionConfig):
        self.config = config
        self._model = None
        self._embedding_cache: Dict[str, 'np.ndarray'] = {}
        self._normalized_cache: Dict[str, str] = {}  # Cache normalized strings
        self._faiss_index = None
        
        logger.info(f"Initializing EmbeddingResolver with FAISS={config.use_faiss}")
    
    def _get_model(self):
        """Lazy load embedding model with error handling."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.config.embedding_model}")
                self._model = SentenceTransformer(self.config.embedding_model)
                logger.info("Model loaded successfully")
            except ImportError as e:
                logger.error("sentence-transformers not installed", exc_info=True)
                logger.warning("Falling back to lexical matching only")
                self.config.use_embeddings = False
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                logger.warning("Falling back to lexical matching only")
                self.config.use_embeddings = False
        return self._model
    
    def get_embeddings_batch(self, texts: List[str]) -> Optional['np.ndarray']:
        """
        Get embeddings for multiple texts efficiently using batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings (n_texts, embedding_dim) or None if failed
        """
        if not self.config.use_embeddings or not texts:
            return None
        
        model = self._get_model()
        if model is None:
            return None
        
        try:
            import numpy as np
            
            # Check cache first
            uncached_indices = []
            uncached_texts = []
            embeddings = [None] * len(texts)
            
            for i, text in enumerate(texts):
                if text in self._embedding_cache:
                    embeddings[i] = self._embedding_cache[text]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
            
            # Compute uncached embeddings in batches
            if uncached_texts:
                logger.debug(f"Computing embeddings for {len(uncached_texts)} texts")
                new_embeddings = model.encode(
                    uncached_texts,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                
                # Cache new embeddings (with size limit)
                for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                    embeddings[idx] = emb
                    if len(self._embedding_cache) < self.config.max_cache_size:
                        self._embedding_cache[text] = emb
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}", exc_info=True)
            return None
    
    def build_faiss_index(self, embeddings: 'np.ndarray') -> Optional['faiss.Index']:
        """
        Build FAISS index for approximate nearest neighbor search.
        
        Args:
            embeddings: Array of embeddings (n_entities, embedding_dim)
            
        Returns:
            FAISS index or None if failed
        """
        if not self.config.use_faiss:
            return None
        
        try:
            import faiss
            import numpy as np
            
            n, dim = embeddings.shape
            logger.info(f"Building FAISS index for {n} entities with dim={dim}")
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Use IVF index for large datasets (>1000 entities)
            if n > 1000:
                nlist = min(int(np.sqrt(n)), 100)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dim)  # Inner product (for normalized = cosine)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings)
                index.add(embeddings)
                index.nprobe = self.config.faiss_nprobe
                logger.info(f"Built IVF index with {nlist} clusters")
            else:
                # For small datasets, use flat index
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                logger.info("Built flat index")
            
            return index
            
        except ImportError:
            logger.warning("faiss-cpu not installed, falling back to exact search")
            return None
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}", exc_info=True)
            return None
    
    def find_similar_pairs(
        self,
        entities: List[Entity],
        threshold: float,
    ) -> List[Tuple[int, int, float]]:
        """
        Find similar entity pairs efficiently using FAISS or exact search.
        
        Args:
            entities: List of entities to compare
            threshold: Similarity threshold
            
        Returns:
            List of (index1, index2, similarity) tuples
        """
        import numpy as np
        
        # Get all entity names
        entity_names = [e.canonical_name for e in entities]
        
        # Compute embeddings in batch
        embeddings = self.get_embeddings_batch(entity_names)
        if embeddings is None:
            logger.warning("Embeddings failed, using lexical matching only")
            return self._find_similar_pairs_lexical(entities, threshold)
        
        similar_pairs = []
        
        # Try FAISS for approximate search
        if self.config.use_faiss:
            index = self.build_faiss_index(embeddings.astype('float32'))
            if index is not None:
                # Search for k nearest neighbors for each entity
                k = min(20, len(entities))  # Look at top 20 similar entities
                
                logger.info(f"Searching for similar pairs with threshold={threshold}")
                distances, indices = index.search(embeddings.astype('float32'), k)
                
                # Extract pairs above threshold
                for i in range(len(entities)):
                    for j_idx in range(k):
                        j = indices[i, j_idx]
                        sim = float(distances[i, j_idx])
                        
                        # Avoid duplicates and self-pairs
                        if i < j and sim >= threshold:
                            similar_pairs.append((i, j, sim))
                
                logger.info(f"Found {len(similar_pairs)} similar pairs using FAISS")
                return similar_pairs
        
        # Fallback to exact search (still optimized with numpy)
        logger.info("Using exact similarity computation")
        return self._find_similar_pairs_exact(embeddings, threshold)
    
    def _find_similar_pairs_exact(
        self,
        embeddings: 'np.ndarray',
        threshold: float,
    ) -> List[Tuple[int, int, float]]:
        """Exact pairwise similarity computation using vectorized operations."""
        import numpy as np
        
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Compute all pairwise similarities at once
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Extract upper triangle (avoid duplicates)
        n = len(embeddings)
        similar_pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_matrix[i, j]
                if sim >= threshold:
                    similar_pairs.append((i, j, float(sim)))
        
        logger.info(f"Found {len(similar_pairs)} similar pairs using exact search")
        return similar_pairs
    
    def _find_similar_pairs_lexical(
        self,
        entities: List[Entity],
        threshold: float,
    ) -> List[Tuple[int, int, float]]:
        """Fallback lexical matching."""
        similar_pairs = []
        
        # Cache normalized names
        normalized = []
        for e in entities:
            if e.canonical_name not in self._normalized_cache:
                if len(self._normalized_cache) < self.config.max_cache_size:
                    self._normalized_cache[e.canonical_name] = normalize_mention(e.canonical_name)
            normalized.append(self._normalized_cache.get(e.canonical_name, normalize_mention(e.canonical_name)))
        
        # Only compare identical normalized strings
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if normalized[i] == normalized[j]:
                    similar_pairs.append((i, j, 1.0))
        
        return similar_pairs


def resolve_entities_optimized(
    entities: List[Entity],
    mentions: List[Mention],
    config: ResolutionConfig,
) -> Tuple[List[Entity], List[Mention], Dict[str, str], List[Relation]]:
    """
    Optimized entity resolution with efficient algorithms.
    
    Improvements:
    - O(n log n) approximate nearest neighbor search with FAISS
    - Batched embedding computation
    - Efficient scipy clustering
    - String operation caching
    
    Returns:
        - Resolved entities
        - Updated mentions with new entity_id
        - Mapping from old entity_id to new entity_id
        - CANDIDATE_SAME_AS relations for ambiguous matches
    """
    if not entities:
        return [], mentions, {}, []
    
    logger.info(f"Starting optimized resolution for {len(entities)} entities")
    
    resolver = EmbeddingResolver(config)
    
    # Group entities by scope
    entity_groups: Dict[str, List[Entity]] = {}
    for e in entities:
        if config.scope == "per_source":
            key = e.source
        elif config.scope == "per_doc":
            doc_mentions = [m for m in mentions if m.entity_id == e.entity_id]
            key = doc_mentions[0].doc_id if doc_mentions else "unknown"
        else:  # global
            key = "global"
        
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(e)
    
    logger.info(f"Grouped into {len(entity_groups)} scopes")
    
    # Resolve within each group
    resolved_entities: List[Entity] = []
    old_to_new: Dict[str, str] = {}
    candidate_relations: List[Relation] = []
    
    for group_key, group_entities in entity_groups.items():
        logger.info(f"Processing group '{group_key}' with {len(group_entities)} entities")
        
        group_resolved, group_mapping, group_candidates = _resolve_entity_group_optimized(
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
    
    logger.info(
        f"Resolution complete: {len(resolved_entities)} entities, "
        f"{len(candidate_relations)} candidate matches"
    )
    
    return resolved_entities, updated_mentions, old_to_new, candidate_relations


def _resolve_entity_group_optimized(
    entities: List[Entity],
    mentions: List[Mention],
    config: ResolutionConfig,
    resolver: EmbeddingResolver,
) -> Tuple[List[Entity], Dict[str, str], List[Relation]]:
    """Optimized resolution using efficient clustering."""
    if len(entities) <= 1:
        return entities, {}, []
    
    # Find similar pairs efficiently (O(n log n) with FAISS or O(n²) with numpy)
    similar_pairs = resolver.find_similar_pairs(entities, config.similarity_threshold)
    
    # Also find candidate pairs
    candidate_pairs = resolver.find_similar_pairs(entities, config.candidate_threshold)
    
    # Filter candidate pairs to only those below similarity threshold
    similar_set = {(i, j) for i, j, _ in similar_pairs}
    candidate_pairs = [(i, j, sim) for i, j, sim in candidate_pairs if (i, j) not in similar_set]
    
    # Use scipy for efficient clustering
    clusters = _cluster_entities_scipy(len(entities), similar_pairs)
    
    # Create CANDIDATE_SAME_AS relations
    candidate_relations = _create_candidate_relations(entities, candidate_pairs)
    
    # Merge entities in each cluster
    resolved_entities: List[Entity] = []
    old_to_new: Dict[str, str] = {}
    
    for cluster in clusters:
        if len(cluster) == 1:
            e = entities[cluster[0]]
            resolved_entities.append(e)
            old_to_new[e.entity_id] = e.entity_id
        else:
            cluster_entities = [entities[i] for i in cluster]
            merged_entity = _merge_entities(cluster_entities, mentions)
            resolved_entities.append(merged_entity)
            
            for e in cluster_entities:
                old_to_new[e.entity_id] = merged_entity.entity_id
    
    logger.debug(f"Merged {len(entities)} entities into {len(resolved_entities)}")
    
    return resolved_entities, old_to_new, candidate_relations


def _cluster_entities_scipy(
    n_entities: int,
    similar_pairs: List[Tuple[int, int, float]],
) -> List[List[int]]:
    """
    Efficient clustering using scipy's connected components.
    
    Much faster than naive agglomerative clustering.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        # Build adjacency matrix as sparse matrix
        if not similar_pairs:
            # No similar pairs, each entity is its own cluster
            return [[i] for i in range(n_entities)]
        
        rows, cols, data = [], [], []
        for i, j, sim in similar_pairs:
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([sim, sim])
        
        # Create sparse adjacency matrix
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_entities, n_entities))
        
        # Find connected components
        n_components, labels = connected_components(adj_matrix, directed=False)
        
        # Group entities by component
        clusters = [[] for _ in range(n_components)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        logger.info(f"Clustered {n_entities} entities into {n_components} components")
        return clusters
        
    except ImportError:
        logger.warning("scipy not installed, falling back to simple clustering")
        return _cluster_entities_simple(n_entities, similar_pairs)


def _cluster_entities_simple(
    n_entities: int,
    similar_pairs: List[Tuple[int, int, float]],
) -> List[List[int]]:
    """Simple union-find clustering as fallback."""
    # Initialize each entity as its own cluster
    parent = list(range(n_entities))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union similar entities
    for i, j, _ in similar_pairs:
        union(i, j)
    
    # Group by root
    clusters_dict = {}
    for i in range(n_entities):
        root = find(i)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(i)
    
    return list(clusters_dict.values())


def _create_candidate_relations(
    entities: List[Entity],
    candidate_pairs: List[Tuple[int, int, float]],
) -> List[Relation]:
    """Create CANDIDATE_SAME_AS relations for ambiguous matches."""
    candidate_relations = []
    
    for i, j, sim in candidate_pairs:
        e1, e2 = entities[i], entities[j]
        
        # Bidirectional relations
        rel1 = Relation(
            source_entity_id=e1.entity_id,
            predicate="CANDIDATE_SAME_AS",
            target_entity_id=e2.entity_id,
            doc_id="",
            chunk_id="",
            source="resolution",
            confidence=float(sim),
            evidence_text=f"Similarity: {sim:.3f}",
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
            evidence_text=f"Similarity: {sim:.3f}",
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
    
    canonical = max(cluster_entities, key=lambda e: entity_mention_counts.get(e.entity_id, 0))
    
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
