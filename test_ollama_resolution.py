"""
Test optimized resolution with Ollama embeddings.
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.common import Entity, Mention, normalize_mention, stable_id
from src.pipeline.resolve_ollama import OllamaEmbeddingResolver, OllamaConfig
from src.pipeline.resolve_optimized import (
    _cluster_entities_scipy,
    _create_candidate_relations,
    _merge_entities,
)


def create_test_entities(n=50):
    """Create test entities."""
    names = [
        "Ion Popescu", "Maria Ionescu", "Ana Dumitru", "Gheorghe Ionescu",
        "Elena Popescu", "Mihai Georgescu", "Ioana Popa", "Vasile Dumitrescu",
        "Cristina Stanescu", "Alexandru Ionescu", "Diana Constantinescu",
        "Andrei Marinescu", "Raluca Petrescu", "Gabriel Niculescu", "Ioana Radu",
    ]
    
    entities = []
    for i in range(n):
        # Create some duplicates
        name = names[i % len(names)]
        if i > 0 and i % 7 == 0:
            name = name + " Jr"  # Slight variation
        
        source = "ro_stories" if i % 2 == 0 else "histnero"
        
        entities.append(Entity(
            entity_id=f"ent_{source}_{i}",
            entity_type="Character" if source == "ro_stories" else "Person",
            canonical_name=name,
            aliases=(),
            is_fictional=(source == "ro_stories"),
            source=source,
            meta={"index": i},
        ))
    
    return entities


def create_test_mentions(entities, n_per_entity=2):
    """Create test mentions."""
    mentions = []
    for ent_idx, entity in enumerate(entities):
        for m_idx in range(n_per_entity):
            mentions.append(Mention(
                mention_id=f"m_{entity.entity_id}_{m_idx}",
                entity_id=entity.entity_id,
                surface=entity.canonical_name,
                start_char=m_idx * 100,
                end_char=m_idx * 100 + len(entity.canonical_name),
                doc_id=f"doc_{ent_idx // 10}",
                chunk_id=f"chunk_{ent_idx}_{m_idx}",
                source=entity.source,
                entity_type=entity.entity_type,
                confidence=0.8,
            ))
    return mentions


def test_ollama_connection():
    """Test connection to Ollama server."""
    print("=" * 80)
    print("TEST 1: Ollama Connection")
    print("=" * 80)
    
    config = OllamaConfig(
        base_url="http://inference.ccrolabs.com",
        model="nomic-embed-text",
    )
    
    try:
        resolver = OllamaEmbeddingResolver(config)
        print(f"[OK] Connected to {config.base_url}")
        print(f"[OK] Using model: {config.model}")
        
        # Test single embedding
        print("\nTesting single embedding...")
        text = "Test text for embedding"
        start = time.time()
        emb = resolver.get_embedding(text)
        elapsed = time.time() - start
        
        if emb:
            print(f"[OK] Got embedding: dimension={len(emb)}, time={elapsed:.3f}s")
            print(f"     First 5 values: {emb[:5]}")
            return resolver
        else:
            print("[X] Failed to get embedding")
            return None
            
    except Exception as e:
        print(f"[X] Connection failed: {e}")
        return None


def test_batch_embeddings(resolver):
    """Test batch embedding computation."""
    print("\n" + "=" * 80)
    print("TEST 2: Batch Embeddings")
    print("=" * 80)
    
    texts = [
        "Ion Popescu",
        "Maria Ionescu", 
        "Ana Dumitru",
        "Elena Popescu",
        "Ion Popescu Jr",  # Similar to first
    ]
    
    print(f"Computing embeddings for {len(texts)} texts...")
    start = time.time()
    embeddings = resolver.get_embeddings_batch(texts, batch_size=5)
    elapsed = time.time() - start
    
    if embeddings and len(embeddings) == len(texts):
        print(f"[OK] Got {len(embeddings)} embeddings in {elapsed:.3f}s")
        print(f"     Avg time per text: {elapsed/len(texts):.3f}s")
        
        # Test similarity
        print("\nSimilarity between texts:")
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = resolver.compute_similarity(texts[i], texts[j])
                print(f"  {texts[i]:<20} <-> {texts[j]:<20} : {sim:.3f}")
    else:
        print("[X] Failed to get batch embeddings")


def test_entity_resolution(resolver):
    """Test full entity resolution with Ollama."""
    print("\n" + "=" * 80)
    print("TEST 3: Entity Resolution with Ollama")
    print("=" * 80)
    
    # Create test data
    entities = create_test_entities(n=30)
    mentions = create_test_mentions(entities, n_per_entity=2)
    
    print(f"Input: {len(entities)} entities, {len(mentions)} mentions")
    print(f"Unique names: {len(set(e.canonical_name for e in entities))}")
    
    # Find similar pairs
    print("\nFinding similar entity pairs...")
    similarity_threshold = 0.85
    candidate_threshold = 0.70
    
    start = time.time()
    similar_pairs = resolver.find_similar_pairs(entities, similarity_threshold)
    elapsed_similar = time.time() - start
    
    print(f"[OK] Found {len(similar_pairs)} similar pairs in {elapsed_similar:.2f}s")
    
    # Find candidate pairs
    candidate_pairs = resolver.find_similar_pairs(entities, candidate_threshold)
    similar_set = {(i, j) for i, j, _ in similar_pairs}
    candidate_pairs = [(i, j, sim) for i, j, sim in candidate_pairs if (i, j) not in similar_set]
    
    print(f"[OK] Found {len(candidate_pairs)} candidate pairs")
    
    # Cluster entities
    print("\nClustering entities...")
    start = time.time()
    clusters = _cluster_entities_scipy(len(entities), similar_pairs)
    elapsed_cluster = time.time() - start
    
    print(f"[OK] Formed {len(clusters)} clusters in {elapsed_cluster:.3f}s")
    
    # Show results
    print("\nCluster details:")
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            names = [entities[idx].canonical_name for idx in cluster]
            print(f"  Cluster {i+1}: {len(cluster)} entities")
            print(f"    Names: {', '.join(names)}")
    
    # Create merged entities
    print("\nMerging entities...")
    merged_entities = []
    old_to_new = {}
    
    for cluster in clusters:
        if len(cluster) == 1:
            e = entities[cluster[0]]
            merged_entities.append(e)
            old_to_new[e.entity_id] = e.entity_id
        else:
            cluster_entities = [entities[i] for i in cluster]
            merged = _merge_entities(cluster_entities, mentions)
            merged_entities.append(merged)
            for e in cluster_entities:
                old_to_new[e.entity_id] = merged.entity_id
    
    print(f"[OK] Reduced from {len(entities)} to {len(merged_entities)} entities")
    print(f"     Merged: {len(entities) - len(merged_entities)} entities")
    
    # Show top merged entities
    merged_with_count = [e for e in merged_entities if len(e.meta.get("merged_from", [])) > 1]
    if merged_with_count:
        print("\nTop merged entities:")
        for e in sorted(merged_with_count, key=lambda x: len(x.meta.get("merged_from", [])), reverse=True)[:5]:
            n_merged = len(e.meta["merged_from"])
            print(f"  - {e.canonical_name}: merged {n_merged} entities")
            if e.aliases:
                print(f"    Aliases: {', '.join(list(e.aliases)[:5])}")
    
    # Show candidate matches
    candidate_relations = _create_candidate_relations(entities, candidate_pairs)
    if candidate_relations:
        print(f"\nCandidate matches (ambiguous): {len(candidate_relations)}")
        for rel in candidate_relations[:5]:
            src = next((e for e in entities if e.entity_id == rel.source_entity_id), None)
            tgt = next((e for e in entities if e.entity_id == rel.target_entity_id), None)
            if src and tgt:
                print(f"  - {src.canonical_name} <-> {tgt.canonical_name} (conf={rel.confidence:.3f})")
    
    # Total stats
    print("\n" + "-" * 80)
    print("SUMMARY:")
    print(f"  Total time: {elapsed_similar + elapsed_cluster:.2f}s")
    print(f"  Embeddings: {elapsed_similar:.2f}s")
    print(f"  Clustering: {elapsed_cluster:.3f}s")
    print(f"  Entities reduced: {len(entities)} -> {len(merged_entities)}")
    print(f"  Cache size: {len(resolver._embedding_cache)} embeddings")


def test_performance_scaling(resolver):
    """Test performance with different sizes."""
    print("\n" + "=" * 80)
    print("TEST 4: Performance Scaling")
    print("=" * 80)
    
    sizes = [20, 50, 100]
    results = []
    
    for n in sizes:
        print(f"\nTesting with {n} entities...")
        entities = create_test_entities(n)
        
        start = time.time()
        similar_pairs = resolver.find_similar_pairs(entities, threshold=0.85)
        elapsed = time.time() - start
        
        throughput = n / elapsed
        results.append((n, elapsed, throughput))
        print(f"  Time: {elapsed:.2f}s, Throughput: {throughput:.1f} entities/sec")
    
    print("\n" + "-" * 80)
    print(f"{'N Entities':<15} {'Time (s)':<15} {'Throughput':<20}")
    print("-" * 80)
    for n, elapsed, throughput in results:
        print(f"{n:<15} {elapsed:<15.2f} {throughput:<20.1f}")


def main():
    """Run all tests."""
    print("=" * 80)
    print("OPTIMIZED RESOLUTION WITH OLLAMA EMBEDDINGS")
    print("=" * 80)
    print("\nUsing Ollama at: http://inference.ccrolabs.com")
    print("Model: nomic-embed-text")
    
    try:
        # Test connection
        resolver = test_ollama_connection()
        if not resolver:
            print("\n[X] Cannot connect to Ollama server")
            print("Make sure Ollama is running at http://inference.ccrolabs.com")
            return
        
        # Run tests
        test_batch_embeddings(resolver)
        test_entity_resolution(resolver)
        test_performance_scaling(resolver)
        
        print("\n" + "=" * 80)
        print("[OK] ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nKey Features Demonstrated:")
        print("  1. Ollama API integration for embeddings")
        print("  2. Batch processing with rate limiting")
        print("  3. Embedding caching for performance")
        print("  4. Efficient scipy clustering")
        print("  5. CANDIDATE_SAME_AS for ambiguous matches")
        
    except KeyboardInterrupt:
        print("\n\n[!] Tests interrupted by user")
    except Exception as e:
        print(f"\n\n[X] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
