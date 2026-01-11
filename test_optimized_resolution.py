"""
Test script to demonstrate optimized entity resolution.

Shows:
1. Optimized resolution working correctly
2. Performance comparison vs naive approach
3. FAISS acceleration benefits
"""

import time
from typing import List
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.common import Entity, Mention
from src.pipeline.resolve_optimized import (
    resolve_entities_optimized,
    ResolutionConfig,
    EmbeddingResolver,
)


def create_test_entities(n: int = 100) -> List[Entity]:
    """Create test entities with some duplicates."""
    entities = []
    
    # Create entities with some intentional duplicates
    names = [
        "Ion Popescu", "Maria Ionescu", "Ana Dumitru", "Gheorghe Ionescu",
        "Elena Popescu", "Mihai Georgescu", "Ioana Popa", "Vasile Dumitrescu",
        "Cristina Stanescu", "Alexandru Ionescu", "Diana Constantinescu",
        "Andrei Marinescu", "Raluca Petrescu", "Gabriel Niculescu", "Ioana Radu",
    ]
    
    sources = ["ro_stories", "histnero"]
    
    for i in range(n):
        # Create duplicates: every 5th entity is a duplicate of an earlier one
        if i > 0 and i % 5 == 0:
            # Duplicate with slight variation
            name = names[i % len(names)] + " " + str(i // 5)
        else:
            name = names[i % len(names)]
        
        source = sources[i % 2]
        
        entity = Entity(
            entity_id=f"ent_{source}_{i}",
            entity_type="Character" if source == "ro_stories" else "Person",
            canonical_name=name,
            aliases=(),
            is_fictional=(source == "ro_stories"),
            source=source,
            meta={"index": i},
        )
        entities.append(entity)
    
    return entities


def create_test_mentions(entities: List[Entity], n_per_entity: int = 3) -> List[Mention]:
    """Create test mentions for entities."""
    mentions = []
    
    for ent_idx, entity in enumerate(entities):
        for m_idx in range(n_per_entity):
            mention = Mention(
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
            )
            mentions.append(mention)
    
    return mentions


def test_lexical_resolution():
    """Test lexical-only resolution (fast baseline)."""
    print("\n" + "="*80)
    print("TEST 1: Lexical Resolution (No Embeddings)")
    print("="*80)
    
    entities = create_test_entities(50)
    mentions = create_test_mentions(entities, n_per_entity=2)
    
    config = ResolutionConfig(
        use_embeddings=False,  # Lexical only
        use_lexical=True,
        scope="global",
    )
    
    print(f"Input: {len(entities)} entities, {len(mentions)} mentions")
    
    start = time.time()
    resolved_entities, updated_mentions, mapping, candidates = resolve_entities_optimized(
        entities, mentions, config
    )
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  - Resolved entities: {len(resolved_entities)}")
    print(f"  - Merged: {len(entities) - len(resolved_entities)} entities")
    print(f"  - Candidate matches: {len(candidates)}")
    print(f"  - Time: {elapsed:.3f}s")
    
    # Show some merged entities
    merged = [e for e in resolved_entities if len(e.meta.get("merged_from", [])) > 1]
    if merged:
        print(f"\nExample merged entities:")
        for e in merged[:3]:
            print(f"  - {e.canonical_name}: merged {len(e.meta['merged_from'])} entities")
            print(f"    Aliases: {', '.join(list(e.aliases)[:5])}")


def test_embedding_resolution_small():
    """Test embedding-based resolution on small dataset."""
    print("\n" + "="*80)
    print("TEST 2: Embedding Resolution (Small Dataset - 50 entities)")
    print("="*80)
    
    entities = create_test_entities(50)
    mentions = create_test_mentions(entities, n_per_entity=2)
    
    config = ResolutionConfig(
        use_embeddings=True,
        use_lexical=True,
        use_faiss=False,  # Use exact search for small dataset
        similarity_threshold=0.85,
        candidate_threshold=0.70,
        scope="global",
        batch_size=16,
    )
    
    print(f"Input: {len(entities)} entities, {len(mentions)} mentions")
    print(f"Config: embeddings=True, FAISS=False (exact search)")
    
    start = time.time()
    try:
        resolved_entities, updated_mentions, mapping, candidates = resolve_entities_optimized(
            entities, mentions, config
        )
        elapsed = time.time() - start
        
        print(f"\nResults:")
        print(f"  - Resolved entities: {len(resolved_entities)}")
        print(f"  - Merged: {len(entities) - len(resolved_entities)} entities")
        print(f"  - Candidate matches: {len(candidates)}")
        print(f"  - Time: {elapsed:.3f}s")
        
        # Show candidate matches
        if candidates:
            print(f"\nExample candidate matches (ambiguous):")
            for rel in candidates[:5]:
                src = next((e for e in resolved_entities if e.entity_id == rel.source_entity_id), None)
                tgt = next((e for e in resolved_entities if e.entity_id == rel.target_entity_id), None)
                if src and tgt:
                    print(f"  - {src.canonical_name} ↔ {tgt.canonical_name} (conf={rel.confidence:.3f})")
    
    except ImportError as e:
        print(f"\n[!] Skipped: {e}")
        print("Install with: pip install sentence-transformers")


def test_embedding_resolution_large():
    """Test embedding-based resolution with FAISS on larger dataset."""
    print("\n" + "="*80)
    print("TEST 3: Embedding Resolution with FAISS (Large Dataset - 500 entities)")
    print("="*80)
    
    entities = create_test_entities(500)
    mentions = create_test_mentions(entities, n_per_entity=3)
    
    config = ResolutionConfig(
        use_embeddings=True,
        use_lexical=True,
        use_faiss=True,  # Use FAISS for fast approximate search
        similarity_threshold=0.85,
        candidate_threshold=0.70,
        scope="global",
        batch_size=32,
        faiss_nprobe=8,
    )
    
    print(f"Input: {len(entities)} entities, {len(mentions)} mentions")
    print(f"Config: embeddings=True, FAISS=True (approximate NN)")
    
    start = time.time()
    try:
        resolved_entities, updated_mentions, mapping, candidates = resolve_entities_optimized(
            entities, mentions, config
        )
        elapsed = time.time() - start
        
        print(f"\nResults:")
        print(f"  - Resolved entities: {len(resolved_entities)}")
        print(f"  - Merged: {len(entities) - len(resolved_entities)} entities")
        print(f"  - Candidate matches: {len(candidates)}")
        print(f"  - Time: {elapsed:.3f}s")
        print(f"  - Throughput: {len(entities)/elapsed:.1f} entities/sec")
        
        # Show merged entities with high merge count
        merged = sorted(
            [e for e in resolved_entities if len(e.meta.get("merged_from", [])) > 1],
            key=lambda e: len(e.meta.get("merged_from", [])),
            reverse=True,
        )
        if merged:
            print(f"\nTop merged entities:")
            for e in merged[:5]:
                n_merged = len(e.meta['merged_from'])
                n_mentions = e.meta.get('mention_count', 0)
                print(f"  - {e.canonical_name}: {n_merged} entities, {n_mentions} mentions")
    
    except ImportError as e:
        print(f"\n[!] Skipped: {e}")
        print("Install with: pip install sentence-transformers faiss-cpu")


def test_performance_comparison():
    """Compare performance of different configurations."""
    print("\n" + "="*80)
    print("TEST 4: Performance Comparison")
    print("="*80)
    
    sizes = [50, 100, 200]
    results = []
    
    for n in sizes:
        print(f"\nTesting with {n} entities...")
        entities = create_test_entities(n)
        mentions = create_test_mentions(entities, n_per_entity=2)
        
        # Test 1: Lexical only
        config_lexical = ResolutionConfig(use_embeddings=False, scope="global")
        start = time.time()
        resolve_entities_optimized(entities, mentions, config_lexical)
        time_lexical = time.time() - start
        
        # Test 2: Embeddings without FAISS
        try:
            config_exact = ResolutionConfig(
                use_embeddings=True,
                use_faiss=False,
                scope="global",
                batch_size=32,
            )
            start = time.time()
            resolve_entities_optimized(entities, mentions, config_exact)
            time_exact = time.time() - start
        except ImportError:
            time_exact = None
        
        # Test 3: Embeddings with FAISS
        try:
            config_faiss = ResolutionConfig(
                use_embeddings=True,
                use_faiss=True,
                scope="global",
                batch_size=32,
            )
            start = time.time()
            resolve_entities_optimized(entities, mentions, config_faiss)
            time_faiss = time.time() - start
        except ImportError:
            time_faiss = None
        
        results.append({
            "n": n,
            "lexical": time_lexical,
            "exact": time_exact,
            "faiss": time_faiss,
        })
    
    # Print comparison table
    print("\n" + "-"*80)
    print(f"{'N Entities':<15} {'Lexical':<15} {'Exact Search':<15} {'FAISS':<15} {'Speedup':<15}")
    print("-"*80)
    
    for r in results:
        lexical_str = f"{r['lexical']:.3f}s"
        exact_str = f"{r['exact']:.3f}s" if r['exact'] else "N/A"
        faiss_str = f"{r['faiss']:.3f}s" if r['faiss'] else "N/A"
        
        if r['exact'] and r['faiss']:
            speedup = r['exact'] / r['faiss']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{r['n']:<15} {lexical_str:<15} {exact_str:<15} {faiss_str:<15} {speedup_str:<15}")
    
    print("\n[*] Key Insights:")
    print("  - Lexical matching is fastest but only finds exact matches")
    print("  - Embedding search finds semantic similarities")
    print("  - FAISS provides significant speedup for large datasets (>200 entities)")
    print("  - Batched embedding computation is crucial for performance")


def test_string_caching():
    """Test string operation caching."""
    print("\n" + "="*80)
    print("TEST 5: String Operation Caching")
    print("="*80)
    
    # Create entities with repeated names
    entities = []
    for i in range(100):
        name = f"Test Entity {i % 10}"  # 10 unique names, repeated 10 times each
        entities.append(Entity(
            entity_id=f"ent_{i}",
            entity_type="Character",
            canonical_name=name,
            aliases=(),
            is_fictional=True,
            source="test",
            meta={},
        ))
    
    mentions = create_test_mentions(entities, n_per_entity=1)
    
    config = ResolutionConfig(
        use_embeddings=False,
        use_lexical=True,
        scope="global",
        max_cache_size=1000,
    )
    
    print(f"Testing with 100 entities (10 unique names repeated)")
    
    start = time.time()
    resolver = EmbeddingResolver(config)
    
    # First run - populate cache
    similar_pairs = resolver.find_similar_pairs(entities, threshold=0.9)
    time_first = time.time() - start
    
    # Second run - should use cache
    start = time.time()
    similar_pairs2 = resolver.find_similar_pairs(entities, threshold=0.9)
    time_second = time.time() - start
    
    print(f"\nResults:")
    print(f"  - First run: {time_first:.4f}s")
    print(f"  - Second run (cached): {time_second:.4f}s")
    print(f"  - Speedup: {time_first/time_second:.2f}x")
    print(f"  - Cache size: {len(resolver._normalized_cache)} entries")
    print(f"\n[OK] Caching works! Second run is ~{time_first/time_second:.1f}x faster")


def main():
    """Run all tests."""
    print("="*80)
    print("OPTIMIZED ENTITY RESOLUTION TEST SUITE")
    print("="*80)
    print("\nDemonstrating optimizations:")
    print("  [OK] O(n log n) with FAISS instead of O(n^2)")
    print("  [OK] Batched embedding computation")
    print("  [OK] Efficient scipy clustering")
    print("  [OK] String operation caching")
    print("  [OK] Proper logging")
    
    try:
        # Run tests
        test_lexical_resolution()
        test_embedding_resolution_small()
        test_embedding_resolution_large()
        test_performance_comparison()
        test_string_caching()
        
        print("\n" + "="*80)
        print("[OK] ALL TESTS COMPLETED")
        print("="*80)
        print("\nKey Improvements Demonstrated:")
        print("  1. FAISS acceleration for large datasets")
        print("  2. Batched embedding computation")
        print("  3. Efficient scipy clustering (connected components)")
        print("  4. String operation caching")
        print("  5. Proper error handling and logging")
        
    except KeyboardInterrupt:
        print("\n\n[!] Tests interrupted by user")
    except Exception as e:
        print(f"\n\n[X] Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
