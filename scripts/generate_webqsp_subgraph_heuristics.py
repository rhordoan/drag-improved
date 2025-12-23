#!/usr/bin/env python3
"""
Generate training heuristics from RoG-WebQSP dataset.
Uses the 'graph' field which contains the gold subgraph triples.
This matches D-RAG's approach of using SPARQL-derived heuristics.
"""

import json
from pathlib import Path
from datasets import load_dataset

def main():
    output_path = Path("data/train_heuristics_webqsp_subgraph.jsonl")
    
    print("=" * 60)
    print("Generating WebQSP Heuristics from RoG-WebQSP")
    print("=" * 60)
    
    # Load RoG-WebQSP from HuggingFace
    print("\nLoading RoG-WebQSP dataset...")
    ds = load_dataset('rmanluo/RoG-webqsp', split='train')
    print(f"  Total samples: {len(ds)}")
    
    # Generate heuristics
    heuristics = []
    samples_with_graph = 0
    
    for sample in ds:
        question = sample.get('question', '')
        graph = sample.get('graph', [])
        q_entity = sample.get('q_entity', [])
        a_entity = sample.get('a_entity', [])
        answer = sample.get('answer', [])
        
        if not question or not graph:
            continue
        
        samples_with_graph += 1
        
        # Extract triples from graph (limit to reasonable size)
        triples = []
        for item in graph[:2000]:  # Limit to prevent huge subgraphs
            if len(item) >= 3:
                s, p, o = item[0], item[1], item[2]
                triples.append([s, p, o])
        
        # Create paths from topic entity to answer entities
        paths = []
        for qe in q_entity[:2]:
            for ae in (a_entity if a_entity else answer)[:2]:
                paths.append([qe, ae])
        
        # If no direct path, use first and last entity in graph
        if not paths and triples:
            first_ent = triples[0][0]
            last_ent = triples[-1][2]
            paths.append([first_ent, last_ent])
        
        heuristic = {
            "question": question,
            "paths": paths,
            "answer": answer[0] if answer else "",
            "graph_size": len(triples),
            "triples": triples[:50]  # Limit for memory (same as CWQ)
        }
        heuristics.append(heuristic)
    
    print(f"  Samples with graphs: {samples_with_graph}")
    print(f"  Generated heuristics: {len(heuristics)}")
    
    # Write heuristics
    print(f"\nWriting heuristics to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for h in heuristics:
            f.write(json.dumps(h) + '\n')
    
    # Stats
    print()
    print("=" * 60)
    print(f"✓ Generated {len(heuristics)} heuristics from WebQSP")
    print()
    print("To train Phase 1 on WebQSP:")
    print(f"  python -m src.trainer.train_phase1 \\")
    print(f"      --heuristics_path {output_path} \\")
    print(f"      --epochs 10 --batch_size 16 \\")
    print(f"      --checkpoint_dir checkpoints_webqsp_subgraph")
    print("=" * 60)

if __name__ == "__main__":
    main()

