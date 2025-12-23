#!/usr/bin/env python3
"""
Generate training heuristics from ComplexWebQuestions dataset.
Uses the 'graph' field which contains the gold subgraph triples.
This matches D-RAG's approach of using SPARQL-derived heuristics.
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    cwq_train = Path("data/cwq/ComplexWebQuestions_train.json")
    output_path = Path("data/train_heuristics_cwq.jsonl")
    kg_output = Path("data/kg/cwq_subgraphs.txt")
    
    print("=" * 60)
    print("Generating CWQ Heuristics from Gold Subgraphs")
    print("=" * 60)
    
    # Read CWQ training data
    print(f"\nReading {cwq_train}...")
    samples = []
    with open(cwq_train) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"  Total samples: {len(samples)}")
    
    # Collect all unique triples for KG construction
    all_triples = set()
    all_entities = set()
    all_relations = set()
    
    # Generate heuristics
    heuristics = []
    samples_with_graph = 0
    
    for sample in samples:
        question = sample.get('question', '')
        graph = sample.get('graph', [])
        q_entity = sample.get('q_entity', [])
        a_entity = sample.get('a_entity', [])
        answer = sample.get('answer', [])
        
        if not question or not graph:
            continue
        
        samples_with_graph += 1
        
        # Extract triples from graph
        triples = []
        for item in graph:
            if len(item) >= 3:
                s, p, o = item[0], item[1], item[2]
                triples.append((s, p, o))
                all_triples.add((s, p, o))
                all_entities.add(s)
                all_entities.add(o)
                all_relations.add(p)
        
        # Create paths from topic entity to answer entities
        # The graph itself represents the reasoning path
        paths = []
        
        # Use q_entity -> a_entity as main path
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
            "triples": [[s, p, o] for s, p, o in triples[:50]]  # Limit for memory
        }
        heuristics.append(heuristic)
    
    print(f"  Samples with graphs: {samples_with_graph}")
    print(f"  Generated heuristics: {len(heuristics)}")
    print(f"  Unique entities: {len(all_entities)}")
    print(f"  Unique relations: {len(all_relations)}")
    print(f"  Total triples: {len(all_triples)}")
    
    # Write CWQ subgraph as KG
    print(f"\nWriting KG to {kg_output}...")
    kg_output.parent.mkdir(parents=True, exist_ok=True)
    with open(kg_output, 'w') as f:
        for s, p, o in sorted(all_triples):
            # Clean up entity/relation names
            s_clean = s.replace('\t', ' ').strip()
            p_clean = p.replace('\t', ' ').strip()
            o_clean = o.replace('\t', ' ').strip()
            if s_clean and p_clean and o_clean:
                f.write(f"{s_clean}\t{p_clean}\t{o_clean}\n")
    
    # Write heuristics
    print(f"Writing heuristics to {output_path}...")
    with open(output_path, 'w') as f:
        for h in heuristics:
            f.write(json.dumps(h) + '\n')
    
    # Stats
    print()
    print("=" * 60)
    print(f"✓ Generated {len(heuristics)} heuristics from CWQ")
    print(f"✓ Created KG with {len(all_triples)} triples")
    print()
    print("To train Phase 1:")
    print(f"  python -m src.trainer.train_phase1 \\")
    print(f"      --kg_path {kg_output} \\")
    print(f"      --heuristics_path {output_path} \\")
    print(f"      --epochs 10 --batch_size 16")
    print("=" * 60)

if __name__ == "__main__":
    main()

