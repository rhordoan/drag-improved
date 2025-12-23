#!/usr/bin/env python3
"""
Generate synthetic heuristics from FB15k-237 triples.
Since FB15k-237 doesn't have natural language questions, we create simple templates.
"""

import json
import random
from pathlib import Path

def read_triples(kg_path):
    """Read triples from TSV file."""
    triples = []
    with open(kg_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

def generate_question(triple):
    """Generate a question from a triple."""
    subj, pred, obj = triple
    
    # Clean up entity/relation names
    subj_clean = subj.replace('/m/', '').replace('_', ' ')
    obj_clean = obj.replace('/m/', '').replace('_', ' ')
    pred_clean = pred.split('/')[-1].replace('_', ' ')
    
    # Simple templates
    templates = [
        f"What is the {pred_clean} of {subj_clean}?",
        f"Find the {pred_clean} for {subj_clean}",
        f"Which {pred_clean} does {subj_clean} have?",
    ]
    
    return random.choice(templates)

def main():
    kg_path = Path("data/kg/fb15k237.txt")
    output_path = Path("data/train_heuristics_fb15k.jsonl")
    
    print(f"Reading triples from {kg_path}...")
    triples = read_triples(kg_path)
    print(f"  Total triples: {len(triples)}")
    
    # Sample subset for training (use 1000 examples)
    sample_triples = random.sample(triples, min(1000, len(triples)))
    
    print(f"\nGenerating {len(sample_triples)} synthetic questions...")
    
    heuristics = []
    for subj, pred, obj in sample_triples:
        question = generate_question((subj, pred, obj))
        
        # Path is just the single edge from subject to object
        heuristic = {
            "question": question,
            "paths": [[subj, obj]],
            "answer": obj
        }
        heuristics.append(heuristic)
    
    # Write to JSONL
    with open(output_path, 'w') as f:
        for h in heuristics:
            f.write(json.dumps(h) + '\n')
    
    print(f"✅ Generated {len(heuristics)} heuristics")
    print(f"   Output: {output_path}")
    print()
    print("Now train with:")
    print(f"  python -m src.trainer.train_phase1 \\")
    print(f"      --kg_path {kg_path} \\")
    print(f"      --heuristics_path {output_path} \\")
    print(f"      --epochs 10")

if __name__ == "__main__":
    random.seed(42)
    main()

