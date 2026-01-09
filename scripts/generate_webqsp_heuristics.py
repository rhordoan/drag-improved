#!/usr/bin/env python3
"""
Generate training heuristics from WebQSP dataset.
Creates paths based on topic entity -> answer entity pairs.
"""

import json
from pathlib import Path
import re

def extract_entity_from_url(url):
    """Extract entity name from Freebase URL."""
    # http://www.freebase.com/view/en/justin_bieber -> justin_bieber
    match = re.search(r'/view/[^/]+/(.+)$', url)
    if match:
        return match.group(1)
    return None

def normalize_entity(entity):
    """Normalize entity string for matching."""
    # Convert to lowercase, replace spaces with underscores
    return entity.lower().replace(' ', '_').replace('-', '_')

def main():
    webqsp_train = Path("data/webqsp/WebQSP.train.json")
    output_path = Path("data/train_heuristics_webqsp.jsonl")
    
    print("=" * 60)
    print("Generating WebQSP Heuristics")
    print("=" * 60)
    
    # Read WebQSP training data
    print(f"\nReading {webqsp_train}...")
    samples = []
    with open(webqsp_train) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"  Total samples: {len(samples)}")
    
    # Generate heuristics
    heuristics = []
    skipped = 0
    
    for sample in samples:
        question = sample.get('question', '')
        answers = sample.get('answers', [])
        url = sample.get('url', '')
        
        if not question or not answers:
            skipped += 1
            continue
        
        # Extract topic entity
        topic_entity = extract_entity_from_url(url)
        if not topic_entity:
            skipped += 1
            continue
        
        # Create paths from topic entity to each answer
        paths = []
        for answer in answers[:3]:  # Limit to first 3 answers
            answer_entity = normalize_entity(answer)
            paths.append([topic_entity, answer_entity])
        
        heuristic = {
            "question": question,
            "paths": paths,
            "answer": answers[0] if answers else "",
            "topic_entity": topic_entity
        }
        heuristics.append(heuristic)
    
    print(f"  Generated: {len(heuristics)} heuristics")
    print(f"  Skipped: {skipped} (missing data)")
    
    # Write heuristics
    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        for h in heuristics:
            f.write(json.dumps(h) + '\n')
    
    print(f"✓ Done!")
    print()
    print("=" * 60)
    print(f"Generated {len(heuristics)} heuristics from WebQSP")
    print()
    print("To train:")
    print(f"  python -m src.trainer.train_phase1 \\")
    print(f"      --heuristics_path {output_path} \\")
    print(f"      --epochs 10 --batch_size 16")
    print("=" * 60)

if __name__ == "__main__":
    main()

