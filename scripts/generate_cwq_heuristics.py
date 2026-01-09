#!/usr/bin/env python3
"""
Generate D-RAG-style "heuristics" JSONL from the RoG-CWQ dataset.

Important:
- CWQ provides official splits: train / dev / test.
- For *proper* validation (to detect overfitting), generate heuristics from the
  official DEV file and pass it to Phase 2 as --val_heuristics_path.

This script produces JSONL where each line contains (at minimum):
  - question: str
  - triples:  List[[head, relation, tail]]
  - paths:    List[List[entity]]  (used to derive 0/1 labels over triples)
  - answer:   str | List[str]
"""

import argparse
import json
from pathlib import Path


def generate_heuristics(input_path: Path, output_path: Path, limit_triples: int = 50):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    print("=" * 60)
    print("Generating CWQ Heuristics from Gold Subgraphs")
    print("=" * 60)
    
    # Read CWQ training data
    print(f"\nReading {input_path}...")
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"  Total samples: {len(samples)}")
    
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
        
        # Preserve answer format (list or string). Phase 2 supports both.
        answer_out = answer
        if isinstance(answer, list):
            answer_out = [str(a) for a in answer if str(a).strip()]
            if len(answer_out) == 1:
                # keep as string for compactness
                answer_out = answer_out[0]
        elif answer is None:
            answer_out = ""

        heuristic = {
            "question": question,
            "paths": paths,
            "answer": answer_out,
            "graph_size": len(triples),
            # Keep per-question subgraph triples (cap for memory/speed)
            "triples": [[s, p, o] for s, p, o in triples[: max(0, int(limit_triples))]],
        }
        heuristics.append(heuristic)
    
    print(f"  Samples with graphs: {samples_with_graph}")
    print(f"  Generated heuristics: {len(heuristics)}")
    
    # Write heuristics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing heuristics to {output_path}...")
    with open(output_path, 'w', encoding="utf-8") as f:
        for h in heuristics:
            f.write(json.dumps(h, ensure_ascii=False) + '\n')
    
    # Stats
    print()
    print("=" * 60)
    print(f"✓ Generated {len(heuristics)} heuristics from CWQ file: {input_path.name}")
    print()
    print("To train Phase 1 (retriever warmup):")
    print("  python -m src.trainer.train_phase1 \\")
    print(f"      --heuristics_path {output_path.as_posix()} \\")
    print("      --epochs 10 --batch_size 16 --lr 5e-5")
    print("=" * 60)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate CWQ heuristics JSONL from a RoG-CWQ split file.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/cwq/ComplexWebQuestions_train.json",
        help="Input CWQ JSONL file (train/dev/test). Default: train.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/train_heuristics_cwq_train.jsonl",
        help="Output heuristics JSONL path. Default: data/train_heuristics_cwq_train.jsonl",
    )
    parser.add_argument(
        "--limit_triples",
        type=int,
        default=50,
        help="Max triples to keep per example (memory/speed knob). Default: 50.",
    )
    args = parser.parse_args()
    generate_heuristics(Path(args.input), Path(args.output), limit_triples=args.limit_triples)


if __name__ == "__main__":
    main()

