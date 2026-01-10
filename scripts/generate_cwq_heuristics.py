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
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple


def _build_undirected_adj(triples: List[List[str]]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = defaultdict(list)
    for t in triples:
        if not t or len(t) < 3:
            continue
        h, _, tail = t[0], t[1], t[2]
        if h and tail:
            adj[h].append(tail)
            adj[tail].append(h)
    return adj


def _bfs_path(adj: Dict[str, List[str]], start: str, goal: str, max_hops: int) -> Optional[List[str]]:
    """
    Return a single shortest path (as entity list) from start->goal using BFS, up to max_hops edges.
    Undirected adjacency.
    """
    if start == goal:
        return [start]
    if start not in adj or goal not in adj:
        return None

    q = deque([(start, 0)])
    parent: Dict[str, Optional[str]] = {start: None}
    depth: Dict[str, int] = {start: 0}

    while q:
        node, d = q.popleft()
        if d >= max_hops:
            continue
        for nb in adj.get(node, []):
            if nb in parent:
                continue
            parent[nb] = node
            depth[nb] = d + 1
            if nb == goal:
                # reconstruct
                path = [goal]
                cur = goal
                while parent[cur] is not None:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append((nb, d + 1))
    return None


def _fallback_edge_paths(adj: Dict[str, List[str]], seeds: List[str], max_paths: int) -> List[List[str]]:
    """
    If we can't find a multi-hop path between q_entity and a_entity within the capped subgraph,
    fall back to a few 1-hop "paths" that correspond to real edges touching a seed entity.
    This ensures num_positive > 0 for more examples (so they don't get filtered away).
    """
    out: List[List[str]] = []
    for s in seeds:
        if s not in adj:
            continue
        for nb in adj[s][: max_paths]:
            if len(out) >= max_paths:
                return out
            out.append([s, nb])
    return out


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
        
        # Extract triples from graph (cap for memory/speed)
        triples = []
        for item in graph:
            if len(item) >= 3:
                s, p, o = item[0], item[1], item[2]
                triples.append((s, p, o))
        
        triples_capped = [[s, p, o] for s, p, o in triples[: max(0, int(limit_triples))]]
        adj = _build_undirected_adj(triples_capped)

        # Build paths which actually correspond to consecutive entity pairs in the subgraph.
        # This is crucial: downstream labeling marks an edge positive iff (path[i], path[i+1])
        # matches some triple head/tail in `triples`.
        paths: List[List[str]] = []

        qe_list = q_entity if isinstance(q_entity, list) else ([q_entity] if q_entity else [])
        ae_list = a_entity if isinstance(a_entity, list) else ([a_entity] if a_entity else [])
        # If a_entity is missing, fall back to answer strings (may not match entities, but try).
        if not ae_list and answer:
            ae_list = answer if isinstance(answer, list) else [answer]

        # Try shortest paths up to 4 hops (paper uses 3-4 hop reasoning for CWQ).
        max_hops = 4
        for qe in qe_list[:3]:
            for ae in ae_list[:3]:
                if len(paths) >= 8:
                    break
                if not isinstance(qe, str) or not isinstance(ae, str):
                    continue
                pth = _bfs_path(adj, qe, ae, max_hops=max_hops)
                if pth and len(pth) >= 2:
                    paths.append(pth)
        
        # Fallback: if no connecting path found, add a few real 1-hop edges touching answer/q entities.
        if not paths:
            seed_entities = []
            seed_entities.extend([x for x in ae_list[:3] if isinstance(x, str)])
            seed_entities.extend([x for x in qe_list[:3] if isinstance(x, str)])
            paths = _fallback_edge_paths(adj, seed_entities, max_paths=4)

        # Last-resort fallback: ensure at least *something* is present if graph is non-empty.
        if not paths and triples_capped:
            h0, _, t0 = triples_capped[0]
            if h0 and t0:
                paths = [[h0, t0]]
        
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
            "triples": triples_capped,
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
    # NOTE: Keep logs ASCII-only for Windows consoles (cp1252) to avoid UnicodeEncodeError.
    print(f"OK: Generated {len(heuristics)} heuristics from CWQ file: {input_path.name}")
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

