import json
import networkx as nx
import argparse
import os

def mine_heuristic_subgraphs(triples_file, dataset_file, output_file):
    """
    Finds shortest paths between topic entities and answer entities in the KG.
    Saves results as train_heuristics.jsonl.
    """
    print("Building networkx graph...")
    G = nx.MultiDiGraph()
    with open(triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                s, p, o = parts
                G.add_edge(s, o, predicate=p)

    print(f"Loading dataset from {dataset_file}...")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    heuristics = []
    print("Mining shortest paths...")
    for idx, item in enumerate(dataset):
        topic_entities = item.get('entities', [])
        answer_entities = item.get('answers', [])
        
        item_paths = []
        for start_node in topic_entities:
            for end_node in answer_entities:
                if start_node in G and end_node in G:
                    try:
                        # Find path up to 3 hops
                        for path in nx.all_simple_paths(G, source=start_node, target=end_node, cutoff=3):
                            item_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        heuristics.append({
            "id": item.get("id", idx),
            "question": item.get("question"),
            "paths": item_paths
        })

    print(f"Saving {len(heuristics)} heuristics to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in heuristics:
            f.write(json.dumps(entry) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine heuristic subgraphs (shortest paths) from KG.")
    parser.add_argument("--kg", type=str, required=True, help="Path to KG triples file.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--output", type=str, default="data/train_heuristics.jsonl", help="Path to save heuristics.")
    args = parser.parse_args()
    
    mine_heuristic_subgraphs(args.kg, args.dataset, args.output)

