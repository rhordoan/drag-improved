"""
Extract Knowledge Graph from RoG-WebQSP/CWQ datasets.
The datasets already contain pre-extracted Freebase subgraphs.
"""
import json
import os
import torch
from torch_geometric.data import Data

def build_kg_from_dataset(dataset_file, output_triples_file=None):
    """
    Extracts all unique triples from the 'graph' field in the RoG dataset
    and builds a unified KG.
    
    Args:
        dataset_file: Path to the jsonl dataset (e.g., data/cwq/ComplexWebQuestions_train.json)
        output_triples_file: Optional path to save extracted triples
    
    Returns:
        Data object, entity map, relation map
    """
    print(f"Extracting KG triples from {dataset_file}...")
    
    entities = {}
    relations = {}
    edges = []
    edge_types = []
    
    def get_id(mapping, item):
        if item not in mapping:
            mapping[item] = len(mapping)
        return mapping[item]
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # The 'graph' field contains the subgraph triples
            graph_triples = item.get('graph', [])
            
            for triple in graph_triples:
                if len(triple) == 3:
                    s, p, o = triple
                    s_id = get_id(entities, s)
                    p_id = get_id(relations, p)
                    o_id = get_id(entities, o)
                    
                    edges.append([s_id, o_id])
                    edge_types.append(p_id)
    
    if len(edges) == 0:
        print("Warning: No edges found. Check if the dataset has a 'graph' field.")
        return None, entities, relations
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    # Initialize node features (can be replaced with pre-trained embeddings)
    x = torch.zeros((len(entities), 128))
    
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    print(f"Built KG with {len(entities)} entities, {len(relations)} relations, {edge_index.shape[1]} edges.")
    
    # Optionally save triples to file
    if output_triples_file:
        os.makedirs(os.path.dirname(output_triples_file), exist_ok=True)
        with open(output_triples_file, 'w', encoding='utf-8') as f:
            for (s_id, o_id), p_id in zip(edges, edge_types):
                # Reverse lookup
                s = [k for k, v in entities.items() if v == s_id][0]
                p = [k for k, v in relations.items() if v == p_id][0]
                o = [k for k, v in entities.items() if v == o_id][0]
                f.write(f"{s}\t{p}\t{o}\n")
        print(f"Saved triples to {output_triples_file}")
    
    return data, entities, relations


def build_combined_kg(dataset_files, output_triples_file=None):
    """
    Builds a unified KG from multiple dataset files.
    """
    print(f"Building combined KG from {len(dataset_files)} files...")
    
    entities = {}
    relations = {}
    edges = []
    edge_types = []
    
    def get_id(mapping, item):
        if item not in mapping:
            mapping[item] = len(mapping)
        return mapping[item]
    
    for dataset_file in dataset_files:
        if not os.path.exists(dataset_file):
            print(f"Warning: {dataset_file} not found, skipping.")
            continue
            
        print(f"  Processing {dataset_file}...")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                graph_triples = item.get('graph', [])
                
                for triple in graph_triples:
                    if len(triple) == 3:
                        s, p, o = triple
                        s_id = get_id(entities, s)
                        p_id = get_id(relations, p)
                        o_id = get_id(entities, o)
                        
                        edges.append([s_id, o_id])
                        edge_types.append(p_id)
    
    if len(edges) == 0:
        print("Warning: No edges found in any dataset.")
        return None, entities, relations
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    x = torch.zeros((len(entities), 128))
    
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    print(f"Combined KG: {len(entities)} entities, {len(relations)} relations, {edge_index.shape[1]} edges.")
    
    if output_triples_file:
        os.makedirs(os.path.dirname(output_triples_file), exist_ok=True)
        # Save entity and relation maps
        with open(output_triples_file.replace('.txt', '_entities.json'), 'w') as f:
            json.dump(entities, f)
        with open(output_triples_file.replace('.txt', '_relations.json'), 'w') as f:
            json.dump(relations, f)
        print(f"Saved entity/relation maps.")
    
    return data, entities, relations


if __name__ == "__main__":
    # Example: Build KG from CWQ dataset
    kg_data, ent_map, rel_map = build_combined_kg([
        "data/cwq/ComplexWebQuestions_train.json",
        "data/cwq/ComplexWebQuestions_dev.json",
    ], output_triples_file="data/kg/extracted_triples.txt")

