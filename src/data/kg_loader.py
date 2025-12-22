import torch
import os
from torch_geometric.data import Data

def load_kg(triples_file):
    """
    Loads a Knowledge Graph from a triples file (format: subject \t predicate \t object).
    Returns a torch_geometric.data.Data object and maps for entities and relations.
    """
    if not os.path.exists(triples_file):
        print(f"Warning: Triples file {triples_file} not found.")
        return None, {}, {}

    entities = {}
    relations = {}
    edges = []
    edge_types = []

    def get_id(mapping, item):
        if item not in mapping:
            mapping[item] = len(mapping)
        return mapping[item]

    print(f"Loading triples from {triples_file}...")
    with open(triples_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            
            s, p, o = parts
            s_id = get_id(entities, s)
            p_id = get_id(relations, p)
            o_id = get_id(entities, o)
            
            edges.append([s_id, o_id])
            edge_types.append(p_id)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    # Node features: for now, just random or zero, to be updated during training or with pre-trained embeddings
    x = torch.zeros((len(entities), 128)) # Example dim 128

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    print(f"Loaded {len(entities)} entities and {len(relations)} relations with {edge_index.shape[1]} edges.")
    return data, entities, relations

if __name__ == "__main__":
    # Example usage
    kg_path = "data/kg/freebase_2hop.txt"
    data, ent_map, rel_map = load_kg(kg_path)

