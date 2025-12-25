"""
Knowledge Graph Loader for D-RAG

Loads KG triples and prepares them for the DRAGRetriever.
Returns PyG Data with proper node features and relation information.
"""

import torch
import os
from torch_geometric.data import Data
from typing import Tuple, Dict, List, Optional


def load_kg(
    triples_file: str,
    node_dim: int = 256,
    init_method: str = "random"
) -> Tuple[Optional[Data], Dict[str, int], Dict[str, int], List[str]]:
    """
    Loads a Knowledge Graph from a triples file (format: subject \t predicate \t object).
    
    Args:
        triples_file: Path to triples file (TSV format: subject \t predicate \t object)
        node_dim: Dimension for node feature initialization
        init_method: How to initialize node features ("random", "zeros", "onehot")
    
    Returns:
        data: torch_geometric.data.Data object with:
            - x: [num_nodes, node_dim] node features
            - edge_index: [2, num_edges] edge connectivity
            - edge_type: [num_edges] relation type indices
        ent_map: Dict mapping entity string -> node index
        rel_map: Dict mapping relation string -> relation index
        rel_texts: List of relation text strings (for encoding)
    """
    if not os.path.exists(triples_file):
        print(f"Warning: Triples file {triples_file} not found.")
        return None, {}, {}, []

    entities = {}
    relations = {}
    edges = []
    edge_types = []
    
    # Store original triple info for fact-level supervision
    triples = []  # List of (subject, predicate, object) strings

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
            triples.append((s, p, o))

    if len(edges) == 0:
        print("Warning: No valid triples found in file.")
        return None, {}, {}, []

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    # Initialize node features
    num_nodes = len(entities)
    if init_method == "random":
        x = torch.randn(num_nodes, node_dim) * 0.02  # Small random init
    elif init_method == "zeros":
        x = torch.zeros(num_nodes, node_dim)
    elif init_method == "onehot":
        # One-hot requires node_dim >= num_nodes, fallback to random if not
        if node_dim >= num_nodes:
            x = torch.eye(num_nodes, node_dim)
        else:
            print(f"Warning: node_dim ({node_dim}) < num_nodes ({num_nodes}), using random init")
            x = torch.randn(num_nodes, node_dim) * 0.02
    else:
        x = torch.randn(num_nodes, node_dim) * 0.02

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    # Store triples in data object for fact-level access
    data.triples = triples
    data.num_relations = len(relations)
    
    # Create relation text list (ordered by index)
    rel_texts = [""] * len(relations)
    for rel_str, rel_idx in relations.items():
        rel_texts[rel_idx] = rel_str
    
    print(f"Loaded {len(entities)} entities and {len(relations)} relations with {edge_index.shape[1]} edges.")
    return data, entities, relations, rel_texts


def create_fact_labels(
    kg_data: Data,
    ent_map: Dict[str, int],
    rel_map: Dict[str, int],
    relevant_triples: List[Tuple[str, str, str]]
) -> torch.Tensor:
    """
    Create fact-level (edge-level) labels from a list of relevant triples.
    GPU-ACCELERATED version using vectorized operations.
    
    Args:
        kg_data: PyG Data object with edge_index and edge_type
        ent_map: Entity string -> index mapping
        rel_map: Relation string -> index mapping
        relevant_triples: List of (subject, predicate, object) tuples that are relevant
    
    Returns:
        labels: [num_edges] binary tensor (1 for relevant facts, 0 otherwise)
    """
    num_edges = kg_data.edge_index.shape[1]
    device = kg_data.edge_index.device
    
    # Convert relevant triples to tensor [num_relevant, 3] on same device
    relevant_list = []
    for s, p, o in relevant_triples:
        if s in ent_map and p in rel_map and o in ent_map:
            relevant_list.append([ent_map[s], rel_map[p], ent_map[o]])
    
    if len(relevant_list) == 0:
        return torch.zeros(num_edges, device=device)
    
    relevant_tensor = torch.tensor(relevant_list, dtype=torch.long, device=device)  # [R, 3]
    
    # Build edge tensor [num_edges, 3]: (head, rel, tail)
    edge_triples = torch.stack([
        kg_data.edge_index[0],  # heads
        kg_data.edge_type,       # relations
        kg_data.edge_index[1]    # tails
    ], dim=1)  # [E, 3]
    
    # GPU-accelerated set membership check
    # Compare each edge against all relevant triples
    # edge_triples: [E, 3], relevant_tensor: [R, 3]
    # We need to check if each edge is in the relevant set
    
    # Method: Use broadcasting to compare
    # Expand dims: edge_triples [E, 1, 3], relevant_tensor [1, R, 3]
    matches = (edge_triples.unsqueeze(1) == relevant_tensor.unsqueeze(0)).all(dim=2)  # [E, R]
    
    # If any relevant triple matches, label is 1
    labels = matches.any(dim=1).float()  # [E]
    
    return labels


def paths_to_triples(
    paths: List[List[str]],
    kg_triples: List[Tuple[str, str, str]]
) -> List[Tuple[str, str, str]]:
    """
    Extract triples from paths for fact-level supervision.
    
    A path like [e1, e2, e3] implies edges (e1->e2) and (e2->e3) exist.
    We find the actual triples containing these entity pairs.
    
    Args:
        paths: List of entity paths (each path is a list of entity strings)
        kg_triples: All triples in the KG
    
    Returns:
        relevant_triples: List of (s, p, o) triples covered by the paths
    """
    # Build lookup: (head, tail) -> list of (head, rel, tail)
    edge_lookup = {}
    for s, p, o in kg_triples:
        key = (s, o)
        if key not in edge_lookup:
            edge_lookup[key] = []
        edge_lookup[key].append((s, p, o))
    
    relevant_triples = []
    seen = set()
    
    for path in paths:
        # Each consecutive pair in path is an edge
        for i in range(len(path) - 1):
            head, tail = path[i], path[i + 1]
            
            # Check both directions (KG may have either)
            for key in [(head, tail), (tail, head)]:
                if key in edge_lookup:
                    for triple in edge_lookup[key]:
                        if triple not in seen:
                            seen.add(triple)
                            relevant_triples.append(triple)
    
    return relevant_triples


class SubgraphDataset:
    """
    Per-question subgraph dataset for D-RAG training.
    Each sample builds its own small PyG graph from its `triples` field.
    This matches the paper's approach and is memory efficient.
    """
    def __init__(
        self, 
        heuristics: List[dict],
        node_dim: int = 256,
        device: torch.device = torch.device('cpu')
    ):
        self.heuristics = heuristics
        self.node_dim = node_dim
        self.device = device
        
        # Filter to samples with valid subgraphs
        self.valid_indices = []
        for i, item in enumerate(heuristics):
            triples = item.get('triples', [])
            paths = item.get('paths', [])
            if len(triples) >= 3 and len(paths) > 0:  # Need some triples and labels
                self.valid_indices.append(i)
        
        print(f"SubgraphDataset: {len(self.valid_indices):,}/{len(heuristics):,} samples have valid subgraphs")
        
        # Pre-compute statistics
        sizes = [len(heuristics[i].get('triples', [])) for i in self.valid_indices[:100]]
        if sizes:
            print(f"  Subgraph sizes (first 100): min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.0f}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def build_subgraph(self, triples: List, device: torch.device) -> Tuple[Data, Dict[str, int], Dict[str, int], List[str]]:
        """
        Build a PyG Data object from a list of triples.
        
        Args:
            triples: List of [head, relation, tail] triples
            device: torch device
        
        Returns:
            data: PyG Data object
            ent_map: entity -> index mapping
            rel_map: relation -> index mapping
            rel_texts: list of relation strings
        """
        ent_map = {}
        rel_map = {}
        edges = []
        edge_types = []
        
        def get_id(mapping, item):
            if item not in mapping:
                mapping[item] = len(mapping)
            return mapping[item]
        
        for triple in triples:
            if len(triple) < 3:
                continue
            s, p, o = triple[0], triple[1], triple[2]
            s_id = get_id(ent_map, s)
            p_id = get_id(rel_map, p)
            o_id = get_id(ent_map, o)
            edges.append([s_id, o_id])
            edge_types.append(p_id)
        
        if len(edges) == 0:
            return None, {}, {}, []
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long, device=device)
        
        # Initialize node features
        num_nodes = len(ent_map)
        x = torch.randn(num_nodes, self.node_dim, device=device) * 0.02
        
        data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
        data.num_relations = len(rel_map)
        
        # Relation texts
        rel_texts = [""] * len(rel_map)
        for rel_str, rel_idx in rel_map.items():
            rel_texts[rel_idx] = rel_str
        
        return data, ent_map, rel_map, rel_texts
    
    def create_labels(
        self, 
        paths: List[List[str]], 
        triples: List, 
        ent_map: Dict[str, int],
        num_edges: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create fact-level labels from paths.
        Labels are 1 for edges that appear in the gold paths.
        """
        # Build triple index: (head, tail) -> edge indices
        triple_to_edges = {}
        for edge_idx, triple in enumerate(triples):
            if len(triple) < 3:
                continue
            h, r, t = triple[0], triple[1], triple[2]
            # Check both directions
            for key in [(h, t), (t, h)]:
                if key not in triple_to_edges:
                    triple_to_edges[key] = []
                triple_to_edges[key].append(edge_idx)
        
        labels = torch.zeros(num_edges, dtype=torch.float32, device=device)
        
        for path in paths:
            for i in range(len(path) - 1):
                head, tail = path[i], path[i + 1]
                for key in [(head, tail), (tail, head)]:
                    if key in triple_to_edges:
                        for edge_idx in triple_to_edges[key]:
                            labels[edge_idx] = 1.0
        
        return labels
    
    def __getitem__(self, idx):
        """
        Returns a complete sample with its own subgraph.
        """
        item = self.heuristics[self.valid_indices[idx]]
        question = item.get('question', '')
        triples = item.get('triples', [])
        paths = item.get('paths', [])
        answer = item.get('answer', '')
        
        # Build subgraph
        subgraph, ent_map, rel_map, rel_texts = self.build_subgraph(triples, self.device)
        
        if subgraph is None:
            # Return empty sample (will be filtered)
            return None
        
        # Create labels
        num_edges = subgraph.edge_index.shape[1]
        labels = self.create_labels(paths, triples, ent_map, num_edges, self.device)
        
        return {
            'question': question,
            'answer': answer,
            # Preserve original textual triples so we can build the paper-style "Provided facts"
            # prompt during Phase 2 (fact index == triple index for this subgraph).
            'triples': triples,
            'subgraph': subgraph,
            'rel_texts': rel_texts,
            'labels': labels,
            'num_positive': int(labels.sum().item()),
            'num_edges': num_edges
        }


# Keep old KGDataset for backward compatibility but mark as deprecated
class KGDataset:
    """
    DEPRECATED: Use SubgraphDataset for per-question subgraph training.
    This class merges all subgraphs into one giant graph, which doesn't scale.
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "KGDataset is deprecated. Use SubgraphDataset for per-question subgraph training. "
            "Each sample in CWQ/WebQSP already has its own subgraph in the 'triples' field."
        )


if __name__ == "__main__":
    # Example usage
    kg_path = "data/kg/freebase_2hop.txt"
    data, ent_map, rel_map, rel_texts = load_kg(kg_path, node_dim=256)
    
    if data is not None:
        print(f"Node features shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Sample relations: {rel_texts[:5]}")
