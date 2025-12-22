import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import json
import os
from src.model.retriever import GNNRetriever
from src.data.kg_loader import load_kg

def train_phase1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load KG
    kg_data, ent_map, rel_map = load_kg(args.kg_path)
    kg_data = kg_data.to(device)
    
    # 2. Load Heuristics
    print(f"Loading heuristics from {args.heuristics_path}...")
    with open(args.heuristics_path, 'r', encoding='utf-8') as f:
        heuristics = [json.loads(line) for line in f]
    
    # 3. Initialize Retriever
    # Example dimensions, adjust to your actual data
    retriever = GNNRetriever(
        node_dim=kg_data.x.shape[1],
        edge_dim=1, # assuming scalar edge attr for now
        hidden_dim=256,
        question_dim=384 # Example: BERT-tiny or similar
    ).to(device)
    
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
    
    # 4. Training Loop
    retriever.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for item in heuristics:
            # Prepare question embedding (simplified, usually pre-computed)
            q_embed = torch.randn(1, 384).to(device) 
            
            # Ground truth labels from heuristics
            # For simplicity, nodes in the shortest path are labeled 1, others 0
            labels = torch.zeros(kg_data.num_nodes).to(device)
            path_nodes = set()
            for path in item['paths']:
                for node in path:
                    if node in ent_map:
                        path_nodes.add(ent_map[node])
            
            for node_idx in path_nodes:
                labels[node_idx] = 1.0
            
            # Forward
            logits = retriever(kg_data.x, kg_data.edge_index, None, q_embed)
            
            # Loss: Binary Cross Entropy
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(heuristics)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(retriever.state_with_dict(), f"checkpoints/phase1_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: GNN Warmup (Supervised Pre-training).")
    parser.add_argument("--kg_path", type=str, required=True, help="Path to KG triples file.")
    parser.add_argument("--heuristics_path", type=str, required=True, help="Path to mined heuristics.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_phase1(args)

