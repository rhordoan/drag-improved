import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import json
import os
from src.model.retriever import GNNRetriever
from src.model.question_encoder import QuestionEncoder
from src.data.kg_loader import load_kg

def train_phase1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load KG
    kg_data, ent_map, rel_map = load_kg(args.kg_path)
    if kg_data is None:
        print("ERROR: Could not load KG. Exiting.")
        return
    kg_data = kg_data.to(device)
    
    # 2. Load Heuristics
    print(f"Loading heuristics from {args.heuristics_path}...")
    with open(args.heuristics_path, 'r', encoding='utf-8') as f:
        heuristics = [json.loads(line) for line in f]
    print(f"Loaded {len(heuristics)} heuristic examples.")
    
    # 3. Initialize Question Encoder
    print("Initializing question encoder...")
    q_encoder = QuestionEncoder().to(device)
    
    # 4. Initialize Retriever
    node_dim = kg_data.x.shape[1]
    retriever = GNNRetriever(
        node_dim=node_dim,
        edge_dim=1,
        hidden_dim=args.hidden_dim,
        question_dim=384  # MiniLM output dim
    ).to(device)
    
    optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
    
    # 5. Training Loop
    retriever.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_samples = 0
        
        for item in heuristics:
            question = item.get('question', '')
            if not question:
                continue
            
            # Encode question
            q_embed = q_encoder([question]).to(device)
            
            # Ground truth labels from heuristics
            labels = torch.zeros(kg_data.num_nodes).to(device)
            path_nodes = set()
            for path in item.get('paths', []):
                for node in path:
                    if node in ent_map:
                        path_nodes.add(ent_map[node])
            
            if len(path_nodes) == 0:
                continue  # Skip if no valid paths
            
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
            num_samples += 1
        
        avg_loss = total_loss / max(num_samples, 1)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Samples: {num_samples}")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(retriever.state_dict(), f"checkpoints/phase1_epoch_{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(retriever.state_dict(), "checkpoints/phase1_best.pt")
            print(f"  -> Saved best checkpoint (loss: {best_loss:.4f})")

    print("Phase 1 training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: GNN Warmup (Supervised Pre-training).")
    parser.add_argument("--kg_path", type=str, required=True, help="Path to KG triples file.")
    parser.add_argument("--heuristics_path", type=str, required=True, help="Path to mined heuristics.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    args = parser.parse_args()
    
    train_phase1(args)
