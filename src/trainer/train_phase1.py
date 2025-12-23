"""
D-RAG Phase 1: Retriever Pre-training with Per-Question Subgraphs

Pre-trains the GNN retriever on per-question subgraphs (SPARQL-derived labels).
Each question has its own small subgraph (~1000 edges) instead of one giant merged graph.

Based on D-RAG paper Appendix G:
- 10 epochs of pre-training
- AdamW optimizer, lr=5e-5, weight_decay=0.001
- BCE + Ranking loss (ρ=0.7)
- Batch size: 16
"""

import torch
import argparse
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.retriever import DRAGRetriever
from src.data.kg_loader import SubgraphDataset


def collate_fn(batch):
    """
    Custom collate function for per-question subgraphs.
    Filters out None samples and returns a list of individual samples.
    """
    # Filter out None samples
    batch = [b for b in batch if b is not None and b.get('num_positive', 0) > 0]
    
    if len(batch) == 0:
        return None
    
    return batch


def train_phase1(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Heuristics (each has its own subgraph!)
    print(f"Loading heuristics from {args.heuristics_path}...")
    with open(args.heuristics_path, 'r', encoding='utf-8') as f:
        heuristics = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(heuristics)} heuristic examples.")
    
    # Show sample structure
    if heuristics:
        sample = heuristics[0]
        print(f"Sample structure:")
        print(f"  - question: {sample.get('question', '')[:80]}...")
        print(f"  - graph_size: {sample.get('graph_size', 0)} triples")
        print(f"  - paths: {len(sample.get('paths', []))} gold paths")
    
    # 2. Create Dataset (per-question subgraphs!)
    dataset = SubgraphDataset(
        heuristics=heuristics,
        node_dim=args.node_dim,
        device=device
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Keep 0 since subgraphs are built on GPU
    )
    
    # 3. Initialize Retriever
    print("\nInitializing DRAGRetriever...")
    retriever = DRAGRetriever(
        node_dim=args.node_dim,
        edge_dim=args.relation_dim,
        hidden_dim=args.hidden_dim,
        instruction_dim=384,  # Sentence-BERT output dim
        relation_dim=args.relation_dim,
        num_reasoning_steps=args.num_reasoning_steps,
        num_heads=4,
        freeze_lm=args.freeze_lm,
        rho=args.rho,
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in retriever.parameters()):,}")
    
    # 4. Setup Optimizer
    optimizer = retriever.get_optimizer(lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # 5. Training Loop
    print(f"\n{'='*60}")
    print(f"Starting Phase 1 training for {args.epochs} epochs")
    print(f"  - Per-question subgraphs (NOT merged graph)")
    print(f"  - Dataset size: {len(dataset)} samples")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Loss weight (ρ): {args.rho}")
    print(f"  - Reasoning steps: {args.num_reasoning_steps}")
    print(f"{'='*60}\n")
    
    retriever.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_rank = 0.0
        num_batches = 0
        total_positives = 0
        total_edges = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            batch_loss = 0.0
            batch_bce = 0.0
            batch_rank = 0.0
            batch_samples = 0
            
            for sample in batch:
                question = sample['question']
                subgraph = sample['subgraph']
                rel_texts = sample['rel_texts']
                labels = sample['labels']
                
                total_positives += sample['num_positive']
                total_edges += sample['num_edges']
                
                # Encode relations for this subgraph
                with torch.no_grad():
                    if len(rel_texts) > 0 and rel_texts[0]:
                        rel_embeds = retriever.encode_relations(rel_texts).to(device)
                    else:
                        rel_embeds = torch.randn(len(rel_texts), args.relation_dim, device=device) * 0.02
                
                # Map to per-edge relation embeddings
                edge_relations = rel_embeds[subgraph.edge_type]
                
                # Forward pass on this sample's subgraph
                fact_probs, node_embeds, fact_embeds = retriever(
                    node_features=subgraph.x,
                    edge_index=subgraph.edge_index,
                    edge_attr=edge_relations,
                    edge_relations=edge_relations,
                    questions=[question],
                    fact_indices=None
                )
                
                # Compute loss
                loss, bce_loss, rank_loss = retriever.compute_loss(
                    fact_probs, labels, margin=args.margin
                )
                
                batch_loss += loss
                batch_bce += bce_loss.item()
                batch_rank += rank_loss.item()
                batch_samples += 1
            
            if batch_samples == 0:
                continue
            
            # Average over samples in batch
            batch_loss = batch_loss / batch_samples
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            epoch_bce += batch_bce / batch_samples
            epoch_rank += batch_rank / batch_samples
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'bce': f'{batch_bce/batch_samples:.4f}',
                'rank': f'{batch_rank/batch_samples:.4f}',
                'edges': f'{total_edges:,}'
            })
        
        # Epoch statistics
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_bce = epoch_bce / max(num_batches, 1)
        avg_rank = epoch_rank / max(num_batches, 1)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, Rank: {avg_rank:.4f}) | "
              f"Positives: {total_positives:,} | Edges: {total_edges:,} | LR: {current_lr:.2e}")
        
        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': retriever.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'args': vars(args)
        }
        torch.save(checkpoint, f"{args.checkpoint_dir}/phase1_epoch_{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, f"{args.checkpoint_dir}/phase1_best.pt")
            print(f"  -> Saved best checkpoint (loss: {best_loss:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Phase 1 training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="D-RAG Phase 1: GNN Retriever Pre-training with Per-Question Subgraphs"
    )
    
    # Data paths - NO KG FILE NEEDED! Each sample has its own subgraph.
    parser.add_argument("--heuristics_path", type=str, required=True,
                        help="Path to heuristics JSONL file (each sample has 'triples' and 'paths')")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    
    # Model architecture
    parser.add_argument("--node_dim", type=int, default=256,
                        help="Node feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="GNN hidden dimension")
    parser.add_argument("--relation_dim", type=int, default=256,
                        help="Relation embedding dimension")
    parser.add_argument("--num_reasoning_steps", type=int, default=3,
                        help="Number of GNN reasoning layers")
    
    # Training hyperparameters (from paper Appendix G)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of pre-training epochs (paper: 10)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (paper: 16)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (paper: 5e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                        help="Weight decay (paper: 0.001)")
    parser.add_argument("--rho", type=float, default=0.7,
                        help="BCE vs Ranking loss weight (paper: 0.7)")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for ranking loss")
    
    # Other options
    parser.add_argument("--freeze_lm", action="store_true",
                        help="Freeze the language model encoder")
    
    args = parser.parse_args()
    
    train_phase1(args)
