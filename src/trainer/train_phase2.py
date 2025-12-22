import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
from src.model.retriever import GNNRetriever
from src.model.sampler import DifferentiableSampler
from src.model.projector import Projector
from src.model.generator import NemotronGenerator
from src.model.question_encoder import QuestionEncoder
from src.data.kg_loader import load_kg
from src.data.qa_dataset import QADataset, collate_fn

def train_phase2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load KG
    kg_data, ent_map, rel_map = load_kg(args.kg_path)
    if kg_data is None:
        print("ERROR: Could not load KG. Exiting.")
        return
    kg_data = kg_data.to(device)
    node_dim = kg_data.x.shape[1]
    
    # 2. Load Dataset
    dataset = QADataset(args.dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 3. Initialize Components
    print("Initializing question encoder...")
    q_encoder = QuestionEncoder().to(device)
    
    print("Initializing GNN retriever...")
    retriever = GNNRetriever(
        node_dim=node_dim,
        edge_dim=1,
        hidden_dim=args.hidden_dim,
        question_dim=384
    ).to(device)
    
    # Load warmup checkpoint
    if args.gnn_checkpoint and os.path.exists(args.gnn_checkpoint):
        print(f"Loading GNN checkpoint from {args.gnn_checkpoint}")
        retriever.load_state_dict(torch.load(args.gnn_checkpoint, map_location=device))
    
    print("Initializing sampler and projector...")
    sampler = DifferentiableSampler(k=args.k_facts, temp=0.5).to(device)
    projector = Projector(gnn_dim=args.hidden_dim, nemotron_dim=2688).to(device)
    
    print("Initializing Nemotron generator...")
    generator = NemotronGenerator(model_id=args.llm_model)
    
    # Verify gradient flow through Mamba before training
    from src.model.generator import verify_gradient_flow
    print("\n--- Gradient Flow Verification ---")
    gradient_ok = verify_gradient_flow(generator, projector, None)
    if not gradient_ok:
        print("WARNING: Gradient flow issue detected. Training may not work correctly.")
        print("Consider using eager attention implementation or disabling gradient checkpointing.")
    print("-----------------------------------\n")
    
    # Optimizer (only trainable params)
    trainable_params = (
        list(retriever.parameters()) +
        list(projector.parameters()) +
        [p for p in generator.model.parameters() if p.requires_grad]
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    # 4. Training Loop
    generator.model.train()
    retriever.train()
    projector.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            questions = batch['questions']
            answers = batch['answers']
            
            if not questions or not answers[0]:
                continue
            
            # 1. Encode questions
            q_embed = q_encoder(questions).to(device)
            
            # 2. GNN forward (process one question at a time for simplicity)
            # For batching, you'd need to handle multiple graphs or aggregate
            logits = retriever(kg_data.x, kg_data.edge_index, None, q_embed[0:1])
            soft_mask = sampler(logits)
            
            # 3. Select top-k facts
            k = min(args.k_facts, soft_mask.shape[0])
            _, top_k_indices = torch.topk(soft_mask, k=k)
            selected_node_embeds = kg_data.x[top_k_indices]
            
            # Ensure gradient flow
            selected_node_embeds = selected_node_embeds * soft_mask[top_k_indices].unsqueeze(-1)
            
            # 4. Project to Nemotron space
            neural_prompt_embeds = projector(selected_node_embeds).unsqueeze(0)
            
            # 5. LLM forward with proper label alignment for Mamba
            # The generator now handles label creation internally
            outputs = generator(
                neural_prompt_embeds, 
                questions[:1], 
                answer_texts=answers[:1]
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"  Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'retriever': retriever.state_dict(),
            'projector': projector.state_dict(),
            'epoch': epoch + 1,
        }, f"checkpoints/phase2_epoch_{epoch+1}.pt")

    print("Phase 2 training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Joint End-to-End Training (D-RAG).")
    parser.add_argument("--kg_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to QA dataset (jsonl).")
    parser.add_argument("--gnn_checkpoint", type=str, help="Path to warmed-up GNN.")
    parser.add_argument("--llm_model", type=str, default="unsloth/Nemotron-3-Nano-30B-A3B-FP8")
    parser.add_argument("--k_facts", type=int, default=10, help="Number of facts to retrieve.")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_phase2(args)
