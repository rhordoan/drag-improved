import torch
import torch.nn.functional as F
import argparse
import os
from src.model.retriever import GNNRetriever
from src.model.sampler import DifferentiableSampler
from src.model.projector import Projector
from src.model.generator import NemotronGenerator
from src.data.kg_loader import load_kg

def train_phase2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load KG
    kg_data, ent_map, rel_map = load_kg(args.kg_path)
    kg_data = kg_data.to(device)
    
    # 2. Initialize Components
    retriever = GNNRetriever(
        node_dim=kg_data.x.shape[1],
        edge_dim=1,
        hidden_dim=256,
        question_dim=384
    ).to(device)
    
    # Load warmup checkpoint
    if args.gnn_checkpoint:
        print(f"Loading GNN checkpoint from {args.gnn_checkpoint}")
        retriever.load_state_dict(torch.load(args.gnn_checkpoint))
    
    sampler = DifferentiableSampler(k=args.k_facts, temp=0.5).to(device)
    projector = Projector(gnn_dim=256, nemotron_dim=2688).to(device)
    generator = NemotronGenerator(model_id=args.llm_model).to(device)
    
    # Optimizer
    # Note: generator already has LoRA applied in __init__
    params = list(retriever.parameters()) + list(projector.parameters()) + list(generator.model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    
    # 3. Training Loop
    generator.model.train()
    retriever.train()
    projector.train()
    
    # Dummy dataset for illustration
    questions = ["what is the name of justin bieber brother?"]
    answers = ["Jazmyn Bieber"]
    
    for epoch in range(args.epochs):
        # ... sample from real data ...
        
        # 1. GNN + Sampler
        # Simplified: one question at a time or grouped by batch
        q_embed = torch.randn(1, 384).to(device)
        logits = retriever(kg_data.x, kg_data.edge_index, None, q_embed)
        soft_mask = sampler(logits) # [num_nodes]
        
        # 2. Project Selected Facts
        # Simplified: weighted average of node embeddings based on soft mask
        # D-RAG usually selects Top-K and projects them individually
        # Here we take the top k indices
        _, top_k_indices = torch.topk(soft_mask, k=args.k_facts)
        selected_node_embeds = kg_data.x[top_k_indices] # [k_facts, gnn_dim]
        
        # Ensure gradient flow: multiply by the soft weights
        selected_node_embeds = selected_node_embeds * soft_mask[top_k_indices].unsqueeze(-1)
        
        # Project to Nemotron space
        # shape: [batch_size, k_facts, 2688]
        neural_prompt_embeds = projector(selected_node_embeds).unsqueeze(0)
        
        # 3. LLM Forward Pass
        # Prepare labels (tokenized answer)
        labels = generator.tokenizer(answers, return_tensors="pt", padding=True).input_ids.to(device)
        
        outputs = generator(neural_prompt_embeds, questions, labels=labels)
        
        loss_gen = outputs.loss
        
        # Optional: Retrieval KL Loss if teacher labels available
        # loss_ret = kl_divergence(logits, teacher_labels)
        # loss = loss_gen + args.alpha_balance * loss_ret
        
        loss = loss_gen
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Joint End-to-End Training (D-RAG).")
    parser.add_argument("--kg_path", type=str, required=True)
    parser.add_argument("--gnn_checkpoint", type=str, help="Path to warmed-up GNN.")
    parser.add_argument("--llm_model", type=str, default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    parser.add_argument("--k_facts", type=int, default=10, help="Number of facts to retrieve.")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_phase2(args)

