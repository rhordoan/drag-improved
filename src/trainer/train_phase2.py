"""
D-RAG Phase 2: Joint End-to-End Training

Jointly trains the retriever with the generator using differentiable sampling.
The retriever learns what facts the generator actually needs to answer questions.

Based on D-RAG paper Appendix G:
- 5 epochs of joint training
- AdamW optimizer, lr=5e-5, weight_decay=0.001
- Differentiable binary Gumbel-Softmax with STE
- Generator: Nemotron (or any causal LM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.retriever import DRAGRetriever
from src.model.sampler import DifferentiableFactSampler
from src.model.projector import Projector
from src.model.generator import NemotronGenerator
from src.data.kg_loader import SubgraphDataset


def collate_fn(batch):
    """Custom collate function for Phase 2."""
    batch = [b for b in batch if b is not None and b.get('num_positive', 0) > 0]
    if len(batch) == 0:
        return None
    return batch


def train_phase2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"{'='*60}")
    print(f"D-RAG Phase 2: Joint End-to-End Training")
    print(f"{'='*60}")
    
    # 1. Load Heuristics (same format as Phase 1, but we need answers too)
    print(f"\nLoading data from {args.heuristics_path}...")
    with open(args.heuristics_path, 'r', encoding='utf-8') as f:
        heuristics = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(heuristics)} examples.")
    
    # 2. Create Dataset (per-question subgraphs)
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
        num_workers=0
    )
    
    # 3. Initialize Retriever and load Phase 1 checkpoint
    print("\nInitializing DRAGRetriever...")
    retriever = DRAGRetriever(
        node_dim=args.node_dim,
        edge_dim=args.relation_dim,
        hidden_dim=args.hidden_dim,
        instruction_dim=384,
        relation_dim=args.relation_dim,
        num_reasoning_steps=args.num_reasoning_steps,
        num_heads=4,
        freeze_lm=True,  # Freeze LM during joint training
        rho=args.rho,
    ).to(device)
    
    # Load Phase 1 checkpoint
    if args.phase1_checkpoint and os.path.exists(args.phase1_checkpoint):
        print(f"Loading Phase 1 checkpoint: {args.phase1_checkpoint}")
        checkpoint = torch.load(args.phase1_checkpoint, map_location=device)
        
        # Handle key name mismatches between Phase 1 checkpoint and current model
        state_dict = checkpoint['model_state_dict']
        model_state = retriever.state_dict()
        
        # Key mapping from old checkpoint to new model structure
        # Phase 1 used: question_encoder.model.*, relation_encoder.question_encoder.model.*
        # Phase 2 expects: instruction_module.encoder.*, relation_encoder.lm_encoder.encoder.*
        key_mapping = {
            # Instruction module (BERT encoder for questions)
            'question_encoder.model.': 'instruction_module.encoder.',
            # Relation encoder's LM (separate BERT weights in old structure -> shared in new, but map anyway)
            'relation_encoder.question_encoder.model.': 'relation_encoder.lm_encoder.encoder.',
        }
        
        # Remap keys from checkpoint
        remapped_dict = {}
        for old_key, value in state_dict.items():
            new_key = old_key
            for old_prefix, new_prefix in key_mapping.items():
                if old_key.startswith(old_prefix):
                    new_key = new_prefix + old_key[len(old_prefix):]
                    break
            remapped_dict[new_key] = value
        
        # Count matches after remapping
        matched_keys = sum(1 for k in remapped_dict if k in model_state)
        missing_keys = [k for k in model_state if k not in remapped_dict]
        unexpected_keys = [k for k in remapped_dict if k not in model_state]
        
        print(f"  Matched keys: {matched_keys}/{len(model_state)}")
        
        if matched_keys > 0:
            retriever.load_state_dict(remapped_dict, strict=False)
            print(f"  Loaded {matched_keys} matching keys from checkpoint")
            if len(missing_keys) > 5:
                print(f"  Missing {len(missing_keys)} keys (will use fresh init)")
            if len(unexpected_keys) > 5:
                print(f"  Ignored {len(unexpected_keys)} unexpected keys")
        else:
            print("  WARNING: No matching keys found. Using fresh model weights.")
        
        print(f"  Checkpoint from epoch {checkpoint.get('epoch', '?')}, loss: {checkpoint.get('loss', '?'):.4f}")
    else:
        print("WARNING: No Phase 1 checkpoint provided! Training from scratch.")
    
    # 4. Initialize Sampler
    print("Initializing Differentiable Sampler...")
    sampler = DifferentiableFactSampler(temp=args.temperature).to(device)
    
    # 5. Initialize Projector (maps GNN fact embeddings to generator space)
    print("Initializing Projector...")
    # Fact embedding dimension: node_dim * 2 + relation_dim (head + tail + relation)
    fact_embed_dim = args.node_dim * 2 + args.relation_dim
    projector = Projector(gnn_dim=fact_embed_dim, nemotron_dim=args.generator_dim).to(device)
    
    # 6. Initialize Generator (Nemotron)
    print(f"Initializing Generator: {args.generator_model}...")
    generator = NemotronGenerator(model_id=args.generator_model, use_lora=True)
    
    # Verify gradient flow
    print("\nVerifying gradient flow through generator...")
    from src.model.generator import verify_gradient_flow
    gradient_ok = verify_gradient_flow(generator, projector, None)
    if not gradient_ok:
        print("WARNING: Gradient flow issue. Training may not work correctly.")
    
    # 7. Setup Optimizer
    # Only train: retriever (fine-tune), projector, generator (LoRA)
    trainable_params = [
        {'params': retriever.parameters(), 'lr': args.lr * 0.1},  # Lower LR for retriever
        {'params': projector.parameters(), 'lr': args.lr},
        {'params': [p for p in generator.model.parameters() if p.requires_grad], 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    
    # 8. Training Loop
    print(f"\n{'='*60}")
    print(f"Starting Phase 2 training for {args.epochs} epochs")
    print(f"  - Dataset size: {len(dataset)} samples")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max facts per sample: {args.k_facts}")
    print(f"  - Generator: {args.generator_model}")
    print(f"{'='*60}\n")
    
    generator.model.train()
    retriever.train()
    projector.train()
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_gen_loss = 0.0
        epoch_ret_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            batch_loss = 0.0
            batch_gen_loss = 0.0
            batch_ret_loss = 0.0
            batch_samples = 0
            
            for sample in batch:
                question = sample['question']
                answer = sample['answer']
                subgraph = sample['subgraph']
                rel_texts = sample['rel_texts']
                retrieval_labels = sample['labels'].to(device)  # From Phase 1 heuristics
                
                if not answer:
                    batch_samples += 1
                    continue
                
                # --- Retriever Forward Pass ---
                # Encode relations for this subgraph
                with torch.no_grad():
                    if len(rel_texts) > 0 and rel_texts[0]:
                        rel_embeds = retriever.encode_relations(rel_texts).to(device)
                    else:
                        rel_embeds = torch.randn(len(rel_texts), args.relation_dim, device=device) * 0.02
                
                edge_relations = rel_embeds[subgraph.edge_type]
                
                # Retriever forward pass
                fact_probs, node_embeds, fact_embeds = retriever(
                    node_features=subgraph.x,
                    edge_index=subgraph.edge_index,
                    edge_attr=edge_relations,
                    edge_relations=edge_relations,
                    questions=[question],
                    fact_indices=None
                )
                
                # --- Differentiable Sampling ---
                # Convert probs to logits for Gumbel-Softmax
                fact_logits = torch.log(fact_probs + 1e-8) - torch.log(1 - fact_probs + 1e-8)
                selection_mask = sampler(fact_logits)  # Binary selection with STE
                
                # Select top-k facts by probability (for efficiency)
                k = min(args.k_facts, selection_mask.shape[0])
                _, top_k_indices = torch.topk(fact_probs, k=k)
                
                # Get selected fact embeddings
                selected_fact_embeds = fact_embeds[top_k_indices]  # [k, fact_dim]
                selected_probs = selection_mask[top_k_indices]  # [k]
                
                # Weight by selection probability (for gradient flow)
                weighted_fact_embeds = selected_fact_embeds * selected_probs.unsqueeze(-1)
                
                # --- Project to Generator Space ---
                neural_prompt_embeds = projector(weighted_fact_embeds)  # [k, generator_dim]
                neural_prompt_embeds = neural_prompt_embeds.unsqueeze(0)  # [1, k, generator_dim]
                
                # Convert to generator's dtype (bfloat16) for compatibility
                # Gradients flow through this conversion automatically
                neural_prompt_embeds = neural_prompt_embeds.to(dtype=torch.bfloat16)
                
                # --- Generator Forward Pass ---
                try:
                    outputs = generator(
                        neural_prompt_embeds, 
                        questions=[question], 
                        answer_texts=[answer]
                    )
                    gen_loss = outputs.loss
                except Exception as e:
                    print(f"Generator error: {e}")
                    continue
                
                # --- Retriever Auxiliary Loss (optional, helps with stability) ---
                ret_loss, _, _ = retriever.compute_loss(fact_probs, retrieval_labels)
                
                # --- Combined Loss ---
                # Generator loss + weighted retriever loss
                loss = gen_loss + args.ret_loss_weight * ret_loss
                
                batch_loss += loss
                batch_gen_loss += gen_loss.item()
                batch_ret_loss += ret_loss.item()
                batch_samples += 1
            
            if batch_samples == 0:
                continue
            
            # Average and backprop
            batch_loss = batch_loss / batch_samples
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(retriever.parameters()) + list(projector.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            epoch_gen_loss += batch_gen_loss / batch_samples
            epoch_ret_loss += batch_ret_loss / batch_samples
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'gen': f'{batch_gen_loss/batch_samples:.4f}',
                'ret': f'{batch_ret_loss/batch_samples:.4f}'
            })
        
        # Epoch statistics
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_gen = epoch_gen_loss / max(num_batches, 1)
        avg_ret = epoch_ret_loss / max(num_batches, 1)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} (Gen: {avg_gen:.4f}, Ret: {avg_ret:.4f}) | "
              f"LR: {current_lr:.2e}")
        
        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'retriever_state_dict': retriever.state_dict(),
            'projector_state_dict': projector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'args': vars(args)
        }
        
        # Save generator LoRA weights separately
        generator.model.save_pretrained(f"{args.checkpoint_dir}/generator_epoch_{epoch+1}")
        torch.save(checkpoint, f"{args.checkpoint_dir}/phase2_epoch_{epoch+1}.pt")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, f"{args.checkpoint_dir}/phase2_best.pt")
            generator.model.save_pretrained(f"{args.checkpoint_dir}/generator_best")
            print(f"  -> Saved best checkpoint (loss: {best_loss:.4f})")
    
    print("\n" + "=" * 60)
    print(f"Phase 2 training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="D-RAG Phase 2: Joint End-to-End Training"
    )
    
    # Data paths
    parser.add_argument("--heuristics_path", type=str, required=True,
                        help="Path to heuristics JSONL (same format as Phase 1)")
    parser.add_argument("--phase1_checkpoint", type=str, required=True,
                        help="Path to Phase 1 retriever checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_phase2",
                        help="Directory to save checkpoints")
    
    # Model architecture
    parser.add_argument("--node_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--relation_dim", type=int, default=256)
    parser.add_argument("--num_reasoning_steps", type=int, default=3)
    parser.add_argument("--generator_dim", type=int, default=2688,
                        help="Generator hidden dimension (2688 for Nemotron)")
    parser.add_argument("--generator_model", type=str, 
                        default="unsloth/Nemotron-3-Nano-30B-A3B-FP8",
                        help="Generator model ID")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of joint training epochs (paper: 5)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (small due to generator memory)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--ret_loss_weight", type=float, default=0.1,
                        help="Weight for retriever auxiliary loss")
    parser.add_argument("--k_facts", type=int, default=10,
                        help="Number of facts to select per question")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Gumbel-Softmax temperature")
    
    args = parser.parse_args()
    
    train_phase2(args)
