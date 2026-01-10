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

import unsloth  # must be imported before any transformers usage (Unsloth patches transformers)
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.model.generator import NemotronGenerator
from src.model.retriever import DRAGRetriever
from src.model.sampler import DifferentiableFactSampler
from src.model.projector import Projector
from src.data.kg_loader import SubgraphDataset
from src.utils.metrics import MetricsAccumulator, compute_retrieval_metrics


def collate_fn_train(batch):
    """
    Collate fn for *training*.
    We skip samples with 0 positives to avoid degenerate retriever supervision batches.
    """
    batch = [b for b in batch if b is not None and b.get("num_positive", 0) > 0]
    if len(batch) == 0:
        return None
    return batch


def collate_fn_val(batch):
    """
    Collate fn for *validation*.
    IMPORTANT: Do NOT drop num_positive==0 samples, otherwise validation/generation
    metrics get computed on a tiny (and biased) subset.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return batch


def validate_epoch(
    val_dataloader,
    retriever,
    projector,
    generator,
    args,
    device,
    run_generation: bool = True,
):
    """
    Run validation pass with full D-RAG metrics (Section 5.1, Appendix F).
    
    Metrics computed:
    - Answer Generation: Hits@1, EM, F1 (full dataset + retrieved subset)
    - Retrieval: Precision, Recall, F1
    - Loss: Generator loss, Retriever loss
    
    Args:
        val_dataloader: Validation data loader
        retriever: DRAGRetriever model
        projector: Projector model
        generator: NemotronGenerator model
        args: Training arguments
        device: Torch device
        run_generation: If True, run actual generation for Hits@1/F1 (slower but accurate)
    
    Returns:
        Dict with all metrics
    """
    retriever.eval()
    projector.eval()
    generator.model.eval()

    metrics_acc = MetricsAccumulator(ret_loss_weight=args.ret_loss_weight)
    forward_errors = 0
    logged_samples = []
    selected_counts = []
    gen_eval_count = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            if batch is None:
                continue

            for sample in batch:
                question = sample['question']
                answer = sample['answer']
                subgraph = sample['subgraph']
                rel_texts = sample['rel_texts']
                triples = sample.get('triples', [])
                retrieval_labels = sample['labels'].to(device)

                if not answer:
                    continue

                # Paper-style multi-answer formatting:
                # If the dataset provides multiple answers, train the generator to emit a bar-separated list.
                # Keep the original list for metrics, but use the joined string for supervised generation loss.
                answer_supervised = answer
                if isinstance(answer, list):
                    answer_supervised = " | ".join([str(a) for a in answer if str(a).strip()])
                
                # Paper-aligned prompt: the generator constructs the full scaffold including "Answer:"
                # We pass the raw question string here.
                question_prompt = question

                # Encode relations
                if len(rel_texts) > 0 and rel_texts[0]:
                    rel_embeds = retriever.encode_relations(rel_texts).to(device)
                else:
                    rel_embeds = torch.randn(len(rel_texts), args.relation_dim, device=device) * 0.02

                edge_relations = rel_embeds[subgraph.edge_type]

                # Retriever forward
                fact_probs, node_embeds, fact_embeds = retriever(
                    node_features=subgraph.x,
                    edge_index=subgraph.edge_index,
                    edge_attr=edge_relations,
                    edge_relations=edge_relations,
                    questions=[question],
                    fact_indices=None,
                )

                # --- Hybrid Selection Strategy (Appendix G: Inference) ---
                # Step 1: Top-k cap (paper uses 100)
                num_facts = fact_probs.shape[0]
                cap_k = min(args.max_facts_cap, num_facts)
                top_probs, top_indices = torch.topk(fact_probs, k=cap_k)
                
                # Step 2: Filter by probability threshold (paper uses 0.01)
                threshold_mask = top_probs >= args.prob_threshold
                selected_indices = top_indices[threshold_mask]
                selected_probs = top_probs[threshold_mask]
                
                # Fallback: if nothing passes threshold, use top-1
                if len(selected_indices) == 0:
                    selected_indices = top_indices[:1]
                    selected_probs = top_probs[:1]
                
                selected_counts.append(int(selected_indices.numel()))
                
                # --- Retrieval Metrics (computed on the selected set) ---
                # Create a binary mask of what was selected
                selected_mask = torch.zeros(num_facts, device=device)
                selected_mask[selected_indices] = 1.0
                ret_metrics = compute_retrieval_metrics(
                    selected_mask, retrieval_labels, threshold=0.5
                )
                metrics_acc.add_retrieval_result(ret_metrics)
                any_relevant_retrieved = ret_metrics['any_relevant_retrieved']

                # Get embeddings for selected facts
                selected_fact_embeds = fact_embeds[selected_indices]
                # Inference uses discrete selected facts (no soft weighting).
                weighted_fact_embeds = selected_fact_embeds

                # Project to generator space
                neural_prompt_embeds = projector(weighted_fact_embeds).unsqueeze(0)
                neural_prompt_embeds = neural_prompt_embeds.to(dtype=torch.bfloat16)

                # Build fact text list aligned to selected_indices (fact index == triple index).
                selected_fact_texts = []
                if triples:
                    for idx in selected_indices.detach().cpu().tolist():
                        if 0 <= idx < len(triples) and len(triples[idx]) >= 3:
                            h, r, t = triples[idx][0], triples[idx][1], triples[idx][2]
                            selected_fact_texts.append(f"{h}, {r}, {t}")
                        else:
                            selected_fact_texts.append("")

                # --- Generator Loss ---
                try:
                    outputs = generator.forward_paper_prompt(
                        neural_prompt_embeds,
                        questions=[question_prompt],
                        fact_texts=[selected_fact_texts],
                        answer_texts=[answer_supervised],
                        eos_loss_weight=args.eos_loss_weight,
                    )
                    gen_loss = outputs.loss.item()
                except Exception:
                    forward_errors += 1
                    continue

                # Retriever loss
                ret_loss, _, _ = retriever.compute_loss(fact_probs, retrieval_labels)
                metrics_acc.add_loss(gen_loss, ret_loss.item())

                # --- Generation Metrics (Hits@1, EM, F1) ---
                # Optional: limit how many val examples actually run free generation (speed knob).
                gen_limit = int(getattr(args, "val_generation_limit", 0) or 0)
                should_generate = run_generation and (gen_limit <= 0 or gen_eval_count < gen_limit)
                if should_generate:
                    try:
                        # Generate answer
                        generated = generator.generate(
                            neural_prompt_embeds,
                            questions=[question_prompt],
                            fact_texts=[selected_fact_texts],
                            max_new_tokens=args.val_max_new_tokens
                        )
                        prediction = generated[0] if generated else ""
                        
                        # Handle answer as list or string
                        ground_truths = answer if isinstance(answer, list) else [answer]
                        
                        metrics_acc.add_generation_result(
                            prediction, ground_truths, any_relevant_retrieved
                        )
                        gen_eval_count += 1

                        # Log a few samples for debugging in logs
                        if args.val_log_samples > 0 and len(logged_samples) < args.val_log_samples:
                            logged_samples.append({
                                "question": question,
                                "ground_truths": ground_truths,
                                "prediction": prediction,
                                "any_relevant_retrieved": bool(any_relevant_retrieved),
                                "num_selected": int(selected_indices.numel()),
                            })
                    except Exception as e:
                        # If generation fails, still count but with empty prediction
                        ground_truths = answer if isinstance(answer, list) else [answer]
                        metrics_acc.add_generation_result("", ground_truths, any_relevant_retrieved)
                        gen_eval_count += 1
                        if args.val_log_samples > 0 and len(logged_samples) < args.val_log_samples:
                            import traceback
                            logged_samples.append({
                                "question": question,
                                "ground_truths": ground_truths,
                                "prediction": "",
                                "any_relevant_retrieved": bool(any_relevant_retrieved),
                                "num_selected": int(selected_indices.numel()),
                                "error": "generation_failed",
                                "error_msg": repr(e),
                                "traceback": traceback.format_exc(limit=5),
                            })
                elif run_generation:
                    # Generation metrics skipped for speed (beyond gen_limit).
                    # Still keep retrieval+loss metrics for full validation.
                    pass

    # Restore train mode
    retriever.train()
    projector.train()
    generator.model.train()

    if forward_errors:
        print(f"  [val] WARNING: generator forward failed on {forward_errors} samples (skipped).")

    out = metrics_acc.get_metrics()
    if selected_counts:
        out["avg_num_selected"] = sum(selected_counts) / len(selected_counts)
        out["min_num_selected"] = min(selected_counts)
        out["max_num_selected"] = max(selected_counts)
    out["gen_eval_count"] = gen_eval_count
    if args.val_log_samples > 0:
        out["_samples"] = logged_samples
    return out


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
    print(f"Loaded {len(heuristics)} training examples.")
    
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
        collate_fn=collate_fn_train,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # 2b. Optional validation set
    val_dataloader = None
    if args.val_heuristics_path and os.path.exists(args.val_heuristics_path):
        print(f"Loading validation data from {args.val_heuristics_path}...")
        with open(args.val_heuristics_path, 'r', encoding='utf-8') as f:
            val_heuristics = [json.loads(line) for line in f if line.strip()]
        print(f"Loaded {len(val_heuristics)} validation examples.")
        
        val_dataset = SubgraphDataset(
            heuristics=val_heuristics,
            node_dim=args.node_dim,
            device=device
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_val,
            num_workers=args.num_workers,
            pin_memory=True if args.num_workers > 0 else False,
            persistent_workers=True if args.num_workers > 0 else False,
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
    generator = NemotronGenerator(
        model_id=args.generator_model,
        revision=args.generator_revision,
        use_lora=True,
    )

    # Simple CPU-side cache for relation embeddings to avoid recomputing encodings
    rel_embed_cache = {}
    
    # Verify gradient flow
    print("\nVerifying gradient flow through generator...")
    from src.model.generator import verify_gradient_flow
    gradient_ok = verify_gradient_flow(generator, projector, None)
    if not gradient_ok:
        print("WARNING: Gradient flow issue. Training may not work correctly.")
    
    # 7. Setup Optimizer
    # Paper-style: Retriever/GNN uses higher LR (e.g. 5e-5), LoRA uses lower LR (e.g. 1e-5).
    lora_lr = args.lora_lr if args.lora_lr is not None else (args.lr * args.lora_lr_scale)
    # Only train: retriever (fine-tune), projector, generator (LoRA)
    trainable_params = [
        {'params': retriever.parameters(), 'lr': args.lr},
        {'params': projector.parameters(), 'lr': args.lr},
        {'params': [p for p in generator.model.parameters() if p.requires_grad], 'lr': lora_lr},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)
    # CosineAnnealingLR uses a single eta_min across param groups; pick a conservative global minimum.
    min_lr = min(pg["lr"] for pg in optimizer.param_groups) * 0.01
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=min_lr
    )
    
    # Helpful LR debug: we have multiple param groups with different base LRs.
    # Note: CosineAnnealingLR is stepped once per epoch below, so LR is constant within an epoch.
    print(
        "Initial learning rates:"
        f" retriever={optimizer.param_groups[0]['lr']:.2e},"
        f" projector={optimizer.param_groups[1]['lr']:.2e},"
        f" lora={optimizer.param_groups[2]['lr']:.2e}"
    )
    
    # 8. Training Loop
    print(f"\n{'='*60}")
    print(f"Starting Phase 2 training for {args.epochs} epochs")
    print(f"  - Train size: {len(dataset)} samples")
    print(f"  - Val size: {len(val_dataloader.dataset) if val_dataloader else 0} samples")
    print(f"  - Val generation: {args.val_generation} (Hits@1/EM/F1)")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max facts cap: {args.max_facts_cap}, prob threshold: {args.prob_threshold}")
    print(f"  - Generator: {args.generator_model}")
    if args.use_grad_norm_balance:
        print(f"  - Loss weighting: PAPER-ALIGNED gradient-norm balancing (rho={args.rho})")
        print(f"    L = L_gen + rho * (||∇L_gen|| / ||∇L_ret||) * L_ret")
    else:
        print(f"  - Loss weighting: static (ret_loss_weight={args.ret_loss_weight})")
    print(f"{'='*60}\n")
    
    generator.model.train()
    retriever.train()
    projector.train()
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_gen_loss = 0.0
        epoch_ret_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue
            
            # Microbatch the generator forward pass to avoid per-sample overhead.
            batch_samples = len(batch)
            if batch_samples == 0:
                continue

            batch_loss_acc = 0.0
            batch_gen_acc = 0.0
            batch_ret_acc = 0.0
            mb_count = 0

            for mb_start in range(0, batch_samples, args.generator_microbatch):
                micro = batch[mb_start:mb_start + args.generator_microbatch]
                if len(micro) == 0:
                    continue

                questions_mb = []
                answers_mb = []
                fact_texts_mb = []
                prompt_embeds_mb = []
                ret_losses_mb = []

                for sample in micro:
                    question = sample['question']
                    answer = sample['answer']
                    subgraph = sample['subgraph']
                    rel_texts = sample['rel_texts']
                    triples = sample.get('triples', [])
                    retrieval_labels = sample['labels'].to(device)

                    if not answer:
                        continue

                    global_step += 1

                    answer_supervised = answer
                    if isinstance(answer, list):
                        answer_supervised = " | ".join([str(a) for a in answer if str(a).strip()])

                    # Relation embedding with cache
                    with torch.no_grad():
                        key = tuple(rel_texts) if rel_texts else None
                        if key and key in rel_embed_cache:
                            rel_embeds = rel_embed_cache[key].to(device)
                        else:
                            if rel_texts and rel_texts[0]:
                                rel_embeds = retriever.encode_relations(rel_texts).to(device)
                            else:
                                rel_embeds = torch.randn(len(rel_texts), args.relation_dim, device=device) * 0.02
                            if key:
                                rel_embed_cache[key] = rel_embeds.detach().cpu()

                    edge_relations = rel_embeds[subgraph.edge_type]

                    fact_probs, node_embeds, fact_embeds = retriever(
                        node_features=subgraph.x,
                        edge_index=subgraph.edge_index,
                        edge_attr=edge_relations,
                        edge_relations=edge_relations,
                        questions=[question],
                        fact_indices=None
                    )

                    fact_logits = torch.log(fact_probs + 1e-8) - torch.log(1 - fact_probs + 1e-8)
                    selection_mask = sampler(fact_logits)

                    num_facts = selection_mask.shape[0]
                    if num_facts > args.max_facts_cap:
                        _, top_cap_indices = torch.topk(fact_probs, k=args.max_facts_cap)
                        cap_mask = torch.zeros_like(selection_mask)
                        cap_mask[top_cap_indices] = 1.0
                        selection_mask = selection_mask * cap_mask

                    selected_indices = (selection_mask > 0.5).nonzero(as_tuple=True)[0]
                    if len(selected_indices) == 0:
                        selected_indices = torch.argmax(fact_probs, keepdim=True)

                    selected_fact_embeds = fact_embeds[selected_indices]
                    selected_mask_values = selection_mask[selected_indices].unsqueeze(-1)
                    selected_fact_embeds = selected_fact_embeds * selected_mask_values

                    neural_prompt_embeds = projector(selected_fact_embeds).unsqueeze(0)
                    neural_prompt_embeds = neural_prompt_embeds.to(dtype=torch.bfloat16)

                    selected_fact_texts = []
                    if triples:
                        for idx in selected_indices.detach().cpu().tolist():
                            if 0 <= idx < len(triples) and len(triples[idx]) >= 3:
                                h, r, t = triples[idx][0], triples[idx][1], triples[idx][2]
                                selected_fact_texts.append(f"{h}, {r}, {t}")
                            else:
                                selected_fact_texts.append("")

                    ret_loss, _, _ = retriever.compute_loss(fact_probs, retrieval_labels)

                    questions_mb.append(question)
                    answers_mb.append(answer_supervised)
                    fact_texts_mb.append(selected_fact_texts)
                    prompt_embeds_mb.append(neural_prompt_embeds)  # [1, K, H]
                    ret_losses_mb.append(ret_loss)

                if len(prompt_embeds_mb) == 0:
                    continue

                max_k = max(pe.shape[1] for pe in prompt_embeds_mb)
                gen_dim = prompt_embeds_mb[0].shape[-1]
                mb_size = len(prompt_embeds_mb)
                batch_prompts = torch.zeros((mb_size, max_k, gen_dim), device=device, dtype=prompt_embeds_mb[0].dtype)
                padded_fact_texts = []
                for i, pe in enumerate(prompt_embeds_mb):
                    k = pe.shape[1]
                    batch_prompts[i, :k, :] = pe.squeeze(0)
                    padded = fact_texts_mb[i] + [""] * (max_k - len(fact_texts_mb[i]))
                    padded_fact_texts.append(padded)

                try:
                    outputs = generator.forward_paper_prompt(
                        batch_prompts,
                        questions=questions_mb,
                        fact_texts=padded_fact_texts,
                        answer_texts=answers_mb,
                        eos_loss_weight=args.eos_loss_weight,
                    )
                    gen_loss = outputs.loss
                except Exception as e:
                    print(f"Generator error: {e}")
                    continue

                ret_loss_mean = torch.stack(ret_losses_mb).mean() if len(ret_losses_mb) > 0 else torch.tensor(0.0, device=device)

                loss = gen_loss + args.ret_loss_weight * ret_loss_mean

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(retriever.parameters()) + list(projector.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                batch_loss_acc += loss.item()
                batch_gen_acc += gen_loss.item()
                batch_ret_acc += ret_loss_mean.item()
                mb_count += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'gen': f'{gen_loss.item():.4f}',
                    'ret': f'{ret_loss_mean.item():.4f}',
                    'lr': f"{optimizer.param_groups[1]['lr']:.2e}",
                })

            if mb_count == 0:
                continue

            epoch_loss += batch_loss_acc / mb_count
            epoch_gen_loss += batch_gen_acc / mb_count
            epoch_ret_loss += batch_ret_acc / mb_count
            num_batches += 1
        
        # Epoch statistics
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_gen = epoch_gen_loss / max(num_batches, 1)
        avg_ret = epoch_ret_loss / max(num_batches, 1)
        
        scheduler.step()
        lrs = scheduler.get_last_lr()
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_loss:.4f} (Gen: {avg_gen:.4f}, Ret: {avg_ret:.4f}) | "
              f"LRs: ret={lrs[0]:.2e}, proj={lrs[1]:.2e}, gen={lrs[2]:.2e}")
        
        # --- Validation ---
        val_metrics = {}
        if val_dataloader is not None:
            val_metrics = validate_epoch(
                val_dataloader, retriever, projector, generator, args, device,
                run_generation=args.val_generation
            )
            
            # Print validation metrics in a structured format
            print(f"\n  --- Validation Metrics ---")
            
            # Loss metrics
            if 'gen_loss' in val_metrics:
                print(f"  Loss: {val_metrics.get('combined_loss', 0):.4f} "
                      f"(Gen: {val_metrics['gen_loss']:.4f}, Ret: {val_metrics['ret_loss']:.4f})")
            
            # Retrieval metrics
            if 'ret_precision' in val_metrics:
                print(f"  Retrieval: P={val_metrics['ret_precision']:.4f}, "
                      f"R={val_metrics['ret_recall']:.4f}, F1={val_metrics['ret_f1']:.4f}")
            
            # Generation metrics (full dataset)
            if 'hits@1' in val_metrics:
                print(f"  Generation (Full): Hits@1={val_metrics['hits@1']:.4f}, "
                      f"EM={val_metrics['em']:.4f}, F1={val_metrics['gen_f1']:.4f}")
            
            # Generation metrics (retrieved subset)
            if 'hits@1_retrieved' in val_metrics:
                ratio = val_metrics.get('retrieved_ratio', 0) * 100
                print(f"  Generation (Retrieved {ratio:.1f}%): "
                      f"Hits@1={val_metrics['hits@1_retrieved']:.4f}, "
                      f"EM={val_metrics['em_retrieved']:.4f}, F1={val_metrics['gen_f1_retrieved']:.4f}")
            if 'avg_num_selected' in val_metrics:
                print(f"  Selected facts: avg={val_metrics['avg_num_selected']:.1f}, "
                      f"min={val_metrics.get('min_num_selected', 0)}, max={val_metrics.get('max_num_selected', 0)}")
            extra_gen = ""
            if args.val_generation:
                gen_limit = int(getattr(args, "val_generation_limit", 0) or 0)
                gen_eval_count = int(val_metrics.get("gen_eval_count", 0) or 0)
                if gen_limit > 0:
                    extra_gen = f", gen_eval={gen_eval_count}/{gen_limit}"
            print(
                f"  (max_cap={args.max_facts_cap}, threshold={args.prob_threshold}, "
                f"max_new_tokens={args.val_max_new_tokens}{extra_gen})"
            )

            # Debug samples
            samples = val_metrics.get("_samples") if isinstance(val_metrics, dict) else None
            if samples:
                print("  Sample predictions:")
                for i, s in enumerate(samples[:args.val_log_samples]):
                    q = (s.get("question", "") or "").strip().replace("\n", " ")
                    pred = (s.get("prediction", "") or "").strip().replace("\n", " ")
                    gts = s.get("ground_truths", [])
                    gts_str = " | ".join(str(x) for x in gts[:3])
                    extra = ""
                    if "num_selected" in s:
                        extra = f" (selected={s['num_selected']})"
                    if "error" in s:
                        extra += f" [error={s['error']}]"
                    print(f"   - Q: {q[:160]}{extra}")
                    print(f"     GT: {gts_str[:160]}")
                    print(f"     Pred: {pred[:160]}")
                    if "error_msg" in s:
                        print(f"     Err: {str(s['error_msg'])[:200]}")
                    if "traceback" in s:
                        tb = (s.get("traceback") or "").strip().replace("\n", " | ")
                        print(f"     TB: {tb[:240]}")
        
        # Use val combined_loss for best checkpoint if available, else train loss.
        # IMPORTANT: If validation failed to compute loss, do NOT default to 0.
        val_combined = (
            val_metrics['combined_loss']
            if (val_metrics and 'combined_loss' in val_metrics)
            else float('inf')
        )
        metric_for_best = val_combined if val_dataloader is not None else avg_loss
        
        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'retriever_state_dict': retriever.state_dict(),
            'projector_state_dict': projector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_metrics': val_metrics if val_dataloader else None,
            'args': vars(args)
        }
        
        # Save generator LoRA weights separately
        generator.model.save_pretrained(f"{args.checkpoint_dir}/generator_epoch_{epoch+1}")
        torch.save(checkpoint, f"{args.checkpoint_dir}/phase2_epoch_{epoch+1}.pt")
        
        if metric_for_best < best_loss:
            best_loss = metric_for_best
            torch.save(checkpoint, f"{args.checkpoint_dir}/phase2_best.pt")
            generator.model.save_pretrained(f"{args.checkpoint_dir}/generator_best")
            print(f"  -> Saved best checkpoint ({'val' if val_dataloader else 'train'} loss: {best_loss:.4f})")
    
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
    parser.add_argument("--val_heuristics_path", type=str, default=None,
                        help="Optional path to validation heuristics JSONL for per-epoch validation")
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
                        default="unsloth/Nemotron-3-Nano-30B-A3B",
                        help="Generator model ID (default uses non-FP8 variant for Unsloth stability)")
    parser.add_argument(
        "--generator_revision",
        type=str,
        # "Previous-to-last" commit on the non-FP8 Nemotron repo (README update),
        # but crucially avoids the FP8 repo's current uninitialized-weight failure in Unsloth.
        default="4e73921165feb0016e8e1d48262910ed060473e2",
        help=(
            "Pin the generator repo revision to avoid breaking changes when trust_remote_code files update. "
            "Default pins a known-good commit for this project."
        ),
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of joint training epochs (paper: 5)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (small due to generator memory)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--rho", type=float, default=0.9,
                        help="Paper-aligned gradient-norm balancing parameter (paper: 0.9). "
                             "Used when --use_grad_norm_balance is set.")
    parser.add_argument("--ret_loss_weight", type=float, default=0.1,
                        help="Static weight for retriever auxiliary loss (used when --use_grad_norm_balance is False)")
    parser.add_argument(
        "--use_grad_norm_balance",
        action="store_true",
        help="Use paper-aligned dynamic gradient-norm balancing: L = L_gen + rho * (||∇L_gen||/||∇L_ret||) * L_ret. "
             "If False, uses static --ret_loss_weight instead.",
    )
    parser.add_argument(
        "--eos_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Optional: upweight EOS token cross-entropy during generator training (>=1.0). "
            "Helps the model learn to stop promptly under greedy decoding without changing the objective."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Gumbel-Softmax temperature")

    # Optimizer detail (Appendix G): LoRA LR is lower than retriever/projector LR.
    parser.add_argument("--lora_lr", type=float, default=None,
                        help="Optional explicit LoRA LR (paper: 1e-5). If not set, uses --lr * --lora_lr_scale.")
    parser.add_argument("--lora_lr_scale", type=float, default=0.2,
                        help="Scale factor for LoRA LR when --lora_lr is not set (default 0.2 => 1e-5 when --lr=5e-5).")
    
    # Retrieval strategy (Section 4.3.3, Appendix G)
    parser.add_argument("--max_facts_cap", type=int, default=100,
                        help="Max facts cap for efficiency (paper: 100). Binary selection happens within this cap.")
    parser.add_argument("--prob_threshold", type=float, default=0.01,
                        help="Probability threshold for inference/validation (paper: 0.01). Facts below this are filtered.")
    
    # Validation options
    parser.add_argument("--val_generation", action="store_true",
                        help="Run generation during validation for Hits@1/EM/F1 (slower but more metrics)")
    parser.add_argument("--val_max_new_tokens", type=int, default=50,
                        help="Max new tokens for validation generation")
    parser.add_argument(
        "--val_generation_limit",
        type=int,
        default=0,
        help="If >0, only run free generation on the first N val examples (speed knob). "
             "Loss and retrieval metrics still computed on all val examples.",
    )
    parser.add_argument("--val_log_samples", type=int, default=3,
                        help="Log N example predictions during validation (0 to disable)")

    # DataLoader workers (CPU parallelism)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes for loading/building per-sample subgraphs. "
             "Set >0 to use multiple CPU cores (each worker has its own cache/state).",
    )

    # Efficiency knob: microbatch the generator forward pass
    parser.add_argument(
        "--generator_microbatch",
        type=int,
        default=4,
        help="Process this many samples at a time in the generator forward pass. "
             "Larger values reduce Python overhead; keep small enough to fit GPU memory.",
    )

    # Diagnostics: gradient-norm imbalance probe (paper-style weighting insight)
    parser.add_argument(
        "--log_grad_norms",
        action="store_true",
        help="Periodically log ||dL_gen/dlogits|| and ||dL_ret/dlogits|| (proxy for weighting imbalance).",
    )
    parser.add_argument(
        "--grad_norm_log_interval",
        type=int,
        default=200,
        help="Log grad-norm diagnostics every N training samples (default: 200).",
    )
    
    args = parser.parse_args()
    
    train_phase2(args)
