# IMPORTANT: Unsloth must be imported FIRST before torch/transformers
from unsloth import FastLanguageModel

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformer_engine.pytorch import fp8_autocast
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

class NemotronGenerator(nn.Module):
    """
    Nemotron-3-Nano wrapper with proper handling for Mamba hybrid architecture.
    
    Key considerations for Mamba backward pass:
    1. Mamba layers process sequentially - prefix tuning works because early tokens
       influence later ones (correct causal direction)
    2. Label alignment must account for prepended neural prompts
    3. Gradient checkpointing with Mamba requires careful state management
    """
    def __init__(
        self,
        model_id: str = "unsloth/Nemotron-3-Nano-30B-A3B",
        revision: str | None = None,
        use_lora: bool = True,
    ):
        super(NemotronGenerator, self).__init__()
        
        rev_str = f" (revision={revision})" if revision else ""
        print(f"Loading Unsloth model and tokenizer: {model_id}{rev_str}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id,
            revision = revision,
            max_seq_length = 2048,
            load_in_4bit = False,
            load_in_8bit = False,
            trust_remote_code = True,
            # Note: Disable unsloth_force_compile for custom inputs_embeds compatibility
            # The Triton kernels may not handle custom embedding injection correctly
            unsloth_force_compile = False,
            attn_implementation = "eager",  # Safer for custom inputs_embeds
        )
        
        # Align tokenizer special tokens with model config.
        # For this Nemotron checkpoint, config typically uses:
        # - eos_token_id = 2 (</s>)
        # - pad_token_id = 999 (<SPECIAL_999>)
        # The tokenizer may default to chat-style <|im_end|> as eos; that can break stopping.
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            try:
                if self.tokenizer.eos_token_id != cfg.eos_token_id:
                    eos_tok = self.tokenizer.convert_ids_to_tokens(cfg.eos_token_id)
                    self.tokenizer.eos_token = eos_tok
                if self.tokenizer.pad_token_id != cfg.pad_token_id:
                    pad_tok = self.tokenizer.convert_ids_to_tokens(cfg.pad_token_id)
                    self.tokenizer.pad_token = pad_tok

                # Keep config in sync (defensive)
                self.model.config.eos_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                # Avoid cache-related warnings / dynamic cache requirements during training & validation.
                # We explicitly pass use_cache=False in forward/generate, but keep config consistent too.
                if hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
            except Exception:
                # If tokenizer doesn't support setting, fall back to passing explicit ids at generate-time.
                pass

        # --- Startup sanity check for stopping tokens (debugging empty / repetitive generations) ---
        try:
            cfg = getattr(self.model, "config", None)
            cfg_eos = getattr(cfg, "eos_token_id", None) if cfg is not None else None
            cfg_pad = getattr(cfg, "pad_token_id", None) if cfg is not None else None
            tok_eos = getattr(self.tokenizer, "eos_token_id", None)
            tok_pad = getattr(self.tokenizer, "pad_token_id", None)
            tok_eos_str = getattr(self.tokenizer, "eos_token", None)
            tok_pad_str = getattr(self.tokenizer, "pad_token", None)
            print(
                "[generator_tokens] "
                f"tokenizer.eos_token_id={tok_eos} eos_token={repr(tok_eos_str)} | "
                f"model.config.eos_token_id={cfg_eos} | "
                f"tokenizer.pad_token_id={tok_pad} pad_token={repr(tok_pad_str)} | "
                f"model.config.pad_token_id={cfg_pad}"
            )
            if cfg_eos is not None and tok_eos is not None and int(cfg_eos) != int(tok_eos):
                print(
                    "[generator_tokens] WARNING: eos_token_id mismatch between tokenizer and model config. "
                    "This can cause immediate stopping, never-stopping, or empty decoded outputs."
                )
            if cfg_pad is not None and tok_pad is not None and int(cfg_pad) != int(tok_pad):
                print(
                    "[generator_tokens] WARNING: pad_token_id mismatch between tokenizer and model config. "
                    "This can cause empty outputs if PAD is generated and then stripped."
                )
        except Exception as e:
            print(f"[generator_tokens] WARNING: failed to print token-id sanity check: {repr(e)}")
        
        if use_lora:
            print("Applying Unsloth optimized LoRA...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = 64,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj",
                                 "in_proj", "out_proj"],
                lora_alpha = 128,
                lora_dropout = 0,
                bias = "none",
                # WARNING: "unsloth" checkpointing may have issues with custom inputs_embeds
                # Use standard gradient checkpointing or disable for initial testing
                use_gradient_checkpointing = True,  # Standard PyTorch checkpointing
                random_state = 3407,
            )
            self.model.print_trainable_parameters()

        # CPU-side tokenization caches to avoid re-tokenizing every step.
        # They store tensors on CPU; we move them to device inside forward_paper_prompt.
        self.prefix_cache = {}
        self.fact_cache = {}

    def forward(self, neural_prompt_embeds, questions, answer_texts=None):
        """
        Forward pass using the paper-style D-RAG prompt:

        Answer the question based on the provided facts.

        Question: {question}

        Provided facts:

        [Projected_GNN_Vector] {head}, {relation}, {tail}
        [Projected_GNN_Vector] {head}, {relation}, {tail}
        ...

        Answer:

        This implements the "Neural Fact Prompt" described in the paper (Figure 5):
        interleaving one structural embedding per fact with the semantic text of the triple.
        
        Args:
            neural_prompt_embeds: [batch_size, k_facts, hidden_size] - projected fact embeddings
            questions: List of question strings (raw question text; do NOT append "Answer:")
            answer_texts: List of answer strings (for training)
        
        Returns:
            CausalLMOutputWithPast with properly computed loss
        """
        raise TypeError(
            "NemotronGenerator.forward signature changed: call forward_paper_prompt("
            "neural_fact_embeds, questions, fact_texts, answer_texts=...) instead."
        )

    def forward_paper_prompt(
        self,
        neural_fact_embeds,
        questions,
        fact_texts,
        answer_texts=None,
        eos_loss_weight: float = 1.0,
    ):
        """
        Paper-style forward pass with interleaved structural + semantic fact prompts.

        Args:
            neural_fact_embeds: [B, K, H] projected fact embeddings (one vector per selected fact)
            questions: List[str] length B (raw question text)
            fact_texts: List[List[str]] length B, each inner list length K (text triples)
            answer_texts: Optional[List[str]] length B (supervised answer text)
        """
        device = self.model.device
        batch_size = neural_fact_embeds.shape[0]

        if len(questions) != batch_size:
            raise ValueError(f"questions length {len(questions)} != batch size {batch_size}")
        if len(fact_texts) != batch_size:
            raise ValueError(f"fact_texts length {len(fact_texts)} != batch size {batch_size}")
        if answer_texts is not None and len(answer_texts) != batch_size:
            raise ValueError(f"answer_texts length {len(answer_texts)} != batch size {batch_size}")
        
        # Verify gradient flow during training only.
        if torch.is_grad_enabled() and answer_texts is not None:
            assert neural_fact_embeds.requires_grad, (
                "neural_fact_embeds must have requires_grad=True for D-RAG training!"
            )

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None; cannot train/stop properly.")
        if pad_id is None:
            pad_id = eos_id

        embed_layer = self.model.get_input_embeddings()
        pad_embed = embed_layer(torch.tensor([pad_id], device=device)).squeeze(0)  # [1, H]

        # Paper prompt scaffold (Appendix C, Figure 5)
        # The prompt ends with "Answer:" on its own line (no trailing space).
        # The supervised answer starts with a leading space for correct tokenization boundary.
        instr = "Answer the question based on the provided facts.\n\n"
        provided_facts_header = "Provided facts:\n\n"
        answer_header = "\nAnswer:"

        sample_embeds = []
        sample_attn = []
        sample_labels = []

        for i in range(batch_size):
            q = questions[i]
            facts_i = fact_texts[i] or []

            # Ensure we only use as many fact vectors as we have fact strings.
            k_vec = int(neural_fact_embeds.shape[1])
            k = min(k_vec, len(facts_i))

            # Prefix text: instruction + question + provided facts header
            prefix_text = f"{instr}Question: {q}\n\n{provided_facts_header}"
            prefix_ids = self.prefix_cache.get(prefix_text)
            if prefix_ids is None:
                prefix_ids = torch.tensor(
                    self.tokenizer(prefix_text, add_special_tokens=False).input_ids,
                    dtype=torch.long,
                    device="cpu",
                )
                self.prefix_cache[prefix_text] = prefix_ids
            # Add BOS once at the beginning of the entire prompt if available.
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                # If cached tensor already has BOS, skip; else prepend.
                if prefix_ids.numel() == 0 or prefix_ids[0].item() != bos_id:
                    prefix_ids = torch.cat(
                        [torch.tensor([bos_id], dtype=torch.long, device="cpu"), prefix_ids],
                        dim=0,
                    )
                    self.prefix_cache[prefix_text] = prefix_ids
            prefix_ids_t = prefix_ids.to(device=device, dtype=torch.long)
            prefix_emb = embed_layer(prefix_ids_t).unsqueeze(0) if prefix_ids_t.numel() > 0 else torch.empty((1, 0, pad_embed.shape[-1]), device=device)

            parts_emb = [prefix_emb]  # list of [1, L, H]
            parts_lab = [torch.full((prefix_emb.shape[1],), -100, device=device, dtype=torch.long)]

            # Interleave each selected fact: [Projected_GNN_Vector] + tokens("{h}, {r}, {t}\n")
            for j in range(k):
                # Structural token (one vector)
                # IMPORTANT: keep 3D shape [1, 1, H] for concatenation with [1, L, H]
                # (Indexing with a single integer would drop the batch dimension.)
                struct = neural_fact_embeds[i:i+1, j:j+1, :]  # [1, 1, H]
                # Semantic text tokens for the triple
                fact_line = f"{str(facts_i[j]).strip()}\n"
                cached_fact = self.fact_cache.get(fact_line)
                if cached_fact is None:
                    cached_fact = torch.tensor(
                        self.tokenizer(fact_line, add_special_tokens=False).input_ids,
                        dtype=torch.long,
                        device="cpu",
                    )
                    self.fact_cache[fact_line] = cached_fact
                fact_ids_t = cached_fact.to(device=device, dtype=torch.long)
                fact_emb = embed_layer(fact_ids_t).unsqueeze(0) if fact_ids_t.numel() > 0 else torch.empty((1, 0, pad_embed.shape[-1]), device=device)

                parts_emb.append(struct)
                parts_lab.append(torch.full((1,), -100, device=device, dtype=torch.long))
                parts_emb.append(fact_emb)
                parts_lab.append(torch.full((fact_emb.shape[1],), -100, device=device, dtype=torch.long))
        
            # Add the answer header ("Answer:\n")
            ans_hdr_ids = self.tokenizer(answer_header, add_special_tokens=False).input_ids
            ans_hdr_ids_t = torch.tensor(ans_hdr_ids, device=device, dtype=torch.long)
            ans_hdr_emb = embed_layer(ans_hdr_ids_t).unsqueeze(0) if ans_hdr_ids_t.numel() > 0 else torch.empty((1, 0, pad_embed.shape[-1]), device=device)
            parts_emb.append(ans_hdr_emb)
            parts_lab.append(torch.full((ans_hdr_emb.shape[1],), -100, device=device, dtype=torch.long))

            # Answer tokens (supervised region): answer + EOS
            # Paper-aligned: prepend a leading space so tokenization matches what follows "Answer:"
            if answer_texts is not None:
                ans_text = str(answer_texts[i])
                # Prepend space for correct tokenization boundary (the prompt ends with "Answer:")
                ans_text_with_space = " " + ans_text if not ans_text.startswith(" ") else ans_text
                ans_ids = self.tokenizer(ans_text_with_space, add_special_tokens=False).input_ids
                ans_ids = ans_ids + [eos_id]
                ans_ids_t = torch.tensor(ans_ids, device=device, dtype=torch.long)
                ans_emb = embed_layer(ans_ids_t).unsqueeze(0)
                parts_emb.append(ans_emb)
                parts_lab.append(ans_ids_t.clone())

                # === EOS Supervision Diagnostic (enable with DRAG_DEBUG_EOS=1) ===
                import os
                if os.environ.get("DRAG_DEBUG_EOS", "") == "1" and i == 0 and not getattr(self, "_eos_debug_printed", False):
                    self._eos_debug_printed = True
                    try:
                        decoded_ans = self.tokenizer.decode(ans_ids, skip_special_tokens=False)
                        supervised_ids = [x.item() for x in ans_ids_t]
                        print(f"\n[EOS_DEBUG] === EOS Supervision Check ===")
                        print(f"[EOS_DEBUG] answer_text (raw): {repr(ans_text)}")
                        print(f"[EOS_DEBUG] answer_text (with space): {repr(ans_text_with_space)}")
                        print(f"[EOS_DEBUG] tokenized ans_ids (before EOS): {self.tokenizer(ans_text_with_space, add_special_tokens=False).input_ids}")
                        print(f"[EOS_DEBUG] tokenized ans_ids (with EOS):   {supervised_ids}")
                        print(f"[EOS_DEBUG] eos_id={eos_id}, last label id={supervised_ids[-1]}, EOS at end? {supervised_ids[-1] == eos_id}")
                        print(f"[EOS_DEBUG] decoded (with specials): {repr(decoded_ans)}")
                        print(f"[EOS_DEBUG] num_supervised_tokens={len(supervised_ids)}")
                        print(f"[EOS_DEBUG] ================================\n")
                    except Exception as e:
                        print(f"[EOS_DEBUG] failed: {repr(e)}")

            # Concat all parts
            emb_i = torch.cat(parts_emb, dim=1)  # [1, L, H]
            lab_i = torch.cat(parts_lab, dim=0)  # [L]

            # Cast structural embeddings to match token embedding dtype (preserves grads).
            if emb_i.dtype != pad_embed.dtype:
                emb_i = emb_i.type_as(pad_embed)

            attn_i = torch.ones((emb_i.shape[1],), device=device, dtype=torch.long)

            sample_embeds.append(emb_i.squeeze(0))  # [L, H]
            sample_attn.append(attn_i)
            sample_labels.append(lab_i)

        # Pad to a batch
        max_len = max(e.shape[0] for e in sample_embeds) if sample_embeds else 0
        hidden = pad_embed.shape[-1]

        batch_embeds = torch.empty((batch_size, max_len, hidden), device=device, dtype=pad_embed.dtype)
        batch_attn = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
        batch_labels = torch.full((batch_size, max_len), -100, device=device, dtype=torch.long)
            
        # Initialize with pad embedding (so masked pads don't inject garbage).
        batch_embeds[:] = pad_embed.to(dtype=batch_embeds.dtype)

        for i in range(batch_size):
            L = sample_embeds[i].shape[0]
            batch_embeds[i, :L, :] = sample_embeds[i].to(dtype=batch_embeds.dtype)
            batch_attn[i, :L] = sample_attn[i]
            batch_labels[i, :L] = sample_labels[i]

        if answer_texts is None:
            batch_labels = None

        # === Full Label Tensor Diagnostic (enable with DRAG_DEBUG_EOS=1) ===
        import os
        if os.environ.get("DRAG_DEBUG_EOS", "") == "1" and batch_labels is not None and not getattr(self, "_label_debug_printed", False):
            self._label_debug_printed = True
            try:
                # Analyze first sample's labels
                labs = batch_labels[0].cpu().tolist()
                masked = sum(1 for x in labs if x == -100)
                supervised = sum(1 for x in labs if x != -100)
                # Find the supervised region
                supervised_ids = [x for x in labs if x != -100]
                supervised_decoded = self.tokenizer.decode(supervised_ids, skip_special_tokens=False) if supervised_ids else ""
                print(f"\n[LABEL_DEBUG] === Full Label Tensor Check (sample 0) ===")
                print(f"[LABEL_DEBUG] total_len={len(labs)}, masked(-100)={masked}, supervised={supervised}")
                print(f"[LABEL_DEBUG] supervised_ids={supervised_ids[:20]}{'...' if len(supervised_ids) > 20 else ''}")
                print(f"[LABEL_DEBUG] supervised_decoded: {repr(supervised_decoded[:100])}{'...' if len(supervised_decoded) > 100 else ''}")
                print(f"[LABEL_DEBUG] last_supervised_id={supervised_ids[-1] if supervised_ids else None}, eos_id={self.tokenizer.eos_token_id}")
                print(f"[LABEL_DEBUG] ==============================================\n")
            except Exception as e:
                print(f"[LABEL_DEBUG] failed: {repr(e)}")

        use_weighted_eos = (
            batch_labels is not None
            and eos_loss_weight is not None
            and float(eos_loss_weight) != 1.0
        )
        labels_for_model = None if use_weighted_eos else batch_labels

        if TE_AVAILABLE:
            with fp8_autocast(enabled=True):
                outputs = self.model(
                    inputs_embeds=batch_embeds,
                    attention_mask=batch_attn,
                    labels=labels_for_model,
                    use_cache=False,
                    return_dict=True,
                )
        else:
            outputs = self.model(
                inputs_embeds=batch_embeds,
                attention_mask=batch_attn,
                labels=labels_for_model,
                use_cache=False,
                return_dict=True,
            )

        # Optional: upweight EOS token positions to teach the model to stop promptly.
        # This keeps the objective as standard token-level cross-entropy, just reweighted.
        if use_weighted_eos:
            logits = outputs.logits  # [B, L, V]
            # Align with Transformers' internal causal shift: predict token t using logits at t-1.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch_labels[:, 1:].contiguous()
            vocab = shift_logits.shape[-1]

            per_tok = F.cross_entropy(
                shift_logits.view(-1, vocab),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.shape[0], shift_labels.shape[1])

            weights = torch.ones_like(shift_labels, dtype=per_tok.dtype, device=per_tok.device)
            weights = weights.masked_fill(shift_labels.eq(-100), 0.0)
            if eos_id is not None:
                weights = weights + shift_labels.eq(int(eos_id)).to(per_tok.dtype) * (float(eos_loss_weight) - 1.0)

            denom = weights.sum().clamp_min(1.0)
            weighted_loss = (per_tok * weights).sum() / denom
            outputs.loss = weighted_loss
        
        return outputs

    def generate(self, neural_fact_embeds, questions, fact_texts, max_new_tokens=100):
        """
        Paper-style generation with interleaved structural + semantic fact prompts.
        
        Uses a custom autoregressive loop to properly handle the hybrid inputs_embeds
        (structural embeddings + token embeddings). This fixes issues where the standard
        Transformers generate() doesn't correctly continue after the initial inputs_embeds.
        """
        # Unsloth models often require switching to inference mode for generate().
        try:
            self.model = FastLanguageModel.for_inference(self.model)
        except Exception:
            pass

        device = self.model.device
        batch_size = neural_fact_embeds.shape[0]
        if len(questions) != batch_size:
            raise ValueError(f"questions length {len(questions)} != batch size {batch_size}")
        if len(fact_texts) != batch_size:
            raise ValueError(f"fact_texts length {len(fact_texts)} != batch size {batch_size}")

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None; cannot stop properly.")
        if pad_id is None:
            pad_id = eos_id

        embed_layer = self.model.get_input_embeddings()
        pad_embed = embed_layer(torch.tensor([pad_id], device=device)).squeeze(0)  # [H]

        # Paper prompt scaffold (must match training exactly)
        instr = "Answer the question based on the provided facts.\n\n"
        provided_facts_header = "Provided facts:\n\n"
        answer_header = "\nAnswer:"

        # Build prompt embeddings for each sample
        sample_embeds = []
        sample_lengths = []
        for i in range(batch_size):
            q = questions[i]
            facts_i = fact_texts[i] or []

            k_vec = int(neural_fact_embeds.shape[1])
            k = min(k_vec, len(facts_i))

            prefix_text = f"{instr}Question: {q}\n\n{provided_facts_header}"
            prefix_ids = self.tokenizer(prefix_text, add_special_tokens=False).input_ids
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None:
                prefix_ids = [bos_id] + prefix_ids
            prefix_ids_t = torch.tensor(prefix_ids, device=device, dtype=torch.long)
            prefix_emb = embed_layer(prefix_ids_t).unsqueeze(0) if prefix_ids_t.numel() > 0 else torch.empty((1, 0, pad_embed.shape[-1]), device=device)

            parts = [prefix_emb]
            for j in range(k):
                # Keep 3D shape for concatenation.
                struct = neural_fact_embeds[i:i+1, j:j+1, :]  # [1, 1, H]
                fact_line = f"{str(facts_i[j]).strip()}\n"
                fact_ids = self.tokenizer(fact_line, add_special_tokens=False).input_ids
                fact_ids_t = torch.tensor(fact_ids, device=device, dtype=torch.long)
                fact_emb = embed_layer(fact_ids_t).unsqueeze(0) if fact_ids_t.numel() > 0 else torch.empty((1, 0, pad_embed.shape[-1]), device=device)
                parts.append(struct)
                parts.append(fact_emb)

            ans_hdr_ids = self.tokenizer(answer_header, add_special_tokens=False).input_ids
            ans_hdr_ids_t = torch.tensor(ans_hdr_ids, device=device, dtype=torch.long)
            ans_hdr_emb = embed_layer(ans_hdr_ids_t).unsqueeze(0) if ans_hdr_ids_t.numel() > 0 else torch.empty((1, 0, pad_embed.shape[-1]), device=device)
            parts.append(ans_hdr_emb)

            emb_i = torch.cat(parts, dim=1)  # [1, L, H]
            if emb_i.dtype != pad_embed.dtype:
                emb_i = emb_i.type_as(pad_embed)
            sample_embeds.append(emb_i.squeeze(0))  # [L, H]
            sample_lengths.append(emb_i.shape[1])
        
        # Pad to batch
        max_prompt_len = max(sample_lengths) if sample_lengths else 0
        hidden = pad_embed.shape[-1]
        
        # Initialize with prompt embeddings
        batch_embeds = torch.zeros((batch_size, max_prompt_len, hidden), device=device, dtype=pad_embed.dtype)
        batch_embeds[:] = pad_embed.to(dtype=batch_embeds.dtype)
        batch_attn = torch.zeros((batch_size, max_prompt_len), device=device, dtype=torch.long)
        
        for i in range(batch_size):
            L = sample_lengths[i]
            batch_embeds[i, :L, :] = sample_embeds[i].to(dtype=batch_embeds.dtype)
            batch_attn[i, :L] = 1

        # Custom autoregressive generation loop
        # This properly handles the hybrid inputs_embeds by manually appending
        # newly generated token embeddings to the context
        generated_tokens = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        
        current_embeds = batch_embeds.clone()
        current_attn = batch_attn.clone()
        
        for step in range(max_new_tokens):
            if all(finished):
                break
            
            # Forward pass
            with torch.no_grad():
                if TE_AVAILABLE:
                    with fp8_autocast(enabled=True):
                        outputs = self.model(
                            inputs_embeds=current_embeds,
                            attention_mask=current_attn,
                            use_cache=False,
                            return_dict=True,
                        )
                else:
                    outputs = self.model(
                        inputs_embeds=current_embeds,
                        attention_mask=current_attn,
                        use_cache=False,
                        return_dict=True,
                    )
            
            # Get logits for the last position of each sequence
            # Need to find the actual last position (not padding)
            logits = outputs.logits  # [batch, seq_len, vocab]
            
            next_tokens = []
            for i in range(batch_size):
                if finished[i]:
                    next_tokens.append(pad_id)
                    continue
                
                # Find the last non-padded position
                seq_len = int(current_attn[i].sum().item())
                last_logits = logits[i, seq_len - 1, :]  # [vocab]
                
                # Greedy: take argmax
                next_token = int(last_logits.argmax().item())
                
                # Block PAD token from being generated
                if next_token == pad_id and pad_id != eos_id:
                    # Get second-best token
                    last_logits[pad_id] = float('-inf')
                    next_token = int(last_logits.argmax().item())
                
                next_tokens.append(next_token)
                generated_tokens[i].append(next_token)
                
                # Check for EOS
                if next_token == eos_id:
                    finished[i] = True
            
            # Append new token embeddings to the context
            next_tokens_t = torch.tensor(next_tokens, device=device, dtype=torch.long)
            next_embeds = embed_layer(next_tokens_t).unsqueeze(1)  # [batch, 1, H]
            next_embeds = next_embeds.to(dtype=current_embeds.dtype)
            
            # Expand current embeddings and attention mask
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            new_attn = torch.ones((batch_size, 1), device=device, dtype=torch.long)
            for i in range(batch_size):
                if finished[i]:
                    new_attn[i, 0] = 0  # Don't attend to padding for finished sequences
            current_attn = torch.cat([current_attn, new_attn], dim=1)

        # Decode generated tokens
        decoded = []
        for i in range(batch_size):
            tokens = generated_tokens[i]
            # Remove EOS if present at the end
            if tokens and tokens[-1] == eos_id:
                tokens = tokens[:-1]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            decoded.append(text.strip())

        # Optional debug output
        import os
        if os.environ.get("DRAG_DEBUG_GENERATE", "") == "1":
            try:
                print(f"[debug_generate] Custom autoregressive generation completed")
                print(f"[debug_generate] batch_size={batch_size}, max_prompt_len={max_prompt_len}")
                for i in range(min(2, batch_size)):
                    print(f"[debug_generate] Sample {i}: generated {len(generated_tokens[i])} tokens")
                    print(f"[debug_generate]   token_ids={generated_tokens[i][:15]}{'...' if len(generated_tokens[i]) > 15 else ''}")
                    print(f"[debug_generate]   decoded={repr(decoded[i][:60])}")
                    # Check if EOS was generated
                    if eos_id in generated_tokens[i]:
                        eos_pos = generated_tokens[i].index(eos_id)
                        print(f"[debug_generate]   EOS generated at position {eos_pos}")
                    else:
                        print(f"[debug_generate]   EOS NOT generated (hit max_new_tokens)")
            except Exception as e:
                print(f"[debug_generate] WARNING: {repr(e)}")

        return decoded


def verify_gradient_flow(generator, projector, sample_input):
    """
    Utility to verify gradients flow correctly through the Mamba model.
    Run this before training to catch gradient issues early.
    """
    print("Verifying gradient flow through Mamba...")
    
    # Create dummy neural prompt directly in bfloat16 with gradient tracking
    # IMPORTANT: Create in target dtype from the start to avoid dtype conversion issues
    neural_prompt = torch.randn(
        1, 5, 2688, 
        requires_grad=True, 
        device=generator.model.device,
        dtype=torch.bfloat16
    )
    
    # Retain grad on this tensor so we can check it after backward
    neural_prompt.retain_grad()
    
    # Dummy question and answer
    questions = ["What is the capital of France?"]
    answers = ["Paris"]
    
    # Forward pass
    outputs = generator.forward_paper_prompt(
        neural_prompt,
        questions=questions,
        fact_texts=[["France, capital_of, Paris"] * neural_prompt.shape[1]],
        answer_texts=answers,
    )
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    if neural_prompt.grad is not None and neural_prompt.grad.abs().sum() > 0:
        print("✅ Gradients ARE flowing through neural_prompt_embeds!")
        print(f"   Gradient norm: {neural_prompt.grad.norm().item():.6f}")
        return True
    else:
        print("❌ WARNING: Gradients are NOT flowing through neural_prompt_embeds!")
        print("   This will break D-RAG training.")
        return False
