#!/usr/bin/env python3
"""
Diagnostic script to verify EOS supervision and token alignment.
Checks:
1. Whether EOS is in supervised labels
2. Whether the token sequence is correct
3. Whether labels align with inputs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
from src.model.generator import NemotronGenerator

def main():
    print("=" * 70)
    print("EOS SUPERVISION DIAGNOSTIC")
    print("=" * 70)
    
    # Load generator
    print("\n[1] Loading generator...")
    generator = NemotronGenerator(
        model_id="unsloth/Nemotron-3-Nano-30B-A3B",
        revision="4e73921165feb0016e8e1d48262910ed060473e2",
        use_lora=True,
    )
    
    # Load the best checkpoint LoRA weights using PEFT's proper loading
    checkpoint_dir = "checkpoints_cwq_phase2_paperprompt/generator_best"
    if os.path.exists(checkpoint_dir):
        print(f"\n[1b] Loading trained LoRA weights from {checkpoint_dir}...")
        try:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                state_dict = load_file(adapter_path)
                # Use PEFT's proper loading function
                set_peft_model_state_dict(generator.model, state_dict)
                print(f"    Successfully loaded {len(state_dict)} adapter tensors!")
            else:
                print(f"    WARNING: {adapter_path} not found!")
        except Exception as e:
            print(f"    Failed to load via set_peft_model_state_dict: {e}")
            # Try alternative: PeftModel.from_pretrained on the base model
            try:
                from peft import PeftModel
                # Get the base model (before PEFT wrapping)
                base_model = generator.model.get_base_model() if hasattr(generator.model, 'get_base_model') else generator.model
                print(f"    Trying PeftModel.from_pretrained...")
                generator.model = PeftModel.from_pretrained(base_model, checkpoint_dir)
                print(f"    Loaded via PeftModel.from_pretrained!")
            except Exception as e2:
                print(f"    Also failed: {e2}")
                import traceback
                traceback.print_exc()
    else:
        print(f"\n[1b] WARNING: No checkpoint found at {checkpoint_dir}, using fresh LoRA!")
    
    tokenizer = generator.tokenizer
    device = generator.model.device
    
    print(f"\n[2] Token IDs:")
    print(f"    EOS token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"    PAD token: {repr(tokenizer.pad_token)} (id={tokenizer.pad_token_id})")
    print(f"    BOS token: {repr(getattr(tokenizer, 'bos_token', None))} (id={getattr(tokenizer, 'bos_token_id', None)})")
    
    # Test examples
    test_cases = [
        ("What is the capital of France?", "Paris"),
        ("Which bodybuilder played Darth Vader?", "David Prowse"),
        ("What state is home to Vicksburg?", "Mississippi"),
    ]
    
    print("\n[3] Tokenization checks:")
    print("-" * 70)
    
    for question, answer in test_cases:
        # How we prepare the answer (with leading space)
        answer_with_space = " " + answer if not answer.startswith(" ") else answer
        
        # Tokenize
        ans_ids = tokenizer(answer_with_space, add_special_tokens=False).input_ids
        ans_ids_with_eos = ans_ids + [tokenizer.eos_token_id]
        
        # Decode back
        decoded = tokenizer.decode(ans_ids, skip_special_tokens=False)
        decoded_with_eos = tokenizer.decode(ans_ids_with_eos, skip_special_tokens=False)
        
        print(f"\n  Answer: {repr(answer)}")
        print(f"  With space: {repr(answer_with_space)}")
        print(f"  Token IDs: {ans_ids}")
        print(f"  Token IDs + EOS: {ans_ids_with_eos}")
        print(f"  Decoded (no EOS): {repr(decoded)}")
        print(f"  Decoded (with EOS): {repr(decoded_with_eos)}")
        print(f"  Last token is EOS? {ans_ids_with_eos[-1] == tokenizer.eos_token_id}")
        
        # Show individual tokens
        print(f"  Individual tokens:")
        for i, tid in enumerate(ans_ids_with_eos):
            tok_str = tokenizer.decode([tid], skip_special_tokens=False)
            print(f"    [{i}] id={tid:5d} -> {repr(tok_str)}")
    
    print("\n" + "-" * 70)
    print("\n[4] Testing forward_paper_prompt label construction:")
    print("-" * 70)
    
    # Create dummy neural embeddings
    hidden_size = generator.model.config.hidden_size
    dummy_fact_embeds = torch.randn(1, 2, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    
    # Test with a simple case
    question = "Which bodybuilder played Darth Vader?"
    answer = "David Prowse"
    fact_texts = [["David Prowse, portrayed, Darth Vader", "Star Wars, character, Darth Vader"]]
    
    # Temporarily enable debug printing
    os.environ["DRAG_DEBUG_EOS"] = "1"
    
    # Reset debug flags
    generator._eos_debug_printed = False
    generator._label_debug_printed = False
    
    print(f"\n  Running forward pass with:")
    print(f"    Question: {question}")
    print(f"    Answer: {answer}")
    print(f"    Facts: {fact_texts[0]}")
    
    try:
        outputs = generator.forward_paper_prompt(
            dummy_fact_embeds,
            questions=[question],
            fact_texts=fact_texts,
            answer_texts=[answer],
            eos_loss_weight=1.0,
        )
        print(f"\n  Forward pass succeeded. Loss: {outputs.loss.item():.4f}")
    except Exception as e:
        print(f"\n  Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-" * 70)
    print("\n[5] Testing generation (does it ever produce EOS?):")
    print("-" * 70)
    
    # Switch to inference
    generator.model.eval()
    
    with torch.no_grad():
        generated = generator.generate(
            dummy_fact_embeds,
            questions=[question],
            fact_texts=fact_texts,
            max_new_tokens=30,
        )
    
    print(f"  Generated: {repr(generated[0])}")
    
    # Try to see the raw token IDs
    print("\n[6] Raw generation token check:")
    print("-" * 70)
    
    # Re-run generation but capture token IDs
    from transformers.generation.utils import GenerationMixin
    
    generator.model.eval()
    embed_layer = generator.model.get_input_embeddings()
    
    # Build prompt embeddings (simplified)
    instr = "Answer the question based on the provided facts.\n\n"
    provided_facts_header = "Provided facts:\n\n"
    answer_header = "\nAnswer:"
    
    prefix_text = f"{instr}Question: {question}\n\n{provided_facts_header}"
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    bos_id = getattr(tokenizer, 'bos_token_id', None)
    if bos_id is not None:
        prefix_ids = [bos_id] + prefix_ids
    prefix_ids_t = torch.tensor(prefix_ids, device=device, dtype=torch.long)
    prefix_emb = embed_layer(prefix_ids_t).unsqueeze(0)
    
    parts = [prefix_emb]
    for j in range(2):
        struct = dummy_fact_embeds[:, j:j+1, :]
        fact_line = f"{fact_texts[0][j]}\n"
        fact_ids = tokenizer(fact_line, add_special_tokens=False).input_ids
        fact_ids_t = torch.tensor(fact_ids, device=device, dtype=torch.long)
        fact_emb = embed_layer(fact_ids_t).unsqueeze(0)
        parts.append(struct)
        parts.append(fact_emb)
    
    ans_hdr_ids = tokenizer(answer_header, add_special_tokens=False).input_ids
    ans_hdr_ids_t = torch.tensor(ans_hdr_ids, device=device, dtype=torch.long)
    ans_hdr_emb = embed_layer(ans_hdr_ids_t).unsqueeze(0)
    parts.append(ans_hdr_emb)
    
    batch_embeds = torch.cat(parts, dim=1).to(dtype=torch.bfloat16)
    batch_attn = torch.ones((1, batch_embeds.shape[1]), device=device, dtype=torch.long)
    
    prompt_len = batch_embeds.shape[1]
    
    gen_model = generator.model
    if hasattr(generator.model, "base_model") and hasattr(generator.model.base_model, "model"):
        gen_model = generator.model.base_model.model
    
    with torch.no_grad():
        gen_outputs = GenerationMixin.generate(
            gen_model,
            inputs_embeds=batch_embeds,
            attention_mask=batch_attn,
            max_new_tokens=30,
            min_new_tokens=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=False,
        )
    
    if gen_outputs.shape[1] > prompt_len:
        new_tokens = gen_outputs[0, prompt_len:]
    else:
        new_tokens = gen_outputs[0]
    
    new_token_ids = new_tokens.cpu().tolist()
    
    print(f"  New token IDs: {new_token_ids}")
    print(f"  EOS id ({tokenizer.eos_token_id}) in output? {tokenizer.eos_token_id in new_token_ids}")
    
    print(f"\n  Token-by-token breakdown:")
    for i, tid in enumerate(new_token_ids[:20]):
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        is_eos = "(EOS)" if tid == tokenizer.eos_token_id else ""
        is_pad = "(PAD)" if tid == tokenizer.pad_token_id else ""
        print(f"    [{i:2d}] id={tid:5d} -> {repr(tok_str):20s} {is_eos}{is_pad}")
    
    if len(new_token_ids) > 20:
        print(f"    ... ({len(new_token_ids) - 20} more tokens)")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()

