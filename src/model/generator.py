import torch
import torch.nn as nn
from unsloth import FastLanguageModel

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
    def __init__(self, model_id="unsloth/Nemotron-3-Nano-30B-A3B-FP8", use_lora=True):
        super(NemotronGenerator, self).__init__()
        
        print(f"Loading Unsloth model and tokenizer: {model_id}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id,
            max_seq_length = 2048,
            load_in_4bit = False,
            load_in_8bit = False,
            trust_remote_code = True,
            # Note: Disable unsloth_force_compile for custom inputs_embeds compatibility
            # The Triton kernels may not handle custom embedding injection correctly
            unsloth_force_compile = False,
            attn_implementation = "eager",  # Safer for custom inputs_embeds
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
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

    def forward(self, neural_prompt_embeds, questions, answer_texts=None):
        """
        Forward pass with proper label alignment for Mamba.
        
        Args:
            neural_prompt_embeds: [batch_size, k_facts, hidden_size] - MUST have requires_grad=True
            questions: List of question strings
            answer_texts: List of answer strings (for training)
        
        Returns:
            CausalLMOutputWithPast with properly computed loss
        """
        device = self.model.device
        batch_size = neural_prompt_embeds.shape[0]
        num_neural_tokens = neural_prompt_embeds.shape[1]
        
        # Verify gradient flow - critical for D-RAG
        assert neural_prompt_embeds.requires_grad, \
            "neural_prompt_embeds must have requires_grad=True for D-RAG training!"
        
        # 1. Tokenize question + answer together for proper label alignment
        if answer_texts is not None:
            # Format: [Question] [Answer]
            full_texts = [f"{q} {a}" for q, a in zip(questions, answer_texts)]
            question_only = questions
        else:
            full_texts = questions
            question_only = questions
        
        # Tokenize the full text (question + answer)
        full_inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        ).to(device)
        
        # Tokenize question only to find where answer starts
        question_inputs = self.tokenizer(
            question_only,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False
        ).to(device)
        
        question_lengths = question_inputs.attention_mask.sum(dim=1)
        
        # 2. Get text embeddings
        text_embeds = self.model.get_input_embeddings()(full_inputs.input_ids)
        
        # 3. Fuse: [Neural Prompts] + [Text Embeddings]
        neural_prompt_embeds = neural_prompt_embeds.to(text_embeds.dtype)
        combined_embeds = torch.cat([neural_prompt_embeds, text_embeds], dim=1)
        
        # 4. Create labels with proper masking
        # Labels should be -100 for positions we don't want to compute loss on
        # Structure: [neural_prompts: -100] [question: -100] [answer: actual tokens]
        
        if answer_texts is not None:
            labels = full_inputs.input_ids.clone()
            
            for i in range(batch_size):
                # Mask out neural prompt positions
                # Mask out question positions (only train on answer)
                q_len = question_lengths[i].item()
                # The first (num_neural_tokens + q_len) positions should be masked
                pass  # Labels are for text portion only
            
            # Prepend -100 for the neural prompt tokens
            neural_prompt_labels = torch.full(
                (batch_size, num_neural_tokens), 
                -100, 
                dtype=torch.long, 
                device=device
            )
            
            # Mask question tokens in the text labels
            for i in range(batch_size):
                q_len = question_lengths[i].item()
                labels[i, :q_len] = -100
            
            # Combine: [neural prompt labels (-100)] + [text labels (masked question, actual answer)]
            combined_labels = torch.cat([neural_prompt_labels, labels], dim=1)
        else:
            combined_labels = None
        
        # 5. Create attention mask for the combined sequence
        neural_prompt_mask = torch.ones(
            (batch_size, num_neural_tokens), 
            dtype=torch.long, 
            device=device
        )
        combined_attention_mask = torch.cat([neural_prompt_mask, full_inputs.attention_mask], dim=1)
        
        # 6. Forward pass
        # Note: For Mamba layers, attention_mask is typically not used (they're causal by design)
        # But we pass it for the Transformer layers in the hybrid model
        if TE_AVAILABLE:
            with fp8_autocast(enabled=True):
                outputs = self.model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    labels=combined_labels,
                    use_cache=False,
                    return_dict=True
                )
        else:
            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=combined_labels,
                use_cache=False,
                return_dict=True
            )
        
        return outputs

    def generate(self, neural_prompt_embeds, questions, max_new_tokens=100):
        """
        Generation with neural prompt prefix.
        """
        device = self.model.device
        batch_size = neural_prompt_embeds.shape[0]
        num_neural_tokens = neural_prompt_embeds.shape[1]
        
        text_inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(device)
        
        text_embeds = self.model.get_input_embeddings()(text_inputs.input_ids)
        
        neural_prompt_embeds = neural_prompt_embeds.to(text_embeds.dtype)
        combined_embeds = torch.cat([neural_prompt_embeds, text_embeds], dim=1)
        
        # Attention mask for combined sequence
        neural_prompt_mask = torch.ones(
            (batch_size, num_neural_tokens), 
            dtype=torch.long, 
            device=device
        )
        combined_attention_mask = torch.cat([neural_prompt_mask, text_inputs.attention_mask], dim=1)
        
        if TE_AVAILABLE:
            with fp8_autocast(enabled=True):
                gen_outputs = self.model.generate(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
        else:
            gen_outputs = self.model.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        return self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)


def verify_gradient_flow(generator, projector, sample_input):
    """
    Utility to verify gradients flow correctly through the Mamba model.
    Run this before training to catch gradient issues early.
    """
    print("Verifying gradient flow through Mamba...")
    
    # Create dummy neural prompt with gradient tracking
    neural_prompt = torch.randn(1, 5, 2688, requires_grad=True, device=generator.model.device)
    neural_prompt = neural_prompt.to(torch.bfloat16)
    
    # Dummy question and answer
    questions = ["What is the capital of France?"]
    answers = ["Paris"]
    
    # Forward pass
    outputs = generator(neural_prompt, questions, answer_texts=answers)
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
