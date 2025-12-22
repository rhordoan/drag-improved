import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

try:
    from transformer_engine.pytorch import fp8_autocast
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

class NemotronGenerator(nn.Module):
    def __init__(self, model_id="unsloth/Nemotron-3-Nano-30B-A3B-FP8", use_lora=True):
        super(NemotronGenerator, self).__init__()
        
        print(f"Loading Unsloth model and tokenizer: {model_id}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id,
            max_seq_length = 2048,
            load_in_4bit = False, # Set to True for 4-bit quantization if needed
            load_in_8bit = False,
            trust_remote_code = True,
            unsloth_force_compile = True, # Optimized inference/training
            attn_implementation = "eager", # Or "flash_attention_2" if supported on H200
        )
        
        # Fix for open-ended generation / batching
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if use_lora:
            print("Applying Unsloth optimized LoRA...")
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = 64, # Matches user's previous preference, or 8-128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj",
                                 "in_proj", "out_proj",], # Optimized targets for Nemotron
                lora_alpha = 128,
                lora_dropout = 0, # Unsloth optimized at 0
                bias = "none",    # Unsloth optimized at "none"
                use_gradient_checkpointing = "unsloth", # 30% less VRAM
                random_state = 3407,
            )
            self.model.print_trainable_parameters()

    def forward(self, neural_prompt_embeds, questions, labels=None):
        """
        Forward pass with Prefix Tuning using Unsloth optimizations.
        neural_prompt_embeds: [batch_size, k_facts, hidden_size]
        questions: List of strings
        """
        # 1. Prepare text inputs
        text_inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)
        
        # 2. Get text embeddings
        # Unsloth model wraps the base model
        text_embeds = self.model.get_input_embeddings()(text_inputs.input_ids)
        
        # 3. Fuse: [Neural Prompts] + [Text Embeddings]
        # Cast neural_prompt_embeds to bfloat16 for the LLM
        neural_prompt_embeds = neural_prompt_embeds.to(torch.bfloat16)
        combined_embeds = torch.cat([neural_prompt_embeds, text_embeds], dim=1)
        
        # 4. Forward
        # Use fp8_autocast if available (optimized for H200)
        if TE_AVAILABLE:
            with fp8_autocast(enabled=True):
                outputs = self.model(
                    inputs_embeds=combined_embeds,
                    labels=labels,
                    use_cache=False
                )
        else:
            outputs = self.model(
                inputs_embeds=combined_embeds,
                labels=labels,
                use_cache=False
            )
        
        return outputs

    def generate(self, neural_prompt_embeds, questions, max_new_tokens=100):
        """
        Generation helper using Unsloth native inference.
        """
        text_inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)
        
        text_embeds = self.model.get_input_embeddings()(text_inputs.input_ids)
        
        neural_prompt_embeds = neural_prompt_embeds.to(torch.bfloat16)
        combined_embeds = torch.cat([neural_prompt_embeds, text_embeds], dim=1)
        
        # Unsloth native inference is faster
        if TE_AVAILABLE:
            with fp8_autocast(enabled=True):
                gen_outputs = self.model.generate(
                    inputs_embeds=combined_embeds,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
        else:
            gen_outputs = self.model.generate(
                inputs_embeds=combined_embeds,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        return self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
