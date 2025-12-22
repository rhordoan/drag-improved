import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

try:
    from transformer_engine.pytorch import fp8_autocast
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

class NemotronGenerator(nn.Module):
    def __init__(self, model_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8", use_lora=True):
        super(NemotronGenerator, self).__init__()
        
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Fix for open-ended generation / batching
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model: {model_id}")
        # H200 optimized load
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        if use_lora:
            print("Applying LoRA (all-linear targeting)...")
            peft_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules="all-linear",
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def forward(self, neural_prompt_embeds, questions, labels=None):
        """
        Forward pass with Prefix Tuning.
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
        # Use peft model's base model if LoRA is applied
        base_model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        text_embeds = base_model.get_input_embeddings()(text_inputs.input_ids)
        
        # 3. Fuse: [Neural Prompts] + [Text Embeddings]
        # Cast neural_prompt_embeds to bfloat16 for the LLM
        neural_prompt_embeds = neural_prompt_embeds.to(torch.bfloat16)
        combined_embeds = torch.cat([neural_prompt_embeds, text_embeds], dim=1)
        
        # 4. Forward
        # During training, set use_cache=False to save memory
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
        Generation helper
        """
        text_inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        ).to(self.model.device)
        
        base_model = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        text_embeds = base_model.get_input_embeddings()(text_inputs.input_ids)
        
        neural_prompt_embeds = neural_prompt_embeds.to(torch.bfloat16)
        combined_embeds = torch.cat([neural_prompt_embeds, text_embeds], dim=1)
        
        # Note: generation with inputs_embeds is supported in HF
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

