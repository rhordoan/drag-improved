# Phase 1 Checkpoints

This document describes the pre-trained Phase 1 checkpoints for D-RAG.

> **🤗 Storage:** Checkpoints are hosted on [Hugging Face Hub](https://huggingface.co/rhordoan/drag-improved-checkpoints) due to GitHub's 100MB file size limit. They are automatically downloaded when you run `./scripts/setup_environment.sh`.

## Available Checkpoints

### CWQ (ComplexWebQuestions)
- **Path**: `checkpoints_cwq_subgraph/phase1_best.pt`
- **Size**: 301 MB
- **Training data**: 27,613 samples from `rmanluo/RoG-cwq`
- **Epochs**: 10
- **Final loss**: 0.2616
- **Training time**: ~3.5 minutes on A100

### WebQSP (WebQuestions Semantic Parses)
- **Path**: `checkpoints_webqsp_subgraph/phase1_best.pt`
- **Size**: 301 MB
- **Training data**: 2,826 samples from `rmanluo/RoG-webqsp`
- **Epochs**: 10
- **Final loss**: ~0.25
- **Training time**: ~30 seconds on A100

## Checkpoint Contents

Each checkpoint file contains:

```python
{
    'epoch': int,                    # Training epoch (1-10)
    'model_state_dict': dict,        # DRAGRetriever weights
    'optimizer_state_dict': dict,    # AdamW optimizer state
    'loss': float,                   # BCE + Ranking loss
    'args': dict                     # Training arguments
}
```

## Model Architecture (DRAGRetriever)

The checkpoints contain weights for:

1. **Instruction Module** (Sentence-BERT)
   - `instruction_module.encoder.*` - 6-layer BERT encoder
   - Encodes questions into instruction vectors

2. **Relation Encoder** (shared with Instruction Module)
   - `relation_encoder.lm_encoder.*` - Reference to instruction module
   - `relation_encoder.projection.*` - MLP to project relations

3. **Graph Reasoning Layers** (3 layers)
   - `reasoning_layers.*.node_transform.*`
   - `reasoning_layers.*.query/key/value.*`
   - `reasoning_layers.*.out_proj.*`

4. **Instruction Update Module**
   - `instruction_update.*` - Updates instruction based on graph state

5. **Fact Scorer**
   - `fact_scorer.scorer.*` - Scores each fact for selection

## Training Configuration

```python
# Common settings for both datasets
node_dim = 256
hidden_dim = 256
relation_dim = 256
instruction_dim = 384  # Sentence-BERT output
num_reasoning_steps = 3
num_heads = 4
rho = 0.7  # BCE vs Ranking loss weight

# Optimizer
optimizer = AdamW(lr=5e-5, weight_decay=0.001)
scheduler = CosineAnnealingLR
batch_size = 16
gradient_clipping = 1.0
```

## Usage

```python
import torch
from src.model.retriever import DRAGRetriever

# Load checkpoint
checkpoint = torch.load('checkpoints_cwq_subgraph/phase1_best.pt')

# Initialize model with same architecture
retriever = DRAGRetriever(
    node_dim=256,
    edge_dim=256,
    hidden_dim=256,
    instruction_dim=384,
    relation_dim=256,
    num_reasoning_steps=3
)

# Load weights
retriever.load_state_dict(checkpoint['model_state_dict'])
```

## Performance

| Dataset | BCE Loss | Ranking Loss | Total Loss |
|---------|----------|--------------|------------|
| CWQ     | 0.092    | 0.656        | 0.262      |
| WebQSP  | ~0.09    | ~0.65        | ~0.25      |

Note: These are Phase 1 losses (supervised on heuristic paths). 
Phase 2 jointly optimizes with the generator.

