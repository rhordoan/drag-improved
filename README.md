# D-RAG: Differentiable Retrieval-Augmented Generation

Implementation of **D-RAG** (Differentiable Retrieval-Augmented Generation) with the **Nemotron-3-Nano-30B** hybrid Mamba-Transformer model.

## 🚀 Quick Start (New Instance)

```bash
# 1. Clone and setup
cd /home/shadeform/nlp/drag-improved
./scripts/setup_environment.sh

# 2. Activate environment
source .venv/bin/activate

# 3. Run Phase 2 training (CWQ)
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_cwq.jsonl \
    --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \
    --epochs 5 \
    --batch_size 1 \
    --generator_model "unsloth/Nemotron-3-Nano-30B-A3B" \
    --checkpoint_dir checkpoints_cwq_phase2
```

---

## 📊 Datasets Used

### Phase 1 Checkpoints (GNN Pre-training)

> **Note:** Checkpoints and heuristics (~700 MB) are automatically downloaded from [🤗 Hugging Face](https://huggingface.co/rhordoan/drag-improved-checkpoints) when you run `setup_environment.sh`.

| Dataset | Samples | Checkpoint | Training Time |
|---------|---------|------------|---------------|
| **CWQ** (ComplexWebQuestions) | 27,613 | `checkpoints_cwq_subgraph/phase1_best.pt` (288 MB) | ~3.5 min |
| **WebQSP** | 2,826 | `checkpoints_webqsp_subgraph/phase1_best.pt` (288 MB) | ~30 sec |

**Data sources:**
- **CWQ**: `rmanluo/RoG-cwq` (Hugging Face) - Complex multi-hop questions
- **WebQSP**: `rmanluo/RoG-webqsp` (Hugging Face) - Single-hop questions

Each sample contains:
- `question`: Natural language question
- `triples`: Per-question subgraph from Freebase (~50 triples per sample)
- `paths`: Gold reasoning paths for supervision
- `answer`: Expected answer

### Heuristics Files

```
data/
├── train_heuristics_cwq.jsonl           # 27,631 CWQ samples (111 MB) - auto-downloaded
├── train_heuristics_webqsp_subgraph.jsonl  # 2,826 WebQSP samples (12 MB) - auto-downloaded
```

**Manual download:**
```bash
python scripts/download_checkpoints.py
```

---

## ⚙️ Environment Setup

### Prerequisites
- **GPU**: H200 (141GB) or B200 (recommended) for 30B model
- **CUDA**: 12.6+ 
- **Python**: 3.12+

### Full Installation

```bash
# 1. Create virtual environment
uv venv .venv
source .venv/bin/activate

# 2. Install core dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install torch_geometric
uv pip install transformers accelerate peft bitsandbytes datasets tqdm

# 3. Install Unsloth
uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth.git"

# 4. Install Mamba dependencies (REQUIRED for Nemotron)
# IMPORTANT: Requires CUDA toolkit for compilation
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6

# Install gcc-11 if needed (for cc1plus)
sudo apt-get update && sudo apt-get install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11

# Install mamba-ssm and causal-conv1d (takes ~8 minutes to compile)
uv pip install mamba-ssm causal-conv1d --no-build-isolation
```

### Troubleshooting Mamba Installation

If `mamba-ssm` fails to compile:

1. **Check CUDA is available:**
   ```bash
   which nvcc  # Should show /usr/local/cuda-12.6/bin/nvcc
   ```

2. **Check g++ is installed:**
   ```bash
   find /usr -name cc1plus  # Should find gcc-11's cc1plus
   ```

3. **Set environment variables:**
   ```bash
   export PATH=/usr/local/cuda-12.6/bin:$PATH
   export CUDA_HOME=/usr/local/cuda-12.6
   ```

---

## 🧠 Training

> **Note:** The paper trains **separate models** for CWQ and WebQSP. They are not combined.

### Phase 1: GNN Pre-training (Per-Question Subgraphs)

Trains the GNN retriever to identify relevant facts based on heuristic paths.
**Pre-trained checkpoints are already included** - skip this if you want to go straight to Phase 2.

#### CWQ Dataset (ComplexWebQuestions)
```bash
# 27,613 samples, ~3.5 min on A100
python -m src.trainer.train_phase1 \
    --heuristics_path data/train_heuristics_cwq.jsonl \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    --node_dim 256 \
    --hidden_dim 256 \
    --relation_dim 256 \
    --num_reasoning_steps 3 \
    --checkpoint_dir checkpoints_cwq_subgraph
```

#### WebQSP Dataset (WebQuestions Semantic Parses)
```bash
# 2,826 samples, ~30 sec on A100
python -m src.trainer.train_phase1 \
    --heuristics_path data/train_heuristics_webqsp_subgraph.jsonl \
    --epochs 10 \
    --batch_size 16 \
    --lr 5e-5 \
    --node_dim 256 \
    --hidden_dim 256 \
    --relation_dim 256 \
    --num_reasoning_steps 3 \
    --checkpoint_dir checkpoints_webqsp_subgraph
```

---

### Phase 2: Joint End-to-End Training

Jointly trains retriever + projector + generator (Nemotron via LoRA).
**Requires H200 (141GB) or B200 (192GB) for the 30B model.**

#### Setup (run once per session)
```bash
# Set CUDA paths (required for Mamba)
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6

# Activate environment
source .venv/bin/activate
```

#### CWQ Dataset - Phase 2
```bash
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_cwq.jsonl \
    --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \
    --epochs 5 \
    --batch_size 1 \
    --lr 5e-5 \
    --k_facts 10 \
    --node_dim 256 \
    --hidden_dim 256 \
    --relation_dim 256 \
    --num_reasoning_steps 3 \
    --generator_model "unsloth/Nemotron-3-Nano-30B-A3B" \
    --checkpoint_dir checkpoints_cwq_phase2
```

#### WebQSP Dataset - Phase 2
```bash
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_webqsp_subgraph.jsonl \
    --phase1_checkpoint checkpoints_webqsp_subgraph/phase1_best.pt \
    --epochs 5 \
    --batch_size 1 \
    --lr 5e-5 \
    --k_facts 10 \
    --node_dim 256 \
    --hidden_dim 256 \
    --relation_dim 256 \
    --num_reasoning_steps 3 \
    --generator_model "unsloth/Nemotron-3-Nano-30B-A3B" \
    --checkpoint_dir checkpoints_webqsp_phase2
```

#### Phase 2 Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--heuristics_path` | required | Path to JSONL with questions + subgraphs |
| `--phase1_checkpoint` | required | Pre-trained GNN checkpoint from Phase 1 |
| `--generator_model` | `unsloth/Nemotron-3-Nano-30B-A3B` | Nemotron model (BF16 recommended) |
| `--epochs` | 5 | Number of joint training epochs |
| `--batch_size` | 1 | Batch size (increase on B200) |
| `--k_facts` | 10 | Number of facts to retrieve per question |
| `--lr` | 5e-5 | Learning rate |
| `--ret_loss_weight` | 0.1 | Weight for retriever auxiliary loss |

---

### Training on B200 (192GB VRAM)

With more VRAM, you can increase batch size for better gradient estimates:

```bash
# B200 optimized settings
python -m src.trainer.train_phase2 \
    --heuristics_path data/train_heuristics_cwq.jsonl \
    --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \
    --epochs 5 \
    --batch_size 4 \
    --k_facts 15 \
    --generator_model "unsloth/Nemotron-3-Nano-30B-A3B" \
    --checkpoint_dir checkpoints_cwq_phase2
```

---

## 📂 Project Structure

```
├── checkpoints_cwq_subgraph/         # Phase 1 CWQ checkpoints
│   └── phase1_best.pt
├── checkpoints_webqsp_subgraph/      # Phase 1 WebQSP checkpoints
│   └── phase1_best.pt
├── data/
│   ├── train_heuristics_cwq.jsonl    # CWQ heuristics with subgraphs
│   └── train_heuristics_webqsp_subgraph.jsonl  # WebQSP heuristics
├── scripts/
│   ├── setup_environment.sh          # Full environment setup
│   ├── generate_cwq_heuristics.py    # Generate CWQ heuristics
│   └── generate_webqsp_subgraph_heuristics.py  # Generate WebQSP heuristics
└── src/
    ├── data/
    │   └── kg_loader.py              # SubgraphDataset for per-question graphs
    ├── model/
    │   ├── retriever.py              # DRAGRetriever (GNN + scoring)
    │   ├── sampler.py                # Gumbel-Softmax differentiable sampling
    │   ├── projector.py              # GNN → LLM dimension bridge
    │   └── generator.py              # Nemotron wrapper with LoRA
    └── trainer/
        ├── train_phase1.py           # GNN pre-training
        └── train_phase2.py           # Joint end-to-end training
```

---

## 🔧 Key Implementation Details

### Per-Question Subgraphs
Unlike approaches that load one giant KG, we use **per-question subgraphs**:
- Each sample has its own small graph (~50 triples)
- Memory efficient (fits on any GPU for Phase 1)
- Matches the paper's approach

### Retriever Architecture (ReaRev-based)
- **Instruction Module**: Sentence-BERT encoder for questions
- **Graph Reasoning**: 3 layers of instruction-conditioned message passing
- **Instruction Update**: Iterative refinement of question representation
- **Fact Scorer**: Bernoulli probability per fact (edge)

### Loss Function
```
L = ρ × L_BCE + (1-ρ) × L_Rank
```
Where ρ = 0.7 (paper default)

### Generator
- **Model**: Nemotron-3-Nano-30B-A3B (BF16)
- **Fine-tuning**: LoRA via Unsloth
- **Injection**: Neural prompts prepended to text embeddings

---

## ⚠️ Hardware Requirements

| Phase | Model | VRAM Required | Recommended GPU |
|-------|-------|---------------|-----------------|
| Phase 1 | GNN only | ~4 GB | Any GPU |
| Phase 2 | GNN + Nemotron 30B | ~140 GB | H200 / B200 |

---

## 📜 Citation

```bibtex
@article{drag2024,
  title={D-RAG: Differentiable Retrieval-Augmented Generation},
  author={...},
  journal={arXiv preprint},
  year={2024}
}
```
