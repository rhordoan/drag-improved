# D-RAG: Differentiable Retrieval-Augmented Generation (H200 + Nemotron Edition)

This repository contains a high-performance implementation of **D-RAG** (Differentiable Retrieval-Augmented Generation), optimized for a **single NVIDIA H200 (141GB)** using the **Unsloth-optimized Nemotron-3-Nano-30B** hybrid Mamba-Transformer model.

## 🚀 Architecture Overview

D-RAG is an end-to-end differentiable pipeline where the retriever learns from the generator's feedback:

1.  **GNN Retriever:** A 2-layer Graph Attention Network (GATConv) that scores facts in a Knowledge Graph subset.
2.  **Differentiable Sampler:** Uses **Gumbel-TopK** to sample a "soft" subgraph, allowing gradients to flow back to the retriever.
3.  **The Projector:** An MLP bridging GNN embeddings into the Nemotron semantic space (dim 2688).
4.  **Generator (Nemotron-3-Nano):** A 30B parameter Hybrid Mamba-Transformer model (3.6B active params via MoE) fine-tuned via **Unsloth's optimized LoRA**. Retrieval context is injected via **Prefix Tuning**.

## 🧠 Model: Nemotron-3-Nano-30B (Unsloth Optimized)

The model uses a hybrid architecture:
*   **Mamba-2 Layers:** For efficient, sequential long-context processing (up to 1M tokens).
*   **Attention Layers:** Interleaved for complex reasoning.
*   **Mixture-of-Experts (MoE):** Only ~3.6B parameters are active per token, making it extremely fast.
*   **Unsloth Acceleration:** Uses Unsloth's `FastLanguageModel` for 2x faster training and 30% less VRAM usage.
*   **FP8 Precision:** Optimized for H200 using `transformer-engine`'s FP8 support within the training loop.

## ⚡ Hardware & Requirements

### Hardware
*   **GPU:** 1x NVIDIA H200 (141GB VRAM).
*   **Precision:** FP8/BF16 mixed precision.

### Software
```bash
# Environment setup (uv recommended)
uv venv .venv
# Activate: source .venv/bin/activate (Unix) or .\.venv\Scripts\Activate.ps1 (Windows)

# Core dependencies (including Unsloth)
uv pip install -U transformers accelerate peft bitsandbytes datasets
uv pip install -U flash-attn --no-build-isolation
uv pip install transformer-engine[pytorch]
uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth.git"
uv pip install torch_geometric
```

## 📂 Project Structure
```text
├── data/
│   ├── webqsp/               # WebQuestions dataset (stanfordnlp/web_questions)
│   ├── cwq/                  # Complex WebQuestions (rmanluo/RoG-cwq)
│   ├── kg/                   # Freebase Knowledge Graph files
│   └── train_heuristics.jsonl # Mined shortest paths for Phase 1
├── src/
│   ├── data/
│   │   ├── download_hf.py    # HF downloader for WebQSP/CWQ
│   │   └── kg_loader.py      # KG triples to PyG Data loader
│   ├── model/
│   │   ├── retriever.py      # GATConv GNN
│   │   ├── sampler.py        # Gumbel-TopK sampler
│   │   ├── projector.py      # GNN -> LLM dimension bridge
│   │   └── generator.py      # Nemotron wrapper (Prefix Tuning + LoRA)
│   ├── trainer/
│   │   ├── train_phase1.py   # GNN Warmup (Supervised)
│   │   └── train_phase2.py   # Joint End-to-End Training
│   └── utils/
│       ├── coverage_check.py # KG entity coverage verification
│       └── subgraph_mining.py # Heuristic mining via networkx
└── scripts/
    └── download_datasets.sh   # Bash entry point for data
```

## 🛠️ Data Preparation

### 1. Download QA Pairs
```bash
bash scripts/download_datasets.sh
```
This pulls `stanfordnlp/web_questions` and `rmanluo/RoG-cwq` via Hugging Face.

### 2. Knowledge Graph & Heuristics
*   Place your trimmed Freebase subset (e.g., `freebase_2hop.txt`) in `data/kg/`.
*   Verify entity coverage:
    ```bash
    python -m src.utils.coverage_check --dataset data/webqsp/WebQSP.train.json --entities data/kg/entities.txt
    ```
*   Mine shortest paths (teacher labels):
    ```bash
    python -m src.utils.subgraph_mining --kg data/kg/freebase_2hop.txt --dataset data/webqsp/WebQSP.train.json
    ```

## 🧠 Training Strategy

### Phase 1: GNN Warmup
Train the Retriever alone to mimic heuristic shortest paths.
```bash
python -m src.trainer.train_phase1 \
    --kg_path data/kg/freebase_2hop.txt \
    --heuristics_path data/train_heuristics.jsonl \
    --epochs 10 \
    --lr 1e-3
```

### Phase 2: Joint End-to-End Training (D-RAG)
Jointly optimize Retriever, Projector, and Nemotron (via LoRA).
```bash
python -m src.trainer.train_phase2 \
    --kg_path data/kg/freebase_2hop.txt \
    --dataset_path data/webqsp/WebQSP.train.json \
    --gnn_checkpoint checkpoints/phase1_best.pt \
    --llm_model unsloth/Nemotron-3-Nano-30B-A3B-FP8 \
    --k_facts 10 \
    --batch_size 1 \
    --lr 5e-5
```

## ⚠️ Implementation Notes

1.  **Prefix Tuning:** Neural fact prompts are prepended to the text embeddings. This ensures the Mamba (sequential) layers process context before the question.
2.  **LoRA Config:** We target `all-linear` modules. This ensures both Transformer (Q/K/V) and Mamba (in_proj/out_proj) components are adapted.
3.  **Memory Management:** Use `gradient_checkpointing` for long sequences. Set `use_cache=False` during training forward passes.
4.  **Tokenizer:** `tokenizer.pad_token` is set to `eos_token` to handle batching for the hybrid architecture.

## 📜 Citation
Original D-RAG paper:
```bibtex
@article{drag2024,
  title={D-RAG: Differentiable Retrieval-Augmented Generation},
  author={Anonymous},
  journal={arXiv preprint},
  year={2024}
}
```
