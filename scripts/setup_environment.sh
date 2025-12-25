#!/bin/bash
# D-RAG Environment Setup Script
# Tested on: Ubuntu 22.04, CUDA 12.6, Python 3.12

set -e

echo "============================================================"
echo "D-RAG Environment Setup"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "src/model/retriever.py" ]; then
    echo "ERROR: Please run this script from the drag-improved directory"
    exit 1
fi

# 1. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Activate venv
source .venv/bin/activate
echo "✓ Virtual environment activated"

# 2. Set CUDA environment
# Prefer newer toolkit if present (Blackwell GPUs require newer ptxas for Triton kernels)
if [ -d "/usr/local/cuda-13.0" ]; then
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    export CUDA_HOME=/usr/local/cuda-13.0
    # Triton defaults to a bundled ptxas; override to use system ptxas which supports newer SMs (e.g., sm_103)
    export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
    echo "✓ CUDA environment set (13.0)"
elif [ -d "/usr/local/cuda-12.8" ]; then
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    export CUDA_HOME=/usr/local/cuda-12.8
    # Even on CUDA 12.8, Triton may use a bundled ptxas which can lag behind and fail on newer SMs.
    # Prefer the system ptxas from the selected toolkit.
    export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
    echo "✓ CUDA environment set (12.8)"
else
    echo "WARNING: No /usr/local/cuda-13.0 or /usr/local/cuda-12.8 found. You may need to set CUDA_HOME/PATH manually."
fi

# 3. Check if gcc-11 is available (needed for mamba-ssm compilation)
echo ""
echo "Checking build dependencies..."
if ! command -v g++-11 &> /dev/null; then
    echo "Installing gcc-11 and g++-11..."
    sudo apt-get update && sudo apt-get install -y gcc-11 g++-11
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-11 \
        --slave /usr/bin/gcov gcov /usr/bin/gcov-11
fi
echo "✓ Build dependencies ready"

# 4. Install Python packages
echo ""
echo "Installing Python dependencies..."
echo "  [1/5] PyTorch..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 > /dev/null 2>&1
echo "  ✓ PyTorch installed"

echo "  [2/5] PyTorch Geometric..."
uv pip install torch_geometric > /dev/null 2>&1
echo "  ✓ PyG installed"

echo "  [3/5] Transformers and utilities..."
uv pip install transformers accelerate peft bitsandbytes datasets tqdm sentencepiece > /dev/null 2>&1
echo "  ✓ Transformers installed"

echo "  [4/5] Unsloth..."
uv pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth.git" > /dev/null 2>&1
echo "  ✓ Unsloth installed"

echo "  [5/5] Mamba dependencies (this takes ~5-8 minutes to compile)..."
uv pip install ninja packaging wheel setuptools > /dev/null 2>&1
uv pip install mamba-ssm causal-conv1d --no-build-isolation
echo "  ✓ Mamba-ssm installed"

# 5. Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
from torch_geometric.data import Data
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Check mamba-ssm
python -c "import mamba_ssm; print('  Mamba-ssm: OK')" 2>/dev/null || echo "  Mamba-ssm: Not installed (may need manual compilation)"

# 6. Download checkpoints and heuristics from Hugging Face
echo ""
echo "============================================================"
echo "Downloading Phase 1 Checkpoints from Hugging Face"
echo "============================================================"

if [ -f "checkpoints_cwq_subgraph/phase1_best.pt" ] && [ -f "data/train_heuristics_cwq.jsonl" ]; then
    echo "✓ Checkpoints already exist, skipping download"
else
    echo ""
    echo "Downloading ~700 MB of checkpoints and heuristics..."
    python scripts/download_checkpoints.py
fi

echo ""
echo "============================================================"
echo "✓ Environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate: source .venv/bin/activate"
echo "  2. Set CUDA:"
echo "     - CUDA 13.0: export PATH=/usr/local/cuda-13.0/bin:\$PATH && export CUDA_HOME=/usr/local/cuda-13.0 && export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas"
echo "     - CUDA 12.8: export PATH=/usr/local/cuda-12.8/bin:\$PATH && export CUDA_HOME=/usr/local/cuda-12.8"
echo "  3. Run Phase 2:"
echo "     python -m src.trainer.train_phase2 \\"
echo "         --heuristics_path data/train_heuristics_cwq.jsonl \\"
echo "         --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \\"
echo "         --epochs 5 --batch_size 1 \\"
echo "         --generator_model 'unsloth/Nemotron-3-Nano-30B-A3B' \\"
echo "         --checkpoint_dir checkpoints_cwq_phase2"
echo "============================================================"

