#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/webqsp data/cwq

echo "Downloading datasets via src/data/download_hf.py..."
python -m src.data.download_hf --output_dir data

echo "Datasets downloaded to data/webqsp and data/cwq"
