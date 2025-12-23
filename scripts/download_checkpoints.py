#!/usr/bin/env python3
"""
Download Phase 1 checkpoints and heuristics files from Hugging Face.

This script downloads the large files that are too big for Git:
- Phase 1 CWQ checkpoint (288 MB)
- Phase 1 WebQSP checkpoint (288 MB)
- CWQ heuristics (111 MB)
- WebQSP heuristics (12 MB)
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Hugging Face repository
REPO_ID = "rhordoancc/drag-improved-checkpoints"

# Files to download
FILES = [
    {
        "filename": "checkpoints_cwq_subgraph/phase1_best.pt",
        "local_path": "checkpoints_cwq_subgraph/phase1_best.pt",
        "size": "288 MB"
    },
    {
        "filename": "checkpoints_webqsp_subgraph/phase1_best.pt",
        "local_path": "checkpoints_webqsp_subgraph/phase1_best.pt",
        "size": "288 MB"
    },
    {
        "filename": "data/train_heuristics_cwq.jsonl",
        "local_path": "data/train_heuristics_cwq.jsonl",
        "size": "111 MB"
    },
    {
        "filename": "data/train_heuristics_webqsp_subgraph.jsonl",
        "local_path": "data/train_heuristics_webqsp_subgraph.jsonl",
        "size": "12 MB"
    }
]

def main():
    print("=" * 70)
    print("Downloading D-RAG Phase 1 Checkpoints and Heuristics from Hugging Face")
    print("=" * 70)
    print(f"\nRepository: {REPO_ID}")
    print(f"Total files: {len(FILES)}")
    print(f"Total size: ~700 MB\n")
    
    # Get project root (parent of scripts/)
    project_root = Path(__file__).parent.parent
    
    for i, file_info in enumerate(FILES, 1):
        filename = file_info["filename"]
        local_path = project_root / file_info["local_path"]
        size = file_info["size"]
        
        print(f"[{i}/{len(FILES)}] Downloading {filename} ({size})...")
        
        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if local_path.exists():
            print(f"  ✓ Already exists, skipping")
            continue
        
        try:
            # Download from Hugging Face
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=project_root,
                local_dir_use_symlinks=False
            )
            print(f"  ✓ Downloaded to {local_path}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"     Please download manually from https://huggingface.co/{REPO_ID}")
            continue
    
    print("\n" + "=" * 70)
    print("✓ Download complete!")
    print("=" * 70)
    print("\nYou can now run Phase 2 training:")
    print("  python -m src.trainer.train_phase2 \\")
    print("      --heuristics_path data/train_heuristics_cwq.jsonl \\")
    print("      --phase1_checkpoint checkpoints_cwq_subgraph/phase1_best.pt \\")
    print("      --epochs 5 --batch_size 1")

if __name__ == "__main__":
    main()

