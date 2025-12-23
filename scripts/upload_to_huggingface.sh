#!/bin/bash
#
# Upload Phase 1 checkpoints and heuristics to Hugging Face Hub
#
# Prerequisites:
#   1. Install huggingface-hub: pip install huggingface-hub
#   2. Login: huggingface-cli login
#   3. Create repository: huggingface-cli repo create drag-improved-checkpoints --type model
#

set -e

REPO_ID="rhordoan/drag-improved-checkpoints"

echo "========================================="
echo "Uploading D-RAG files to Hugging Face"
echo "========================================="
echo ""
echo "Repository: $REPO_ID"
echo ""

# Check if logged in
echo "Checking Hugging Face authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "❌ Not logged in to Hugging Face!"
    echo ""
    echo "Please run: huggingface-cli login"
    echo "Then create a repository:"
    echo "  huggingface-cli repo create drag-improved-checkpoints --type model"
    exit 1
fi

echo "✓ Authenticated"
echo ""

# Files to upload (from /tmp where we extracted them)
declare -a FILES=(
    "/tmp/phase1_best_cwq.pt:checkpoints_cwq_subgraph/phase1_best.pt"
    "/tmp/phase1_best_webqsp.pt:checkpoints_webqsp_subgraph/phase1_best.pt"
    "/tmp/train_heuristics_cwq.jsonl:data/train_heuristics_cwq.jsonl"
    "/tmp/train_heuristics_webqsp_subgraph.jsonl:data/train_heuristics_webqsp_subgraph.jsonl"
)

for file_pair in "${FILES[@]}"; do
    IFS=':' read -r local_path remote_path <<< "$file_pair"
    
    if [ ! -f "$local_path" ]; then
        echo "❌ File not found: $local_path"
        echo "   Please ensure files are extracted to /tmp/"
        exit 1
    fi
    
    size=$(du -h "$local_path" | cut -f1)
    echo "Uploading $remote_path ($size)..."
    
    huggingface-cli upload "$REPO_ID" "$local_path" "$remote_path" --quiet || {
        echo "❌ Upload failed for $remote_path"
        exit 1
    }
    
    echo "✓ Uploaded $remote_path"
    echo ""
done

echo "========================================="
echo "✓ All files uploaded successfully!"
echo "========================================="
echo ""
echo "View your files at:"
echo "  https://huggingface.co/$REPO_ID"
echo ""
echo "Others can now download with:"
echo "  python scripts/download_checkpoints.py"

