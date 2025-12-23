# Upload Instructions for Hugging Face

This document explains how to upload the Phase 1 checkpoints and heuristics to Hugging Face Hub.

## Prerequisites

1. **Hugging Face account:** Create one at https://huggingface.co/join
2. **Hugging Face token:** Get your token at https://huggingface.co/settings/tokens (with `write` permission)

## Step 1: Login to Hugging Face

```bash
# Login with your token
huggingface-cli login
# Paste your token when prompted
```

## Step 2: Create Repository

```bash
# Create a new model repository
huggingface-cli repo create drag-improved-checkpoints --type model
```

This will create: `https://huggingface.co/YOUR_USERNAME/drag-improved-checkpoints`

## Step 3: Prepare Files

The files are currently in `/tmp/` (extracted from the git history):

```bash
ls -lh /tmp/phase1_best_*.pt /tmp/train_heuristics_*.jsonl
```

You should see:
- `/tmp/phase1_best_cwq.pt` (288 MB)
- `/tmp/phase1_best_webqsp.pt` (288 MB)
- `/tmp/train_heuristics_cwq.jsonl` (111 MB)
- `/tmp/train_heuristics_webqsp_subgraph.jsonl` (12 MB)

## Step 4: Upload Files

Run the upload script:

```bash
cd /home/shadeform/nlp/drag-improved
./scripts/upload_to_huggingface.sh
```

This will upload all 4 files to your Hugging Face repository.

## Step 5: Upload README

```bash
# Copy the Hugging Face README
cp /tmp/HF_README.md /tmp/README.md

# Upload it
huggingface-cli upload rhordoan/drag-improved-checkpoints /tmp/README.md README.md
```

## Step 6: Verify

Visit your repository at:
```
https://huggingface.co/rhordoan/drag-improved-checkpoints
```

You should see:
- ✅ `checkpoints_cwq_subgraph/phase1_best.pt`
- ✅ `checkpoints_webqsp_subgraph/phase1_best.pt`
- ✅ `data/train_heuristics_cwq.jsonl`
- ✅ `data/train_heuristics_webqsp_subgraph.jsonl`
- ✅ `README.md`

## Step 7: Test Download

Test that others can download your files:

```bash
# From a fresh clone
cd /home/shadeform/nlp/drag-improved
python scripts/download_checkpoints.py
```

This should download all files from your Hugging Face repository.

## Troubleshooting

### Upload Timeout
If upload times out for large files:
```bash
# Upload one file at a time
huggingface-cli upload rhordoan/drag-improved-checkpoints \
    /tmp/phase1_best_cwq.pt \
    checkpoints_cwq_subgraph/phase1_best.pt
```

### Authentication Error
```bash
# Re-login
huggingface-cli logout
huggingface-cli login
```

### Repository Already Exists
If the repository name is taken, use:
```bash
# Use a different name
huggingface-cli repo create drag-improved-checkpoints-v1 --type model
```

Then update `REPO_ID` in:
- `scripts/download_checkpoints.py`
- `scripts/upload_to_huggingface.sh`

## After Upload

Once uploaded, update the repository documentation to point to your Hugging Face repository URL in:
- `README.md` (search for "huggingface.co/rhordoan")
- `CHECKPOINTS.md`
- `scripts/download_checkpoints.py`

Then commit and push to GitHub:
```bash
git add .
git commit -m "Update Hugging Face links"
git push origin main
```

---

**Note:** If you're not `rhordoan`, you'll need to update all references to `rhordoan/drag-improved-checkpoints` to your own username/repo in the scripts and documentation.

