#!/usr/bin/env python3
"""
Quick script to create a validation split from training heuristics.
Takes last N examples as validation (preserves order for reproducibility).
"""
import json
import argparse
from pathlib import Path

def create_val_split(train_path, val_size=500, output_dir=None):
    """Split last val_size examples into validation set."""
    train_path = Path(train_path)
    
    if output_dir is None:
        output_dir = train_path.parent
    else:
        output_dir = Path(output_dir)
    
    # Read all examples
    print(f"Reading {train_path}...")
    with open(train_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f if line.strip()]
    
    total = len(examples)
    print(f"Total examples: {total}")
    
    if val_size >= total:
        print(f"ERROR: val_size ({val_size}) >= total examples ({total})")
        return
    
    # Split: first (total - val_size) for train, last val_size for val
    train_examples = examples[:-val_size]
    val_examples = examples[-val_size:]
    
    # Generate output paths
    stem = train_path.stem
    if stem.endswith('_train'):
        train_out = output_dir / f"{stem}.jsonl"
        val_out = output_dir / f"{stem.replace('_train', '_val')}.jsonl"
    else:
        train_out = output_dir / f"{stem}_train.jsonl"
        val_out = output_dir / f"{stem}_val.jsonl"
    
    # Write new train file
    print(f"Writing {len(train_examples)} train examples to {train_out}")
    with open(train_out, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    # Write val file
    print(f"Writing {len(val_examples)} val examples to {val_out}")
    with open(val_out, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"\nDone!")
    print(f"  Train: {train_out} ({len(train_examples)} examples)")
    print(f"  Val:   {val_out} ({len(val_examples)} examples)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val split")
    parser.add_argument("--input", type=str, required=True,
                        help="Input heuristics JSONL file")
    parser.add_argument("--val_size", type=int, default=500,
                        help="Number of examples for validation (default: 500)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    create_val_split(args.input, args.val_size, args.output_dir)











