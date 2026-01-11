"""
Monitor the progress of the full corpus RoLit-KG pipeline run.

Usage:
    python monitor_pipeline.py --log_file logs/full_corpus_run.log
"""

import argparse
import time
import re
from pathlib import Path


def parse_log_file(log_path):
    """Parse log file to extract progress information."""
    if not Path(log_path).exists():
        return None
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Extract key metrics
    info = {
        'started': False,
        'stage': None,
        'documents': None,
        'chunks': None,
        'entities_pre': None,
        'entities_post': None,
        'relations': None,
        'elapsed': None,
        'completed': False,
    }
    
    # Check if started
    if 'ROLIT-KG OPTIMIZED PIPELINE' in content:
        info['started'] = True
    
    # Extract stage
    stages = [
        'Stage 1: Ingesting',
        'Stage 2: Normalizing',
        'Stage 3: Chunking',
        'Stage 4: Extracting',
        'Stage 5: Validating',
        'Stage 6: Resolving',
        'Stage 7: Grounding',
        'Stage 8: Persisting',
        'Stage 9: Computing metrics',
        'Stage 10: Computing graph analytics',
        'Stage 11: Generating Neo4j',
    ]
    
    for stage in reversed(stages):
        if stage in content:
            info['stage'] = stage
            break
    
    # Extract metrics
    doc_match = re.search(r'Total documents: (\d+)', content)
    if doc_match:
        info['documents'] = int(doc_match.group(1))
    
    chunk_match = re.search(r'Created (\d+) chunks', content)
    if chunk_match:
        info['chunks'] = int(chunk_match.group(1))
    
    entities_pre_match = re.search(r'Extracted (\d+) mentions, (\d+) entities', content)
    if entities_pre_match:
        info['entities_pre'] = int(entities_pre_match.group(2))
    
    entities_post_match = re.search(r'Entities \(post-resolution\): (\d+)', content)
    if entities_post_match:
        info['entities_post'] = int(entities_post_match.group(1))
    
    relations_match = re.search(r'Relations: (\d+)', content)
    if relations_match:
        info['relations'] = int(relations_match.group(1))
    
    elapsed_match = re.search(r'Total time: ([\d.]+)s', content)
    if elapsed_match:
        info['elapsed'] = float(elapsed_match.group(1))
    
    # Check if completed
    if 'PIPELINE COMPLETE' in content:
        info['completed'] = True
    
    return info


def format_time(seconds):
    """Format seconds as human-readable string."""
    if seconds is None:
        return "N/A"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_progress(info):
    """Print formatted progress information."""
    print("\n" + "=" * 80)
    print("ROLIT-KG FULL CORPUS PIPELINE - PROGRESS MONITOR")
    print("=" * 80)
    
    if not info or not info['started']:
        print("Status: Pipeline not started yet...")
        return
    
    print(f"Status: {'COMPLETED ✓' if info['completed'] else 'RUNNING...'}")
    print(f"Current Stage: {info['stage'] or 'Starting...'}")
    print()
    
    if info['documents']:
        print(f"Documents: {info['documents']:,}")
    
    if info['chunks']:
        print(f"Chunks: {info['chunks']:,}")
    
    if info['entities_pre']:
        print(f"Entities (extracted): {info['entities_pre']:,}")
    
    if info['entities_post']:
        reduction = 100 * (1 - info['entities_post'] / info['entities_pre']) if info['entities_pre'] else 0
        print(f"Entities (resolved): {info['entities_post']:,} ({reduction:.1f}% reduction)")
    
    if info['relations']:
        print(f"Relations: {info['relations']:,}")
    
    if info['elapsed']:
        print(f"\nElapsed time: {format_time(info['elapsed'])}")
    
    print("=" * 80)


def monitor_pipeline(log_path, refresh_interval=10):
    """Monitor pipeline progress in real-time."""
    print(f"Monitoring log file: {log_path}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring...")
    
    try:
        while True:
            info = parse_log_file(log_path)
            print_progress(info)
            
            if info and info['completed']:
                print("\nPipeline completed successfully!")
                break
            
            time.sleep(refresh_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor RoLit-KG pipeline progress")
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/full_corpus_run.log",
        help="Path to log file"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print status once and exit"
    )
    
    args = parser.parse_args()
    
    if args.once:
        info = parse_log_file(args.log_file)
        print_progress(info)
    else:
        monitor_pipeline(args.log_file, args.refresh)


if __name__ == "__main__":
    main()
