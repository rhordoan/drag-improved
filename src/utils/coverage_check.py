import json
import os
import argparse

def check_coverage(dataset_file, entities_map):
    """
    Checks what percentage of question entities in the dataset are present in the KG.
    """
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found.")
        return

    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    total_entities = 0
    found_entities = 0

    for item in data:
        # Note: 'entities' field depends on the dataset format
        # For WebQSP (stanfornlp), we might need to extract from URL or separate metadata
        # Assuming entities are provided in an 'entities' list of MIDs
        q_entities = item.get('entities', [])
        for ent in q_entities:
            total_entities += 1
            if ent in entities_map:
                found_entities += 1

    coverage = (found_entities / total_entities * 100) if total_entities > 0 else 0
    print(f"Coverage for {dataset_file}: {coverage:.2f}% ({found_entities}/{total_entities})")
    
    if coverage < 90:
        print("WARNING: Coverage is less than 90%. You may have the wrong KG dump.")
    return coverage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check entity coverage against KG.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset jsonl/json.")
    parser.add_argument("--entities", type=str, required=True, help="Path to entities map (tab-separated or similar).")
    args = parser.parse_args()
    
    # Load entities_map from the provided file
    if not os.path.exists(args.entities):
        print(f"Entities file {args.entities} not found.")
    else:
        entities_set = set()
        print(f"Loading entities from {args.entities}...")
        with open(args.entities, 'r', encoding='utf-8') as f:
            for line in f:
                # Handle both single MID per line and tab-separated triples
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    entities_set.add(parts[0])
                if len(parts) >= 3:
                    entities_set.add(parts[2])
        
        check_coverage(args.dataset, entities_set)

