#!/usr/bin/env python3
"""
Generate D-RAG training heuristics from WebQSP semantic parses.
Extracts paths from SPARQL queries following the paper's approach.
"""

import json
from pathlib import Path
import re

def parse_sparql_path(sparql, topic_mid):
    """
    Extract entity path from SPARQL query.
    Returns list of (entity, relation) tuples.
    """
    # Find all triple patterns like: ns:m.xxx ns:relation.path ?y
    # Pattern: ns:ENTITY ns:RELATION ?var
    pattern = r'ns:(m\.\w+|[ge]\.\w+)\s+ns:([\w\.]+)'
    matches = re.findall(pattern, sparql)
    
    path_entities = [topic_mid]
    relations = []
    
    for entity, relation in matches:
        if not entity.startswith(topic_mid.split('.')[0]):  # Skip if not proper entity
            relations.append(relation)
    
    return path_entities, relations

def main():
    webqsp_train = Path("data/WebQSP/data/WebQSP.train.json")
    output_path = Path("data/train_heuristics_webqsp_full.jsonl")
    
    print("=" * 70)
    print("Generating WebQSP Heuristics from SPARQL Semantic Parses")
    print("=" * 70)
    print()
    print("Source: ACL 2016 - WebQuestionsSP Dataset")
    print("Reference: [Yih, Richardson, Meek, Chang & Suh, 2016]")
    print()
    
    # Load WebQSP
    print(f"Reading {webqsp_train}...")
    with open(webqsp_train) as f:
        data = json.load(f)
    
    questions_data = data['Questions']
    print(f"  Total questions: {len(questions_data)}")
    print()
    
    # Generate heuristics
    heuristics = []
    skipped = 0
    relation_chains = []
    
    for q_data in questions_data:
        question = q_data.get('RawQuestion', '')
        parses = q_data.get('Parses', [])
        
        if not question or not parses:
            skipped += 1
            continue
        
        # Use first parse
        parse = parses[0]
        topic_mid = parse.get('TopicEntityMid', '')
        topic_name = parse.get('TopicEntityName', '')
        answers = parse.get('Answers', [])
        inferential_chain = parse.get('InferentialChain', [])
        sparql = parse.get('Sparql', '')
        
        if not topic_mid or not answers:
            skipped += 1
            continue
        
        # Build path from topic entity through relations to answer entities
        # InferentialChain gives us the relations
        paths = []
        
        # Extract answer MIDs from SPARQL or use answer text
        answer_entities = []
        for answer in answers[:3]:  # Limit to 3 answers
            answer_mid = answer.get('AnswerArgument', '')
            answer_text = answer.get('Answer', '')
            
            if answer_mid:
                answer_entities.append(answer_mid)
            elif answer_text:
                # Create normalized entity ID
                normalized = answer_text.lower().replace(' ', '_')
                answer_entities.append(normalized)
        
        # Create paths: topic -> answer (simplified)
        for ans_ent in answer_entities[:3]:
            if inferential_chain:
                # Path with intermediate relations
                path = [topic_mid] + inferential_chain + [ans_ent]
            else:
                # Direct path
                path = [topic_mid, ans_ent]
            paths.append(path)
        
        # Collect relation chains for stats
        if inferential_chain:
            relation_chains.append(len(inferential_chain))
        
        heuristic = {
            "question": question,
            "paths": paths,
            "answer": answers[0].get('Answer', '') if answers else "",
            "topic_entity": topic_mid,
            "topic_name": topic_name,
            "relations": inferential_chain,
            "num_answers": len(answers)
        }
        heuristics.append(heuristic)
    
    print(f"Generated heuristics: {len(heuristics)}")
    print(f"Skipped (missing data): {skipped}")
    print()
    
    # Stats
    if relation_chains:
        avg_chain = sum(relation_chains) / len(relation_chains)
        max_chain = max(relation_chains)
        print(f"Relation chain stats:")
        print(f"  Average length: {avg_chain:.1f}")
        print(f"  Max length: {max_chain}")
        print()
    
    # Write heuristics
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        for h in heuristics:
            f.write(json.dumps(h) + '\n')
    
    print()
    print("=" * 70)
    print(f"✓ Generated {len(heuristics)} heuristics from WebQSP")
    print()
    print("Note: To match FB15k-237 entities, you need a WebQSP-aligned")
    print("      Freebase subset. The paper likely used a specific KB dump.")
    print()
    print("For now, continue training with CWQ (27K samples) which has")
    print("built-in gold subgraphs in the 'graph' field.")
    print("=" * 70)

if __name__ == "__main__":
    main()

