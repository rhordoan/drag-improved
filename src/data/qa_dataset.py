import json
import os
from torch.utils.data import Dataset

class QADataset(Dataset):
    """
    Dataset for loading WebQSP or CWQ question-answer pairs.
    """
    def __init__(self, data_file):
        self.data = []
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Dataset file {data_file} not found.")
        
        print(f"Loading dataset from {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Normalize field names across datasets
                question = item.get('question', item.get('Question', ''))
                answers = item.get('answers', item.get('Answers', []))
                
                # Handle different answer formats
                if isinstance(answers, list) and len(answers) > 0:
                    if isinstance(answers[0], dict):
                        # CWQ format: list of dicts with 'answer' key
                        answers = [a.get('answer', str(a)) for a in answers]
                    # else: WebQSP format: list of strings
                
                # Get entities if available (for KG linking)
                entities = item.get('entities', item.get('TopicEntityMid', []))
                if isinstance(entities, str):
                    entities = [entities]
                
                self.data.append({
                    'question': question,
                    'answers': answers,
                    'entities': entities,
                    'id': item.get('id', item.get('ID', len(self.data)))
                })
        
        print(f"Loaded {len(self.data)} examples.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Returns lists for questions and answers (suitable for tokenizers).
    """
    questions = [item['question'] for item in batch]
    answers = [item['answers'][0] if item['answers'] else '' for item in batch]
    entities = [item['entities'] for item in batch]
    ids = [item['id'] for item in batch]
    
    return {
        'questions': questions,
        'answers': answers,
        'entities': entities,
        'ids': ids
    }

