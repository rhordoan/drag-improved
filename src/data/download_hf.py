import os
import argparse
from datasets import load_dataset

def download_datasets(output_base):
    # Create directories
    webqsp_dir = os.path.join(output_base, 'webqsp')
    cwq_dir = os.path.join(output_base, 'cwq')
    os.makedirs(webqsp_dir, exist_ok=True)
    os.makedirs(cwq_dir, exist_ok=True)

    # 1. WebQSP (stanfordnlp/web_questions)
    print("Downloading WebQSP (stanfordnlp/web_questions)...")
    try:
        webqsp = load_dataset("stanfordnlp/web_questions")
        webqsp['train'].to_json(os.path.join(webqsp_dir, 'WebQSP.train.json'))
        webqsp['test'].to_json(os.path.join(webqsp_dir, 'WebQSP.test.json'))
        print(f"Saved WebQSP to {webqsp_dir}")
    except Exception as e:
        print(f"Error downloading stanfordnlp/web_questions: {e}")

    # 2. Complex WebQuestions (drt/complex_web_questions -> rmanluo/RoG-cwq)
    print("Downloading Complex WebQuestions (falling back to rmanluo/RoG-cwq due to loading script issues)...")
    try:
        # drt/complex_web_questions is unsupported in datasets 3.0+ (loading scripts removed)
        # using rmanluo/RoG-cwq which is the standard pre-processed version for Graph QA
        cwq = load_dataset("rmanluo/RoG-cwq")
        cwq['train'].to_json(os.path.join(cwq_dir, 'ComplexWebQuestions_train.json'))
        
        split_name = 'validation' if 'validation' in cwq else 'dev' if 'dev' in cwq else None
        if split_name:
            cwq[split_name].to_json(os.path.join(cwq_dir, 'ComplexWebQuestions_dev.json'))
        
        cwq['test'].to_json(os.path.join(cwq_dir, 'ComplexWebQuestions_test.json'))
        print(f"Saved CWQ to {cwq_dir}")
    except Exception as e:
        print(f"Error downloading CWQ: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WebQSP and CWQ datasets from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="data", help="Base directory to save datasets.")
    args = parser.parse_args()
    
    download_datasets(args.output_dir)
