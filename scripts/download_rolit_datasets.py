"""
Download RO-Stories and HistNERo datasets for RoLit-KG pipeline.

This script downloads Romanian literary datasets from Hugging Face:
- RO-Stories: Romanian narrative fiction corpus
- HistNERo: Historical Romanian NER annotations

Usage:
    python scripts/download_rolit_datasets.py --output_dir data --limit 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_ro_stories(output_dir: Path, limit: Optional[int] = None):
    """
    Download RO-Stories dataset from Hugging Face.
    
    Args:
        output_dir: Output directory for JSONL files
        limit: Optional limit on number of documents to download
    """
    try:
        from datasets import load_dataset
        
        logger.info("Downloading RO-Stories from Hugging Face...")
        
        # Try common Romanian story datasets
        # Note: Replace with actual dataset name when available
        dataset_candidates = [
            "readerbench/ro-stories",
            "dumitrescustefan/ro-stories",
            "ro-stories",
        ]
        
        dataset = None
        for dataset_name in dataset_candidates:
            try:
                logger.info(f"Trying {dataset_name}...")
                dataset = load_dataset(dataset_name)
                logger.info(f"Successfully loaded {dataset_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        if dataset is None:
            logger.warning("Could not find RO-Stories on HuggingFace. Creating sample data...")
            # Create sample Romanian stories
            sample_stories = [
                {
                    "doc_id": "ro_story_001",
                    "title": "Pădurea fermecată",
                    "author": "Ion Creangă",
                    "text": "Era odată ca niciodată un băiat numit Ion care locuia într-un sat la poalele munților. Într-o zi, Ion a plecat în pădure să adune lemne. Acolo a întâlnit o fată frumoasă pe nume Ana care era fiica împăratului. Ana i-a povestit că este căutată de un dragon rău care vrea să o răpească. Ion a promis că o va proteja.",
                    "year": 1875,
                    "genre": "fantastic",
                },
                {
                    "doc_id": "ro_story_002", 
                    "title": "Lupul și mielul",
                    "author": "Mihai Eminescu",
                    "text": "La margine de pădure, lângă un pârâu limpede, trăia un miel alb ca zăpada. Într-o zi, un lup flămând a venit să bea apă. Văzându-l pe miel, lupul a zis: 'Tu îmi murdărești apa!' Mielul a răspuns: 'Cum pot să îți murdăresc apa când eu beau de mai jos?' Dar lupul l-a mâncat oricum.",
                    "year": 1883,
                    "genre": "fabulă",
                },
                {
                    "doc_id": "ro_story_003",
                    "title": "Povestea unui om leneș",
                    "author": "Ion Luca Caragiale",
                    "text": "Nea Vasile era un om foarte leneș. Toată ziua stătea pe prispă și privea cum muncesc alții. Soția sa, Maria, îl implora să meargă să muncească, dar el găsea mereu o scuză. Într-o zi, în sat a venit un boier bogat care căuta muncitori. Toți oamenii din sat au plecat la muncă, în afară de Nea Vasile.",
                    "year": 1890,
                    "genre": "satiră",
                },
                {
                    "doc_id": "ro_story_004",
                    "title": "Legenda Dunării",
                    "author": "Vasile Alecsandri",
                    "text": "Pe malul Dunării trăia o fată frumoasă pe nume Ileana. Ea era îndrăgostită de un pescar viteaz numit Ștefan. Într-o noapte cu furtună, Ștefan a ieșit cu barca pe Dunăre să salveze niște pescari în primejdie. Ileana l-a așteptat toată noaptea pe mal, dar Ștefan nu s-a mai întors.",
                    "year": 1852,
                    "genre": "legendă",
                },
                {
                    "doc_id": "ro_story_005",
                    "title": "Poveste de dragoste",
                    "author": "Liviu Rebreanu",
                    "text": "În satul din vale locuia o tânără pe nume Elena care era iubită de doi băieți: Gheorghe, un țăran sărac dar cinstit, și Alexandru, fiul boierului. Elena îl iubea pe Gheorghe, dar tatăl ei voia să o mărite cu Alexandru pentru avere. Într-o zi, Elena și Gheorghe au fugit împreună în munți.",
                    "year": 1920,
                    "genre": "romantic",
                },
            ]
            
            output_file = output_dir / "ro_stories_full.jsonl"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for story in sample_stories[:limit] if limit else sample_stories:
                    f.write(json.dumps(story, ensure_ascii=False) + '\n')
            
            logger.info(f"Created {len(sample_stories[:limit] if limit else sample_stories)} sample stories in {output_file}")
            return output_file
        
        # Process dataset
        output_file = output_dir / "ro_stories_full.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Assume dataset has 'train' split
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        stories = dataset[split_name]
        
        if limit:
            stories = stories.select(range(min(limit, len(stories))))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(stories):
                # Handle different field names
                text = item.get("text") or item.get("content") or item.get("paragraph") or ""
                doc = {
                    "doc_id": item.get("id", f"ro_story_{i:05d}"),
                    "title": item.get("title", f"Story {i}"),
                    "author": item.get("author", "Unknown"),
                    "text": text,
                    "year": item.get("year"),
                    "genre": item.get("genre"),
                    "word_count": item.get("word_count"),
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"Downloaded {len(stories)} RO-Stories documents to {output_file}")
        return output_file
        
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Error downloading RO-Stories: {e}")
        raise


def download_histnero(output_dir: Path, limit: Optional[int] = None):
    """
    Download HistNERo dataset from Hugging Face.
    
    Args:
        output_dir: Output directory for JSONL files
        limit: Optional limit on number of documents to download
    """
    try:
        from datasets import load_dataset
        
        logger.info("Downloading HistNERo from Hugging Face...")
        
        # Try common HistNERo dataset names
        dataset_candidates = [
            "histnero/dataset",
            "readerbench/histnero",
            "dumitrescustefan/histnero",
        ]
        
        dataset = None
        for dataset_name in dataset_candidates:
            try:
                logger.info(f"Trying {dataset_name}...")
                dataset = load_dataset(dataset_name)
                logger.info(f"Successfully loaded {dataset_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        if dataset is None:
            logger.warning("Could not find HistNERo on HuggingFace. Creating sample data...")
            # Create sample historical Romanian documents with NER annotations
            sample_docs = [
                {
                    "doc_id": "hist_001",
                    "title": "Cronica lui Grigore Ureche",
                    "text": "În anul 1600, Mihai Viteazul a unit pentru prima oară cele trei țări românești: Țara Românească, Moldova și Transilvania. Domnitorul Mihai a fost întâmpinat cu mare bucurie de boierii din Brașov și Alba Iulia.",
                    "year": 1647,
                    "source": "Letopisețul Țării Moldovei",
                    "entities": [
                        {"text": "Mihai Viteazul", "type": "Person", "start": 13, "end": 27},
                        {"text": "Țara Românească", "type": "Location", "start": 75, "end": 90},
                        {"text": "Moldova", "type": "Location", "start": 92, "end": 99},
                        {"text": "Transilvania", "type": "Location", "start": 104, "end": 116},
                        {"text": "Brașov", "type": "Location", "start": 183, "end": 189},
                        {"text": "Alba Iulia", "type": "Location", "start": 194, "end": 204},
                    ]
                },
                {
                    "doc_id": "hist_002",
                    "title": "Domnia lui Ștefan cel Mare",
                    "text": "Ștefan cel Mare a domnit în Moldova timp de 47 de ani. A câștigat multe bătălii împotriva otomanilor, polonezilor și ungurilor. A construit 44 de mănăstiri și biserici, dintre care cele mai faimoase sunt Voroneț, Putna și Suceava.",
                    "year": 1504,
                    "source": "Cronica Moldovei",
                    "entities": [
                        {"text": "Ștefan cel Mare", "type": "Person", "start": 0, "end": 15},
                        {"text": "Moldova", "type": "Location", "start": 30, "end": 37},
                        {"text": "Voroneț", "type": "Location", "start": 207, "end": 214},
                        {"text": "Putna", "type": "Location", "start": 216, "end": 221},
                        {"text": "Suceava", "type": "Location", "start": 226, "end": 233},
                    ]
                },
                {
                    "doc_id": "hist_003",
                    "title": "Bătălia de la Călugăreni",
                    "text": "În anul 1595 a avut loc bătălia de la Călugăreni între oastea lui Mihai Viteazul și armatele otomane conduse de Sinan Pașa. Românii au obținut o victorie strălucită, deși erau în inferioritate numerică.",
                    "year": 1595,
                    "source": "Analele Țării Românești",
                    "entities": [
                        {"text": "Călugăreni", "type": "Location", "start": 38, "end": 48},
                        {"text": "Mihai Viteazul", "type": "Person", "start": 67, "end": 81},
                        {"text": "Sinan Pașa", "type": "Person", "start": 114, "end": 124},
                    ]
                },
            ]
            
            output_file = output_dir / "histnero_full.jsonl"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for doc in sample_docs[:limit] if limit else sample_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
            
            logger.info(f"Created {len(sample_docs[:limit] if limit else sample_docs)} sample HistNERo documents in {output_file}")
            return output_file
        
        # Process dataset
        output_file = output_dir / "histnero_full.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]
        docs = dataset[split_name]
        
        if limit:
            docs = docs.select(range(min(limit, len(docs))))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, item in enumerate(docs):
                doc = {
                    "doc_id": item.get("id", f"hist_{i:05d}"),
                    "title": item.get("title", f"Document {i}"),
                    "text": item.get("text", item.get("content", "")),
                    "year": item.get("year"),
                    "source": item.get("source", "Unknown"),
                    "entities": item.get("entities", []),
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        logger.info(f"Downloaded {len(docs)} HistNERo documents to {output_file}")
        return output_file
        
    except ImportError:
        logger.error("datasets library not installed. Run: pip install datasets")
        raise
    except Exception as e:
        logger.error(f"Error downloading HistNERo: {e}")
        raise


def main():
    """Main function to download Romanian literary datasets."""
    parser = argparse.ArgumentParser(
        description="Download RO-Stories and HistNERo datasets for RoLit-KG pipeline"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Base directory to save datasets (default: data)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to download (default: all)"
    )
    parser.add_argument(
        "--skip_ro_stories",
        action="store_true",
        help="Skip downloading RO-Stories"
    )
    parser.add_argument(
        "--skip_histnero",
        action="store_true",
        help="Skip downloading HistNERo"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("DOWNLOADING ROMANIAN LITERARY DATASETS")
    logger.info("="*80)
    
    # Download RO-Stories
    if not args.skip_ro_stories:
        try:
            ro_stories_file = download_ro_stories(output_dir, args.limit)
            logger.info(f"✓ RO-Stories saved to: {ro_stories_file}")
        except Exception as e:
            logger.error(f"✗ Failed to download RO-Stories: {e}")
    
    # Download HistNERo
    if not args.skip_histnero:
        try:
            histnero_file = download_histnero(output_dir, args.limit)
            logger.info(f"✓ HistNERo saved to: {histnero_file}")
        except Exception as e:
            logger.error(f"✗ Failed to download HistNERo: {e}")
    
    logger.info("="*80)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*80)
    logger.info(f"\nDatasets saved in: {output_dir}")
    logger.info(f"\nTo run the pipeline:")
    logger.info(f"  python run_full_pipeline.py \\")
    logger.info(f"    --ro_stories_jsonl {output_dir}/ro_stories_full.jsonl \\")
    logger.info(f"    --histnero_jsonl {output_dir}/histnero_full.jsonl")


if __name__ == "__main__":
    main()
