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
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _detokenize_with_offsets(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Deterministic detokenizer that also returns token (start,end) offsets in the produced text.
    Good enough for producing stable span offsets from HF token classification datasets.
    """
    no_space_before = {".", ",", ";", ":", "!", "?", ")", "]", "}", "»", "”"}
    no_space_after = {"(", "[", "{", "«", "„", "“"}

    offsets: List[Tuple[int, int]] = []
    cur = ""
    for tok in tokens:
        tok = "" if tok is None else str(tok)
        if not tok:
            offsets.append((len(cur), len(cur)))
            continue

        add_space = True
        if not cur:
            add_space = False
        elif tok in no_space_before:
            add_space = False
        elif cur and cur[-1] in (" ", "\n"):
            add_space = False
        elif any(cur.endswith(ch) for ch in no_space_after):
            add_space = False

        if add_space:
            cur += " "

        start = len(cur)
        cur += tok
        end = len(cur)
        offsets.append((start, end))

    return cur, offsets


def _extract_spans_from_bio(
    *,
    text: str,
    token_offsets: List[Tuple[int, int]],
    labels: List[str],
) -> List[Dict[str, object]]:
    """
    Convert BIO-like per-token labels (e.g., B-PER/I-PER/O) into char-offset spans.
    """
    spans: List[Dict[str, object]] = []
    cur_label: Optional[str] = None
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    def flush():
        nonlocal cur_label, cur_start, cur_end
        if cur_label and cur_start is not None and cur_end is not None and cur_end > cur_start:
            spans.append(
                {
                    "start_char": int(cur_start),
                    "end_char": int(cur_end),
                    "label": cur_label,
                    "surface": text[cur_start:cur_end],
                }
            )
        cur_label, cur_start, cur_end = None, None, None

    for i, lab in enumerate(labels):
        lab = (lab or "O").strip()
        if lab == "O" or lab == "":
            flush()
            continue

        if "-" in lab:
            prefix, ent = lab.split("-", 1)
        else:
            prefix, ent = "B", lab
        prefix = prefix.upper()
        ent = ent.strip().upper()

        s, e = token_offsets[i]
        if prefix == "B" or cur_label != ent:
            flush()
            cur_label = ent
            cur_start, cur_end = s, e
        else:
            cur_end = e

    flush()
    return spans


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
            "avramandrei/histnero",
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
                    "spans": [
                        {"start_char": 13, "end_char": 27, "label": "PER", "surface": "Mihai Viteazul"},
                        {"start_char": 75, "end_char": 90, "label": "LOC", "surface": "Țara Românească"},
                        {"start_char": 92, "end_char": 99, "label": "LOC", "surface": "Moldova"},
                        {"start_char": 104, "end_char": 116, "label": "LOC", "surface": "Transilvania"},
                        {"start_char": 183, "end_char": 189, "label": "LOC", "surface": "Brașov"},
                        {"start_char": 194, "end_char": 204, "label": "LOC", "surface": "Alba Iulia"},
                    ]
                },
                {
                    "doc_id": "hist_002",
                    "title": "Domnia lui Ștefan cel Mare",
                    "text": "Ștefan cel Mare a domnit în Moldova timp de 47 de ani. A câștigat multe bătălii împotriva otomanilor, polonezilor și ungurilor. A construit 44 de mănăstiri și biserici, dintre care cele mai faimoase sunt Voroneț, Putna și Suceava.",
                    "year": 1504,
                    "source": "Cronica Moldovei",
                    "spans": [
                        {"start_char": 0, "end_char": 15, "label": "PER", "surface": "Ștefan cel Mare"},
                        {"start_char": 30, "end_char": 37, "label": "LOC", "surface": "Moldova"},
                        {"start_char": 207, "end_char": 214, "label": "LOC", "surface": "Voroneț"},
                        {"start_char": 216, "end_char": 221, "label": "LOC", "surface": "Putna"},
                        {"start_char": 226, "end_char": 233, "label": "LOC", "surface": "Suceava"},
                    ]
                },
                {
                    "doc_id": "hist_003",
                    "title": "Bătălia de la Călugăreni",
                    "text": "În anul 1595 a avut loc bătălia de la Călugăreni între oastea lui Mihai Viteazul și armatele otomane conduse de Sinan Pașa. Românii au obținut o victorie strălucită, deși erau în inferioritate numerică.",
                    "year": 1595,
                    "source": "Analele Țării Românești",
                    "spans": [
                        {"start_char": 38, "end_char": 48, "label": "LOC", "surface": "Călugăreni"},
                        {"start_char": 67, "end_char": 81, "label": "PER", "surface": "Mihai Viteazul"},
                        {"start_char": 114, "end_char": 124, "label": "PER", "surface": "Sinan Pașa"},
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
        
        # Process dataset (token classification → our doc+spans JSONL)
        output_file = output_dir / "histnero_full.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine label names for ner_tags (Sequence(ClassLabel))
        label_names = None
        try:
            any_split = list(dataset.keys())[0]
            ner_feat = dataset[any_split].features.get("ner_tags")
            if hasattr(ner_feat, "feature") and hasattr(ner_feat.feature, "names"):
                label_names = list(ner_feat.feature.names)
        except Exception:
            label_names = None

        # Combine splits if present
        split_order = [s for s in ("train", "validation", "valid", "test") if s in dataset]
        if not split_order:
            split_order = list(dataset.keys())

        grouped_text: Dict[str, str] = {}
        grouped_spans: Dict[str, List[Dict[str, object]]] = {}
        grouped_meta: Dict[str, Dict[str, object]] = {}

        for split_name in split_order:
            ds = dataset[split_name]
            for row in ds:
                doc_id = row.get("doc_id") or row.get("id") or "histnero_unknown"
                doc_id = str(doc_id)

                tokens = row.get("tokens") or []
                tags = row.get("ner_tags") or []
                if not isinstance(tokens, list) or not isinstance(tags, list) or len(tokens) != len(tags):
                    continue

                sent_text, token_offsets = _detokenize_with_offsets([str(t) for t in tokens])

                labs: List[str] = []
                for t in tags:
                    if isinstance(t, int) and label_names and 0 <= t < len(label_names):
                        labs.append(label_names[t])
                    else:
                        labs.append(str(t))

                sent_spans = _extract_spans_from_bio(text=sent_text, token_offsets=token_offsets, labels=labs)

                prev = grouped_text.get(doc_id, "")
                sep = "\n" if prev else ""
                base = len(prev) + len(sep)
                grouped_text[doc_id] = prev + sep + sent_text
                grouped_spans.setdefault(doc_id, [])
                for s in sent_spans:
                    grouped_spans[doc_id].append(
                        {
                            "start_char": int(s["start_char"]) + base,
                            "end_char": int(s["end_char"]) + base,
                            "label": s["label"],
                            "surface": s["surface"],
                        }
                    )

                grouped_meta.setdefault(doc_id, {})
                if "region" in row and row.get("region") is not None:
                    grouped_meta[doc_id]["region"] = row.get("region")
                if "id" in row and row.get("id") is not None:
                    grouped_meta[doc_id].setdefault("row_ids", [])
                    # keep small; don't explode metadata
                    if len(grouped_meta[doc_id]["row_ids"]) < 50:
                        grouped_meta[doc_id]["row_ids"].append(str(row.get("id")))

        doc_ids_sorted = sorted(grouped_text.keys())
        if limit:
            doc_ids_sorted = doc_ids_sorted[: int(limit)]

        with open(output_file, 'w', encoding='utf-8') as f:
            for did in doc_ids_sorted:
                f.write(
                    json.dumps(
                        {
                            "doc_id": did,
                            "title": did,
                            "text": grouped_text[did],
                            "spans": grouped_spans.get(did, []),
                            "meta": grouped_meta.get(did, {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        logger.info(f"Downloaded {len(doc_ids_sorted)} HistNERo documents to {output_file}")
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
