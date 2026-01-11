# RoLit-KG Analytics Guide

This guide shows how to analyze the RoLit-KG outputs produced by `run_full_pipeline.py`.

## Where the outputs live

A run writes to:

- `outputs/<run>/artifacts/`
  - `docs.jsonl`, `chunks.jsonl`, `mentions.jsonl`, `entities.jsonl`, `relations.jsonl`
  - `*_raw.jsonl` versions when `--checkpoint_after_extract` is used
- `outputs/<run>/reports/`
  - `metrics.json`, `report.md`
- `outputs/<run>/cypher/`
  - `constraints.cypher`, `load.cypher`
- `outputs/<run>/summary.json`

## Quick sanity checks

### 1) Overall counts

Open:
- `outputs/<run>/summary.json`
- `outputs/<run>/reports/report.md`

These are the fastest way to confirm:
- documents/chunks processed
- extracted mentions/entities/relations
- post-resolution entity count

### 2) Spot-check high-degree hubs (noise)

Check the “Top hubs” section in `outputs/<run>/reports/report.md`.

If you see hubs dominated by stopwords or punctuation artifacts, tune:
- `src/pipeline/extract.py` stopwords / regex rules
- the resolution mode (`lexical` vs embeddings-based)

## Programmatic analytics (Python)

### Load artifacts

```python
import json
from pathlib import Path

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

run_dir = Path("outputs/rolit_kg_full_corpus")

entities = list(read_jsonl(run_dir / "artifacts" / "entities.jsonl"))
relations = list(read_jsonl(run_dir / "artifacts" / "relations.jsonl"))

print("entities:", len(entities))
print("relations:", len(relations))
```

### Run built-in graph analytics

RoLit-KG includes a lightweight analytics module:

```python
from src.pipeline.analytics import compute_graph_analytics

metrics = compute_graph_analytics(
    entities=entities,
    relations=relations,
    output_path=str(run_dir / "reports" / "analytics.json"),
)

print(metrics.node_count, metrics.edge_count)
```

Notes for large graphs:
- Analytics can be memory-heavy if you include extremely dense relation types.
- If you emit `CANDIDATE_SAME_AS`, consider filtering it out for analytics.

### Filter relations for cleaner analytics

```python
filtered_relations = [r for r in relations if r.get("predicate") != "CANDIDATE_SAME_AS"]

from src.pipeline.analytics import compute_graph_analytics
compute_graph_analytics(
    entities=entities,
    relations=filtered_relations,
    output_path=str(run_dir / "reports" / "analytics_no_candidates.json"),
)
```

## Neo4j analytics (Cypher)

After importing `cypher/constraints.cypher` and `cypher/load.cypher`:

### Top entities by degree

```cypher
MATCH (e:Entity)-[r]-()
RETURN e.canonical_name AS name, count(r) AS degree
ORDER BY degree DESC
LIMIT 25;
```

### Interaction graph among characters only

```cypher
MATCH (a:Entity)-[r:INTERACTS_WITH]->(b:Entity)
WHERE a.entity_type = 'Character' AND b.entity_type = 'Character'
RETURN a.canonical_name, b.canonical_name, r.confidence
LIMIT 50;
```

### Find works with most mentions

```cypher
MATCH (m:Mention)
RETURN m.doc_id AS doc_id, count(*) AS mentions
ORDER BY mentions DESC
LIMIT 20;
```

## Recommended workflow for “production” analysis

- Use `--checkpoint_after_extract` so extraction artifacts are always preserved.
- On full corpus, start with `--resolution_mode lexical` to produce a stable KG quickly.
- Once the KG is stable, upgrade resolution to embeddings-based ANN/kNN (so it scales beyond O(n²)).

