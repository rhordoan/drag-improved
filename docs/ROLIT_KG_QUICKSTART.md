# RoLit-KG Quick Start Guide

## 5-Minute Tutorial

### Step 1: Verify Installation

```bash
cd C:\Users\Roberto\Documents\Github\drag-improved
python --version  # Should be 3.8+
```

### Step 2: Install Dependencies

```bash
pip install pyyaml datasets
# Optional for Neo4j loading:
pip install neo4j
```

### Step 3: Run Sample Pipeline

```bash
set PYTHONPATH=%CD%
python scripts\run_rolit_kg.py ^
    --config configs\rolit_kg.yaml ^
    --limit_ro_stories 2 ^
    --limit_histnero 2 ^
    --no_neo4j
```

**Expected output** (in ~15 seconds):
```
RoLit-KG MVP complete. Output: data\processed\rolit_kg\rolitkg_mvp
- Cypher constraints: data\processed\rolit_kg\rolitkg_mvp\cypher\constraints.cypher
- Cypher load:        data\processed\rolit_kg\rolitkg_mvp\cypher\load.cypher
- Report:             data\processed\rolit_kg\rolitkg_mvp\reports\report.md
```

### Step 4: Inspect Results

```bash
type data\processed\rolit_kg\rolitkg_mvp\reports\report.md
```

You should see:
- **Entities**: Characters, Persons, Locations, Events
- **Relations**: INTERACTS_WITH, LOCATED_IN
- **Grounding rate**: % of fictional entities linked to historical figures

### Step 5: View Generated Cypher

```bash
type data\processed\rolit_kg\rolitkg_mvp\cypher\load.cypher
```

This is ready to paste into Neo4j Browser or load via `cypher-shell`.

---

## Load into Neo4j (Optional)

### Option A: Using Neo4j Desktop

1. **Install Neo4j Desktop**: https://neo4j.com/download/
2. **Create a new database** (set password: `password`)
3. **Start the database**
4. **Open Neo4j Browser** (http://localhost:7474)
5. **Run constraints**:
   ```bash
   # Copy/paste from constraints.cypher
   CREATE CONSTRAINT work_id_unique IF NOT EXISTS FOR (w:Work) REQUIRE w.work_id IS UNIQUE;
   CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
   CREATE CONSTRAINT mention_id_unique IF NOT EXISTS FOR (m:Mention) REQUIRE m.mention_id IS UNIQUE;
   ```
6. **Load data**:
   ```bash
   # Copy/paste from load.cypher (may take 1-2 minutes for 50+ docs)
   ```

### Option B: Using Docker

```bash
# Start Neo4j
docker run -d --name neo4j-rolit ^
  -p 7474:7474 -p 7687:7687 ^
  -e NEO4J_AUTH=neo4j/password ^
  neo4j:latest

# Wait 30 seconds for startup
timeout /t 30

# Run pipeline with direct loading
python scripts\run_rolit_kg.py ^
    --config configs\rolit_kg.yaml ^
    --limit_ro_stories 10
```

---

## Query Examples (Neo4j Browser)

### 1. List all works
```cypher
MATCH (w:Work)
RETURN w.title, w.author, w.source
ORDER BY w.title;
```

### 2. Characters in "Fata din pădure"
```cypher
MATCH (w:Work {title: "Fata din pădure"})-[:HAS_CHARACTER]->(c:Character)
RETURN c.canonical_name, c.aliases;
```

### 3. Top character interactions
```cypher
MATCH (c1:Character)-[r:INTERACTS_WITH]->(c2:Character)
RETURN c1.canonical_name AS from,
       c2.canonical_name AS to,
       r.confidence,
       r.evidence_chunk_id
ORDER BY r.confidence DESC
LIMIT 10;
```

### 4. Grounded historical figures
```cypher
MATCH (c:Character)-[:BASED_ON]->(p:Person)
RETURN c.canonical_name AS fictional,
       p.canonical_name AS historical,
       p.source;
```

### 5. Entity provenance (all mentions of "Ion")
```cypher
MATCH (m:Mention)-[:MENTIONS]->(e:Entity {canonical_name: "Ion"})
RETURN m.doc_id, m.surface, m.start_char, m.end_char, m.confidence
ORDER BY m.doc_id, m.start_char;
```

More queries in: `docs/rolit_kg_starter_queries.cypher`

---

## Process Full Corpus

Once you've tested the sample, scale up:

```bash
# Option A (recommended): Use the full production runner (fast + resumable)
# This processes the full RO-Stories corpus (12,516) + HistNERo (if present),
# checkpoints raw artifacts after extraction, and uses scalable lexical resolution.
python run_full_pipeline.py ^
  --ro_stories_jsonl data\ro_stories_full.jsonl ^
  --histnero_jsonl data\histnero_full.jsonl ^
  --output_dir outputs\rolit_kg_full_corpus ^
  --run_name rolit_kg_full_12516_lexical ^
  --resolution_mode lexical ^
  --checkpoint_after_extract

# Option B: Config-driven runner (MVP)
# Process 100 stories + all HistNERo data
python scripts\run_rolit_kg.py ^
  --config configs\rolit_kg.yaml ^
  --limit_ro_stories 100 ^
  --output_dir outputs\rolit_kg_full
```

**Expected runtime**:
- **100 stories**: ~5-10 minutes (regex NER, no GPU)
- **Full corpus (12,516)**: depends mostly on resolution strategy
  - `--resolution_mode lexical`: tens of minutes (CPU)
  - `--resolution_mode ollama`: not recommended for full corpus without ANN/kNN (can OOM / be too slow)

**Outputs (production runner)**:
- `outputs\rolit_kg_full_corpus\artifacts\*` (JSONL artifacts)
- `outputs\rolit_kg_full_corpus\cypher\constraints.cypher`
- `outputs\rolit_kg_full_corpus\cypher\load.cypher`
- `outputs\rolit_kg_full_corpus\reports\report.md`
- `outputs\rolit_kg_full_corpus\summary.json`

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```bash
set PYTHONPATH=%CD%
```

### Neo4j connection refused
Check Neo4j is running:
```bash
docker ps | grep neo4j
# Or use --no_neo4j flag
```

### Too many stopwords extracted as entities
Edit `src/pipeline/extract.py` line 48-56, add more words to `_STOPWORDS_RO`.

---

## Next Steps

- **Upgrade NER**: Install `transformers` and set `extraction.method: "hf_ner"` in config
- **Tune thresholds**: Adjust `resolution.similarity_threshold` (default: 0.90)
- **Custom data**: Add your own JSONL corpora (see `docs/ROLIT_KG_README.md` § Custom Data)
- **Explore graph**: Use Neo4j Bloom for visual exploration

---

## Full Documentation

- **Complete guide**: `docs/ROLIT_KG_README.md`
- **Implementation details**: `docs/ROLIT_KG_IMPLEMENTATION_SUMMARY.md`
- **Starter queries**: `docs/rolit_kg_starter_queries.cypher`
