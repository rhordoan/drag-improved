"""
Sanity-check Ollama JSON-mode relation extraction.

Runs a controlled example where at least one relation should be extractable.
"""

from src.pipeline.common import Chunk, Document, Mention, stable_id
from src.pipeline.extract import extract_relations_ollama


def main() -> None:
    text = "Ion Creangă a mers la Brașov cu Maria Ionescu."

    doc = Document(doc_id="doc_test", source="ro_stories", title=None, text=text)
    chunk = Chunk(
        chunk_id="chunk_test",
        doc_id=doc.doc_id,
        source=doc.source,
        start_char=0,
        end_char=len(text),
        text=text,
    )

    # Predefined entities/mentions (IDs must match what LLM is allowed to use)
    ents = [
        ("e1", "Ion Creangă", "Character", 0, 10),
        ("e2", "Brașov", "Location", 22, 28),
        ("e3", "Maria Ionescu", "Character", 33, 46),
    ]
    mentions = []
    for eid, name, typ, s, e in ents:
        mentions.append(
            Mention(
                mention_id=stable_id("m", doc.doc_id, chunk.chunk_id, eid),
                entity_id=eid,
                surface=name,
                start_char=s,
                end_char=e,
                doc_id=doc.doc_id,
                chunk_id=chunk.chunk_id,
                source=doc.source,
                entity_type=typ,
                confidence=0.9,
            )
        )

    rels = extract_relations_ollama(
        doc=doc,
        chunk=chunk,
        mentions=mentions,
        base_url="http://inference.ccrolabs.com",
        model="llama3.2:3b",
        timeout_s=60,
        max_relations=10,
    )

    print(f"relations={len(rels)}")
    for r in rels:
        # Keep output ASCII-safe for Windows consoles
        print(f"{r.source_entity_id} -[{r.predicate}]-> {r.target_entity_id} (conf={r.confidence:.2f})")
        if r.evidence_text:
            print(f"  evidence={r.evidence_text[:120]}")


if __name__ == "__main__":
    main()

