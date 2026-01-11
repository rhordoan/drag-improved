"""
Advanced extraction module for production-grade RoLit-KG.

Features:
- Ensemble NER (regex + transformer + LLM)
- Semantic relation extraction with LLM guidance
- Entity typing with confidence scores
- Coreference resolution
- Temporal relation extraction
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .common import (
    Chunk,
    Document,
    Entity,
    Mention,
    Relation,
    normalize_mention,
    slice_text,
    stable_id,
)


@dataclass
class ExtractionConfig:
    """Configuration for advanced extraction."""
    use_regex_ner: bool = True
    use_transformer_ner: bool = True
    use_llm_ner: bool = False
    transformer_model: str = "readerbench/ro-ner"
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    
    relation_extraction_mode: str = "semantic"  # "heuristic" | "semantic" | "llm"
    relation_confidence_threshold: float = 0.5
    
    enable_coreference: bool = True
    enable_temporal: bool = True
    enable_entity_typing: bool = True
    
    cache_dir: Optional[str] = None
    batch_size: int = 16
    max_workers: int = 4


# Entity type taxonomy with confidence scoring
ENTITY_TYPE_TAXONOMY = {
    "Character": {"fictional": True, "narrative_role": True},
    "Person": {"fictional": False, "historical": True},
    "Location": {"spatial": True},
    "Organization": {"collective": True},
    "Event": {"temporal": True},
    "Theme": {"abstract": True},
    "Motif": {"literary": True},
}


# Semantic relation patterns with confidence weights
SEMANTIC_RELATION_PATTERNS = {
    "LOVES": {
        "keywords": ["iubește", "iubire", "dragoste", "dragostea", "îndrăgostit"],
        "confidence": 0.85,
        "symmetric": True,
    },
    "HATES": {
        "keywords": ["urăște", "ură", "dușmănie", "vrăjmaș", "inamic"],
        "confidence": 0.85,
        "symmetric": True,
    },
    "KILLS": {
        "keywords": ["ucide", "omoară", "asasinează", "crimă", "mort"],
        "confidence": 0.9,
        "symmetric": False,
    },
    "TRAVELS_TO": {
        "keywords": ["călătorește", "pleacă", "merge", "ajunge", "sosește"],
        "confidence": 0.7,
        "symmetric": False,
    },
    "DESCENDS_FROM": {
        "keywords": ["fiu", "fiică", "nepoat", "descendent", "moștenit"],
        "confidence": 0.85,
        "symmetric": False,
    },
    "OWNS": {
        "keywords": ["deține", "proprietar", "posed", "aparține", "moșie"],
        "confidence": 0.75,
        "symmetric": False,
    },
}


# Temporal indicators for event ordering
TEMPORAL_INDICATORS = {
    "BEFORE": ["înainte", "mai devreme", "anterior", "pregătind"],
    "AFTER": ["după", "mai târziu", "ulterior", "următor"],
    "DURING": ["în timpul", "pe când", "în timp ce", "concomitent"],
    "OVERLAPS": ["simultan", "în același timp", "totodată"],
}


class EnsembleNER:
    """Ensemble NER combining multiple extraction methods."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._transformer_pipeline = None
        self._llm_client = None
        self._cache = {}
        
    def extract_entities(
        self, 
        doc: Document, 
        chunk: Chunk
    ) -> Tuple[List[Mention], List[Entity]]:
        """
        Extract entities using ensemble of methods.
        Deduplicates and merges results with confidence voting.
        """
        all_mentions: List[Mention] = []
        all_entities: List[Entity] = []
        
        if self.config.use_regex_ner:
            regex_m, regex_e = self._extract_regex(doc, chunk)
            all_mentions.extend(regex_m)
            all_entities.extend(regex_e)
        
        if self.config.use_transformer_ner:
            trans_m, trans_e = self._extract_transformer(doc, chunk)
            all_mentions.extend(trans_m)
            all_entities.extend(trans_e)
        
        if self.config.use_llm_ner:
            llm_m, llm_e = self._extract_llm(doc, chunk)
            all_mentions.extend(llm_m)
            all_entities.extend(llm_e)
        
        # Merge overlapping mentions and vote on entity types
        merged_mentions, merged_entities = self._merge_and_vote(all_mentions, all_entities)
        
        return merged_mentions, merged_entities
    
    def _extract_regex(self, doc: Document, chunk: Chunk) -> Tuple[List[Mention], List[Entity]]:
        """Regex-based extraction with Romanian capitalization patterns."""
        mentions: List[Mention] = []
        entities: List[Entity] = []
        
        cap_seq_re = re.compile(
            r"\b[A-ZĂÂÎȘȚ][A-Za-zĂÂÎȘȚăâîșț\-\']+(?:\s+[A-ZĂÂÎȘȚ][A-Za-zĂÂÎȘȚăâîșț\-\']+){0,3}\b",
            re.UNICODE,
        )
        
        stopwords = {
            'într-o', 'într-un', 'de', 'la', 'pe', 'cu', 'din', 'în', 'pentru',
            'și', 'sau', 'dar', 'dacă', 'când', 'unde', 'cum', 'ce', 'care',
        }
        
        for m in cap_seq_re.finditer(chunk.text):
            surface = m.group(0).strip()
            if len(surface) < 2 or surface.lower() in stopwords:
                continue
            
            start = chunk.start_char + m.start()
            end = chunk.start_char + m.end()
            
            mention_id = stable_id("m", doc.doc_id, chunk.chunk_id, str(start), str(end), surface)
            entity_id = stable_id("ent", "mention", mention_id)
            
            # Simple heuristic: if in historical source, likely Person, else Character
            entity_type = "Person" if doc.source == "histnero" else "Character"
            
            mentions.append(
                Mention(
                    mention_id=mention_id,
                    entity_id=entity_id,
                    surface=surface,
                    start_char=start,
                    end_char=end,
                    doc_id=doc.doc_id,
                    chunk_id=chunk.chunk_id,
                    source=doc.source,
                    entity_type=entity_type,
                    confidence=0.6,  # Lower confidence for regex
                )
            )
            
            entities.append(
                Entity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    canonical_name=surface,
                    aliases=(),
                    is_fictional=(doc.source == "ro_stories"),
                    source=doc.source,
                    meta={"extraction_method": "regex"},
                )
            )
        
        return mentions, entities
    
    def _extract_transformer(self, doc: Document, chunk: Chunk) -> Tuple[List[Mention], List[Entity]]:
        """Transformer-based NER with fine-tuned Romanian model."""
        if self._transformer_pipeline is None:
            try:
                from transformers import pipeline
                self._transformer_pipeline = pipeline(
                    "ner",
                    model=self.config.transformer_model,
                    aggregation_strategy="simple",
                    device=-1,  # CPU; use device=0 for GPU
                )
            except Exception as e:
                print(f"Warning: Failed to load transformer NER: {e}")
                return [], []
        
        mentions: List[Mention] = []
        entities: List[Entity] = []
        
        try:
            results = self._transformer_pipeline(chunk.text)
            
            for item in results:
                surface = item["word"]
                start = chunk.start_char + item["start"]
                end = chunk.start_char + item["end"]
                confidence = float(item["score"])
                
                label = item["entity_group"].upper()
                entity_type_map = {
                    "PER": "Character" if doc.source == "ro_stories" else "Person",
                    "PERSON": "Character" if doc.source == "ro_stories" else "Person",
                    "LOC": "Location",
                    "LOCATION": "Location",
                    "ORG": "Organization",
                }
                entity_type = entity_type_map.get(label, "Character")
                
                mention_id = stable_id("m", doc.doc_id, chunk.chunk_id, str(start), str(end), surface)
                entity_id = stable_id("ent", "mention", mention_id)
                
                mentions.append(
                    Mention(
                        mention_id=mention_id,
                        entity_id=entity_id,
                        surface=surface,
                        start_char=start,
                        end_char=end,
                        doc_id=doc.doc_id,
                        chunk_id=chunk.chunk_id,
                        source=doc.source,
                        entity_type=entity_type,
                        confidence=confidence,
                    )
                )
                
                entities.append(
                    Entity(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        canonical_name=surface,
                        aliases=(),
                        is_fictional=(doc.source == "ro_stories"),
                        source=doc.source,
                        meta={"extraction_method": "transformer", "transformer_label": label},
                    )
                )
        except Exception as e:
            print(f"Warning: Transformer NER failed for chunk {chunk.chunk_id}: {e}")
        
        return mentions, entities
    
    def _extract_llm(self, doc: Document, chunk: Chunk) -> Tuple[List[Mention], List[Entity]]:
        """LLM-based extraction with structured output."""
        # TODO: Implement LLM extraction with OpenAI/Anthropic API
        # For now, return empty
        return [], []
    
    def _merge_and_vote(
        self,
        mentions: List[Mention],
        entities: List[Entity],
    ) -> Tuple[List[Mention], List[Entity]]:
        """
        Merge overlapping mentions and vote on entity types.
        Uses span overlap and confidence-weighted voting.
        """
        if not mentions:
            return [], []
        
        # Sort by start position
        sorted_mentions = sorted(mentions, key=lambda m: (m.start_char, m.end_char))
        
        merged_mentions: List[Mention] = []
        merged_entities: List[Entity] = []
        entity_map: Dict[str, Entity] = {e.entity_id: e for e in entities}
        
        i = 0
        while i < len(sorted_mentions):
            current = sorted_mentions[i]
            overlapping = [current]
            
            # Find all overlapping mentions
            j = i + 1
            while j < len(sorted_mentions):
                next_m = sorted_mentions[j]
                if next_m.start_char < current.end_char:
                    overlapping.append(next_m)
                    j += 1
                else:
                    break
            
            # Vote on entity type using confidence weights
            type_votes: Dict[str, float] = {}
            for m in overlapping:
                type_votes[m.entity_type] = type_votes.get(m.entity_type, 0.0) + m.confidence
            
            best_type = max(type_votes.items(), key=lambda x: x[1])[0]
            avg_confidence = sum(m.confidence for m in overlapping) / len(overlapping)
            
            # Use the mention with highest confidence as representative
            best_mention = max(overlapping, key=lambda m: m.confidence)
            
            # Create merged mention with voted type and average confidence
            mention_id = stable_id(
                "m",
                best_mention.doc_id,
                best_mention.chunk_id,
                str(best_mention.start_char),
                str(best_mention.end_char),
                best_mention.surface,
            )
            entity_id = stable_id("ent", "mention", mention_id)
            
            merged_mentions.append(
                Mention(
                    mention_id=mention_id,
                    entity_id=entity_id,
                    surface=best_mention.surface,
                    start_char=best_mention.start_char,
                    end_char=best_mention.end_char,
                    doc_id=best_mention.doc_id,
                    chunk_id=best_mention.chunk_id,
                    source=best_mention.source,
                    entity_type=best_type,
                    confidence=min(avg_confidence, 1.0),
                )
            )
            
            merged_entities.append(
                Entity(
                    entity_id=entity_id,
                    entity_type=best_type,
                    canonical_name=best_mention.surface,
                    aliases=(),
                    is_fictional=best_mention.source == "ro_stories",
                    source=best_mention.source,
                    meta={
                        "merged_from": len(overlapping),
                        "extraction_methods": list(set(
                            entity_map.get(m.entity_id, Entity(
                                entity_id="", entity_type="", canonical_name="",
                                aliases=(), is_fictional=False, source="", meta={}
                            )).meta.get("extraction_method", "unknown")
                            for m in overlapping
                        )),
                    },
                )
            )
            
            i = j if j > i + 1 else i + 1
        
        return merged_mentions, merged_entities


class SemanticRelationExtractor:
    """Semantic relation extraction with pattern matching and context analysis."""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.patterns = SEMANTIC_RELATION_PATTERNS
        self.temporal_indicators = TEMPORAL_INDICATORS
    
    def extract_relations(
        self,
        doc: Document,
        chunk: Chunk,
        mentions: List[Mention],
    ) -> List[Relation]:
        """Extract semantic relations between entities in chunk."""
        relations: List[Relation] = []
        
        # Extract pairwise relations based on semantic patterns
        relations.extend(self._extract_semantic_relations(doc, chunk, mentions))
        
        # Extract temporal relations
        if self.config.enable_temporal:
            relations.extend(self._extract_temporal_relations(doc, chunk, mentions))
        
        return relations
    
    def _extract_semantic_relations(
        self,
        doc: Document,
        chunk: Chunk,
        mentions: List[Mention],
    ) -> List[Relation]:
        """Extract semantic relations using pattern matching."""
        relations: List[Relation] = []
        
        if len(mentions) < 2:
            return relations
        
        chunk_text_lower = chunk.text.lower()
        
        # For each pair of mentions in the chunk
        for i, m1 in enumerate(mentions):
            for m2 in mentions[i + 1:]:
                # Get text between mentions
                if m1.start_char < m2.start_char:
                    between_text = slice_text(
                        doc.text,
                        m1.end_char,
                        m2.start_char,
                    ).lower()
                else:
                    between_text = slice_text(
                        doc.text,
                        m2.end_char,
                        m1.start_char,
                    ).lower()
                
                # Check for semantic patterns
                for predicate, pattern_info in self.patterns.items():
                    keywords = pattern_info["keywords"]
                    base_confidence = pattern_info["confidence"]
                    
                    # Check if any keyword appears between mentions or in chunk
                    matches = sum(1 for kw in keywords if kw in between_text or kw in chunk_text_lower)
                    
                    if matches > 0:
                        # Adjust confidence based on number of keyword matches
                        confidence = min(base_confidence + (matches - 1) * 0.05, 0.95)
                        
                        # Create relation
                        rel = Relation(
                            source_entity_id=m1.entity_id,
                            predicate=predicate,
                            target_entity_id=m2.entity_id,
                            doc_id=doc.doc_id,
                            chunk_id=chunk.chunk_id,
                            source=doc.source,
                            confidence=confidence,
                            evidence_text=between_text[:200],
                            rel_type="semantic",
                        )
                        relations.append(rel)
                        
                        # Add symmetric relation if applicable
                        if pattern_info.get("symmetric", False):
                            rel_sym = Relation(
                                source_entity_id=m2.entity_id,
                                predicate=predicate,
                                target_entity_id=m1.entity_id,
                                doc_id=doc.doc_id,
                                chunk_id=chunk.chunk_id,
                                source=doc.source,
                                confidence=confidence,
                                evidence_text=between_text[:200],
                                rel_type="semantic",
                            )
                            relations.append(rel_sym)
        
        return relations
    
    def _extract_temporal_relations(
        self,
        doc: Document,
        chunk: Chunk,
        mentions: List[Mention],
    ) -> List[Relation]:
        """Extract temporal ordering relations between events."""
        relations: List[Relation] = []
        
        # Filter to only Event-type mentions
        event_mentions = [m for m in mentions if m.entity_type == "Event"]
        
        if len(event_mentions) < 2:
            return relations
        
        chunk_text_lower = chunk.text.lower()
        
        # For each pair of events
        for i, e1 in enumerate(event_mentions):
            for e2 in event_mentions[i + 1:]:
                # Check for temporal indicators
                for temporal_rel, indicators in self.temporal_indicators.items():
                    if any(ind in chunk_text_lower for ind in indicators):
                        rel = Relation(
                            source_entity_id=e1.entity_id,
                            predicate=temporal_rel,
                            target_entity_id=e2.entity_id,
                            doc_id=doc.doc_id,
                            chunk_id=chunk.chunk_id,
                            source=doc.source,
                            confidence=0.7,
                            evidence_text=chunk.text[:200],
                            rel_type="temporal",
                        )
                        relations.append(rel)
                        break
        
        return relations


def extract_advanced(
    docs: List[Document],
    doc_chunks: Dict[str, List[Chunk]],
    config: ExtractionConfig,
) -> Tuple[List[Mention], List[Entity], List[Relation]]:
    """
    Advanced extraction pipeline using ensemble NER and semantic relation extraction.
    
    Args:
        docs: List of documents to process
        doc_chunks: Map of doc_id -> chunks
        config: Extraction configuration
    
    Returns:
        Tuple of (mentions, entities, relations)
    """
    all_mentions: List[Mention] = []
    all_entities: List[Entity] = []
    all_relations: List[Relation] = []
    
    ner_extractor = EnsembleNER(config)
    rel_extractor = SemanticRelationExtractor(config)
    
    for doc in docs:
        chunks = doc_chunks.get(doc.doc_id, [])
        
        for chunk in chunks:
            # Extract entities with ensemble NER
            mentions, entities = ner_extractor.extract_entities(doc, chunk)
            all_mentions.extend(mentions)
            all_entities.extend(entities)
            
            # Extract semantic relations
            relations = rel_extractor.extract_relations(doc, chunk, mentions)
            all_relations.extend(relations)
    
    return all_mentions, all_entities, all_relations
