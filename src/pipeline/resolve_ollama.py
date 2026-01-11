"""
Ollama-based embedding resolver for entity resolution.

Uses Ollama API for embeddings instead of sentence-transformers.
"""

import logging
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama embeddings."""
    base_url: str = "http://inference.ccrolabs.com"
    model: str = "nomic-embed-text"  # or "mxbai-embed-large", "all-minilm"
    timeout: int = 30


class OllamaEmbeddingResolver:
    """Entity resolution using Ollama embeddings."""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self._embedding_cache: Dict[str, List[float]] = {}
        self._session = requests.Session()
        
        logger.info(f"Initializing Ollama resolver: {config.base_url}, model={config.model}")
        
        # Test connection
        try:
            self._test_connection()
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.config.base_url}: {e}")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            response = self._session.post(
                f"{self.config.base_url}/api/embeddings",
                json={
                    "model": self.config.model,
                    "prompt": text,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            
            data = response.json()
            embedding = data.get("embedding")
            
            if embedding:
                # Cache the result
                self._embedding_cache[text] = embedding
                return embedding
            else:
                logger.error(f"No embedding in response: {data}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout getting embedding for: {text[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> Optional[List[List[float]]]:
        """
        Get embeddings for multiple texts.
        
        Ollama doesn't have native batch API, so we process in chunks.
        """
        import numpy as np
        
        embeddings = []
        
        # Check cache first
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if text in self._embedding_cache:
                embeddings.append(self._embedding_cache[text])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Get uncached embeddings
        if uncached_texts:
            logger.info(f"Getting embeddings for {len(uncached_texts)} texts")
            
            for i, text in enumerate(uncached_texts):
                emb = self.get_embedding(text)
                if emb:
                    idx = uncached_indices[i]
                    embeddings[idx] = emb
                
                # Rate limiting - don't hammer the server
                if i < len(uncached_texts) - 1 and i % batch_size == 0:
                    import time
                    time.sleep(0.1)
        
        # Check if we got all embeddings
        if None in embeddings:
            logger.warning(f"{embeddings.count(None)} texts failed to embed")
            # Replace None with zero vectors (or filter them out)
            dim = len(next(e for e in embeddings if e is not None))
            embeddings = [e if e else [0.0] * dim for e in embeddings]
        
        return embeddings
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        import numpy as np
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        similarity = float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        )
        return similarity
    
    def find_similar_pairs(
        self,
        entities: List,
        threshold: float,
    ) -> List[tuple]:
        """
        Find similar entity pairs using Ollama embeddings.
        
        Args:
            entities: List of Entity objects
            threshold: Similarity threshold
            
        Returns:
            List of (index1, index2, similarity) tuples
        """
        import numpy as np
        
        # Get all entity names
        entity_names = [e.canonical_name for e in entities]
        
        logger.info(f"Computing embeddings for {len(entity_names)} entities")
        
        # Get embeddings in batch
        embeddings = self.get_embeddings_batch(entity_names)
        if not embeddings:
            logger.error("Failed to get embeddings")
            return []
        
        embeddings = np.array(embeddings)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        # Compute similarity matrix (upper triangle only)
        similar_pairs = []
        n = len(entities)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= threshold:
                    similar_pairs.append((i, j, sim))
        
        logger.info(f"Found {len(similar_pairs)} similar pairs (threshold={threshold})")
        return similar_pairs
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "size": len(self._embedding_cache),
            "cache_enabled": True,
        }
