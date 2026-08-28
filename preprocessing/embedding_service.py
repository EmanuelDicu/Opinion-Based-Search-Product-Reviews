"""Embedding service for semantic vector representations.

This module provides dense vector embeddings for product reviews and aspects,
enabling semantic similarity search in Elasticsearch.

Architecture inspired by Recipe-Search project:
- Uses ollama with qwen3-embedding:4b model
- Creates embeddings for review text + aspect information
- Supports batch processing for efficiency
"""

from typing import List, Dict, Optional, Union
import sys

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from preprocessing.config import PreprocessingConfig


class EmbeddingService:
    """Service for generating dense vector embeddings using ollama.
    
    Similar to the Recipe-Search project, this uses qwen3-embedding:4b
    to create semantic embeddings that enable meaning-based search.
    """
    
    def __init__(self, model: str = None):
        """Initialize the embedding service.
        
        Args:
            model: Embedding model name. Defaults to config setting.
        """
        self.model = model or PreprocessingConfig.EMBEDDING_MODEL
        self.dimension = PreprocessingConfig.EMBEDDING_DIMENSION
        self._available = OLLAMA_AVAILABLE
        
        if not self._available:
            print("⚠ Warning: ollama not installed. Embeddings will be empty.", file=sys.stderr)
        else:
            print(f"✓ Embedding service initialized with {self.model}")
            if not self._verify_model():
                self._available = False  # Disable so we don't spam errors per document
    
    def _verify_model(self) -> bool:
        """Verify the embedding model is available (Ollama reachable and model loaded)."""
        try:
            test_response = ollama.embeddings(model=self.model, prompt="test")
            actual_dim = len(test_response.get('embedding', []))
            if actual_dim > 0:
                self.dimension = actual_dim
                print(f"✓ Embedding model verified. Dimension: {self.dimension}")
                return True
        except Exception as e:
            print(f"⚠ Ollama unreachable or model not loaded: {e}", file=sys.stderr)
            print("  Embeddings will be skipped. To fix:", file=sys.stderr)
            print("  - Run Ollama on the host and set OLLAMA_HOST (e.g. http://host.docker.internal:11434 in Docker)", file=sys.stderr)
            print(f"  - Pull the model: ollama pull {self.model}", file=sys.stderr)
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if embedding service is available."""
        return self._available
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the dense vector
        """
        if not self._available or not text:
            return []
        
        try:
            response = ollama.embeddings(model=self.model, prompt=text)
            return response.get('embedding', [])
        except Exception as e:
            self._available = False  # Stop retrying and avoid repeated error messages
            print(f"Ollama connection failed (embeddings disabled for this run): {e}", file=sys.stderr)
            print("  Ensure Ollama is running and OLLAMA_HOST is set (e.g. http://host.docker.internal:11434 in Docker).", file=sys.stderr)
            return []
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts.
        
        Note: ollama doesn't have native batch support, so this
        processes sequentially but can be optimized with async.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not self._available:
            return [[] for _ in texts]
        
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            embeddings.append(emb)
        return embeddings
    
    def create_review_embedding(self, record: Dict) -> List[float]:
        """Create a semantic embedding for a review record.
        
        Combines aspect, opinion, and review text into a single
        embedding that captures the semantic meaning.
        
        Similar to Recipe-Search combining title + ingredients.
        
        Args:
            record: Dictionary with aspect, opinion, review_text, etc.
            
        Returns:
            Dense vector embedding
        """
        # Build combined text for embedding (like Recipe-Search combines title + ingredients)
        parts = []
        
        # Include product name if available
        if record.get('product_name'):
            parts.append(f"Product: {record['product_name']}")
        
        # Include aspect
        aspect = record.get('aspect', '')
        if aspect:
            parts.append(f"Aspect: {aspect}")
        
        # Include opinion
        opinion = record.get('opinion', '')
        if opinion and opinion != 'mentioned':
            parts.append(f"Opinion: {opinion}")
        
        # Include sentiment
        sentiment = record.get('sentiment', 'neutral')
        parts.append(f"Sentiment: {sentiment}")
        
        # Include review excerpt
        review_text = record.get('review_text', '')
        if review_text:
            parts.append(f"Review: {review_text[:300]}")
        
        combined_text = " | ".join(parts)
        return self.get_embedding(combined_text)
    
    def create_query_embedding(
        self, 
        query: str, 
        aspect_hint: Optional[str] = None,
        sentiment_hint: Optional[str] = None
    ) -> List[float]:
        """Create a semantic embedding for a search query.
        
        Optionally incorporates aspect and sentiment hints to
        improve search relevance.
        
        Args:
            query: User's search query
            aspect_hint: Optional aspect category to focus on
            sentiment_hint: Optional sentiment preference (positive/negative)
            
        Returns:
            Dense vector embedding for the query
        """
        parts = [query]
        
        if aspect_hint:
            parts.append(f"Aspect: {aspect_hint}")
        
        if sentiment_hint:
            parts.append(f"Looking for: {sentiment_hint} reviews")
        
        combined_query = " | ".join(parts)
        return self.get_embedding(combined_query)


# Singleton instance for convenience
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
