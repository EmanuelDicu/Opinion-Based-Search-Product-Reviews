"""Elasticsearch indexing utilities with dense vector support.

This module handles indexing of review extractions with semantic embeddings,
enabling both keyword-based and vector similarity search.

Architecture inspired by Recipe-Search project:
- Dense vector field for KNN/ANN similarity search
- Combined text + vector indexing for hybrid search
- Bulk indexing with progress tracking

Dependencies: elasticsearch>=8.0, tqdm (see requirements.txt).
"""

from typing import List, Dict, Generator, Optional, Tuple

from elasticsearch import Elasticsearch  # type: ignore[import-untyped]
from elasticsearch.helpers import bulk, BulkIndexError  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from preprocessing.config import PreprocessingConfig
from preprocessing.embedding_service import EmbeddingService, get_embedding_service


class ElasticsearchIndexer:
    """Index extracted data in Elasticsearch with semantic embeddings.
    
    Features:
    - Dense vector field for semantic similarity search (like Recipe-Search)
    - Text fields for keyword/BM25 search
    - Hybrid search capability combining both approaches
    - Batch embedding generation for efficiency
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """Initialize Elasticsearch client and embedding service.
        
        Args:
            embedding_service: Optional embedding service instance.
                              Creates default if not provided.
        """
        es_config = PreprocessingConfig.get_elasticsearch_config()
        host = es_config["host"]
        port = es_config["port"]
        scheme = es_config["scheme"]
        url = f"{scheme}://{host}:{port}"
        self.es = Elasticsearch(
            hosts=[url],
            verify_certs=False,
            ssl_show_warn=False,
        )
        self.index_name = PreprocessingConfig.ELASTICSEARCH_INDEX
        self.embedding_service = embedding_service or get_embedding_service()
        self.embedding_dimension = PreprocessingConfig.EMBEDDING_DIMENSION
        
        # Update dimension from embedding service if available
        if self.embedding_service.is_available:
            self.embedding_dimension = self.embedding_service.dimension
    
    def is_available(self) -> bool:
        """Check if Elasticsearch is available."""
        try:
            return self.es.ping()
        except Exception as e:
            print(f"Elasticsearch connection failed: {e}")
            return False
    
    def create_index(self, delete_existing: bool = False) -> bool:
        """Create Elasticsearch index with dense vector mapping.
        
        Creates an index optimized for hybrid search:
        - Text fields for keyword/BM25 search
        - Dense vector field for KNN similarity search
        
        Args:
            delete_existing: If True, delete existing index first
            
        Returns:
            True if index created successfully
        """
        if not self.is_available():
            print("Elasticsearch is not available. Skipping index creation.")
            return False
        
        # Delete existing index if requested
        if delete_existing and self.es.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        if self.es.indices.exists(index=self.index_name):
            print(f"Index '{self.index_name}' already exists. Skipping creation.")
            return True
        
        # Mapping with dense vector support (similar to Recipe-Search)
        mapping = {
            "properties": {
                # Core fields
                "product_id": {"type": "keyword"},
                "product_name": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "aspect": {
                    "type": "keyword",  # Use keyword for exact matching
                    "fields": {
                        "text": {"type": "text", "analyzer": "standard"}
                    }
                },
                "opinion": {"type": "text", "analyzer": "standard"},
                "review_text": {"type": "text", "analyzer": "standard"},
                "sentiment": {"type": "keyword"},
                "rating": {"type": "float"},
                
                # Dense vector for semantic search (like Recipe-Search's recipe_vector)
                "review_vector": {
                    "type": "dense_vector",
                    "dims": self.embedding_dimension,
                    "index": True,  # Enable ANN search
                    "similarity": "cosine"  # Use cosine similarity
                },
                
                # Metadata for inferred high-level features
                "is_inferred": {"type": "boolean"},
                "is_implicit": {"type": "boolean"},
                "feature_type": {"type": "keyword"},
                "source_aspects": {"type": "keyword"},
                
                # Additional metadata
                "timestamp": {"type": "date"},
            }
        }
        
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,  # For development; increase for production
        }
        
        try:
            print(f"Creating index '{self.index_name}' with dense vector mapping...")
            self.es.indices.create(
                index=self.index_name,
                mappings=mapping,
                settings=settings
            )
            print(f"✓ Index '{self.index_name}' created successfully")
            print(f"  - Vector dimension: {self.embedding_dimension}")
            print(f"  - Similarity: cosine")
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def index_documents(
        self, 
        extractions: List[Dict],
        batch_size: int = 100,
        show_progress: bool = True,
        embedding_limit: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Index documents with embeddings in Elasticsearch.
        
        Generates embeddings for each document and indexes them
        using bulk operations for efficiency.
        
        Args:
            extractions: List of aspect-opinion records to index
            batch_size: Number of documents to process per batch
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (success_count, failed_count)
        """
        if not self.is_available():
            print("Elasticsearch is not available. Skipping indexing.")
            return (0, len(extractions))
        
        if not self.es.indices.exists(index=self.index_name):
            if not self.create_index():
                return (0, len(extractions))
        
        if not extractions:
            print("No documents to index")
            return (0, 0)
        
        embedding_limit = embedding_limit if embedding_limit is not None else PreprocessingConfig.DATASET_LIMIT
        print(f"Starting bulk indexing of {len(extractions)} documents...")
        print(f"  - Generating embeddings with {self.embedding_service.model} (limit: {embedding_limit})")
        print(f"  - Batch size: {batch_size}")
        
        total_success = 0
        total_failed = 0
        
        # Process in batches with progress bar
        iterator = range(0, len(extractions), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Indexing", unit="batch")
        
        embedding_count = 0
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(extractions))
            batch = extractions[batch_start:batch_end]
            
            # Generate actions with embeddings
            actions = []
            for i, extraction in enumerate(batch):
                doc_id = f"{extraction.get('product_id', 'unknown')}_{extraction.get('aspect', 'general')}_{batch_start + i}"
                
                # Generate embedding only for the first N docs
                embedding = None
                if embedding_limit and embedding_count < embedding_limit:
                    embedding = self.embedding_service.create_review_embedding(extraction)
                    embedding_count += 1
                
                # Build document
                doc = {
                    **extraction,
                    "review_vector": embedding if embedding else None
                }
                
                # Remove None vector if embedding failed
                if not embedding:
                    doc.pop("review_vector", None)
                
                action = {
                    "_index": self.index_name,
                    "_id": doc_id,
                    "_source": doc
                }
                actions.append(action)
            
            # Bulk index the batch
            try:
                success, failed = bulk(
                    self.es,
                    actions,
                    chunk_size=batch_size,
                    request_timeout=120,
                    raise_on_error=False
                )
                total_success += success
                if isinstance(failed, list):
                    total_failed += len(failed)
            except BulkIndexError as e:
                print(f"Bulk indexing error: {len(e.errors)} errors")
                total_failed += len(e.errors)
            except Exception as e:
                print(f"Error during bulk indexing: {e}")
                total_failed += len(actions)
        
        # Refresh index to make documents immediately searchable
        try:
            self.es.indices.refresh(index=self.index_name)
            print(f"✓ Index '{self.index_name}' refreshed")
        except Exception as e:
            print(f"Warning: Could not refresh index: {e}")
        
        print(f"\n✓ Indexing complete:")
        print(f"  - Successfully indexed: {total_success}")
        print(f"  - Failed: {total_failed}")
        if embedding_limit:
            print(f"  - Embeddings generated: {min(embedding_count, embedding_limit)} / {len(extractions)}")
        
        return (total_success, total_failed)
    
    def index_documents_generator(
        self, 
        extractions_generator: Generator[Dict, None, None],
        batch_size: int = 100
    ) -> Tuple[int, int]:
        """Index documents from a generator (memory-efficient for large datasets).
        
        Similar to Recipe-Search's yield_recipes_from_csv approach.
        
        Args:
            extractions_generator: Generator yielding extraction records
            batch_size: Number of documents to accumulate before bulk indexing
            
        Returns:
            Tuple of (success_count, failed_count)
        """
        if not self.is_available():
            print("Elasticsearch is not available. Skipping indexing.")
            return (0, 0)
        
        if not self.es.indices.exists(index=self.index_name):
            if not self.create_index():
                return (0, 0)
        
        total_success = 0
        total_failed = 0
        batch = []
        count = 0
        
        print("Starting streaming bulk indexing...")
        
        for extraction in extractions_generator:
            batch.append(extraction)
            count += 1
            
            if len(batch) >= batch_size:
                # Process batch
                success, failed = self._index_batch(batch, count - len(batch))
                total_success += success
                total_failed += failed
                batch = []
                
                if count % 1000 == 0:
                    print(f"  ... processed {count} documents ...")
        
        # Process remaining documents
        if batch:
            success, failed = self._index_batch(batch, count - len(batch))
            total_success += success
            total_failed += failed
        
        # Refresh index
        try:
            self.es.indices.refresh(index=self.index_name)
        except Exception:
            pass
        
        print(f"\n✓ Streaming indexing complete:")
        print(f"  - Total processed: {count}")
        print(f"  - Successfully indexed: {total_success}")
        print(f"  - Failed: {total_failed}")
        
        return (total_success, total_failed)
    
    def _index_batch(self, batch: List[Dict], start_idx: int) -> Tuple[int, int]:
        """Index a batch of documents with embeddings."""
        actions = []
        
        for i, extraction in enumerate(batch):
            doc_id = f"{extraction.get('product_id', 'unknown')}_{extraction.get('aspect', 'general')}_{start_idx + i}"
            
            # Generate embedding
            embedding = self.embedding_service.create_review_embedding(extraction)
            
            doc = {**extraction}
            if embedding:
                doc["review_vector"] = embedding
            
            actions.append({
                "_index": self.index_name,
                "_id": doc_id,
                "_source": doc
            })
        
        try:
            success, failed = bulk(
                self.es,
                actions,
                request_timeout=120,
                raise_on_error=False
            )
            failed_count = len(failed) if isinstance(failed, list) else 0
            return (success, failed_count)
        except Exception as e:
            print(f"Batch indexing error: {e}")
            return (0, len(batch))
    
    def delete_index(self) -> bool:
        """Delete the index."""
        if not self.is_available():
            return False
        
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                print(f"✓ Index '{self.index_name}' deleted")
            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False
    
    def get_index_stats(self) -> Optional[Dict]:
        """Get statistics about the index."""
        if not self.is_available():
            return None
        
        try:
            if not self.es.indices.exists(index=self.index_name):
                return None
            
            stats = self.es.indices.stats(index=self.index_name)
            count = self.es.count(index=self.index_name)
            
            return {
                "index_name": self.index_name,
                "document_count": count["count"],
                "size_bytes": stats["_all"]["total"]["store"]["size_in_bytes"],
            }
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return None

