"""Preprocessing package for Opinion-Based Search.

Pipeline stages:
1. Data loading with strict validation (spam, noise, duplicate filtering)
2. Aspect extraction via LLM (qwen3:8b) or keyword fallback
3. Weighted feature inference (asymmetric pos/neg weights per aspect)
4. Dense vector embeddings (qwen3-embedding:4b)
5. Elasticsearch indexing with KNN support
"""

from preprocessing.config import PreprocessingConfig
from preprocessing.text_processor import TextProcessor
from preprocessing.dataset_loader import DatasetLoader, ReviewValidator
from preprocessing.aspect_extractor import AspectOpinionExtractor
from preprocessing.embedding_service import EmbeddingService, get_embedding_service
from preprocessing.elasticsearch_indexer import ElasticsearchIndexer
from preprocessing.feature_inference import FeatureInferenceEngine, infer_high_level_features

__all__ = [
    "PreprocessingConfig",
    "TextProcessor", 
    "DatasetLoader",
    "ReviewValidator",
    "AspectOpinionExtractor",
    "EmbeddingService",
    "get_embedding_service",
    "ElasticsearchIndexer",
    "FeatureInferenceEngine",
    "infer_high_level_features",
]
