"""Configuration module for the backend application.

This module provides configuration for the Opinion-Based Search backend,
including Elasticsearch settings optimized for hybrid search with dense vectors.
"""

import os
from typing import Dict

class Config:
    """Application configuration."""
    
    # Flask settings
    FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Elasticsearch settings
    ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    ELASTICSEARCH_PORT = int(os.getenv('ELASTICSEARCH_PORT', 9200))
    ELASTICSEARCH_SCHEME = os.getenv('ELASTICSEARCH_SCHEME', 'http')
    # Use the vector-enabled index (matches preprocessing config)
    ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'product_reviews_vector')
    
    # Data settings
    DATA_DIR = os.getenv('DATA_DIR', '/app/data')
    EXTRACTIONS_FILE = os.path.join(DATA_DIR, 'extractions.json')
    
    # CORS settings
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # Ollama settings (for query embedding and summarization)
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    CHAT_MODEL = os.getenv('CHAT_MODEL', 'qwen3:8b')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'qwen3-embedding:4b')
    
    # OpenAI summarization settings
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    OPENAI_API_KEY_PATH = os.getenv('OPENAI_API_KEY_PATH', '/app/api_key.txt')
    
    # LLM summarization: number of related reviews to pass to the model (top N)
    LLM_SUMMARY_MAX_REVIEWS = int(os.getenv('LLM_SUMMARY_MAX_REVIEWS', '20'))
    
    @classmethod
    def get_elasticsearch_config(cls) -> Dict[str, any]:
        """Get Elasticsearch connection configuration."""
        return {
            'host': cls.ELASTICSEARCH_HOST,
            'port': cls.ELASTICSEARCH_PORT,
            'scheme': cls.ELASTICSEARCH_SCHEME
        }

