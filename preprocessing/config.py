"""Configuration for preprocessing pipeline.

This module configures the neuro-symbolic preprocessing pipeline that:
1. Uses LLM (ollama) for intelligent aspect extraction from reviews
2. Creates dense vector embeddings for semantic search
3. Implements weighted feature inference from aspect-level signals
"""

import os
from typing import Dict, List, Tuple

class PreprocessingConfig:
    """Preprocessing configuration for the Opinion-Based Search system."""
    
    # =========================================================================
    # Dataset settings
    # =========================================================================
    DATASET_FILE = os.getenv('DATASET_FILE', 'Cell_Phones_and_Accessories.jsonl')
    DATASET_LIMIT = int(os.getenv('DATASET_LIMIT', '10000'))
    
    # =========================================================================
    # Data Quality Filters (ETL validation)
    # =========================================================================
    MIN_REVIEW_LENGTH = 20          # Minimum characters for valid review
    MAX_REVIEW_LENGTH = 5000        # Maximum characters (truncate beyond)
    MIN_WORD_COUNT = 5              # Minimum words for semantic value
    MAX_CAPS_RATIO = 0.5            # Max ratio of uppercase chars (spam filter)
    MAX_PUNCTUATION_RATIO = 0.3     # Max ratio of punctuation (spam filter)
    MAX_DIGIT_RATIO = 0.4           # Max ratio of digits (noise filter)
    MIN_RATING = 1.0                # Valid rating range
    MAX_RATING = 5.0
    REQUIRE_VERIFIED_PURCHASE = False  # If True, only use verified purchases
    
    # Spam/noise patterns to filter out
    SPAM_PATTERNS: Tuple[str, ...] = (
        "click here", "buy now", "free gift", "limited time",
        "act now", "order today", "special offer", "discount code",
    )
    
    # =========================================================================
    # Elasticsearch settings
    # =========================================================================
    ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    ELASTICSEARCH_PORT = int(os.getenv('ELASTICSEARCH_PORT', '9200'))
    ELASTICSEARCH_SCHEME = os.getenv('ELASTICSEARCH_SCHEME', 'http')
    ELASTICSEARCH_INDEX = os.getenv('ELASTICSEARCH_INDEX', 'product_reviews_vector')
    
    # =========================================================================
    # Data settings
    # =========================================================================
    DATA_DIR = os.getenv('DATA_DIR', '/app/data')
    EXTRACTIONS_FILE = os.path.join(DATA_DIR, 'extractions.json')
    
    # =========================================================================
    # Ollama LLM settings
    # =========================================================================
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    CHAT_MODEL = os.getenv('CHAT_MODEL', 'qwen3:8b')
    # Extraction uses a smaller, faster model by default (qwen2.5:3b). Set EXTRACTION_CHAT_MODEL=qwen3:8b for quality over speed.
    EXTRACTION_CHAT_MODEL = os.getenv('EXTRACTION_CHAT_MODEL', 'qwen2.5:3b')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'qwen3-embedding:4b')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '2560'))
    # Timeout in seconds for each Ollama extraction request (avoids blocking forever)
    EXTRACTION_REQUEST_TIMEOUT = int(os.getenv('EXTRACTION_REQUEST_TIMEOUT', '300'))
    
    # =========================================================================
    # Processing settings
    # =========================================================================
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    # Reviews per Ollama call (smaller = faster per call, less likely to hang)
    EXTRACTION_BATCH_SIZE = int(os.getenv('EXTRACTION_BATCH_SIZE', '50'))
    # If true, only send reviews that contain aspect keywords to LLM; others use keyword extraction only
    EXTRACTION_LLM_PRESELECT = os.getenv('EXTRACTION_LLM_PRESELECT', 'true').lower() == 'true'
    # Max number of reviews to use for aspect extraction (0 = no limit; analyze full dataset)
    EXTRACTION_REVIEW_LIMIT = int(os.getenv('EXTRACTION_REVIEW_LIMIT', '0'))
    USE_LLM_EXTRACTION = os.getenv('USE_LLM_EXTRACTION', 'true').lower() == 'true'
    
    # =========================================================================
    # Inferred feature controls
    # =========================================================================
    MAX_INFERRED_FEATURES = int(os.getenv('MAX_INFERRED_FEATURES', '1000'))
    MIN_INFERRED_CONFIDENCE = float(os.getenv('MIN_INFERRED_CONFIDENCE', '0.6'))
    MIN_INFERRED_ASPECTS = int(os.getenv('MIN_INFERRED_ASPECTS', '2'))
    
    # =========================================================================
    # Aspect categories (Level 1 concepts)
    # =========================================================================
    ASPECT_CATEGORIES: List[str] = [
        "camera", "battery", "screen", "processor", "design",
        "sound", "software", "price", "durability", "connectivity", "storage",
    ]
    
    # =========================================================================
    # WEIGHTED High-Level Features (Level 2 concepts)
    # 
    # Each aspect has TWO weights:
    #   - pos_weight: How much a POSITIVE sentiment in this aspect contributes
    #   - neg_weight: How much a NEGATIVE sentiment in this aspect contributes
    # 
    # Higher weight = more important for that sentiment direction.
    # Example: For gaming, processor lag (negative) is critical, so neg_weight=2.0
    # =========================================================================
    HIGH_LEVEL_FEATURES: Dict[str, Dict] = {
        "gaming_experience": {
            "description": "Suitability for mobile gaming",
            "aspects": {
                # aspect: (pos_weight, neg_weight)
                "processor": (1.5, 2.0),  # Performance critical; lag is deal-breaker
                "screen":    (1.2, 1.0),  # Good display helps; bad display less critical
                "battery":   (0.8, 1.5),  # Nice to have; drain is annoying
            },
        },
        "photography_capability": {
            "description": "Camera quality for photos and videos",
            "aspects": {
                "camera": (2.0, 2.0),  # Camera is everything here
            },
        },
        "cinematic_viewing": {
            "description": "Quality of media consumption",
            "aspects": {
                "screen": (1.5, 1.5),  # Display quality is key
                "sound":  (1.2, 1.0),  # Audio enhances; bad audio tolerable
            },
        },
        "all_day_endurance": {
            "description": "Battery life for power users",
            "aspects": {
                "battery":   (2.0, 2.0),  # Battery is the feature
                "processor": (0.5, 1.0),  # Efficiency helps; power-hungry hurts
            },
        },
        "value_for_money": {
            "description": "Overall value proposition",
            "aspects": {
                "price":     (1.5, 2.0),  # Value critical; overpriced is rejection
                "processor": (0.8, 0.5),  # Good perf adds value
                "camera":    (0.8, 0.5),  # Good camera adds value
                "battery":   (0.8, 0.5),  # Good battery adds value
            },
        },
        "field_utility": {
            "description": "Outdoor/field use suitability",
            "aspects": {
                "battery":    (1.5, 1.5),
                "durability": (1.5, 2.0),  # Fragile is a deal-breaker outdoors
                "design":     (0.8, 0.5),
            },
        },
        "business_productivity": {
            "description": "Professional/productivity use",
            "aspects": {
                "processor": (1.2, 1.5),  # Crashes/lag hurt productivity
                "software":  (1.5, 2.0),  # Software reliability critical
                "battery":   (1.0, 1.0),
            },
        },
        "thermal_stability": {
            "description": "Stays cool under load",
            "aspects": {
                "processor": (1.0, 2.0),  # Overheating is major negative
                "design":    (0.8, 1.0),
            },
        },
        "rugged_lifestyle": {
            "description": "Durability for accidents/harsh use",
            "aspects": {
                "durability": (2.0, 2.0),
                "design":     (1.0, 1.0),
            },
        },
        "audio_fidelity": {
            "description": "Music/audio playback quality",
            "aspects": {
                "sound":        (2.0, 2.0),
                "connectivity": (0.5, 0.5),  # Bluetooth codec support
            },
        },
        "night_photography": {
            "description": "Low-light photo quality",
            "aspects": {
                "camera":    (2.0, 2.0),
                "processor": (0.5, 0.5),  # Processing helps night mode
            },
        },
        "one_hand_comfort": {
            "description": "Ease of single-hand use",
            "aspects": {
                "design": (2.0, 2.0),  # Size/weight is the feature
                "screen": (0.8, 1.0),  # Large screen hurts
            },
        },
        "minimalist_experience": {
            "description": "Clean, bloat-free software",
            "aspects": {
                "software": (2.0, 2.0),
                "design":   (0.5, 0.3),
            },
        },
        "fast_charging": {
            "description": "Speed of charging",
            "aspects": {
                "battery": (2.0, 2.0),  # Charging speed
            },
        },
        "future_longevity": {
            "description": "Stays fast for 3+ years",
            "aspects": {
                "processor": (1.2, 1.0),
                "software":  (1.5, 1.5),  # Update support critical
                "storage":   (0.8, 1.0),
            },
        },
    }
    
    @classmethod
    def get_elasticsearch_config(cls) -> Dict[str, any]:
        """Get Elasticsearch connection configuration."""
        return {
            'host': cls.ELASTICSEARCH_HOST,
            'port': cls.ELASTICSEARCH_PORT,
            'scheme': cls.ELASTICSEARCH_SCHEME
        }
    
    @classmethod
    def get_ollama_url(cls) -> str:
        """Get the Ollama API base URL."""
        return cls.OLLAMA_HOST

