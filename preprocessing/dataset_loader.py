"""Dataset loading with strict validation and noise filtering.

Implements heuristic filters to ensure data quality for semantic analysis:
- Length validation
- Rating validation  
- Spam/noise detection (caps, punctuation, digits)
- Duplicate detection
- Content quality checks
"""

import json
import os
import re
import hashlib
from typing import Optional, List, Dict, Set, Tuple
from dataclasses import dataclass

from preprocessing.config import PreprocessingConfig


@dataclass
class ValidationStats:
    """Track validation statistics for reporting."""
    total_read: int = 0
    passed: int = 0
    failed_json: int = 0
    failed_length: int = 0
    failed_rating: int = 0
    failed_spam: int = 0
    failed_noise: int = 0
    failed_duplicate: int = 0
    failed_quality: int = 0


class ReviewValidator:
    """Validates and filters reviews based on quality heuristics."""
    
    def __init__(self):
        self.cfg = PreprocessingConfig
        self._seen_hashes: Set[str] = set()  # For duplicate detection
        
    def validate(self, item: Dict) -> Tuple[bool, str]:
        """Validate a review item. Returns (is_valid, rejection_reason)."""
        text = item.get('text', item.get('review_text', ''))
        
        # 1. Length validation
        if len(text) < self.cfg.MIN_REVIEW_LENGTH:
            return False, "too_short"
        if len(text) > self.cfg.MAX_REVIEW_LENGTH:
            text = text[:self.cfg.MAX_REVIEW_LENGTH]  # Truncate, don't reject
            
        # 2. Word count check
        words = text.split()
        if len(words) < self.cfg.MIN_WORD_COUNT:
            return False, "too_few_words"
            
        # 3. Rating validation
        rating = item.get('rating')
        if rating is not None:
            try:
                rating = float(rating)
                if not (self.cfg.MIN_RATING <= rating <= self.cfg.MAX_RATING):
                    return False, "invalid_rating"
            except (ValueError, TypeError):
                return False, "invalid_rating"
                
        # 4. Verified purchase filter (if enabled)
        if self.cfg.REQUIRE_VERIFIED_PURCHASE:
            if not item.get('verified_purchase', False):
                return False, "unverified"
                
        # 5. Spam pattern detection
        text_lower = text.lower()
        for pattern in self.cfg.SPAM_PATTERNS:
            if pattern in text_lower:
                return False, "spam_pattern"
                
        # 6. Noise heuristics (excessive caps, punctuation, digits)
        if not self._check_noise_ratios(text):
            return False, "noise_ratio"
            
        # 7. Quality heuristics
        if not self._check_content_quality(text):
            return False, "low_quality"
            
        # 8. Duplicate detection (content hash)
        content_hash = self._hash_content(text)
        if content_hash in self._seen_hashes:
            return False, "duplicate"
        self._seen_hashes.add(content_hash)
        
        return True, "ok"
    
    def _check_noise_ratios(self, text: str) -> bool:
        """Check for spam indicators via character ratio analysis."""
        if not text:
            return False
            
        total = len(text)
        if total == 0:
            return False
            
        # Uppercase ratio (ALL CAPS REVIEWS are often spam/bots)
        upper_count = sum(1 for c in text if c.isupper())
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count > 0 and (upper_count / alpha_count) > self.cfg.MAX_CAPS_RATIO:
            return False
            
        # Punctuation ratio (excessive !!!!! or ????? is noise)
        punct_count = sum(1 for c in text if c in '!?.,;:()[]{}@#$%^&*')
        if (punct_count / total) > self.cfg.MAX_PUNCTUATION_RATIO:
            return False
            
        # Digit ratio (reviews that are mostly numbers are usually garbage)
        digit_count = sum(1 for c in text if c.isdigit())
        if (digit_count / total) > self.cfg.MAX_DIGIT_RATIO:
            return False
            
        return True
    
    def _check_content_quality(self, text: str) -> bool:
        """Check for meaningful content (not just filler)."""
        text_lower = text.lower()
        
        # Reject single-word repeated reviews like "good good good good"
        words = text_lower.split()
        if len(words) >= 3:
            unique_words = set(words)
            if len(unique_words) <= 2:  # Only 1-2 unique words repeated
                return False
                
        # Reject reviews that are just product names or "N/A"
        filler_patterns = ['n/a', 'na', 'none', 'no comment', 'test', '...', '---']
        if text_lower.strip() in filler_patterns:
            return False
            
        return True
    
    def _hash_content(self, text: str) -> str:
        """Create content hash for duplicate detection."""
        # Normalize: lowercase, remove extra spaces, remove punctuation
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = ' '.join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()


class DatasetLoader:
    """Load and validate datasets with strict quality filtering."""

    @staticmethod
    def load_dataset(limit: Optional[int] = None, verbose: bool = True) -> List[Dict]:
        """Load dataset with validation and filtering.
        
        Args:
            limit: Maximum number of VALID reviews to return
            verbose: Print detailed statistics
            
        Returns:
            List of validated review records
        """
        # limit <= 0 means "no limit" (read full dataset)
        if limit is None:
            limit = PreprocessingConfig.DATASET_LIMIT
        
        if verbose:
            print(f"Loading dataset with validation (target: {limit} valid reviews)...")
            print(f"Dataset file: {PreprocessingConfig.DATASET_FILE}")

        # Find dataset file
        dataset_file = DatasetLoader._find_dataset_file()
        if not dataset_file:
            print("⚠ Dataset not found, using sample data")
            return DatasetLoader._get_sample_data()
            
        if verbose:
            print(f"Loading from: {dataset_file}")
        
        # Load and validate
        validator = ReviewValidator()
        stats = ValidationStats()
        dataset = []
        
        with open(dataset_file, 'r', encoding='utf-8') as fp:
            for line in fp:
                stats.total_read += 1
                line = line.strip()
                if not line:
                    continue
                    
                # Parse JSON
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    stats.failed_json += 1
                    continue
                
                # Validate
                is_valid, reason = validator.validate(item)
                
                if is_valid:
                    # Normalize the item for downstream processing
                    normalized = DatasetLoader._normalize_item(item)
                    dataset.append(normalized)
                    stats.passed += 1
                    
                    if limit and limit > 0 and stats.passed >= limit:
                        break
                else:
                    # Track rejection reason
                    if reason == "too_short" or reason == "too_few_words":
                        stats.failed_length += 1
                    elif reason == "invalid_rating":
                        stats.failed_rating += 1
                    elif reason == "spam_pattern":
                        stats.failed_spam += 1
                    elif reason == "noise_ratio":
                        stats.failed_noise += 1
                    elif reason == "duplicate":
                        stats.failed_duplicate += 1
                    else:
                        stats.failed_quality += 1
        
        # Print statistics
        if verbose:
            DatasetLoader._print_stats(stats)
            
        return dataset
    
    @staticmethod
    def _find_dataset_file() -> Optional[str]:
        """Find the dataset file in possible locations."""
        project_root = os.path.dirname(os.path.dirname(__file__))
        paths = [
            PreprocessingConfig.DATASET_FILE,
            os.path.join(PreprocessingConfig.DATA_DIR, PreprocessingConfig.DATASET_FILE),
            os.path.join("data", PreprocessingConfig.DATASET_FILE),
            os.path.join(project_root, "data", PreprocessingConfig.DATASET_FILE),
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    
    @staticmethod
    def _normalize_item(item: Dict) -> Dict:
        """Normalize field names for consistent downstream processing."""
        text = item.get('text', item.get('review_text', ''))
        
        # Truncate if needed
        if len(text) > PreprocessingConfig.MAX_REVIEW_LENGTH:
            text = text[:PreprocessingConfig.MAX_REVIEW_LENGTH]
            
        return {
            'text': text,
            'product_id': item.get('parent_asin', item.get('asin', 'unknown')),
            'asin': item.get('asin'),
            'parent_asin': item.get('parent_asin'),
            'product_name': item.get('title', item.get('product_name', '')),
            'rating': item.get('rating'),
            'verified_purchase': item.get('verified_purchase', False),
        }
    
    @staticmethod
    def _print_stats(stats: ValidationStats):
        """Print validation statistics."""
        total_rejected = (stats.failed_json + stats.failed_length + stats.failed_rating +
                         stats.failed_spam + stats.failed_noise + stats.failed_duplicate +
                         stats.failed_quality)
        
        print(f"\n✓ Validation complete:")
        print(f"  - Total read:      {stats.total_read:,}")
        print(f"  - Passed:          {stats.passed:,} ({100*stats.passed/max(stats.total_read,1):.1f}%)")
        print(f"  - Rejected:        {total_rejected:,}")
        if total_rejected > 0:
            print(f"    ├─ Invalid JSON: {stats.failed_json:,}")
            print(f"    ├─ Too short:    {stats.failed_length:,}")
            print(f"    ├─ Bad rating:   {stats.failed_rating:,}")
            print(f"    ├─ Spam:         {stats.failed_spam:,}")
            print(f"    ├─ Noise:        {stats.failed_noise:,}")
            print(f"    ├─ Duplicate:    {stats.failed_duplicate:,}")
            print(f"    └─ Low quality:  {stats.failed_quality:,}")

    @staticmethod
    def _get_sample_data() -> List[Dict]:
        """Fallback sample data for testing."""
        return [
            {"text": "For gaming this phone is incredibly smooth with high refresh rate and very fast processor. Even demanding games run without frame drops.",
             "product_id": "SAMSUNG_S23", "product_name": "Samsung Galaxy S23", "rating": 5},
            {"text": "Battery drains faster when gaming all day, but it still lasts through a normal day of mixed use with moderate screen time.",
             "product_id": "SAMSUNG_S23", "product_name": "Samsung Galaxy S23", "rating": 4},
            {"text": "The AMOLED screen is bright and colors pop beautifully. Great for watching movies and browsing photos on the go.",
             "product_id": "SAMSUNG_S23", "product_name": "Samsung Galaxy S23", "rating": 5},
            {"text": "The iPhone 15 Pro handles heavy games well but the phone gets warm after long gaming sessions which is concerning.",
             "product_id": "IPHONE_15_PRO", "product_name": "Apple iPhone 15 Pro", "rating": 4},
            {"text": "Battery life is solid for everyday use, and iOS feels very smooth with no lag even with many apps open.",
             "product_id": "IPHONE_15_PRO", "product_name": "Apple iPhone 15 Pro", "rating": 5},
            {"text": "The camera is fantastic, especially for video recording. Great stabilization and impressive low-light performance.",
             "product_id": "IPHONE_15_PRO", "product_name": "Apple iPhone 15 Pro", "rating": 5},
            {"text": "Pixel 8 is not the fastest for gaming but it is perfectly fine for casual games and everyday productivity apps.",
             "product_id": "PIXEL_8", "product_name": "Google Pixel 8", "rating": 4},
            {"text": "Battery life is excellent and the phone stays cool even under heavy navigation and continuous camera use.",
             "product_id": "PIXEL_8", "product_name": "Google Pixel 8", "rating": 5},
            {"text": "The camera quality is outstanding for still photos, with very natural colors and excellent detail preservation.",
             "product_id": "PIXEL_8", "product_name": "Google Pixel 8", "rating": 5},
        ]

