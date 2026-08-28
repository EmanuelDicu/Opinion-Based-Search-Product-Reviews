"""Fallback search service using in-memory data."""

import json
import os
from typing import Dict, List, Optional

from backend.config import Config


class FallbackSearchService:
    """Fallback search service for when Elasticsearch is unavailable."""

    def __init__(self) -> None:
        self.data: List[Dict] = []
        self._load_data()

    def _load_data(self) -> None:
        file_path = Config.EXTRACTIONS_FILE
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    self.data = json.load(f)
                print(f"Loaded {len(self.data)} fallback records from {file_path}")
            except Exception as exc:
                print(f"Error loading fallback data: {exc}")
                self.data = []
        else:
            print(f"Fallback data file not found: {file_path}")
            self.data = []

    def search(
        self,
        query: str,
        aspect: Optional[str] = None,
        product: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search using in-memory data with optional aspect and product filters."""
        results: List[Dict] = []
        query_lower = query.lower()
        product_lower = product.lower() if product else None
        aspect_lower_filter = aspect.lower() if aspect else None

        # Extract products from query for better matching
        mentioned_products = self._extract_products_from_query(query_lower)
        if mentioned_products:
            # If products are mentioned, prioritize matching them
            product_lower = mentioned_products[0] if not product_lower else product_lower

        for item in self.data:
            review_text_lower = item.get("review_text", "").lower()
            opinion_lower = item.get("opinion", "").lower()
            aspect_lower = item.get("aspect", "").lower()
            product_name_lower = item.get("product_name", "").lower()

            if not self._matches_query(query_lower, review_text_lower, opinion_lower, aspect_lower):
                continue
            if aspect_lower_filter and aspect_lower_filter not in aspect_lower:
                continue
            
            # Improved product matching with normalization
            if product_lower or mentioned_products:
                if not self._matches_product(product_lower or mentioned_products[0], product_name_lower, review_text_lower, opinion_lower):
                    continue

            results.append({"_source": item})
            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _matches_query(query: str, review: str, opinion: str, aspect: str) -> bool:
        return query in review or query in opinion or query in aspect

    @staticmethod
    def _matches_product(product: str, product_name: str, review: str, opinion: str) -> bool:
        """Improved product matching with normalization."""
        # Normalize product names (e.g., "iphone" matches "iPhone", "IPHONE", etc.)
        product_normalized = product.lower().strip()
        product_name_normalized = product_name.lower().strip()
        
        # Direct match
        if product_normalized in product_name_normalized or product_name_normalized in product_normalized:
            return True
        
        # Check if product brand/model keywords match
        # e.g., "iphone" matches "Apple iPhone 15 Pro"
        brand_keywords = {
            'iphone': ['iphone', 'apple'],
            'samsung': ['samsung', 'galaxy'],
            'samsung galaxy': ['samsung', 'galaxy'],
            'google pixel': ['pixel', 'google'],
            'oneplus': ['oneplus', 'one plus'],
            'xiaomi': ['xiaomi', 'mi'],
            'huawei': ['huawei'],
            'oppo': ['oppo'],
            'vivo': ['vivo'],
            'motorola': ['motorola', 'moto'],
            'nokia': ['nokia'],
            'sony': ['sony', 'xperia'],
            'lg': ['lg'],
        }
        
        # Check keyword matching
        keywords = brand_keywords.get(product_normalized, [product_normalized])
        if any(kw in product_name_normalized for kw in keywords):
            return True
        
        # Fallback: check if product appears in review text
        return product_normalized in review or product_normalized in opinion
    
    @staticmethod
    def _extract_products_from_query(query: str) -> List[str]:
        """Extract product names/brands from query with normalization."""
        products = []
        query_lower = query.lower()
        
        # Common product name patterns (normalized) - order matters (more specific first)
        product_patterns = [
            ('google pixel', ['pixel', 'google pixel']),
            ('iphone', ['iphone', 'apple iphone', 'iphone 15', 'iphone 14', 'iphone 13', 'iphone 12', 'iphone 11']),
            ('samsung galaxy', ['galaxy', 'samsung galaxy', 'galaxy s23', 'galaxy s24', 'galaxy s22']),
            ('samsung', ['samsung']),
            ('oneplus', ['oneplus', 'one plus']),
            ('xiaomi', ['xiaomi', 'mi']),
            ('huawei', ['huawei']),
            ('oppo', ['oppo']),
            ('vivo', ['vivo']),
            ('motorola', ['motorola', 'moto']),
            ('nokia', ['nokia']),
            ('sony', ['sony', 'xperia']),
            ('lg', ['lg']),
        ]
        
        for normalized_name, variations in product_patterns:
            if any(var in query_lower for var in variations):
                products.append(normalized_name)
                # Remove matched variations from query to avoid duplicate matches
                for var in variations:
                    query_lower = query_lower.replace(var, '')
        
        return products

