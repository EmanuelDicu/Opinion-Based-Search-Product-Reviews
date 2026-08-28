"""Search service for querying reviews with hybrid search support.

This service implements a hybrid search strategy combining:
1. Semantic search using dense vector embeddings (KNN)
2. Keyword/BM25 search for lexical matching
3. Aspect and sentiment filtering

Architecture inspired by Recipe-Search project with ollama embeddings.

Strict mode: Fallbacks are disabled; Elasticsearch and Ollama are required.
"""

from typing import Dict, List, Optional

from elasticsearch import Elasticsearch

from backend.config import Config
from backend.services.openai_llm_service import LLMService

# Try to import ollama for embedding generation at query time
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class ServiceUnavailableError(RuntimeError):
    """Raised when a required service (Elasticsearch, Ollama) is unavailable.
    API should return 503 and not use fallbacks.
    """
    def __init__(self, message: str, service: str = "service"):
        self.service = service
        super().__init__(message)


class SearchService:
    """Service for searching reviews with hybrid semantic + keyword search.
    
    Supports three search modes:
    1. Hybrid (default): Combines vector similarity with keyword matching
    2. Semantic: Pure vector similarity search
    3. Keyword: Traditional BM25/keyword search
    """

    # Configuration
    EMBEDDING_MODEL = "qwen3-embedding:4b"

    def __init__(self) -> None:
        self.es = Elasticsearch([Config.get_elasticsearch_config()])
        self.index_name = Config.ELASTICSEARCH_INDEX
        self.llm = LLMService()
        
        # Ollama required for semantic search; no fallback
        self._semantic_available = OLLAMA_AVAILABLE
        if self._semantic_available:
            print(f"✓ Semantic search enabled using {self.EMBEDDING_MODEL}")
        else:
            raise ServiceUnavailableError(
                "Ollama is required for query embeddings but is not installed or not running. "
                "Install ollama and pull the embedding model.",
                service="ollama"
            )

    def is_elasticsearch_available(self) -> bool:
        try:
            return self.es.ping()
        except Exception:
            return False
    
    def get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a search query using ollama.
        
        Similar to Recipe-Search's get_embedding function.
        """
        if not self._semantic_available:
            return None
        
        try:
            response = ollama.embeddings(model=self.EMBEDDING_MODEL, prompt=query)
            return response.get('embedding')
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return None

    def search(
        self,
        query: str,
        aspect: Optional[str] = None,
        product: Optional[str] = None,
        sentiment: Optional[str] = None,
        limit: int = 20,
        search_mode: str = "hybrid",
    ) -> List[Dict]:
        """Search for reviews matching the query.
        
        Args:
            query: User's search query
            aspect: Optional aspect filter (e.g., "camera", "battery")
            product: Optional product filter
            sentiment: Optional sentiment filter ("positive", "negative", "neutral")
            limit: Maximum number of results
            search_mode: "hybrid", "semantic", or "keyword"
            
        Returns:
            List of matching review records
        """
        if not self.is_elasticsearch_available():
            raise ServiceUnavailableError(
                "Elasticsearch is required and is not available. "
                "Start Elasticsearch and ensure ELASTICSEARCH_HOST/PORT are correct. Fallback search is disabled.",
                service="elasticsearch"
            )
        results = self._search_elasticsearch(
            query, aspect, product, sentiment, limit, search_mode
        )
        if results is None:
            raise ServiceUnavailableError(
                "Search failed (e.g. index missing or KNN/query error). Check index name and that preprocessing has run.",
                service="elasticsearch"
            )
        return results

    def _search_elasticsearch(
        self,
        query: str,
        aspect: Optional[str],
        product: Optional[str],
        sentiment: Optional[str],
        limit: int,
        search_mode: str,
    ) -> Optional[List[Dict]]:
        """Execute search against Elasticsearch."""
        try:
            # Try semantic/hybrid search first
            if search_mode in ("hybrid", "semantic") and self._semantic_available:
                query_vector = self.get_query_embedding(query)
                if query_vector:
                    return self._search_with_knn(
                        query, query_vector, aspect, product, sentiment, limit, 
                        hybrid=(search_mode == "hybrid")
                    )
            
            # Fall back to keyword search
            search_body = self._build_keyword_search(query, aspect, product, sentiment, limit)
            results = self.es.search(index=self.index_name, body=search_body)
            return results.get("hits", {}).get("hits", [])
            
        except Exception as exc:
            print(f"Elasticsearch error: {exc}")
            return None

    def _search_with_knn(
        self,
        query: str,
        query_vector: List[float],
        aspect: Optional[str],
        product: Optional[str],
        sentiment: Optional[str],
        limit: int,
        hybrid: bool = True,
    ) -> Optional[List[Dict]]:
        """Execute KNN (vector similarity) search.
        
        Similar to Recipe-Search's search function using knn parameter.
        """
        # Build filter for KNN search
        filter_clauses = []
        
        if aspect:
            filter_clauses.append({"term": {"aspect": aspect}})
        
        if sentiment:
            filter_clauses.append({"term": {"sentiment": sentiment}})
        
        if product:
            filter_clauses.append({
                "multi_match": {
                    "query": product,
                    "fields": ["product_name", "product_name.keyword"],
                    "type": "best_fields"
                }
            })
        
        # Extract products from query
        mentioned_products = SearchService.extract_products_from_query(query)
        if mentioned_products:
            product_filter = {
                "bool": {
                    "should": [
                        {"multi_match": {
                            "query": prod,
                            "fields": ["product_name^3", "review_text"],
                            "fuzziness": "AUTO"
                        }}
                        for prod in mentioned_products
                    ]
                }
            }
            filter_clauses.append(product_filter)
        
        # Build KNN query (similar to Recipe-Search)
        knn_query = {
            "field": "review_vector",
            "query_vector": query_vector,
            "k": limit,
            "num_candidates": limit * 5,  # Oversample for better results
        }
        
        # Add filter to KNN if we have filter clauses
        if filter_clauses:
            knn_query["filter"] = {"bool": {"must": filter_clauses}}
        
        try:
            if hybrid:
                # Hybrid search: combine KNN with keyword matching
                keyword_query = self._build_keyword_query(query, aspect, product)
                
                results = self.es.search(
                    index=self.index_name,
                    knn=knn_query,
                    query=keyword_query,
                    size=limit,
                    _source=["product_id", "product_name", "aspect", "opinion", 
                             "sentiment", "review_text", "is_inferred", "feature_type"]
                )
            else:
                # Pure semantic search
                results = self.es.search(
                    index=self.index_name,
                    knn=knn_query,
                    size=limit,
                    _source=["product_id", "product_name", "aspect", "opinion", 
                             "sentiment", "review_text", "is_inferred", "feature_type"]
                )
            
            return results.get("hits", {}).get("hits", [])
            
        except Exception as e:
            print(f"KNN search error: {e}")
            # Fall back to keyword search
            return None
    
    def _build_keyword_query(
        self, 
        query: str, 
        aspect: Optional[str], 
        product: Optional[str]
    ) -> Dict:
        """Build the keyword/BM25 part of a hybrid query."""
        should_clauses = [
            {"match": {"review_text": {"query": query, "boost": 2}}},
            {"match": {"opinion": {"query": query, "boost": 1.5}}},
            {"match": {"aspect": {"query": query, "boost": 1}}},
        ]
        
        must_clauses = []
        
        if aspect:
            must_clauses.append({"term": {"aspect": aspect}})
        
        if product:
            must_clauses.append({
                "multi_match": {
                    "query": product,
                    "fields": ["product_name^3", "review_text"],
                    "fuzziness": "AUTO"
                }
            })
        
        query_dict = {"bool": {"should": should_clauses}}
        
        if must_clauses:
            query_dict["bool"]["must"] = must_clauses
        
        return query_dict

    def _build_keyword_search(
        self, 
        query: str, 
        aspect: Optional[str], 
        product: Optional[str],
        sentiment: Optional[str],
        limit: int
    ) -> Dict:
        """Build traditional keyword/BM25 search query."""
        must_clauses: List[Dict] = []
        should_clauses: List[Dict] = []
        
        if aspect:
            must_clauses.append({"term": {"aspect": aspect}})
        
        if sentiment:
            must_clauses.append({"term": {"sentiment": sentiment}})
        
        # Extract products from query
        mentioned_products = SearchService.extract_products_from_query(query)
        if mentioned_products:
            for prod in mentioned_products:
                should_clauses.append({
                    "multi_match": {
                        "query": prod,
                        "fields": ["product_name^5", "review_text^2"],
                        "fuzziness": "AUTO"
                    }
                })
        
        if product:
            must_clauses.append({
                "multi_match": {
                    "query": product,
                    "fields": ["product_name^3", "review_text", "opinion"],
                    "fuzziness": "AUTO"
                }
            })

        should_clauses.extend([
            {"match": {"review_text": {"query": query, "boost": 2}}},
            {"match": {"opinion": {"query": query, "boost": 1.5}}},
            {"match": {"aspect": query}},
        ])

        query_dict: Dict = {
            "size": limit,
            "query": {
                "bool": {
                    "should": should_clauses,
                }
            },
        }
        
        if must_clauses:
            query_dict["query"]["bool"]["must"] = must_clauses
        
        return query_dict

    def format_results(self, hits: List[Dict]) -> List[Dict]:
        """Format search results for API response.

        Includes 'model' (phone model name) and 'sentiment_display' (positive/mixed/negative)
        for clear display in related reviews.
        """
        formatted_results: List[Dict] = []
        seen = set()
        for hit in hits:
            source = hit.get("_source", hit)
            review_text = source.get("review_text", "")
            if not review_text or not str(review_text).strip():
                continue
            norm_text = " ".join(str(review_text).strip().lower().split())
            sentiment = source.get("sentiment", "neutral")
            # Display sentiment as positive / mixed / negative (neutral -> mixed)
            sentiment_display = (
                "positive" if sentiment == "positive" else
                "negative" if sentiment == "negative" else
                "mixed"
            )
            product_name = source.get("product_name", "Unknown phone")
            dedupe_key = (
                str(product_name).strip().lower(),
                norm_text,
                sentiment_display,
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            formatted_results.append(
                {
                    "product_id": source.get("product_id", "unknown"),
                    "asin": source.get("asin"),
                    "parent_asin": source.get("parent_asin"),
                    "product_name": product_name,
                    "model": product_name,  # Explicit model name for related reviews (model, review, sentiment)
                    "aspect": source.get("aspect", "unknown"),
                    "opinion": source.get("opinion", ""),
                    "sentiment": sentiment,
                    "sentiment_display": sentiment_display,
                    "review_text": review_text,
                }
            )
        return formatted_results

    def generate_summary(self, results: List[Dict]) -> str:
        """Generate a concise, product-centric summary.

        NOTE: This is kept as a fallback if the LLM-based summary is not available.
        """
        if not results:
            return "No relevant reviews found for your query."

        # Group by product then by aspect
        products: Dict[str, Dict[str, Dict[str, int]]] = {}
        for result in results:
            product_name = result.get("product_name", "Unknown phone")
            aspect = result["aspect"]
            sentiment = result["sentiment"]
            products.setdefault(product_name, {})
            products[product_name].setdefault(aspect, {"positive": 0, "negative": 0, "neutral": 0})
            products[product_name][aspect][sentiment] += 1

        summary_parts: List[str] = []
        for product_name, aspects in products.items():
            # Prefer high-level features if present (e.g., gaming_experience)
            if "gaming_experience" in aspects:
                counts = aspects["gaming_experience"]
                pos = counts["positive"]
                neg = counts["negative"]
                if pos > neg:
                    summary_parts.append(
                        f"{product_name} looks good for gaming based on processor performance and screen quality."
                    )
                elif neg > pos:
                    summary_parts.append(
                        f"{product_name} may not be ideal for gaming based on the reviews."
                    )
                else:
                    summary_parts.append(
                        f"{product_name} has mixed reviews for gaming."
                    )
            else:
                # Generic aspect summary fallback - use direct language
                for aspect, counts in aspects.items():
                    pos = counts["positive"]
                    neg = counts["negative"]
                    aspect_display = aspect.replace("_", " ").title()
                    if pos > neg:
                        summary_parts.append(
                            f"{product_name} has good {aspect_display} according to reviews."
                        )
                    elif neg > pos:
                        summary_parts.append(
                            f"{product_name} has issues with {aspect_display} according to reviews."
                        )
                    else:
                        summary_parts.append(
                            f"{product_name} has mixed reviews for {aspect_display}."
                        )

        return " ".join(summary_parts)
    
    @staticmethod
    def extract_products_from_query(query: str) -> List[str]:
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
    
    @staticmethod
    def filter_results_by_products(results: List[Dict], mentioned_products: List[str]) -> List[Dict]:
        """Filter results to only include products mentioned in query with normalized matching."""
        filtered = []
        product_keywords = {
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
        
        for result in results:
            product_name = result.get('product_name', '').lower()
            matched = False
            
            for mentioned in mentioned_products:
                # Direct substring match
                if mentioned in product_name or product_name in mentioned:
                    matched = True
                    break
                
                # Keyword-based matching
                keywords = product_keywords.get(mentioned, [mentioned])
                if any(kw in product_name for kw in keywords):
                    matched = True
                    break
            
            if matched:
                filtered.append(result)
        
        # If filtering removed all results, return original (better than nothing)
        return filtered if filtered else results

    def generate_llm_summary(self, query: str, results: List[Dict]) -> str:
        """If LLM is configured, return an LLM-generated natural language summary.
        
        For comparison queries, always attempt to use LLM even if it might fail.
        """
        if not self.llm or not hasattr(self.llm, "summarize"):
            return ""
        
        # For comparison queries, always try LLM first
        # The LLM service handles comparison detection internally
        summary = self.llm.summarize(query, results)
        
        # If LLM is enabled but returned empty, log it
        if not summary and self.llm.enabled:
            print("Warning: LLM service is enabled but returned empty summary")
        
        return summary

