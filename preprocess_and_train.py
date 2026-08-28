"""
Preprocessing and ML Pipeline for Opinion-Based Search

This script implements the complete neuro-symbolic preprocessing pipeline:

1. **Data Ingestion**: Load product reviews from JSONL dataset
2. **Aspect Extraction**: Use LLM (default qwen2.5:3b for speed; config EXTRACTION_CHAT_MODEL) for aspect-opinion extraction
3. **Feature Inference**: Apply SWRL-like rules to derive Level 2 features (e.g., gaming_experience)
4. **Embedding Generation**: Create semantic vectors using qwen3-embedding:4b
5. **Indexing**: Store in Elasticsearch with dense vectors for hybrid search

Architecture inspired by:
- Recipe-Search project (ollama embeddings + Elasticsearch)
- "Inferring Features from Product Reviews" technical document

Usage:
    python preprocess_and_train.py [--limit N] [--no-llm] [--recreate-index]
    RECREATE_INDEX=true  # in Docker: delete and recreate ES index on startup
    SKIP_PREPROCESSING_IF_INDEX_EXISTS=true  # skip pipeline when index already exists
    NO_LLM=true  # in Docker: keyword-only extraction (no Ollama for aspect extraction)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional, Generator
from tqdm import tqdm

from preprocessing.config import PreprocessingConfig
from preprocessing.dataset_loader import DatasetLoader
from preprocessing.smartphone_filter import filter_smartphone_reviews
from preprocessing.text_processor import TextProcessor
from preprocessing.aspect_extractor import AspectOpinionExtractor
from preprocessing.elasticsearch_indexer import ElasticsearchIndexer
from preprocessing.feature_inference import FeatureInferenceEngine
from preprocessing.embedding_service import EmbeddingService, get_embedding_service

try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None


def _elasticsearch_index_exists() -> bool:
    """Return True if the configured Elasticsearch index already exists."""
    if Elasticsearch is None:
        return False
    try:
        es_config = PreprocessingConfig.get_elasticsearch_config()
        host = es_config["host"]
        port = es_config["port"]
        scheme = es_config["scheme"]
        url = f"{scheme}://{host}:{port}"
        es = Elasticsearch(hosts=[url], verify_certs=False, ssl_show_warn=False)
        if not es.ping():
            return False
        return es.indices.exists(index=PreprocessingConfig.ELASTICSEARCH_INDEX)
    except Exception:
        return False


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Opinion-Based Search Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocess_and_train.py                    # Run with defaults
  python preprocess_and_train.py --limit 1000       # Process only 1000 reviews
  python preprocess_and_train.py --no-llm           # Use keyword extraction (faster)
  python preprocess_and_train.py --recreate-index   # Delete and recreate ES index
        """
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of reviews to process (default: from config)"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based extraction (use keyword fallback)"
    )
    
    
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete existing Elasticsearch index and recreate"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Batch size for Elasticsearch indexing (default: 100)"
    )
    
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Only save extractions to file, skip Elasticsearch indexing"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for extractions (default: from config)"
    )
    
    parser.add_argument(
        "--no-smartphone-filter",
        action="store_true",
        help="Skip Ollama-based filter that keeps only smartphone reviews (not accessories)"
    )

    args = parser.parse_args()
    # Allow Docker/env to force recreate index (RECREATE_INDEX=true)
    if os.getenv("RECREATE_INDEX", "").lower() in ("1", "true", "yes"):
        args.recreate_index = True
    # Allow Docker/env to disable LLM extraction (NO_LLM=true → keyword-only)
    if os.getenv("NO_LLM", "").lower() in ("1", "true", "yes"):
        args.no_llm = True
    return args


def process_reviews(
    dataset: List[Dict],
    use_llm: bool = True,
    show_progress: bool = True
) -> List[Dict]:
    """Process reviews and extract aspects/opinions.

    Uses batched LLM extraction when use_llm=True (one Ollama call per batch of
    EXTRACTION_BATCH_SIZE reviews). Falls back to keyword extraction per review
    when LLM is disabled or batch fails.

    Args:
        dataset: List of review records from the dataset
        use_llm: Whether to use LLM extraction
        show_progress: Whether to show progress bar

    Returns:
        List of aspect-opinion extraction records
    """
    all_extractions: List[Dict] = []
    batch_size = PreprocessingConfig.EXTRACTION_BATCH_SIZE

    print("\n" + "=" * 50)
    print("Stage 1: Aspect-Opinion Extraction")
    print("=" * 50)

    extractor = AspectOpinionExtractor(use_llm=use_llm)

    # Build list of items to process (same validation as before)
    extraction_limit = getattr(PreprocessingConfig, "EXTRACTION_REVIEW_LIMIT", 200)
    to_process: List[Dict] = []
    stats = {
        "total_reviews": 0,
        "skipped_short": 0,
        "skipped_empty": 0,
    }
    for item in dataset:
        stats["total_reviews"] += 1
        review_text = item.get("text", item.get("review_text", ""))
        if not review_text:
            stats["skipped_empty"] += 1
            continue
        cleaned_text = TextProcessor.preprocess_text(review_text)
        if len(cleaned_text) < 15:
            stats["skipped_short"] += 1
            continue
        to_process.append({
            "text": cleaned_text,
            "review_text": cleaned_text,
            "product_id": item.get("parent_asin", item.get("asin", item.get("product_id", "unknown"))),
            "asin": item.get("asin"),
            "parent_asin": item.get("parent_asin"),
            "product_name": item.get("product_name", item.get("title", "")),
            "title": item.get("title", item.get("product_name", "")),
            "rating": item.get("rating"),
        })
        if extraction_limit and len(to_process) >= extraction_limit:
            break

    processed = len(to_process)
    if not to_process:
        print(f"\n✓ No reviews to process after validation.")
        return []

    if extraction_limit and processed >= extraction_limit:
        print(f"  Extraction limited to {extraction_limit} reviews (EXTRACTION_REVIEW_LIMIT).")

    # Pre-selection: only send aspect-rich reviews to LLM (others use keyword extraction only)
    preselect = getattr(PreprocessingConfig, "EXTRACTION_LLM_PRESELECT", True)
    llm_candidates: List[Dict] = []
    keyword_only: List[Dict] = []
    if use_llm and preselect:
        for item in to_process:
            text = item.get("text") or item.get("review_text") or ""
            if extractor.has_aspect_keyword(text):
                llm_candidates.append(item)
            else:
                keyword_only.append(item)
        if keyword_only:
            print(f"  Pre-selection: {len(llm_candidates)} reviews → LLM, {len(keyword_only)} → keyword-only")
    else:
        llm_candidates = to_process
        keyword_only = []

    # Keyword-only path (no LLM)
    extraction_count = 0
    implicit_count = 0
    if keyword_only:
        kw_extractions = extractor.extract_keyword_only_batch(keyword_only)
        extraction_count += len(kw_extractions)
        implicit_count += sum(1 for e in kw_extractions if e.get("is_implicit"))
        all_extractions.extend(kw_extractions)

    # LLM batches (with timeout; on timeout/error fall back to keyword for that batch)
    batches = [llm_candidates[i : i + batch_size] for i in range(0, len(llm_candidates), batch_size)]
    if batches:
        print(f"  Processing {len(llm_candidates)} reviews in {len(batches)} LLM batch(es) of up to {batch_size}")
    iterator = tqdm(batches, desc="Extracting aspects (batch)") if show_progress else batches
    for idx, batch in enumerate(iterator):
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(batch=idx + 1, total=len(batches))
        extractions = extractor.extract_aspects_opinions_batch(batch)
        extraction_count += len(extractions)
        implicit_count += sum(1 for e in extractions if e.get("is_implicit"))
        all_extractions.extend(extractions)

    print(f"\n✓ Extraction complete:")
    print(f"  - Total reviews: {stats['total_reviews']}")
    print(f"  - Processed: {processed}")
    print(f"  - Skipped (empty): {stats['skipped_empty']}")
    print(f"  - Skipped (short): {stats['skipped_short']}")
    print(f"  - LLM batches: {len(batches)}")
    print(f"  - Keyword-only: {len(keyword_only)}")
    print(f"  - Aspect-opinion pairs: {extraction_count}")
    print(f"  - Implicit aspects: {implicit_count}")

    return all_extractions


def infer_features(extractions: List[Dict]) -> List[Dict]:
    """Infer high-level features using weighted aspect aggregation.
    
    Uses asymmetric weights where positive/negative sentiment 
    can have different importance per aspect.
    
    Args:
        extractions: List of aspect-opinion records
        
    Returns:
        Extractions augmented with inferred features
    """
    print("\n" + "=" * 50)
    print("Stage 2: Weighted Feature Inference")
    print("=" * 50)
    
    inference_engine = FeatureInferenceEngine()
    enriched_extractions = inference_engine.infer_features(extractions)
    
    # Count inferred features
    inferred_count = sum(1 for e in enriched_extractions if e.get("is_inferred"))
    print(f"\n✓ Inference complete:")
    print(f"  - Original extractions: {len(extractions)}")
    print(f"  - Inferred features: {inferred_count}")
    print(f"  - Total records: {len(enriched_extractions)}")
    
    return enriched_extractions


def save_extractions_to_file(
    extractions: List[Dict], 
    filename: Optional[str] = None
) -> str:
    """Save extractions to JSON file as backup.
    
    Args:
        extractions: List of extraction records
        filename: Output file path (default: from config)
        
    Returns:
        Path to saved file
    """
    filename = filename or PreprocessingConfig.EXTRACTIONS_FILE
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
    
    print(f"\nSaving extractions to {filename}...")
    
    # Remove vector field for JSON save (too large)
    extractions_for_save = []
    for ext in extractions:
        ext_copy = {k: v for k, v in ext.items() if k != "review_vector"}
        extractions_for_save.append(ext_copy)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(extractions_for_save, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved {len(extractions)} extractions to {filename}")
    return filename


def index_in_elasticsearch(
    extractions: List[Dict],
    batch_size: int = 100,
    recreate_index: bool = False
) -> bool:
    """Index extractions in Elasticsearch with embeddings.
    
    Creates dense vector embeddings and indexes for hybrid search.
    
    Args:
        extractions: List of extraction records
        batch_size: Batch size for bulk indexing
        recreate_index: Whether to delete and recreate index
        
    Returns:
        True if indexing successful
    """
    print("\n" + "=" * 50)
    print("Stage 3: Elasticsearch Indexing")
    print("=" * 50)
    
    indexer = ElasticsearchIndexer()

    if not indexer.embedding_service.is_available:
        print("✗ Embeddings are required. Ollama must be running with the embedding model.")
        print(f"  Run: ollama pull {PreprocessingConfig.EMBEDDING_MODEL}")
        print("  Set OLLAMA_HOST if running in Docker (e.g. http://host.docker.internal:11434)")
        return False

    if not indexer.is_available():
        print("✗ Elasticsearch is not available.")
        print("  Make sure Elasticsearch is running on",
              f"{PreprocessingConfig.ELASTICSEARCH_HOST}:{PreprocessingConfig.ELASTICSEARCH_PORT}")
        return False

    # Create or recreate index
    if recreate_index:
        indexer.delete_index()
    
    if not indexer.create_index(delete_existing=recreate_index):
        return False
    
    # Index documents with embeddings
    success, failed = indexer.index_documents(
        extractions,
        batch_size=batch_size,
        show_progress=True
    )
    
    # Print index stats
    stats = indexer.get_index_stats()
    if stats:
        print(f"\n✓ Index statistics:")
        print(f"  - Index name: {stats['index_name']}")
        print(f"  - Document count: {stats['document_count']}")
        print(f"  - Size: {stats['size_bytes'] / 1024 / 1024:.2f} MB")
    
    return success > 0


def main():
    """Main pipeline execution."""
    args = parse_arguments()

    # Option: skip full pipeline if index already exists (e.g. Docker restart)
    skip_if_index_exists = os.getenv("SKIP_PREPROCESSING_IF_INDEX_EXISTS", "").lower() in ("1", "true", "yes")
    if skip_if_index_exists and not args.recreate_index:
        if _elasticsearch_index_exists():
            print("\n✓ Elasticsearch index already exists. Skipping preprocessing (SKIP_PREPROCESSING_IF_INDEX_EXISTS=true).")
            return 0
    
    print("\n" + "=" * 60)
    print("  Opinion-Based Search - Preprocessing Pipeline")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  - Dataset file: {PreprocessingConfig.DATASET_FILE}")
    limit_label = args.limit if args.limit is not None else "ALL"
    print(f"  - Limit: {limit_label}")
    print(f"  - LLM extraction: {'disabled' if args.no_llm else 'enabled'}")
    print(f"  - Embedding model: {PreprocessingConfig.EMBEDDING_MODEL}")
    print(f"  - ES index: {PreprocessingConfig.ELASTICSEARCH_INDEX}")
    
    # 1. Load dataset
    print("\n" + "-" * 50)
    print("Loading dataset...")
    dataset = DatasetLoader.load_dataset(
        limit=args.limit if args.limit is not None else PreprocessingConfig.DATASET_LIMIT
    )
    
    if not dataset:
        print("✗ No dataset available. Exiting.")
        return 1
    
    print(f"✓ Loaded {len(dataset)} reviews")
    
    # 1b. Filter to smartphones only (exclude accessories: cases, chargers, etc.)
    if not args.no_smartphone_filter:
        print("\n" + "-" * 50)
        print("Filtering to smartphone reviews only (excluding accessories)...")
        dataset = filter_smartphone_reviews(
            dataset,
            batch_size=12,
            use_llm=not args.no_llm,
            verbose=True,
        )
        if not dataset:
            print("✗ No smartphone reviews after filtering. Exiting.")
            return 1
        print(f"✓ {len(dataset)} smartphone reviews to process")
    else:
        print("  (Smartphone filter skipped)")
    
    # 2. Extract aspects and opinions
    extractions = process_reviews(
        dataset,
        use_llm=not args.no_llm
    )
    
    if not extractions:
        print("✗ No extractions found. Exiting.")
        return 1
    
    # 3. Infer high-level features (weighted)
    enriched_extractions = infer_features(extractions)
    
    # 4. Save to file
    output_file = args.output or PreprocessingConfig.EXTRACTIONS_FILE
    save_extractions_to_file(enriched_extractions, output_file)
    
    # 5. Index in Elasticsearch (unless save-only mode)
    if not args.save_only:
        if not index_in_elasticsearch(
            enriched_extractions,
            batch_size=args.batch_size,
            recreate_index=args.recreate_index
        ):
            print("\n✗ Indexing failed. Exiting with error.")
            return 1

    print("\n" + "=" * 60)
    print("  Pipeline completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
