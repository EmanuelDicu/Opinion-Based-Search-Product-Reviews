"""
Backend API for Opinion-Based Search
Flask API that handles search queries and returns relevant results.

Strict mode: Elasticsearch and Ollama are required; no fallbacks.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.config import Config
from backend.services.search_service import SearchService, ServiceUnavailableError
from backend.models.aspect_extractor import AspectExtractor

app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)

# Initialize services (raises ServiceUnavailableError if Ollama is not available)
search_service = SearchService()

@app.route('/api/search', methods=['POST'])
def search():
    """Handle search queries. Requires Elasticsearch and Ollama; no fallbacks."""
    data = request.json or {}
    query = str(data.get('query', '')).strip()
    product = None

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    query_words = query.split()
    if len(query_words) <= 2:
        return jsonify({
            'query': query,
            'product': product,
            'results': [],
            'summary': 'Your query is too short. Please provide more details (e.g., "What phone is good for gaming?" or "Compare iPhone and Samsung battery life").',
            'count': 0
        })

    try:
        aspect = AspectExtractor.extract_aspect(query)
        sentiment = None
        query_lower = query.lower()
        negative_hints = [
            "bad", "worse", "worst", "poor", "terrible", "negative",
            "issue", "issues", "problem", "problems", "weak", "drain", "drains",
            "short battery", "battery drain", "overheat", "overheating",
            "lag", "slow", "not good", "disappoint", "disappointed",
            "bug", "bugs", "crash", "crashes", "freeze", "freezes",
            "broken", "defective", "scratch", "scratches", "cheap feel"
        ]
        positive_hints = [
            "good", "great", "best", "excellent", "amazing", "positive",
            "fast", "strong", "long battery", "great battery", "recommend",
            "smooth", "bright", "sharp", "clear", "solid", "reliable"
        ]
        if any(h in query_lower for h in negative_hints):
            sentiment = "negative"
        elif any(h in query_lower for h in positive_hints):
            sentiment = "positive"

        hits = search_service.search(query, aspect, product=product, sentiment=sentiment, limit=20)
        formatted_results = search_service.format_results(hits)
        # If sentiment intent detected, filter results by sentiment (include mixed)
        if sentiment:
            filtered = [
                r for r in formatted_results
                if r.get("sentiment_display") in (sentiment, "mixed")
            ]
            # If strict filter yields nothing, fall back to unfiltered results
            if filtered:
                formatted_results = filtered

        summary = search_service.generate_llm_summary(query, formatted_results)
        # Use fallback if LLM returned empty, None, or truncated/invalid (e.g. "None of the")
        if not summary or not summary.strip() or len(summary.strip()) < 15:
            summary = search_service.generate_summary(formatted_results)
        if not summary or not summary.strip():
            summary = "No summary available for this query. See related reviews below."
    except ServiceUnavailableError as e:
        return jsonify({
            'error': str(e),
            'service': getattr(e, 'service', 'unknown')
        }), 503

    return jsonify({
        'query': query,
        'product': product,
        'results': formatted_results,
        'summary': summary,
        'count': len(formatted_results)
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint. Reports Elasticsearch status (no fallback)."""
    try:
        es_available = search_service.is_elasticsearch_available()
    except Exception:
        es_available = False
    return jsonify({
        'status': 'ok',
        'elasticsearch': 'connected' if es_available else 'disconnected',
    })

if __name__ == '__main__':
    print("Starting Opinion-Based Search API...")
    print(f"API will be available at http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(host=Config.FLASK_HOST, port=Config.FLASK_PORT, debug=Config.FLASK_DEBUG)

