"""Aspect and opinion extraction from reviews using LLM-based extraction.

This module implements a neuro-symbolic approach to aspect extraction:
1. Uses LLM (ollama; default qwen2.5:3b for speed, configurable) for aspect identification
2. Extracts both explicit and implicit aspects from review text
3. Falls back to keyword-based extraction when LLM is unavailable

Based on the methodology from "Inferring Features from Product Reviews":
- Level 0: Explicit Attribute Layer (raw specs)
- Level 1: Implicit Aspect Layer (inferred from context)
- Level 2: Inferred Concept Layer (high-level features)
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional, Any

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not installed. Using keyword-based extraction.")

from preprocessing.config import PreprocessingConfig
from preprocessing.text_processor import TextProcessor


@dataclass(frozen=True)
class AspectLexicon:
    """Lexicon for keyword-based aspect detection (fallback)."""
    name: str
    keywords: Sequence[str]


@dataclass
class ExtractedAspect:
    """Structured output from aspect extraction."""
    aspect: str
    opinion: str
    sentiment: str
    confidence: float = 1.0
    is_implicit: bool = False


class AspectOpinionExtractor:
    """Extract aspects and opinion snippets from review text.
    
    Uses a hybrid approach:
    1. LLM-based extraction (primary) - default qwen2.5:3b for speed, configurable via EXTRACTION_CHAT_MODEL
    2. Keyword-based extraction (fallback) - when LLM unavailable
    
    This implements the "Double Propagation" and "Implicit Aspect Clue" 
    concepts from the research paper, augmented with LLM capabilities.
    """

    # Keyword lexicons for fallback extraction
    ASPECT_LEXICONS: Sequence[AspectLexicon] = (
        AspectLexicon("camera", [
            "camera", "photo", "picture", "image", "lens", "zoom", "selfie", 
            "portrait", "night mode", "video", "recording", "megapixel", "mp"
        ]),
        AspectLexicon("battery", [
            "battery", "charge", "charging", "power", "endurance", "life",
            "drain", "last", "mah", "wireless charging", "fast charging"
        ]),
        AspectLexicon("screen", [
            "screen", "display", "resolution", "brightness", "clarity", "amoled",
            "oled", "lcd", "refresh rate", "hz", "hdr", "color", "vivid"
        ]),
        AspectLexicon("processor", [
            "processor", "cpu", "chip", "performance", "speed", "fast", "slow",
            "lag", "snapdragon", "a17", "gaming", "multitasking", "responsive"
        ]),
        AspectLexicon("design", [
            "design", "look", "appearance", "build", "material", "finish",
            "premium", "plastic", "glass", "metal", "weight", "thin", "size"
        ]),
        AspectLexicon("sound", [
            "sound", "speaker", "audio", "volume", "quality", "music",
            "bass", "stereo", "dolby", "headphone", "earbuds"
        ]),
        AspectLexicon("software", [
            "software", "os", "interface", "ui", "app", "system", "android",
            "ios", "update", "bloatware", "smooth", "feature"
        ]),
        AspectLexicon("price", [
            "price", "cost", "expensive", "cheap", "value", "affordable",
            "worth", "money", "budget", "premium", "flagship"
        ]),
        AspectLexicon("durability", [
            "durable", "durability", "waterproof", "water resistant", "ip68",
            "drop", "scratch", "gorilla glass", "rugged", "sturdy"
        ]),
        AspectLexicon("connectivity", [
            "5g", "4g", "lte", "wifi", "bluetooth", "nfc", "signal", 
            "network", "connection", "reception"
        ]),
        AspectLexicon("storage", [
            "storage", "memory", "gb", "ram", "internal", "expandable",
            "sd card", "space", "capacity"
        ]),
    )

    # Implicit Aspect Clues (IAC) mapping - adjectives that imply aspects
    IMPLICIT_ASPECT_CLUES: Dict[str, str] = {
        # Battery-related clues
        "dies quickly": "battery",
        "lasts forever": "battery",
        "drains fast": "battery",
        "all day": "battery",
        
        # Performance-related clues
        "snappy": "processor",
        "sluggish": "processor",
        "buttery smooth": "processor",
        "freezes": "processor",
        "crashes": "processor",
        
        # Design-related clues
        "heavy": "design",
        "lightweight": "design",
        "sleek": "design",
        "bulky": "design",
        "premium feel": "design",
        "cheap feel": "design",
        
        # Screen-related clues
        "gorgeous": "screen",
        "vibrant": "screen",
        "washed out": "screen",
        "eye strain": "screen",
        
        # Camera-related clues
        "sharp photos": "camera",
        "blurry": "camera",
        "grainy": "camera",
        "crisp images": "camera",
        
        # Sound-related clues
        "tinny": "sound",
        "loud": "sound",
        "muffled": "sound",
        "clear audio": "sound",
    }

    # Sentiment indicators
    POSITIVE_INDICATORS = (
        "good", "great", "excellent", "amazing", "love", "perfect", "best",
        "wonderful", "smooth", "fast", "fantastic", "awesome", "outstanding",
        "superb", "brilliant", "impressive", "remarkable", "exceptional",
        "satisfied", "happy", "pleased", "recommend", "worth"
    )
    
    NEGATIVE_INDICATORS = (
        "bad", "poor", "terrible", "awful", "worst", "hate", "disappointed",
        "slow", "hot", "horrible", "frustrating", "annoying", "useless",
        "broken", "defective", "regret", "waste", "overpriced", "problem"
    )

    def __init__(self, use_llm: bool = None):
        """Initialize the extractor.

        Args:
            use_llm: Whether to use LLM extraction. If None, uses config setting.
        """
        self.use_llm = use_llm if use_llm is not None else PreprocessingConfig.USE_LLM_EXTRACTION
        self.chat_model = getattr(
            PreprocessingConfig, "EXTRACTION_CHAT_MODEL", None
        ) or PreprocessingConfig.CHAT_MODEL
        self._llm_available = OLLAMA_AVAILABLE and self.use_llm
        self._request_timeout = getattr(
            PreprocessingConfig, "EXTRACTION_REQUEST_TIMEOUT", 300
        )
        self._ollama_client = None
        if self._llm_available and OLLAMA_AVAILABLE:
            try:
                self._ollama_client = ollama.Client(timeout=self._request_timeout)
            except Exception:
                self._ollama_client = None

        if self._llm_available:
            print(f"✓ LLM-based extraction enabled using {self.chat_model} (timeout={self._request_timeout}s)")
        else:
            print("⚠ Using keyword-based extraction (LLM disabled or unavailable)")

    def extract_aspects_opinions(
        self, 
        review_text: str, 
        product_id: str, 
        product_name: Optional[str] = None,
        rating: Optional[float] = None
    ) -> List[Dict]:
        """Extract aspects and opinions from a review.
        
        This method implements the hybrid neuro-symbolic approach:
        1. Try LLM-based extraction first (neural)
        2. Fall back to keyword-based extraction (symbolic)
        3. Apply implicit aspect clue detection
        
        Args:
            review_text: The review text to analyze
            product_id: Product identifier
            product_name: Optional product name
            rating: Optional rating (1-5) to inform sentiment
            
        Returns:
            List of aspect-opinion records
        """
        if not review_text or len(review_text.strip()) < 10:
            return []

        # Try LLM-based extraction first
        if self._llm_available:
            try:
                results = self._extract_with_llm(review_text, product_id, product_name, rating)
                if results:
                    return results
            except Exception as e:
                print(f"LLM extraction failed: {e}. Falling back to keywords.")

        # Fallback to keyword-based extraction
        return self._extract_with_keywords(review_text, product_id, product_name, rating)

    def has_aspect_keyword(self, text: str) -> bool:
        """Return True if the text contains any aspect-related keyword (for pre-selection)."""
        if not text or len(text.strip()) < 10:
            return False
        text_lower = text.lower()
        return any(self._contains_any(text_lower, lex.keywords) for lex in self.ASPECT_LEXICONS)

    def extract_keyword_only_batch(self, items: List[Dict]) -> List[Dict]:
        """Run keyword-based extraction only (no LLM) on a list of items. Fast fallback."""
        results: List[Dict] = []
        for item in items:
            text = (item.get("text") or item.get("review_text") or "").strip()
            if not text or len(text) < 10:
                continue
            extracted = self._extract_with_keywords(
                text,
                item.get("product_id", "unknown"),
                item.get("product_name") or item.get("title"),
                item.get("rating"),
            )
            if item.get("asin") or item.get("parent_asin"):
                for rec in extracted:
                    if item.get("asin"):
                        rec["asin"] = item.get("asin")
                    if item.get("parent_asin"):
                        rec["parent_asin"] = item.get("parent_asin")
            results.extend(extracted)
        return results

    def extract_aspects_opinions_batch(
        self,
        items: List[Dict],
        truncate_chars: int = 100,
    ) -> List[Dict]:
        """Extract aspects/opinions from a batch of reviews with one LLM call.

        Args:
            items: List of dicts with 'text' or 'review_text', 'product_id',
                   'product_name' (optional), 'rating' (optional).
            truncate_chars: Max characters per review in the batch prompt.

        Returns:
            Flat list of aspect-opinion records (same format as extract_aspects_opinions).
        """
        if not items:
            return []
        if not self._llm_available:
            results: List[Dict] = []
            for item in items:
                text = (item.get("text") or item.get("review_text") or "").strip()
                if not text or len(text) < 10:
                    continue
                results.extend(
                    self._extract_with_keywords(
                        text,
                        item.get("product_id", "unknown"),
                        item.get("product_name") or item.get("title"),
                        item.get("rating"),
                    )
                )
            return results
        try:
            return self._extract_batch_with_llm(items, truncate_chars=truncate_chars)
        except TimeoutError as e:
            print(f"Batch LLM timed out ({e}). Using keyword extraction for this batch.", flush=True)
            return self.extract_keyword_only_batch(items)
        except Exception as e:
            msg = "timed out" if "timeout" in str(e).lower() else str(e)
            print(f"Batch LLM failed ({msg}). Using keyword extraction for this batch.", flush=True)
            return self.extract_keyword_only_batch(items)

    def _extract_batch_with_llm(
        self,
        items: List[Dict],
        truncate_chars: int = 100,
    ) -> List[Dict]:
        """One Ollama call for many reviews; returns flat list of extraction records."""
        batch_size = len(items)
        print(f"  [LLM] Sending batch of {batch_size} reviews (timeout={self._request_timeout}s)...", flush=True)
        lines: List[str] = []
        for i, item in enumerate(items, start=1):
            text = (item.get("text") or item.get("review_text") or "").strip()
            snippet = text[:truncate_chars] + ("..." if len(text) > truncate_chars else "")
            rating = item.get("rating")
            line = f"Review {i}: \"{snippet}\""
            if rating is not None:
                line += f" (rating {rating}/5)"
            lines.append(line)

        system_prompt = """You are an expert at analyzing product reviews to extract structured opinions.

For EACH numbered review, identify:
1. ASPECTS: Product features discussed (camera, battery, screen, processor, design, sound, software, price, durability, connectivity, storage)
2. OPINIONS: The specific opinion for each aspect
3. SENTIMENT: positive, negative, or neutral
4. IMPLICIT: true if the aspect is implied (e.g. "dies quickly" -> battery)

Output ONLY a single JSON object. Keys are review numbers as strings "1", "2", ... "N". Value for each key is an array of aspect-opinion objects.
Format: {"1": [{"aspect": "category", "opinion": "brief text", "sentiment": "positive|negative|neutral", "is_implicit": false}], "2": [...], ...}
If a review has no aspects, use an empty array: "3": []
No other text, no markdown."""

        user_prompt = f"""Extract aspect-opinion pairs for each of these {batch_size} reviews.

{chr(10).join(lines)}

Return one JSON object with keys "1" to "{batch_size}" and arrays of {{"aspect", "opinion", "sentiment", "is_implicit"}}."""

        client = self._ollama_client if self._ollama_client is not None else ollama
        try:
            if self._ollama_client is not None:
                response = self._ollama_client.chat(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": 0.1, "num_predict": 8000},
                )
            else:
                response = ollama.chat(
                    model=self.chat_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": 0.1, "num_predict": 8000},
                )
        except Exception as e:
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise TimeoutError(f"Ollama request timed out after {self._request_timeout}s") from e
            raise
        content = (response.get("message") or {}).get("content") or ""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.lower().startswith("json"):
                content = content[4:].strip()
            content = content.strip()
        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx == -1 or end_idx <= start_idx:
            raise ValueError("No JSON object in batch response")
        data = json.loads(content[start_idx:end_idx])
        if not isinstance(data, dict):
            raise ValueError("Batch response is not a JSON object")

        results: List[Dict] = []
        for i, item in enumerate(items, start=1):
            extractions = data.get(str(i)) or data.get(i)
            if not isinstance(extractions, list):
                continue
            review_text = (item.get("text") or item.get("review_text") or "")[:500]
            product_id = item.get("product_id", "unknown")
            product_name = item.get("product_name") or item.get("title")
            rating = item.get("rating")
            for ext in extractions:
                if not isinstance(ext, dict):
                    continue
                aspect = (ext.get("aspect") or "").lower().strip()
                if not aspect or aspect not in PreprocessingConfig.ASPECT_CATEGORIES:
                    aspect = self._map_to_category(aspect)
                    if not aspect:
                        continue
                record = {
                    "product_id": product_id,
                    "aspect": aspect,
                    "opinion": (ext.get("opinion") or "")[:200],
                    "review_text": review_text,
                    "sentiment": self._normalize_sentiment(ext.get("sentiment", "neutral")),
                    "is_implicit": bool(ext.get("is_implicit", False)),
                }
                if item.get("asin"):
                    record["asin"] = item.get("asin")
                if item.get("parent_asin"):
                    record["parent_asin"] = item.get("parent_asin")
                if product_name:
                    record["product_name"] = product_name
                if rating is not None:
                    record["rating"] = rating
                results.append(record)
        print(f"  [LLM] Batch done: {len(results)} extractions.", flush=True)
        return results

    def _extract_with_llm(
        self, 
        review_text: str, 
        product_id: str, 
        product_name: Optional[str],
        rating: Optional[float]
    ) -> List[Dict]:
        """Extract aspects using LLM with Chain-of-Thought prompting.
        
        Implements the Syn-Chain (Syntax-Opinion-Sentiment Reasoning Chain)
        approach from the research paper for accurate aspect-based extraction.
        """
        system_prompt = """You are an expert at analyzing product reviews to extract structured opinions.

For each review, identify:
1. ASPECTS: What product features/attributes are discussed (e.g., camera, battery, screen, processor, design, sound, software, price, durability, connectivity, storage)
2. OPINIONS: The specific opinion expressed about each aspect
3. SENTIMENT: Whether the opinion is positive, negative, or neutral
4. IMPLICIT ASPECTS: Features implied but not explicitly named (e.g., "dies quickly" implies battery)

Use Chain-of-Thought reasoning:
- Step 1: Parse the syntax to identify opinion-target pairs
- Step 2: Map targets to aspect categories
- Step 3: Classify sentiment considering context and negations

Output ONLY valid JSON array with no additional text:
[{"aspect": "category", "opinion": "brief opinion text", "sentiment": "positive|negative|neutral", "is_implicit": false}]

If no aspects found, return empty array: []"""

        user_prompt = f"""Analyze this product review and extract all aspect-opinion pairs:

Review: "{review_text[:1000]}"
{f'Rating: {rating}/5' if rating else ''}

Extract aspects using Chain-of-Thought:
1. First identify what features are mentioned or implied
2. Then determine the opinion and sentiment for each

Return ONLY the JSON array:"""

        try:
            client = self._ollama_client if self._ollama_client is not None else ollama
            if self._ollama_client is not None:
                response = self._ollama_client.chat(
                    model=self.chat_model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    options={'temperature': 0.1}
                )
            else:
                response = ollama.chat(
                    model=self.chat_model,
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    options={'temperature': 0.1}
                )
            content = response['message']['content'].strip()
            
            # Parse JSON response - handle potential markdown code blocks
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            # Find JSON array in response
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx]
            
            extractions = json.loads(content)
            
            if not isinstance(extractions, list):
                return []
            
            # Convert to standard format
            results = []
            for ext in extractions:
                if not isinstance(ext, dict):
                    continue
                    
                aspect = ext.get('aspect', '').lower().strip()
                if not aspect or aspect not in PreprocessingConfig.ASPECT_CATEGORIES:
                    # Try to map to known category
                    aspect = self._map_to_category(aspect)
                    if not aspect:
                        continue
                
                record = {
                    "product_id": product_id,
                    "aspect": aspect,
                    "opinion": ext.get('opinion', '')[:200],
                    "review_text": review_text[:500],
                    "sentiment": self._normalize_sentiment(ext.get('sentiment', 'neutral')),
                    "is_implicit": ext.get('is_implicit', False),
                }
                if product_name:
                    record["product_name"] = product_name
                if rating:
                    record["rating"] = rating
                    
                results.append(record)
            
            return results
            
        except json.JSONDecodeError:
            return []
        except Exception as e:
            raise e

    def _extract_with_keywords(
        self, 
        review_text: str, 
        product_id: str, 
        product_name: Optional[str],
        rating: Optional[float]
    ) -> List[Dict]:
        """Fallback keyword-based extraction using lexicons and IAC mapping."""
        results: List[Dict] = []
        text_lower = review_text.lower()
        detected_aspects = set()

        # 1. Explicit aspect extraction using lexicons
        for lex in self.ASPECT_LEXICONS:
            if not self._contains_any(text_lower, lex.keywords):
                continue
            
            if lex.name in detected_aspects:
                continue
            detected_aspects.add(lex.name)

            sentiment = self._detect_sentiment(text_lower, rating)
            opinion = self._extract_opinion(review_text, lex.keywords)

            record = {
                "product_id": product_id,
                "aspect": lex.name,
                "opinion": opinion,
                "review_text": review_text[:500],
                "sentiment": sentiment,
                "is_implicit": False,
            }
            if product_name:
                record["product_name"] = product_name
            if rating:
                record["rating"] = rating

            results.append(record)

        # 2. Implicit aspect extraction using IAC mapping
        for clue, aspect in self.IMPLICIT_ASPECT_CLUES.items():
            if clue in text_lower and aspect not in detected_aspects:
                detected_aspects.add(aspect)
                
                # Determine sentiment from the clue itself
                sentiment = self._sentiment_from_clue(clue)
                
                record = {
                    "product_id": product_id,
                    "aspect": aspect,
                    "opinion": f"[Implicit: {clue}]",
                    "review_text": review_text[:500],
                    "sentiment": sentiment,
                    "is_implicit": True,
                }
                if product_name:
                    record["product_name"] = product_name
                if rating:
                    record["rating"] = rating

                results.append(record)

        return results

    def _map_to_category(self, aspect: str) -> Optional[str]:
        """Map a free-form aspect to a known category."""
        aspect_lower = aspect.lower()
        
        # Direct mapping
        category_keywords = {
            "camera": ["camera", "photo", "picture", "lens", "video"],
            "battery": ["battery", "power", "charge", "charging"],
            "screen": ["screen", "display", "display quality"],
            "processor": ["processor", "performance", "speed", "cpu", "chip"],
            "design": ["design", "build", "look", "appearance", "material"],
            "sound": ["sound", "speaker", "audio", "volume"],
            "software": ["software", "os", "interface", "app", "system"],
            "price": ["price", "value", "cost", "money"],
            "durability": ["durability", "durable", "waterproof", "rugged"],
            "connectivity": ["connectivity", "wifi", "bluetooth", "5g", "network"],
            "storage": ["storage", "memory", "ram", "space"],
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in aspect_lower for kw in keywords):
                return category
        
        return None

    @staticmethod
    def _contains_any(text_lower: str, keywords: Sequence[str]) -> bool:
        """Check if text contains any of the keywords."""
        return any(keyword in text_lower for keyword in keywords)

    @classmethod
    def _detect_sentiment(cls, text_lower: str, rating: Optional[float] = None) -> str:
        """Detect sentiment using keywords and optional rating."""
        pos_count = sum(1 for word in cls.POSITIVE_INDICATORS if word in text_lower)
        neg_count = sum(1 for word in cls.NEGATIVE_INDICATORS if word in text_lower)
        
        # Use rating as additional signal
        if rating is not None:
            if rating >= 4:
                pos_count += 2
            elif rating <= 2:
                neg_count += 2
        
        if pos_count > neg_count:
            return "positive"
        if neg_count > pos_count:
            return "negative"
        return "neutral"

    @staticmethod
    def _normalize_sentiment(sentiment: str) -> str:
        """Normalize sentiment to standard values."""
        sentiment = sentiment.lower().strip()
        if sentiment in ("positive", "pos", "good", "+"):
            return "positive"
        if sentiment in ("negative", "neg", "bad", "-"):
            return "negative"
        return "neutral"

    @staticmethod
    def _sentiment_from_clue(clue: str) -> str:
        """Determine sentiment from an implicit aspect clue."""
        positive_clues = [
            "lasts forever", "all day", "snappy", "buttery smooth", 
            "lightweight", "sleek", "premium feel", "gorgeous", "vibrant",
            "sharp photos", "crisp images", "loud", "clear audio"
        ]
        negative_clues = [
            "dies quickly", "drains fast", "sluggish", "freezes", "crashes",
            "heavy", "bulky", "cheap feel", "washed out", "eye strain",
            "blurry", "grainy", "tinny", "muffled"
        ]
        
        if clue in positive_clues:
            return "positive"
        if clue in negative_clues:
            return "negative"
        return "neutral"

    @staticmethod
    def _extract_opinion(text: str, keywords: Sequence[str]) -> str:
        """Extract a context window around the first matching keyword."""
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                return TextProcessor.extract_opinion_context(text, keyword, context_size=60)
        return "mentioned"

