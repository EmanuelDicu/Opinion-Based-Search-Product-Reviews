"""Weighted feature inference from aspect-opinion extractions.

Simplified neuro-symbolic inference using asymmetric weights:
- Each aspect has separate weights for positive vs negative sentiment
- Allows modeling that e.g. "processor lag" is more damaging to gaming 
  than "good processor" is beneficial

Weight interpretation:
  aspects: { "processor": (pos_weight=1.5, neg_weight=2.0) }
  - Positive processor sentiment contributes 1.5x to positive score
  - Negative processor sentiment contributes 2.0x to negative score
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from preprocessing.config import PreprocessingConfig


@dataclass
class AspectStats:
    """Aggregated sentiment counts for an aspect."""
    positive: int = 0
    negative: int = 0
    neutral: int = 0

    def total(self) -> int:
        return self.positive + self.negative + self.neutral
    
    def net_score(self) -> float:
        """Normalized score: +1 (all positive) to -1 (all negative)."""
        total = self.total()
        if total == 0:
            return 0.0
        return (self.positive - self.negative) / total


class FeatureInferenceEngine:
    """Infers high-level features using weighted aspect aggregation.
    
    For each feature, computes weighted positive and negative scores
    from constituent aspects, then determines overall sentiment.
    """

    def __init__(self) -> None:
        self.features = PreprocessingConfig.HIGH_LEVEL_FEATURES
        print(f"✓ Feature inference engine initialized ({len(self.features)} features)")

    def infer_features(self, extractions: List[Dict]) -> List[Dict]:
        """Augment extractions with inferred high-level features."""
        if not extractions:
            return extractions

        # Build product profiles: {product_id: {aspect: AspectStats}}
        profiles = self._build_profiles(extractions)
        print(f"Built profiles for {len(profiles)} products from {len(extractions)} extractions")
        
        # Infer features for each product
        inferred: List[Dict] = []
        for product_id, profile in profiles.items():
            for feature_name, feature_config in self.features.items():
                record = self._infer_single(product_id, profile, feature_name, feature_config)
                if record:
                    inferred.append(record)

        # Apply global cap and confidence filtering
        max_inferred = getattr(PreprocessingConfig, "MAX_INFERRED_FEATURES", 1000)
        if max_inferred and len(inferred) > max_inferred:
            inferred = sorted(inferred, key=lambda r: r.get("confidence", 0), reverse=True)[:max_inferred]

        print(f"Inferred {len(inferred)} high-level feature records")
        return extractions + inferred

    def _build_profiles(self, extractions: List[Dict]) -> Dict[str, Dict]:
        """Aggregate aspect sentiments per product."""
        profiles: Dict[str, Dict] = {}
        
        for item in extractions:
            pid = item.get("product_id", "unknown")
            aspect = str(item.get("aspect", "")).lower().strip()
            sentiment = str(item.get("sentiment", "")).lower().strip()
            
            if not aspect or sentiment not in ("positive", "negative", "neutral"):
                continue
                
            if pid not in profiles:
                profiles[pid] = {
                    "product_name": item.get("product_name", "Unknown"),
                    "aspects": defaultdict(AspectStats),
                }
            
            stats = profiles[pid]["aspects"][aspect]
            setattr(stats, sentiment, getattr(stats, sentiment) + 1)
            
        return profiles

    def _infer_single(
        self, 
        product_id: str, 
        profile: Dict, 
        feature_name: str, 
        feature_config: Dict
    ) -> Dict | None:
        """Infer sentiment for a single feature using weighted scoring."""
        
        aspect_weights = feature_config.get("aspects", {})
        if not aspect_weights:
            return None
            
        product_aspects = profile["aspects"]
        
        # Compute weighted scores
        pos_score = 0.0
        neg_score = 0.0
        total_weight = 0.0
        contributing_aspects = []
        
        for aspect, (pos_w, neg_w) in aspect_weights.items():
            stats = product_aspects.get(aspect)
            if not stats or stats.total() == 0:
                continue
                
            # Normalize counts to [0, 1] range
            total = stats.total()
            pos_ratio = stats.positive / total
            neg_ratio = stats.negative / total
            
            # Apply asymmetric weights
            pos_score += pos_ratio * pos_w
            neg_score += neg_ratio * neg_w
            total_weight += max(pos_w, neg_w)  # Normalize by max possible
            
            contributing_aspects.append((aspect, stats))
        
        # Need at least one contributing aspect
        if not contributing_aspects:
            return None

        min_aspects = getattr(PreprocessingConfig, "MIN_INFERRED_ASPECTS", 2)
        if min_aspects and len(contributing_aspects) < min_aspects:
            return None
            
        # Determine sentiment from weighted scores
        if total_weight > 0:
            pos_score /= total_weight
            neg_score /= total_weight
            
        sentiment, confidence = self._compute_sentiment(pos_score, neg_score)
        min_conf = getattr(PreprocessingConfig, "MIN_INFERRED_CONFIDENCE", 0.6)
        if min_conf and confidence < min_conf:
            return None
        
        # Build record
        return {
            "product_id": product_id,
            "product_name": profile["product_name"],
            "aspect": feature_name,
            "opinion": self._build_opinion(feature_config, contributing_aspects, sentiment),
            "sentiment": sentiment,
            "confidence": confidence,
            "is_inferred": True,
            "feature_type": feature_name,
            "source_aspects": list(aspect_weights.keys()),
        }

    def _compute_sentiment(self, pos_score: float, neg_score: float) -> Tuple[str, float]:
        """Determine sentiment and confidence from weighted scores."""
        diff = pos_score - neg_score
        
        # Thresholds for classification
        if diff > 0.15:
            sentiment = "positive"
            confidence = min(1.0, 0.5 + diff)
        elif diff < -0.15:
            sentiment = "negative"
            confidence = min(1.0, 0.5 - diff)
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(diff) * 2  # Higher confidence when truly balanced
            
        return sentiment, round(confidence, 2)

    def _build_opinion(
        self, 
        feature_config: Dict, 
        contributing_aspects: List[Tuple[str, AspectStats]], 
        sentiment: str
    ) -> str:
        """Build descriptive opinion string."""
        description = feature_config.get("description", "")
        
        aspect_summary = ", ".join(
            f"{asp}({s.positive}+/{s.negative}-)" 
            for asp, s in contributing_aspects
        )
        
        return f"{description} [{sentiment.upper()}] Based on: {aspect_summary}"


def infer_high_level_features(extractions: List[Dict]) -> List[Dict]:
    """Convenience function to run feature inference."""
    return FeatureInferenceEngine().infer_features(extractions)


