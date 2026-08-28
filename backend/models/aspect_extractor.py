"""Aspect extraction utilities."""

from typing import Dict, Optional

class AspectExtractor:
    """Extract aspects from user queries."""
    
    ASPECT_KEYWORDS = {
        'camera': ['camera', 'photo', 'picture', 'image', 'lens', 'zoom', 'selfie'],
        'battery': ['battery', 'charge', 'charging', 'power', 'endurance'],
        'screen': ['screen', 'display', 'resolution', 'brightness', 'clarity'],
        'processor': ['processor', 'cpu', 'chip', 'performance', 'speed'],
        'design': ['design', 'look', 'appearance', 'build', 'material'],
        'sound': ['sound', 'speaker', 'audio', 'volume'],
        'software': ['software', 'os', 'interface', 'ui', 'app'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value'],
        # High-level inferred features (Level 2 concepts)
        'gaming_experience': ['gaming', 'game', 'games', 'gamer', 'play games', 'fps', 'frame rate'],
        'cinematic_viewing': ['movie', 'movies', 'video', 'videos', 'cinema', 'watch films'],
        'field_utility': ['hiking', 'field work', 'outdoors', 'rugged', 'durable'],
        'outdoor_navigation': ['navigation', 'maps', 'outdoor navigation', 'sunlight', 'bright screen']
    }
    
    @classmethod
    def extract_aspect(cls, query: str) -> Optional[str]:
        """Extract aspect from user query."""
        query_lower = query.lower()
        for aspect, keywords in cls.ASPECT_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                return aspect
        return None
    
    @classmethod
    def get_all_aspects(cls) -> list:
        """Get all available aspects."""
        return list(cls.ASPECT_KEYWORDS.keys())

