"""Text preprocessing utilities."""

import re


class TextProcessor:
    """Utility functions for lightweight text cleanup and context extraction."""

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and normalize review text."""
        if not text:
            return ""

        text = re.sub(r"<[^>]+>", "", text)  # strip HTML
        text = re.sub(r"[^\w\s.,!?;:\-]", "", text)  # remove non-basic chars
        text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
        return text

    @staticmethod
    def extract_opinion_context(text: str, keyword: str, context_size: int = 50) -> str:
        """Return a small window of text around the keyword, if present."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()

        idx = text_lower.find(keyword_lower)
        if idx == -1:
            return "mentioned"

        start = max(0, idx - context_size)
        end = min(len(text), idx + context_size)
        return text[start:end].strip()

