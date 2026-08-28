"""LLM-based summarization of search results using OpenAI."""

from typing import List, Dict
import os

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from backend.config import Config


def _load_openai_key() -> str:
    """Load OpenAI API key from env or mounted file."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    key_path = os.getenv("OPENAI_API_KEY_PATH", "/app/api_key.txt")
    try:
        with open(key_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


class LLMService:
    """Summarization via OpenAI (default: gpt-4o-mini)."""

    def __init__(self) -> None:
        self.model = getattr(Config, "OPENAI_MODEL", "gpt-4o-mini")
        self.max_reviews = Config.LLM_SUMMARY_MAX_REVIEWS
        self._api_key = _load_openai_key()
        self.enabled = OPENAI_AVAILABLE and bool(self._api_key)
        self._client = OpenAI(api_key=self._api_key) if self.enabled else None

        if self.enabled:
            print(f"✓ OpenAI summarization enabled: {self.model} (max {self.max_reviews} reviews)")
        else:
            print("⚠ OpenAI summarization disabled (missing openai package or API key)")

    def summarize(self, query: str, results: List[Dict]) -> str:
        """Return a natural language answer from the list of related reviews."""
        if not self.enabled or not results:
            return ""

        capped = results[: self.max_reviews]
        max_review_chars = 300

        lines = []
        for i, r in enumerate(capped, start=1):
            product = r.get("product_name", r.get("model", "Unknown"))
            sentiment = r.get("sentiment_display", r.get("sentiment", "neutral"))
            if sentiment == "neutral":
                sentiment = "mixed"
            text = (r.get("review_text", "") or "").strip()
            if len(text) > max_review_chars:
                text = text[: max_review_chars] + "..."
            lines.append(f"Review {i}: {product} — {sentiment}")
            lines.append(text)
            lines.append("")

        context_text = "\n".join(lines).strip() or "No reviews found."

        system_prompt = (
            "You are an assistant that helps users choose smartphones based on real user reviews. "
            "You will be given the user's question and a list of related reviews (product name, sentiment, and review text). "
            "Write a short, helpful summary (2-4 sentences) that directly answers or suggests an answer to the user's question. "
            "Name specific products where relevant. Do not repeat the review text; synthesize and suggest."
        )
        user_prompt = (
            f"User question: {query}\n\n"
            f"Related reviews:\n{context_text}\n\n"
            "Write a short summary (2-4 sentences) that suggests an answer to the user's question based on these reviews. "
            "Name specific products where relevant."
        )

        max_context_chars = 14000
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars] + "\n...[truncated]"
            user_prompt = (
                f"User question: {query}\n\nRelated reviews:\n{context_text}\n\n"
                "Write a short summary (2-4 sentences) that suggests an answer to the user's question. "
                "Name specific products where relevant."
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"OpenAI summarization error: {exc}")
            return ""
