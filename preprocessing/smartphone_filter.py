"""Filter reviews to keep only those about smartphones (not accessories).

Uses review title + start of review text with fast keyword heuristics (no LLM).
The dataset has no product title per review—only review title and text—so we
detect accessory reviews from content (e.g. "case", "charger", "screen protector").
"""

from typing import List, Dict

# Phrases/words that indicate the REVIEW is about an ACCESSORY (reject)
# Checked in (review title + first 400 chars of review text), lowercased
ACCESSORY_CONTENT_PHRASES = (
    " case ", " case.", " case,", " case!", " case for", "phone case", "cases ",
    " cover ", " cover.", " cover for", "phone cover", "covers ",
    " charger", "charger ", " charging cable", " charging cord", " cable ", "cable ",
    " usb cable", " usb ", " cable for", "cord ", " adapter ",
    " screen protector", "screen protector", "tempered glass", " glass protector",
    " phone holder", " holder ", " car mount", " desk stand", " stand ",
    " mount ", " dock ", " cradle ",
    " earbuds", " earphones", " earphone", " headphones", " headphone",
    " stylus", " battery pack", " power bank", " charging station",
    " wallet case", " pouch ", " holster", " strap ", " lanyard",
    " ring holder", " phone ring", " kickstand", " grip ",
    " armband", " belt clip", " carrying case", " bluetooth speaker",
    " fitbit cable", " lost the cable", " install ",
    " screen guard", " screen film", " glass ", " tempered ", " protector ",
    " magsafe", " mag safe", " magnetic mount", " phone grip", " popsocket", " pop socket",
    " otterbox", " spigen", " ringke", " uag ", " caseology", " esr ",
    " bumper", " shell", " sleeve", " folio", " flip case", " wallet ",
    " wrist strap", " lanyard", " camera lens protector", " lens protector",
    " protector for", " case for", " cover for", " charger for", " cable for",
    " fits my iphone", " fits my galaxy", " fits iphone", " fits galaxy",
    " for my iphone", " for my galaxy", " for iphone ", " for galaxy ",
    " for samsung ", " for pixel ",
    # Non-phone gadgets to exclude
    " tablet ", " ipad ", " kindle ", " fire tablet", " galaxy tab",
    " smartwatch", " smart watch", " watch band", " watch strap", " apple watch",
    " fitbit", " garmin", " fitness tracker", " activity tracker",
    " drone ", " quadcopter", " action camera", " gopro",
    " vr headset", " game controller", " controller ",
    " keyboard ", " mouse ", " selfie stick", " tripod", " microphone", " mic ",
    " replacement battery", " battery replacement", " screen replacement",
    " nintendo switch", " switch ", " airtag", " air tag", " stylus ", " pen ",
)

# Require a phone brand mention in review title/text
PHONE_BRAND_KEYWORDS = (
    "iphone", "samsung", "galaxy", "pixel", "google pixel", "oneplus",
    "xiaomi", "redmi", "huawei", "oppo", "vivo", "motorola", "moto",
    "nokia", "sony", "xperia", "lg", "asus", "realme", "infinix", "tecno",
)

def _get_content_snippet(item: Dict, max_text_len: int = 400) -> str:
    """Review title + start of review text for classification."""
    title = (item.get("product_name") or item.get("title") or "").strip()
    text = (item.get("text") or item.get("review_text") or "").strip()
    if len(text) > max_text_len:
        text = text[:max_text_len] + " "
    return (title + " " + text).lower()


def _content_looks_accessory(snippet: str) -> bool:
    """True if the snippet clearly describes an accessory product."""
    return any(phrase in snippet for phrase in ACCESSORY_CONTENT_PHRASES)


def _mentions_phone_brand(snippet: str) -> bool:
    """True if snippet mentions a known phone brand/model keyword."""
    return any(kw in snippet for kw in PHONE_BRAND_KEYWORDS)


def is_smartphone_review(item: Dict) -> bool:
    """Return True if this review appears to be about a smartphone (not an accessory).

    Uses review title + start of review text. Fast, no LLM.
    - Reject if content contains accessory phrases (case, charger, screen protector, etc.).
    - Keep otherwise (so we keep phone reviews and ambiguous ones).
    """
    snippet = _get_content_snippet(item)
    if not snippet.strip():
        return False
    if not _mentions_phone_brand(snippet):
        return False
    return not _content_looks_accessory(snippet)


def filter_smartphone_reviews(
    items: List[Dict],
    batch_size: int = 0,
    model: str = "",
    use_llm: bool = False,
    verbose: bool = True,
) -> List[Dict]:
    """Keep only reviews that appear to be about smartphones (not accessories).

    Uses a fast content-based heuristic (review title + text snippet). No Ollama.
    batch_size, model, and use_llm are ignored.

    Args:
        items: List of dicts with 'title'/'product_name' and 'text'/'review_text'.
        batch_size: Ignored.
        model: Ignored.
        use_llm: Ignored.
        verbose: Print stats.

    Returns:
        Subset of items classified as smartphone-related.
    """
    if not items:
        return []
    filtered = [item for item in items if is_smartphone_review(item)]
    if verbose:
        print(f"✓ Smartphone filter (content-based): kept {len(filtered)} / {len(items)} reviews.")
    return filtered
