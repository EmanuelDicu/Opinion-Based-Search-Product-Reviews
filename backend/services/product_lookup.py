"""Amazon product lookup for resolving product names by ASIN."""

from typing import Optional, Dict
import os
import requests


PHONE_BRAND_KEYWORDS = (
    "iphone", "samsung", "galaxy", "pixel", "google pixel", "oneplus",
    "xiaomi", "redmi", "huawei", "oppo", "vivo", "motorola", "moto",
    "nokia", "sony", "xperia", "lg", "asus", "realme", "infinix", "tecno",
)


def _looks_like_phone_name(name: str) -> bool:
    text = (name or "").lower()
    return any(kw in text for kw in PHONE_BRAND_KEYWORDS)


class AmazonProductLookup:
    """Fetch product name from an Amazon product API by ASIN/parent ASIN.

    Configure via env:
      - AMAZON_PRODUCT_API_URL: base URL (e.g. https://api.example.com/product)
      - AMAZON_PRODUCT_API_KEY: API key (optional)
      - AMAZON_PRODUCT_API_HOST: host header (optional, for RapidAPI)
    """

    def __init__(self) -> None:
        self.base_url = os.getenv("AMAZON_PRODUCT_API_URL", "").strip()
        self.api_key = os.getenv("AMAZON_PRODUCT_API_KEY", "").strip()
        self.api_host = os.getenv("AMAZON_PRODUCT_API_HOST", "").strip()

    def resolve_phone_name(self, asin: Optional[str], parent_asin: Optional[str]) -> Optional[str]:
        """Return product name if it looks like a phone; otherwise None."""
        if not self.base_url:
            return None

        for candidate in [asin, parent_asin]:
            if not candidate:
                continue
            name = self._fetch_name(candidate)
            if name and _looks_like_phone_name(name):
                return name
        return None

    def _fetch_name(self, asin: str) -> Optional[str]:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        if self.api_host:
            headers["X-API-Host"] = self.api_host

        try:
            # If URL includes {asin}, substitute directly; else pass asin as query param
            if "{asin}" in self.base_url:
                url = self.base_url.format(asin=asin)
                resp = requests.get(url, headers=headers, timeout=10)
            else:
                resp = requests.get(self.base_url, headers=headers, params={"asin": asin}, timeout=10)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return _extract_name(data)
        except Exception:
            return None


def _extract_name(data: Dict) -> Optional[str]:
    """Try common fields across Amazon product APIs."""
    if not data:
        return None

    # Common flat fields
    for key in ("title", "name", "product_title", "productTitle"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # PA API-like nested format
    try:
        item_info = data.get("ItemInfo") or data.get("itemInfo") or {}
        title = item_info.get("Title") or item_info.get("title") or {}
        display = title.get("DisplayValue") or title.get("displayValue")
        if isinstance(display, str) and display.strip():
            return display.strip()
    except Exception:
        pass

    # RapidAPI-like nested
    for key in ("product", "data", "result"):
        obj = data.get(key)
        if isinstance(obj, dict):
            for k in ("title", "name", "product_title"):
                val = obj.get(k)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    return None
