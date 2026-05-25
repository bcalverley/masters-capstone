# Pulls live market prices from the Pokemon TCG API and applies grade multipliers.

import requests
from config import TARGET_SETS

API_BASE = "https://api.pokemontcg.io/v2"

# Approximate market multipliers vs raw ungraded price.
# These are reasonable industry averages — adjust to suit buying preferences.
GRADE_MULTIPLIERS = {
    "Ungraded": 1.0,
    "PSA 10":   4.0,
    "PSA 9":    2.0,
    "PSA 8":    1.3,
    "PSA 7":    1.0,
    "BGS 9.5":  5.0,
    "CGC 10":   3.5,
    "CGC 9":    1.8,
}

VARIANT_MAP = {
    "Normal":       "normal",
    "Holo":         "holofoil",
    "Reverse Holo": "reverseHolofoil",
}

_price_cache: dict = {}  # cache results so we don't hit the API twice for the same card


def fetch_card_prices(set_code: str, card_number: str) -> dict:
    # returns the TCGPlayer prices dict for a card, or {} if unavailable
    label = f"{set_code} {card_number}"
    if label in _price_cache:
        return _price_cache[label]

    api_id = TARGET_SETS.get(set_code)
    if not api_id:
        _price_cache[label] = {}
        return {}

    try:
        resp = requests.get(
            f"{API_BASE}/cards",
            params={"q": f"set.id:{api_id} number:{card_number}", "pageSize": 1},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            _price_cache[label] = {}
            return {}
        prices = data[0].get("tcgplayer", {}).get("prices", {})
        _price_cache[label] = prices
        return prices
    except Exception:
        return {}


def get_price(
    set_code: str,
    card_number: str,
    variant: str = "Normal",
    grade: str = "Ungraded",
) -> float | None:
    # returns the adjusted market price for the given variant and grade, or None if unavailable
    # falls back to normal pricing if the specific variant has no data
    prices = fetch_card_prices(set_code, card_number)
    api_variant = VARIANT_MAP.get(variant, "normal")

    variant_data = prices.get(api_variant) or prices.get("normal") or {}
    market = variant_data.get("market") or variant_data.get("mid")

    if market is None:
        return None

    multiplier = GRADE_MULTIPLIERS.get(grade, 1.0)
    return round(market * multiplier, 2)
