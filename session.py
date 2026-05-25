import csv
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SessionItem:
    set_code: str
    card_number: str
    card_name: str
    variant: str           # "Normal" | "Holo" | "Reverse Holo"
    grade: str             # "Ungraded" | "PSA 10" | ...
    market_price: Optional[float]
    predicted_label: str
    confidence: float


class BuyingSession:
    def __init__(self, offer_tiers=(80, 75, 60)):
        self.items: list[SessionItem] = []
        self.offer_tiers = list(offer_tiers)

    def add_item(self, item: SessionItem) -> None:
        self.items.append(item)

    def remove_item(self, index: int) -> None:
        if 0 <= index < len(self.items):
            del self.items[index]

    def clear(self) -> None:
        self.items.clear()

    def market_total(self) -> float:
        return round(sum(i.market_price or 0.0 for i in self.items), 2)

    def offer_amount(self, pct: int) -> float:
        return round(self.market_total() * pct / 100, 2)

    def export_receipt(self, filepath: str) -> None:
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["#", "Set", "Card #", "Name", "Variant", "Grade", "Market Price"])
            for i, item in enumerate(self.items, 1):
                price_str = f"${item.market_price:.2f}" if item.market_price else "N/A"
                writer.writerow([
                    i, item.set_code, item.card_number, item.card_name,
                    item.variant, item.grade, price_str,
                ])
            writer.writerow([])
            total = self.market_total()
            writer.writerow(["", "", "", "", "", "Market Total", f"${total:.2f}"])
            for pct in self.offer_tiers:
                writer.writerow([
                    "", "", "", "", "", f"Offer ({pct}%)",
                    f"${self.offer_amount(pct):.2f}",
                ])
            writer.writerow([])
            writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
