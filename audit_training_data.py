"""
Audits trainingassets/ and reports coverage for the 5 target sets.
Run before and after download_training_data.py to confirm readiness.

Usage:
    python audit_training_data.py
"""

from collections import defaultdict
from config import TRAINING_DIR, TARGET_SETS

MIN_IMAGES = 5  # cards below this are flagged


def audit():
    if not TRAINING_DIR.exists():
        print(f"Training directory not found: {TRAINING_DIR}")
        print("Run download_training_data.py first.")
        return

    target_set_codes = set(TARGET_SETS.keys())
    set_stats = defaultdict(lambda: {"cards": 0, "images": 0, "low": []})
    other_sets = set()

    for card_dir in sorted(TRAINING_DIR.iterdir()):
        if not card_dir.is_dir():
            continue
        parts = card_dir.name.split()
        if len(parts) < 2:
            continue

        set_code = parts[0]
        images = [
            f for f in card_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        n = len(images)

        if set_code not in target_set_codes:
            other_sets.add(set_code)
            continue

        set_stats[set_code]["cards"] += 1
        set_stats[set_code]["images"] += n
        if n < MIN_IMAGES:
            set_stats[set_code]["low"].append((card_dir.name, n))

    print("\n══ Training Data Audit ════════════════════════════════")
    total_cards = total_images = 0

    for code in sorted(target_set_codes):
        s = set_stats[code]
        if s["cards"] == 0:
            print(f"  ✗  {code}: NOT FOUND — run download_training_data.py")
            continue

        avg = s["images"] / s["cards"]
        flag = "⚠" if avg < MIN_IMAGES else "✓"
        print(f"  {flag}  {code}: {s['cards']} cards, {s['images']} images  ({avg:.1f} avg/card)")

        if s["low"]:
            sample = ", ".join(f"{n}×{name}" for name, n in s["low"][:5])
            tail = f"  +{len(s['low'])-5} more" if len(s["low"]) > 5 else ""
            print(f"       Low-image cards: {sample}{tail}")

        total_cards  += s["cards"]
        total_images += s["images"]

    print(f"\n  Total: {total_cards} cards, {total_images} images across target sets")

    if other_sets:
        print(f"\n  Other sets present (not targeted): {', '.join(sorted(other_sets))}")

    print("═══════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    audit()
