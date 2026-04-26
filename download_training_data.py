"""
Downloads official card images from the Pokémon TCG API for the 5 target sets,
then generates 15 webcam-simulating augmented training images per card.

Run once before training (internet connection required):
    python download_training_data.py

Safe to re-run — folders that already have enough images are skipped.
"""

import io
import random
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter

from config import TRAINING_DIR, TARGET_SETS

IMAGES_PER_CARD = 15
API_BASE = "https://api.pokemontcg.io/v2"


# ── Card number normalisation ──────────────────────────────────────────────────

def normalise_number(raw):
    """
    API returns zero-padded numbers like '001'. The evaluation set and Supabase
    use un-padded integers like '1'. Alphanumeric numbers (TG01, SWSH001) are
    kept as-is since they can't be meaningfully cast to int.
    """
    try:
        return str(int(raw))
    except ValueError:
        return raw


# ── API helpers ────────────────────────────────────────────────────────────────

def fetch_cards_for_set(api_id):
    cards = []
    page = 1
    while True:
        resp = requests.get(
            f"{API_BASE}/cards",
            params={"q": f"set.id:{api_id}", "pageSize": 250, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("data", [])
        cards.extend(batch)
        if len(cards) >= data.get("totalCount", 0):
            break
        page += 1
        time.sleep(0.2)
    return cards


def download_image(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ── Augmentation pipeline ──────────────────────────────────────────────────────

def random_perspective_warp(arr, max_distortion=0.08):
    h, w = arr.shape[:2]
    d = int(min(w, h) * max_distortion)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), h - random.randint(0, d)],
        [random.randint(0, d), h - random.randint(0, d)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(arr, M, (w, h))


def augment_for_webcam(source_pil):
    """Return one augmented PIL image simulating a real webcam capture."""
    img = ImageEnhance.Brightness(source_pil).enhance(random.uniform(0.55, 1.45))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.30))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.80, 1.20))

    if random.random() > 0.40:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.8)))

    arr = np.array(img)
    arr = random_perspective_warp(arr, max_distortion=0.08)

    noise = np.random.normal(0, random.uniform(2, 10), arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# ── Per-set download ───────────────────────────────────────────────────────────

def process_set(set_code, api_id):
    print(f"\n── {set_code}  (API id: {api_id}) " + "─" * 40)
    try:
        cards = fetch_cards_for_set(api_id)
    except Exception as e:
        print(f"   ERROR fetching card list: {e}")
        return

    print(f"   {len(cards)} cards found")
    downloaded = skipped = errors = 0

    for card in cards:
        number = normalise_number(card["number"])
        label = f"{set_code} {number}"
        folder = TRAINING_DIR / label

        existing = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if len(existing) >= IMAGES_PER_CARD:
            skipped += 1
            continue

        img_url = card["images"].get("large") or card["images"].get("small")
        if not img_url:
            print(f"   WARNING: no image URL for {label}")
            errors += 1
            continue

        try:
            folder.mkdir(parents=True, exist_ok=True)
            source = download_image(img_url)
            time.sleep(0.15)

            needed = IMAGES_PER_CARD - len(existing)
            for i in range(needed):
                aug = augment_for_webcam(source)
                aug.save(folder / f"aug_{i:03d}.jpg", quality=92)

            downloaded += 1
            print(f"   + {label}")

        except Exception as e:
            print(f"   ! {label}: {e}")
            errors += 1

    print(f"   Done — {downloaded} downloaded, {skipped} already complete, {errors} errors")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    total_cards = sum(
        len(list((TRAINING_DIR / f"{s} *").parent.glob(f"{s} *")))
        if (TRAINING_DIR).exists() else 0
        for s in TARGET_SETS
    )

    print("Pokémon TCG Training Data Downloader")
    print(f"Target sets : {', '.join(TARGET_SETS.keys())}")
    print(f"Images/card : {IMAGES_PER_CARD}")
    print(f"Output      : {TRAINING_DIR}")

    for set_code, api_id in TARGET_SETS.items():
        process_set(set_code, api_id)

    print("\nAll sets processed.")
    print("Next step: python train.py")


if __name__ == "__main__":
    main()
