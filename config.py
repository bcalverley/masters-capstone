import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# works whether running as a normal script or a packaged .exe
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f"WARNING: .env not found at {ENV_PATH} — Supabase lookups will be unavailable.")

SUPABASE_URL = "https://ihamwjmbxcjzmpeyaehd.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # loaded from .env, database.py handles missing key

# the 5 sets the model is trained on — maps set code to Pokemon TCG API set id
TARGET_SETS = {
    "JTG": "sv9",       # Journey Together (2025)
    "PRE": "sv8pt5",    # Prismatic Evolutions (2025)
    "SCR": "sv7",       # Stellar Crown (2024)
    "SFA": "sv6pt5",    # Shrouded Fable (2024)
    "SSP": "sv8",       # Surging Sparks (2024)
}

TRAINING_DIR = BASE_DIR / "trainingassets"
CAPTURE_DIR  = BASE_DIR / "captures"
CAPTURE_PATH = CAPTURE_DIR / "inventory_1.jpg"

CAPTURE_DIR.mkdir(exist_ok=True)

# 160x160 gives enough detail to distinguish card artwork
# changing this requires retraining the model
IMG_SIZE             = (160, 160)
BATCH_SIZE           = 16
EPOCHS               = 25
CONFIDENCE_THRESHOLD = 0.70

CARD_ASPECT = 88 / 63  # standard Pokemon card aspect ratio

# offer tiers as a percentage of market value
OFFER_TIERS = [80, 75, 60]

VARIANTS = ["Normal", "Holo", "Reverse Holo"]

GRADES = [
    "Ungraded",
    "PSA 10", "PSA 9", "PSA 8", "PSA 7",
    "BGS 9.5",
    "CGC 10", "CGC 9",
]
