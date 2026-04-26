import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ==================================================
# Resolve directories — handles both live and frozen
# (PyInstaller sets sys.frozen and sys._MEIPASS)
# ==================================================
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# Explicitly load .env from the base directory
# ==================================================
ENV_PATH = BASE_DIR / ".env"

if not ENV_PATH.exists():
    raise RuntimeError(f".env file not found at: {ENV_PATH}")

load_dotenv(dotenv_path=ENV_PATH)

# ==================================================
# Supabase configuration
# ==================================================
SUPABASE_URL = "https://ihamwjmbxcjzmpeyaehd.supabase.co"
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_KEY:
    raise RuntimeError(
        f"SUPABASE_KEY not found in .env at {ENV_PATH}\n"
        f"Add a line: SUPABASE_KEY=<your_key>"
    )

# ==================================================
# Target sets — the 5 sets the model is trained on.
# Maps internal set code → Pokémon TCG API set id.
# ==================================================
TARGET_SETS = {
    "JTG": "sv9",       # Journey Together  (2025)
    "PRE": "sv8pt5",    # Prismatic Evolutions (2025)
    "SCR": "sv7",       # Stellar Crown (2024)
    "SFA": "sv6pt5",    # Shrouded Fable (2024)
    "SSP": "sv8",       # Surging Sparks (2024)
}

# ==================================================
# Application paths
# ==================================================
TRAINING_DIR = BASE_DIR / "trainingassets"
CAPTURE_DIR  = BASE_DIR / "captures"
CAPTURE_PATH = CAPTURE_DIR / "inventory_1.jpg"

CAPTURE_DIR.mkdir(exist_ok=True)

# ==================================================
# Model configuration
# ==================================================
# 96×96 is the MobileNetV2 minimum and gives enough
# resolution to distinguish fine card details.
# NOTE: changing this requires retraining the model.
IMG_SIZE             = (96, 96)
BATCH_SIZE           = 16
EPOCHS               = 25
CONFIDENCE_THRESHOLD = 0.70

# Pokémon card aspect ratio
CARD_ASPECT = 88 / 63
