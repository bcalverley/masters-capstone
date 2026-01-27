import os
from pathlib import Path
from dotenv import load_dotenv

# ==================================================
# Resolve project root (same folder as this file)
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

# ==================================================
# Explicitly load .env from project root
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
        f"SUPABASE_KEY not found in .env\n"
        f"Expected line:\nSUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImloYW13am1ieGNqem1wZXlhZWhkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDY3NTEwNzgsImV4cCI6MjA2MjMyNzA3OH0.YSt-yp0vQkemlYepXkL-90SC7V-C8fjHU0BK1A4m9FE"
    )

# ==================================================
# Application paths
# ==================================================
TRAINING_DIR = BASE_DIR / "trainingassets"
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_PATH = CAPTURE_DIR / "inventory_1.jpg"

# ==================================================
# Model configuration
# ==================================================
IMG_SIZE = (64, 64)
BATCH_SIZE = 12
EPOCHS = 25
CONFIDENCE_THRESHOLD = 0.70

# Pokémon card aspect ratio
CARD_ASPECT = 88 / 63
