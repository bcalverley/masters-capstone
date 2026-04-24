import os
import random
import shutil
from pathlib import Path

from train import get_training_set

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "trainingassets"
EVAL_DIR = BASE_DIR / "evaluation_set"

IMAGES_PER_CLASS = 1  # change to 2 or 3 if desired

# Clear old evaluation folder if it exists
if EVAL_DIR.exists():
    shutil.rmtree(EVAL_DIR)

EVAL_DIR.mkdir()

print("\nGenerating evaluation set from training assets...\n")

training_set = get_training_set()
class_labels = list(training_set.class_indices.keys())

total_images = 0

for label in class_labels:
    source_folder = TRAIN_DIR / label
    dest_folder = EVAL_DIR / label
    dest_folder.mkdir(parents=True, exist_ok=True)

    images = [f for f in os.listdir(source_folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        continue

    selected = random.sample(
        images,
        min(IMAGES_PER_CLASS, len(images))
    )

    for img in selected:
        shutil.copy(
            source_folder / img,
            dest_folder / img
        )
        total_images += 1

print("Done.")
print(f"Total images copied: {total_images}")
print(f"Evaluation folder created at:\n{EVAL_DIR}")