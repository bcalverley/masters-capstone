"""
Run this once (from the project root, with trainingassets/ present) before
building the packaged application.  It writes class_labels.json which the
packaged app loads instead of scanning trainingassets/ at startup.

Usage:
    python export_labels.py
"""
import json
from train import get_training_set

training_set = get_training_set()

with open("class_labels.json", "w") as f:
    json.dump(training_set.class_indices, f, indent=2, sort_keys=True)

print(f"Exported {len(training_set.class_indices)} class labels to class_labels.json")
