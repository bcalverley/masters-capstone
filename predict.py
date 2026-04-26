import json
import sys
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model


def _resolve(filename):
    """Return the correct path whether running live or frozen by PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / filename
    return Path(__file__).resolve().parent / filename


MODEL_PATH  = _resolve("card_model.keras")
LABELS_PATH = _resolve("class_labels.json")

_class_labels = None  # module-level cache


def _load_class_labels():
    if not LABELS_PATH.exists():
        raise RuntimeError(
            f"class_labels.json not found at {LABELS_PATH}.\n"
            "Run export_labels.py before packaging."
        )
    with open(LABELS_PATH) as f:
        class_indices = json.load(f)  # {class_name: int_index}
    return {v: k for k, v in class_indices.items()}  # {int_index: class_name}


def load_trained_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Train the model first.")
    return load_model(MODEL_PATH)


def predict_card(model, processed_image, top_k=3):
    """
    Returns:
        predicted_label   (str)
        confidence        (float)
        top_k_labels      (list[str])
        raw_probabilities (np.ndarray)
    """
    global _class_labels
    if _class_labels is None:
        _class_labels = _load_class_labels()

    result       = model.predict(processed_image, verbose=0)
    probabilities   = result[0]
    predicted_index = int(np.argmax(probabilities))
    confidence      = float(np.max(probabilities))

    predicted_label = _class_labels[predicted_index]

    top_indices  = np.argsort(probabilities)[-top_k:][::-1]
    top_k_labels = [_class_labels[i] for i in top_indices]

    return predicted_label, confidence, top_k_labels, probabilities
