import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

MODEL_PATH = Path("card_model.keras")


def load_trained_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Train the model first.")
    return load_model(MODEL_PATH)


def predict_card(model, training_set, processed_image, top_k=3):
    """
    Returns:
        predicted_label (str)
        confidence (float)
        top_k_labels (list)
        raw_probabilities (np.array)
    """

    result = model.predict(processed_image, verbose=0)

    probabilities = result[0]
    predicted_index = np.argmax(probabilities)
    confidence = float(np.max(probabilities))

    class_labels = {v: k for k, v in training_set.class_indices.items()}

    predicted_label = class_labels[predicted_index]

    # Top-k predictions
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_labels = [class_labels[i] for i in top_indices]

    return predicted_label, confidence, top_k_labels, probabilities