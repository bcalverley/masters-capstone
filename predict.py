import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

MODEL_PATH = Path("card_model.keras")

def load_trained_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("Trained model not found. Train the model first.")
    return load_model(MODEL_PATH)

def predict_card(model, training_set, processed_image):
    """
    Returns:
        predicted_label (str)
        confidence (float between 0 and 1)
    """
    result = model.predict(processed_image)

    predicted_index = np.argmax(result)
    confidence = float(np.max(result))

    class_labels = {v: k for k, v in training_set.class_indices.items()}
    predicted_label = class_labels[predicted_index]

    return predicted_label, confidence

