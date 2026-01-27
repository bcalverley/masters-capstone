import numpy as np


def predict_card(model, training_set, captured_image):
    result = model.predict(captured_image)
    predicted_index = np.argmax(result)

    class_labels = {v: k for k, v in training_set.class_indices.items()}
    return class_labels[predicted_index]
