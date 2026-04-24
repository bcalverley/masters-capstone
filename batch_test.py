import os
import csv
from datetime import datetime
from keras.preprocessing import image
import numpy as np
from config import IMG_SIZE, CONFIDENCE_THRESHOLD
from predict import predict_card


def preprocess_image(filepath):
    img = image.load_img(filepath, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def run_batch_test(model, training_set, folder_path):

    results = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                filepath = os.path.join(root, file)
                true_label = os.path.basename(root)

                processed = preprocess_image(filepath)

                prediction, confidence, top_k_labels, _ = predict_card(
                    model,
                    training_set,
                    processed
                )

                # -----------------------------
                # Statistical evaluation logic
                # -----------------------------

                if confidence < CONFIDENCE_THRESHOLD:
                    accepted = 0
                    correct = 0
                    top3_correct = 0
                else:
                    accepted = 1
                    correct = 1 if prediction == true_label else 0
                    top3_correct = 1 if true_label in top_k_labels else 0

                results.append([
                    file,
                    true_label,
                    prediction,
                    confidence,
                    accepted,
                    correct,
                    top3_correct
                ])

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"batch_results_{timestamp}.csv"

    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "true_label",
            "predicted_label",
            "confidence",
            "accepted",
            "top1_correct",
            "top3_correct"
        ])
        writer.writerows(results)

    return output_file, len(results)