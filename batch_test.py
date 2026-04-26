import os
import csv
from datetime import datetime
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import IMG_SIZE, CONFIDENCE_THRESHOLD
from predict import predict_card, load_trained_model
from database import lookup_card


def preprocess_image(filepath):
    img = image.load_img(filepath, target_size=IMG_SIZE)
    arr = image.img_to_array(img)          # [0, 255]
    arr = preprocess_input(arr)            # [-1, 1] for MobileNetV2
    return np.expand_dims(arr, axis=0)


def run_batch_test(model, folder_path):

    results = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                filepath = os.path.join(root, file)
                true_label = os.path.basename(root)

                processed = preprocess_image(filepath)

                prediction, confidence, top_k_labels, probabilities = predict_card(
                    model,
                    processed
                )

                # Top-k confidences derived from the raw probability array
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_k_confs = [float(probabilities[i]) for i in top_indices]

                top2_label = top_k_labels[1] if len(top_k_labels) > 1 else ""
                top3_label = top_k_labels[2] if len(top_k_labels) > 2 else ""
                top2_conf  = top_k_confs[1]  if len(top_k_confs)  > 1 else 0.0
                top3_conf  = top_k_confs[2]  if len(top_k_confs)  > 2 else 0.0
                conf_margin = confidence - top2_conf

                # Parse set identifier components
                parts = prediction.split()
                set_code    = parts[0] if len(parts) > 0 else ""
                card_number = parts[1] if len(parts) > 1 else ""

                # Database lookup for card metadata
                try:
                    card_info = lookup_card(prediction)
                except Exception:
                    card_info = None
                card_name = card_info["name"]   if card_info else ""
                rarity    = card_info["rarity"] if card_info else ""

                # -----------------------------
                # Statistical evaluation logic
                # -----------------------------

                if confidence < CONFIDENCE_THRESHOLD:
                    accepted    = 0
                    correct     = 0
                    top3_correct = 0
                else:
                    accepted    = 1
                    correct     = 1 if prediction == true_label else 0
                    top3_correct = 1 if true_label in top_k_labels else 0

                results.append([
                    file,
                    true_label,
                    prediction,
                    confidence,
                    top2_label,
                    top2_conf,
                    top3_label,
                    top3_conf,
                    conf_margin,
                    set_code,
                    card_number,
                    card_name,
                    rarity,
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
            "top2_label",
            "top2_confidence",
            "top3_label",
            "top3_confidence",
            "confidence_margin",
            "set_code",
            "card_number",
            "card_name",
            "rarity",
            "accepted",
            "top1_correct",
            "top3_correct"
        ])
        writer.writerows(results)

    return output_file, len(results)
