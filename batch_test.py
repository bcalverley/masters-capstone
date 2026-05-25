import csv
import os
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image as PILImage, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import IMG_SIZE, CONFIDENCE_THRESHOLD
from database import lookup_card
from predict import predict_card, load_trained_model


def _normalize_label(folder_name):
    """'JTG18' or 'JTG018' → 'JTG 18'  (matches trainingassets folder format).
    Leaves already-spaced labels like 'JTG 18' unchanged."""
    m = re.match(r'^([A-Za-z]+)\s*0*(\d+)$', folder_name.strip())
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    return folder_name


def preprocess_image(filepath):
    img = PILImage.open(filepath)
    img = ImageOps.exif_transpose(img)          # honour phone rotation EXIF tag
    if img.width > img.height:                  # card photographed landscape — rotate to portrait
        img = img.rotate(90, expand=True)
    img = img.resize(IMG_SIZE, PILImage.LANCZOS).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def get_image_paths(folder_path):
    """Return sorted list of (filepath, normalized_label) for all images under folder_path.
    Folder names like 'JTG18' are normalized to 'JTG 18' to match model class labels."""
    paths = []
    for root, dirs, files in os.walk(folder_path):
        label = _normalize_label(os.path.basename(root))
        for file in sorted(files):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append((os.path.join(root, file), label))
    return paths


def predict_single(model, filepath):
    """Run prediction on one image; return a dict of all result fields."""
    processed = preprocess_image(filepath)
    prediction, confidence, top_k_labels, probabilities = predict_card(model, processed)

    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_k_confs = [float(probabilities[i]) for i in top_indices]

    top2_label = top_k_labels[1] if len(top_k_labels) > 1 else ""
    top3_label = top_k_labels[2] if len(top_k_labels) > 2 else ""
    top2_conf  = top_k_confs[1]  if len(top_k_confs)  > 1 else 0.0
    top3_conf  = top_k_confs[2]  if len(top_k_confs)  > 2 else 0.0

    parts       = prediction.split()
    set_code    = parts[0] if parts else ""
    card_number = parts[1] if len(parts) > 1 else ""

    try:
        card_info = lookup_card(prediction)
    except Exception:
        card_info = None
    card_name = card_info["name"]   if card_info else ""
    rarity    = card_info["rarity"] if card_info else ""

    return {
        "prediction":   prediction,
        "confidence":   confidence,
        "top2_label":   top2_label,  "top2_conf":  top2_conf,
        "top3_label":   top3_label,  "top3_conf":  top3_conf,
        "conf_margin":  confidence - top2_conf,
        "set_code":     set_code,    "card_number": card_number,
        "card_name":    card_name,   "rarity":     rarity,
        "accepted":     confidence >= CONFIDENCE_THRESHOLD,
        "top_k_labels": top_k_labels,
    }


REVIEW_HEADERS = [
    "filename", "true_label", "predicted_label", "confidence",
    "top2_label", "top2_confidence", "top3_label", "top3_confidence",
    "confidence_margin", "true_set_code", "predicted_set_code", "card_number",
    "card_name", "rarity", "accepted", "top1_correct", "top3_correct",
    "variant", "grade", "market_price", "added_to_session",
]


def write_results_csv(rows, output_path):
    """Write interactive review results with per-row data + summary + per-set breakdown."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REVIEW_HEADERS)
        for row in rows:
            writer.writerow([row.get(h, "") for h in REVIEW_HEADERS])

        total = len(rows)
        if total == 0:
            return

        accepted = sum(1 for r in rows if r.get("accepted"))
        top1     = sum(1 for r in rows if r.get("top1_correct"))
        top3     = sum(1 for r in rows if r.get("top3_correct"))
        added    = sum(1 for r in rows if r.get("added_to_session"))
        pct      = lambda n: f"{n / total * 100:.1f}%"

        writer.writerow([])
        writer.writerow(["=== SUMMARY ==="])
        writer.writerow(["Total reviewed",                total])
        writer.writerow(["Accepted (>=70% confidence)",   accepted,       pct(accepted)])
        writer.writerow(["Rejected (<70% confidence)",    total-accepted, pct(total-accepted)])
        writer.writerow(["Top-1 correct",                 top1,           pct(top1)])
        writer.writerow(["Top-3 correct",                 top3,           pct(top3)])
        writer.writerow(["Added to receipt",              added,          pct(added)])

        # Per-set breakdown
        set_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "accepted": 0})
        for r in rows:
            s = r.get("true_set_code") or "unknown"
            set_stats[s]["total"]    += 1
            set_stats[s]["top1"]     += int(r.get("top1_correct", 0))
            set_stats[s]["top3"]     += int(r.get("top3_correct", 0))
            set_stats[s]["accepted"] += int(r.get("accepted", 0))

        writer.writerow([])
        writer.writerow(["=== BY SET ==="])
        writer.writerow(["Set", "Cards", "Accepted", "Top-1 Correct", "Top-1 %", "Top-3 Correct", "Top-3 %"])
        for s, st in sorted(set_stats.items()):
            n = st["total"]
            writer.writerow([
                s, n, st["accepted"],
                st["top1"], f"{st['top1']/n*100:.1f}%" if n else "",
                st["top3"], f"{st['top3']/n*100:.1f}%" if n else "",
            ])


def run_batch_test(model, folder_path):

    results = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                filepath = os.path.join(root, file)
                true_label = _normalize_label(os.path.basename(root))

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

    os.makedirs("batch_results", exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file   = os.path.join("batch_results", f"batch_results_{timestamp}.csv")
    excel_file = os.path.join("batch_results", f"batch_results_{timestamp}.xlsx")

    with open(csv_file, mode="w", newline="") as f:
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

    try:
        from generate_excel import build_excel
        build_excel(csv_file, excel_file)
    except Exception:
        pass

    return csv_file, len(results)
