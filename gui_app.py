import csv
import os
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import cv2
import customtkinter as ctk
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from batch_test import run_batch_test
from camera_capture import capture_image_with_buttons
from config import IMG_SIZE, CONFIDENCE_THRESHOLD
from database import lookup_card
from predict import predict_card, load_trained_model

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SCAN_LOG_PATH = "scan_log.csv"
SCAN_LOG_HEADERS = [
    "timestamp", "source", "predicted_label", "confidence",
    "top2_label", "top2_confidence", "top3_label", "top3_confidence",
    "confidence_margin", "set_code", "card_number", "card_name", "rarity",
    "reverse_holo",
]


class CardScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokémon Card Scanner")
        self.root.geometry("540x560")
        self.root.resizable(False, False)

        self.last_scan = None

        # ── Title ──────────────────────────────────────────────────────
        ctk.CTkLabel(
            root,
            text="Pokémon Card Scanner",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(pady=(28, 4))

        self.status_label = ctk.CTkLabel(
            root,
            text="Loading model...",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
        )
        self.status_label.pack(pady=(0, 18))
        root.update()

        self.model = load_trained_model()
        self.status_label.configure(text="Model ready. Choose an input method.")

        # ── Action buttons ─────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(root, fg_color="transparent")
        btn_frame.pack(pady=(0, 18))

        ctk.CTkButton(
            btn_frame,
            text="Capture Image",
            command=self.on_capture_clicked,
            width=158,
        ).grid(row=0, column=0, padx=6)

        ctk.CTkButton(
            btn_frame,
            text="Upload Image",
            command=self.on_upload_clicked,
            width=158,
        ).grid(row=0, column=1, padx=6)

        ctk.CTkButton(
            btn_frame,
            text="Batch Test Folder",
            command=self.on_batch_test_clicked,
            width=328,
            fg_color=("gray78", "gray25"),
            hover_color=("gray68", "gray32"),
            text_color=("gray10", "gray90"),
        ).grid(row=1, column=0, columnspan=2, pady=(10, 0))

        # ── Result card ────────────────────────────────────────────────
        result_card = ctk.CTkFrame(root, corner_radius=14)
        result_card.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        ctk.CTkLabel(
            result_card,
            text="RESULT",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="gray50",
        ).pack(anchor="w", padx=18, pady=(14, 0))

        self.result_label = ctk.CTkLabel(
            result_card,
            text="Waiting for input...",
            font=ctk.CTkFont(size=13),
            wraplength=460,
            justify="center",
        )
        self.result_label.pack(expand=True, padx=18, pady=8)

        # Annotation row — hidden until a successful single scan
        self.annotation_frame = ctk.CTkFrame(result_card, fg_color="transparent")
        self.reverse_holo_var = tk.BooleanVar()
        ctk.CTkCheckBox(
            self.annotation_frame,
            text="Reverse Holo",
            variable=self.reverse_holo_var,
        ).pack(side="left", padx=(0, 12))
        ctk.CTkButton(
            self.annotation_frame,
            text="Log Scan",
            command=self.log_scan,
            width=100,
        ).pack(side="left")

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(self, filepath):
        img = image.load_img(filepath, target_size=IMG_SIZE)
        arr = image.img_to_array(img)      # [0, 255]
        arr = preprocess_input(arr)        # [-1, 1] for MobileNetV2
        return np.expand_dims(arr, axis=0)

    def preprocess_cv_frame(self, frame):
        frame_rgb = frame[:, :, ::-1]
        resized = cv2.resize(frame_rgb, IMG_SIZE)
        arr = preprocess_input(resized.astype(np.float32))
        return np.expand_dims(arr, axis=0)

    # ------------------------------------------------------------------
    # Shared prediction dispatch
    # ------------------------------------------------------------------

    def _run_prediction(self, processed, source):
        prediction, confidence, top_k_labels, probabilities = predict_card(
            self.model,
            processed
        )
        self.display_result(prediction, confidence, top_k_labels, probabilities, source)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def on_upload_clicked(self):
        file_path = filedialog.askopenfilename(
            title="Select card image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        self.result_label.configure(text="Running prediction...")
        self.root.update()

        processed = self.preprocess_image(file_path)
        self._run_prediction(processed, os.path.basename(file_path))

    def on_capture_clicked(self):
        self.result_label.configure(text="Opening camera...")
        self.root.update()

        frame = capture_image_with_buttons()
        if frame is None:
            self.result_label.configure(text="Capture cancelled.")
            return

        self.result_label.configure(text="Running prediction...")
        self.root.update()

        processed = self.preprocess_cv_frame(frame)
        self._run_prediction(processed, "camera")

    def on_batch_test_clicked(self):
        folder_path = filedialog.askdirectory(
            title="Select folder containing test images"
        )
        if not folder_path:
            return

        self.annotation_frame.pack_forget()
        self.last_scan = None
        self.result_label.configure(text="Running batch test...")
        self.root.update()

        output_file, count = run_batch_test(self.model, folder_path)

        self.result_label.configure(
            text=(
                f"Batch complete.\n\n"
                f"Images processed: {count}\n"
                f"Results saved to:\n{output_file}"
            )
        )

    # ------------------------------------------------------------------
    # Result display
    # ------------------------------------------------------------------

    def display_result(self, prediction, confidence, top_k_labels, probabilities, source="unknown"):
        confidence_pct = confidence * 100
        self.annotation_frame.pack_forget()
        self.last_scan = None

        if confidence < CONFIDENCE_THRESHOLD:
            self.result_label.configure(
                text=f"No card detected\n\nConfidence: {confidence_pct:.1f}%"
            )
            return

        # Resolve top-k confidences from the raw probability array
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_k_confs = [float(probabilities[i]) for i in top_indices]
        top2_label  = top_k_labels[1] if len(top_k_labels) > 1 else ""
        top3_label  = top_k_labels[2] if len(top_k_labels) > 2 else ""
        top2_conf   = top_k_confs[1]  if len(top_k_confs)  > 1 else 0.0
        top3_conf   = top_k_confs[2]  if len(top_k_confs)  > 2 else 0.0
        conf_margin = confidence - top2_conf

        parts       = prediction.split()
        set_code    = parts[0] if len(parts) > 0 else ""
        card_number = parts[1] if len(parts) > 1 else ""

        try:
            card_info = lookup_card(prediction)
        except Exception:
            card_info = None
        card_name = card_info["name"]   if card_info else ""
        rarity    = card_info["rarity"] if card_info else ""

        if card_info:
            text = (
                f"Predicted: {prediction}  ({confidence_pct:.1f}%)\n"
                f"Name: {card_name}  |  Rarity: {rarity}\n"
                f"Set: {set_code}  |  Number: {card_number}\n\n"
                f"2nd: {top2_label} ({top2_conf * 100:.1f}%)   "
                f"3rd: {top3_label} ({top3_conf * 100:.1f}%)\n"
                f"Margin: {conf_margin * 100:.1f}%"
            )
        else:
            text = (
                f"Predicted: {prediction}  ({confidence_pct:.1f}%)\n"
                f"No database entry found.\n\n"
                f"2nd: {top2_label} ({top2_conf * 100:.1f}%)   "
                f"3rd: {top3_label} ({top3_conf * 100:.1f}%)\n"
                f"Margin: {conf_margin * 100:.1f}%"
            )

        self.result_label.configure(text=text)

        self.last_scan = {
            "source":            source,
            "predicted_label":   prediction,
            "confidence":        confidence,
            "top2_label":        top2_label,
            "top2_confidence":   top2_conf,
            "top3_label":        top3_label,
            "top3_confidence":   top3_conf,
            "confidence_margin": conf_margin,
            "set_code":          set_code,
            "card_number":       card_number,
            "card_name":         card_name,
            "rarity":            rarity,
        }

        self.reverse_holo_var.set(False)
        self.annotation_frame.pack(pady=(0, 14))

    # ------------------------------------------------------------------
    # Scan logging
    # ------------------------------------------------------------------

    def log_scan(self):
        if self.last_scan is None:
            return

        file_exists = os.path.isfile(SCAN_LOG_PATH)
        with open(SCAN_LOG_PATH, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCAN_LOG_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reverse_holo": int(self.reverse_holo_var.get()),
                **self.last_scan,
            })

        self.annotation_frame.pack_forget()
        self.last_scan = None
        self.status_label.configure(text="Scan logged.")
        self.root.after(2000, lambda: self.status_label.configure(
            text="Model ready. Choose an input method."
        ))


def main():
    root = ctk.CTk()
    app = CardScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
