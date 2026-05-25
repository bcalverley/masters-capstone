import csv
import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image as PILImage, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from batch_review import BatchReviewWindow
from batch_test import get_image_paths
from camera_capture import capture_image_with_buttons
from config import IMG_SIZE, CONFIDENCE_THRESHOLD, VARIANTS, GRADES, OFFER_TIERS
from database import lookup_card
from predict import predict_card, load_trained_model
from price_lookup import get_price
from session import BuyingSession, SessionItem

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SCAN_LOG_PATH = "scan_log.csv"
SCAN_LOG_HEADERS = [
    "timestamp", "source", "predicted_label", "confidence",
    "top2_label", "top2_confidence", "top3_label", "top3_confidence",
    "confidence_margin", "set_code", "card_number", "card_name", "rarity",
    "variant", "grade", "market_price",
]


class CardScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokemon Card Scanner — Buying Tool")
        self.root.geometry("960x660")
        self.root.resizable(False, False)

        self.last_scan = None
        self._current_price = None
        self.session = BuyingSession(offer_tiers=OFFER_TIERS)

        self._build_ui()

        self.status_label.configure(text="Loading model...")
        root.update()
        self.model = load_trained_model()
        self.status_label.configure(text="Model ready.")

    #
    # UI construction
    #

    def _build_ui(self):
        self.root.columnconfigure(0, weight=0, minsize=340)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        left = ctk.CTkFrame(self.root, corner_radius=0)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left, text="Card Scanner",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, pady=(20, 2))

        self.status_label = ctk.CTkLabel(
            left, text="",
            font=ctk.CTkFont(size=11), text_color="gray60",
        )
        self.status_label.grid(row=1, column=0, pady=(0, 12))

        # Scan buttons
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=16, pady=(0, 12))
        btn_frame.columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            btn_frame, text="Capture", command=self.on_capture_clicked, width=142,
        ).grid(row=0, column=0, padx=4)
        ctk.CTkButton(
            btn_frame, text="Upload", command=self.on_upload_clicked, width=142,
        ).grid(row=0, column=1, padx=4)

        # Result card
        result_card = ctk.CTkFrame(left, corner_radius=10)
        result_card.grid(row=3, column=0, padx=16, pady=(0, 10), sticky="ew")
        result_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            result_card, text="SCAN RESULT",
            font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))

        self.result_label = ctk.CTkLabel(
            result_card, text="Scan a card to begin.",
            font=ctk.CTkFont(size=12), wraplength=296, justify="center",
        )
        self.result_label.grid(row=1, column=0, padx=14, pady=(4, 12))

        # Condition controls
        cond_frame = ctk.CTkFrame(left, corner_radius=10)
        cond_frame.grid(row=4, column=0, padx=16, pady=(0, 10), sticky="ew")
        cond_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            cond_frame, text="CONDITION",
            font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.variant_var = tk.StringVar(value="Normal")
        ctk.CTkSegmentedButton(
            cond_frame, values=VARIANTS, variable=self.variant_var,
            command=self._on_condition_changed, width=308,
        ).grid(row=1, column=0, padx=14, pady=(0, 8))

        self.grade_var = tk.StringVar(value="Ungraded")
        ctk.CTkOptionMenu(
            cond_frame, values=GRADES, variable=self.grade_var,
            command=self._on_condition_changed, width=308,
        ).grid(row=2, column=0, padx=14, pady=(0, 12))

        # Price display
        price_frame = ctk.CTkFrame(left, corner_radius=10)
        price_frame.grid(row=5, column=0, padx=16, pady=(0, 10), sticky="ew")
        price_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            price_frame, text="MARKET PRICE",
            font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))

        self.price_label = ctk.CTkLabel(
            price_frame, text="—",
            font=ctk.CTkFont(size=30, weight="bold"), text_color="#4fc3f7",
        )
        self.price_label.grid(row=1, column=0, pady=(2, 12))

        # Add to Session
        self.add_btn = ctk.CTkButton(
            left, text="Add to Session",
            command=self.add_to_session,
            height=44, font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2e7d32", hover_color="#388e3c",
            state="disabled",
        )
        self.add_btn.grid(row=6, column=0, padx=16, pady=(0, 8), sticky="ew")

        # Batch test (secondary action)
        ctk.CTkButton(
            left, text="Batch Test Folder",
            command=self.on_batch_test_clicked, height=32,
            fg_color=("gray78", "gray25"), hover_color=("gray68", "gray32"),
            text_color=("gray10", "gray90"),
        ).grid(row=7, column=0, padx=16, pady=(0, 20), sticky="ew")

    def _build_right_panel(self):
        right = ctk.CTkFrame(self.root, corner_radius=0, fg_color=("gray90", "gray15"))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(right, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="SESSION",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        self.item_count_label = ctk.CTkLabel(
            header, text="0 items",
            font=ctk.CTkFont(size=11), text_color="gray60",
        )
        self.item_count_label.grid(row=0, column=1, sticky="e")

        # Scrollable item list
        self.session_scroll = ctk.CTkScrollableFrame(right, corner_radius=8)
        self.session_scroll.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 8))
        self.session_scroll.columnconfigure(0, weight=1)

        self._render_empty_state()

        # Totals section
        totals_frame = ctk.CTkFrame(right, corner_radius=10)
        totals_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        totals_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            totals_frame, text="TOTALS",
            font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 4))

        self.total_labels = {}
        rows = [("Market Total", "market")] + [(f"Offer @ {p}%", p) for p in OFFER_TIERS]
        for i, (label_text, key) in enumerate(rows):
            ctk.CTkLabel(
                totals_frame, text=label_text,
                font=ctk.CTkFont(size=12),
                text_color="white" if key == "market" else "gray70",
            ).grid(row=i + 1, column=0, sticky="w", padx=14, pady=2)

            val = ctk.CTkLabel(
                totals_frame, text="$0.00",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#4fc3f7" if key == "market" else "gray80",
            )
            val.grid(row=i + 1, column=1, sticky="e", padx=14, pady=2)
            self.total_labels[key] = val

        # Action buttons
        action_frame = ctk.CTkFrame(right, fg_color="transparent")
        action_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(4, 16))
        action_frame.columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            action_frame, text="Clear Session",
            command=self.clear_session, height=36,
            fg_color=("gray78", "gray30"), hover_color=("gray68", "gray38"),
            text_color=("gray10", "gray90"),
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(
            action_frame, text="Export Receipt",
            command=self.export_receipt, height=36,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    #
    # Image preprocessing
    #

    def preprocess_image(self, filepath):
        img = PILImage.open(filepath)
        img = ImageOps.exif_transpose(img)      # honour phone rotation EXIF tag
        if img.width > img.height:              # card photographed landscape — rotate to portrait
            img = img.rotate(90, expand=True)
        img = img.resize(IMG_SIZE, PILImage.LANCZOS).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    def preprocess_cv_frame(self, frame):
        frame_rgb = frame[:, :, ::-1]
        resized = cv2.resize(frame_rgb, IMG_SIZE)
        arr = preprocess_input(resized.astype(np.float32))
        return np.expand_dims(arr, axis=0)

    #
    # Condition change -> re-fetch price
    #

    def _on_condition_changed(self, _=None):
        if self.last_scan is not None:
            self._fetch_price_async()

    def _fetch_price_async(self):
        self.price_label.configure(text="...", text_color="gray60")
        self.add_btn.configure(state="disabled")

        scan = self.last_scan
        variant = self.variant_var.get()
        grade = self.grade_var.get()

        def worker():
            price = get_price(scan["set_code"], scan["card_number"], variant, grade)
            self.root.after(0, lambda: self._set_price(price))

        threading.Thread(target=worker, daemon=True).start()

    def _set_price(self, price):
        self._current_price = price
        if price is not None:
            self.price_label.configure(text=f"${price:.2f}", text_color="#4fc3f7")
        else:
            self.price_label.configure(text="N/A", text_color="gray60")
        self.add_btn.configure(state="normal")

    #
    # Shared prediction dispatch
    #

    def _run_prediction(self, processed, source):
        prediction, confidence, top_k_labels, probabilities = predict_card(
            self.model, processed
        )
        self.display_result(prediction, confidence, top_k_labels, probabilities, source)

    #
    # Button handlers
    #

    def on_upload_clicked(self):
        file_path = filedialog.askopenfilename(
            title="Select card image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
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
        folder_path = filedialog.askdirectory(title="Select folder of card images")
        if not folder_path:
            return

        image_paths = get_image_paths(folder_path)
        if not image_paths:
            self.result_label.configure(text="No images found in selected folder.")
            return

        self.result_label.configure(text=f"Opening review for {len(image_paths)} cards...")
        BatchReviewWindow(self.root, self.model, image_paths, self.session, self._on_batch_complete)

    def _on_batch_complete(self):
        self._refresh_session_ui()
        self.result_label.configure(text="Batch review complete. Session updated.")

    #
    # Result display
    #

    def display_result(self, prediction, confidence, top_k_labels, probabilities, source="unknown"):
        confidence_pct = confidence * 100
        self.last_scan = None
        self._current_price = None
        self.add_btn.configure(state="disabled")
        self.price_label.configure(text="—", text_color="#4fc3f7")

        if confidence < CONFIDENCE_THRESHOLD:
            self.result_label.configure(
                text=f"No card detected\nConfidence: {confidence_pct:.1f}%"
            )
            return

        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_k_confs = [float(probabilities[i]) for i in top_indices]
        top2_label = top_k_labels[1] if len(top_k_labels) > 1 else ""
        top3_label = top_k_labels[2] if len(top_k_labels) > 2 else ""
        top2_conf  = top_k_confs[1]  if len(top_k_confs)  > 1 else 0.0
        top3_conf  = top_k_confs[2]  if len(top_k_confs)  > 2 else 0.0
        conf_margin = confidence - top2_conf

        parts       = prediction.split()
        set_code    = parts[0] if parts else ""
        card_number = parts[1] if len(parts) > 1 else ""

        try:
            card_info = lookup_card(prediction)
        except Exception:
            card_info = None
        card_name = card_info["name"]   if card_info else ""
        rarity    = card_info["rarity"] if card_info else ""

        if card_info:
            text = (
                f"{prediction}  ({confidence_pct:.1f}%)\n"
                f"{card_name}  ·  {rarity}\n\n"
                f"2nd: {top2_label} ({top2_conf * 100:.1f}%)\n"
                f"3rd: {top3_label} ({top3_conf * 100:.1f}%)"
            )
        else:
            text = (
                f"{prediction}  ({confidence_pct:.1f}%)\n"
                f"No database entry found.\n\n"
                f"2nd: {top2_label} ({top2_conf * 100:.1f}%)\n"
                f"3rd: {top3_label} ({top3_conf * 100:.1f}%)"
            )

        self.result_label.configure(text=text)

        self.last_scan = {
            "source": source,
            "predicted_label": prediction,
            "confidence": confidence,
            "top2_label": top2_label, "top2_confidence": top2_conf,
            "top3_label": top3_label, "top3_confidence": top3_conf,
            "confidence_margin": conf_margin,
            "set_code": set_code, "card_number": card_number,
            "card_name": card_name, "rarity": rarity,
        }

        self.variant_var.set("Normal")
        self.grade_var.set("Ungraded")
        self._fetch_price_async()

    #
    # Session management
    #

    def add_to_session(self):
        if self.last_scan is None:
            return

        scan = self.last_scan
        item = SessionItem(
            set_code=scan["set_code"],
            card_number=scan["card_number"],
            card_name=scan["card_name"],
            variant=self.variant_var.get(),
            grade=self.grade_var.get(),
            market_price=self._current_price,
            predicted_label=scan["predicted_label"],
            confidence=scan["confidence"],
        )
        self.session.add_item(item)
        self._log_to_csv(scan, item)
        self._refresh_session_ui()

        self.last_scan = None
        self._current_price = None
        self.add_btn.configure(state="disabled")
        self.price_label.configure(text="—", text_color="#4fc3f7")
        self.result_label.configure(text="Added! Scan next card.")

    def clear_session(self):
        self.session.clear()
        self._refresh_session_ui()

    def export_receipt(self):
        if not self.session.items:
            return
        os.makedirs("receipts", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path  = os.path.join("receipts", f"receipt_{timestamp}.csv")
        pdf_path  = os.path.join("receipts", f"receipt_{timestamp}.pdf")
        self.session.export_receipt(csv_path)
        try:
            from receipt_pdf import export_receipt_pdf
            export_receipt_pdf(self.session, pdf_path)
            self.status_label.configure(text=f"Receipt saved: receipts/receipt_{timestamp}.pdf")
        except Exception as e:
            self.status_label.configure(text=f"CSV saved (PDF failed: {e})")

    #
    # Session UI rendering
    #

    def _render_empty_state(self):
        label = ctk.CTkLabel(
            self.session_scroll, text="No cards in session yet.",
            font=ctk.CTkFont(size=12), text_color="gray50",
        )
        label.pack(pady=40)

    def _refresh_session_ui(self):
        for widget in self.session_scroll.winfo_children():
            widget.destroy()

        items = self.session.items
        if not items:
            self._render_empty_state()
        else:
            for idx, item in enumerate(items):
                self._add_session_row(idx, item)

        count = len(items)
        self.item_count_label.configure(
            text=f"{count} item{'s' if count != 1 else ''}"
        )
        self._update_totals()

    def _add_session_row(self, idx, item):
        row = ctk.CTkFrame(self.session_scroll, corner_radius=8)
        row.pack(fill="x", pady=3, padx=4)
        row.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            row, text=f"{idx + 1}.",
            font=ctk.CTkFont(size=11), text_color="gray50", width=24,
        ).grid(row=0, column=0, padx=(8, 4), pady=8, sticky="w")

        info_frame = ctk.CTkFrame(row, fg_color="transparent")
        info_frame.grid(row=0, column=1, sticky="ew", pady=6)

        name_text = item.card_name if item.card_name else item.predicted_label
        ctk.CTkLabel(
            info_frame, text=name_text,
            font=ctk.CTkFont(size=12, weight="bold"), anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            info_frame,
            text=f"{item.variant}  ·  {item.grade}  ·  {item.set_code} #{item.card_number}",
            font=ctk.CTkFont(size=10), text_color="gray60", anchor="w",
        ).pack(anchor="w")

        price_text = f"${item.market_price:.2f}" if item.market_price else "N/A"
        ctk.CTkLabel(
            row, text=price_text,
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#4fc3f7",
        ).grid(row=0, column=2, padx=8)

        ctk.CTkButton(
            row, text="x", width=28, height=28,
            fg_color="gray25", hover_color="#c62828",
            font=ctk.CTkFont(size=11),
            command=lambda i=idx: self._remove_item(i),
        ).grid(row=0, column=3, padx=(0, 8))

    def _remove_item(self, idx):
        self.session.remove_item(idx)
        self._refresh_session_ui()

    def _update_totals(self):
        total = self.session.market_total()
        self.total_labels["market"].configure(text=f"${total:.2f}")
        for pct in OFFER_TIERS:
            self.total_labels[pct].configure(text=f"${self.session.offer_amount(pct):.2f}")

    #
    # CSV logging
    #

    def _log_to_csv(self, scan, item):
        file_exists = os.path.isfile(SCAN_LOG_PATH)
        with open(SCAN_LOG_PATH, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCAN_LOG_HEADERS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source":            scan["source"],
                "predicted_label":   scan["predicted_label"],
                "confidence":        scan["confidence"],
                "top2_label":        scan["top2_label"],
                "top2_confidence":   scan["top2_confidence"],
                "top3_label":        scan["top3_label"],
                "top3_confidence":   scan["top3_confidence"],
                "confidence_margin": scan["confidence_margin"],
                "set_code":          scan["set_code"],
                "card_number":       scan["card_number"],
                "card_name":         scan["card_name"],
                "rarity":            scan["rarity"],
                "variant":           item.variant,
                "grade":             item.grade,
                "market_price":      item.market_price if item.market_price else "",
            })


def main():
    root = ctk.CTk()
    app = CardScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
