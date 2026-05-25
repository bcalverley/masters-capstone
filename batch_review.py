import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import customtkinter as ctk
from PIL import Image as PILImage, ImageOps

from batch_test import predict_single, write_results_csv
from config import VARIANTS, GRADES, CONFIDENCE_THRESHOLD
from price_lookup import get_price
from session import SessionItem


class BatchReviewWindow(ctk.CTkToplevel):
    def __init__(self, parent, model, image_paths, session, on_complete=None):
        super().__init__(parent)
        self.title("Batch Review")
        self.geometry("940x760")
        self.minsize(940, 680)
        self.resizable(False, True)
        self.grab_set()
        self.lift()
        self.focus_force()

        self.model       = model
        self.image_paths = image_paths   # list of (filepath, folder_label)
        self.session     = session
        self.on_complete = on_complete

        self.current_idx    = 0
        self.results        = []
        self._pred          = None
        self._current_price = None
        self._ctk_img_ref   = None   # prevent GC of CTkImage

        self._build_ui()
        self.after(80, lambda: self._load_card(0))

    #
    # UI
    #

    def _build_ui(self):
        self.columnconfigure(0, weight=0, minsize=292)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(12, 4))
        hdr.columnconfigure(1, weight=1)

        ctk.CTkLabel(hdr, text="Batch Review",
                     font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, sticky="w")

        self.progress_label = ctk.CTkLabel(hdr, text=f"0 of {len(self.image_paths)}",
                                           font=ctk.CTkFont(size=12), text_color="gray60")
        self.progress_label.grid(row=0, column=1, sticky="e")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 8))
        self.progress_bar.set(0)

        # ── Left: image + labels ───────────────────────────────────────
        left = ctk.CTkFrame(self, corner_radius=10)
        left.grid(row=2, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
        left.columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(left, text="Loading...", width=252, height=320)
        self.image_label.grid(row=0, column=0, padx=10, pady=(12, 6))

        self.filename_label = ctk.CTkLabel(left, text="", font=ctk.CTkFont(size=10),
                                           text_color="gray60", wraplength=264, justify="center")
        self.filename_label.grid(row=1, column=0, padx=10)

        self.accept_badge = ctk.CTkLabel(left, text="",
                                         font=ctk.CTkFont(size=11, weight="bold"))
        self.accept_badge.grid(row=2, column=0, pady=(8, 0))

        ctk.CTkLabel(left, text="FOLDER / TRUE LABEL",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50"
                     ).grid(row=3, column=0, pady=(10, 0))

        self.folder_label_widget = ctk.CTkLabel(left, text="—",
                                                font=ctk.CTkFont(size=13, weight="bold"))
        self.folder_label_widget.grid(row=4, column=0, padx=10, pady=(2, 14))

        # ── Right: review controls ─────────────────────────────────────
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=2, column=1, sticky="nsew", padx=(0, 16), pady=(0, 16))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)   # spacer pushes buttons down

        # Top predictions
        pred_frame = ctk.CTkFrame(right, corner_radius=10)
        pred_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        pred_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(pred_frame, text="TOP PREDICTIONS",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50"
                     ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(10, 4))

        self.pred_rows = []
        rank_colors = ["#4fc3f7", "white", "gray60"]
        for i in range(3):
            ctk.CTkLabel(pred_frame, text=f"#{i+1}",
                         font=ctk.CTkFont(size=11, weight="bold"),
                         text_color=rank_colors[i], width=28
                         ).grid(row=i+1, column=0, padx=(14, 6), pady=4)

            name_lbl = ctk.CTkLabel(pred_frame, text="—",
                                    font=ctk.CTkFont(size=12), anchor="w",
                                    text_color=rank_colors[i])
            name_lbl.grid(row=i+1, column=1, sticky="ew", padx=4, pady=4)

            conf_lbl = ctk.CTkLabel(pred_frame, text="—",
                                    font=ctk.CTkFont(size=12, weight="bold"),
                                    text_color=rank_colors[i], width=68)
            conf_lbl.grid(row=i+1, column=2, padx=(4, 14), pady=4)

            self.pred_rows.append((name_lbl, conf_lbl))

        ctk.CTkLabel(pred_frame, text="").grid(row=4, column=0, pady=(0, 2))

        # Actual card entry
        actual_frame = ctk.CTkFrame(right, corner_radius=10)
        actual_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        actual_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(actual_frame, text="ACTUAL CARD (edit if folder label is wrong)",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50"
                     ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.actual_entry = ctk.CTkEntry(actual_frame, placeholder_text="e.g. SFA 45")
        self.actual_entry.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        self.actual_entry.bind("<KeyRelease>", self._on_actual_changed)

        self.match_label = ctk.CTkLabel(actual_frame, text="",
                                        font=ctk.CTkFont(size=11), text_color="gray60")
        self.match_label.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 10))

        # Condition
        cond_frame = ctk.CTkFrame(right, corner_radius=10)
        cond_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        cond_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(cond_frame, text="CONDITION",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50"
                     ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.variant_var = tk.StringVar(value="Normal")
        ctk.CTkSegmentedButton(cond_frame, values=VARIANTS, variable=self.variant_var,
                               command=self._on_condition_changed
                               ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        self.grade_var = tk.StringVar(value="Ungraded")
        ctk.CTkOptionMenu(cond_frame, values=GRADES, variable=self.grade_var,
                          command=self._on_condition_changed
                          ).grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))

        # Spacer
        ctk.CTkFrame(right, fg_color="transparent", height=1).grid(row=4, column=0, sticky="nsew")

        # Price + action buttons (anchored to bottom)
        bottom = ctk.CTkFrame(right, fg_color="transparent")
        bottom.grid(row=5, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        price_frame = ctk.CTkFrame(bottom, corner_radius=10)
        price_frame.pack(fill="x", pady=(0, 8))
        price_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(price_frame, text="MARKET PRICE",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color="gray50"
                     ).grid(row=0, column=0, sticky="w", padx=14, pady=(8, 0))

        self.price_label = ctk.CTkLabel(price_frame, text="—",
                                        font=ctk.CTkFont(size=24, weight="bold"),
                                        text_color="#4fc3f7")
        self.price_label.grid(row=1, column=0, pady=(2, 8))

        btn_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        btn_frame.pack(fill="x")
        btn_frame.columnconfigure((0, 1), weight=1)

        ctk.CTkButton(btn_frame, text="Add to Receipt & Next →",
                      command=self._add_and_next, height=44,
                      font=ctk.CTkFont(size=13, weight="bold"),
                      fg_color="#2e7d32", hover_color="#388e3c"
                      ).grid(row=0, column=0, padx=(0, 4), sticky="ew")

        ctk.CTkButton(btn_frame, text="Next (don't add) →",
                      command=self._skip_and_next, height=44,
                      fg_color=("gray78", "gray30"), hover_color=("gray68", "gray38"),
                      text_color=("gray10", "gray90")
                      ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

    #
    # Card loading
    #

    def _load_card(self, idx):
        filepath, folder_label = self.image_paths[idx]
        total = len(self.image_paths)

        self.progress_label.configure(text=f"Card {idx + 1} of {total}")
        self.progress_bar.set(idx / total)
        self.title(f"Batch Review — Card {idx + 1} of {total}")

        # Thumbnail
        try:
            img = PILImage.open(filepath)
            img = ImageOps.exif_transpose(img)  # honour phone rotation EXIF tag
            if img.width > img.height:          # card photographed landscape — rotate to portrait
                img = img.rotate(90, expand=True)
            img.thumbnail((252, 320))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(img.width, img.height))
            self._ctk_img_ref = ctk_img
            self.image_label.configure(image=ctk_img, text="")
        except Exception:
            self.image_label.configure(image=None, text="[No preview]")

        self.filename_label.configure(text=os.path.basename(filepath))
        self.folder_label_widget.configure(text=folder_label)

        # Prediction (synchronous — user has time to review while reading)
        try:
            self._pred = predict_single(self.model, filepath)
        except Exception as e:
            self._pred = None
            self.accept_badge.configure(text=f"PREDICTION ERROR: {e}", text_color="#ff8a65")
            return
        pred = self._pred

        # Acceptance badge
        if pred["accepted"]:
            self.accept_badge.configure(text="ACCEPTED", text_color="#4caf50")
        else:
            self.accept_badge.configure(
                text=f"LOW CONFIDENCE — {pred['confidence']*100:.1f}%", text_color="#ff8a65"
            )

        # Prediction rows
        tops = [
            (pred["prediction"], pred["confidence"]),
            (pred["top2_label"], pred["top2_conf"]),
            (pred["top3_label"], pred["top3_conf"]),
        ]
        for i, (label, conf) in enumerate(tops):
            self.pred_rows[i][0].configure(text=label or "—")
            self.pred_rows[i][1].configure(text=f"{conf*100:.1f}%" if label else "—")

        # Actual card pre-filled with folder label (ground truth)
        self.actual_entry.delete(0, "end")
        self.actual_entry.insert(0, folder_label)
        self._update_match_label()

        # Reset condition + price
        self.variant_var.set("Normal")
        self.grade_var.set("Ungraded")
        self._current_price = None
        self.price_label.configure(text="—", text_color="#4fc3f7")
        self._fetch_price_async()

    #
    # Actual card field
    #

    def _on_actual_changed(self, _=None):
        self._update_match_label()

    def _update_match_label(self):
        if self._pred is None:
            return
        actual = self.actual_entry.get().strip()
        pred   = self._pred["prediction"]
        if not actual:
            self.match_label.configure(text="", text_color="gray60")
        elif actual == pred:
            self.match_label.configure(text="Matches prediction", text_color="#4caf50")
        else:
            self.match_label.configure(text=f"Prediction was: {pred}", text_color="#ef9a9a")

    #
    # Condition / price
    #

    def _on_condition_changed(self, _=None):
        self._fetch_price_async()

    def _fetch_price_async(self):
        if self._pred is None:
            return
        self.price_label.configure(text="...", text_color="gray60")
        pred    = self._pred
        variant = self.variant_var.get()
        grade   = self.grade_var.get()

        def worker():
            price = get_price(pred["set_code"], pred["card_number"], variant, grade)
            self.after(0, lambda: self._set_price(price))

        threading.Thread(target=worker, daemon=True).start()

    def _set_price(self, price):
        self._current_price = price
        if price is not None:
            self.price_label.configure(text=f"${price:.2f}", text_color="#4fc3f7")
        else:
            self.price_label.configure(text="N/A", text_color="gray60")

    #
    # Navigation
    #

    def _record_result(self, add_to_session):
        if self._pred is None:
            self._advance()
            return
        filepath, folder_label = self.image_paths[self.current_idx]
        actual_label = self.actual_entry.get().strip() or folder_label
        pred         = self._pred

        top1_correct = int(pred["prediction"] == actual_label)
        top3_correct = int(actual_label in pred["top_k_labels"])

        true_parts    = actual_label.split()
        true_set_code = true_parts[0] if true_parts else ""

        self.results.append({
            "filename":           os.path.basename(filepath),
            "true_label":         actual_label,
            "predicted_label":    pred["prediction"],
            "confidence":         round(pred["confidence"], 4),
            "top2_label":         pred["top2_label"],
            "top2_confidence":    round(pred["top2_conf"], 4),
            "top3_label":         pred["top3_label"],
            "top3_confidence":    round(pred["top3_conf"], 4),
            "confidence_margin":  round(pred["conf_margin"], 4),
            "true_set_code":      true_set_code,
            "predicted_set_code": pred["set_code"],
            "card_number":        pred["card_number"],
            "card_name":          pred["card_name"],
            "rarity":             pred["rarity"],
            "accepted":           int(pred["accepted"]),
            "top1_correct":       top1_correct,
            "top3_correct":       top3_correct,
            "variant":            self.variant_var.get(),
            "grade":              self.grade_var.get(),
            "market_price":       self._current_price if self._current_price is not None else "",
            "added_to_session":   int(add_to_session),
        })

        if add_to_session:
            self.session.add_item(SessionItem(
                set_code=pred["set_code"],
                card_number=pred["card_number"],
                card_name=pred["card_name"],
                variant=self.variant_var.get(),
                grade=self.grade_var.get(),
                market_price=self._current_price,
                predicted_label=pred["prediction"],
                confidence=pred["confidence"],
            ))

    def _add_and_next(self):
        self._record_result(add_to_session=True)
        self._advance()

    def _skip_and_next(self):
        self._record_result(add_to_session=False)
        self._advance()

    def _advance(self):
        self.current_idx += 1
        if self.current_idx >= len(self.image_paths):
            self._show_complete()
        else:
            self._load_card(self.current_idx)

    #
    # Completion
    #

    def _show_complete(self):
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="Complete!")
        self.title("Batch Review — Complete")

        rows     = self.results
        total    = len(rows)
        accepted = sum(1 for r in rows if r["accepted"])
        top1     = sum(1 for r in rows if r["top1_correct"])
        top3     = sum(1 for r in rows if r["top3_correct"])
        added    = sum(1 for r in rows if r["added_to_session"])
        pct      = lambda n: f"{n/total*100:.0f}%" if total else "—"

        os.makedirs("batch_results", exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path   = os.path.join("batch_results", f"batch_review_{timestamp}.csv")
        excel_path = os.path.join("batch_results", f"batch_review_{timestamp}.xlsx")

        write_results_csv(rows, csv_path)
        try:
            from generate_excel import build_excel
            build_excel(csv_path, excel_path)
            files_msg = f"batch_results/batch_review_{timestamp}.csv\nbatch_results/batch_review_{timestamp}.xlsx"
        except Exception:
            files_msg = f"batch_results/batch_review_{timestamp}.csv"

        msg = (
            f"Review complete  —  {total} card{'s' if total != 1 else ''}\n\n"
            f"Accepted (>=70%):   {accepted}/{total}  ({pct(accepted)})\n"
            f"Top-1 correct:        {top1}/{total}  ({pct(top1)})\n"
            f"Top-3 correct:        {top3}/{total}  ({pct(top3)})\n"
            f"Added to receipt:    {added}/{total}\n\n"
            f"Results saved to:\n{files_msg}"
        )
        messagebox.showinfo("Review Complete", msg, parent=self)

        if self.on_complete:
            self.on_complete()

        self.destroy()
