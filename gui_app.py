import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from keras.preprocessing import image
import cv2

from predict import predict_card, load_trained_model
from train import get_training_set
from database import lookup_card
from config import IMG_SIZE, CONFIDENCE_THRESHOLD
from camera_capture import capture_image_with_buttons


class CardScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokémon Card Scanner")
        self.root.geometry("520x420")
        self.root.resizable(False, False)

        # =========================
        # Main container
        # =========================
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # =========================
        # Title
        # =========================
        title_label = ttk.Label(
            main_frame,
            text="Pokémon Card Scanner",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # =========================
        # Status label
        # =========================
        self.status_label = ttk.Label(
            main_frame,
            text="Loading model...",
            font=("Segoe UI", 11)
        )
        self.status_label.pack(pady=(0, 10))
        self.root.update()

        # =========================
        # Load model + classes ONCE
        # =========================
        self.model = load_trained_model()
        self.training_set = get_training_set()

        self.status_label.config(text="Model ready. Choose input method.")

        # =========================
        # Button frame
        # =========================
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.capture_button = ttk.Button(
            button_frame,
            text="Capture Image",
            command=self.on_capture_clicked
        )
        self.capture_button.grid(row=0, column=0, padx=10)

        self.upload_button = ttk.Button(
            button_frame,
            text="Upload Image",
            command=self.on_upload_clicked
        )
        self.upload_button.grid(row=0, column=1, padx=10)

        # =========================
        # Result section
        # =========================
        result_frame = ttk.LabelFrame(
            main_frame,
            text="Result",
            padding=15
        )
        result_frame.pack(fill="both", expand=True, pady=20)

        self.result_label = ttk.Label(
            result_frame,
            text="Waiting for input...",
            font=("Segoe UI", 11),
            wraplength=460,
            justify="center"
        )
        self.result_label.pack()

    # =========================
    # Image preprocessing
    # =========================
    def preprocess_image(self, filepath):
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def preprocess_cv_frame(self, frame):
        frame_rgb = frame[:, :, ::-1]  # BGR → RGB
        frame_resized = cv2.resize(frame_rgb, IMG_SIZE)
        frame_array = frame_resized / 255.0
        return np.expand_dims(frame_array, axis=0)

    # =========================
    # Upload handler
    # =========================
    def on_upload_clicked(self):
        file_path = filedialog.askopenfilename(
            title="Select card image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        self.result_label.config(text="Running prediction...")
        self.root.update()

        processed_image = self.preprocess_image(file_path)

        prediction, confidence = predict_card(
            self.model,
            self.training_set,
            processed_image
        )

        self.display_result(prediction, confidence)

    # =========================
    # Camera handler
    # =========================
    def on_capture_clicked(self):
        self.result_label.config(text="Opening camera...")
        self.root.update()

        frame = capture_image_with_buttons()

        if frame is None:
            self.result_label.config(text="Capture cancelled.")
            return

        self.result_label.config(text="Running prediction...")
        self.root.update()

        processed_image = self.preprocess_cv_frame(frame)

        prediction, confidence = predict_card(
            self.model,
            self.training_set,
            processed_image
        )

        self.display_result(prediction, confidence)

    # =========================
    # Unified result display
    # =========================
    def display_result(self, prediction, confidence):
        confidence_pct = confidence * 100

        if confidence < CONFIDENCE_THRESHOLD:
            self.result_label.config(
                text=(
                    "No card detected\n\n"
                    f"Confidence: {confidence_pct:.1f}%"
                )
            )
            return

        card_info = lookup_card(prediction)

        if card_info:
            result_text = (
                f"Predicted Card: {prediction}\n"
                f"Confidence: {confidence_pct:.1f}%\n\n"
                f"Name: {card_info['name']}\n"
                f"Set: {card_info['set_code']}\n"
                f"Number: {card_info['card_number']}\n"
                f"Rarity: {card_info['rarity']}"
            )
        else:
            result_text = (
                f"Predicted Card: {prediction}\n"
                f"Confidence: {confidence_pct:.1f}%\n\n"
                "No database entry found for this card."
            )

        self.result_label.config(text=result_text)


# =========================
# App entry point
# =========================
def main():
    root = tk.Tk()
    app = CardScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
