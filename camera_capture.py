import cv2
import tkinter as tk
from tkinter import ttk


def capture_image_with_buttons():
    """
    Opens a webcam preview window and a small Tkinter control window.
    Returns a captured frame (numpy array) or None if cancelled.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None

    captured_frame = {"image": None}
    running = True

    # -------------------------
    # Control window
    # -------------------------
    control_root = tk.Tk()
    control_root.title("Camera Controls")
    control_root.geometry("220x120")
    control_root.resizable(False, False)

    def on_capture():
        nonlocal running
        captured_frame["image"] = current_frame.copy()
        running = False
        control_root.destroy()

    def on_cancel():
        nonlocal running
        running = False
        control_root.destroy()

    ttk.Label(control_root, text="Align card, then capture").pack(pady=8)
    ttk.Button(control_root, text="Capture", command=on_capture).pack(pady=5)
    ttk.Button(control_root, text="Cancel", command=on_cancel).pack(pady=5)

    # -------------------------
    # Camera loop
    # -------------------------
    current_frame = None

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame

        # Draw a guide rectangle (centered)
        h, w, _ = frame.shape
        box_w, box_h = int(w * 0.6), int(h * 0.9)
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Camera Preview", frame)

        # Allow Tkinter + OpenCV to coexist
        control_root.update()
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    return captured_frame["image"]
