import cv2
import numpy as np
from keras.preprocessing import image
from config import IMG_SIZE, CARD_ASPECT, CAPTURE_PATH


def capture_card():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("\nAlign card | 's' capture | 'q' quit\n")

    captured_image = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        clean_frame = frame.copy()
        h, w, _ = frame.shape

        card_h = int(h * 0.8)
        card_w = int(card_h / CARD_ASPECT)

        if card_w > w:
            card_w = int(w * 0.8)
            card_h = int(card_w * CARD_ASPECT)

        x1 = (w - card_w) // 2
        y1 = (h - card_h) // 2
        x2 = x1 + card_w
        y2 = y1 + card_h

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Card Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            crop = clean_frame[y1:y2, x1:x2]
            cv2.imwrite(CAPTURE_PATH, crop)

            resized = cv2.resize(crop, IMG_SIZE)
            img_array = image.img_to_array(resized) / 255.0
            captured_image = np.expand_dims(img_array, axis=0)
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_image is None:
        raise RuntimeError("No image captured")

    return captured_image
