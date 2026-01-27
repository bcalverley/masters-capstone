import os
from train import train_model
from webcam import capture_card
from predict import predict_card
from database import lookup_card
from config import CAPTURE_DIR

os.makedirs(CAPTURE_DIR, exist_ok=True)

from pathlib import Path
from config import TRAINING_DIR

if not TRAINING_DIR.exists():
    print("Training data not found.")
    print("The application is installed correctly.")
    print("Please add training images to:")
    print(TRAINING_DIR)
    input("\nPress Enter to exit...")
    exit(0)

model, training_set = train_model()

captured_image = capture_card()

prediction = predict_card(model, training_set, captured_image)
print("\nPredicted class:", prediction)

card = lookup_card(prediction)

if card:
    print("\n===== CARD INFO =====")
    print(card["name"], "|", card["set_code"], "|", card["card_number"], "|", card["rarity"])
else:
    print("No matching card found.")
