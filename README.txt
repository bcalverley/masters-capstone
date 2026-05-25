Pokemon Card Scanner — Buying Tool
====================================
Masters Capstone Project


OVERVIEW
--------
A desktop application that uses computer vision to identify Pokemon cards from a
webcam or uploaded photo, look up live market prices, and help card shops make
buying decisions. The model is built on MobileNetV2 transfer learning and trained
on augmented images downloaded from the official Pokemon TCG API.

Target sets: JTG, PRE, SCR, SFA, SSP (Scarlet & Violet era, 2024-2025)
Cards outside these sets are rejected when confidence falls below 70%.


REQUIREMENTS
------------
Python 3.11
  pip install -r requirements.txt

The trained model (card_model.keras) and class label file (class_labels.json) are
not included in the repo because of file size. To generate them, run the training
pipeline in order:

  1. python download_training_data.py   # downloads card images from the Pokemon TCG API
  2. python train.py                    # trains the model (30-60 min on CPU)
  3. python export_labels.py            # saves the class label index to class_labels.json


RUNNING THE APP
---------------
  python gui_app.py


HOW IT WORKS
------------
1. A card is scanned via webcam or uploaded as an image file.
2. The image is resized to 160x160, normalized, and passed through the model.
3. The model outputs a confidence score for each card class it was trained on.
4. If the top prediction is >= 70% confident, it is accepted and the card identity
   is returned. Below 70%, the result is rejected as unrecognised.
5. The app looks up the live TCGPlayer market price for the identified card.
6. Cards can be added to a session, which tracks market totals and calculates
   offer amounts at 80%, 75%, and 60% of market value.
7. At the end of a session, a PDF receipt is exported to the receipts/ folder.


KEY FILES
---------
  gui_app.py                 Main application window (two-panel buying tool UI)
  batch_review.py            Card-by-card batch review window for testing folders of images
  predict.py                 Runs the model and returns the top prediction and confidence
  price_lookup.py            Fetches live TCGPlayer market prices with grade multipliers
  session.py                 Tracks the buying session and handles CSV receipt export
  receipt_pdf.py             Generates the formatted PDF receipt
  config.py                  All configuration settings in one place
  camera_capture.py          Webcam capture window

  train.py                   MobileNetV2 training script (two-phase: head then fine-tune)
  download_training_data.py  Downloads card images from the Pokemon TCG API and augments them
  export_labels.py           Saves the class label index after training
  audit_training_data.py     Checks training data coverage per set before training
  batch_test.py              Batch prediction logic used by the batch review window
  generate_evaluation_set.py Pulls a random sample from training assets to use as an eval set
  generate_excel.py          Converts a batch results CSV into a formatted Excel workbook
  generate_report.py         Generates a detailed evaluation report CSV from batch results


ARCHITECTURE
------------
- Model:       MobileNetV2 pretrained on ImageNet, fine-tuned on ~13,000 augmented
               card images across 896 classes (5 target sets x ~179 cards average)
- Training:    Two-phase — phase 1 trains the classification head with the base frozen,
               phase 2 unfreezes the top 30 layers and fine-tunes at a lower learning rate
- Augmentation: Each card gets 15 training images generated from the official API render
               with random brightness, contrast, blur, perspective warp, and noise applied
               to simulate real webcam conditions
- Validation:  99.33% accuracy on the augmented validation set (15% held-out split).
               Real-world accuracy on physical cards varies by set — SFA and SSP both
               reached ~90% on accepted predictions. JTG performed poorly due to reverse
               holo card finishes that are not present in the flat API training images.
- Pricing:     Pokemon TCG API -> TCGPlayer market data, results cached per session
- Database:    Supabase (optional) for card name and rarity lookup — app works without it
- UI:          CustomTkinter dark mode, 960x660 fixed layout
