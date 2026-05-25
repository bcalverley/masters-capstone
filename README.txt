Pokemon Card Scanner — Buying Tool
====================================
Masters Capstone Project


OVERVIEW
--------
A desktop application that uses computer vision (MobileNetV2 transfer learning)
to identify Pokemon cards from a webcam or uploaded image, retrieve live market
prices from the TCGPlayer API, and calculate offer tiers for a buying session.

Target sets: JTG, PRE, SCR, SFA, SSP (Scarlet & Violet era, 2024-2025)
Out-of-scope cards (MTG, older sets, etc.) are rejected via a 70% confidence threshold.


REQUIREMENTS
------------
Python 3.11
  pip install -r requirements.txt

The trained model (card_model.keras) and class labels (class_labels.json) are
not tracked in git due to file size. Generate them by running the training pipeline:

  1. python download_training_data.py   # download ~13,000 images from PokemonTCG API
  2. python train.py                    # train MobileNetV2 (~30-60 min on CPU)
  3. python export_labels.py            # generate class_labels.json


RUNNING THE APP
---------------
  python gui_app.py


KEY FILES
---------
  gui_app.py                 Main application (two-panel buying tool UI)
  batch_review.py            Interactive card-by-card batch review window
  predict.py                 Model inference
  price_lookup.py            TCGPlayer live price lookup with grade multipliers
  session.py                 Buying session state and receipt export
  config.py                  All configuration constants
  camera_capture.py          Webcam capture

  train.py                   MobileNetV2 training (two-phase: head then fine-tune)
  download_training_data.py  Downloads card images from PokemonTCG API + augmentation
  export_labels.py           Regenerates class_labels.json from trained model
  audit_training_data.py     Reports training data coverage per set
  batch_test.py              Batch prediction engine (used by batch_review.py)

  build.bat                  PyInstaller build script
  PokemonCardScanner.spec    PyInstaller spec file


ARCHITECTURE
------------
- Model:       MobileNetV2 pretrained on ImageNet, fine-tuned on 13,440 augmented
               card images across 896 classes (5 target sets)
- Validation:  99.33% accuracy on held-out 15% validation split
- Pricing:     PokemonTCG API -> TCGPlayer market data, cached per session
- Database:    Supabase (optional) for card metadata lookup; app runs without it
- UI:          CustomTkinter dark mode, 960x660 two-panel layout
