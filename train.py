import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import TRAINING_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS

MODEL_PATH = "card_model.keras"


def _make_datagen(subset):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # maps [0,255] → [-1,1]
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.12,
        brightness_range=[0.85, 1.15],
        shear_range=0.08,
        # horizontal_flip intentionally off — card orientation is meaningful
        fill_mode="nearest",
        validation_split=0.15,
    )
    return datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset=subset,
        shuffle=(subset == "training"),
    )


def _build_model(num_classes):
    base = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # frozen during phase 1

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    return model, base


def train_model():
    train_gen = _make_datagen("training")
    val_gen   = _make_datagen("validation")
    num_classes = train_gen.num_classes

    print(f"\nClasses : {num_classes}")
    print(f"Train   : {len(train_gen)} batches")
    print(f"Val     : {len(val_gen)} batches")

    model, base = _build_model(num_classes)

    # ── Phase 1: train classification head, base frozen ───────────────
    print("\n===== PHASE 1: Training head (base frozen) =====\n")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=15,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
    )

    # ── Phase 2: unfreeze top 30 layers and fine-tune ─────────────────
    print("\n===== PHASE 2: Fine-tuning top 30 base layers =====\n")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # lower LR to avoid destroying pretrained weights
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[
            EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
            ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        ],
    )

    print(f"\nBest model saved to {MODEL_PATH}")
    print("Next: python export_labels.py")
    return model, train_gen


def get_training_set():
    """Returns a generator used only for class-index mapping (export_labels.py)."""
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    return datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False,
    )


if __name__ == "__main__":
    train_model()
