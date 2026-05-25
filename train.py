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
        # horizontal flip off — a mirrored card is a different card visually
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
    base.trainable = False  # keep the pretrained weights frozen during phase 1

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

    # Phase 1 — train just the new classification head, base weights stay frozen
    print("\nPhase 1: training classification head (base frozen)\n")
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

    # Phase 2 — unfreeze the top 30 layers and fine-tune at a lower learning rate
    print("\nPhase 2: fine-tuning top 30 base layers\n")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # lower learning rate so we don't overwrite the pretrained weights
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
    # used by export_labels.py to get the class index mapping
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
