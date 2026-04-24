import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D
from config import TRAINING_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS


MODEL_PATH = "card_model.keras"


def train_model():
    """
    Trains CNN using strong augmentation to simulate
    real-world capture variability.
    """

    # ==============================
    # Enhanced augmentation strategy
    # ==============================
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        fill_mode="nearest"
    )

    training_set = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse"
    )

    # ==============================
    # CNN Architecture
    # ==============================
    cnn = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),

        Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(2, 2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(training_set.num_classes, activation="softmax")
    ])

    cnn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n===== TRAINING MODEL =====\n")

    cnn.fit(training_set, epochs=EPOCHS)
    cnn.save(MODEL_PATH)

    return cnn, training_set


def get_training_set():
    """
    Loads training set without augmentation.
    Used for class index mapping.
    """
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    training_set = datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=False
    )

    return training_set