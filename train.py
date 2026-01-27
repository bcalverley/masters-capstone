import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D
from config import TRAINING_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS


def train_model():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False
    )

    training_set = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse"
    )

    cnn = Sequential([
        Input(shape=(64, 64, 3)),
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

    return cnn, training_set
