"""
model.py
--------
CNN architecture for binary skin cancer classification (Benign vs Malignant).
Architecture follows the Kaggle reference with Conv2D → BatchNorm → MaxPool → Dropout blocks,
adapted from 7-class softmax to 1-neuron sigmoid for binary output.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Dense,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2


def build_cnn_model(input_shape=(100, 125, 3)):
    """
    Build a CNN for binary skin-lesion classification.

    Architecture (3 conv blocks + dense head):
        Block 1: 2×Conv2D(32) → BN → MaxPool → Dropout(0.2)
        Block 2: 2×Conv2D(64) → BN → MaxPool → Dropout(0.3)
        Block 3: 2×Conv2D(128) → BN → MaxPool → Dropout(0.3)
        Head   : Flatten → Dense(256) → BN → Dropout(0.3) → Dense(1, sigmoid)
    """
    model = Sequential(name="SkinCancer_CNN")

    # ── Block 1 ──────────────────────────────────────
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     input_shape=input_shape, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # ── Block 2 ──────────────────────────────────────
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # ── Block 3 ──────────────────────────────────────
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # ── Dense Head ───────────────────────────────────
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output: 1 neuron + sigmoid → binary classification
    model.add(Dense(1, activation='sigmoid'))

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
    print("\nModel architecture ready for binary classification.")
