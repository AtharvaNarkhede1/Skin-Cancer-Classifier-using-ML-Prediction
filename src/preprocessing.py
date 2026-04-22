"""
preprocessing.py
-----------------
Data loading, preprocessing, and augmentation for the skin cancer dataset.
Loads images directly from folder structure:
    data/train/
        benign/     → label 0
        malignant/  → label 1

Binary classification: Benign (0) vs Malignant (1).
"""

import numpy as np
import os
from PIL import Image
from glob import glob
from io import BytesIO
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# Constants — used everywhere for consistency
# ──────────────────────────────────────────────
IMG_HEIGHT = 100
IMG_WIDTH = 125
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

CLASSES = {
    'benign': 0,
    'malignant': 1,
}


def load_and_resize_image(image_path):
    """Load a single image, resize to (IMG_WIDTH, IMG_HEIGHT), return as uint8 array."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # PIL uses (width, height)
    return np.asarray(img, dtype='uint8')


def prepare_dataset(data_dir):
    """
    Full pipeline: scan benign/malignant folders → load images → normalise → split.

    Parameters
    ----------
    data_dir : str
        Path to the train directory containing 'benign' and 'malignant' subfolders.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test  (all numpy arrays)
    """
    images = []
    labels = []

    for class_name, label in CLASSES.items():
        folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(
                f"Expected folder not found: {folder}\n"
                f"Make sure '{class_name}' subfolder exists inside {data_dir}"
            )

        # Accept .jpg, .jpeg, .png
        paths = (
            glob(os.path.join(folder, '*.jpg')) +
            glob(os.path.join(folder, '*.jpeg')) +
            glob(os.path.join(folder, '*.png'))
        )[:300]  # LIMIT TO 300 FOR BETTER ACCURACY

        print(f"  [{class_name.upper()}]  found {len(paths)} images (label={label})")

        for i, path in enumerate(paths):
            try:
                img = load_and_resize_image(path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"    Skipping {os.path.basename(path)}: {e}")

            if (i + 1) % 500 == 0:
                print(f"    Loaded {i + 1}/{len(paths)} from {class_name}...")

    X = np.array(images, dtype='float32') / 255.0   # normalise to [0, 1]
    y = np.array(labels, dtype='float32')

    print(f"\nTotal images loaded : {len(X)}")
    print(f"  Benign            : {(y == 0).sum()}")
    print(f"  Malignant         : {(y == 1).sum()}")

    # Split: 70 % train, 15 % val, 15 % test  (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Train : {len(X_train)}  |  Val : {len(X_val)}  |  Test : {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_data_augmentor():
    """Returns an ImageDataGenerator for training-time augmentation."""
    return ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.12,
        width_shift_range=0.12,
        height_shift_range=0.12,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.85, 1.15],
    )


def preprocess_single_image(image_bytes):
    """
    Preprocess a single image (from raw bytes) exactly the same way as training.
    Returns a (1, IMG_HEIGHT, IMG_WIDTH, 3) float32 array normalised to [0, 1].
    """
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.asarray(img, dtype='float32') / 255.0
    return np.expand_dims(img, axis=0)


if __name__ == "__main__":
    print("Preprocessing module ready.")
    print(f"Image shape used: {IMG_SHAPE}")
