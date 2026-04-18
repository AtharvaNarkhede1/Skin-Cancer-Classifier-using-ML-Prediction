"""
preprocessing.py
-----------------
Data loading, preprocessing, and augmentation for HAM10000 skin cancer dataset.
Binary classification: Malignant (mel) vs Benign (all others).
"""

import numpy as np
import pandas as pd
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

# Lesion type readable names
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma',
}


def load_metadata(data_dir):
    """
    Load HAM10000 metadata CSV and map each image_id to its file path.
    Adds binary label: 1 = Malignant (mel), 0 = Benign (everything else).
    """
    csv_path = os.path.join(data_dir, 'HAM10000_metadata.csv')
    df = pd.read_csv(csv_path)

    # Build { image_id : full_path } from both image folders
    imageid_path_dict = {}
    for folder in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
        folder_path = os.path.join(data_dir, folder)
        for filepath in glob(os.path.join(folder_path, '*.jpg')):
            image_id = os.path.splitext(os.path.basename(filepath))[0]
            imageid_path_dict[image_id] = filepath

    df['path'] = df['image_id'].map(imageid_path_dict)
    df['cell_type'] = df['dx'].map(LESION_TYPE_DICT)

    # Binary label: mel → 1 (Malignant), rest → 0 (Benign)
    df['label'] = (df['dx'] == 'mel').astype(int)

    # Fill missing age with mean
    df['age'] = df['age'].fillna(int(df['age'].mean()))

    # Drop rows with missing image paths
    df = df.dropna(subset=['path'])

    return df


def load_and_resize_image(image_path):
    """Load a single image, resize to (IMG_WIDTH, IMG_HEIGHT), return as uint8 array."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # PIL uses (width, height)
    return np.asarray(img, dtype='uint8')


def prepare_dataset(data_dir):
    """
    Full pipeline: load metadata → load images → normalise → split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test (all numpy).
    """
    df = load_metadata(data_dir)

    print(f"Total samples : {len(df)}")
    print(f"  Benign      : {(df['label'] == 0).sum()}")
    print(f"  Malignant   : {(df['label'] == 1).sum()}")

    # Load all images (shows progress)
    print("Loading images...")
    images = []
    labels = []
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            img = load_and_resize_image(row['path'])
            images.append(img)
            labels.append(row['label'])
        except Exception as e:
            print(f"  Skipping {row['image_id']}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"  Loaded {i + 1}/{len(df)} images...")

    X = np.array(images, dtype='float32') / 255.0   # normalise to [0, 1]
    y = np.array(labels, dtype='float32')

    print(f"Images loaded: {len(X)}")

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
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.12,
        height_shift_range=0.12,
        horizontal_flip=True,
        vertical_flip=True,
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
