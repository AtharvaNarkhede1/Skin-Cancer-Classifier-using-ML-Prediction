"""
train.py
--------
Training pipeline for the skin cancer binary classifier.
Loads images from data/train/{benign,malignant} folders,
builds CNN, trains with augmentation, evaluates on test set,
and saves model + plots.

Usage:
    python -m src.train
    (run from project root)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# Matplotlib backend that works without a display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Ensure project root is on sys.path ───────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import prepare_dataset, get_data_augmentor, IMG_SHAPE
from src.model import build_cnn_model


def train():
    """Full training pipeline."""

    # ── Paths ────────────────────────────────────────────
    # Dataset: data/train/benign  and  data/train/malignant
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'train')
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, 'skin_cancer_model.keras')

    # ── 1. Load & preprocess data ────────────────────────
    print("=" * 60)
    print("  SKIN CANCER CLASSIFIER — TRAINING")
    print("  Dataset: folder-based (benign / malignant)")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(data_dir)

    # ── 2. Compute class weights (dataset may be imbalanced) ─
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"\nClass weights: {class_weight}")

    # ── 3. Data augmentation ─────────────────────────────
    datagen = get_data_augmentor()
    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, y_train, batch_size=32)

    # ── 4. Build model ───────────────────────────────────
    model = build_cnn_model(input_shape=IMG_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # ── 5. Callbacks ─────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── 6. Train ─────────────────────────────────────────
    print("\nStarting training...\n")
    history = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # ── 7. Evaluate on test set ──────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION ON TEST SET")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_acc:.4f}")

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    print("\nClassification Report:")
    print(classification_report(y_test.astype(int), y_pred,
                                target_names=['Benign', 'Malignant']))

    # ── 8. Save training plots ───────────────────────────
    _plot_history(history, model_dir)
    _plot_confusion_matrix(y_test.astype(int), y_pred, model_dir)

    print(f"\n✓ Model saved to  : {model_save_path}")
    print(f"✓ Plots saved to  : {model_dir}/")
    print("=" * 60)


# ─── Helper: training curves ─────────────────────────────
def _plot_history(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved -> {path}")


# ─── Helper: confusion matrix ────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved -> {path}")


if __name__ == "__main__":
    train()
