"""
model.py
========
Arsitektur CNN-LSTM untuk autentikasi gait.

Struktur:
  Input (128, 6)
    -> Conv1D Block 1: Conv1D(64) -> BN -> ReLU -> MaxPool
    -> Conv1D Block 2: Conv1D(128) -> BN -> ReLU -> MaxPool
    -> LSTM(128) -> LSTM(64)
    -> Dropout(0.3)
    -> Dense(64) -> ReLU
    -> Dense(1) -> Sigmoid (binary verification)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

from config import (
    WINDOW_SIZE, N_CHANNELS, CNN_FILTERS, CNN_KERNEL,
    LSTM_UNITS, DROPOUT_RATE, DENSE_UNITS,
    BATCH_SIZE, LEARNING_RATE, CENTRALIZED_EPOCHS, PATIENCE,
    MODELS_DIR
)

log = logging.getLogger(__name__)


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_cnn_lstm(
    input_shape: Tuple[int, int] = (WINDOW_SIZE, N_CHANNELS),
    summary: bool = False
) -> keras.Model:
    """
    Bangun model CNN-LSTM untuk binary gait verification.

    Args:
        input_shape: (window_size, n_channels) = (128, 6)
        summary:     cetak model summary jika True

    Returns:
        model Keras yang belum di-compile
    """
    inp = keras.Input(shape=input_shape, name="sensor_input")

    # ── CNN Block 1 ──────────────────────────────────────────────────────────
    x = layers.Conv1D(
        filters=CNN_FILTERS[0],
        kernel_size=CNN_KERNEL,
        padding="same",
        name="conv1"
    )(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool1")(x)

    # ── CNN Block 2 ──────────────────────────────────────────────────────────
    x = layers.Conv1D(
        filters=CNN_FILTERS[1],
        kernel_size=CNN_KERNEL,
        padding="same",
        name="conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool2")(x)

    # ── LSTM Layers ───────────────────────────────────────────────────────────
    x = layers.Dropout(DROPOUT_RATE, name="dropout_lstm")(x)
    x = layers.LSTM(LSTM_UNITS[0], return_sequences=True, name="lstm1")(x)
    x = layers.LSTM(LSTM_UNITS[1], return_sequences=False, name="lstm2")(x)

    # ── Classification Head ───────────────────────────────────────────────────
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense1")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_out")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="GaitAuth_CNN_LSTM")

    if summary:
        model.summary()

    return model


def compile_model(model: keras.Model) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")]
    )
    return model


# ─── Training Utilities ───────────────────────────────────────────────────────

def get_class_weight(y: np.ndarray) -> Dict[int, float]:
    """Hitung class weight untuk menangani imbalance."""
    n_total = len(y)
    n_pos   = y.sum()
    n_neg   = n_total - n_pos
    w_pos   = n_total / (2 * n_pos) if n_pos > 0 else 1.0
    w_neg   = n_total / (2 * n_neg) if n_neg > 0 else 1.0
    return {0: w_neg, 1: w_pos}


def train_centralized(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = CENTRALIZED_EPOCHS,
    model_name: str = "centralized"
) -> keras.callbacks.History:
    """
    Latih model secara terpusat (semua data, tidak federated).
    Digunakan untuk Skenario 1 (baseline).
    """
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
            monitor="val_loss", save_best_only=True, verbose=0
        )
    ]

    class_weight = get_class_weight(y_train)
    log.info(f"Mulai training centralized | epochs={epochs} | class_weight={class_weight}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    return history


def get_weights(model: keras.Model):
    """Ambil bobot model sebagai list numpy arrays."""
    return [w.numpy() for w in model.weights]


def set_weights(model: keras.Model, weights):
    """Set bobot model dari list numpy arrays."""
    for var, val in zip(model.weights, weights):
        var.assign(val)


# ─── Authentication Metrics ───────────────────────────────────────────────────

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Hitung Equal Error Rate (EER) dan threshold-nya.

    EER adalah titik di mana FAR (False Acceptance Rate) = FRR (False Rejection Rate).

    Returns:
        (eer, threshold)
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr   # FRR = 1 - TPR

    # Cari titik paling dekat dengan FAR = FRR
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


def compute_far_frr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    Hitung FAR dan FRR pada threshold tertentu.

    FAR (False Acceptance Rate): impostor yang diterima sebagai genuine
    FRR (False Rejection Rate):  genuine yang ditolak
    """
    y_pred = (y_scores >= threshold).astype(int)
    impostor_mask = y_true == 0
    genuine_mask  = y_true == 1

    far = np.sum((y_pred == 1) & impostor_mask) / np.sum(impostor_mask)
    frr = np.sum((y_pred == 0) & genuine_mask)  / np.sum(genuine_mask)
    return float(far), float(frr)


def evaluate_authentication(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluasi lengkap sistem autentikasi.

    Returns dict dengan: accuracy, precision, recall, f1, far, frr, eer, auc
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    y_scores = model.predict(X_test, verbose=0).flatten()
    eer, eer_threshold = compute_eer(y_test, y_scores)
    far, frr = compute_far_frr(y_test, y_scores, eer_threshold)

    # Gunakan EER threshold sebagai decision threshold
    y_pred = (y_scores >= eer_threshold).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "far":       far,
        "frr":       frr,
        "eer":       eer,
        "auc":       roc_auc_score(y_test, y_scores),
        "threshold": eer_threshold
    }

    log.info(
        f"Evaluation | acc={metrics['accuracy']:.4f} | "
        f"EER={metrics['eer']:.4f} | AUC={metrics['auc']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    model = build_cnn_lstm(summary=True)
    compile_model(model)

    # Dummy forward pass
    dummy_X = np.random.randn(16, WINDOW_SIZE, N_CHANNELS).astype(np.float32)
    dummy_y = np.random.randint(0, 2, 16).astype(np.float32)
    loss = model.train_on_batch(dummy_X, dummy_y)
    log.info(f"Dummy forward pass OK | loss={loss[0]:.4f}")
