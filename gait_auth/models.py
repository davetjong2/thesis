"""
models.py
=========
Arsitektur model deep learning untuk gait authentication.

Model yang tersedia:
  1. CNN-LSTM (baseline)       — Conv1D spatial + LSTM temporal
  2. CNN-GRU                   — Conv1D + GRU (lebih ringan dari LSTM)
  3. TCN (Temporal Conv Net)   — Dilated causal convolutions
  4. TransformerEncoder        — Multi-head self-attention
  5. InceptionTime             — Multi-scale convolutional kernels

Semua model menerima input (batch, 128, 6) dan output (batch, 1) sigmoid.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
import logging

from config import (
    INPUT_SHAPE, LEARNING_RATE, BATCH_SIZE,
    MAX_EPOCHS, PATIENCE,
)

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. CNN-LSTM (Baseline)
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn_lstm(
    input_shape: tuple = INPUT_SHAPE,
    cnn_filters: list = None,
    lstm_units: list = None,
    dropout: float = 0.3,
    name: str = "CNN_LSTM",
) -> Model:
    """
    Hybrid CNN-LSTM: Conv1D layers extract spatial features across sensor
    channels, LSTM layers capture temporal gait dynamics.

    Architecture:
        Conv1D(64) → BN → ReLU → MaxPool
        Conv1D(128) → BN → ReLU → MaxPool
        Dropout(0.3)
        LSTM(128, return_seq) → LSTM(64)
        Dense(64) → Dropout → Dense(1, sigmoid)

    Total params: ~350K
    """
    if cnn_filters is None:
        cnn_filters = [64, 128]
    if lstm_units is None:
        lstm_units = [128, 64]

    inp = layers.Input(shape=input_shape, name="sensor_input")
    x = inp

    # CNN blocks
    for i, f in enumerate(cnn_filters):
        x = layers.Conv1D(f, kernel_size=3, padding="same", name=f"conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = layers.Activation("relu", name=f"relu{i+1}")(x)
        x = layers.MaxPooling1D(pool_size=2, name=f"pool{i+1}")(x)

    x = layers.Dropout(dropout, name="dropout_cnn")(x)

    # LSTM blocks
    x = layers.LSTM(lstm_units[0], return_sequences=True, name="lstm1")(x)
    x = layers.LSTM(lstm_units[1], return_sequences=False, name="lstm2")(x)

    # Classifier
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="dropout_out")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name=name)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2. CNN-GRU
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn_gru(
    input_shape: tuple = INPUT_SHAPE,
    cnn_filters: list = None,
    gru_units: list = None,
    dropout: float = 0.3,
    name: str = "CNN_GRU",
) -> Model:
    """
    CNN-GRU: Sama seperti CNN-LSTM tetapi GRU ~33% lebih sedikit parameter
    (tidak ada cell state terpisah). Cocok untuk deployment on-device.

    GRU gates: reset gate + update gate (vs LSTM: forget + input + output).
    Referensi: Cho et al. (2014), "Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation"

    Total params: ~280K
    """
    if cnn_filters is None:
        cnn_filters = [64, 128]
    if gru_units is None:
        gru_units = [128, 64]

    inp = layers.Input(shape=input_shape, name="sensor_input")
    x = inp

    for i, f in enumerate(cnn_filters):
        x = layers.Conv1D(f, kernel_size=3, padding="same", name=f"conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn{i+1}")(x)
        x = layers.Activation("relu", name=f"relu{i+1}")(x)
        x = layers.MaxPooling1D(pool_size=2, name=f"pool{i+1}")(x)

    x = layers.Dropout(dropout, name="dropout_cnn")(x)

    x = layers.GRU(gru_units[0], return_sequences=True, name="gru1")(x)
    x = layers.GRU(gru_units[1], return_sequences=False, name="gru2")(x)

    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="dropout_out")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inp, outputs=out, name=name)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TCN (Temporal Convolutional Network)
# ══════════════════════════════════════════════════════════════════════════════

def _residual_block(
    x, filters: int, kernel_size: int, dilation_rate: int, dropout: float, block_id: int
):
    """
    Satu residual block TCN:
        Conv1D(dilated, causal) → BN → ReLU → Dropout
        Conv1D(dilated, causal) → BN → ReLU → Dropout
        + residual connection (1x1 conv jika dimensi berbeda)

    Dilated convolution memperluas receptive field secara eksponensial
    tanpa menambah parameter: receptive_field = Σ (kernel-1)*dilation + 1
    """
    res = x

    # First dilated conv
    x = layers.Conv1D(
        filters, kernel_size, padding="causal",
        dilation_rate=dilation_rate, name=f"tcn_conv{block_id}a"
    )(x)
    x = layers.BatchNormalization(name=f"tcn_bn{block_id}a")(x)
    x = layers.Activation("relu", name=f"tcn_relu{block_id}a")(x)
    x = layers.SpatialDropout1D(dropout, name=f"tcn_drop{block_id}a")(x)

    # Second dilated conv
    x = layers.Conv1D(
        filters, kernel_size, padding="causal",
        dilation_rate=dilation_rate, name=f"tcn_conv{block_id}b"
    )(x)
    x = layers.BatchNormalization(name=f"tcn_bn{block_id}b")(x)
    x = layers.Activation("relu", name=f"tcn_relu{block_id}b")(x)
    x = layers.SpatialDropout1D(dropout, name=f"tcn_drop{block_id}b")(x)

    # Residual connection
    if res.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding="same", name=f"tcn_res{block_id}")(res)

    x = layers.Add(name=f"tcn_add{block_id}")([x, res])
    return x


def build_tcn(
    input_shape: tuple = INPUT_SHAPE,
    n_filters: int = 64,
    kernel_size: int = 3,
    n_blocks: int = 4,
    dropout: float = 0.2,
    name: str = "TCN",
) -> Model:
    """
    Temporal Convolutional Network (Bai et al., 2018).

    Keunggulan vs RNN:
      - Parallelisasi penuh (tidak sequential)
      - Gradient stabil (tidak vanishing/exploding)
      - Receptive field terkontrol via dilation

    Dilation rates: [1, 2, 4, 8] → receptive field = 62 samples @ kernel=3
    Dengan 128 timesteps input, cukup untuk menangkap 1 siklus gait penuh.

    Total params: ~120K
    """
    inp = layers.Input(shape=input_shape, name="sensor_input")
    x = inp

    for i in range(n_blocks):
        dilation = 2 ** i
        x = _residual_block(x, n_filters, kernel_size, dilation, dropout, block_id=i)

    # Global pooling → classifier
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="dropout_out")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inp, outputs=out, name=name)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Transformer Encoder
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(layers.Layer):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Memberikan informasi posisi temporal ke input Transformer,
    karena self-attention sendiri bersifat permutation-invariant.
    """
    def __init__(self, max_len: int = 128, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def build(self, input_shape):
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe = np.zeros((self.max_len, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term[:self.d_model // 2])
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


def _transformer_encoder_block(x, d_model, n_heads, ff_dim, dropout, block_id):
    """
    Satu Transformer encoder block:
        MultiHeadAttention → Add&Norm → FFN → Add&Norm

    Self-attention memungkinkan setiap timestep 'melihat' seluruh
    sequence, menangkap dependensi jarak jauh tanpa batasan
    receptive field fixed.
    """
    # Multi-Head Self-Attention
    attn_out = layers.MultiHeadAttention(
        num_heads=n_heads, key_dim=d_model // n_heads,
        name=f"mha_{block_id}"
    )(x, x)
    attn_out = layers.Dropout(dropout, name=f"attn_drop_{block_id}")(attn_out)
    x = layers.LayerNormalization(name=f"ln1_{block_id}")(x + attn_out)

    # Feed-Forward Network
    ffn = layers.Dense(ff_dim, activation="relu", name=f"ffn1_{block_id}")(x)
    ffn = layers.Dense(d_model, name=f"ffn2_{block_id}")(ffn)
    ffn = layers.Dropout(dropout, name=f"ffn_drop_{block_id}")(ffn)
    x = layers.LayerNormalization(name=f"ln2_{block_id}")(x + ffn)

    return x


def build_transformer(
    input_shape: tuple = INPUT_SHAPE,
    d_model: int = 64,
    n_heads: int = 4,
    n_blocks: int = 3,
    ff_dim: int = 128,
    dropout: float = 0.2,
    name: str = "Transformer",
) -> Model:
    """
    Transformer Encoder for time-series classification.

    Mengadopsi arsitektur encoder-only dari Vaswani et al. (2017),
    diadaptasi untuk sinyal sensor 1D:
      - Proyeksi linear: (128,6) → (128, d_model)
      - Positional encoding: sinusoidal
      - N encoder blocks
      - Global average pooling → classifier

    Keunggulan: menangkap dependensi jangka panjang tanpa masalah
    vanishing gradient yang dimiliki LSTM.

    Total params: ~150K
    """
    inp = layers.Input(shape=input_shape, name="sensor_input")

    # Project input channels to d_model
    x = layers.Dense(d_model, name="input_proj")(inp)
    x = PositionalEncoding(max_len=input_shape[0], d_model=d_model, name="pos_enc")(x)

    for i in range(n_blocks):
        x = _transformer_encoder_block(x, d_model, n_heads, ff_dim, dropout, block_id=i)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="dropout_out")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inp, outputs=out, name=name)


# ══════════════════════════════════════════════════════════════════════════════
# 5. InceptionTime
# ══════════════════════════════════════════════════════════════════════════════

def _inception_module(x, n_filters: int, module_id: int):
    """
    Satu InceptionTime module: 3 branch konvolusi paralel + 1 MaxPool branch.

    Kernel sizes: [1, 3, 5, MaxPool+1x1]

    Setiap branch menangkap pattern pada skala temporal yang berbeda:
      - Kernel 1: point-wise features
      - Kernel 3: short-range patterns (~30ms @ 100Hz)
      - Kernel 5: medium-range patterns (~50ms @ 100Hz)
      - MaxPool:  dominant amplitude features

    Referensi: Fawaz et al. (2020), "InceptionTime: Finding AlexNet for
    Time Series Classification"
    """
    # Bottleneck 1x1
    bottleneck = layers.Conv1D(
        n_filters, 1, padding="same", name=f"inc{module_id}_bottleneck"
    )(x)

    # Parallel convolutions
    conv1 = layers.Conv1D(
        n_filters, 1, padding="same", name=f"inc{module_id}_conv1"
    )(bottleneck)
    conv3 = layers.Conv1D(
        n_filters, 3, padding="same", name=f"inc{module_id}_conv3"
    )(bottleneck)
    conv5 = layers.Conv1D(
        n_filters, 5, padding="same", name=f"inc{module_id}_conv5"
    )(bottleneck)

    # MaxPool branch
    pool = layers.MaxPooling1D(3, strides=1, padding="same", name=f"inc{module_id}_pool")(x)
    pool_conv = layers.Conv1D(
        n_filters, 1, padding="same", name=f"inc{module_id}_pool_conv"
    )(pool)

    # Concatenate all branches
    x = layers.Concatenate(name=f"inc{module_id}_concat")([conv1, conv3, conv5, pool_conv])
    x = layers.BatchNormalization(name=f"inc{module_id}_bn")(x)
    x = layers.Activation("relu", name=f"inc{module_id}_relu")(x)
    return x


def build_inceptiontime(
    input_shape: tuple = INPUT_SHAPE,
    n_filters: int = 32,
    n_modules: int = 3,
    dropout: float = 0.3,
    name: str = "InceptionTime",
) -> Model:
    """
    InceptionTime: multi-scale temporal feature extraction.

    Menggabungkan convolution kernels berbagai ukuran dalam satu module,
    memungkinkan model menangkap pola gait pada skala temporal berbeda
    secara simultan. Residual connections setiap 2 module.

    Total params: ~180K
    """
    inp = layers.Input(shape=input_shape, name="sensor_input")
    x = inp
    residual = x

    for i in range(n_modules):
        x = _inception_module(x, n_filters, module_id=i)

        # Residual connection setiap 2 module
        if (i + 1) % 2 == 0:
            if residual.shape[-1] != x.shape[-1]:
                residual = layers.Conv1D(
                    x.shape[-1], 1, padding="same",
                    name=f"inc_shortcut_{i}"
                )(residual)
            x = layers.Add(name=f"inc_res_{i}")([x, residual])
            x = layers.Activation("relu", name=f"inc_res_relu_{i}")(x)
            residual = x

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="dropout_out")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    return Model(inputs=inp, outputs=out, name=name)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — Akses semua model via nama string
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "CNN-LSTM":      build_cnn_lstm,
    "CNN-GRU":       build_cnn_gru,
    "TCN":           build_tcn,
    "Transformer":   build_transformer,
    "InceptionTime": build_inceptiontime,
}

def get_model(model_name: str, **kwargs) -> Model:
    """Build model by name from the registry."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' tidak dikenali. "
            f"Pilihan: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


def compile_model(
    model: Model,
    lr: float = LEARNING_RATE,
    class_weight: dict = None,
) -> Model:
    """
    Compile model dengan konfigurasi standar.
    - Optimizer: Adam (Kingma & Ba, 2015)
    - Loss: Binary Cross-Entropy
    - Metrics: accuracy, AUC, precision, recall
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def get_callbacks(model_name: str, patience: int = PATIENCE):
    """Standard training callbacks: EarlyStopping + ReduceLR + ModelCheckpoint."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_eer(y_true, y_scores):
    """
    Compute Equal Error Rate: titik di mana FAR = FRR.
    EER < 5% dianggap standar baik untuk sistem biometrik.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    frr = 1 - tpr
    # Cari titik di mana |FAR - FRR| minimal
    idx = np.nanargmin(np.abs(fpr - frr))
    eer = (fpr[idx] + frr[idx]) / 2
    return eer, thresholds[idx]


def evaluate_auth_metrics(model, X_test, y_test):
    """
    Evaluasi lengkap: accuracy, precision, recall, F1, FAR, FRR, EER, AUC.
    Returns dict dengan semua metrik.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix,
    )

    y_scores = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    eer, eer_thresh = compute_eer(y_test, y_scores)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "far":       far,
        "frr":       frr,
        "eer":       eer,
        "eer_threshold": eer_thresh,
        "auc":       roc_auc_score(y_test, y_scores),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_params":  model.count_params(),
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_model_comparison(results: dict):
    """Pretty-print comparison table."""
    header = (
        f"{'Model':<16} {'Params':>8} {'Acc':>7} {'Prec':>7} "
        f"{'Rec':>7} {'F1':>7} {'EER':>7} {'AUC':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in results.items():
        print(
            f"{name:<16} {m['n_params']:>8,} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1_score']:>7.4f} {m['eer']:>7.4f} {m['auc']:>7.4f}"
        )
    print("=" * len(header) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Quick test all models
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n===== Model Architecture Summary =====\n")
    for name, builder in MODEL_REGISTRY.items():
        model = builder()
        compile_model(model)
        print(f"\n{'─'*60}")
        print(f"  {name}: {model.count_params():,} parameters")
        print(f"{'─'*60}")
        model.summary(print_fn=lambda line: print(f"  {line}"))
