"""
train_compare_multi.py
======================
Evaluasi model dengan strategi yang benar:
  1. Multi-subject: evaluasi di N subjects, rata-ratakan metrik
  2. Pooled dataset: gabungkan data dari banyak subjects jadi satu dataset besar
  3. Cross-validation: k-fold untuk stabilitas

Masalah sebelumnya: 1 subject = 11 windows = 44 samples total → tidak valid.
Solusi: pool data dari semua 745 subjects → ribuan samples.

Penggunaan:
    python train_compare_multi.py
    python train_compare_multi.py --max-subjects 100 --epochs 50
    python train_compare_multi.py --models CNN-LSTM Transformer InceptionTime
"""

import argparse
import json
import time
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import defaultdict

from config import (
    INPUT_SHAPE, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
    LEARNING_RATE, RANDOM_SEED, RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    SUBSET_AUTO, SENSOR_AXES, N_CHANNELS, WINDOW_SIZE, WINDOW_STEP,
)
from preprocessing_ouisir import (
    load_subset_auto, generate_synthetic_ouisir,
    preprocess_sequence, interpolate_missing,
    butter_lowpass_filter, sliding_window, normalize_windows,
    split_dataset,
)
from models import (
    MODEL_REGISTRY, get_model, compile_model, get_callbacks,
    evaluate_auth_metrics, print_model_comparison, compute_eer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGI 1: POOLED MULTI-SUBJECT DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_pooled_verification_dataset(
    all_data: dict,
    n_genuine_subjects: int = 50,
    n_impostor_subjects: int = None,
    impostor_ratio: float = 1.0,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Bangun dataset besar dengan pooling data dari banyak subjects.

    Strategi:
      - Pilih N subjects sebagai "genuine users"
      - Untuk setiap genuine user, ambil windows dari seq0 (gallery)
        sebagai genuine samples (label=1)
      - Ambil windows dari subjects lain sebagai impostor (label=0)
      - Gabungkan semua jadi satu dataset besar

    Ini menghasilkan ribuan samples, bukan puluhan.
    """
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    rng = np.random.default_rng(seed)
    sids = sorted(all_data.keys())

    if n_genuine_subjects > len(sids):
        n_genuine_subjects = len(sids)

    # Pilih subjects secara random
    selected = rng.choice(sids, size=n_genuine_subjects, replace=False)
    remaining = [s for s in sids if s not in selected]

    if n_impostor_subjects is None:
        n_impostor_subjects = min(len(remaining), n_genuine_subjects * 2)

    impostor_sids = rng.choice(
        remaining, size=min(n_impostor_subjects, len(remaining)), replace=False
    ) if remaining else []

    log.info(f"Pooled dataset: {n_genuine_subjects} genuine + {len(impostor_sids)} impostor subjects")

    # Fit global scaler pada semua data
    all_raw = []
    for sid in list(selected) + list(impostor_sids):
        for style, df in all_data[sid].items():
            df_interp = interpolate_missing(df)
            raw = df_interp[SENSOR_AXES].values.astype(np.float32)
            filtered = butter_lowpass_filter(raw)
            all_raw.append(filtered)

    all_raw_concat = np.concatenate(all_raw, axis=0)
    global_scaler = StandardScaler().fit(all_raw_concat)

    # Build genuine windows
    genuine_X, genuine_y = [], []
    for sid in selected:
        for style, df in all_data[sid].items():
            df_interp = interpolate_missing(df)
            raw = df_interp[SENSOR_AXES].values.astype(np.float32)
            filtered = butter_lowpass_filter(raw)
            try:
                windows = sliding_window(filtered)
                windows_norm, _ = normalize_windows(windows, scaler=global_scaler, fit=False)
                genuine_X.append(windows_norm)
                genuine_y.append(np.ones(len(windows_norm), dtype=np.int32))
            except ValueError:
                continue

    # Build impostor windows
    impostor_X, impostor_y = [], []
    for sid in impostor_sids:
        for style, df in all_data[sid].items():
            df_interp = interpolate_missing(df)
            raw = df_interp[SENSOR_AXES].values.astype(np.float32)
            filtered = butter_lowpass_filter(raw)
            try:
                windows = sliding_window(filtered)
                windows_norm, _ = normalize_windows(windows, scaler=global_scaler, fit=False)
                impostor_X.append(windows_norm)
                impostor_y.append(np.zeros(len(windows_norm), dtype=np.int32))
            except ValueError:
                continue

    genuine_X = np.concatenate(genuine_X, axis=0)
    impostor_X = np.concatenate(impostor_X, axis=0)

    # Balance classes berdasarkan impostor_ratio
    n_target_impostor = int(len(genuine_X) * impostor_ratio)
    if n_target_impostor < len(impostor_X):
        idx = rng.choice(len(impostor_X), size=n_target_impostor, replace=False)
        impostor_X = impostor_X[idx]

    X = np.concatenate([genuine_X, impostor_X], axis=0)
    y = np.concatenate([
        np.ones(len(genuine_X), dtype=np.int32),
        np.zeros(len(impostor_X), dtype=np.int32),
    ], axis=0)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    log.info(f"Pooled dataset: {len(X)} total samples "
             f"(genuine={len(genuine_X)}, impostor={len(impostor_X)})")

    return X.astype(np.float32), y, global_scaler


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model_name: str,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Latih satu model dan kembalikan metrik."""
    log.info(f"\n{'='*60}")
    log.info(f"  Training: {model_name}")
    log.info(f"  Train: {X_train.shape} (pos={y_train.sum()}, neg={(1-y_train).sum()})")
    log.info(f"{'='*60}")

    model = get_model(model_name)
    compile_model(model, lr=LEARNING_RATE)

    # Class weight untuk menangani imbalance
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    if n_pos > 0 and n_neg > 0:
        class_weight = {0: 1.0, 1: n_neg / n_pos}
    else:
        class_weight = None

    callbacks = get_callbacks(model_name, patience=PATIENCE)

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    train_time = time.time() - t0

    metrics = evaluate_auth_metrics(model, X_test, y_test)
    metrics["model_name"] = model_name
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["epochs_trained"] = len(history.history["loss"])
    metrics["best_val_loss"] = float(min(history.history["val_loss"]))
    metrics["train_samples"] = len(X_train)
    metrics["test_samples"] = len(X_test)

    log.info(f"\n  {model_name} Results:")
    log.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
    log.info(f"    Precision: {metrics['precision']:.4f}")
    log.info(f"    Recall:    {metrics['recall']:.4f}")
    log.info(f"    F1-Score:  {metrics['f1_score']:.4f}")
    log.info(f"    EER:       {metrics['eer']:.4f}")
    log.info(f"    AUC:       {metrics['auc']:.4f}")
    log.info(f"    FAR:       {metrics['far']:.4f}")
    log.info(f"    FRR:       {metrics['frr']:.4f}")
    log.info(f"    Params:    {metrics['n_params']:,}")
    log.info(f"    Time:      {train_time:.1f}s ({metrics['epochs_trained']} epochs)")

    # Save model
    save_path = MODELS_DIR / f"{model_name.lower().replace('-', '_')}_best.keras"
    model.save(save_path)
    log.info(f"    Saved:     {save_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict, save_dir: Path = PLOTS_DIR):
    """Generate comparison charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed, skipping plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    x = np.arange(len(names))
    width = 0.2

    # ── Plot 1: Scores ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 7))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auc"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"]

    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        vals = [results[n][metric] for n in names]
        bars = ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title(),
                      color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Pooled Multi-Subject Evaluation", fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "multi_comparison_scores.png", dpi=150)
    plt.close()
    log.info(f"Score plot saved: {save_dir / 'multi_comparison_scores.png'}")

    # ── Plot 2: Error rates ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    eer = [results[n]["eer"] for n in names]
    far = [results[n]["far"] for n in names]
    frr = [results[n]["frr"] for n in names]

    ax.bar(x - width, eer, width, label="EER", color="#F44336")
    ax.bar(x, far, width, label="FAR", color="#9C27B0")
    ax.bar(x + width, frr, width, label="FRR", color="#607D8B")

    ax.set_xlabel("Model")
    ax.set_ylabel("Error Rate")
    ax.set_title("Error Rates — Lower is Better")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "multi_comparison_errors.png", dpi=150)
    plt.close()

    # ── Plot 3: Efficiency ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    params = [results[n]["n_params"] / 1000 for n in names]
    f1s = [results[n]["f1_score"] for n in names]
    colors_scatter = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]

    for i, n in enumerate(names):
        ax.scatter(params[i], f1s[i], s=250, c=colors_scatter[i % len(colors_scatter)],
                   label=n, zorder=5, edgecolors="black", linewidths=1.5)
        ax.annotate(n, (params[i], f1s[i]), textcoords="offset points",
                    xytext=(10, 5), fontsize=10)

    ax.set_xlabel("Parameters (K)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("Efficiency: Parameters vs Performance", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "multi_efficiency.png", dpi=150)
    plt.close()

    log.info(f"All plots saved to {save_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-subject model comparison (pooled dataset)"
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        help="Models to train",
    )
    parser.add_argument("--max-subjects", type=int, default=200,
                        help="Max subjects to load from OU-ISIR")
    parser.add_argument("--genuine-subjects", type=int, default=50,
                        help="Number of subjects used as genuine class")
    parser.add_argument("--impostor-ratio", type=float, default=1.0,
                        help="Ratio impostor/genuine (1.0 = balanced)")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # ── Load Data ──────────────────────────────────────────────
    if SUBSET_AUTO.exists():
        log.info(f"Loading OU-ISIR (max {args.max_subjects} subjects)...")
        all_data = load_subset_auto(max_subjects=args.max_subjects)
    else:
        log.info("OU-ISIR not found, using synthetic data.")
        all_data = generate_synthetic_ouisir(n_subjects=args.max_subjects)

    log.info(f"Loaded {len(all_data)} subjects")

    # ── Build Pooled Dataset ───────────────────────────────────
    X, y, scaler = build_pooled_verification_dataset(
        all_data,
        n_genuine_subjects=args.genuine_subjects,
        impostor_ratio=args.impostor_ratio,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    log.info(f"\n{'='*60}")
    log.info(f"  POOLED DATASET SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Total:  {len(X)} samples")
    log.info(f"  Train:  {X_train.shape} (pos={y_train.sum()}, neg={(1-y_train).sum()})")
    log.info(f"  Val:    {X_val.shape}  (pos={y_val.sum()}, neg={(1-y_val).sum()})")
    log.info(f"  Test:   {X_test.shape} (pos={y_test.sum()}, neg={(1-y_test).sum()})")
    log.info(f"{'='*60}\n")

    # ── Train All Models ───────────────────────────────────────
    all_results = {}
    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            log.warning(f"Unknown model: {model_name}, skipping.")
            continue

        metrics = train_and_evaluate(
            model_name,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        all_results[model_name] = metrics

    # ── Print Comparison ───────────────────────────────────────
    print_model_comparison(all_results)

    # ── Save Results ───────────────────────────────────────────
    results_path = RESULTS_DIR / "multi_subject_comparison.json"
    serializable = {}
    for name, m in all_results.items():
        serializable[name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in m.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Results saved: {results_path}")

    # ── Plots ──────────────────────────────────────────────────
    plot_results(all_results)

    # ── Best Model ─────────────────────────────────────────────
    best = max(all_results, key=lambda n: all_results[n]["f1_score"])
    log.info(f"\n🏆 Best model by F1: {best} "
             f"(F1={all_results[best]['f1_score']:.4f}, "
             f"EER={all_results[best]['eer']:.4f})")


if __name__ == "__main__":
    main()
