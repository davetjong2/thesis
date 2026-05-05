"""
train_compare.py
================
Latih dan bandingkan semua arsitektur model pada dataset OU-ISIR.

Penggunaan:
    python train_compare.py                     # Semua model, data sintetis jika OU-ISIR tidak ada
    python train_compare.py --models CNN-LSTM TCN
    python train_compare.py --max-subjects 100
    python train_compare.py --epochs 50 --batch-size 64
"""

import argparse
import json
import time
import logging
import numpy as np
import tensorflow as tf
from pathlib import Path

from config import (
    INPUT_SHAPE, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
    LEARNING_RATE, RANDOM_SEED, RESULTS_DIR, PLOTS_DIR, MODELS_DIR,
    SUBSET_AUTO,
)
from preprocessing_ouisir import (
    load_subset_auto, generate_synthetic_ouisir,
    build_verification_dataset, split_dataset,
)
from models import (
    MODEL_REGISTRY, get_model, compile_model, get_callbacks,
    evaluate_auth_metrics, print_model_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Reproducibility
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def train_single_model(
    model_name: str,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    save_model: bool = True,
) -> dict:
    """
    Latih satu model dan evaluasi.

    Returns:
        dict: metrik evaluasi + training time + history
    """
    log.info(f"\n{'='*60}")
    log.info(f"  Training: {model_name}")
    log.info(f"{'='*60}")

    # Build & compile
    model = get_model(model_name)
    compile_model(model, lr=LEARNING_RATE)

    # Class weight
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    class_weight = {0: 1.0, 1: n_neg / n_pos} if n_pos > 0 else None

    callbacks = get_callbacks(model_name, patience=PATIENCE)

    # Train
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

    # Evaluate
    metrics = evaluate_auth_metrics(model, X_test, y_test)
    metrics["model_name"] = model_name
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["epochs_trained"] = len(history.history["loss"])
    metrics["best_val_loss"] = float(min(history.history["val_loss"]))

    log.info(f"\n  {model_name} Results:")
    log.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
    log.info(f"    F1-Score:  {metrics['f1_score']:.4f}")
    log.info(f"    EER:       {metrics['eer']:.4f}")
    log.info(f"    AUC:       {metrics['auc']:.4f}")
    log.info(f"    Params:    {metrics['n_params']:,}")
    log.info(f"    Time:      {train_time:.1f}s ({metrics['epochs_trained']} epochs)")

    # Save model
    if save_model:
        save_path = MODELS_DIR / f"{model_name.lower().replace('-', '_')}_best.keras"
        model.save(save_path)
        log.info(f"    Saved:     {save_path}")

    return metrics


def plot_comparison(results: dict, save_dir: Path = PLOTS_DIR):
    """Generate comparison bar charts."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed. Skipping plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())

    # ── Plot 1: Accuracy, F1, AUC ──────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.25

    acc  = [results[n]["accuracy"] for n in names]
    f1   = [results[n]["f1_score"] for n in names]
    auc  = [results[n]["auc"] for n in names]

    bars1 = ax.bar(x - width, acc, width, label="Accuracy", color="#4CAF50")
    bars2 = ax.bar(x, f1, width, label="F1-Score", color="#2196F3")
    bars3 = ax.bar(x + width, auc, width, label="AUC", color="#FF9800")

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy / F1 / AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "model_comparison_scores.png", dpi=150)
    plt.close()

    # ── Plot 2: EER, FAR, FRR ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    eer = [results[n]["eer"] for n in names]
    far = [results[n]["far"] for n in names]
    frr = [results[n]["frr"] for n in names]

    bars1 = ax.bar(x - width, eer, width, label="EER", color="#F44336")
    bars2 = ax.bar(x, far, width, label="FAR", color="#9C27B0")
    bars3 = ax.bar(x + width, frr, width, label="FRR", color="#607D8B")

    ax.set_xlabel("Model")
    ax.set_ylabel("Error Rate")
    ax.set_title("Model Comparison: EER / FAR / FRR (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_dir / "model_comparison_errors.png", dpi=150)
    plt.close()

    # ── Plot 3: Params vs Performance ─────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    params = [results[n]["n_params"] / 1000 for n in names]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]

    for i, n in enumerate(names):
        ax.scatter(params[i], f1[i], s=200, c=colors[i % len(colors)],
                   label=n, zorder=5, edgecolors="black")
        ax.annotate(n, (params[i], f1[i]), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Parameters (K)")
    ax.set_ylabel("F1-Score")
    ax.set_title("Efficiency: Parameters vs Performance")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "model_efficiency.png", dpi=150)
    plt.close()

    log.info(f"Plots saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train & compare gait models")
    parser.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        help="Model names to train"
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--target-subject", type=str, default=None,
                        help="Subject ID for verification (default: first in dataset)")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    # ── Load Data ──────────────────────────────────────────────
    if SUBSET_AUTO.exists():
        log.info(f"Loading OU-ISIR from {SUBSET_AUTO}")
        all_data = load_subset_auto(max_subjects=args.max_subjects)
    else:
        log.info("OU-ISIR not found. Using synthetic data.")
        n_subj = args.max_subjects or 20
        all_data = generate_synthetic_ouisir(n_subjects=n_subj)

    sids = sorted(all_data.keys())
    target = args.target_subject or sids[0]
    log.info(f"Target subject: {target} | Total subjects: {len(sids)}")

    # ── Build Dataset ──────────────────────────────────────────
    X, y, scaler = build_verification_dataset(all_data, target_sid=target)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    log.info(f"\nDataset shapes:")
    log.info(f"  Train: {X_train.shape} | Pos: {y_train.sum()} | Neg: {(1-y_train).sum()}")
    log.info(f"  Val:   {X_val.shape}")
    log.info(f"  Test:  {X_test.shape}")

    # ── Train All Models ───────────────────────────────────────
    all_results = {}
    for model_name in args.models:
        if model_name not in MODEL_REGISTRY:
            log.warning(f"Unknown model: {model_name}. Skipping.")
            continue

        metrics = train_single_model(
            model_name,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_model=not args.no_save,
        )
        all_results[model_name] = metrics

    # ── Print Comparison ───────────────────────────────────────
    print_model_comparison(all_results)

    # ── Save Results JSON ──────────────────────────────────────
    results_path = RESULTS_DIR / "model_comparison.json"
    # Convert numpy types for JSON serialization
    serializable = {}
    for name, m in all_results.items():
        serializable[name] = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in m.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # ── Generate Plots ─────────────────────────────────────────
    plot_comparison(all_results)

    # ── Best Model Summary ─────────────────────────────────────
    best = min(all_results, key=lambda n: all_results[n]["eer"])
    log.info(f"\n🏆 Best model by EER: {best} (EER={all_results[best]['eer']:.4f})")


if __name__ == "__main__":
    main()
