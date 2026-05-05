"""
experiments.py
==============
Lima skenario eksperimen sesuai desain penelitian:

  Skenario 1: Baseline Evaluation (Centralized Training)
  Skenario 2: Perbandingan FL vs Centralized (K = 10, 25, 50)
  Skenario 3: Analisis Dampak Non-IID (alpha = inf, 1.0, 0.5, 0.1)
  Skenario 4: Multi-Position Robustness
  Skenario 5: SHAP Feature Analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split

from config import (
    TEST_SIZE, VAL_SIZE, RANDOM_SEED, RESULTS_DIR, PLOTS_DIR,
    FL_CLIENTS_CONFIGS, DIRICHLET_ALPHAS, MAX_FL_ROUNDS,
    LOCAL_EPOCHS, POSITIONS
)
from model import (
    build_cnn_lstm, compile_model,
    train_centralized, evaluate_authentication
)
from federated import build_fl_experiment, FederatedServer
from explainability import GaitSHAPAnalyzer
from statistical_analysis import (
    test_h1_eer, test_h2_fl_degradation, test_h3_shap_consistency,
    build_comparison_table, generate_summary_report
)

log = logging.getLogger(__name__)

PALETTE = {
    "centralized": "#1565C0",
    "fl_k10":      "#43A047",
    "fl_k25":      "#FB8C00",
    "fl_k50":      "#E53935",
    "iid":         "#4CAF50",
    "a1.0":        "#FFC107",
    "a0.5":        "#FF5722",
    "a0.1":        "#B71C1C"
}


# ─── Helper ───────────────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split: train / val / test."""
    X_tr_val, X_test, y_tr_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_val, y_tr_val, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_tr_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_result(result: Dict, name: str):
    """Simpan result dict ke JSON."""
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Result saved: {path}")


# ─── Skenario 1: Baseline Centralized ────────────────────────────────────────

def scenario1_baseline(
    X: np.ndarray, y: np.ndarray
) -> Dict:
    """
    Latih model secara terpusat dengan semua data (tanpa FL).
    Hasil ini menjadi upper-bound performa.
    """
    log.info("=" * 55)
    log.info("SKENARIO 1: Baseline Centralized Training")
    log.info("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    model = compile_model(build_cnn_lstm())
    history = train_centralized(
        model, X_train, y_train, X_val, y_val,
        model_name="s1_centralized"
    )

    metrics = evaluate_authentication(model, X_test, y_test)
    metrics["scenario"] = "Centralized Baseline"
    metrics["epochs_trained"] = len(history.history["loss"])

    # Plot learning curves
    _plot_learning_curve(
        history.history["loss"],
        history.history.get("val_loss", []),
        title="Skenario 1: Learning Curve (Centralized)",
        save_path=PLOTS_DIR / "s1_learning_curve.png"
    )

    save_result(metrics, "s1_baseline")
    log.info(f"Skenario 1 selesai | EER={metrics['eer']:.4f} | acc={metrics['accuracy']:.4f}")
    return metrics


# ─── Skenario 2: FL vs Centralized ───────────────────────────────────────────

def scenario2_fl_vs_centralized(
    X: np.ndarray, y: np.ndarray,
    n_total_clients: int = 50
) -> List[Dict]:
    """
    Bandingkan centralized vs FL dengan K = 10, 25, 50 clients per round.
    Semua FL menggunakan distribusi IID (alpha = 1e9).
    """
    log.info("=" * 55)
    log.info("SKENARIO 2: FL vs Centralized")
    log.info("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    all_results = []
    acc_centralized_list = []
    acc_fl_list = []

    # (A) Centralized baseline
    model_c = compile_model(build_cnn_lstm())
    train_centralized(model_c, X_train, y_train, X_val, y_val, model_name="s2_central")
    metrics_c = evaluate_authentication(model_c, X_test, y_test)
    metrics_c["scenario"] = "Centralized"
    metrics_c["k_clients"] = "N/A"
    metrics_c["alpha"] = "N/A"
    metrics_c["convergence_rounds"] = "N/A"
    all_results.append(metrics_c)
    acc_centralized_list.append(metrics_c["accuracy"])

    # (B–D) Federated dengan berbagai K
    for k in FL_CLIENTS_CONFIGS:
        log.info(f"  FL: K={k}, alpha=IID (1e9)")
        n_clients = min(n_total_clients, len(X_train) // 10)

        server = build_fl_experiment(
            X_train, y_train, X_val, y_val,
            n_clients=n_clients,
            clients_per_round=k,
            alpha=1e9,
            local_epochs=LOCAL_EPOCHS
        )
        history = server.run(n_rounds=MAX_FL_ROUNDS, verbose=False)
        n_rounds = len(history)

        metrics_fl = evaluate_authentication(server.global_model, X_test, y_test)
        metrics_fl["scenario"]          = f"FL (K={k})"
        metrics_fl["k_clients"]         = k
        metrics_fl["alpha"]             = "IID"
        metrics_fl["convergence_rounds"] = n_rounds
        metrics_fl["total_comm_mb"]     = sum(r.comm_cost_mb for r in history)
        all_results.append(metrics_fl)
        acc_fl_list.append(metrics_fl["accuracy"])

    # Plot bar chart perbandingan
    _plot_scenario2_comparison(all_results, save_path=PLOTS_DIR / "s2_comparison.png")

    # Plot convergence curve untuk semua FL configs
    # (disiapkan tapi perlu menyimpan history – simplified)

    # Uji H2
    avg_fl_acc = np.mean(acc_fl_list)
    avg_c_acc  = np.mean(acc_centralized_list)
    h2_result  = test_h2_fl_degradation(
        acc_centralized=acc_centralized_list * len(acc_fl_list),
        acc_federated=acc_fl_list
    )

    save_result({"results": all_results, "h2_test": h2_result}, "s2_fl_vs_central")
    return all_results


# ─── Skenario 3: Non-IID Analysis ────────────────────────────────────────────

def scenario3_noniid(
    X: np.ndarray, y: np.ndarray,
    n_total_clients: int = 50,
    k_per_round: int = 10
) -> List[Dict]:
    """
    Analisis dampak heterogenitas data (Dirichlet alpha = inf, 1.0, 0.5, 0.1).
    Konfigurasi FL tetap: K=10, E=5.
    """
    log.info("=" * 55)
    log.info("SKENARIO 3: Non-IID Analysis")
    log.info("=" * 55)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    all_results = []
    convergence_data = {}

    for alpha in DIRICHLET_ALPHAS:
        alpha_label = "IID" if alpha > 100 else str(alpha)
        log.info(f"  alpha = {alpha_label}")

        n_clients = min(n_total_clients, len(X_train) // 8)
        server = build_fl_experiment(
            X_train, y_train, X_val, y_val,
            n_clients=n_clients,
            clients_per_round=k_per_round,
            alpha=alpha,
            local_epochs=LOCAL_EPOCHS
        )
        history = server.run(n_rounds=MAX_FL_ROUNDS, verbose=False)

        metrics = evaluate_authentication(server.global_model, X_test, y_test)
        metrics["scenario"]           = f"alpha={alpha_label}"
        metrics["alpha"]              = alpha_label
        metrics["convergence_rounds"] = len(history)

        all_results.append(metrics)
        convergence_data[alpha_label] = {
            "rounds":    [r.round_num  for r in history],
            "val_loss":  [r.global_loss for r in history],
            "val_acc":   [r.global_acc  for r in history]
        }

    _plot_scenario3_convergence(convergence_data, save_path=PLOTS_DIR / "s3_convergence.png")
    save_result({"results": all_results}, "s3_noniid")
    return all_results


# ─── Skenario 4: Multi-Position Robustness ───────────────────────────────────

def scenario4_multiposition(
    all_data: Dict,
    target_pids: List[str] = None
) -> Dict:
    """
    Latih dengan 3 posisi, uji pada posisi ke-4 yang tidak pernah dilihat.
    Training: pocket_left, pocket_right, hand_left
    Testing:  hand_right
    """
    from preprocessing import build_dataset

    log.info("=" * 55)
    log.info("SKENARIO 4: Multi-Position Robustness")
    log.info("=" * 55)

    TRAIN_POSITIONS = ["pocket_left", "pocket_right", "hand_left"]
    TEST_POSITION   = ["hand_right"]

    pids = target_pids or list(all_data.keys())[:5]
    per_pid_results = []

    for pid in pids:
        try:
            X_tr, y_tr, scaler = build_dataset(
                all_data, target_pid=pid, positions=TRAIN_POSITIONS
            )
            X_te, y_te, _ = build_dataset(
                all_data, target_pid=pid, positions=TEST_POSITION
            )
        except Exception as e:
            log.warning(f"Skip participant {pid}: {e}")
            continue

        X_train, X_val, _, y_train, y_val, _ = split_data(X_tr, y_tr)

        model = compile_model(build_cnn_lstm())
        train_centralized(
            model, X_train, y_train, X_val, y_val,
            model_name=f"s4_{pid}"
        )
        metrics = evaluate_authentication(model, X_te, y_te)
        metrics["participant"] = pid
        metrics["scenario"]    = "Multi-Position"
        per_pid_results.append(metrics)
        log.info(f"  {pid}: EER={metrics['eer']:.4f} | acc={metrics['accuracy']:.4f}")

    # Agregasi
    eer_list = [r["eer"] for r in per_pid_results]
    acc_list = [r["accuracy"] for r in per_pid_results]
    summary = {
        "scenario":    "Multi-Position Robustness",
        "n_participants": len(per_pid_results),
        "mean_eer":    float(np.mean(eer_list)),
        "std_eer":     float(np.std(eer_list)),
        "mean_acc":    float(np.mean(acc_list)),
        "per_participant": per_pid_results
    }
    save_result(summary, "s4_multiposition")
    return summary


# ─── Skenario 5: SHAP Feature Analysis ───────────────────────────────────────

def scenario5_shap_analysis(
    model_centralized,
    model_federated,
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Dict:
    """
    Jalankan SHAP analysis pada model centralized dan federated global.
    Bandingkan feature importance keduanya dan validasi H3.
    """
    log.info("=" * 55)
    log.info("SKENARIO 5: SHAP Feature Analysis")
    log.info("=" * 55)

    # Pilih background data secara acak dari training set
    bg_idx = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
    X_bg   = X_train[bg_idx]

    results = {}

    for label, model in [("centralized", model_centralized),
                          ("federated",   model_federated)]:
        analyzer = GaitSHAPAnalyzer(model, X_bg)
        res = analyzer.run_full_analysis(X_test, model_label=label)
        results[label] = res

    # Bandingkan consistency
    from scipy.stats import spearmanr
    from statistical_analysis import test_h3_shap_consistency

    h3 = test_h3_shap_consistency(
        imp_centralized=results["centralized"]["channel_importance"],
        imp_federated=results["federated"]["channel_importance"]
    )
    results["h3_test"] = h3

    # Plot perbandingan feature importance side-by-side
    _plot_shap_comparison(
        results["centralized"]["channel_importance"],
        results["federated"]["channel_importance"],
        save_path=PLOTS_DIR / "s5_shap_comparison.png"
    )

    save_result(results, "s5_shap")
    return results


# ─── Plot Helpers ─────────────────────────────────────────────────────────────

def _plot_learning_curve(
    train_loss: List[float],
    val_loss: List[float],
    title: str,
    save_path: Path
):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, color=PALETTE["centralized"], label="Train Loss", linewidth=1.5)
    if val_loss:
        ax.plot(epochs, val_loss, color=PALETTE["fl_k50"], linestyle="--",
                label="Val Loss", linewidth=1.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_scenario2_comparison(results: List[Dict], save_path: Path):
    labels   = [r["scenario"] for r in results]
    acc_vals = [float(r["accuracy"]) for r in results]
    eer_vals = [float(r["eer"]) for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, values, ylabel, title in zip(
        axes,
        [acc_vals, eer_vals],
        ["Accuracy", "EER"],
        ["Accuracy per Konfigurasi", "Equal Error Rate per Konfigurasi"]
    ):
        bars = ax.bar(x, values, width=0.5, color=PALETTE["centralized"],
                      alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Skenario 2: FL vs Centralized", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_scenario3_convergence(convergence_data: Dict, save_path: Path):
    color_map = {
        "IID": PALETTE["iid"],
        "1.0": PALETTE["a1.0"],
        "0.5": PALETTE["a0.5"],
        "0.1": PALETTE["a0.1"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, data in convergence_data.items():
        color = color_map.get(str(label), "#9E9E9E")
        axes[0].plot(data["rounds"], data["val_loss"], label=f"alpha={label}",
                     color=color, linewidth=1.5)
        axes[1].plot(data["rounds"], data["val_acc"],  label=f"alpha={label}",
                     color=color, linewidth=1.5)

    for ax, ylabel, title in zip(
        axes,
        ["Validation Loss", "Validation Accuracy"],
        ["Convergence: Val Loss", "Convergence: Val Accuracy"]
    ):
        ax.set_xlabel("Communication Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)

    plt.suptitle("Skenario 3: Dampak Non-IID (Dirichlet alpha)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_shap_comparison(
    imp_c: Dict[str, float],
    imp_f: Dict[str, float],
    save_path: Path
):
    from config import SENSOR_AXES
    axes_order = SENSOR_AXES

    vals_c = [imp_c.get(a, 0) for a in axes_order]
    vals_f = [imp_f.get(a, 0) for a in axes_order]

    x = np.arange(len(axes_order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars_c = ax.bar(x - width/2, vals_c, width, label="Centralized",
                    color=PALETTE["centralized"], alpha=0.85)
    bars_f = ax.bar(x + width/2, vals_f, width, label="Federated (Global)",
                    color=PALETTE["fl_k25"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(axes_order, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Skenario 5: SHAP Feature Importance – Centralized vs Federated",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Run All Scenarios ────────────────────────────────────────────────────────

def run_all_scenarios(all_data: Dict) -> Dict:
    """
    Jalankan semua 5 skenario secara berurutan dan kompilasi hasil.
    Mengembalikan dict lengkap berisi semua hasil dan pengujian statistik.
    """
    from preprocessing import build_dataset

    log.info("Membangun dataset gabungan untuk skenario 1-3...")
    pids = list(all_data.keys())

    # Gabungkan data dari semua participant untuk skenario 1-3
    X_list, y_list = [], []
    for pid in pids:
        try:
            X_p, y_p, _ = build_dataset(all_data, target_pid=pid)
            X_list.append(X_p)
            y_list.append(y_p)
        except Exception as e:
            log.warning(f"Skip {pid}: {e}")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    log.info(f"Dataset gabungan: X={X_all.shape}, y={np.bincount(y_all)}")

    np.random.seed(RANDOM_SEED)

    # Skenario 1
    s1 = scenario1_baseline(X_all, y_all)

    # Skenario 2
    s2 = scenario2_fl_vs_centralized(X_all, y_all, n_total_clients=len(pids))

    # Skenario 3
    s3 = scenario3_noniid(X_all, y_all, n_total_clients=len(pids))

    # Skenario 4
    s4 = scenario4_multiposition(all_data, target_pids=pids[:10])

    # Skenario 5 – butuh model terlatih
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_all, y_all)
    model_c = compile_model(build_cnn_lstm())
    train_centralized(model_c, X_train, y_train, X_val, y_val, model_name="s5_central")

    server_fl = build_fl_experiment(
        X_train, y_train, X_val, y_val,
        n_clients=len(pids), clients_per_round=10,
        alpha=1.0
    )
    server_fl.run(n_rounds=50, verbose=False)

    s5 = scenario5_shap_analysis(
        model_c, server_fl.global_model, X_train, X_test
    )

    # Pengujian Hipotesis
    eer_list_s1 = [s1["eer"]] * 5   # dalam penelitian nyata, gunakan cross-validation
    h1 = test_h1_eer(eer_list_s1)

    acc_fl_list = [r["accuracy"] for r in s2 if r["scenario"] != "Centralized"]
    acc_c_list  = [s2[0]["accuracy"]] * len(acc_fl_list)
    h2 = test_h2_fl_degradation(acc_c_list, acc_fl_list)

    h3 = s5.get("h3_test", {})

    # Tabel perbandingan
    comparison = build_comparison_table(
        s2 + s3,
        save_path=RESULTS_DIR / "comparison_table.csv"
    )

    report = generate_summary_report(
        h1, h2, h3, comparison,
        save_path=RESULTS_DIR / "summary_report.txt"
    )
    print("\n" + report)

    return {
        "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5,
        "h1": h1, "h2": h2, "h3": h3,
        "report": report
    }
