"""
statistical_analysis.py
========================
Pengujian hipotesis statistik untuk menjawab Research Questions.

H1: EER < 5%      -- one-sample t-test
H2: Degradasi FL vs centralized < 5pp  -- paired t-test
H3: Spearman rho SHAP centralized vs federated >= 0.70
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Optional
import logging
from pathlib import Path

from config import (
    ALPHA_STAT, EER_THRESHOLD, ACC_DEGRADATION, SHAP_CORR_MIN,
    RESULTS_DIR, SENSOR_AXES
)

log = logging.getLogger(__name__)


# ─── H1: EER < 5% ─────────────────────────────────────────────────────────────

def test_h1_eer(
    eer_scores: List[float],
    threshold: float = EER_THRESHOLD,
    alpha: float = ALPHA_STAT
) -> Dict:
    """
    Uji H1: apakah EER rata-rata secara statistik < threshold (5%).

    Menggunakan one-sample one-tailed t-test.
    H0: mu_EER >= threshold
    H1: mu_EER < threshold

    Args:
        eer_scores: list EER dari setiap fold / participant
        threshold:  nilai null hypothesis (default 0.05)

    Returns:
        dict hasil pengujian
    """
    eer_arr = np.array(eer_scores)
    n       = len(eer_arr)

    # one-sample t-test: apakah mean berbeda dari threshold
    t_stat, p_two = stats.ttest_1samp(eer_arr, popmean=threshold)
    # one-tailed (H1: mean < threshold) → p = p_two / 2 jika t < 0
    p_one = p_two / 2 if t_stat < 0 else 1.0 - p_two / 2

    reject_h0 = p_one < alpha

    result = {
        "hypothesis":  "H1: EER < 5%",
        "n_samples":   n,
        "mean_eer":    float(eer_arr.mean()),
        "std_eer":     float(eer_arr.std(ddof=1)),
        "t_statistic": float(t_stat),
        "p_value":     float(p_one),
        "alpha":       alpha,
        "reject_h0":   reject_h0,
        "conclusion":  (
            f"H1 DITERIMA: EER rata-rata ({eer_arr.mean():.4f}) "
            f"secara statistik lebih kecil dari {threshold} (p={p_one:.4f})"
        ) if reject_h0 else (
            f"H1 DITOLAK: EER rata-rata ({eer_arr.mean():.4f}) "
            f"tidak terbukti lebih kecil dari {threshold} (p={p_one:.4f})"
        )
    }

    log.info(result["conclusion"])
    return result


# ─── H2: Degradasi FL < 5pp ───────────────────────────────────────────────────

def test_h2_fl_degradation(
    acc_centralized: List[float],
    acc_federated: List[float],
    max_degradation: float = ACC_DEGRADATION,
    alpha: float = ALPHA_STAT
) -> Dict:
    """
    Uji H2: apakah degradasi akurasi FL vs centralized tidak melebihi 5pp.

    Menggunakan paired t-test pada selisih akurasi.
    H0: mean_degradation > max_degradation
    H1: mean_degradation <= max_degradation

    Args:
        acc_centralized: list akurasi model centralized per fold/participant
        acc_federated:   list akurasi model federated per fold/participant
    """
    degradation = np.array(acc_centralized) - np.array(acc_federated)

    t_stat, p_two = stats.ttest_1samp(degradation, popmean=max_degradation)
    # H1: mean_degradation < max_degradation → p = p_two/2 jika t < 0
    p_one = p_two / 2 if t_stat < 0 else 1.0 - p_two / 2

    reject_h0 = p_one < alpha

    result = {
        "hypothesis":       "H2: Degradasi FL vs Centralized < 5pp",
        "mean_degradation": float(degradation.mean()),
        "std_degradation":  float(degradation.std(ddof=1)),
        "max_allowed":      max_degradation,
        "t_statistic":      float(t_stat),
        "p_value":          float(p_one),
        "alpha":            alpha,
        "reject_h0":        reject_h0,
        "conclusion":       (
            f"H2 DITERIMA: Degradasi rata-rata ({degradation.mean():.4f}) "
            f"tidak melebihi {max_degradation} (p={p_one:.4f})"
        ) if reject_h0 else (
            f"H2 DITOLAK: Degradasi ({degradation.mean():.4f}) "
            f"melebihi batas {max_degradation} (p={p_one:.4f})"
        )
    }

    log.info(result["conclusion"])
    return result


# ─── H3: Spearman SHAP Consistency ────────────────────────────────────────────

def test_h3_shap_consistency(
    imp_centralized: Dict[str, float],
    imp_federated: Dict[str, float],
    min_rho: float = SHAP_CORR_MIN,
    alpha: float = ALPHA_STAT
) -> Dict:
    """
    Uji H3: apakah Spearman rank correlation SHAP importance
    antara model centralized dan federated >= 0.70.

    Args:
        imp_centralized: {sensor_axis: mean_abs_shap} dari model centralized
        imp_federated:   {sensor_axis: mean_abs_shap} dari model federated
    """
    axes  = SENSOR_AXES
    vals_c = [imp_centralized.get(a, 0) for a in axes]
    vals_f = [imp_federated.get(a, 0)   for a in axes]

    rho, p_val = stats.spearmanr(vals_c, vals_f)

    h3_confirmed = (rho >= min_rho) and (p_val < alpha)

    result = {
        "hypothesis":    "H3: Spearman SHAP rho >= 0.70",
        "spearman_rho":  float(rho),
        "p_value":       float(p_val),
        "min_rho":       min_rho,
        "alpha":         alpha,
        "h3_confirmed":  h3_confirmed,
        "conclusion":    (
            f"H3 TERKONFIRMASI: rho={rho:.4f} >= {min_rho} (p={p_val:.4f}), "
            "SHAP centralized dan federated konsisten."
        ) if h3_confirmed else (
            f"H3 TIDAK TERKONFIRMASI: rho={rho:.4f} < {min_rho} atau p={p_val:.4f}"
        )
    }

    log.info(result["conclusion"])
    return result


# ─── Comparison Table Builder ─────────────────────────────────────────────────

def build_comparison_table(
    scenario_results: List[Dict],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Buat tabel perbandingan hasil semua skenario eksperimen.

    Args:
        scenario_results: list of dict, masing-masing berisi kunci:
            scenario, accuracy, precision, recall, f1, far, frr, eer, auc,
            convergence_rounds (opsional), comm_cost_mb (opsional)
    """
    df = pd.DataFrame(scenario_results)

    # Format kolom numerik
    numeric_cols = ["accuracy", "precision", "recall", "f1", "far", "frr", "eer", "auc"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    if save_path:
        df.to_csv(save_path, index=False)
        log.info(f"Tabel perbandingan disimpan: {save_path}")

    return df


# ─── Summary Report ───────────────────────────────────────────────────────────

def generate_summary_report(
    h1_result: Dict,
    h2_result: Dict,
    h3_result: Dict,
    comparison_df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> str:
    """
    Buat laporan ringkasan hasil pengujian hipotesis dan evaluasi.
    """
    lines = [
        "=" * 70,
        "RINGKASAN HASIL PENELITIAN",
        "Gait Authentication dengan Explainable Federated Learning",
        "=" * 70,
        "",
        "PENGUJIAN HIPOTESIS",
        "-" * 40,
        f"[H1] {h1_result['conclusion']}",
        f"     Mean EER: {h1_result['mean_eer']:.4f} +/- {h1_result['std_eer']:.4f}",
        f"     t={h1_result['t_statistic']:.3f}, p={h1_result['p_value']:.4f}",
        "",
        f"[H2] {h2_result['conclusion']}",
        f"     Mean degradasi: {h2_result['mean_degradation']:.4f} pp",
        f"     t={h2_result['t_statistic']:.3f}, p={h2_result['p_value']:.4f}",
        "",
        f"[H3] {h3_result['conclusion']}",
        f"     Spearman rho={h3_result['spearman_rho']:.4f}, p={h3_result['p_value']:.4f}",
        "",
        "TABEL PERBANDINGAN SKENARIO",
        "-" * 40,
        comparison_df.to_string(index=False),
        "=" * 70
    ]
    report = "\n".join(lines)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        log.info(f"Summary report disimpan: {save_path}")

    return report
